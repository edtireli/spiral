from __future__ import annotations

import gzip
import io
import json
import tarfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.academic_finetune import CORPUS_SCHEMA, MANIFEST_SCHEMA
from scripts.academic_finetune.corpus import CollectionConfig, CorpusCollector, compile_corpus
from scripts.academic_finetune.sources import (
    ARXIV_API,
    NCBI_EFETCH,
    NCBI_ESEARCH,
    ArxivSource,
    PubMedBaselineSource,
    PubMedSource,
    SourceDocument,
    parse_arxiv_feed,
    parse_pubmed_esearch,
    parse_pubmed_xml,
)
from scripts.academic_finetune.text import (
    academic_unit_rejection_reason,
    canonicalize_citations,
    claims_are_feasible,
    claims_are_safe,
    context_is_safe,
    context_target_overlap,
    clean_pmc_xml,
    clean_prose,
    clean_tex_archive,
    document_rejection_reason,
    neutral_claims,
    plan_content_recall,
    plan_target_overlap,
)


FIXTURES = Path(__file__).parents[1] / "scripts" / "academic_finetune" / "fixtures"
CUTOFF = date(2021, 12, 31)


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class FixtureFetcher:
    """Strict fixture transport: an unexpected network shape fails the test."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> bytes:
        if self.fail:
            raise AssertionError("completed collection attempted network access")
        parameters = dict(params or {})
        self.calls.append((url, parameters))
        if url == ARXIV_API:
            query = str(parameters["search_query"])
            return fixture("arxiv_hep_th.xml" if "hep-th" in query else "arxiv_hep_ph.xml")
        if url == NCBI_ESEARCH:
            return fixture("pubmed_esearch.xml")
        if url == NCBI_EFETCH and parameters.get("db") == "pubmed":
            return fixture("pubmed.xml")
        if url == NCBI_EFETCH and parameters.get("db") == "pmc":
            return fixture("pmc.xml")
        raise AssertionError(f"unexpected fixture request: {url} {parameters}")


def sources(fetcher: FixtureFetcher):
    return [
        ArxivSource("hep-th", cutoff=CUTOFF, fetcher=fetcher),
        ArxivSource("hep-ph", cutoff=CUTOFF, fetcher=fetcher),
        PubMedSource(
            "hasabstract[text]",
            cutoff=CUTOFF,
            email="fixture@example.org",
            fetcher=fetcher,
        ),
    ]


def test_offline_parsers_enforce_latest_revision_cutoff_and_provenance() -> None:
    hep_th, total, start = parse_arxiv_feed(
        fixture("arxiv_hep_th.xml"),
        category="hep-th",
        cutoff=CUTOFF,
        query="cat:hep-th",
    )
    assert (total, start) == (3, 0)
    assert {document.source_id for document in hep_th} == {"2101.00001v2", "1911.00004v1"}
    assert all(document.latest_version <= CUTOFF.isoformat() for document in hep_th)
    assert hep_th[0].landing_url.endswith("2101.00001v2")
    assert hep_th[0].license.startswith("http://arxiv.org/licenses/")

    ids, count, returned_start = parse_pubmed_esearch(fixture("pubmed_esearch.xml"))
    assert ids == ("31415926", "16180339", "27182818")
    assert (count, returned_start) == (3, 0)
    pubmed = parse_pubmed_xml(fixture("pubmed.xml"), cutoff=CUTOFF, query="hasabstract[text]")
    assert {document.source_id for document in pubmed} == {"31415926", "16180339"}
    with_pmc = next(document for document in pubmed if document.source_id == "31415926")
    assert "PMC7654321" in with_pmc.artifact_url
    assert with_pmc.latest_version == "2020-10-12"
    # MEDLINE can refresh indexing years later without changing the authors'
    # publication into post-cutoff prose.
    assert with_pmc.metadata_revised == "2023-01-05"


def test_collection_is_bounded_resumable_and_does_not_persist_contact(tmp_path: Path) -> None:
    fetcher = FixtureFetcher()
    collector = CorpusCollector(
        tmp_path / "cache",
        CollectionConfig(
            maximum_documents_per_source=10,
            page_size=10,
            maximum_scanned_per_source=30,
        ),
    )
    documents = collector.collect(sources(fetcher))
    assert len(documents) == 6
    first_call_count = len(fetcher.calls)
    assert first_call_count == 4  # two Atom pages plus PubMed search+fetch
    state_text = collector.state_path.read_text(encoding="utf-8")
    assert "fixture@example.org" not in state_text
    assert "contact_sha256" in state_text

    # A completed checkpoint performs no fetches, while cached record hashes are
    # still verified during loading.
    resumed = collector.collect(sources(FixtureFetcher(fail=True)))
    assert [document.document_id for document in resumed] == [
        document.document_id for document in documents
    ]
    assert len(fetcher.calls) == first_call_count


def test_compiler_is_deterministic_balanced_author_safe_and_leak_free(tmp_path: Path) -> None:
    collector = CorpusCollector(
        tmp_path / "cache",
        CollectionConfig(
            maximum_documents_per_source=10,
            page_size=10,
            maximum_scanned_per_source=30,
        ),
    )
    documents = collector.collect(sources(FixtureFetcher()))
    output = tmp_path / "academic_corpus.jsonl"
    manifest = compile_corpus(documents, output_path=output, require_trainable_splits=False)
    first_bytes = output.read_bytes()
    first_manifest_bytes = (tmp_path / "academic_corpus.jsonl.manifest.json").read_bytes()
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["trainable"] is False
    assert manifest["non_trainable_reasons"]
    assert manifest["source_strata"] == ["arxiv:hep-ph", "arxiv:hep-th", "pubmed"]
    assert len(set(manifest["counts"]["by_stratum"].values())) == 1
    assert manifest["counts"]["documents_by_stratum"] == {
        "arxiv:hep-ph": 2,
        "arxiv:hep-th": 2,
        "pubmed": 2,
    }
    assert set(manifest["counts"]["by_split"]) == {"train", "validation", "test"}
    assert manifest["counts"]["documents_by_publication_year"]

    records = [json.loads(line) for line in first_bytes.decode().splitlines()]
    assert records and all(record["schema_version"] == CORPUS_SCHEMA for record in records)
    document_splits: dict[str, set[str]] = {}
    author_splits: dict[str, set[str]] = {}
    for record in records:
        document_splits.setdefault(record["document"]["document_id"], set()).add(record["split"])
        for author in record["document"]["authors"]:
            author_splits.setdefault(author.casefold(), set()).add(record["split"])
        assert record["input"]["construction_method"] == "bounded_semantic_roles_v3"
        assert claims_are_feasible(record["input"]["claims"], record["target"])
        assert plan_content_recall(record["input"]["claims"], record["target"]) >= 0.55
        assert claims_are_safe(record["input"]["claims"], record["target"])
        assert record["provenance"]["plan_target_overlap"]["longest_common_ngram"] < 5
        assert record["provenance"]["text_unit_sha256"]
    assert all(len(splits) == 1 for splits in document_splits.values())
    assert all(len(splits) == 1 for splits in author_splits.values())
    assert all(
        len({record["task_type"] for record in records if record["document"]["document_id"] == identity}) == 1
        for identity in document_splits
    )
    assert all(
        sum(record["document"]["document_id"] == identity for record in records) == 1
        for identity, splits in document_splits.items()
        if next(iter(splits)) != "train"
    )

    compile_corpus(documents, output_path=output, require_trainable_splits=False)
    assert output.read_bytes() == first_bytes
    assert (tmp_path / "academic_corpus.jsonl.manifest.json").read_bytes() == first_manifest_bytes

    with pytest.raises(ValueError, match="every split/source stratum"):
        compile_corpus(documents, output_path=tmp_path / "not-production.jsonl")


def test_target_overlap_gate_rejects_clause_copying() -> None:
    target = (
        "However, the calibrated measurement constrains the interaction strength "
        "without establishing decisive evidence for a new resonance."
    )
    copied = ["The calibrated measurement constrains the interaction strength without establishing evidence."]
    assert not claims_are_safe(copied, target)
    assert plan_target_overlap(copied, target)["longest_common_ngram"] >= 5
    planned = neutral_claims(target)
    assert claims_are_safe(planned, target)
    assert target.casefold() not in " ".join(planned).casefold()

    argument = (
        "The new state is energetically stable, but we leave its phenomenological "
        "study to later work."
    )
    argument_plan = " ".join(neutral_claims(argument)).casefold()
    assert "new state" in argument_plan
    assert "energetic stability" in argument_plan
    assert "phenomenological study" in argument_plan
    assert "deferred" in argument_plan
    assert claims_are_safe(neutral_claims(argument), argument)

    qualified = (
        "The approximation remains accurate to approximately five percent, although "
        "severe sampling losses limit its interpretation."
    )
    qualified_plan = " ".join(neutral_claims(qualified)).casefold()
    assert "five percent" in qualified_plan
    assert "sampling losses" in qualified_plan
    assert "limitation" in qualified_plan

    result_object = "The deviations induce severe observational signal losses."
    result_plan = " ".join(neutral_claims(result_object)).casefold()
    assert "deviations" in result_plan
    assert "observational signal losses" in result_plan


def test_tex_citations_are_canonical_and_declared() -> None:
    value = (
        r"Earlier work \cite{Kitaev} and a later comparison \citep[Sec. 2]{QHE,KK} "
        r"support the bounded conclusion [19]."
    )
    canonical = canonicalize_citations(value)
    assert canonical == "Earlier work [1] and a later comparison [2] support the bounded conclusion [3]."


def test_notice_promotional_and_malformed_units_are_rejected() -> None:
    assert document_rejection_reason(
        "RETRACTED ARTICLE: An unreliable result",
        "A nominal abstract remains here.",
    ) == "notice_title"
    assert document_rejection_reason(
        "A nominal article",
        "The Publisher has retracted this article in agreement with the editor.",
    ) == "notice_prose"
    assert academic_unit_rejection_reason(
        "This CME-accredited series is available at https://example.org/program."
    ) == "promotional_or_web_boilerplate"
    assert academic_unit_rejection_reason(
        "The measured response (including the reference group remains stable."
    ) == "unbalanced_delimiter"


def test_context_overlap_gate_rejects_answer_copy_but_allows_shared_terms() -> None:
    target = (
        "The renormalized amplitude remains finite because the counterterm cancels "
        "the logarithmic divergence."
    )
    topical_context = (
        "We next study the renormalized amplitude in the same perturbative regime, "
        "using a local counterterm fixed by the preceding symmetry argument."
    )
    copied_context = f"Earlier draft: {target} The discussion then continues."
    long_copy = (
        "An earlier note says the renormalized amplitude remains finite because the "
        "counterterm cancels the logarithmic divergence in this calculation."
    )
    assert context_is_safe(topical_context, target)
    assert not context_is_safe(copied_context, target)
    assert context_target_overlap(copied_context, target)["exact_target_in_context"] is True
    assert not context_is_safe(long_copy, target)


def test_production_split_has_eight_independent_components_per_stratum(
    tmp_path: Path,
) -> None:
    documents: list[SourceDocument] = []
    for stratum in ("arxiv:hep-ph", "arxiv:hep-th", "pubmed"):
        provider = "pubmed" if stratum == "pubmed" else "arxiv"
        for index in range(26):
            documents.append(SourceDocument(
                provider=provider,
                stratum=stratum,
                source_id=f"{stratum}-{index}",
                title=f"Controlled academic measurement {stratum} {index}",
                authors=(f"Independent Author {stratum} {index}",),
                published="2020-01-01",
                latest_version="2020-01-01",
                abstract=(
                    f"The calibrated {stratum} analysis of sample {index} establishes a bounded relation "
                    "between the observable response and the controlled theoretical assumption. "
                    f"An independent comparison reports {index + 20} percent agreement and "
                    "preserves the stated limitation under the secondary measurement protocol."
                ),
                body="",
                landing_url=f"https://example.invalid/{stratum}/{index}",
                artifact_url="",
                metadata_endpoint="fixture",
                content_endpoint="",
                query="fixture",
                extraction="fixture_abstract",
            ))
    output = tmp_path / "production.jsonl"
    manifest = compile_corpus(documents, output_path=output)
    assert manifest["trainable"] is True
    for stratum in ("arxiv:hep-ph", "arxiv:hep-th", "pubmed"):
        for split in ("validation", "test"):
            key = f"{stratum}|{split}"
            assert manifest["counts"]["documents_by_stratum_split"][key] >= 8
            assert manifest["split_diagnostics"]["components_by_stratum_split"][key] >= 8
    records = [json.loads(line) for line in output.read_text().splitlines()]
    heldout_documents = Counter(
        record["document"]["document_id"]
        for record in records
        if record["split"] in {"validation", "test"}
    )
    assert heldout_documents and set(heldout_documents.values()) == {1}


def test_compiler_skips_title_copied_into_abstract_instead_of_failing_document(
    tmp_path: Path,
) -> None:
    # The first abstract sentence substantially repeats the title, while the
    # second sentence remains a useful held-out realization example.
    document = SourceDocument(
        provider="arxiv",
        stratum="arxiv:hep-th",
        source_id="fixture-title-copy",
        title=(
            "Renormalized amplitudes in curved backgrounds with a local counterterm "
            "for interacting scalar quantum fields"
        ),
        authors=("One Author",),
        published="2021-01-01",
        latest_version="2021-01-01",
        abstract=(
            "Renormalized amplitudes in curved backgrounds with a local counterterm "
            "for interacting scalar quantum fields provide the subject of this detailed "
            "theoretical analysis. "
            "Nevertheless, the finite remainder depends on the boundary condition "
            "and therefore limits the universality of the construction."
        ),
        body="",
        landing_url="https://arxiv.org/abs/fixture-title-copy",
        artifact_url="",
        metadata_endpoint=ARXIV_API,
        content_endpoint="",
        query="fixture",
        extraction="atom_abstract",
    )
    # Exercise the private record compiler directly: corpus-level compilation
    # additionally requires all three balanced strata and production splits.
    from scripts.academic_finetune.corpus import _base_examples

    examples = _base_examples(document, CUTOFF)
    assert examples
    assert all("provide the subject" not in row["target"] for row in examples)
    assert any("finite remainder" in row["target"] for row in examples)


def test_tex_pmc_and_boilerplate_extraction_are_prose_only() -> None:
    tex = rb"""\documentclass{article}
\begin{document}
\section{Discussion}
The controlled result remains valid under the stated assumptions, although the boundary case requires a separate argument.
\begin{equation}x^2 + y^2 = z^2\end{equation}
\section{References}
This bibliography sentence must disappear.
\end{document}
"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("paper/main.tex")
        info.size = len(tex)
        archive.addfile(info, io.BytesIO(tex))
    extracted = clean_tex_archive(buffer.getvalue())
    assert "controlled result remains valid" in extracted
    assert "bibliography sentence" not in extracted
    assert "x^2" not in extracted

    pmc = clean_pmc_xml(fixture("pmc.xml"))
    assert "convergent observations" in pmc
    assert "reference-list sentence" not in pmc
    assert "received:" not in clean_prose("Received: 1 January\n\nA substantive result remains.").casefold()


def test_streamed_pubmed_baseline_has_uncompressed_budget(tmp_path: Path) -> None:
    path = tmp_path / "baseline.xml.gz"
    path.write_bytes(gzip.compress(fixture("pubmed.xml")))
    source = PubMedBaselineSource(path, cutoff=CUTOFF)
    page = source.fetch_page(0, 10)
    assert {document.source_id for document in page.documents} == {"31415926", "16180339"}
    assert page.exhausted is True

    constrained = PubMedBaselineSource(path, cutoff=CUTOFF, maximum_uncompressed_bytes=64)
    with pytest.raises(ValueError, match="uncompressed size limit"):
        constrained.fetch_page(0, 10)


def test_collector_ignores_exfat_appledouble_metadata(tmp_path: Path) -> None:
    collector = CorpusCollector(tmp_path / "cache", CollectionConfig(
        maximum_documents_per_source=1,
        maximum_scanned_per_source=1,
        page_size=1,
    ))
    metadata = collector.raw_directory / "arxiv_hep-th" / "._record.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"\x00\x05\x16\x07AppleDouble-not-json")

    assert collector._raw_record_paths() == []


def test_post_cutoff_or_missing_source_cannot_compile(tmp_path: Path) -> None:
    document = SourceDocument(
        provider="arxiv",
        stratum="arxiv:hep-th",
        source_id="2101.1v9",
        title="Unsafe revision",
        authors=("Author",),
        published="2021-01-01",
        latest_version="2022-01-01",
        abstract=(
            "This sentence is deliberately long enough to resemble academic prose, "
            "but its source revision crosses the fixed temporal boundary."
        ),
        body="",
        landing_url="https://arxiv.org/abs/2101.1v9",
        artifact_url="",
        metadata_endpoint=ARXIV_API,
        content_endpoint="",
        query="fixture",
        extraction="atom_abstract",
    )
    with pytest.raises(ValueError, match="post-cutoff"):
        compile_corpus([document], output_path=tmp_path / "bad.jsonl")
