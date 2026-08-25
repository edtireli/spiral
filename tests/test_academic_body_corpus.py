from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.academic_finetune.body_corpus import (
    BODY_ATTESTATION_SCHEMA,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    BodyCorpusError,
    build_body_source_cache,
    compile_body_corpus,
    extract_jats_body,
    extract_tex_body,
    load_body_source_cache,
)
from scripts.academic_finetune.build_body_corpus import parser as body_corpus_parser
from scripts.academic_finetune.sources import (
    SourceDocument,
    canonical_json,
    raw_record_sha256,
    sha256_text,
)
from scripts.academic_finetune.structure_corpus import STRUCTURE_CACHE_SCHEMA
from scripts.academic_finetune.structure_extract import (
    parse_jats_structure,
    parse_tex_structure_archive,
)
from scripts.academic_finetune.text import sentences


class _ExactTokenizer:
    """Small deterministic stand-in for the pinned Qwen chat template."""

    def __init__(self, *, overflow_completion_fragment: str = "") -> None:
        self.overflow_completion_fragment = overflow_completion_fragment
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        if (
            len(messages) == 2
            and self.overflow_completion_fragment
            and self.overflow_completion_fragment
            in str(messages[1]["content"])
        ):
            return list(range(DEFAULT_MAX_SEQUENCE_LENGTH + 1))
        length = 8 + sum(
            len(str(message["content"]).split()) for message in messages
        )
        if len(messages) == 2:
            length += 1
        return list(range(length))


def _token_gate(tokenizer: _ExactTokenizer | None = None) -> dict:
    return {
        "tokenizer": tokenizer or _ExactTokenizer(),
        "tokenizer_identity": {
            "identity": "fixture-qwen-body-chat-template-v1",
            "loader": "injected-test-tokenizer",
        },
    }


def test_sentence_split_keeps_trailing_citations_with_the_preceding_sentence() -> None:
    value = (
        "It suggests a bounded interpretation. [1], [2] Despite this evidence, "
        "the limitation remains material. [3] A final sentence follows."
    )

    assert sentences(value) == [
        "It suggests a bounded interpretation. [1], [2]",
        "Despite this evidence, the limitation remains material. [3]",
        "A final sentence follows.",
    ]
    assert sentences("Smith et al. [1] reported a bounded result.") == [
        "Smith et al. [1] reported a bounded result."
    ]


def _tex_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _document(stratum: str, index: int) -> SourceDocument:
    provider = "pubmed" if stratum == "pubmed" else "arxiv"
    return SourceDocument(
        provider=provider,
        stratum=stratum,
        source_id=f"{stratum}-{index}",
        title=f"Body provenance study {stratum} {index}",
        authors=(f"Independent Author {stratum} {index}",),
        published="2020-02-03",
        latest_version="2021-01-04",
        abstract=f"ABSTRACT-ONLY-{stratum}-{index} must never enter a body target.",
        body="",
        landing_url=f"https://example.invalid/{stratum}/{index}",
        artifact_url=f"https://example.invalid/artifact/{stratum}/{index}",
        metadata_endpoint="fixture:metadata",
        content_endpoint="fixture:content",
        query=(
            "(2020/01/01:2020/12/31[Date - Publication])"
            if stratum == "pubmed"
            else f"fixture:{stratum}"
        ),
        extraction="fixture_metadata_abstract_only",
        license="https://creativecommons.org/licenses/by/4.0/",
    )


def _tex_payload(marker: str) -> bytes:
    source = rf"""\documentclass{{article}}
\title{{{marker} body study}}
\begin{{document}}
\begin{{abstract}}ARTIFACT-ABSTRACT-{marker} must be excluded from prose.\end{{abstract}}
CUSTOM-FRONT-{marker} is unsectioned and must also be excluded from prose.
\section{{Introduction}}
The {marker} analysis defines a controlled physical question and states the
scope under which the comparison is informative. A second sentence records
the assumptions needed to interpret the subsequent result without overclaiming.

The {marker} evidence remains stable under the bounded perturbation considered
in this study. Nevertheless, the conclusion is restricted to the stated regime
and does not establish behavior outside the measured or calculated domain.
\end{{document}}
""".encode()
    return _tex_archive({"paper/main.tex": source})


def _jats_payload(marker: str, *, pmid: str = "pubmed-3", year: int = 2020) -> bytes:
    return f"""<article><front><article-meta>
      <article-id pub-id-type="pmid">{pmid}</article-id>
      <title-group><article-title>{marker} body study</article-title></title-group>
      <pub-date pub-type="epub"><year>{year}</year><month>2</month><day>3</day></pub-date>
      <copyright-year>{year}</copyright-year>
      <abstract><p>ARTIFACT-ABSTRACT-{marker} must be excluded.</p></abstract>
    </article-meta></front><body>
      <p>CUSTOM-FRONT-{marker} is unsectioned and must be excluded.</p>
      <sec><title>Introduction</title>
        <p>The {marker} analysis defines a controlled biomedical question and states the scope under which the comparison is informative. A second sentence records the assumptions needed to interpret the subsequent result without overclaiming.</p>
        <p>The {marker} evidence remains stable under the bounded intervention considered in this study. Nevertheless, the conclusion is restricted to the stated cohort and does not establish behavior outside the measured population.</p>
      </sec>
    </body></article>""".encode()


def _write_structure_record(
    root: Path,
    document: SourceDocument,
    payload: bytes,
) -> None:
    structure_value = (
        parse_jats_structure(payload).to_dict()
        if document.provider == "pubmed"
        else parse_tex_structure_archive(payload).to_dict()
    )
    artifact_hash = hashlib.sha256(payload).hexdigest()
    artifact_name = f"{sha256_text(document.document_id)}.bin"
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / artifact_name).write_bytes(payload)
    directory = document.stratum.replace(":", "_")
    record_path = root / "records" / directory / f"{sha256_text(document.document_id)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": STRUCTURE_CACHE_SCHEMA,
        "document_id": document.document_id,
        "source_stratum": document.stratum,
        "metadata_raw_record_sha256": raw_record_sha256(document),
        "artifact_sha256": artifact_hash,
        "artifact_filename": artifact_name,
        "content_endpoint": document.content_endpoint,
        "status": "ready",
        "structure_sha256": sha256_text(canonical_json(structure_value)),
        "structure": structure_value,
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")


def test_tex_body_excludes_front_matter_assets_and_all_back_matter() -> None:
    payload = _tex_archive({
        "paper/main.tex": rb"""\documentclass{article}
\title{A Main Body Study}
\begin{document}
\begin{abstract}ABSTRACT UNIQUE MARKER that must not enter body prose.\end{abstract}
CUSTOM UNSECTIONED ABSTRACT MARKER that must not enter body prose.
\input{sections/main}
\section*{Acknowledgments}
ACKNOWLEDGMENT UNIQUE MARKER that must not enter body prose.
\appendix
\section{Auxiliary checks}
APPENDIX UNIQUE MARKER that must not enter body prose.
\bibliography{paper}
\end{document}
""",
        "paper/sections/main.tex": rb"""\section{Introduction}
The first main-body paragraph motivates the bounded scientific comparison and
states the regime in which the analysis will be interpreted.

\begin{figure}\caption{FIGURE CAPTION UNIQUE MARKER}\end{figure}

The second main-body paragraph records the principal inference while preserving
the limitation required by the available evidence \citep[see][]{clean-source}.

The broken mathematical paragraph obtains $x+y$ at leading order and must be rejected.

The broken object-reference paragraph follows Fig.~\ref{fig:missing} and must be rejected.
\subsection{Controlled $x$ result}
The nested main-body paragraph explains why the observed response supports the
stated conclusion without extending beyond the tested domain.

Aizawa is grateful to the host institute for its invitation and warm hospitality.

The positive cosmological canstant was deliberately misspelled in the source.
""",
    })

    extraction = extract_tex_body(payload)

    assert extraction.title == "A Main Body Study"
    assert [paragraph.section_title for paragraph in extraction.paragraphs] == [
        "Introduction",
        "Introduction",
        "Untitled section",
    ]
    assert [paragraph.section_path for paragraph in extraction.paragraphs] == [
        (1,),
        (1,),
        (1, 1),
    ]
    assert [paragraph.document_paragraph_index for paragraph in extraction.paragraphs] == [
        1,
        2,
        3,
    ]
    assert "[1]" in extraction.paragraphs[1].text
    assert extraction.provenance["paragraph_rejection_counts"]["inline_math"] == 1
    assert extraction.provenance["paragraph_rejection_counts"]["object_cross_reference"] == 1
    assert extraction.provenance["paragraph_rejection_counts"]["acknowledgment_prose"] == 1
    assert extraction.provenance["paragraph_rejection_counts"]["known_source_text_defect"] == 1
    excluded = extraction.body_text
    for marker in (
        "ABSTRACT UNIQUE MARKER",
        "CUSTOM UNSECTIONED ABSTRACT MARKER",
        "FIGURE CAPTION UNIQUE MARKER",
        "ACKNOWLEDGMENT UNIQUE MARKER",
        "APPENDIX UNIQUE MARKER",
    ):
        assert marker not in excluded


def test_jats_body_uses_nested_section_provenance_and_excludes_nonbody() -> None:
    payload = b"""<article><front><article-meta>
      <title-group><article-title>A Biomedical Body Study</article-title></title-group>
      <abstract><p>ABSTRACT UNIQUE MARKER must be excluded.</p></abstract>
    </article-meta></front><body>
      <p>CUSTOM UNSECTIONED ABSTRACT MARKER must be excluded.</p>
      <custom-wrapper><sec><title>Methods</title>
        <p>The mathematical estimate <inline-formula><tex-math>x+y</tex-math></inline-formula> is deliberately rejected from style supervision.</p>
        <p>This preceding fragment introduces the formula and must be rejected from style supervision:</p>
        <disp-formula><tex-math>x+y</tex-math></disp-formula>
        <p>This following fragment depends on the omitted formula and must also be rejected from style supervision.</p>
        <p>The object description follows <xref ref-type="fig" rid="f1">Figure 1</xref> and is deliberately rejected from style supervision.</p>
        <p>The main methods paragraph defines a bounded study cohort and records the intervention used for the controlled comparison [<xref ref-type="bibr" rid="r1">1</xref>].</p>
        <fig><caption><p>FIGURE CAPTION UNIQUE MARKER</p></caption></fig>
        <wrapper><sec><title>Analysis</title>
          <p>The nested analysis paragraph explains the estimation procedure and preserves the uncertainty needed to interpret the result.</p>
          <table-wrap><caption><p>TABLE CAPTION UNIQUE MARKER</p></caption></table-wrap>
        </sec></wrapper>
      </sec></custom-wrapper>
      <sec sec-type="ack"><title>Acknowledgments</title><p>ACK UNIQUE MARKER</p></sec>
      <sec sec-type="appendix"><title>Appendix A</title><p>APPENDIX UNIQUE MARKER</p></sec>
    </body><back><ref-list><ref><mixed-citation>REFERENCE UNIQUE MARKER</mixed-citation></ref></ref-list></back>
    </article>"""

    extraction = extract_jats_body(payload)

    assert [paragraph.section_title for paragraph in extraction.paragraphs] == [
        "Methods",
        "Analysis",
    ]
    assert [paragraph.section_path for paragraph in extraction.paragraphs] == [
        (1,),
        (1, 1),
    ]
    assert "[1]" in extraction.paragraphs[0].text
    assert "[[1]]" not in extraction.paragraphs[0].text
    assert extraction.provenance["paragraph_rejection_counts"]["inline_or_display_math"] == 1
    assert extraction.provenance["paragraph_rejection_counts"]["object_cross_reference"] == 1
    assert extraction.provenance["paragraph_rejection_counts"]["display_math_context"] == 2
    for marker in (
        "ABSTRACT UNIQUE MARKER",
        "CUSTOM UNSECTIONED ABSTRACT MARKER",
        "FIGURE CAPTION UNIQUE MARKER",
        "TABLE CAPTION UNIQUE MARKER",
        "ACK UNIQUE MARKER",
        "APPENDIX UNIQUE MARKER",
        "REFERENCE UNIQUE MARKER",
    ):
        assert marker not in extraction.body_text


def test_large_unbalanced_inline_math_remains_bounded_and_is_rejected() -> None:
    malformed = (
        "$"
        + "The bounded evidence remains interpretable despite malformed syntax. " * 18_000
        + "FINAL BODY SUFFIX MARKER remains available for downstream extraction."
    )
    source = (
        "\\documentclass{article}\n\\begin{document}\n\\section{Main}\n"
        + malformed
        + "\n\\end{document}\n"
    ).encode()

    started = time.monotonic()
    extraction = extract_tex_body(source)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0
    assert extraction.paragraphs == ()
    assert extraction.provenance["paragraph_rejection_counts"] == {"inline_math": 1}


def test_body_cache_and_corpus_attest_body_only_provenance(tmp_path: Path) -> None:
    structure_cache = tmp_path / "structures"
    body_cache = tmp_path / "bodies"
    documents = [
        _document("arxiv:hep-ph", 1),
        _document("arxiv:hep-th", 2),
        _document("pubmed", 3),
    ]
    payloads = [_tex_payload("HEP-PH"), _tex_payload("HEP-TH"), _jats_payload("PUBMED")]
    for document, payload in zip(documents, payloads, strict=True):
        _write_structure_record(structure_cache, document, payload)

    sources, cache_manifest = build_body_source_cache(
        documents,
        structure_cache=structure_cache,
        body_cache=body_cache,
    )
    loaded_sources, loaded_manifest = load_body_source_cache(body_cache)
    assert loaded_sources == sources
    assert loaded_manifest == cache_manifest
    assert cache_manifest["body_only_attestation"]["abstract_fields_empty"] is True
    assert cache_manifest["counts"]["body_source_documents"] == 3

    output = tmp_path / "body.jsonl"
    manifest = compile_body_corpus(
        sources,
        output_path=output,
        body_cache_manifest=cache_manifest,
        balance_sources=False,
        require_trainable_splits=False,
        minimum_documents_per_split_stratum=0,
        minimum_examples_per_split_stratum=0,
        minimum_components_per_split_stratum=0,
        **_token_gate(),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert rows
    assert manifest["body_only_attestation"]["schema_version"] == BODY_ATTESTATION_SCHEMA
    assert manifest["counts"]["body_only_example_ratio"] == 1.0
    assert manifest["counts"]["abstract_examples"] == 0
    audit = manifest["body_only_attestation"]["deterministic_quality_audit"]
    assert audit["sample_examples"] == audit["automated_artifact_gate_passed"]
    assert (tmp_path / audit["filename"]).is_file()
    assert all(row["provenance"]["body_only"] is True for row in rows)
    assert all(row["provenance"]["exact_training_tokens"] <= 448 for row in rows)
    assert all(row["provenance"]["body_section_path"] for row in rows)
    assert all(row["provenance"]["body_document_paragraph_index"] >= 1 for row in rows)
    serialized = output.read_bytes()
    second = tmp_path / "body-second.jsonl"
    second_manifest = compile_body_corpus(
        sources,
        output_path=second,
        body_cache_manifest=cache_manifest,
        balance_sources=False,
        require_trainable_splits=False,
        minimum_documents_per_split_stratum=0,
        minimum_examples_per_split_stratum=0,
        minimum_components_per_split_stratum=0,
        **_token_gate(),
    )
    assert second.read_bytes() == serialized
    assert second_manifest["corpus_sha256"] == manifest["corpus_sha256"]
    all_targets = "\n".join(str(row["target"]) for row in rows)
    assert "ABSTRACT-ONLY" not in all_targets
    assert "ARTIFACT-ABSTRACT" not in all_targets
    assert "CUSTOM-FRONT" not in all_targets


def test_exact_token_gate_rejects_whole_rows_before_split_and_balance(
    tmp_path: Path,
) -> None:
    structure_cache = tmp_path / "structures"
    documents = [
        _document("arxiv:hep-ph", 1),
        _document("arxiv:hep-th", 2),
        _document("pubmed", 3),
    ]
    payloads = [_tex_payload("HEP-PH"), _tex_payload("HEP-TH"), _jats_payload("PUBMED")]
    for document, payload in zip(documents, payloads, strict=True):
        _write_structure_record(structure_cache, document, payload)
    sources, cache_manifest = build_body_source_cache(
        documents,
        structure_cache=structure_cache,
        body_cache=tmp_path / "bodies",
    )
    tokenizer = _ExactTokenizer(
        overflow_completion_fragment="defines a controlled biomedical question"
    )
    output = tmp_path / "exact-gated-body.jsonl"

    manifest = compile_body_corpus(
        sources,
        output_path=output,
        body_cache_manifest=cache_manifest,
        require_trainable_splits=False,
        minimum_documents_per_split_stratum=0,
        minimum_examples_per_split_stratum=0,
        minimum_components_per_split_stratum=0,
        **_token_gate(tokenizer),
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    gate = manifest["gates"]["exact_training_token_gate"]
    assert gate["stage"] == (
        "after_candidate_dedup_before_author_split_and_source_example_balance"
    )
    assert gate["candidates_rejected"] == 1
    assert gate["candidates_rejected_by_stratum"] == {"pubmed": 1}
    assert gate["candidates_rejected_by_task_type"] in (
        {"sentence": 1},
        {"paragraph": 1},
    )
    assert gate["largest_candidate_tokens"] == 449
    assert gate["largest_accepted_tokens"] <= 448
    assert gate["derived_rows"] == 0
    assert gate["tokenizer"]["identity"] == "fixture-qwen-body-chat-template-v1"
    assert manifest["body_only_attestation"][
        "exact_token_gate_before_split_and_source_example_balance"
    ] is True
    assert manifest["body_only_attestation"]["targets_truncated_or_partitioned"] == 0
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert all(row["provenance"]["exact_training_tokens"] <= 448 for row in rows)
    assert all("derivation" not in row for row in rows)
    assert all(
        "defines a controlled biomedical question" not in str(row["target"])
        for row in rows
    )
    assert len(set(manifest["counts"]["by_stratum"].values())) == 1
    full_calls = [messages for messages, _kwargs in tokenizer.calls if len(messages) == 2]
    prompt_calls = [messages for messages, _kwargs in tokenizer.calls if len(messages) == 1]
    assert len(full_calls) == gate["candidates_measured"]
    assert len(prompt_calls) == gate["candidates_measured"]


def test_body_compiler_refuses_an_unauthenticated_exact_token_gate(
    tmp_path: Path,
) -> None:
    structure_cache = tmp_path / "structures"
    documents = [
        _document("arxiv:hep-ph", 1),
        _document("arxiv:hep-th", 2),
        _document("pubmed", 3),
    ]
    for document, payload in zip(
        documents,
        [_tex_payload("HEP-PH"), _tex_payload("HEP-TH"), _jats_payload("PUBMED")],
        strict=True,
    ):
        _write_structure_record(structure_cache, document, payload)
    sources, _manifest = build_body_source_cache(
        documents,
        structure_cache=structure_cache,
        body_cache=tmp_path / "bodies",
    )

    with pytest.raises(BodyCorpusError, match="tokenizer_model_path or an injected tokenizer"):
        compile_body_corpus(sources, output_path=tmp_path / "missing.jsonl")
    with pytest.raises(BodyCorpusError, match="explicit tokenizer_identity"):
        compile_body_corpus(
            sources,
            output_path=tmp_path / "missing-identity.jsonl",
            tokenizer=_ExactTokenizer(),
        )


def test_body_corpus_cli_requires_the_pinned_tokenizer_model() -> None:
    required = [
        "--metadata-cache", "metadata",
        "--structure-cache", "structures",
        "--body-cache", "bodies",
        "--output", "corpus.jsonl",
    ]
    with pytest.raises(SystemExit):
        body_corpus_parser().parse_args(required)
    parsed = body_corpus_parser().parse_args([
        *required,
        "--tokenizer-model", "Qwen3.8-27B-4bit",
    ])
    assert parsed.tokenizer_model == Path("Qwen3.8-27B-4bit")
    assert parsed.max_sequence_length == 448


def test_crosslists_are_collapsed_only_when_bibliography_and_artifact_agree(
    tmp_path: Path,
) -> None:
    structure_cache = tmp_path / "structures"
    winner = _document("arxiv:hep-ph", 5)
    duplicate = replace(winner, stratum="arxiv:hep-th", query="fixture:hep-th")
    payload = _tex_payload("CROSSLIST")
    _write_structure_record(structure_cache, winner, payload)

    sources, manifest = build_body_source_cache(
        [duplicate, winner],
        structure_cache=structure_cache,
        body_cache=tmp_path / "bodies",
    )
    assert len(sources) == 1
    assert sources[0].document.stratum == "arxiv:hep-ph"
    assert manifest["counts"]["cross_list_duplicates_removed"] == 1
    assert manifest["cross_list_deduplication"][0]["selected_stratum"] == "arxiv:hep-ph"

    disagreeing = replace(duplicate, title="A conflicting cross-list title")
    with pytest.raises(BodyCorpusError, match="duplicate source documents disagree"):
        build_body_source_cache(
            [winner, disagreeing],
            structure_cache=structure_cache,
            body_cache=tmp_path / "bad-bodies",
        )


def test_body_cache_rejects_tampered_artifact(tmp_path: Path) -> None:
    structure_cache = tmp_path / "structures"
    document = _document("arxiv:hep-ph", 7)
    payload = _tex_payload("TAMPER")
    _write_structure_record(structure_cache, document, payload)
    artifact = next((structure_cache / "artifacts").iterdir())
    artifact.write_bytes(payload + b"tampered")

    with pytest.raises(BodyCorpusError, match="artifact hash mismatch"):
        build_body_source_cache(
            [document],
            structure_cache=structure_cache,
            body_cache=tmp_path / "bodies",
        )


def test_jats_post_cutoff_date_overrides_false_pre_cutoff_metadata(
    tmp_path: Path,
) -> None:
    structure_cache = tmp_path / "structures"
    document = _document("pubmed", 9)
    payload = _jats_payload("POST-CUTOFF", pmid=document.source_id, year=2023)
    _write_structure_record(structure_cache, document, payload)

    sources, manifest = build_body_source_cache(
        [document],
        structure_cache=structure_cache,
        body_cache=tmp_path / "bodies",
    )

    assert sources == []
    temporal = manifest["jats_temporal_attestation"]
    assert temporal["artifacts_checked"] == 1
    assert temporal["eligible_artifacts"] == 0
    assert temporal["rejected_artifacts"] == 1
    assert temporal["rejection_flag_counts"] == {
        "metadata_publication_year_mismatch": 1,
        "post_cutoff_publication_or_copyright_year": 1,
        "query_publication_range_mismatch": 1,
    }
    assert temporal["receipts"][0]["jats_attested_years"] == [2023]
