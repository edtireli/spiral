from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.academic_finetune.build_structure_corpus import parser as corpus_parser
from scripts.academic_finetune.sources import (
    SourceDocument,
    canonical_json,
    raw_record_sha256,
    sha256_text,
)
from scripts.academic_finetune.structure_corpus import (
    ALL_TASKS,
    PAPER_STRUCTURE_SCHEMA,
    STRUCTURE_CACHE_SCHEMA,
    STRUCTURE_CORPUS_SCHEMA,
    STRUCTURE_MANIFEST_SCHEMA,
    STRUCTURE_PROMPT_CONTRACT,
    StructureHydrator,
    compile_structure_corpus,
    load_structure_cache,
    normalize_section_role,
)


def _tex_archive(source: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("paper.tex")
        info.size = len(source)
        archive.addfile(info, io.BytesIO(source))
    return buffer.getvalue()


class _Fetcher:
    def __init__(self, payload: bytes = b"", *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> bytes:
        if self.fail:
            raise AssertionError("resumed hydration attempted a network request")
        self.calls.append((url, dict(params or {})))
        return self.payload


class _ExactTokenizer:
    """Deterministic chat-template tokenizer used without model weights."""

    def __init__(
        self, *, overflow_task: str = "", overflow_replay_fragment: str = "",
        exact_limit_task: str = "",
    ) -> None:
        self.overflow_task = overflow_task
        self.overflow_replay_fragment = overflow_replay_fragment
        self.exact_limit_task = exact_limit_task
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        if len(messages) == 2:
            prompt = str(messages[0]["content"])
            completion = str(messages[1]["content"])
            if (
                self.overflow_task
                and f"Task: {self.overflow_task}\n" in prompt
            ) or (
                self.overflow_replay_fragment
                and self.overflow_replay_fragment in completion
            ):
                return list(range(449))
            if (
                self.exact_limit_task
                and f"Task: {self.exact_limit_task}\n" in prompt
            ):
                return list(range(448))
        length = 8 + sum(len(str(message["content"]).split()) for message in messages)
        if len(messages) == 2:
            length += 1
        return list(range(length))


def _token_gate(tokenizer: _ExactTokenizer | None = None) -> dict:
    return {
        "tokenizer": tokenizer or _ExactTokenizer(),
        "tokenizer_identity": {
            "identity": "fixture-qwen-chat-template-v1",
            "loader": "injected-test-tokenizer",
        },
    }


def _document(stratum: str, index: int) -> SourceDocument:
    provider = "pubmed" if stratum == "pubmed" else "arxiv"
    return SourceDocument(
        provider=provider,
        stratum=stratum,
        source_id=f"{stratum}-{index}",
        title=f"Controlled architecture study {stratum} {index}",
        authors=(f"Independent Architect {stratum} {index}",),
        published="2020-02-03",
        latest_version="2021-01-04",
        abstract=(
            f"This bounded {stratum} study {index} connects a stated question to a controlled "
            "analysis and preserves the principal limitation when interpreting the result."
        ),
        body="",
        landing_url=f"https://example.invalid/{stratum}/{index}",
        artifact_url=(
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{7000000 + index}/"
            if provider == "pubmed"
            else f"https://export.arxiv.org/e-print/{index}"
        ),
        metadata_endpoint="fixture:metadata",
        content_endpoint="fixture:content",
        query="fixture",
        extraction="fixture_structure",
        license="https://creativecommons.org/licenses/by/4.0/",
    )


def _node(
    title: str,
    path: list[int],
    words: int,
    *,
    children: list[dict] | None = None,
    figures: int = 0,
    tables: int = 0,
) -> dict:
    children = children or []
    paragraph_count = max(1, words // 100)
    direct_paragraph_count = max(
        0, paragraph_count - sum(child["paragraph_count"] for child in children)
    )
    placements = [
        {
            "kind": kind,
            "order": order,
            "after_paragraph": min(order, direct_paragraph_count),
            "identifier": f"{kind}-{'.'.join(map(str, path))}-{order}",
        }
        for kind, count in (("figure", figures), ("table", tables))
        for order in range(1, count + 1)
    ]
    return {
        "title": title,
        "level": len(path),
        "order": path[-1],
        "path": path,
        "included_in_main": True,
        "exclusion_reason": "",
        "direct_word_count": words - sum(child["word_count"] for child in children),
        "direct_paragraph_count": direct_paragraph_count,
        "word_count": words,
        "paragraph_count": paragraph_count,
        "direct_figure_count": figures,
        "direct_table_count": tables,
        "figure_count": figures + sum(child["figure_count"] for child in children),
        "table_count": tables + sum(child["table_count"] for child in children),
        "placements": placements,
        "children": children,
    }


def _structure(document: SourceDocument) -> dict:
    biomedical = document.provider == "pubmed"
    middle = "Methods" if biomedical else "Theoretical Framework"
    third = "Results" if biomedical else "Phenomenological Analysis"
    sections = [
        _node("Introduction", [1], 320),
        _node(
            middle,
            [2],
            760,
            figures=1,
            children=[
                _node("Boundary Conditions", [2, 1], 330),
                _node("Effective Action" if not biomedical else "Study Cohort", [2, 2], 390),
            ],
        ),
        _node(third, [3], 680, figures=1, tables=1),
        _node("Discussion" if biomedical else "Conclusions and Outlook", [4], 360),
    ]
    return {
        "schema_version": PAPER_STRUCTURE_SCHEMA,
        "source_format": "pmc_jats" if biomedical else "arxiv_tex",
        "title": document.title,
        "sections": sections,
        "appendices": [],
        "abstract_word_count": 58,
        "abstract_paragraph_count": 1,
        "unsectioned_word_count": 0,
        "unsectioned_paragraph_count": 0,
        "unsectioned_placements": [],
        "counts": {
            "main_words": 2120,
            "main_paragraphs": 19,
            "main_figures": 2,
            "main_tables": 1,
        },
        "provenance": {
            "parser": "fixture",
            "parser_version": "1",
            "source_sha256": sha256_text(f"artifact:{document.document_id}"),
            "source_bytes": 1,
            "root_member": "fixture",
            "included_members": ["fixture"],
            "expanded_source_sha256": "",
            "warnings": [],
            "identifiers": [],
        },
    }


def _cache_record(document: SourceDocument) -> dict:
    structure = _structure(document)
    return {
        "schema_version": STRUCTURE_CACHE_SCHEMA,
        "document_id": document.document_id,
        "metadata_raw_record_sha256": raw_record_sha256(document),
        "artifact_sha256": sha256_text(f"artifact:{document.document_id}"),
        "content_endpoint": document.content_endpoint,
        "status": "ready",
        "structure_sha256": sha256_text(canonical_json(structure)),
        "structure": structure,
    }


def _replay(document: SourceDocument, index: int) -> dict:
    target = (
        f"The controlled analysis for document {document.source_id} replay {index} "
        "supports the bounded conclusion while retaining the stated limitation."
    )
    return {
        "schema_version": "spiral.academic-plan-prose.v1",
        "example_id": sha256_text(f"replay:{document.document_id}:{index}"),
        "split": "train",
        "task_type": "sentence",
        "source": {
            "provider": document.provider,
            "stratum": document.stratum,
            "source_id": document.source_id,
            "landing_url": document.landing_url,
            "artifact_url": document.artifact_url,
            "query": document.query,
        },
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "authors": list(document.authors),
            "published": document.published,
            "latest_version": document.latest_version,
            "metadata_revised": "",
        },
        "input": {
            "context": f"The preceding argument defines case {index}.",
            "claims": ["bounded conclusion", "limitation retained"],
            "rhetorical_relation": "qualification",
            "certainty": "bounded",
            "citation_count": 0,
            "citation_slots": [],
            "construction_method": "bounded_semantic_roles_v3",
        },
        "target": target,
        "provenance": {"raw_record_sha256": raw_record_sha256(document)},
    }


def test_role_normalization_is_semantic_and_discipline_aware() -> None:
    assert normalize_section_role("Materials and Methods", discipline="biomedical_science") == "methods"
    assert normalize_section_role("Methods", discipline="theoretical_physics") == "method_or_setup"
    assert normalize_section_role("Dualities at Finite Density", discipline="theoretical_physics") == "domain_development"
    assert normalize_section_role("Cohort Characteristics", discipline="biomedical_science") == "domain_section"


def test_structure_compiler_is_deterministic_nested_bounded_and_author_safe(tmp_path: Path) -> None:
    documents = [
        _document(stratum, index)
        for stratum in ("arxiv:hep-ph", "arxiv:hep-th", "pubmed")
        for index in range(5)
    ]
    structures = {document.document_id: _cache_record(document) for document in documents}
    replay = [_replay(document, index) for document in documents for index in range(2)]
    crosslisted = replace(
        documents[0],
        stratum="arxiv:hep-th",
        query="fixture cross-list",
    )
    documents_with_crosslist = [*documents, crosslisted]
    output = tmp_path / "paper-structure.jsonl"

    manifest = compile_structure_corpus(
        documents_with_crosslist,
        structures,
        output_path=output,
        replay_records=replay,
        minimum_per_split_stratum=1,
        **_token_gate(),
    )
    first_corpus = output.read_bytes()
    first_manifest = (tmp_path / "paper-structure.jsonl.manifest.json").read_bytes()
    assert manifest["schema_version"] == STRUCTURE_MANIFEST_SCHEMA
    assert manifest["corpus_schema_version"] == STRUCTURE_CORPUS_SCHEMA
    assert manifest["prompt_contract"] == STRUCTURE_PROMPT_CONTRACT
    assert manifest["trainable"] is True
    assert manifest["counts"]["prose_replay_ratio"] == pytest.approx(0.2)
    assert manifest["counts"]["cross_list_document_ids"] == 1
    assert manifest["counts"]["cross_list_duplicates_removed"] == 1
    assert manifest["deduplication"]["cross_list_pairs"] == {
        "arxiv:hep-ph|arxiv:hep-th": 1,
    }
    assert manifest["deduplication"]["cross_list_duplicates"][0][
        "selected_stratum"] == "arxiv:hep-ph"
    assert set(manifest["counts"]["by_task_type"]) == set(ALL_TASKS)
    exact_gate = manifest["gates"]["exact_training_token_gate"]
    assert exact_gate["max_sequence_length"] == 448
    assert exact_gate["largest_accepted_tokens"] <= 448
    assert exact_gate["candidates_rejected"] == 0
    assert exact_gate["derived_rows"] == 0
    assert exact_gate["tokenizer"]["identity"] == "fixture-qwen-chat-template-v1"

    records = [json.loads(line) for line in first_corpus.decode().splitlines()]
    assert records and all(record["schema_version"] == STRUCTURE_CORPUS_SCHEMA for record in records)
    document_splits: dict[str, set[str]] = defaultdict(set)
    author_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        document_splits[record["document"]["document_id"]].add(record["split"])
        for author in record["document"]["authors"]:
            author_splits[author.casefold()].add(record["split"])
        assert record["provenance"]["exact_training_tokens"] <= 448
        assert 0 < record["provenance"]["exact_prompt_offset"] < record[
            "provenance"]["exact_training_tokens"]
        if record["task_type"] == "prose_replay":
            assert isinstance(record["target"], str) and record["target"]
            assert record["input"]["unit"] in {"sentence", "paragraph"}
        else:
            assert isinstance(record["target"], dict)
            schema = record["input"]["response_schema"]
            assert schema["type"] == "object"
            assert set(schema["required"]) <= set(record["target"])
    assert all(len(splits) == 1 for splits in document_splits.values())
    assert all(len(splits) == 1 for splits in author_splits.values())

    nested_order = [
        record for record in records
        if record["task_type"] == "order_structure" and record["target"]["parent_id"] != "paper"
    ]
    nested_restore = [
        record for record in records
        if record["task_type"] == "restore_section"
        and record["target"]["missing_section"]["parent_id"] != "paper"
    ]
    assert nested_order and nested_restore
    assert all(record["target"]["parent_path"] == [2] for record in nested_order)
    assert all(len(record["target"]["missing_section"]["path"]) == 2 for record in nested_restore)

    compile_structure_corpus(
        reversed(documents_with_crosslist),
        structures,
        output_path=output,
        replay_records=reversed(replay),
        minimum_per_split_stratum=1,
        **_token_gate(),
    )
    assert output.read_bytes() == first_corpus
    assert (tmp_path / "paper-structure.jsonl.manifest.json").read_bytes() == first_manifest


def test_exact_gate_rejects_whole_structure_and_replay_candidates_before_mix_and_split(
    tmp_path: Path,
) -> None:
    documents = [
        _document(stratum, index)
        for stratum in ("arxiv:hep-ph", "arxiv:hep-th", "pubmed")
        for index in range(5)
    ]
    structures = {document.document_id: _cache_record(document) for document in documents}
    replay = [_replay(document, index) for document in documents for index in range(2)]
    tokenizer = _ExactTokenizer(
        overflow_task="repair_structure",
        overflow_replay_fragment="replay 1 supports",
        exact_limit_task="recognize_role",
    )
    output = tmp_path / "exact-gated.jsonl"

    manifest = compile_structure_corpus(
        documents,
        structures,
        output_path=output,
        replay_records=replay,
        minimum_per_split_stratum=1,
        require_trainable_splits=False,
        tokenizer=tokenizer,
        tokenizer_identity={
            "identity": "fixture-selective-overflow-v1",
            "loader": "injected-test-tokenizer",
        },
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    gate = manifest["gates"]["exact_training_token_gate"]
    # Six structure candidates and two replay candidates are measured per source
    # document before either the 4:1 selector or connected-author split runs.
    assert gate["candidates_measured"] == len(documents) * 8
    assert gate["candidates_rejected_by_task_type"] == {
        "prose_replay": len(documents),
        "repair_structure": len(documents),
    }
    assert gate["candidates_rejected"] == len(documents) * 2
    assert gate["largest_candidate_tokens"] == 449
    assert gate["largest_accepted_tokens"] == 448
    assert gate["derived_rows"] == 0
    assert manifest["rejections"]["examples"] == {
        "over_exact_token_budget:prose_replay": len(documents),
        "over_exact_token_budget:repair_structure": len(documents),
    }
    assert manifest["counts"]["prose_replay_ratio"] == pytest.approx(0.2)
    assert all(row["task_type"] != "repair_structure" for row in rows)
    assert all("replay 1 supports" not in str(row["target"]) for row in rows)
    assert all(row["provenance"]["exact_training_tokens"] <= 448 for row in rows)
    assert len({row["example_id"] for row in rows}) == len(rows)
    assert not any("derivation" in row for row in rows)

    full_calls = [messages for messages, _kwargs in tokenizer.calls if len(messages) == 2]
    prompt_only_calls = [messages for messages, _kwargs in tokenizer.calls if len(messages) == 1]
    assert len(full_calls) == gate["candidates_measured"]
    assert len(prompt_only_calls) == gate["candidates_measured"]
    assert any(
        messages[0]["content"].startswith(
            "Complete one academic paper-structure task from the evidence below.\n"
        )
        for messages in full_calls
    )
    assert any(
        messages[0]["content"].startswith("Reconstruct one missing unit")
        for messages in full_calls
    )


def test_structure_compiler_refuses_an_unauthenticated_token_gate(tmp_path: Path) -> None:
    documents = [
        _document(stratum, 0)
        for stratum in ("arxiv:hep-ph", "arxiv:hep-th", "pubmed")
    ]
    structures = {document.document_id: _cache_record(document) for document in documents}

    with pytest.raises(ValueError, match="tokenizer_model_path or an injected tokenizer"):
        compile_structure_corpus(
            documents,
            structures,
            output_path=tmp_path / "missing-tokenizer.jsonl",
            replay_records=[],
            require_trainable_splits=False,
            minimum_per_split_stratum=1,
        )
    with pytest.raises(ValueError, match="explicit tokenizer_identity"):
        compile_structure_corpus(
            documents,
            structures,
            output_path=tmp_path / "missing-identity.jsonl",
            replay_records=[],
            require_trainable_splits=False,
            minimum_per_split_stratum=1,
            tokenizer=_ExactTokenizer(),
        )


def test_structure_compiler_cli_requires_the_pinned_tokenizer_model() -> None:
    required = [
        "--metadata-cache", "metadata",
        "--structure-cache", "structures",
        "--output", "corpus.jsonl",
    ]
    with pytest.raises(SystemExit):
        corpus_parser().parse_args(required)
    parsed = corpus_parser().parse_args([
        *required,
        "--tokenizer-model", "Qwen3.8-27B-4bit",
        "--compile-only",
    ])
    assert parsed.tokenizer_model == Path("Qwen3.8-27B-4bit")
    assert parsed.max_sequence_length == 448


def test_structure_cache_loader_hash_checks_and_skips_unavailable(tmp_path: Path) -> None:
    document = _document("arxiv:hep-th", 1)
    ready = _cache_record(document)
    record_directory = tmp_path / "records" / "arxiv_hep-th"
    record_directory.mkdir(parents=True)
    ready_path = record_directory / "ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    unavailable = {
        "schema_version": STRUCTURE_CACHE_SCHEMA,
        "document_id": "pubmed:no-pmc",
        "status": "unavailable",
        "reason": "pmcid_missing",
    }
    (record_directory / "unavailable.json").write_text(json.dumps(unavailable), encoding="utf-8")
    assert set(load_structure_cache(tmp_path)) == {document.document_id}

    ready["structure"]["title"] = "tampered"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_structure_cache(tmp_path)


def test_structure_cache_collapses_identical_crosslists_and_rejects_disagreement(
    tmp_path: Path,
) -> None:
    hep_ph = _document("arxiv:hep-ph", 2)
    hep_th = replace(hep_ph, stratum="arxiv:hep-th", query="cross-list")
    ph_record = _cache_record(hep_ph)
    ph_record["source_stratum"] = hep_ph.stratum
    th_record = dict(ph_record)
    th_record["source_stratum"] = hep_th.stratum
    th_record["metadata_raw_record_sha256"] = raw_record_sha256(hep_th)
    for stratum, record in (("arxiv_hep-ph", ph_record), ("arxiv_hep-th", th_record)):
        directory = tmp_path / "records" / stratum
        directory.mkdir(parents=True)
        (directory / "same.json").write_text(json.dumps(record), encoding="utf-8")

    loaded = load_structure_cache(tmp_path)
    assert set(loaded) == {hep_ph.document_id}
    assert loaded[hep_ph.document_id]["source_stratum"] == "arxiv:hep-ph"
    assert loaded.duplicate_records == [{
        "document_id": hep_ph.document_id,
        "selected_stratum": "arxiv:hep-ph",
        "duplicate_stratum": "arxiv:hep-th",
        "selected_record": "arxiv_hep-ph/same.json",
        "duplicate_record": "arxiv_hep-th/same.json",
        "artifact_sha256": ph_record["artifact_sha256"],
        "structure_sha256": ph_record["structure_sha256"],
    }]

    th_record["artifact_sha256"] = "f" * 64
    (tmp_path / "records" / "arxiv_hep-th" / "same.json").write_text(
        json.dumps(th_record), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_structure_cache(tmp_path)


def test_structure_hydrator_retains_official_artifact_and_resumes_without_network(
    tmp_path: Path,
) -> None:
    document = _document("arxiv:hep-th", 9)
    payload = _tex_archive(
        rb"""\documentclass{article}
\begin{document}
\section{Introduction}
This paragraph motivates a controlled theoretical comparison.
\section{Formalism}
This paragraph defines the bounded model and its assumptions.
\section{Conclusions}
This paragraph states the result and retains its limitation.
\end{document}
"""
    )
    arxiv = _Fetcher(payload)
    pubmed = _Fetcher(fail=True)
    cache = tmp_path / "structure-cache"
    first = StructureHydrator(
        cache,
        arxiv_fetcher=arxiv,
        pubmed_fetcher=pubmed,
    ).hydrate_one(document)
    assert first["status"] == "ready"
    assert first["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(arxiv.calls) == 1
    assert list((cache / "artifacts").glob("*.bin"))
    assert set(load_structure_cache(cache)) == {document.document_id}

    resumed = StructureHydrator(
        cache,
        arxiv_fetcher=_Fetcher(fail=True),
        pubmed_fetcher=_Fetcher(fail=True),
    ).hydrate_one(document)
    assert resumed == first


def test_structure_hydrator_uses_but_does_not_persist_ncbi_api_key(tmp_path: Path) -> None:
    document = _document("pubmed", 4)
    pmc_payload = (
        Path(__file__).parents[1] / "scripts" / "academic_finetune" / "fixtures" / "pmc.xml"
    ).read_bytes()
    pubmed = _Fetcher(pmc_payload)
    record = StructureHydrator(
        tmp_path / "pmc-cache",
        arxiv_fetcher=_Fetcher(fail=True),
        pubmed_fetcher=pubmed,
        ncbi_api_key="fixture-secret-key",
    ).hydrate_one(document)
    assert record["status"] == "ready"
    assert pubmed.calls[0][1]["api_key"] == "fixture-secret-key"
    record_text = next((tmp_path / "pmc-cache" / "records").glob("*/*.json")).read_text()
    assert "fixture-secret-key" not in record_text
