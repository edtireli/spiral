"""Deterministic paper-architecture corpus collection and compilation.

This module deliberately keeps three artifacts separate:

* :class:`SourceDocument` JSON records contain bibliographic metadata;
* structure-cache records contain the parsed TeX/JATS hierarchy;
* the emitted JSONL contains bounded structure-learning tasks.

The separation lets a parser upgrade rebuild structure records from cached
official artifacts without changing the already-audited prose corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

from scripts.academic_finetune.corpus import _author_safe_splits, _normal_author
from scripts.academic_finetune.sources import (
    ARXIV_EPRINT,
    NCBI_EFETCH,
    PoliteFetcher,
    SourceDocument,
    canonical_json,
    load_source_documents,
    raw_record_sha256,
    sha256_text,
)

STRUCTURE_CORPUS_SCHEMA = "spiral.academic-paper-structure.v1"
STRUCTURE_MANIFEST_SCHEMA = "spiral.academic-structure-corpus-manifest.v1"
STRUCTURE_CACHE_SCHEMA = "spiral.academic-paper-structure-cache.v1"
STRUCTURE_PROMPT_CONTRACT = "spiral.academic-paper-blueprint.v1"
PAPER_STRUCTURE_SCHEMA = "spiral.paper-structure.v1"

STRUCTURE_TASKS = (
    "recognize_role",
    "order_structure",
    "budget_structure",
    "restore_section",
    "repair_structure",
    "brief_to_blueprint",
)
ALL_TASKS = STRUCTURE_TASKS + ("prose_replay",)
REQUIRED_STRATA = ("arxiv:hep-ph", "arxiv:hep-th", "pubmed")
SPLITS = ("train", "validation", "test")
DEFAULT_CUTOFF = date(2021, 12, 31)
DEFAULT_MAX_SEQUENCE_LENGTH = 448
DEFAULT_MINIMUM_COMPONENTS = 8
TARGET_REPLAY_RATIO = 0.20


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def metadata_cache_paths(path: Path) -> list[Path]:
    """Return real SourceDocument records from a collector cache or raw folder."""

    root = path / "raw" if (path / "raw").is_dir() else path
    return sorted(
        candidate
        for candidate in root.glob("*/*.json")
        if not candidate.name.startswith("._")
    )


def load_metadata_cache(path: Path) -> list[SourceDocument]:
    paths = metadata_cache_paths(path)
    if not paths:
        raise ValueError(f"no SourceDocument records found beneath {path}")
    return load_source_documents(paths)


def load_replay_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid replay JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"replay record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def _value_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"expected mapping/dataclass structure, got {type(value).__name__}")


def _cache_identity(document_id: str) -> str:
    return sha256_text(document_id)


def _cache_record_path(cache_directory: Path, document: SourceDocument) -> Path:
    stratum = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.stratum)
    return cache_directory / "records" / stratum / f"{_cache_identity(document.document_id)}.json"


def _cache_artifact_path(cache_directory: Path, document: SourceDocument) -> Path:
    return cache_directory / "artifacts" / f"{_cache_identity(document.document_id)}.bin"


class StructureCacheRecords(dict[str, dict[str, Any]]):
    """Ready cache mapping with duplicate cross-list provenance attached."""

    def __init__(self) -> None:
        super().__init__()
        self.record_paths: dict[str, str] = {}
        self.duplicate_records: list[dict[str, Any]] = []


def _cached_stratum(record: Mapping[str, Any], candidate: Path) -> str:
    explicit = str(record.get("source_stratum") or "").strip()
    if explicit:
        return explicit
    directory = candidate.parent.name
    if directory == "arxiv_hep-ph":
        return "arxiv:hep-ph"
    if directory == "arxiv_hep-th":
        return "arxiv:hep-th"
    if directory == "pubmed":
        return "pubmed"
    return directory


def load_structure_cache(path: Path) -> StructureCacheRecords:
    """Load and hash-check ready structure records; unavailable records are skipped."""

    root = path / "records" if (path / "records").is_dir() else path
    result = StructureCacheRecords()
    for candidate in sorted(root.glob("*/*.json")):
        if candidate.name.startswith("._"):
            continue
        record = json.loads(candidate.read_text(encoding="utf-8"))
        if record.get("schema_version") != STRUCTURE_CACHE_SCHEMA:
            raise ValueError(f"unexpected structure cache schema in {candidate}")
        if record.get("status") != "ready":
            continue
        structure = record.get("structure")
        if not isinstance(structure, dict):
            raise TypeError(f"missing structure object in {candidate}")
        structure_hash = sha256_text(canonical_json(structure))
        if structure_hash != record.get("structure_sha256"):
            raise ValueError(f"structure cache hash mismatch: {candidate}")
        document_id = str(record.get("document_id", ""))
        if not document_id:
            raise ValueError(f"empty structure document id in {candidate}")
        if document_id in result:
            existing = result[document_id]
            if (
                str(existing.get("artifact_sha256") or "")
                != str(record.get("artifact_sha256") or "")
                or str(existing.get("structure_sha256") or "")
                != str(record.get("structure_sha256") or "")
                or canonical_json(existing.get("structure"))
                != canonical_json(record.get("structure"))
            ):
                raise ValueError(
                    "duplicate structure cache records disagree for "
                    f"{document_id}: {result.record_paths[document_id]} and {candidate}"
                )
            selected_path = Path(result.record_paths[document_id])
            result.duplicate_records.append({
                "document_id": document_id,
                "selected_stratum": _cached_stratum(existing, selected_path),
                "duplicate_stratum": _cached_stratum(record, candidate),
                "selected_record": f"{selected_path.parent.name}/{selected_path.name}",
                "duplicate_record": f"{candidate.parent.name}/{candidate.name}",
                "artifact_sha256": str(record.get("artifact_sha256") or ""),
                "structure_sha256": str(record.get("structure_sha256") or ""),
            })
            continue
        result[document_id] = record
        result.record_paths[document_id] = str(candidate)
    return result


class StructureHydrator:
    """Resume-safe official artifact fetch and structure parsing.

    Parsed records are keyed by document identity. A successful record is only
    reused when its metadata hash still matches; raw official artifacts are
    retained separately so parser changes need not re-download them.
    """

    def __init__(
        self,
        cache_directory: Path,
        *,
        arxiv_fetcher: Any,
        pubmed_fetcher: Any,
        ncbi_api_key: str = "",
        retry_failed: bool = False,
    ) -> None:
        self.cache_directory = cache_directory
        self.arxiv_fetcher = arxiv_fetcher
        self.pubmed_fetcher = pubmed_fetcher
        self.ncbi_api_key = ncbi_api_key.strip()
        self.retry_failed = retry_failed

    @staticmethod
    def _pmcid(document: SourceDocument) -> str:
        match = re.search(r"\bPMC\d+\b", document.artifact_url, flags=re.IGNORECASE)
        return match.group(0).upper() if match else ""

    def _official_payload(self, document: SourceDocument) -> tuple[bytes, str, str]:
        if document.provider == "arxiv":
            endpoint = f"{ARXIV_EPRINT}{document.source_id}"
            return self.arxiv_fetcher.get(endpoint), endpoint, "arxiv_tex"
        if document.provider == "pubmed":
            pmcid = self._pmcid(document)
            if not pmcid:
                raise LookupError("pmcid_missing")
            parameters = {"db": "pmc", "id": pmcid, "retmode": "xml"}
            if self.ncbi_api_key:
                parameters["api_key"] = self.ncbi_api_key
            return self.pubmed_fetcher.get(NCBI_EFETCH, params=parameters), NCBI_EFETCH, "pmc_jats"
        raise LookupError(f"unsupported_provider:{document.provider}")

    def hydrate_one(self, document: SourceDocument) -> dict[str, Any]:
        if date.fromisoformat(document.published) > DEFAULT_CUTOFF:
            raise ValueError(f"post-cutoff source document rejected: {document.document_id}")
        if date.fromisoformat(document.latest_version) > DEFAULT_CUTOFF:
            raise ValueError(f"post-cutoff source revision rejected: {document.document_id}")
        record_path = _cache_record_path(self.cache_directory, document)
        metadata_hash = raw_record_sha256(document)
        if record_path.exists():
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            if existing.get("document_id") != document.document_id:
                raise ValueError(f"structure cache identity collision: {record_path}")
            if existing.get("metadata_raw_record_sha256") != metadata_hash:
                raise ValueError(
                    f"metadata changed for cached structure {document.document_id}; use a new cache"
                )
            if existing.get("status") == "ready" or not self.retry_failed:
                return existing

        artifact_path = _cache_artifact_path(self.cache_directory, document)
        endpoint = ""
        source_format = ""
        try:
            if artifact_path.exists():
                payload = artifact_path.read_bytes()
                source_format = "arxiv_tex" if document.provider == "arxiv" else "pmc_jats"
                endpoint = (
                    f"{ARXIV_EPRINT}{document.source_id}"
                    if document.provider == "arxiv"
                    else NCBI_EFETCH
                )
            else:
                payload, endpoint, source_format = self._official_payload(document)
                _atomic_bytes(artifact_path, payload)
        except LookupError as exc:
            record = {
                "schema_version": STRUCTURE_CACHE_SCHEMA,
                "document_id": document.document_id,
                "source_stratum": document.stratum,
                "metadata_raw_record_sha256": metadata_hash,
                "status": "unavailable",
                "reason": str(exc),
            }
            _atomic_text(record_path, json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            return record

        from scripts.academic_finetune.structure_extract import (
            parse_jats_structure,
            parse_tex_structure_archive,
        )

        try:
            parsed = (
                parse_tex_structure_archive(payload)
                if source_format == "arxiv_tex"
                else parse_jats_structure(payload)
            )
            structure = _value_dict(parsed)
            if structure.get("schema_version") != PAPER_STRUCTURE_SCHEMA:
                raise ValueError("parser emitted an unexpected paper-structure schema")
        # Source archives are untrusted research artifacts and the bounded
        # parsers can legitimately surface several stdlib decoder/container
        # exception families. Cache the normalized failure without making one
        # malformed paper abort a resumable collection.
        except Exception as exc:  # noqa: BLE001
            record = {
                "schema_version": STRUCTURE_CACHE_SCHEMA,
                "document_id": document.document_id,
                "source_stratum": document.stratum,
                "metadata_raw_record_sha256": metadata_hash,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_filename": artifact_path.name,
                "status": "parse_error",
                "reason": f"{type(exc).__name__}:{str(exc)[:240]}",
            }
            _atomic_text(record_path, json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            return record

        record = {
            "schema_version": STRUCTURE_CACHE_SCHEMA,
            "document_id": document.document_id,
            "source_stratum": document.stratum,
            "metadata_raw_record_sha256": metadata_hash,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "artifact_filename": artifact_path.name,
            "content_endpoint": endpoint,
            "status": "ready",
            "structure_sha256": sha256_text(canonical_json(structure)),
            "structure": structure,
        }
        _atomic_text(record_path, json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return record

    def hydrate(self, documents: Iterable[SourceDocument]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for document in sorted(documents, key=lambda value: (value.stratum, value.document_id)):
            counts[self.hydrate_one(document)["status"]] += 1
        return dict(sorted(counts.items()))


def _normal_heading(value: str) -> str:
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", value or "")
    value = re.sub(r"[^\w\s-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.casefold().split())


def discipline_for(document: SourceDocument) -> str:
    if document.stratum == "arxiv:hep-th":
        return "theoretical_physics"
    if document.stratum == "arxiv:hep-ph":
        return "particle_phenomenology"
    return "biomedical_science"


def genre_for(document: SourceDocument, structure: Mapping[str, Any]) -> str:
    title = _normal_heading(document.title)
    headings = " ".join(
        _normal_heading(str(node.get("title", "")))
        for node in structure.get("sections", [])
        if isinstance(node, Mapping)
    )
    if document.provider == "pubmed":
        if "systematic review" in title or "meta analysis" in title:
            return "systematic_review"
        if "review" in title:
            return "narrative_review"
        if re.search(r"\b(case report|case series)\b", title):
            return "clinical_case"
        if "methods" in headings and "results" in headings:
            return "empirical_biomedical_article"
        return "biomedical_article"
    if document.stratum == "arxiv:hep-ph":
        return "phenomenology_article"
    return "theory_article"


def normalize_section_role(title: str, *, discipline: str) -> str:
    """Map heading semantics to a small discipline-aware role vocabulary.

    Unknown physics sections remain ``domain_development`` and unknown
    biomedical sections remain ``domain_section``; position alone never forces
    an IMRaD role.
    """

    heading = _normal_heading(title)
    rules: tuple[tuple[str, str], ...] = (
        (r"\b(introduction|motivation|overview)\b", "introduction"),
        (r"\b(background|preliminaries|preliminary|related work)\b", "background"),
        (r"\b(materials? and methods?|methods?|methodology|study design|participants?|patients?)\b", "methods"),
        (r"\b(experimental setup|detector|data set|dataset|data collection)\b", "experimental_setup"),
        (r"\b(model|formalism|framework|theoretical setup|setup and notation|notation)\b", "formalism"),
        (r"\b(calculation|derivation|proof|amplitudes?|cross sections?|phenomenology)\b", "analysis"),
        (r"\b(results?|findings?)\b", "results"),
        (r"\b(discussion|interpretation|implications?)\b", "discussion"),
        (r"\b(conclusions?|summary|outlook|concluding remarks?)\b", "conclusion"),
        (r"\b(limitations?)\b", "limitations"),
        (r"\b(acknowledg(e)?ments?)\b", "acknowledgements"),
        (r"\b(references?|bibliography)\b", "references"),
        (r"\b(appendix|appendices|supplementary|supplement)\b", "appendix"),
    )
    for pattern, role in rules:
        if re.search(pattern, heading):
            if role == "methods" and discipline != "biomedical_science":
                return "method_or_setup"
            return role
    return "domain_development" if discipline != "biomedical_science" else "domain_section"


def _children(node: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(child) for child in node.get("children", []) if isinstance(child, Mapping)]


def _path(node: Mapping[str, Any], fallback: Sequence[int]) -> tuple[int, ...]:
    raw = node.get("path", fallback)
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(value, int) for value in raw):
        return tuple(int(value) for value in raw)
    return tuple(fallback)


def _flatten_nodes(
    nodes: Sequence[Mapping[str, Any]],
    *,
    prefix: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(nodes, start=1):
        node = dict(raw)
        path = _path(node, (*prefix, index))
        node["path"] = list(path)
        node["id"] = "s" + ".".join(str(value) for value in path)
        result.append(node)
        result.extend(_flatten_nodes(_children(node), prefix=path))
    return result


def _included(node: Mapping[str, Any]) -> bool:
    return bool(node.get("included_in_main", True))


def _main_roots(structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in _flatten_nodes(
            [dict(value) for value in structure.get("sections", []) if isinstance(value, Mapping)]
        )
        if len(node["path"]) == 1 and _included(node)
    ]


def _all_main_nodes(structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in _flatten_nodes(
            [dict(value) for value in structure.get("sections", []) if isinstance(value, Mapping)]
        )
        if _included(node)
    ]


def _integer(node: Mapping[str, Any], key: str, fallback: str = "") -> int:
    value = node.get(key, node.get(fallback, 0) if fallback else 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _node_id(node: Mapping[str, Any]) -> str:
    existing = str(node.get("id", ""))
    if existing:
        return existing
    return "s" + ".".join(str(value) for value in _path(node, (1,)))


def _outline_node(node: Mapping[str, Any], *, discipline: str) -> dict[str, Any]:
    return {
        "id": _node_id(node),
        "heading": str(node.get("title", "")).strip(),
        "path": list(_path(node, (1,))),
        "role": normalize_section_role(str(node.get("title", "")), discipline=discipline),
        "words": _integer(node, "word_count", "direct_word_count"),
    }


def _budget_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "figures": _integer(node, "figure_count", "direct_figure_count"),
        "id": _node_id(node),
        "paragraphs": _integer(node, "paragraph_count", "direct_paragraph_count"),
        "tables": _integer(node, "table_count", "direct_table_count"),
        "words": _integer(node, "word_count", "direct_word_count"),
    }


def _paper_counts(structure: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = structure.get("counts", {})
    counts = counts if isinstance(counts, Mapping) else {}
    section_words = sum(_integer(node, "word_count", "direct_word_count") for node in roots)
    section_paragraphs = sum(
        _integer(node, "paragraph_count", "direct_paragraph_count") for node in roots
    )
    return {
        "abstract_words": _integer(structure, "abstract_word_count"),
        "section_words": section_words,
        "section_paragraphs": section_paragraphs,
        "unsectioned_words": _integer(structure, "unsectioned_word_count"),
        "figures": _integer(counts, "main_figures"),
        "tables": _integer(counts, "main_tables"),
    }


def _blueprint(
    structure: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], *, discipline: str, genre: str
) -> dict[str, Any]:
    # Root order itself encodes the root path. Nested parent/path supervision is
    # supplied by order_structure, restore_section and repair_structure, so the
    # paper-level target need not repeat root paths and exceed the 448-token row.
    return {
        "paper_counts": _paper_counts(structure, roots),
        "sections": [
            {
                "heading": str(node.get("title", "")).strip(),
                "id": _node_id(node),
                "role": normalize_section_role(str(node.get("title", "")), discipline=discipline),
                "words": _integer(node, "word_count", "direct_word_count"),
            }
            for node in roots
        ],
    }


def _approx_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else canonical_json(value)
    # JSON punctuation and word-like spans are deliberately counted separately.
    # This is conservative for current Qwen tokenizers and is independently
    # visible in every row's provenance.
    return len(re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE))


def _bounded_abstract(value: str, maximum_words: int = 32) -> str:
    words = re.findall(r"\S+", re.sub(r"\s+", " ", value).strip())
    return " ".join(words[:maximum_words])


def _deterministic_choice(values: Sequence[Any], seed: str) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return min(values, key=lambda value: sha256_text(f"{seed}\0{canonical_json(value)}"))


def _deterministic_shuffle(values: Sequence[Any], seed: str) -> list[Any]:
    result = sorted(values, key=lambda value: sha256_text(f"{seed}\0{canonical_json(value)}"))
    if len(result) > 1 and list(values) == result:
        result = result[1:] + result[:1]
    return result


def _generic_input(
    instruction: str,
    context: Mapping[str, Any],
    required: Sequence[str],
    *,
    constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "context": dict(context),
        "constraints": {
            "evidence": "observed_paper_structure_only",
            **dict(constraints or {}),
        },
        "response_schema": {
            "type": "object",
            "required": list(required),
        },
    }


def _task_payloads(
    document: SourceDocument,
    structure: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any], str]]:
    roots = _main_roots(structure)
    nodes = _all_main_nodes(structure)
    if len(roots) < 2 or not nodes:
        return []
    discipline = discipline_for(document)
    genre = genre_for(document, structure)
    blueprint = _blueprint(structure, roots, discipline=discipline, genre=genre)
    seed_base = f"{document.document_id}\0structure-v1"

    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        groups[tuple(_path(node, (1,)))[:-1]].append(node)
    groups = {
        parent: sorted(values, key=lambda value: list(_path(value, (1,))))
        for parent, values in groups.items()
        if len(values) >= 2
    }
    nested_groups = {parent: values for parent, values in groups.items() if parent}
    order_parent = _deterministic_choice(
        list(nested_groups) if nested_groups else list(groups),
        f"{seed_base}\0order-parent",
    )
    order_nodes = groups[order_parent]
    order_parent_id = "paper" if not order_parent else "s" + ".".join(map(str, order_parent))
    chosen = _deterministic_choice(nodes, f"{seed_base}\0role")
    chosen_path = list(_path(chosen, (1,)))
    siblings = [
        node
        for node in nodes
        if list(_path(node, (1,)))[:-1] == chosen_path[:-1]
    ]
    sibling_order = sorted(siblings, key=lambda node: list(_path(node, (1,))))
    chosen_index = next(index for index, node in enumerate(sibling_order) if _node_id(node) == _node_id(chosen))
    neighbour_headings = [
        str(node.get("title", ""))
        for node in sibling_order[max(0, chosen_index - 1): chosen_index + 2]
        if _node_id(node) != _node_id(chosen)
    ]

    payloads: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    payloads.append((
        "recognize_role",
        _generic_input(
            "Classify the rhetorical role of the observed section heading.",
            {
                "discipline": discipline,
                "genre": genre,
                "heading": str(chosen.get("title", "")),
                "level": len(chosen_path),
                "neighbour_headings": neighbour_headings,
            },
            ("role",),
        ),
        {"role": normalize_section_role(str(chosen.get("title", "")), discipline=discipline)},
        f"section:{_node_id(chosen)}",
    ))

    observed = [
        {"heading": str(node.get("title", "")), "id": _node_id(node)}
        for node in order_nodes
    ]
    shuffled = _deterministic_shuffle(observed, f"{seed_base}\0order")
    payloads.append((
        "order_structure",
        _generic_input(
            "Restore the observed main-section order.",
            {
                "discipline": discipline,
                "genre": genre,
                "parent_id": order_parent_id,
                "parent_path": list(order_parent),
                "shuffled_sections": shuffled,
            },
            ("parent_id", "parent_path", "ordered_section_ids"),
            constraints={"preserve_section_ids": True},
        ),
        {
            "ordered_section_ids": [_node_id(node) for node in order_nodes],
            "parent_id": order_parent_id,
            "parent_path": list(order_parent),
        },
        f"siblings:{order_parent_id}",
    ))

    counts = _paper_counts(structure, roots)
    payloads.append((
        "budget_structure",
        _generic_input(
            "Recover the exact observed section length allocation.",
            {
                "discipline": discipline,
                "genre": genre,
                "section_words": counts["section_words"],
                "sections": [
                    {"heading": str(node.get("title", "")), "id": _node_id(node)}
                    for node in roots
                ],
            },
            ("section_words", "section_budgets"),
            constraints={"budgets_sum_to_section_words": True},
        ),
        {
            "section_budgets": [_budget_node(node) for node in roots],
            "section_words": counts["section_words"],
        },
        "main_section_budgets",
    ))

    nested_nodes = [node for node in nodes if len(_path(node, (1,))) > 1]
    removable = nested_nodes or roots[1:-1] or roots
    missing = _deterministic_choice(removable, f"{seed_base}\0restore")
    missing_path = list(_path(missing, (1,)))
    missing_parent_path = missing_path[:-1]
    missing_parent_id = (
        "paper" if not missing_parent_path else "s" + ".".join(map(str, missing_parent_path))
    )
    missing_siblings = sorted(
        [
            node
            for node in nodes
            if list(_path(node, (1,)))[:-1] == missing_parent_path
        ],
        key=lambda node: list(_path(node, (1,))),
    )
    remaining = [
        {"heading": str(node.get("title", "")), "id": _node_id(node)}
        for node in missing_siblings
        if _node_id(node) != _node_id(missing)
    ]
    missing_target = _outline_node(missing, discipline=discipline)
    payloads.append((
        "restore_section",
        _generic_input(
            "Restore the omitted observed section and its exact insertion path.",
            {
                "discipline": discipline,
                "genre": genre,
                "parent_id": missing_parent_id,
                "parent_path": missing_parent_path,
                "incomplete_sections": remaining,
            },
            ("missing_section",),
            constraints={"preserve_observed_order": True},
        ),
        {
            "missing_section": {
                **missing_target,
                "parent_id": missing_parent_id,
            },
        },
        f"missing:{_node_id(missing)}",
    ))

    repair_nodes = order_nodes
    if len(repair_nodes) > 4:
        window_start = int(sha256_text(f"{seed_base}\0repair-window")[:8], 16) % (
            len(repair_nodes) - 3
        )
        repair_nodes = repair_nodes[window_start:window_start + 4]
    correct_sections = [_outline_node(node, discipline=discipline) for node in repair_nodes]
    corrupted_sections = [dict(value) for value in correct_sections]
    if len(corrupted_sections) > 1:
        first = int(sha256_text(f"{seed_base}\0swap")[:8], 16) % len(corrupted_sections)
        second = (first + 1) % len(corrupted_sections)
        corrupted_sections[first], corrupted_sections[second] = (
            corrupted_sections[second],
            corrupted_sections[first],
        )
    altered = int(sha256_text(f"{seed_base}\0role-corruption")[:8], 16) % len(corrupted_sections)
    corrupted_sections[altered]["role"] = "introduction" if corrupted_sections[altered]["role"] != "introduction" else "results"
    candidate = {
        "parent_id": order_parent_id,
        "parent_path": list(order_parent),
        "sections": corrupted_sections,
    }
    repaired = {
        "parent_id": order_parent_id,
        "parent_path": list(order_parent),
        "sections": correct_sections,
    }
    payloads.append((
        "repair_structure",
        _generic_input(
            "Repair the corrupted blueprint to the observed order, roles, paths, and budgets.",
            {"candidate_blueprint": candidate},
            ("parent_id", "parent_path", "sections"),
            constraints={"repair_all_corruptions": True},
        ),
        repaired,
        f"sibling_repair:{order_parent_id}",
    ))

    payloads.append((
        "brief_to_blueprint",
        _generic_input(
            "Construct the paper blueprint observed for this title and bounded abstract brief.",
            {
                "discipline": discipline,
                "genre": genre,
                "title": document.title,
                "abstract_brief": _bounded_abstract(document.abstract),
            },
            ("paper_counts", "sections"),
        ),
        blueprint,
        "paper_blueprint",
    ))
    return payloads


def _base_fields(
    document: SourceDocument,
    cache_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = {
        "provider": document.provider,
        "stratum": document.stratum,
        "source_id": document.source_id,
        "landing_url": document.landing_url,
        "artifact_url": document.artifact_url,
        "license": document.license,
        "query": document.query,
    }
    document_fields = {
        "document_id": document.document_id,
        "title": document.title,
        "authors": list(document.authors),
        "published": document.published,
        "latest_version": document.latest_version,
        "metadata_revised": document.metadata_revised,
    }
    structure = cache_record["structure"]
    provenance = {
        "cutoff": DEFAULT_CUTOFF.isoformat(),
        "metadata_endpoint": document.metadata_endpoint,
        "content_endpoint": cache_record.get("content_endpoint", document.content_endpoint),
        "metadata_raw_record_sha256": raw_record_sha256(document),
        "artifact_sha256": cache_record.get("artifact_sha256", ""),
        "structure_sha256": cache_record["structure_sha256"],
        "structure_parser": structure.get("provenance", {}),
    }
    return source, document_fields, provenance


def _structure_examples(
    document: SourceDocument,
    cache_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    structure = cache_record["structure"]
    source, document_fields, provenance = _base_fields(document, cache_record)
    result: list[dict[str, Any]] = []
    for task_type, task_input, target, locator in _task_payloads(document, structure):
        token_estimate = _approx_tokens(task_input) + _approx_tokens(target) + 32
        identity = sha256_text(
            f"{document.document_id}\0{task_type}\0{locator}\0{canonical_json(target)}"
        )
        result.append({
            "schema_version": STRUCTURE_CORPUS_SCHEMA,
            "example_id": identity,
            "split": "",
            "task_type": task_type,
            "source": source,
            "document": document_fields,
            "input": task_input,
            "target": target,
            "provenance": {
                **provenance,
                "locator": locator,
                "deterministic_seed_sha256": sha256_text(
                    f"{document.document_id}\0structure-v1\0{task_type}"
                ),
                "approx_training_tokens": token_estimate,
            },
        })
    return result


def _transform_replay(
    record: Mapping[str, Any],
    documents: Mapping[str, SourceDocument],
) -> dict[str, Any] | None:
    old_input = record.get("input")
    old_document = record.get("document")
    old_source = record.get("source")
    target = record.get("target")
    unit = record.get("task_type")
    if (
        not isinstance(old_input, Mapping)
        or not isinstance(old_document, Mapping)
        or not isinstance(old_source, Mapping)
        or unit not in {"sentence", "paragraph"}
        or not isinstance(target, str)
        or not target.strip()
    ):
        return None
    required = ("context", "claims", "rhetorical_relation", "certainty", "citation_count")
    if any(key not in old_input for key in required):
        return None
    document_id = str(old_document.get("document_id", ""))
    metadata = documents.get(document_id)
    published = str(old_document.get("published", ""))
    latest = str(old_document.get("latest_version", ""))
    try:
        if date.fromisoformat(published) > DEFAULT_CUTOFF or date.fromisoformat(latest) > DEFAULT_CUTOFF:
            return None
    except ValueError:
        return None
    task_input = {
        "unit": unit,
        **{key: old_input[key] for key in required},
        "citation_slots": old_input.get("citation_slots", []),
        "construction_method": old_input.get("construction_method", "bounded_semantic_roles_v3"),
    }
    token_estimate = _approx_tokens(task_input) + _approx_tokens(target) + 32
    source = dict(old_source)
    source["license"] = metadata.license if metadata is not None else str(source.get("license", ""))
    identity = sha256_text(f"prose-replay-v1\0{record.get('example_id', '')}\0{sha256_text(target)}")
    return {
        "schema_version": STRUCTURE_CORPUS_SCHEMA,
        "example_id": identity,
        "split": "",
        "task_type": "prose_replay",
        "source": source,
        "document": dict(old_document),
        "input": task_input,
        "target": target,
        "provenance": {
            "cutoff": DEFAULT_CUTOFF.isoformat(),
            "replay_source_schema": record.get("schema_version", ""),
            "replay_source_example_id": record.get("example_id", ""),
            "replay_original_split": record.get("split", ""),
            "metadata_raw_record_sha256": (
                raw_record_sha256(metadata) if metadata is not None else record.get("provenance", {}).get("raw_record_sha256", "")
            ),
            "approx_training_tokens": token_estimate,
        },
    }


def _tokenizer_receipt(value: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """Validate the deterministic identity required for an injected tokenizer."""

    if isinstance(value, str):
        if not value.strip():
            raise ValueError("injected tokenizer_identity must not be empty")
        return {"identity": value.strip(), "loader": "injected"}
    if not isinstance(value, Mapping):
        raise ValueError(
            "an injected tokenizer requires an explicit tokenizer_identity receipt")
    receipt = dict(value)
    identity = receipt.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("injected tokenizer_identity receipt has no stable identity")
    try:
        canonical_json(receipt)
    except (TypeError, ValueError) as exc:
        raise ValueError("injected tokenizer_identity receipt is not stable JSON") from exc
    return receipt


def _training_measurement_row(example: Mapping[str, Any]) -> dict[str, Any]:
    """Render exactly the row later consumed by MLX-LM's completion dataset."""

    task_type = str(example["task_type"])
    prompt_input = example["input"]
    if task_type == "prose_replay":
        # Keep replay byte-for-byte aligned with training_support.format_structure_prompt.
        from scripts.academic_finetune.training_support import format_academic_prompt

        prompt = format_academic_prompt({
            "task_type": prompt_input["unit"],
            "input": {
                key: value for key, value in prompt_input.items() if key != "unit"
            },
        })
        completion = str(example["target"]).strip()
    else:
        # This shared runtime/training contract owns the structure instruction text.
        from spiral.academic_structure_contract import (
            canonical_json_text,
            format_structure_prompt,
        )

        prompt = format_structure_prompt(task_type, prompt_input)
        completion = canonical_json_text(example["target"])
    return {
        "prompt": prompt,
        "completion": completion,
        "example_id": example["example_id"],
        "task_type": task_type,
    }


def _exact_token_filter(
    examples: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject complete rendered rows that MLX-LM would truncate.

    This intentionally has no shortening or partition branch. Stage-two structure
    and replay examples either fit whole or do not enter the corpus.
    """

    from scripts.academic_finetune.bound_training_data import (
        BoundTrainingDataError,
        measure_completion_sequence,
    )

    accepted: list[dict[str, Any]] = []
    measured_by_task: Counter[str] = Counter()
    rejected_by_task: Counter[str] = Counter()
    largest_candidate = 0
    largest_accepted = 0
    for example in examples:
        task_type = str(example["task_type"])
        try:
            measurement = measure_completion_sequence(
                tokenizer, _training_measurement_row(example))
        except BoundTrainingDataError as exc:
            raise ValueError(
                f"exact token measurement failed for {example['example_id']}: {exc}") from exc
        measured_by_task[task_type] += 1
        largest_candidate = max(largest_candidate, measurement.total_tokens)
        if measurement.total_tokens > max_sequence_length:
            rejected_by_task[task_type] += 1
            continue
        largest_accepted = max(largest_accepted, measurement.total_tokens)
        bounded = dict(example)
        bounded["provenance"] = {
            **dict(example["provenance"]),
            "exact_training_tokens": measurement.total_tokens,
            "exact_prompt_offset": measurement.prompt_offset,
            "exact_completion_tokens": measurement.completion_tokens,
        }
        accepted.append(bounded)
    return accepted, {
        "candidates_measured": len(examples),
        "candidates_measured_by_task_type": dict(sorted(measured_by_task.items())),
        "candidates_rejected": sum(rejected_by_task.values()),
        "candidates_rejected_by_task_type": dict(sorted(rejected_by_task.items())),
        "largest_candidate_tokens": largest_candidate,
        "largest_accepted_tokens": largest_accepted,
    }


def _round_robin_take(
    values: Sequence[dict[str, Any]],
    limit: int,
    *,
    bucket_fields: Sequence[str],
) -> list[dict[str, Any]]:
    if limit >= len(values):
        return list(values)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        key_parts: list[str] = []
        for field in bucket_fields:
            if field == "stratum":
                key_parts.append(str(value["source"]["stratum"]))
            else:
                key_parts.append(str(value[field]))
        buckets[tuple(key_parts)].append(value)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: sha256_text(item["example_id"]))
    result: list[dict[str, Any]] = []
    index = 0
    while len(result) < limit:
        advanced = False
        for key in sorted(buckets):
            if index < len(buckets[key]):
                result.append(buckets[key][index])
                advanced = True
                if len(result) == limit:
                    break
        if not advanced:
            break
        index += 1
    return result


def _select_with_exact_replay_ratio(
    structure_examples: Sequence[dict[str, Any]],
    replay_examples: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select four structure rows per replay row (exactly 20% final replay)."""

    if not replay_examples or len(structure_examples) < 4:
        return list(structure_examples), []
    replay_count = min(len(replay_examples), len(structure_examples) // 4)
    structure_count = replay_count * 4
    selected_structure = _round_robin_take(
        structure_examples,
        structure_count,
        bucket_fields=("stratum", "task_type"),
    )
    selected_replay = _round_robin_take(
        replay_examples,
        replay_count,
        bucket_fields=("stratum",),
    )
    return selected_structure, selected_replay


def _deduplicate_examples(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for value in sorted(values, key=lambda item: item["example_id"]):
        if value["example_id"] in chosen:
            continue
        chosen[value["example_id"]] = value
    return list(chosen.values())


def _author_safety(examples: Sequence[Mapping[str, Any]]) -> bool:
    author_splits: dict[str, set[str]] = defaultdict(set)
    document_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        split = str(example["split"])
        document = example["document"]
        document_splits[str(document["document_id"])].add(split)
        authors = [_normal_author(str(value)) for value in document.get("authors", [])]
        for author in authors:
            if author:
                author_splits[author].add(split)
    return all(len(value) == 1 for value in document_splits.values()) and all(
        len(value) == 1 for value in author_splits.values()
    )


def _aggregate_input_hash(records: Iterable[Mapping[str, Any]], field: str) -> str:
    values = sorted(str(record.get(field, "")) for record in records if record.get(field))
    return sha256_text("\n".join(values))


def _document_crosslist_identity(document: SourceDocument) -> dict[str, Any]:
    value = document.to_dict()
    # A cross-list necessarily has a different source stratum and query. Those
    # are collection provenance, while every bibliographic/content/artifact
    # field must remain byte-identical or compilation fails closed.
    for key in ("stratum", "query", "content_sha256", "raw_record_sha256"):
        value.pop(key, None)
    return value


def _deduplicate_crosslisted_documents(
    documents: Iterable[SourceDocument],
) -> tuple[list[SourceDocument], list[dict[str, Any]]]:
    by_identity: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in documents:
        by_identity[document.document_id].append(document)
    selected: list[SourceDocument] = []
    duplicates: list[dict[str, Any]] = []
    for document_id, group in sorted(by_identity.items()):
        ordered = sorted(group, key=lambda value: (value.stratum, value.document_id))
        winner = ordered[0]
        if len(ordered) > 1:
            strata = [value.stratum for value in ordered]
            if (winner.provider != "arxiv" or len(set(strata)) != len(strata)
                    or any(
                        _document_crosslist_identity(value)
                        != _document_crosslist_identity(winner)
                        for value in ordered[1:]
                    )):
                raise ValueError(
                    f"duplicate source documents disagree for {document_id}: "
                    + ", ".join(strata)
                )
            duplicates.append({
                "document_id": document_id,
                "observed_strata": strata,
                "selected_stratum": winner.stratum,
                "duplicates_removed": len(ordered) - 1,
            })
        selected.append(winner)
    return selected, duplicates


def compile_structure_corpus(
    documents: Iterable[SourceDocument],
    structure_records: Mapping[str, Mapping[str, Any]],
    *,
    output_path: Path,
    replay_records: Iterable[Mapping[str, Any]] = (),
    cutoff: date = DEFAULT_CUTOFF,
    required_strata: Sequence[str] = REQUIRED_STRATA,
    balance_sources: bool = True,
    require_trainable_splits: bool = True,
    minimum_per_split_stratum: int = DEFAULT_MINIMUM_COMPONENTS,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    tokenizer_model_path: Path | None = None,
    tokenizer: Any | None = None,
    tokenizer_identity: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Compile cached structures into a byte-stable, exactly token-gated curriculum."""

    if cutoff > DEFAULT_CUTOFF:
        raise ValueError("academic structure corpus cutoff cannot exceed 2021-12-31")
    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or not 128 <= max_sequence_length <= DEFAULT_MAX_SEQUENCE_LENGTH
    ):
        raise ValueError("max_sequence_length must be an integer in [128, 448]")
    if tokenizer is None:
        if tokenizer_model_path is None:
            raise ValueError(
                "exact structure compilation requires tokenizer_model_path or an "
                "injected tokenizer")
        if tokenizer_identity is not None:
            raise ValueError(
                "tokenizer_identity cannot override the pinned production tokenizer receipt")
        from scripts.academic_finetune.bound_training_data import load_tokenizer_only

        tokenizer, exact_tokenizer_receipt = load_tokenizer_only(tokenizer_model_path)
    else:
        if tokenizer_model_path is not None:
            raise ValueError(
                "tokenizer_model_path and an injected tokenizer are mutually exclusive")
        exact_tokenizer_receipt = _tokenizer_receipt(tokenizer_identity)
    replay_record_values = list(replay_records)
    deduplicated_documents, crosslist_duplicates = _deduplicate_crosslisted_documents(
        documents
    )
    by_identity: dict[str, SourceDocument] = {}
    for document in sorted(
            deduplicated_documents, key=lambda value: (value.stratum, value.document_id)):
        if date.fromisoformat(document.published) > cutoff or date.fromisoformat(document.latest_version) > cutoff:
            raise ValueError(f"post-cutoff source document rejected: {document.document_id}")
        by_identity[document.document_id] = document

    eligible: dict[str, list[SourceDocument]] = defaultdict(list)
    skipped_structures = Counter()
    for document in by_identity.values():
        record = structure_records.get(document.document_id)
        if not record:
            skipped_structures["missing_structure"] += 1
            continue
        if record.get("status", "ready") != "ready" or not isinstance(record.get("structure"), Mapping):
            skipped_structures[f"status:{record.get('status', 'invalid')}"] += 1
            continue
        structure = dict(record["structure"])
        cached_metadata_hash = str(record.get("metadata_raw_record_sha256", ""))
        if cached_metadata_hash and cached_metadata_hash != raw_record_sha256(document):
            raise ValueError(f"structure metadata hash mismatch for {document.document_id}")
        expected_hash = sha256_text(canonical_json(structure))
        if record.get("structure_sha256") != expected_hash:
            raise ValueError(f"structure hash mismatch for {document.document_id}")
        if structure.get("schema_version") != PAPER_STRUCTURE_SCHEMA:
            raise ValueError(f"unexpected paper structure schema for {document.document_id}")
        from scripts.academic_finetune.structure_extract import PaperStructure

        # Reconstructing the frozen parser object verifies hierarchy types,
        # aggregate counts and parser provenance rather than trusting a JSON
        # object merely because its outer hash is internally consistent.
        PaperStructure.from_dict(structure)
        expected_format = "pmc_jats" if document.provider == "pubmed" else "arxiv_tex"
        if structure.get("source_format") != expected_format:
            raise ValueError(f"structure source format mismatch for {document.document_id}")
        artifact_hash = str(record.get("artifact_sha256", ""))
        parser_source_hash = str(structure.get("provenance", {}).get("source_sha256", ""))
        if artifact_hash and parser_source_hash != artifact_hash:
            raise ValueError(f"structure artifact provenance mismatch for {document.document_id}")
        eligible[document.stratum].append(document)
    missing = sorted(set(required_strata) - set(eligible))
    if missing:
        raise ValueError(f"structure corpus is missing required source strata: {', '.join(missing)}")
    unexpected = sorted(set(eligible) - set(required_strata))
    if unexpected:
        raise ValueError(f"structure corpus has unconfigured source strata: {', '.join(unexpected)}")
    selected_documents = [document for values in eligible.values() for document in values]
    if balance_sources:
        target = min(len(eligible[stratum]) for stratum in required_strata)
        selected_documents = [
            document
            for stratum in sorted(required_strata)
            for document in sorted(
                eligible[stratum],
                key=lambda value: (
                    str(structure_records[value.document_id]["structure_sha256"]),
                    value.document_id,
                ),
            )[:target]
        ]

    structure_candidates = _deduplicate_examples(
        example
        for document in selected_documents
        for example in _structure_examples(
            document,
            structure_records[document.document_id],
        )
    )
    transformed_replay_candidates = _deduplicate_examples(
        transformed
        for record in replay_record_values
        if (transformed := _transform_replay(
            record,
            by_identity,
        )) is not None
    )
    bounded_candidates, exact_token_gate = _exact_token_filter(
        [*structure_candidates, *transformed_replay_candidates],
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
    )
    structure_examples = [
        example for example in bounded_candidates
        if example["task_type"] != "prose_replay"
    ]
    replay_candidates = [
        example for example in bounded_candidates
        if example["task_type"] == "prose_replay"
    ]
    available_structure_examples = len(structure_examples)
    available_replay_examples = len(replay_candidates)
    structure_examples, replay_examples = _select_with_exact_replay_ratio(
        structure_examples, replay_candidates
    )
    structure_rows_dropped_for_replay_mix = available_structure_examples - len(
        structure_examples
    )
    examples = [*structure_examples, *replay_examples]
    if not examples:
        raise ValueError("structure and replay inputs produced no bounded training examples")

    split_diagnostics = _author_safe_splits(
        examples,
        required_strata=required_strata,
        minimum_heldout_components_per_stratum=minimum_per_split_stratum,
    )
    examples.sort(
        key=lambda value: (
            value["split"],
            value["source"]["stratum"],
            value["task_type"],
            value["example_id"],
        )
    )
    if not _author_safety(examples):
        raise AssertionError("connected author/document split safety failed")

    split_counts = Counter(str(example["split"]) for example in examples)
    stratum_counts = Counter(str(example["source"]["stratum"]) for example in examples)
    task_counts = Counter(str(example["task_type"]) for example in examples)
    document_split: dict[str, str] = {}
    document_stratum: dict[str, str] = {}
    for example in examples:
        identity = str(example["document"]["document_id"])
        document_split[identity] = str(example["split"])
        document_stratum[identity] = str(example["source"]["stratum"])
    documents_by_stratum_split = Counter(
        f"{document_stratum[identity]}|{split}" for identity, split in document_split.items()
    )
    examples_by_stratum_split = Counter(
        f"{example['source']['stratum']}|{example['split']}" for example in examples
    )
    components_by_stratum_split = Counter(split_diagnostics.get("components_by_stratum_split", {}))
    non_trainable_reasons: list[str] = []
    for stratum in required_strata:
        for split in SPLITS:
            key = f"{stratum}|{split}"
            if documents_by_stratum_split[key] < minimum_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {documents_by_stratum_split[key]} documents; requires {minimum_per_split_stratum}"
                )
            if examples_by_stratum_split[key] < minimum_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {examples_by_stratum_split[key]} examples; requires {minimum_per_split_stratum}"
                )
            if components_by_stratum_split[key] < minimum_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {components_by_stratum_split[key]} author components; requires {minimum_per_split_stratum}"
                )
    missing_tasks = sorted(set(ALL_TASKS) - set(task_counts))
    if missing_tasks:
        non_trainable_reasons.append("missing task types: " + ", ".join(missing_tasks))
    replay_ratio = task_counts["prose_replay"] / len(examples)
    if not math.isclose(replay_ratio, TARGET_REPLAY_RATIO, abs_tol=1e-12):
        non_trainable_reasons.append(
            f"prose replay ratio is {replay_ratio:.6f}; requires {TARGET_REPLAY_RATIO:.6f}"
        )
    if structure_rows_dropped_for_replay_mix > 3:
        non_trainable_reasons.append(
            "insufficient prose replay coverage required dropping "
            f"{structure_rows_dropped_for_replay_mix} structure examples"
        )
    trainable = not non_trainable_reasons
    if require_trainable_splits and not trainable:
        raise ValueError(
            "structure corpus is not trainable: " + "; ".join(non_trainable_reasons[:5])
        )

    jsonl = "".join(canonical_json(example) + "\n" for example in examples)
    _atomic_text(output_path, jsonl)
    selected_cache_records = [
        structure_records[document.document_id]
        for document in selected_documents
        if document.document_id in structure_records
    ]
    licensed_documents = {
        example["document"]["document_id"]
        for example in examples
        if str(example["source"].get("license", "")).strip()
    }
    cache_duplicate_records = list(
        getattr(structure_records, "duplicate_records", []) or [])
    crosslist_strata = Counter(
        "|".join(row["observed_strata"]) for row in crosslist_duplicates
    )
    duplicates_removed = sum(
        int(row["duplicates_removed"]) for row in crosslist_duplicates
    )
    manifest = {
        "schema_version": STRUCTURE_MANIFEST_SCHEMA,
        "corpus_schema_version": STRUCTURE_CORPUS_SCHEMA,
        "prompt_contract": STRUCTURE_PROMPT_CONTRACT,
        "cutoff": cutoff.isoformat(),
        "trainable": trainable,
        "non_trainable_reasons": non_trainable_reasons,
        "source_strata": sorted(stratum_counts),
        "balanced_sources": balance_sources,
        "corpus_sha256": hashlib.sha256(jsonl.encode("utf-8")).hexdigest(),
        "output_filename": output_path.name,
        "intended_use": {
            "component": "spiral_research_paper_planner",
            "planner_only": True,
            "target_modality": "json_paper_architecture",
            "excluded_component": "spiralchat_general_conversation",
            "prose_replay_purpose": "stage_two_forgetting_regularizer_only",
        },
        "counts": {
            "examples": len(examples),
            "documents": len(document_split),
            "structure_examples": len(structure_examples),
            "available_structure_examples": available_structure_examples,
            "structure_examples_dropped_for_replay_mix": structure_rows_dropped_for_replay_mix,
            "prose_replay_examples": len(replay_examples),
            "available_prose_replay_examples": available_replay_examples,
            "cross_list_document_ids": len(crosslist_duplicates),
            "cross_list_duplicates_removed": duplicates_removed,
            "structure_cache_duplicates_collapsed": len(cache_duplicate_records),
            "prose_replay_ratio": replay_ratio,
            "by_split": dict(sorted(split_counts.items())),
            "by_stratum": dict(sorted(stratum_counts.items())),
            "by_task_type": dict(sorted(task_counts.items())),
            "documents_by_stratum_split": dict(sorted(documents_by_stratum_split.items())),
            "examples_by_stratum_split": dict(sorted(examples_by_stratum_split.items())),
            "licensed_documents": len(licensed_documents),
            "unlicensed_or_unspecified_documents": len(document_split) - len(licensed_documents),
        },
        "gates": {
            "exact_training_token_gate": {
                "method": "mlx_lm.CompletionsDataset.apply_chat_template parity",
                "max_sequence_length": max_sequence_length,
                "overflow_policy": "reject_candidate_never_truncate_or_partition",
                "derived_rows": 0,
                "tokenizer": exact_tokenizer_receipt,
                **exact_token_gate,
            },
            "largest_approx_training_tokens": max(
                int(example["provenance"]["approx_training_tokens"]) for example in examples
            ),
            "pre_2022_only": True,
            "connected_author_document_splits": True,
            "exact_prose_replay_ratio": TARGET_REPLAY_RATIO,
            "canonical_jsonl": True,
            "structure_targets_are_json_objects": True,
            "prose_replay_targets_are_nonempty_strings": True,
            "observed_order_paths_and_integer_budgets_only": True,
        },
        "trainability_thresholds": {
            "documents_examples_author_components_per_split_stratum": minimum_per_split_stratum,
            "required_task_types": list(ALL_TASKS),
        },
        "split_policy": (
            "connected document-author components; deterministic constrained three-way split; "
            "all structure and replay rows for a document remain together"
        ),
        "split_diagnostics": split_diagnostics,
        "input_hashes": {
            "metadata_records_sha256": _aggregate_input_hash(
                (
                    {"hash": raw_record_sha256(document)}
                    for document in selected_documents
                ),
                "hash",
            ),
            "structure_records_sha256": _aggregate_input_hash(
                selected_cache_records,
                "structure_sha256",
            ),
            "replay_examples_sha256": _aggregate_input_hash(
                replay_examples,
                "example_id",
            ),
        },
        "determinism": {
            "corruptions": "sha256(document_id, task_type, structure-v1)",
            "selection": "sha256 identity with task/stratum round-robin",
            "target_serialization": "canonical compact sorted JSON at dataset boundary",
        },
        "deduplication": {
            "cross_list_policy": (
                "identical arXiv document_id/artifact/structure records collapse to the "
                "lexicographically first source stratum before source balancing"
            ),
            "selected_precedence": ["arxiv:hep-ph", "arxiv:hep-th"],
            "cross_list_duplicates": crosslist_duplicates,
            "cross_list_pairs": dict(sorted(crosslist_strata.items())),
            "structure_cache_duplicate_records": cache_duplicate_records,
            "disagreement_policy": "fail_closed",
        },
        "structure_semantics": {
            "roles": "heading-semantic and discipline-aware; position does not impose IMRaD",
            "budgets": "integer counts copied from parser output; no synthetic length targets",
            "hierarchy": "parser path identifiers and observed sibling order",
            "back_matter": "excluded from main-section training by parser included_in_main flag",
        },
        "rejections": {
            "structures": dict(sorted(skipped_structures.items())),
            "examples": {
                f"over_exact_token_budget:{task_type}": count
                for task_type, count in exact_token_gate[
                    "candidates_rejected_by_task_type"].items()
            },
            "replay_candidates_rejected": max(
                0, len(replay_record_values) - len(replay_candidates)
            ),
            "replay_candidates_invalid_or_duplicate_before_token_gate": max(
                0, len(replay_record_values) - len(transformed_replay_candidates)
            ),
            "replay_candidates_over_exact_token_budget": exact_token_gate[
                "candidates_rejected_by_task_type"].get("prose_replay", 0),
        },
    }
    manifest_path = output_path.with_name(f"{output_path.name}.manifest.json")
    _atomic_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest


def official_fetchers(
    *, email: str, ncbi_api_key: str = ""
) -> tuple[PoliteFetcher, PoliteFetcher]:
    """Construct policy-compliant clients for the structure hydration CLI."""

    if not email.strip():
        raise ValueError("a contact email is required for official academic endpoints")
    user_agent = f"SpiralAcademicStructure/1.0 (mailto:{email.strip()})"
    arxiv = PoliteFetcher(user_agent=user_agent, min_interval_seconds=3.0)
    pubmed = PoliteFetcher(
        user_agent=user_agent,
        min_interval_seconds=0.12 if ncbi_api_key.strip() else 0.4,
    )
    return arxiv, pubmed


__all__ = [
    "ALL_TASKS",
    "STRUCTURE_CACHE_SCHEMA",
    "STRUCTURE_CORPUS_SCHEMA",
    "STRUCTURE_MANIFEST_SCHEMA",
    "STRUCTURE_PROMPT_CONTRACT",
    "StructureHydrator",
    "compile_structure_corpus",
    "discipline_for",
    "genre_for",
    "load_metadata_cache",
    "load_replay_corpus",
    "load_structure_cache",
    "normalize_section_role",
    "official_fetchers",
]
