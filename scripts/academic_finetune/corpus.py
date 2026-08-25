"""Resumable collection and deterministic plan-to-prose corpus compilation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.academic_finetune import CORPUS_SCHEMA, MANIFEST_SCHEMA
from scripts.academic_finetune.sources import (
    AcademicSource,
    SourceDocument,
    canonical_json,
    load_source_documents,
    raw_record_sha256,
    sha256_text,
)
from scripts.academic_finetune.text import (
    canonicalize_citations,
    claims_are_feasible,
    certainty,
    citation_count,
    citation_markers,
    context_is_safe,
    context_target_overlap,
    document_rejection_reason,
    neutral_claims,
    plan_content_recall,
    plan_target_overlap,
    paragraphs,
    rhetorical_relation,
    sentences,
    usable_paragraph,
    usable_sentence,
)


REQUIRED_STRATA = ("arxiv:hep-ph", "arxiv:hep-th", "pubmed")
MAX_SENTENCE_EXAMPLES_PER_DOCUMENT = 8
MAX_PARAGRAPH_EXAMPLES_PER_DOCUMENT = 4
MINIMUM_HELDOUT_COMPONENTS_PER_STRATUM = 8
SPLIT_POLICY = (
    "connected document-author components; deterministic constrained per-stratum "
    "holdouts >=8 documents/components; one held-out example per document"
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class CollectionConfig:
    cutoff: date = date(2021, 12, 31)
    maximum_documents_per_source: int = 100
    page_size: int = 25
    maximum_scanned_per_source: int = 500

    def __post_init__(self) -> None:
        if self.cutoff > date(2021, 12, 31):
            raise ValueError("academic corpus cutoff cannot exceed 2021-12-31")
        if not 1 <= self.maximum_documents_per_source <= 100_000:
            raise ValueError("maximum_documents_per_source is out of bounds")
        if not 1 <= self.page_size <= 100:
            raise ValueError("page_size must be in [1, 100]")
        if self.maximum_scanned_per_source < self.maximum_documents_per_source:
            raise ValueError("maximum_scanned_per_source must cover the document limit")


class CorpusCollector:
    """Page-level checkpointing; a crash can at most repeat one idempotent page."""

    def __init__(self, cache_directory: Path, config: CollectionConfig) -> None:
        self.cache_directory = cache_directory
        self.config = config
        self.raw_directory = cache_directory / "raw"
        self.state_path = cache_directory / "collection-state.json"

    def _configuration(self, sources: Sequence[AcademicSource]) -> dict[str, Any]:
        return {
            "collector_revision": 2,
            "cutoff": self.config.cutoff.isoformat(),
            "maximum_documents_per_source": self.config.maximum_documents_per_source,
            "page_size": self.config.page_size,
            "maximum_scanned_per_source": self.config.maximum_scanned_per_source,
            "sources": [
                dict(source.descriptor())
                for source in sorted(sources, key=lambda value: value.checkpoint_key)
            ],
        }

    def _load_state(self, sources: Sequence[AcademicSource]) -> dict[str, Any]:
        configuration = self._configuration(sources)
        config_hash = sha256_text(canonical_json(configuration))
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("configuration_sha256") != config_hash:
                raise ValueError(
                    "collection cache belongs to a different configuration; use another cache directory"
                )
            return state
        return {
            "schema_version": "spiral.academic-collection-state.v1",
            "configuration": configuration,
            "configuration_sha256": config_hash,
            "sources": {},
        }

    def _save_state(self, state: Mapping[str, Any]) -> None:
        _atomic_text(self.state_path, json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    def _record_path(self, document: SourceDocument) -> Path:
        stratum = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.stratum)
        identity = sha256_text(f"{document.stratum}\0{document.source_id}")
        return self.raw_directory / stratum / f"{identity}.json"

    def _write_document(self, document: SourceDocument) -> bool:
        path = self._record_path(document)
        if path.exists():
            cached = load_source_documents([path])[0]
            if cached.document_id != document.document_id:
                raise ValueError(f"cache identity collision: {path}")
            return False
        value = document.to_dict()
        _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return True

    def _raw_record_paths(self) -> list[Path]:
        # exFAT creates AppleDouble siblings named `._<record>.json`. They are
        # filesystem metadata, not corpus records, and must never enter JSON
        # parsing or the deterministic source identity.
        return sorted(
            path
            for path in self.raw_directory.glob("*/*.json")
            if not path.name.startswith("._")
        )

    def collect(self, sources: Sequence[AcademicSource]) -> list[SourceDocument]:
        if len({source.checkpoint_key for source in sources}) != len(sources):
            raise ValueError("each configured source partition must have a unique checkpoint key")
        state = self._load_state(sources)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._save_state(state)
        partitions_by_stratum: dict[str, list[str]] = defaultdict(list)
        for source in sources:
            partitions_by_stratum[source.stratum].append(source.checkpoint_key)
        for keys in partitions_by_stratum.values():
            keys.sort()
        for source in sorted(sources, key=lambda value: value.checkpoint_key):
            partition_keys = partitions_by_stratum[source.stratum]
            partition_index = partition_keys.index(source.checkpoint_key)
            document_base, document_remainder = divmod(
                self.config.maximum_documents_per_source, len(partition_keys)
            )
            document_limit = document_base + (1 if partition_index < document_remainder else 0)
            scan_base, scan_remainder = divmod(
                self.config.maximum_scanned_per_source, len(partition_keys)
            )
            scan_limit = scan_base + (1 if partition_index < scan_remainder else 0)
            source_state = state["sources"].setdefault(
                source.checkpoint_key,
                {"offset": 0, "accepted": 0, "accepted_ids": [], "scanned": 0, "complete": False},
            )
            source_state.setdefault("accepted_ids", [])
            if document_limit == 0 or scan_limit == 0:
                source_state["complete"] = True
                self._save_state(state)
                continue
            while (
                not source_state["complete"]
                and source_state["accepted"] < document_limit
                and source_state["scanned"] < scan_limit
            ):
                remaining_scan = scan_limit - source_state["scanned"]
                page_size = min(self.config.page_size, remaining_scan)
                page = source.fetch_page(int(source_state["offset"]), page_size)
                accepted_this_page = 0
                for document in page.documents:
                    if source_state["accepted"] + accepted_this_page >= document_limit:
                        break
                    identity = document.document_id
                    if identity in source_state["accepted_ids"]:
                        continue
                    path = self._record_path(document)
                    if path.exists():
                        # A crash between the atomic record and checkpoint write is safe.
                        load_source_documents([path])
                    else:
                        self._write_document(source.hydrate(document))
                    source_state["accepted_ids"].append(identity)
                    accepted_this_page += 1
                source_state["accepted"] += accepted_this_page
                source_state["offset"] = page.next_offset
                source_state["scanned"] += page.scanned
                source_state["complete"] = bool(
                    page.exhausted
                    or source_state["accepted"] >= document_limit
                    or source_state["scanned"] >= scan_limit
                )
                self._save_state(state)
                if page.scanned == 0:
                    source_state["complete"] = True
                    self._save_state(state)
                    break
        return load_source_documents(self._raw_record_paths())


def _normal_author(value: str) -> str:
    normal = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in normal).split())


def _deduplicate_documents(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    chosen: dict[str, SourceDocument] = {}
    for document in sorted(documents, key=lambda value: (value.stratum, value.document_id)):
        if not document.prose.strip():
            continue
        # Exact normalized-prose dedup catches cross-lists and repeated baseline shards.
        key = sha256_text(re.sub(r"\s+", " ", document.prose).casefold().strip())
        chosen.setdefault(key, document)
    return list(chosen.values())


def _base_examples(document: SourceDocument, cutoff: date) -> list[dict[str, Any]]:
    prose_paragraphs = paragraphs(document.prose)
    raw_hash = raw_record_sha256(document)
    base_source = {
        "provider": document.provider,
        "source_id": document.source_id,
        "stratum": document.stratum,
        "landing_url": document.landing_url,
        "artifact_url": document.artifact_url,
        "query": document.query,
    }
    base_document = {
        "document_id": document.document_id,
        "title": document.title,
        "authors": list(document.authors),
        "published": document.published,
        "latest_version": document.latest_version,
        "metadata_revised": document.metadata_revised,
        "content_sha256": document.content_sha256,
    }
    sentence_examples: list[dict[str, Any]] = []
    paragraph_examples: list[dict[str, Any]] = []
    previous_paragraph = f"Document title: {document.title}"
    for paragraph_index, paragraph in enumerate(prose_paragraphs):
        paragraph_sentences = sentences(paragraph)
        context_sentences: list[str] = []
        for sentence_index, target in enumerate(paragraph_sentences):
            if not usable_sentence(target):
                context_sentences.append(target)
                continue
            context = " ".join(context_sentences[-2:]).strip() or previous_paragraph
            context = canonicalize_citations(context)
            target = canonicalize_citations(target)
            if context_is_safe(context, target):
                example = _example(
                    document=document,
                    source=base_source,
                    document_fields=base_document,
                    task_type="sentence",
                    context=context,
                    target=target,
                    locator=f"paragraph:{paragraph_index + 1};sentence:{sentence_index + 1}",
                    raw_hash=raw_hash,
                    cutoff=cutoff,
                )
                if example is not None:
                    sentence_examples.append(example)
            context_sentences.append(target)
        paragraph_target = canonicalize_citations(paragraph)
        paragraph_context = canonicalize_citations(previous_paragraph)
        if usable_paragraph(paragraph) and context_is_safe(paragraph_context, paragraph_target):
            example = _example(
                    document=document,
                    source=base_source,
                    document_fields=base_document,
                    task_type="paragraph",
                    context=paragraph_context,
                    target=paragraph_target,
                    locator=f"paragraph:{paragraph_index + 1}",
                    raw_hash=raw_hash,
                    cutoff=cutoff,
                )
            if example is not None:
                paragraph_examples.append(example)
        previous_paragraph = paragraph
    # Full papers must not drown abstracts or short papers. Earlier locators are
    # preferred because abstracts/introduction prose carries explicit argument
    # structure and fewer appendix fragments.
    # A document contributes sentence or paragraph supervision, never both.
    # This prevents each sentence target from reappearing inside a paragraph
    # completion and keeps the apparent sample size honest.
    paragraph_first = int(sha256_text(f"{document.document_id}\0task-v2")[:8], 16) % 4 == 0
    preferred = paragraph_examples if paragraph_first else sentence_examples
    fallback = sentence_examples if paragraph_first else paragraph_examples
    chosen = preferred or fallback
    cap = (
        MAX_PARAGRAPH_EXAMPLES_PER_DOCUMENT
        if chosen and chosen[0]["task_type"] == "paragraph"
        else MAX_SENTENCE_EXAMPLES_PER_DOCUMENT
    )
    return chosen[:cap]


def _example(
    *,
    document: SourceDocument,
    source: Mapping[str, Any],
    document_fields: Mapping[str, Any],
    task_type: str,
    context: str,
    target: str,
    locator: str,
    raw_hash: str,
    cutoff: date,
) -> dict[str, Any] | None:
    if not context_is_safe(context, target):
        raise ValueError(f"context leakage gate failed for {document.document_id} {locator}")
    unit_hash = sha256_text(target)
    identity = sha256_text(f"{document.document_id}\0{task_type}\0{locator}\0{unit_hash}")
    claims = neutral_claims(target, maximum=6, paragraph=task_type == "paragraph")
    if not claims_are_feasible(claims, target):
        return None
    overlap = plan_target_overlap(claims, target)
    return {
        "schema_version": CORPUS_SCHEMA,
        "example_id": identity,
        "split": "",
        "task_type": task_type,
        "source": dict(source),
        "document": dict(document_fields),
        "input": {
            "context": context,
            "claims": claims,
            "rhetorical_relation": rhetorical_relation(target),
            "certainty": certainty(target),
            "citation_count": citation_count(target),
            "citation_slots": citation_markers(target),
            "construction_method": "bounded_semantic_roles_v3",
        },
        "target": target,
        "provenance": {
            "metadata_endpoint": document.metadata_endpoint,
            "content_endpoint": document.content_endpoint,
            "extraction": document.extraction,
            "locator": locator,
            "text_unit_sha256": unit_hash,
            "raw_record_sha256": raw_hash,
            "cutoff": cutoff.isoformat(),
            "plan_target_overlap": overlap,
            "plan_target_content_recall": round(plan_content_recall(claims, target), 6),
            "context_target_overlap": context_target_overlap(context, target),
        },
    }


def _deduplicate_examples(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for example in sorted(examples, key=lambda value: value["example_id"]):
        key = sha256_text(re.sub(r"\s+", " ", example["target"]).casefold().strip())
        chosen.setdefault(key, example)
    return list(chosen.values())


def _round_robin_examples(values: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep source balance without dropping every example from a document."""

    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        by_document[value["document"]["document_id"]].append(value)
    for bucket in by_document.values():
        bucket.sort(key=lambda value: value["example_id"])
    result: list[dict[str, Any]] = []
    round_index = 0
    document_ids = sorted(by_document)
    while len(result) < limit:
        added = False
        for document_id in document_ids:
            bucket = by_document[document_id]
            if round_index < len(bucket):
                result.append(bucket[round_index])
                added = True
                if len(result) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return result


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _author_safe_splits(
    examples: list[dict[str, Any]],
    *,
    required_strata: Sequence[str],
    minimum_heldout_components_per_stratum: int,
) -> dict[str, Any]:
    document_authors: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        document_id = example["document"]["document_id"]
        authors = {_normal_author(value) for value in example["document"]["authors"] if _normal_author(value)}
        # Anonymous documents remain independent components.
        document_authors[document_id].update(authors or {f"anonymous:{document_id}"})
    finder = _UnionFind(document_authors)
    author_owner: dict[str, str] = {}
    for document_id in sorted(document_authors):
        for author in sorted(document_authors[document_id]):
            previous = author_owner.setdefault(author, document_id)
            finder.union(previous, document_id)
    component_documents: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(document_authors):
        component_documents[finder.find(document_id)].append(document_id)
    document_stratum = {
        example["document"]["document_id"]: example["source"]["stratum"]
        for example in examples
    }
    component_stratum_documents: dict[str, Counter[str]] = {}
    for root, documents in component_documents.items():
        component_stratum_documents[root] = Counter(
            document_stratum[document_id] for document_id in documents
        )

    total_documents = Counter(document_stratum.values())
    total_components = Counter(
        stratum
        for counts in component_stratum_documents.values()
        for stratum in counts
    )
    heldout_document_targets = {
        stratum: (
            minimum_heldout_components_per_stratum
            if total_documents[stratum] >= 2 * minimum_heldout_components_per_stratum + 1
            else max(0, (total_documents[stratum] - 1) // 2)
        )
        for stratum in required_strata
    }
    heldout_component_targets = {
        stratum: (
            minimum_heldout_components_per_stratum
            if total_components[stratum] >= 2 * minimum_heldout_components_per_stratum + 1
            else max(0, (total_components[stratum] - 1) // 2)
        )
        for stratum in required_strata
    }

    component_splits: dict[str, str] = {}
    unassigned = set(component_documents)
    for desired_index, desired in enumerate(("validation", "test")):
        assigned_documents = Counter()
        assigned_components = Counter()
        while True:
            document_deficit = {
                stratum: max(0, heldout_document_targets[stratum] - assigned_documents[stratum])
                for stratum in required_strata
            }
            component_deficit = {
                stratum: max(0, heldout_component_targets[stratum] - assigned_components[stratum])
                for stratum in required_strata
            }
            if not any(document_deficit.values()) and not any(component_deficit.values()):
                break
            later_holdouts = 1 - desired_index
            candidates: list[tuple[int, int, int, str, str]] = []
            for root in sorted(unassigned):
                counts = component_stratum_documents[root]
                remaining_documents = Counter(
                    document_stratum[document_id]
                    for candidate in unassigned
                    if candidate != root
                    for document_id in component_documents[candidate]
                )
                remaining_components = Counter(
                    stratum
                    for candidate in unassigned
                    if candidate != root
                    for stratum in component_stratum_documents[candidate]
                )
                if any(
                    remaining_documents[stratum]
                    < later_holdouts * heldout_document_targets[stratum] + 1
                    or remaining_components[stratum]
                    < later_holdouts * heldout_component_targets[stratum] + 1
                    for stratum in required_strata
                ):
                    continue
                document_gain = sum(
                    min(document_deficit[stratum], counts[stratum])
                    for stratum in required_strata
                )
                component_gain = sum(
                    component_deficit[stratum] > 0 and counts[stratum] > 0
                    for stratum in required_strata
                )
                if document_gain == 0 and component_gain == 0:
                    continue
                overshoot = sum(
                    max(0, counts[stratum] - document_deficit[stratum])
                    for stratum in required_strata
                )
                priority = sha256_text(f"{desired}\0{root}")
                candidates.append((component_gain, document_gain, -overshoot, priority, root))
            if not candidates:
                break
            chosen = max(candidates)[-1]
            component_splits[chosen] = desired
            unassigned.remove(chosen)
            counts = component_stratum_documents[chosen]
            assigned_documents.update(counts)
            assigned_components.update(counts.keys())

    for root in unassigned:
        component_splits[root] = "train"

    document_split: dict[str, str] = {
        document_id: component_splits[root]
        for root, documents in component_documents.items()
        for document_id in documents
    }
    # A deliberately tiny pilot cannot meet the production minima. Keep its
    # byte-stable three-way smoke shape without calling it trainable.
    component_sizes = Counter(
        finder.find(example["document"]["document_id"])
        for example in examples
    )
    if len(component_documents) >= 3:
        for desired in ("validation", "test", "train"):
            if desired in component_splits.values():
                continue
            candidates = [
                root
                for root in component_documents
                if Counter(component_splits.values())[component_splits[root]] > 1
            ]
            if not candidates:
                break
            chosen = min(candidates, key=lambda root: (component_sizes[root], sha256_text(root)))
            component_splits[chosen] = desired
            for document_id in component_documents[chosen]:
                document_split[document_id] = desired
    for example in examples:
        example["split"] = document_split[example["document"]["document_id"]]
    components_by_stratum_split = Counter(
        f"{stratum}|{component_splits[root]}"
        for root, counts in component_stratum_documents.items()
        for stratum in counts
    )
    return {
        "components": len(component_documents),
        "largest_component_examples": max(component_sizes.values(), default=0),
        "component_counts_by_split": dict(sorted(Counter(component_splits.values()).items())),
        "components_by_stratum_split": dict(sorted(components_by_stratum_split.items())),
        "heldout_component_target_per_stratum": minimum_heldout_components_per_stratum,
    }


def _reduce_heldout_correlation(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one independent validation/test target per document."""

    result = [example for example in examples if example["split"] == "train"]
    by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        if example["split"] != "train":
            by_document[example["document"]["document_id"]].append(example)
    for document_id, values in sorted(by_document.items()):
        result.append(
            min(
                values,
                key=lambda value: sha256_text(
                    f"{document_id}\0heldout-v2\0{value['example_id']}"
                ),
            )
        )
    return result


def compile_corpus(
    documents: Iterable[SourceDocument],
    *,
    output_path: Path,
    cutoff: date = date(2021, 12, 31),
    balance_sources: bool = True,
    required_strata: Sequence[str] = REQUIRED_STRATA,
    require_trainable_splits: bool = True,
    minimum_documents_per_split_stratum: int = MINIMUM_HELDOUT_COMPONENTS_PER_STRATUM,
    minimum_examples_per_split_stratum: int = MINIMUM_HELDOUT_COMPONENTS_PER_STRATUM,
    minimum_components_per_split_stratum: int = MINIMUM_HELDOUT_COMPONENTS_PER_STRATUM,
    candidate_filter: Callable[
        [Sequence[dict[str, Any]]], Sequence[dict[str, Any]]
    ] | None = None,
) -> dict[str, Any]:
    """Compile cached sources atomically; output is byte-stable for equal inputs."""

    if cutoff > date(2021, 12, 31):
        raise ValueError("academic corpus cutoff cannot exceed 2021-12-31")
    deduplicated_documents = _deduplicate_documents(documents)
    rejected_documents = Counter()
    selected_documents: list[SourceDocument] = []
    for document in deduplicated_documents:
        rejection = document_rejection_reason(document.title, document.prose)
        if rejection:
            rejected_documents[rejection] += 1
            continue
        selected_documents.append(document)
    for document in selected_documents:
        published = date.fromisoformat(document.published)
        latest = date.fromisoformat(document.latest_version)
        if published > cutoff or latest > cutoff:
            raise ValueError(f"post-cutoff source document rejected: {document.document_id}")
    documents_by_stratum: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in selected_documents:
        documents_by_stratum[document.stratum].append(document)
    missing = sorted(set(required_strata) - set(documents_by_stratum))
    if missing:
        raise ValueError(f"corpus is missing required source strata: {', '.join(missing)}")
    if balance_sources and documents_by_stratum:
        document_target = min(len(documents_by_stratum[stratum]) for stratum in required_strata)
        selected_documents = [
            document
            for stratum in sorted(documents_by_stratum)
            for document in sorted(
                documents_by_stratum[stratum],
                key=lambda value: (value.content_sha256, value.document_id),
            )[:document_target]
        ]
    examples = _deduplicate_examples(
        example
        for document in selected_documents
        for example in _base_examples(document, cutoff)
    )
    # Token-bound or policy gates belong here: every complete candidate is
    # measured before author-component splitting, held-out reduction, and the
    # final per-stratum example balance.  The callback may only retain or
    # annotate rows; callers that need derived examples must use a separate,
    # explicitly attested compilation path.
    if candidate_filter is not None:
        originals = {
            str(example["example_id"]): example for example in examples
        }
        original_ids = set(originals)
        filtered = list(candidate_filter(tuple(examples)))
        filtered_ids = [str(example.get("example_id", "")) for example in filtered]
        if len(filtered_ids) != len(set(filtered_ids)):
            raise ValueError("candidate_filter returned duplicate example identities")
        unexpected_ids = sorted(set(filtered_ids) - original_ids)
        if unexpected_ids:
            raise ValueError(
                "candidate_filter derived or replaced corpus rows: "
                + ", ".join(unexpected_ids[:3])
            )
        for example in filtered:
            identity = str(example["example_id"])
            original = originals[identity]
            for key, value in original.items():
                if key == "provenance":
                    continue
                if example.get(key) != value:
                    raise ValueError(
                        "candidate_filter changed a frozen corpus row: " + identity
                    )
            original_provenance = original.get("provenance", {})
            filtered_provenance = example.get("provenance", {})
            if not isinstance(filtered_provenance, Mapping) or any(
                filtered_provenance.get(key) != value
                for key, value in original_provenance.items()
            ):
                raise ValueError(
                    "candidate_filter changed frozen corpus provenance: " + identity
                )
        examples = filtered
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_stratum[example["source"]["stratum"]].append(example)
    missing_examples = sorted(set(required_strata) - set(by_stratum))
    unexpected_examples = sorted(set(by_stratum) - set(required_strata))
    if missing_examples:
        raise ValueError(f"required source strata produced no usable prose: {', '.join(missing_examples)}")
    if unexpected_examples:
        raise ValueError(f"corpus contains unconfigured source strata: {', '.join(unexpected_examples)}")
    split_diagnostics = _author_safe_splits(
        examples,
        required_strata=required_strata,
        minimum_heldout_components_per_stratum=minimum_components_per_split_stratum,
    )
    examples = _reduce_heldout_correlation(examples)
    by_stratum = defaultdict(list)
    for example in examples:
        by_stratum[example["source"]["stratum"]].append(example)
    if balance_sources and by_stratum:
        target = min(len(values) for values in by_stratum.values())
        examples = [
            example
            for stratum in sorted(by_stratum)
            for example in _round_robin_examples(by_stratum[stratum], target)
        ]
    examples.sort(key=lambda value: (value["split"], value["source"]["stratum"], value["example_id"]))
    split_counts = Counter(example["split"] for example in examples)
    stratum_counts = Counter(example["source"]["stratum"] for example in examples)
    type_counts = Counter(example["task_type"] for example in examples)
    document_ids = {example["document"]["document_id"] for example in examples}
    document_stratum = {
        example["document"]["document_id"]: example["source"]["stratum"] for example in examples
    }
    document_split = {
        example["document"]["document_id"]: example["split"] for example in examples
    }
    documents_by_stratum_count = Counter(document_stratum.values())
    documents_by_split_count = Counter(document_split.values())
    documents_by_stratum_split = Counter(
        f"{document_stratum[identity]}|{document_split[identity]}" for identity in document_ids
    )
    examples_by_stratum_split = Counter(
        f"{example['source']['stratum']}|{example['split']}" for example in examples
    )
    components_by_stratum_split = Counter(
        split_diagnostics.get("components_by_stratum_split", {})
    )
    non_trainable_reasons: list[str] = []
    expected_splits = ("train", "validation", "test")
    for stratum in required_strata:
        for split in expected_splits:
            key = f"{stratum}|{split}"
            document_count = documents_by_stratum_split[key]
            example_count = examples_by_stratum_split[key]
            if document_count < minimum_documents_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {document_count} documents; requires {minimum_documents_per_split_stratum}"
                )
            if example_count < minimum_examples_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {example_count} examples; requires {minimum_examples_per_split_stratum}"
                )
            component_count = components_by_stratum_split[key]
            if component_count < minimum_components_per_split_stratum:
                non_trainable_reasons.append(
                    f"{key} has {component_count} author components; "
                    f"requires {minimum_components_per_split_stratum}"
                )
    trainable = bool(examples) and not non_trainable_reasons
    if require_trainable_splits and not trainable:
        raise ValueError(
            "corpus lacks minimum author-safe coverage in every split/source stratum; "
            "collect more documents or explicitly build a non-trainable pilot: "
            + "; ".join(non_trainable_reasons[:4])
        )
    jsonl = "".join(canonical_json(example) + "\n" for example in examples)
    _atomic_text(output_path, jsonl)

    example_years = Counter(example["document"]["published"][:4] for example in examples)
    document_years = Counter(
        next(
            example["document"]["published"][:4]
            for example in examples
            if example["document"]["document_id"] == identity
        )
        for identity in document_ids
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "corpus_schema_version": CORPUS_SCHEMA,
        "cutoff": cutoff.isoformat(),
        "source_strata": sorted(stratum_counts),
        "balanced_sources": balance_sources,
        "trainable": trainable,
        "non_trainable_reasons": non_trainable_reasons,
        "trainability_thresholds": {
            "documents_per_split_stratum": minimum_documents_per_split_stratum,
            "examples_per_split_stratum": minimum_examples_per_split_stratum,
            "author_components_per_split_stratum": minimum_components_per_split_stratum,
        },
        "counts": {
            "examples": len(examples),
            "documents": len(document_ids),
            "by_split": dict(sorted(split_counts.items())),
            "by_stratum": dict(sorted(stratum_counts.items())),
            "by_task_type": dict(sorted(type_counts.items())),
            "documents_by_stratum": dict(sorted(documents_by_stratum_count.items())),
            "documents_by_split": dict(sorted(documents_by_split_count.items())),
            "documents_by_stratum_split": dict(sorted(documents_by_stratum_split.items())),
            "examples_by_stratum_split": dict(sorted(examples_by_stratum_split.items())),
            "examples_by_publication_year": dict(sorted(example_years.items())),
            "documents_by_publication_year": dict(sorted(document_years.items())),
        },
        "corpus_sha256": hashlib.sha256(jsonl.encode("utf-8")).hexdigest(),
        "output_filename": output_path.name,
        "split_policy": SPLIT_POLICY,
        "split_diagnostics": split_diagnostics,
        "deduplication": (
            "normalized exact document prose and target text sha256; one deterministic "
            "task granularity per document; one held-out target per document"
        ),
        "source_hygiene": {
            "rejected_documents_by_reason": dict(sorted(rejected_documents.items())),
            "notice_titles_and_prose": "reject_document",
            "promotional_urls_cme_and_malformed_units": "reject_unit",
            "citation_normalization": "paper-specific markers mapped by occurrence to numeric local slots",
            "temporal_scope": (
                "paper publication/latest-version cutoff; metadata revision date retained as provenance"
            ),
        },
        "per_document_caps": {
            "sentence": MAX_SENTENCE_EXAMPLES_PER_DOCUMENT,
            "paragraph": MAX_PARAGRAPH_EXAMPLES_PER_DOCUMENT,
        },
        "task_feasibility": {
            "sentence": (
                "bounded semantic proposition/detail slots; all quantities and at least 55% "
                "of unique content tokens; no copied five-token run"
            ),
            "paragraph": (
                "one proposition plus bounded detail slots per sentence; polarity is an "
                "attribute; all quantities and at least 55% content coverage"
            ),
        },
    }
    manifest_path = output_path.with_name(f"{output_path.name}.manifest.json")
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest
