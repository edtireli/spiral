#!/usr/bin/env python3
"""Model-free held-out evaluation for the academic paper-structure adapter.

Predictions are JSONL rows keyed by ``example_id`` with literal ``base`` and
``adapter`` output strings.  The six structure tasks require one exact JSON
object; prose-replay rows are coverage-counted here and remain semantically
scored by the existing prose evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


STRUCTURE_EVALUATION_SCHEMA = "spiral.academic-structure-evaluation.v1"
STRUCTURE_EXAMPLE_EVALUATION_SCHEMA = "spiral.academic-structure-example-evaluation.v1"
STRUCTURE_PREDICTION_SCHEMA = "spiral.academic-structure-predictions.v1"
STRUCTURE_TASKS = (
    "recognize_role",
    "order_structure",
    "budget_structure",
    "restore_section",
    "repair_structure",
    "brief_to_blueprint",
)
REPLAY_TASK = "prose_replay"
ARMS = ("base", "adapter")
SUMMARY_FILENAME = "structure_evaluation_summary.json"
EXAMPLES_FILENAME = "structure_evaluation_examples.jsonl"
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
MAX_COUNT = 1_000_000_000


class StructureEvaluationError(ValueError):
    """A malformed corpus, prediction packet, or coverage set."""


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_number:{value}")


def _strict_json(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def parse_exact_json_object(value: str) -> tuple[dict[str, Any] | None, str]:
    """Return an object only when the entire output is strict finite JSON."""

    try:
        parsed = _strict_json(value)
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError) as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(parsed, dict):
        return None, "top_level_not_object"
    return parsed, ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StructureEvaluationError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_corpus(path: Path, *, split: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise StructureEvaluationError(f"cannot read corpus {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                raise StructureEvaluationError(f"{path}:{line_number}: corpus row is too large")
            try:
                value = _strict_json(line)
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise StructureEvaluationError(
                    f"{path}:{line_number}: invalid corpus JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise StructureEvaluationError(f"{path}:{line_number}: corpus row must be an object")
            example_id = str(value.get("example_id") or "")
            if not example_id or example_id in seen:
                raise StructureEvaluationError(
                    f"{path}:{line_number}: missing or duplicate corpus example_id"
                )
            seen.add(example_id)
            if split is None or value.get("split") == split:
                _validate_corpus_row(value, path=path, line_number=line_number)
                rows.append(value)
    if not rows:
        label = "all splits" if split is None else f"split {split!r}"
        raise StructureEvaluationError(f"corpus contains no rows for {label}")
    return sorted(rows, key=lambda row: str(row["example_id"]))


def _validate_corpus_row(row: Mapping[str, Any], *, path: Path, line_number: int) -> None:
    task = str(row.get("task_type") or "")
    prefix = f"{path}:{line_number}"
    if task not in (*STRUCTURE_TASKS, REPLAY_TASK):
        raise StructureEvaluationError(f"{prefix}: unsupported structure task {task!r}")
    if task == REPLAY_TASK:
        if not isinstance(row.get("target"), str):
            raise StructureEvaluationError(f"{prefix}: prose_replay target must be a string")
        return
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise StructureEvaluationError(f"{prefix}: structure target must be an object")
    required = _response_required(row)
    if any(key not in target for key in required):
        raise StructureEvaluationError(f"{prefix}: target omits response-schema required keys")
    if not _task_schema_valid(task, target, target):
        raise StructureEvaluationError(f"{prefix}: target does not match {task} schema")


def _load_predictions(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise StructureEvaluationError(f"cannot read predictions {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                raise StructureEvaluationError(f"{path}:{line_number}: prediction row is too large")
            try:
                value = _strict_json(line)
            except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                raise StructureEvaluationError(
                    f"{path}:{line_number}: invalid prediction JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise StructureEvaluationError(f"{path}:{line_number}: prediction row must be an object")
            schema = value.get("schema_version", STRUCTURE_PREDICTION_SCHEMA)
            if schema != STRUCTURE_PREDICTION_SCHEMA:
                raise StructureEvaluationError(
                    f"{path}:{line_number}: unsupported prediction schema {schema!r}"
                )
            example_id = str(value.get("example_id") or "")
            if not example_id or example_id in result:
                raise StructureEvaluationError(
                    f"{path}:{line_number}: missing or duplicate prediction example_id"
                )
            arms: dict[str, str] = {}
            for arm in ARMS:
                output = value.get(arm)
                if not isinstance(output, str):
                    raise StructureEvaluationError(
                        f"{path}:{line_number}: {arm} must be a literal output string"
                    )
                arms[arm] = output
            result[example_id] = arms
    return result


def _response_required(row: Mapping[str, Any]) -> tuple[str, ...]:
    task_input = row.get("input")
    if not isinstance(task_input, Mapping):
        raise StructureEvaluationError("structure row input must be an object")
    response_schema = task_input.get("response_schema")
    if not isinstance(response_schema, Mapping):
        raise StructureEvaluationError("structure row is missing input.response_schema")
    raw = response_schema.get("required")
    if not isinstance(raw, list) or not all(isinstance(key, str) and key for key in raw):
        raise StructureEvaluationError("structure response_schema.required must be a string array")
    if len(set(raw)) != len(raw):
        raise StructureEvaluationError("structure response_schema.required contains duplicates")
    return tuple(raw)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= MAX_COUNT


def _safe_count(value: Any) -> int:
    return int(value) if _is_count(value) else 0


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_string_list(value: Any, *, unique: bool = False) -> bool:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return False
    return not unique or len(set(value)) == len(value)


def _is_path(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in value)
    )


def _target_item_keys(target: Mapping[str, Any], key: str) -> set[str]:
    values = target.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], Mapping):
        return set()
    return {str(field) for field in values[0]}


def _valid_records(
    value: Any,
    *,
    required: set[str],
    path_required: bool = False,
) -> bool:
    if not isinstance(value, list):
        return False
    ids: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or any(key not in item for key in required):
            return False
        for key in required:
            field = item[key]
            if key in {"id", "heading", "role", "parent_id"} and not _is_string(field):
                return False
            if key in {"words", "paragraphs", "figures", "tables"} and not _is_count(field):
                return False
            if key == "path" and not _is_path(field, allow_empty=False):
                return False
        if "id" in item:
            ids.append(str(item["id"]))
    return len(ids) == len(set(ids))


def _valid_single_record(value: Any, required: set[str]) -> bool:
    return _valid_records([value], required=required, path_required="path" in required)


def _labels_value(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping) and isinstance(item.get("label"), str):
            result.append(str(item["label"]))
        else:
            return None
    return tuple(result)


def _violation_labels(value: Mapping[str, Any]) -> tuple[str, ...] | None:
    for key in ("violation_labels", "violations"):
        if key in value:
            return _labels_value(value[key])
    return None


def _task_schema_valid(
    task: str,
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
) -> bool:
    if any(key not in candidate for key in target):
        return False
    target_labels = _violation_labels(target)
    if target_labels is not None and _violation_labels(candidate) is None:
        return False
    if task == "recognize_role":
        return _is_string(candidate.get("role"))
    if task == "order_structure":
        return (
            _is_string(candidate.get("parent_id"))
            and _is_path(candidate.get("parent_path"), allow_empty=True)
            and _is_string_list(candidate.get("ordered_section_ids"), unique=True)
        )
    if task == "budget_structure":
        required = _target_item_keys(target, "section_budgets") or {
            "id", "paragraphs", "words"
        }
        return _is_count(candidate.get("section_words")) and _valid_records(
            candidate.get("section_budgets"), required=required
        )
    if task == "restore_section":
        missing = target.get("missing_section")
        required = {str(key) for key in missing} if isinstance(missing, Mapping) else {
            "id", "heading", "path", "role", "words", "parent_id"
        }
        return _valid_single_record(candidate.get("missing_section"), required)
    if task == "repair_structure":
        required = _target_item_keys(target, "sections") or {
            "id", "heading", "path", "role", "words"
        }
        return (
            _is_string(candidate.get("parent_id"))
            and _is_path(candidate.get("parent_path"), allow_empty=True)
            and _valid_records(candidate.get("sections"), required=required)
        )
    if task == "brief_to_blueprint":
        target_counts = target.get("paper_counts")
        candidate_counts = candidate.get("paper_counts")
        if not isinstance(target_counts, Mapping) or not isinstance(candidate_counts, Mapping):
            return False
        if any(key not in candidate_counts or not _is_count(candidate_counts[key]) for key in target_counts):
            return False
        required = _target_item_keys(target, "sections") or {"id", "heading", "role", "words"}
        return _valid_records(candidate.get("sections"), required=required)
    return False


def _normal_label(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip().casefold().replace("-", "_"))


def _normal_heading(value: Any) -> str:
    return " ".join(re.sub(r"[^\w\s-]+", " ", str(value), flags=re.UNICODE).casefold().split())


def _set_f1(expected: set[Any], actual: set[Any]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _counter_f1(expected: Counter[Any], actual: Counter[Any]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _section_records(task: str, value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if task == "order_structure":
        return [
            {"id": identifier}
            for identifier in _list_or_empty(value.get("ordered_section_ids"))
            if isinstance(identifier, str)
        ]
    if task == "budget_structure":
        raw = value.get("section_budgets")
    elif task in {"repair_structure", "brief_to_blueprint"}:
        raw = value.get("sections")
    elif task == "restore_section":
        raw = [value.get("missing_section")]
    else:
        raw = []
    return [item for item in _list_or_empty(raw) if isinstance(item, Mapping)]


def _record_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(item["id"]) for item in records if isinstance(item.get("id"), str)]


def _pairwise_order_accuracy(expected: Sequence[str], actual: Sequence[str]) -> float:
    expected_unique = list(dict.fromkeys(expected))
    actual_position = {identifier: index for index, identifier in enumerate(dict.fromkeys(actual))}
    if len(expected_unique) < 2:
        return 1.0 if all(identifier in actual_position for identifier in expected_unique) else 0.0
    correct = 0
    pairs = 0
    for first_index, first in enumerate(expected_unique[:-1]):
        for second in expected_unique[first_index + 1 :]:
            pairs += 1
            if (
                first in actual_position
                and second in actual_position
                and actual_position[first] < actual_position[second]
            ):
                correct += 1
    return correct / pairs


def _field_accuracy(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    field: str,
    *,
    normalize: Any = lambda value: value,
) -> float:
    expected_map = {
        str(item["id"]): normalize(item[field])
        for item in expected
        if isinstance(item.get("id"), str) and field in item
    }
    actual_map = {
        str(item["id"]): normalize(item[field])
        for item in actual
        if isinstance(item.get("id"), str) and field in item
    }
    identities = set(expected_map) | set(actual_map)
    if not identities:
        return 1.0
    return sum(
        identity in expected_map
        and identity in actual_map
        and expected_map[identity] == actual_map[identity]
        for identity in identities
    ) / len(identities)


def _word_mae(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> float:
    expected_map = {
        str(item["id"]): _safe_count(item.get("words"))
        for item in expected
        if isinstance(item.get("id"), str)
    }
    actual_map = {
        str(item["id"]): _safe_count(item.get("words"))
        for item in actual
        if isinstance(item.get("id"), str)
    }
    identities = set(expected_map) | set(actual_map)
    if not identities:
        return 0.0
    return sum(abs(expected_map.get(key, 0) - actual_map.get(key, 0)) for key in identities) / len(
        identities
    )


def _parent_and_path(
    task: str,
    value: Mapping[str, Any],
) -> tuple[Any, Any]:
    if task in {"order_structure", "repair_structure"}:
        return value.get("parent_id"), value.get("parent_path")
    if task == "restore_section":
        section = _mapping_or_empty(value.get("missing_section"))
        path = section.get("path")
        parent_path = path[:-1] if isinstance(path, list) and path else None
        return section.get("parent_id"), parent_path
    return None, None


def _semantic_metrics(
    task: str,
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if task == "recognize_role":
        metrics["role_accuracy"] = float(
            _normal_label(candidate.get("role", "")) == _normal_label(target.get("role", ""))
        )
    expected_records = _section_records(task, target)
    actual_records = _section_records(task, candidate)
    if expected_records:
        expected_ids = _record_ids(expected_records)
        actual_ids = _record_ids(actual_records)
        metrics["section_id_f1"] = _set_f1(set(expected_ids), set(actual_ids))
        metrics["order_pairwise_accuracy"] = _pairwise_order_accuracy(expected_ids, actual_ids)
        if any("role" in item for item in expected_records):
            metrics["role_accuracy"] = _field_accuracy(
                expected_records,
                actual_records,
                "role",
                normalize=_normal_label,
            )
        if any("words" in item for item in expected_records):
            metrics["word_mae"] = _word_mae(expected_records, actual_records)

    if task in {"order_structure", "restore_section", "repair_structure"}:
        expected_parent, expected_parent_path = _parent_and_path(task, target)
        actual_parent, actual_parent_path = _parent_and_path(task, candidate)
        metrics["parent_accuracy"] = float(actual_parent == expected_parent)
        metrics["parent_path_accuracy"] = float(actual_parent_path == expected_parent_path)
        if task == "order_structure":
            metrics["path_accuracy"] = metrics["parent_path_accuracy"]
        elif task == "restore_section":
            expected_section = _mapping_or_empty(target.get("missing_section"))
            actual_section = _mapping_or_empty(candidate.get("missing_section"))
            metrics["section_path_accuracy"] = float(
                actual_section.get("path") == expected_section.get("path")
            )
            metrics["path_accuracy"] = metrics["section_path_accuracy"]
        else:
            section_path = _field_accuracy(
                expected_records,
                actual_records,
                "path",
                normalize=lambda value: tuple(value) if isinstance(value, list) else (),
            )
            metrics["section_path_accuracy"] = section_path
            metrics["path_accuracy"] = (metrics["parent_path_accuracy"] + section_path) / 2

    if task == "budget_structure":
        budgets = _list_or_empty(candidate.get("section_budgets"))
        expected_total = _safe_count(target.get("section_words"))
        candidate_total = candidate.get("section_words")
        valid_words = all(
            isinstance(item, Mapping) and _is_count(item.get("words")) for item in budgets
        )
        allocated = sum(_safe_count(item.get("words")) for item in budgets if isinstance(item, Mapping))
        metrics["budget_total_compliance"] = float(
            _is_count(candidate_total)
            and int(candidate_total) == expected_total
            and valid_words
            and allocated == expected_total
        )

    target_labels = _violation_labels(target)
    if target_labels is not None:
        candidate_labels = _violation_labels(candidate) or ()
        metrics["violation_label_f1"] = _set_f1(
            {_normal_label(value) for value in target_labels},
            {_normal_label(value) for value in candidate_labels},
        )

    if task == "brief_to_blueprint":
        expected_headings = Counter(
            _normal_heading(item.get("heading", "")) for item in expected_records
        )
        actual_headings = Counter(
            _normal_heading(item.get("heading", "")) for item in actual_records
        )
        expected_roles = {
            (str(item.get("id", "")), _normal_label(item.get("role", "")))
            for item in expected_records
        }
        actual_roles = {
            (str(item.get("id", "")), _normal_label(item.get("role", "")))
            for item in actual_records
        }
        metrics["blueprint_heading_overlap"] = _counter_f1(expected_headings, actual_headings)
        metrics["blueprint_role_overlap"] = _set_f1(expected_roles, actual_roles)
        metrics["blueprint_order_overlap"] = _pairwise_order_accuracy(
            _record_ids(expected_records), _record_ids(actual_records)
        )
    return metrics


def score_structure_candidate(row: Mapping[str, Any], output: str) -> dict[str, Any]:
    """Score one arm output without invoking a model or accepting loose JSON."""

    task = str(row.get("task_type") or "")
    if task == REPLAY_TASK:
        return {
            "parsed": None,
            "error": "",
            "metrics": {"outputs_counted": 1},
        }
    if task not in STRUCTURE_TASKS:
        raise StructureEvaluationError(f"unsupported structure task {task!r}")
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise StructureEvaluationError("structure target must be an object")
    parsed, error = parse_exact_json_object(output)
    candidate: Mapping[str, Any] = parsed or {}
    required = _response_required(row)
    required_valid = parsed is not None and all(key in parsed for key in required)
    schema_valid = parsed is not None and required_valid and _task_schema_valid(task, parsed, target)
    metrics = {
        "json_validity": float(parsed is not None),
        "required_key_validity": float(required_valid),
        "schema_validity": float(schema_valid),
        **_semantic_metrics(task, target, candidate),
    }
    _assert_finite(metrics)
    return {"parsed": parsed, "error": error, "metrics": metrics}


def _assert_finite(value: Any, path: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise StructureEvaluationError(f"{path} contains a non-finite metric")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _aggregate_arm(examples: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    by_task_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        by_task_rows[str(example["task_type"])].append(example)
    by_task: dict[str, Any] = {}
    overall_metrics: dict[str, list[float]] = defaultdict(list)
    replay_count = 0
    structure_count = 0
    for task in sorted(by_task_rows):
        rows = by_task_rows[task]
        if task == REPLAY_TASK:
            replay_count += len(rows)
            by_task[task] = {
                "example_count": len(rows),
                "metrics": {"outputs_counted": len(rows)},
            }
            continue
        structure_count += len(rows)
        values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            scores = row[arm]["metrics"]
            for metric, score in scores.items():
                values[str(metric)].append(float(score))
                overall_metrics[str(metric)].append(float(score))
        by_task[task] = {
            "example_count": len(rows),
            "metrics": {
                metric: sum(scores) / len(scores)
                for metric, scores in sorted(values.items())
            },
        }
    result = {
        "overall": {
            "example_count": len(examples),
            "structure_example_count": structure_count,
            "replay_outputs_counted": replay_count,
            "metrics": {
                metric: sum(scores) / len(scores)
                for metric, scores in sorted(overall_metrics.items())
            },
        },
        "by_task": by_task,
    }
    _assert_finite(result)
    return result


def evaluate_structure_predictions(
    corpus_path: Path,
    predictions_path: Path,
    output_directory: Path,
    *,
    split: str | None = "test",
) -> dict[str, Any]:
    """Evaluate exactly covered rows and atomically emit deterministic artifacts."""

    corpus = _load_corpus(corpus_path, split=split)
    predictions = _load_predictions(predictions_path)
    expected_ids = {str(row["example_id"]) for row in corpus}
    observed_ids = set(predictions)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise StructureEvaluationError(
            "predictions do not exactly cover evaluated examples; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    examples: list[dict[str, Any]] = []
    for row in corpus:
        example_id = str(row["example_id"])
        entry: dict[str, Any] = {
            "schema_version": STRUCTURE_EXAMPLE_EVALUATION_SCHEMA,
            "example_id": example_id,
            "split": str(row.get("split", "")),
            "task_type": str(row["task_type"]),
            "source_stratum": str(_mapping_or_empty(row.get("source")).get("stratum", "")),
            "target": row["target"],
        }
        for arm in ARMS:
            text = predictions[example_id][arm]
            score = score_structure_candidate(row, text)
            entry[arm] = {"text": text, **score}
        examples.append(entry)

    examples_payload = b"".join(_canonical_line(example) for example in examples)
    output_directory.mkdir(parents=True, exist_ok=True)
    examples_path = output_directory / EXAMPLES_FILENAME
    _atomic_write(examples_path, examples_payload)
    task_counts = Counter(str(row["task_type"]) for row in corpus)
    summary = {
        "schema_version": STRUCTURE_EVALUATION_SCHEMA,
        "split": "all" if split is None else split,
        "counts": {
            "examples": len(corpus),
            "structure_examples": sum(task_counts[task] for task in STRUCTURE_TASKS),
            "prose_replay_examples": task_counts[REPLAY_TASK],
            "by_task": dict(sorted(task_counts.items())),
        },
        "corpus_sha256": _sha256_file(corpus_path),
        "predictions_sha256": _sha256_file(predictions_path),
        "examples_filename": EXAMPLES_FILENAME,
        "examples_sha256": hashlib.sha256(examples_payload).hexdigest(),
        "arms": {arm: _aggregate_arm(examples, arm) for arm in ARMS},
        "metric_direction": {
            "word_mae": "lower_is_better",
            "all_other_fraction_metrics": "higher_is_better",
        },
        "replay_policy": (
            "prose_replay outputs are coverage-counted only; semantic prose regression remains "
            "the responsibility of scripts/academic_finetune/evaluate.py"
        ),
    }
    _assert_finite(summary)
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(output_directory / SUMMARY_FILENAME, summary_payload)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--corpus", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
        help="corpus split to evaluate (default: test)",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        summary = evaluate_structure_predictions(
            arguments.corpus,
            arguments.predictions,
            arguments.output_dir,
            split=None if arguments.split == "all" else arguments.split,
        )
    except StructureEvaluationError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"evaluated {summary['counts']['examples']} examples; "
        f"wrote {arguments.output_dir / SUMMARY_FILENAME} and "
        f"{arguments.output_dir / EXAMPLES_FILENAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXAMPLES_FILENAME",
    "STRUCTURE_EVALUATION_SCHEMA",
    "STRUCTURE_PREDICTION_SCHEMA",
    "SUMMARY_FILENAME",
    "StructureEvaluationError",
    "evaluate_structure_predictions",
    "main",
    "parse_exact_json_object",
    "parser",
    "score_structure_candidate",
]
