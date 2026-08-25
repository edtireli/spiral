"""Model-free support for Spiral's Qwen3.8 academic QLoRA harness.

The functions in this module deliberately do not import MLX.  They validate the
corpus and local checkpoint, construct the exact MLX-LM inputs, and publish
content-addressed receipts before the heavyweight process is allowed to start.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import queue
import re
import signal
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from spiral.academic_structure_contract import (
    format_structure_prompt as format_structure_task_prompt,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - MLX production is macOS-only
    fcntl = None


CORPUS_SCHEMA = "spiral.academic-plan-prose.v1"
STRUCTURE_CORPUS_SCHEMA = "spiral.academic-paper-structure.v1"
DATASET_SCHEMA = "spiral.academic-mlx-dataset.v1"
ADAPTER_SCHEMA = "spiral.academic-adapter.v1"
PREFLIGHT_SCHEMA = "spiral.academic-preflight.v1"
TRAINING_METRIC_SCHEMA = "spiral.academic-training-metric.v1"
TRAINING_RUN_SCHEMA = "spiral.academic-training-run.v1"
TRAINING_DATA_VIEW_SCHEMA = "spiral.academic-training-data-view.v1"
CHECKPOINT_SCHEMA = "spiral.academic-checkpoint.v2"
CHECKPOINT_POINTER_SCHEMA = "spiral.academic-checkpoint-pointer.v2"
TRAINING_SUPERVISOR_EVENT_SCHEMA = "spiral.academic-training-supervisor-event.v1"
TRAINING_SUPERVISOR_STATUS_SCHEMA = "spiral.academic-training-supervisor-status.v1"
PROFILE_ID = "academic-hep-pubmed-v1"
PROMPT_CONTRACT = CORPUS_SCHEMA
STRUCTURE_PROFILE_ID = "academic-hep-pubmed-structure-v1"
STRUCTURE_PROMPT_CONTRACT = "spiral.academic-paper-blueprint.v1"
STRUCTURE_CORPUS_MANIFEST_SCHEMA = "spiral.academic-structure-corpus-manifest.v1"
STRUCTURE_EXACT_TOKEN_GATE_METHOD = (
    "mlx_lm.CompletionsDataset.apply_chat_template parity"
)
STRUCTURE_EXACT_TOKEN_OVERFLOW_POLICY = (
    "reject_candidate_never_truncate_or_partition"
)
PARENT_ADAPTER_SCHEMA = "spiral.academic-parent-adapter.v1"
ADAPTER_LINEAGE_SCHEMA = "spiral.academic-adapter-lineage.v1"
STRUCTURE_TASK_TYPES = frozenset({
    "recognize_role",
    "order_structure",
    "budget_structure",
    "restore_section",
    "repair_structure",
    "brief_to_blueprint",
})
STRUCTURE_REPLAY_TASK = "prose_replay"
STRUCTURE_LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.v_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
SUPPORTED_PROFILE_CONTRACTS = {
    PROFILE_ID: PROMPT_CONTRACT,
    STRUCTURE_PROFILE_ID: STRUCTURE_PROMPT_CONTRACT,
}
CORPUS_SCHEMA_BY_PROMPT_CONTRACT = {
    PROMPT_CONTRACT: CORPUS_SCHEMA,
    STRUCTURE_PROMPT_CONTRACT: STRUCTURE_CORPUS_SCHEMA,
}
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_LAYER_COUNT = 64
EXPECTED_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
EXPECTED_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
EXPECTED_CONFIG_SHA256 = "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"
EXPECTED_WEIGHT_INDEX_SHA256 = "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
EXPECTED_WEIGHT_INVENTORY_SHA256 = "8126a3fd4aef3346254965791eedc5a5468bf7fcf46bdd95ef29dd13266ed589"
EXPECTED_WEIGHT_FILES = (
    {
        "path": "model-00001-of-00003.safetensors",
        "size_bytes": 5_343_268_662,
        "sha256": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    },
    {
        "path": "model-00002-of-00003.safetensors",
        "size_bytes": 5_354_185_130,
        "sha256": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    },
    {
        "path": "model-00003-of-00003.safetensors",
        "size_bytes": 5_357_087_557,
        "sha256": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    },
)
REQUIRED_STRATA = frozenset({"arxiv:hep-th", "arxiv:hep-ph", "pubmed"})
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXACT_CITATION_RE = re.compile(
    r"(?:\[[0-9,;\-– ]+\]|\([A-Z][A-Za-z'’\-]+(?: et al\.)?,? \d{4}[a-z]?\))")
LEAKAGE_TOKEN_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


class HarnessError(RuntimeError):
    """A fail-closed validation error with a user-actionable message."""


TRAIN_METRIC_RE = re.compile(
    r"^Iter (?P<iteration>\d+): Train loss (?P<train_loss>\S+), "
    r"Learning Rate (?P<learning_rate>\S+), It/sec (?P<iterations_per_second>\S+), "
    r"Tokens/sec (?P<tokens_per_second>\S+), Trained Tokens (?P<trained_tokens>\d+), "
    r"Peak mem (?P<peak_memory_gb>\S+) GB$")
VAL_METRIC_RE = re.compile(
    r"^Iter (?P<iteration>\d+): Val loss (?P<val_loss>\S+), "
    r"Val took (?P<validation_seconds>\S+)s$")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json(value))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_toml_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HarnessError(f"cannot read training config {path}: {exc}") from exc
    validate_training_config(config)
    return config


def validate_training_config(config: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if config.get("schema_version") != "spiral.academic-qlora-config.v1":
        errors.append("unsupported or missing config schema_version")
    profile = config.get("profile", {})
    profile_id = profile.get("id") if isinstance(profile, Mapping) else None
    prompt_contract = (
        profile.get("prompt_contract") if isinstance(profile, Mapping) else None
    )
    expected_prompt = SUPPORTED_PROFILE_CONTRACTS.get(str(profile_id))
    if expected_prompt is None:
        errors.append(
            "profile.id must be academic-hep-pubmed-v1 or "
            "academic-hep-pubmed-structure-v1"
        )
    elif prompt_contract != expected_prompt:
        errors.append(
            f"profile.prompt_contract must be {expected_prompt!r} for {profile_id!r}"
        )

    base = config.get("base_model", {})
    if base.get("model_id") != EXPECTED_MODEL_ID:
        errors.append(f"base_model.model_id must be immutable {EXPECTED_MODEL_ID!r}")
    if base.get("model_type") != EXPECTED_MODEL_TYPE:
        errors.append(f"base_model.model_type must be {EXPECTED_MODEL_TYPE!r}")
    if base.get("architecture") != EXPECTED_ARCHITECTURE:
        errors.append(f"base_model.architecture must be {EXPECTED_ARCHITECTURE!r}")
    revision = str(base.get("revision", ""))
    if not REVISION_RE.fullmatch(revision):
        errors.append("base_model.revision must be an immutable 40-character SHA-1")
    elif revision != EXPECTED_REVISION:
        errors.append(f"base_model.revision must be immutable {EXPECTED_REVISION}")
    for field in ("config_sha256", "weight_index_sha256", "weight_inventory_sha256"):
        if not SHA256_RE.fullmatch(str(base.get(field, ""))):
            errors.append(f"base_model.{field} must pin an exact SHA-256")
    weight_files = base.get("weight_files")
    inventory_lines: list[str] = []
    inventory_bytes = 0
    seen_weight_paths: set[str] = set()
    if not isinstance(weight_files, list) or not weight_files:
        errors.append("base_model.weight_files must pin the exact shard inventory")
    else:
        for index, row in enumerate(weight_files):
            if not isinstance(row, Mapping):
                errors.append(f"base_model.weight_files[{index}] must be an object")
                continue
            path = str(row.get("path", ""))
            size = row.get("size_bytes")
            digest = str(row.get("sha256", ""))
            if (not path or Path(path).name != path or path in seen_weight_paths
                    or not isinstance(size, int) or isinstance(size, bool) or size <= 0
                    or not SHA256_RE.fullmatch(digest)):
                errors.append(
                    f"base_model.weight_files[{index}] must pin a unique shard name, "
                    "positive byte size, and SHA-256")
                continue
            seen_weight_paths.add(path)
            inventory_bytes += size
            inventory_lines.append(f"{path}\0{size}\0{digest}\n")
        if inventory_lines:
            configured_inventory = sha256_bytes(
                "".join(sorted(inventory_lines)).encode("utf-8"))
            if configured_inventory != base.get("weight_inventory_sha256"):
                errors.append(
                    "base_model.weight_inventory_sha256 does not match base_model.weight_files")
            if base.get("weight_bytes") != inventory_bytes:
                errors.append("base_model.weight_bytes does not match the pinned shard sizes")
            if tuple(weight_files) != EXPECTED_WEIGHT_FILES:
                errors.append("base_model.weight_files is not the exact Qwen3.8 shard inventory")
    expected_digests = {
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "weight_index_sha256": EXPECTED_WEIGHT_INDEX_SHA256,
        "weight_inventory_sha256": EXPECTED_WEIGHT_INVENTORY_SHA256,
    }
    for field, expected in expected_digests.items():
        if base.get(field) != expected:
            errors.append(f"base_model.{field} is not the pinned Qwen3.8 identity")
    quant = base.get("quantization", {})
    if quant.get("bits") != 4 or quant.get("group_size") != 64:
        errors.append("the base must be the verified 4-bit, group-size-64 MLX checkpoint")

    training = config.get("training", {})
    if training.get("fine_tune_type") != "lora":
        errors.append("training.fine_tune_type must be 'lora' (QLoRA comes from the quantized base)")
    if training.get("batch_size") != 1:
        errors.append("training.batch_size must be 1 for the 27B memory boundary")
    if training.get("mask_prompt") is not True:
        errors.append("training.mask_prompt must be true (completion-only loss)")
    if training.get("grad_checkpoint") is not True:
        errors.append("training.grad_checkpoint must be true")
    accumulation = _positive_int(training.get("grad_accumulation_steps"), "training.grad_accumulation_steps", errors)
    iterations = _positive_int(training.get("iterations"), "training.iterations", errors)
    save_every = _positive_int(training.get("save_every"), "training.save_every", errors)
    num_layers = _positive_int(training.get("num_layers"), "training.num_layers", errors)
    if num_layers and num_layers > EXPECTED_LAYER_COUNT:
        errors.append(f"training.num_layers cannot exceed {EXPECTED_LAYER_COUNT}")
    if accumulation and save_every and save_every % accumulation:
        errors.append("training.save_every must land on a gradient-accumulation update boundary")
    if accumulation and iterations and iterations % accumulation:
        errors.append("training.iterations must be divisible by grad_accumulation_steps")
    if not isinstance(training.get("seed"), int) or isinstance(training.get("seed"), bool):
        errors.append("training.seed must be an integer")

    lora = config.get("lora", {})
    keys = lora.get("keys")
    if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
        errors.append("lora.keys must be a non-empty list of module paths")
        keys = []
    full_attention = {"self_attn.q_proj", "self_attn.v_proj"}
    linear_attention = {"linear_attn.in_proj_qkv", "linear_attn.out_proj"}
    if not full_attention.intersection(keys):
        errors.append("lora.keys must target Qwen3.5 full-attention layers")
    if not linear_attention.intersection(keys):
        errors.append("lora.keys must target Qwen3.5 linear-attention layers")
    allowed = full_attention | linear_attention | {
        "self_attn.k_proj", "self_attn.o_proj",
        "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    unknown = sorted(set(keys) - allowed)
    if unknown:
        errors.append(f"lora.keys contains unverified Qwen3.5 paths: {', '.join(unknown)}")
    for name in ("rank",):
        _positive_int(lora.get(name), f"lora.{name}", errors)
    for name in ("scale", "dropout"):
        value = lora.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            errors.append(f"lora.{name} must be finite")

    if profile_id == STRUCTURE_PROFILE_ID:
        exact_training = {
            "iterations": 1200,
            "learning_rate": 0.000002,
            "max_seq_length": 448,
            "num_layers": 4,
            "seed": 25082026,
        }
        for field_name, expected in exact_training.items():
            if training.get(field_name) != expected:
                errors.append(
                    f"structure profile training.{field_name} must be {expected!r}")
        if tuple(keys) != STRUCTURE_LORA_KEYS:
            errors.append(
                "structure profile lora.keys must preserve the stage-one topology")
        if (
            lora.get("rank") != 16
            or lora.get("scale") != 32.0
            or lora.get("dropout") != 0.0
        ):
            errors.append(
                "structure profile LoRA topology must remain rank 16, scale 32, "
                "dropout 0")

    if errors:
        raise HarnessError("invalid academic QLoRA config:\n- " + "\n- ".join(errors))


def _positive_int(value: Any, name: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{name} must be a positive integer")
        return 0
    return value


def _prompt_contract(config: Mapping[str, Any] | None) -> str:
    if config is None:
        return PROMPT_CONTRACT
    profile = config.get("profile")
    if not isinstance(profile, Mapping):
        raise HarnessError("academic QLoRA config has no profile contract")
    selected = str(profile.get("prompt_contract", ""))
    if selected not in CORPUS_SCHEMA_BY_PROMPT_CONTRACT:
        raise HarnessError(f"unsupported academic prompt contract {selected!r}")
    return selected


def _corpus_schema(config: Mapping[str, Any] | None) -> str:
    return CORPUS_SCHEMA_BY_PROMPT_CONTRACT[_prompt_contract(config)]


def read_corpus_records(
    path: Path, *, config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
    else:
        candidates = [path]
    if not candidates:
        raise HarnessError(f"no JSONL corpus files found at {path}")
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise HarnessError(f"{candidate}:{line_number}: invalid JSON: {exc}") from exc
                    if not isinstance(row, dict):
                        raise HarnessError(f"{candidate}:{line_number}: record must be an object")
                    records.append(row)
        except OSError as exc:
            raise HarnessError(f"cannot read corpus {candidate}: {exc}") from exc
    validate_corpus_records(records, config=config)
    return records


def _normalised_author(value: str) -> str:
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in unicodedata.normalize("NFKC", value).casefold()
        ).split()
    )


def _validate_json_tree(
    value: Any, *, label: str, errors: list[str], depth: int = 0,
    nodes: list[int] | None = None,
) -> None:
    """Reject non-JSON, non-finite, or pathologically large structure values."""

    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > 2_048:
        errors.append(f"{label}: JSON value exceeds 2,048 nodes")
        return
    if depth > 12:
        errors.append(f"{label}: JSON nesting exceeds 12 levels")
        return
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            if not value.strip():
                errors.append(f"{label}: JSON strings must be non-empty")
            elif any(ord(character) < 32 and character not in "\n\t" for character in value):
                errors.append(f"{label}: JSON strings contain control characters")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{label}: JSON numbers must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(
                item, label=f"{label}[{index}]", errors=errors,
                depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{label}: JSON object keys must be non-empty strings")
                continue
            _validate_json_tree(
                item, label=f"{label}.{key}", errors=errors,
                depth=depth + 1, nodes=nodes)
        return
    errors.append(f"{label}: value of type {type(value).__name__} is not JSON")


def _validate_json_schema_instance(
    value: Any, schema: Mapping[str, Any], *, label: str, errors: list[str],
) -> None:
    """Validate the deterministic JSON-Schema subset emitted by the compiler."""

    supported_keywords = {
        "type", "const", "enum", "required", "properties",
        "additionalProperties", "items", "minItems", "minLength", "minimum",
    }
    unsupported = sorted(set(schema) - supported_keywords)
    if unsupported:
        errors.append(
            f"{label}: response_schema contains unsupported keyword(s): "
            + ", ".join(unsupported))
        return
    selected_type = schema.get("type")
    type_matches = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if selected_type is not None:
        if selected_type not in type_matches:
            errors.append(f"{label}: response_schema contains unsupported type {selected_type!r}")
            return
        if not type_matches[selected_type]:
            errors.append(f"{label}: target does not satisfy response_schema type {selected_type}")
            return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{label}: target does not satisfy response_schema const")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or value not in enum):
        errors.append(f"{label}: target does not satisfy response_schema enum")
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            errors.append(f"{label}: response_schema.required must be a string array")
            required = []
        for key in required:
            if key not in value:
                errors.append(f"{label}: target is missing response_schema key {key!r}")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping) or any(
            not isinstance(key, str) or not isinstance(spec, Mapping)
            for key, spec in properties.items()
        ):
            errors.append(f"{label}: response_schema.properties must be an object")
            properties = {}
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(
                    f"{label}: target has response_schema-forbidden key(s): "
                    + ", ".join(extra))
        for key, item in value.items():
            child = properties.get(key)
            if isinstance(child, Mapping):
                _validate_json_schema_instance(
                    item, child, label=f"{label}.{key}", errors=errors)
    if isinstance(value, list):
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and len(value) < minimum:
            errors.append(f"{label}: target has fewer than response_schema.minItems")
        items = schema.get("items")
        if items is not None and not isinstance(items, Mapping):
            errors.append(f"{label}: response_schema.items must be an object")
        elif isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_json_schema_instance(
                    item, items, label=f"{label}[{index}]", errors=errors)
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and not isinstance(minimum, bool) and len(value) < minimum:
            errors.append(f"{label}: target is shorter than response_schema.minLength")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{label}: target is below response_schema.minimum")


def _nonnegative_integer(
    value: Any, *, label: str, errors: list[str],
) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{label}: must be a non-negative integer")
        return None
    return value


def _validate_budget_rows(
    rows: Any, *, label: str, errors: list[str], require_paragraphs: bool,
) -> tuple[int, int] | None:
    if not isinstance(rows, list) or not rows:
        errors.append(f"{label}: must be a non-empty array")
        return None
    identifiers: set[str] = set()
    total_words = 0
    valid_words = 0
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{row_label}: must be an object")
            continue
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(f"{row_label}.id: must be a non-empty string")
        elif identifier in identifiers:
            errors.append(f"{row_label}.id: duplicate section id {identifier!r}")
        else:
            identifiers.add(identifier)
        words = _nonnegative_integer(
            row.get("words"), label=f"{row_label}.words", errors=errors)
        if words is not None:
            total_words += words
            valid_words += 1
        if require_paragraphs:
            _nonnegative_integer(
                row.get("paragraphs"),
                label=f"{row_label}.paragraphs", errors=errors)
        for optional_count in ("figures", "tables"):
            if optional_count in row:
                _nonnegative_integer(
                    row.get(optional_count),
                    label=f"{row_label}.{optional_count}", errors=errors)
    return total_words, valid_words


def _validate_structure_semantics(
    task_type: Any, target: Mapping[str, Any], *, label: str,
    errors: list[str],
) -> None:
    """Enforce arithmetic invariants that JSON Schema alone cannot express."""

    if task_type == "budget_structure":
        section_words = _nonnegative_integer(
            target.get("section_words"),
            label=f"{label}.section_words", errors=errors)
        budget_result = _validate_budget_rows(
            target.get("section_budgets"),
            label=f"{label}.section_budgets", errors=errors,
            require_paragraphs=True)
        if section_words is not None and budget_result is not None:
            budget_words, valid_words = budget_result
            budgets = target.get("section_budgets")
            if (
                isinstance(budgets, list)
                and valid_words == len(budgets)
                and budget_words != section_words
            ):
                errors.append(
                    f"{label}: section_budgets words sum to {budget_words}, "
                    f"not section_words {section_words}")
        return

    if task_type != "brief_to_blueprint":
        return
    paper_counts = target.get("paper_counts")
    if not isinstance(paper_counts, Mapping):
        errors.append(f"{label}.paper_counts: must be an object")
        section_words = None
    else:
        count_values = {
            key: _nonnegative_integer(
                paper_counts.get(key), label=f"{label}.paper_counts.{key}",
                errors=errors)
            for key in (
                "abstract_words", "section_words", "section_paragraphs",
                "unsectioned_words", "figures", "tables",
            )
        }
        section_words = count_values["section_words"]
    sections = target.get("sections")
    section_result = _validate_budget_rows(
        sections, label=f"{label}.sections", errors=errors,
        require_paragraphs=False)
    if isinstance(sections, list):
        for index, section in enumerate(sections):
            if not isinstance(section, Mapping):
                continue
            for field_name in ("heading", "role"):
                value = section.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"{label}.sections[{index}].{field_name}: "
                        "must be a non-empty string")
    if section_words is not None and section_result is not None:
        budget_words, valid_words = section_result
        if (
            isinstance(sections, list)
            and valid_words == len(sections)
            and budget_words != section_words
        ):
            errors.append(
                f"{label}: section words sum to {budget_words}, "
                f"not paper_counts.section_words {section_words}")


def _validate_structure_corpus_records(
    records: Sequence[Mapping[str, Any]], *, require_all_strata: bool,
) -> None:
    if not records:
        raise HarnessError("the academic structure corpus is empty")
    errors: list[str] = []
    seen_ids: set[str] = set()
    document_splits: dict[str, str] = {}
    author_splits: dict[str, str] = {}
    splits: set[str] = set()
    strata: set[str] = set()
    observed_tasks: set[str] = set()
    allowed_tasks = STRUCTURE_TASK_TYPES | {STRUCTURE_REPLAY_TASK}
    for index, row in enumerate(records):
        label = f"record {index + 1}"
        if row.get("schema_version") != STRUCTURE_CORPUS_SCHEMA:
            errors.append(
                f"{label}: schema_version must be {STRUCTURE_CORPUS_SCHEMA!r}")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id.strip():
            errors.append(f"{label}: example_id is required")
        elif example_id in seen_ids:
            errors.append(f"{label}: duplicate example_id {example_id!r}")
        else:
            seen_ids.add(example_id)
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            errors.append(f"{label}: split must be train, validation, or test")
        else:
            splits.add(str(split))
        task_type = row.get("task_type")
        if task_type not in allowed_tasks:
            errors.append(
                f"{label}: task_type must be one of {', '.join(sorted(allowed_tasks))}")
        else:
            observed_tasks.add(str(task_type))
        source = row.get("source")
        if not isinstance(source, Mapping):
            errors.append(f"{label}: source object is required")
        else:
            stratum = source.get("stratum")
            if stratum not in REQUIRED_STRATA:
                errors.append(
                    f"{label}: source.stratum must be arxiv:hep-th, arxiv:hep-ph, or pubmed")
            else:
                strata.add(str(stratum))
        document = row.get("document")
        if not isinstance(document, Mapping) or not document.get("document_id"):
            errors.append(f"{label}: document.document_id is required")
        elif split in {"train", "validation", "test"}:
            document_id = str(document["document_id"])
            previous = document_splits.setdefault(document_id, str(split))
            if previous != split:
                errors.append(
                    f"{label}: document {document_id!r} leaks across {previous!r} and {split!r}")
            authors = document.get("authors")
            if not isinstance(authors, list) or any(
                not isinstance(author, str) for author in authors
            ):
                errors.append(f"{label}: document.authors must be a list of strings")
            else:
                for author in authors:
                    normalised = _normalised_author(author)
                    if not normalised:
                        continue
                    author_previous = author_splits.setdefault(normalised, str(split))
                    if author_previous != split:
                        errors.append(
                            f"{label}: author {author!r} leaks across "
                            f"{author_previous!r} and {split!r}")
        prompt_input = row.get("input")
        if not isinstance(prompt_input, Mapping) or not prompt_input:
            errors.append(f"{label}: input must be a non-empty JSON object")
            continue
        _validate_json_tree(prompt_input, label=f"{label}.input", errors=errors)
        target = row.get("target")
        if task_type == STRUCTURE_REPLAY_TASK:
            if not isinstance(target, str) or not target.strip():
                errors.append(f"{label}: prose_replay target must be non-empty prose")
                continue
            unit = prompt_input.get("unit")
            if unit not in {"sentence", "paragraph"}:
                errors.append(
                    f"{label}: prose_replay input.unit must be sentence or paragraph")
            context = prompt_input.get("context")
            if not isinstance(context, str):
                errors.append(f"{label}: prose_replay input.context must be a string")
            elif target.strip():
                leakage_reason = target_context_leakage_reason(target, context)
                if leakage_reason:
                    errors.append(
                        f"{label}: target leaks into input.context ({leakage_reason})")
            claims = prompt_input.get("claims")
            if not isinstance(claims, list) or not claims or any(
                not isinstance(claim, str) or not claim.strip() for claim in claims
            ):
                errors.append(
                    f"{label}: prose_replay input.claims must contain non-empty strings")
            for field_name in ("rhetorical_relation", "certainty"):
                if not isinstance(prompt_input.get(field_name), str) or not str(
                    prompt_input.get(field_name, "")
                ).strip():
                    errors.append(f"{label}: prose_replay input.{field_name} is required")
            citation_count = prompt_input.get("citation_count")
            citation_slots = prompt_input.get("citation_slots")
            expected_slots = EXACT_CITATION_RE.findall(target)
            if (
                not isinstance(citation_count, int)
                or isinstance(citation_count, bool)
                or citation_count < 0
            ):
                errors.append(
                    f"{label}: prose_replay input.citation_count must be a non-negative integer")
            if citation_slots != expected_slots:
                errors.append(
                    f"{label}: prose_replay input.citation_slots must exactly match target")
            if isinstance(citation_count, int) and citation_count != len(expected_slots):
                errors.append(
                    f"{label}: prose_replay input.citation_count does not match target")
        else:
            if not isinstance(target, Mapping) or not target:
                errors.append(
                    f"{label}: structure target must be a non-empty JSON object")
                continue
            _validate_json_tree(target, label=f"{label}.target", errors=errors)
            response_schema = prompt_input.get("response_schema")
            if not isinstance(response_schema, Mapping) or response_schema.get("type") != "object":
                errors.append(
                    f"{label}: structure input.response_schema must describe an object")
            else:
                _validate_json_schema_instance(
                    target, response_schema, label=f"{label}.target", errors=errors)
            _validate_structure_semantics(
                task_type, target, label=f"{label}.target", errors=errors)
            try:
                encoded = canonical_json(target)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}: structure target is not canonical JSON: {exc}")
            else:
                if len(encoded) > 64 * 1024:
                    errors.append(f"{label}: canonical structure target exceeds 64 KiB")
    missing_splits = {"train", "validation", "test"} - splits
    if missing_splits:
        errors.append(
            f"corpus is missing required split(s): {', '.join(sorted(missing_splits))}")
    if require_all_strata:
        missing_strata = REQUIRED_STRATA - strata
        if missing_strata:
            errors.append(
                f"corpus is missing required source strata: {', '.join(sorted(missing_strata))}")
    if not observed_tasks.intersection(STRUCTURE_TASK_TYPES):
        errors.append("structure corpus contains no structure-supervision task")
    if errors:
        raise HarnessError(
            "invalid academic structure corpus:\n- " + "\n- ".join(errors[:50]))


def validate_corpus_records(
    records: Sequence[Mapping[str, Any]], *, require_all_strata: bool = False,
    config: Mapping[str, Any] | None = None,
) -> None:
    if _corpus_schema(config) == STRUCTURE_CORPUS_SCHEMA:
        _validate_structure_corpus_records(
            records, require_all_strata=require_all_strata)
        return
    if not records:
        raise HarnessError("the academic corpus is empty")
    errors: list[str] = []
    seen_ids: set[str] = set()
    document_splits: dict[str, str] = {}
    author_splits: dict[str, str] = {}
    splits: set[str] = set()
    strata: set[str] = set()
    for index, row in enumerate(records):
        label = f"record {index + 1}"
        if row.get("schema_version") != CORPUS_SCHEMA:
            errors.append(f"{label}: schema_version must be {CORPUS_SCHEMA!r}")
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id.strip():
            errors.append(f"{label}: example_id is required")
        elif example_id in seen_ids:
            errors.append(f"{label}: duplicate example_id {example_id!r}")
        else:
            seen_ids.add(example_id)
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            errors.append(f"{label}: split must be train, validation, or test")
        else:
            splits.add(str(split))
        if row.get("task_type") not in {"sentence", "paragraph"}:
            errors.append(f"{label}: task_type must be sentence or paragraph")
        source = row.get("source")
        if not isinstance(source, Mapping):
            errors.append(f"{label}: source object is required")
        else:
            stratum = source.get("stratum")
            if stratum not in REQUIRED_STRATA:
                errors.append(
                    f"{label}: source.stratum must be arxiv:hep-th, arxiv:hep-ph, or pubmed"
                )
            else:
                strata.add(str(stratum))
        document = row.get("document")
        if not isinstance(document, Mapping) or not document.get("document_id"):
            errors.append(f"{label}: document.document_id is required")
        elif split in {"train", "validation", "test"}:
            document_id = str(document["document_id"])
            previous = document_splits.setdefault(document_id, str(split))
            if previous != split:
                errors.append(
                    f"{label}: document {document_id!r} leaks across {previous!r} and {split!r}"
                )
            authors = document.get("authors")
            if not isinstance(authors, list) or any(not isinstance(author, str) for author in authors):
                errors.append(f"{label}: document.authors must be a list of strings")
            else:
                for author in authors:
                    normalized = " ".join(
                        "".join(
                            character if character.isalnum() else " "
                            for character in unicodedata.normalize("NFKC", author).casefold()
                        ).split()
                    )
                    if not normalized:
                        continue
                    author_previous = author_splits.setdefault(normalized, str(split))
                    if author_previous != split:
                        errors.append(
                            f"{label}: author {author!r} leaks across {author_previous!r} and {split!r}"
                        )
        target = row.get("target")
        if not isinstance(target, str) or not target.strip():
            errors.append(f"{label}: target must be non-empty")
        prompt_input = row.get("input")
        if not isinstance(prompt_input, Mapping):
            errors.append(f"{label}: input object is required")
        else:
            if not isinstance(prompt_input.get("context"), str):
                errors.append(f"{label}: input.context must be a string")
            elif isinstance(target, str) and target.strip():
                leakage_reason = target_context_leakage_reason(
                    target, str(prompt_input["context"]))
                if leakage_reason:
                    errors.append(
                        f"{label}: target leaks into input.context ({leakage_reason})")
            claims = prompt_input.get("claims")
            if not isinstance(claims, list) or not claims or any(not isinstance(claim, str) or not claim.strip() for claim in claims):
                errors.append(f"{label}: input.claims must contain non-empty strings")
            if not isinstance(prompt_input.get("rhetorical_relation"), str) or not prompt_input.get("rhetorical_relation", "").strip():
                errors.append(f"{label}: input.rhetorical_relation is required")
            if not isinstance(prompt_input.get("certainty"), str) or not prompt_input.get("certainty", "").strip():
                errors.append(f"{label}: input.certainty is required")
            citation_count = prompt_input.get("citation_count")
            if not isinstance(citation_count, int) or isinstance(citation_count, bool) or citation_count < 0:
                errors.append(f"{label}: input.citation_count must be a non-negative integer")
            citation_slots = prompt_input.get("citation_slots")
            expected_slots = EXACT_CITATION_RE.findall(target) if isinstance(target, str) else []
            if citation_slots != expected_slots:
                errors.append(
                    f"{label}: input.citation_slots must be the exact ordered citation markers present in target")
            if isinstance(citation_count, int) and citation_count != len(expected_slots):
                errors.append(f"{label}: input.citation_count does not match target citation markers")
    missing_splits = {"train", "validation", "test"} - splits
    if missing_splits:
        errors.append(f"corpus is missing required split(s): {', '.join(sorted(missing_splits))}")
    if require_all_strata:
        missing_strata = REQUIRED_STRATA - strata
        if missing_strata:
            errors.append(f"corpus is missing required source strata: {', '.join(sorted(missing_strata))}")
    if errors:
        raise HarnessError("invalid academic corpus:\n- " + "\n- ".join(errors[:50]))


def _leakage_tokens(text: str) -> list[str]:
    return [
        token.casefold().replace("’", "'")
        for token in LEAKAGE_TOKEN_RE.findall(unicodedata.normalize("NFKC", text))
    ]


def _longest_common_token_run(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the longest contiguous token run using O(min(n, m)) memory."""

    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    longest = 0
    for right_token in right:
        current = [0] * (len(left) + 1)
        for index, left_token in enumerate(left, 1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest


def target_context_leakage_reason(target: str, context: str) -> str | None:
    """Detect copied/near-copied supervision at the final training boundary.

    The corpus constructor applies its own provenance-aware gate.  This independent,
    model-free check protects training even when callers bypass that constructor.
    It intentionally operates on normalized tokens, so punctuation/case edits cannot
    hide a copied target while short conventional academic phrases remain allowed.
    """

    target_tokens = _leakage_tokens(target)
    context_tokens = _leakage_tokens(context)
    if not target_tokens or not context_tokens:
        return None
    target_text = " ".join(target_tokens)
    context_text = " ".join(context_tokens)
    if target_text in context_text:
        return "normalized target is contained verbatim"
    if len(target_tokens) >= 12:
        run = _longest_common_token_run(target_tokens, context_tokens)
        threshold = max(10, math.ceil(len(target_tokens) * 0.55))
        if run >= threshold:
            return f"contiguous copied run is {run} tokens"
    if len(target_tokens) >= 10:
        width = 4
        target_ngrams = Counter(
            tuple(target_tokens[index:index + width])
            for index in range(len(target_tokens) - width + 1)
        )
        context_ngrams = Counter(
            tuple(context_tokens[index:index + width])
            for index in range(len(context_tokens) - width + 1)
        )
        overlap = sum((target_ngrams & context_ngrams).values())
        if overlap / sum(target_ngrams.values()) >= 0.85:
            return "at least 85% of target four-token shingles occur in context"
    return None


def format_academic_prompt(row: Mapping[str, Any]) -> str:
    prompt_input = row["input"]
    claims = "\n".join(f"{number}. {claim.strip()}" for number, claim in enumerate(prompt_input["claims"], 1))
    unit = row["task_type"]
    citation_count = int(prompt_input["citation_count"])
    citation_slots = ", ".join(prompt_input["citation_slots"]) or "none"
    return (
        "Reconstruct one missing unit of human academic prose from the plan below.\n"
        f"Write exactly one {unit}; return only that {unit}. Do not describe the task.\n"
        "Preserve the requested epistemic restraint and do not invent facts or citations.\n\n"
        f"Previous context:\n{prompt_input['context'].strip()}\n\n"
        f"Claims to express:\n{claims}\n\n"
        f"Logical relation: {prompt_input['rhetorical_relation'].strip()}\n"
        f"Certainty: {prompt_input['certainty'].strip()}\n"
        f"Citation markers required ({citation_count}): {citation_slots}\n"
    )


def format_structure_prompt(row: Mapping[str, Any]) -> str:
    """Render one stage-two example without leaking its canonical target."""

    prompt_input = row["input"]
    if row["task_type"] == STRUCTURE_REPLAY_TASK:
        replay = {
            "task_type": prompt_input["unit"],
            "input": {
                key: value for key, value in prompt_input.items() if key != "unit"
            },
        }
        return format_academic_prompt(replay)
    return format_structure_task_prompt(str(row["task_type"]), prompt_input)


def _structure_exact_token_gate(
    corpus_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    *, config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the compiler's exact MLX chat-template boundary receipt.

    Stage two deliberately has no partition/truncation semantics.  Preserve the
    compiler receipt in the prepared dataset so the training process can prove
    that its selected model exposes the same tokenizer identity.
    """

    gates = corpus_manifest.get("gates")
    gate = gates.get("exact_training_token_gate") if isinstance(gates, Mapping) else None
    if not isinstance(gate, Mapping):
        raise HarnessError(
            "academic structure corpus manifest has no exact training-token gate")
    training = config.get("training") if isinstance(config, Mapping) else None
    configured_limit = (
        training.get("max_seq_length") if isinstance(training, Mapping) else None
    )
    limit = gate.get("max_sequence_length")
    if (
        gate.get("method") != STRUCTURE_EXACT_TOKEN_GATE_METHOD
        or gate.get("overflow_policy") != STRUCTURE_EXACT_TOKEN_OVERFLOW_POLICY
        or gate.get("derived_rows") != 0
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 1
        or limit != configured_limit
    ):
        raise HarnessError(
            "academic structure corpus exact training-token gate is incompatible")
    tokenizer = gate.get("tokenizer")
    if (
        not isinstance(tokenizer, Mapping)
        or not isinstance(tokenizer.get("identity"), str)
        or not tokenizer["identity"].strip()
    ):
        raise HarnessError(
            "academic structure corpus exact training-token gate has no tokenizer identity")
    try:
        canonical_json(tokenizer)
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "academic structure corpus tokenizer receipt is not stable JSON") from exc

    largest_row = 0
    for index, row in enumerate(records, 1):
        provenance = row.get("provenance")
        total = provenance.get("exact_training_tokens") if isinstance(
            provenance, Mapping) else None
        offset = provenance.get("exact_prompt_offset") if isinstance(
            provenance, Mapping) else None
        completion = provenance.get("exact_completion_tokens") if isinstance(
            provenance, Mapping) else None
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or not isinstance(completion, int)
            or isinstance(completion, bool)
            or not 0 < offset < total <= limit
            or completion != total - offset
        ):
            raise HarnessError(
                f"academic structure corpus record {index} has no valid exact "
                "training-token receipt")
        largest_row = max(largest_row, total)
    largest_accepted = gate.get("largest_accepted_tokens")
    if (
        not isinstance(largest_accepted, int)
        or isinstance(largest_accepted, bool)
        or not largest_row <= largest_accepted <= limit
    ):
        raise HarnessError(
            "academic structure corpus exact training-token maximum is incompatible")
    measured = gate.get("candidates_measured")
    rejected = gate.get("candidates_rejected")
    if (
        not isinstance(measured, int)
        or isinstance(measured, bool)
        or measured < len(records)
        or not isinstance(rejected, int)
        or isinstance(rejected, bool)
        or rejected < 0
        or rejected > measured
    ):
        raise HarnessError(
            "academic structure corpus exact training-token counts are incompatible")
    return dict(gate)


def prepare_mlx_dataset(
    corpus_path: Path, output_dir: Path, *, require_all_strata: bool = True,
    require_trainable: bool = True, config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not corpus_path.is_file():
        raise HarnessError("training preparation requires the combined corpus JSONL file")
    prompt_contract = _prompt_contract(config)
    corpus_schema = _corpus_schema(config)
    structure_mode = prompt_contract == STRUCTURE_PROMPT_CONTRACT
    records = read_corpus_records(corpus_path, config=config)
    validate_corpus_records(
        records, require_all_strata=require_all_strata, config=config)
    corpus_manifest_path = corpus_path.with_name(f"{corpus_path.name}.manifest.json")
    corpus_manifest = _read_json(corpus_manifest_path, "academic corpus manifest")
    expected_manifest_schema = (
        STRUCTURE_CORPUS_MANIFEST_SCHEMA
        if structure_mode else "spiral.academic-corpus-manifest.v1"
    )
    if corpus_manifest.get("schema_version", corpus_manifest.get("schema")) != expected_manifest_schema:
        raise HarnessError("academic corpus manifest has the wrong schema")
    if structure_mode:
        if corpus_manifest.get("corpus_schema_version") != corpus_schema:
            raise HarnessError(
                "academic structure corpus manifest has the wrong corpus schema")
        if corpus_manifest.get("prompt_contract") != prompt_contract:
            raise HarnessError(
                "academic structure corpus manifest has the wrong prompt contract")
    if set(corpus_manifest.get("source_strata") or []) != REQUIRED_STRATA:
        raise HarnessError("academic corpus manifest does not attest the exact three source strata")
    if require_trainable and corpus_manifest.get("trainable") is not True:
        raise HarnessError("academic corpus manifest is not marked trainable")
    raw_corpus_sha256 = sha256_file(corpus_path)
    if corpus_manifest.get("corpus_sha256") != raw_corpus_sha256:
        raise HarnessError("academic corpus bytes do not match the corpus manifest")
    structure_exact_token_gate = (
        _structure_exact_token_gate(corpus_manifest, records, config=config)
        if structure_mode else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_names = {"train": "train", "validation": "valid", "test": "test"}
    by_split: dict[str, list[bytes]] = {name: [] for name in split_names.values()}
    split_strata: dict[str, dict[str, int]] = {name: {} for name in split_names.values()}
    split_task_types: dict[str, dict[str, int]] = {name: {} for name in split_names.values()}
    canonical_source = hashlib.sha256()
    for row in sorted(records, key=lambda item: (str(item["split"]), str(item["example_id"]))):
        canonical_source.update(canonical_json(row))
        split = split_names[str(row["split"])]
        completion = (
            row["target"].strip()
            if row["task_type"] == STRUCTURE_REPLAY_TASK
            else canonical_json(row["target"]).decode("utf-8").rstrip("\n")
        ) if structure_mode else row["target"].strip()
        mlx_row = {
            "prompt": (
                format_structure_prompt(row)
                if structure_mode else format_academic_prompt(row)
            ),
            "completion": completion,
            "example_id": row["example_id"],
            "source_stratum": row["source"]["stratum"],
            "task_type": row["task_type"],
        }
        by_split[split].append(canonical_json(mlx_row))
        stratum = str(row["source"]["stratum"])
        task_type = str(row["task_type"])
        split_strata[split][stratum] = split_strata[split].get(stratum, 0) + 1
        split_task_types[split][task_type] = split_task_types[split].get(task_type, 0) + 1

    manifest_splits: dict[str, Any] = {}
    for split, lines in by_split.items():
        payload = b"".join(lines)
        destination = output_dir / f"{split}.jsonl"
        atomic_write_bytes(destination, payload)
        manifest_splits[split] = {
            "path": destination.name,
            "count": len(lines),
            "sha256": sha256_bytes(payload),
            "source_strata": dict(sorted(split_strata[split].items())),
            "task_types": dict(sorted(split_task_types[split].items())),
        }
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "prompt_contract": prompt_contract,
        "source_corpus": str(corpus_path.resolve()),
        "source_corpus_sha256": canonical_source.hexdigest(),
        "source_corpus_file_sha256": raw_corpus_sha256,
        "source_corpus_manifest": str(corpus_manifest_path.resolve()),
        "source_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "format": "mlx_lm.completions",
        "completion_only_loss": True,
        "target_context_leakage_gate": {
            "version": "normalized-token-containment-common-run-shingle-v1",
            "exact_normalized_containment": "reject",
            "minimum_common_run_tokens": 10,
            "common_run_target_fraction": 0.55,
            "four_token_shingle_target_fraction": 0.85,
        },
        "splits": manifest_splits,
    }
    if structure_mode:
        manifest["target_context_leakage_gate"].update({
            "scope": "prose_replay_only",
            "structure_target_overlap": (
                "allowed where required by observed-structure task semantics"),
        })
        manifest.update({
            "profile_id": STRUCTURE_PROFILE_ID,
            "corpus_schema_version": STRUCTURE_CORPUS_SCHEMA,
            "source_corpus_manifest_schema": STRUCTURE_CORPUS_MANIFEST_SCHEMA,
            "target_contract": {
                "structure_tasks": sorted(STRUCTURE_TASK_TYPES),
                "structure_target": "canonical_json_object",
                "prose_replay_task": STRUCTURE_REPLAY_TASK,
                "prose_replay_target": "nonempty_academic_prose",
            },
            "source_exact_training_token_gate": structure_exact_token_gate,
        })
    atomic_write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def load_dataset_manifest(
    data_dir: Path, *, config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = data_dir / "dataset_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read prepared dataset manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != DATASET_SCHEMA or manifest.get("completion_only_loss") is not True:
        raise HarnessError("prepared dataset manifest does not guarantee completion-only data")
    prompt_contract = _prompt_contract(config)
    if manifest.get("prompt_contract") != prompt_contract:
        raise HarnessError("prepared dataset manifest has the wrong prompt contract")
    if prompt_contract == STRUCTURE_PROMPT_CONTRACT:
        if (
            manifest.get("profile_id") != STRUCTURE_PROFILE_ID
            or manifest.get("corpus_schema_version") != STRUCTURE_CORPUS_SCHEMA
            or manifest.get("source_corpus_manifest_schema")
            != STRUCTURE_CORPUS_MANIFEST_SCHEMA
        ):
            raise HarnessError(
                "prepared structure dataset manifest has an incompatible contract")
        target_contract = manifest.get("target_contract")
        if not isinstance(target_contract, Mapping) or (
            target_contract.get("structure_tasks") != sorted(STRUCTURE_TASK_TYPES)
            or target_contract.get("structure_target") != "canonical_json_object"
            or target_contract.get("prose_replay_task") != STRUCTURE_REPLAY_TASK
            or target_contract.get("prose_replay_target")
            != "nonempty_academic_prose"
        ):
            raise HarnessError(
                "prepared structure dataset has no exact target contract")
        source_gate = manifest.get("source_exact_training_token_gate")
        source_tokenizer = (
            source_gate.get("tokenizer") if isinstance(source_gate, Mapping) else None
        )
        configured_limit = config.get("training", {}).get("max_seq_length") if isinstance(
            config, Mapping) and isinstance(config.get("training"), Mapping) else None
        if (
            not isinstance(source_gate, Mapping)
            or source_gate.get("method") != STRUCTURE_EXACT_TOKEN_GATE_METHOD
            or source_gate.get("overflow_policy")
            != STRUCTURE_EXACT_TOKEN_OVERFLOW_POLICY
            or source_gate.get("derived_rows") != 0
            or source_gate.get("max_sequence_length") != configured_limit
            or not isinstance(source_tokenizer, Mapping)
            or not isinstance(source_tokenizer.get("identity"), str)
            or not source_tokenizer["identity"].strip()
        ):
            raise HarnessError(
                "prepared structure dataset has no compatible exact training-token gate")
    for split in ("train", "valid", "test"):
        entry = manifest.get("splits", {}).get(split, {})
        file_path = data_dir / str(entry.get("path", ""))
        if not file_path.is_file():
            raise HarnessError(f"prepared {split} split is missing")
        if sha256_file(file_path) != entry.get("sha256"):
            raise HarnessError(f"prepared {split} split hash does not match its manifest")
        if not isinstance(entry.get("count"), int) or entry["count"] <= 0:
            raise HarnessError(f"prepared {split} split is empty")
    return manifest


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} {path} must contain an object")
    return value


def detect_local_revision(model_path: Path) -> str | None:
    for part in reversed(model_path.resolve().parts):
        if REVISION_RE.fullmatch(part):
            return part
    for name in (".spiral-base-revision.json", "base_revision.json"):
        receipt = model_path / name
        if receipt.is_file():
            revision = str(_read_json(receipt, "base revision receipt").get("revision", ""))
            if REVISION_RE.fullmatch(revision):
                return revision
    # ``huggingface_hub download --local-dir`` does not create a snapshots/<sha>
    # component. It does, however, record the immutable commit as the first line
    # of every download metadata file. Require the config and weight-index
    # attestations to agree; the exact file hashes are checked separately.
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    metadata_paths = [
        metadata_root / "config.json.metadata",
        metadata_root / "model.safetensors.index.json.metadata",
    ]
    if all(path.is_file() for path in metadata_paths):
        revisions: set[str] = set()
        try:
            for path in metadata_paths:
                revision = path.read_text(encoding="utf-8").splitlines()[0].strip()
                if not REVISION_RE.fullmatch(revision):
                    return None
                revisions.add(revision)
        except (OSError, IndexError, UnicodeError):
            return None
        if len(revisions) == 1:
            return revisions.pop()
    return None


def safetensors_header(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise HarnessError(f"truncated safetensors file: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 1 or header_length > min(size - 8, 128 * 1024 * 1024):
                raise HarnessError(f"invalid safetensors header length in {path}")
            header_raw = handle.read(header_length)
        header = json.loads(header_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        raise HarnessError(f"invalid safetensors file {path}: {exc}") from exc
    maximum_end = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, Mapping):
            raise HarnessError(f"invalid tensor entry {name!r} in {path}")
        offsets = spec.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) for value in offsets):
            raise HarnessError(f"invalid tensor offsets for {name!r} in {path}")
        start, end = offsets
        if start < 0 or end < start:
            raise HarnessError(f"invalid tensor range for {name!r} in {path}")
        maximum_end = max(maximum_end, end)
    if 8 + header_length + maximum_end != size:
        raise HarnessError(f"safetensors payload length mismatch in {path}")
    return header


def model_weight_inventory(model_path: Path) -> tuple[set[str], list[Path]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise HarnessError("model.safetensors.index.json has no weight_map")
        keys = {str(key) for key in weight_map}
        shard_names = sorted({str(value) for value in weight_map.values()})
        shards = [model_path / name for name in shard_names]
    else:
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            raise HarnessError(f"no safetensors weights found in {model_path}")
        keys = set()
        for shard in shards:
            keys.update(key for key in safetensors_header(shard) if key != "__metadata__")
    missing = [path.name for path in shards if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise HarnessError(f"model is missing weight shard(s): {', '.join(missing)}")
    return keys, shards


def selected_target_inventory(
    model_config: Mapping[str, Any], target_keys: Sequence[str], num_layers: int,
    *, weight_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    text_config = model_config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise HarnessError("Qwen3.5 wrapper is missing text_config")
    layer_count = text_config.get("num_hidden_layers")
    layer_types = text_config.get("layer_types")
    if layer_count != EXPECTED_LAYER_COUNT or not isinstance(layer_types, list) or len(layer_types) != layer_count:
        raise HarnessError(
            f"expected the {EXPECTED_LAYER_COUNT}-layer Qwen3.5 text stack; got {layer_count!r} layers"
        )
    if num_layers <= 0 or num_layers > layer_count:
        raise HarnessError(f"num_layers must be between 1 and {layer_count}")
    selected = list(range(layer_count - num_layers, layer_count))
    per_key = {key: 0 for key in target_keys}
    per_layer: dict[str, list[str]] = {}
    weight_key_set = set(weight_keys) if weight_keys is not None else None
    for layer_index in selected:
        kind = layer_types[layer_index]
        available_prefix = "linear_attn." if kind == "linear_attention" else "self_attn."
        matched: list[str] = []
        for target in target_keys:
            structurally_available = target.startswith("mlp.") or target.startswith(available_prefix)
            if not structurally_available:
                continue
            if weight_key_set is not None:
                suffix = f"layers.{layer_index}.{target}.weight"
                if not any(key.endswith(suffix) for key in weight_key_set):
                    continue
            matched.append(target)
            per_key[target] += 1
        if not matched:
            raise HarnessError(
                f"selected layer {layer_index} ({kind}) has no verified LoRA target; "
                "configure both self_attn and linear_attn target families"
            )
        per_layer[str(layer_index)] = matched
    return {
        "selected_layers": selected,
        "layer_type_counts": {
            "full_attention": sum(layer_types[index] == "full_attention" for index in selected),
            "linear_attention": sum(layer_types[index] == "linear_attention" for index in selected),
        },
        "target_path_counts": {key: value for key, value in per_key.items() if value},
        "targets_per_layer": per_layer,
        "total_target_modules": sum(per_key.values()),
    }


def validate_local_base_model(model_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    base = config["base_model"]
    if not model_path.is_dir():
        raise HarnessError(
            f"local base checkpoint is missing: {model_path}. Download the pinned revision separately; "
            "the harness will not download or load 27B weights during preflight."
        )
    model_config_path = model_path / "config.json"
    model_config = _read_json(model_config_path, "model config")
    errors: list[str] = []
    if model_config.get("model_type") != base["model_type"]:
        errors.append(f"model_type is {model_config.get('model_type')!r}, expected {base['model_type']!r}")
    architectures = model_config.get("architectures")
    if not isinstance(architectures, list) or base["architecture"] not in architectures:
        errors.append(f"architectures does not contain {base['architecture']!r}")
    quant = model_config.get("quantization") or model_config.get("quantization_config") or {}
    if quant.get("bits") != base["quantization"]["bits"] or quant.get("group_size") != base["quantization"]["group_size"]:
        errors.append("checkpoint quantization does not match the configured 4-bit/group-64 base")
    revision = detect_local_revision(model_path)
    if revision != base["revision"]:
        errors.append(
            f"local revision is {revision or 'unattested'}, expected immutable {base['revision']}; "
            "use a Hugging Face snapshots/<sha> directory or a base revision receipt"
        )
    if errors:
        raise HarnessError("incompatible local base checkpoint:\n- " + "\n- ".join(errors))
    weight_keys, shards = model_weight_inventory(model_path)
    total_bytes = sum(path.stat().st_size for path in shards)
    minimum_bytes = int(config.get("resources", {}).get("minimum_model_bytes", 14_000_000_000))
    if total_bytes < minimum_bytes:
        raise HarnessError(
            f"checkpoint weight shards total only {total_bytes:,} bytes; expected at least {minimum_bytes:,}"
        )
    targets = selected_target_inventory(
        model_config,
        list(config["lora"]["keys"]),
        int(config["training"]["num_layers"]),
        weight_keys=weight_keys,
    )
    weight_files: list[dict[str, Any]] = []
    inventory_digest = hashlib.sha256()
    for shard in sorted(shards, key=lambda value: value.name):
        shard_digest = sha256_file(shard)
        shard_size = shard.stat().st_size
        inventory_digest.update(
            f"{shard.name}\0{shard_size}\0{shard_digest}\n".encode("utf-8"))
        weight_files.append({
            "path": shard.name,
            "size_bytes": shard_size,
            "sha256": shard_digest,
        })
    index_path = model_path / "model.safetensors.index.json"
    config_digest = sha256_file(model_config_path)
    index_digest = sha256_file(index_path) if index_path.is_file() else None
    inventory_digest_hex = inventory_digest.hexdigest()
    pinned_weight_files = sorted(
        (dict(row) for row in base.get("weight_files", [])),
        key=lambda row: str(row.get("path", "")),
    )
    observed_weight_files = sorted(weight_files, key=lambda row: row["path"])
    identity_errors: list[str] = []
    if config_digest != base.get("config_sha256"):
        identity_errors.append("config.json SHA-256 differs from the pinned Qwen3.8 base")
    if index_digest != base.get("weight_index_sha256"):
        identity_errors.append("weight index SHA-256 differs from the pinned Qwen3.8 base")
    if total_bytes != base.get("weight_bytes"):
        identity_errors.append("total weight bytes differ from the pinned Qwen3.8 base")
    if observed_weight_files != pinned_weight_files:
        identity_errors.append("weight shard inventory differs from the pinned Qwen3.8 base")
    if inventory_digest_hex != base.get("weight_inventory_sha256"):
        identity_errors.append("weight inventory SHA-256 differs from the pinned Qwen3.8 base")
    if identity_errors:
        raise HarnessError(
            "incompatible local base checkpoint:\n- " + "\n- ".join(identity_errors))
    return {
        "model_id": base["model_id"],
        "revision": revision,
        "model_type": model_config["model_type"],
        "architecture": base["architecture"],
        "config_sha256": config_digest,
        "weight_index_sha256": index_digest,
        "weight_shards": len(shards),
        "weight_bytes": total_bytes,
        "weight_files": weight_files,
        "weight_inventory_sha256": inventory_digest_hex,
        "quantization": dict(base["quantization"]),
        "target_inventory": targets,
    }


_MEMORY_PRESSURE_PERCENT_RE = re.compile(
    r"^System-wide memory free percentage:\s*([0-9]{1,3})%\s*$",
    re.MULTILINE,
)


def _parse_memory_pressure_percentage(output: str) -> int:
    """Parse the stable, machine-readable fact emitted by memory_pressure -Q."""
    matches = _MEMORY_PRESSURE_PERCENT_RE.findall(output)
    if len(matches) != 1:
        raise ValueError("expected exactly one system-wide memory free percentage")
    percentage = int(matches[0])
    if not 0 <= percentage <= 100:
        raise ValueError("system-wide memory free percentage is outside 0..100")
    return percentage


def _vm_stat_available_bytes(output: str) -> int:
    page_match = re.search(r"page size of (\d+) bytes", output)
    if not page_match:
        raise ValueError("vm_stat output does not declare its page size")
    page_size = int(page_match.group(1))
    if page_size <= 0:
        raise ValueError("vm_stat page size must be positive")
    counters: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\.", line)
        if match:
            counters[match.group(1)] = int(match.group(2))
    available_names = (
        "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    if not any(name in counters for name in available_names):
        raise ValueError("vm_stat output does not contain available-page counters")
    return sum(counters.get(name, 0) for name in available_names) * page_size


def system_resources(path: Path) -> dict[str, Any]:
    total = 0
    available = 0
    available_source: str | None = None
    available_percentage: float | None = None
    available_probe_error: str | None = None
    if platform.system() == "Darwin":
        try:
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            pressure_output = subprocess.check_output(
                ["/usr/bin/memory_pressure", "-Q"], text=True)
        except (OSError, subprocess.SubprocessError):
            pressure_output = None
        if pressure_output is not None:
            try:
                percentage = _parse_memory_pressure_percentage(pressure_output)
            except (TypeError, ValueError):
                # A successful command with an unknown output contract must not
                # silently fall through to a weaker estimate after output drift.
                available_probe_error = "malformed /usr/bin/memory_pressure -Q output"
            else:
                available = total * percentage // 100
                available_source = "memory_pressure"
                available_percentage = float(percentage)
        else:
            try:
                available = _vm_stat_available_bytes(
                    subprocess.check_output(["vm_stat"], text=True))
            except (OSError, subprocess.SubprocessError, TypeError, ValueError):
                available_probe_error = "memory_pressure unavailable and vm_stat output unusable"
            else:
                available_source = "vm_stat"
                if total > 0:
                    available_percentage = available * 100.0 / total
    elif Path("/proc/meminfo").is_file():
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
    disk = shutil.disk_usage(path if path.exists() else path.parent)
    facts = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "disk_free_bytes": disk.free,
    }
    if platform.system() == "Darwin":
        facts.update({
            "memory_available_source": available_source or "unavailable",
            "memory_available_percentage": available_percentage,
        })
        if available_probe_error:
            facts["memory_available_probe_error"] = available_probe_error
    return facts


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def python_package_versions(python_executable: str) -> dict[str, str | None]:
    """Query the exact interpreter selected for MLX without importing its models."""
    probe = (
        "import importlib.metadata,json;"
        "names=('mlx','mlx-lm');"
        "out={};"
        "exec(\"for n in names:\\n try: out[n.replace('-', '_')]=importlib.metadata.version(n)"
        "\\n except importlib.metadata.PackageNotFoundError: out[n.replace('-', '_')]=None\");"
        "print(json.dumps(out,sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", probe], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(f"cannot run selected MLX Python {python_executable!r}: {exc}") from exc
    if completed.returncode:
        raise HarnessError(
            f"selected MLX Python failed its package probe: {completed.stderr.strip()[:300]}")
    try:
        versions = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError("selected MLX Python returned an invalid package probe") from exc
    if not isinstance(versions, dict):
        raise HarnessError("selected MLX Python returned an invalid package map")
    return {"mlx": versions.get("mlx"), "mlx_lm": versions.get("mlx_lm")}


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> None:
        if self.errors:
            raise HarnessError("academic QLoRA preflight failed:\n- " + "\n- ".join(self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "facts": self.facts,
        }


def run_preflight(
    config: Mapping[str, Any], model_path: Path, data_dir: Path, output_dir: Path,
    *, resource_facts: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str | None] | None = None,
) -> PreflightReport:
    report = PreflightReport()
    try:
        model_receipt = validate_local_base_model(model_path, config)
        report.facts["base_model"] = model_receipt
    except HarnessError as exc:
        report.errors.append(str(exc))
    try:
        dataset = load_dataset_manifest(data_dir, config=config)
        report.facts["dataset"] = dataset
    except HarnessError as exc:
        report.errors.append(str(exc))

    resources = dict(resource_facts or system_resources(output_dir))
    report.facts["resources"] = resources
    required = config.get("resources", {})
    gib = 1024 ** 3
    total_required = float(required.get("minimum_total_memory_gib", 32)) * gib
    available_required = float(required.get("minimum_available_memory_gib", 24)) * gib
    disk_required = float(required.get("minimum_free_disk_gib", 32)) * gib
    if resources.get("platform") != "Darwin" or resources.get("machine") != "arm64":
        report.errors.append("MLX QLoRA requires an Apple-silicon (Darwin/arm64) host")
    if int(resources.get("memory_total_bytes", 0)) < total_required:
        report.errors.append(
            f"physical unified memory is below the strict {total_required / gib:.1f} GiB floor"
        )
    if int(resources.get("memory_available_bytes", 0)) < available_required:
        report.errors.append(
            f"available unified memory is below {available_required / gib:.1f} GiB; stop other inference first"
        )
    if int(resources.get("disk_free_bytes", 0)) < disk_required:
        report.errors.append(f"free disk is below the strict {disk_required / gib:.1f} GiB floor")

    versions = dict(package_versions or {
        "mlx": installed_version("mlx"),
        "mlx_lm": installed_version("mlx-lm"),
    })
    report.facts["packages"] = versions
    version_limits = config.get("compatibility", {})
    for package, config_key in (("mlx", "required_mlx"), ("mlx_lm", "required_mlx_lm")):
        version = versions.get(package)
        required_version = str(version_limits.get(config_key, ""))
        if not version:
            report.errors.append(f"{package} is not installed; install the pinned MLX training environment")
        elif not required_version:
            report.errors.append(f"training config does not pin an exact {package} version")
        elif version != required_version:
            report.errors.append(
                f"{package} {version} does not match the required reproducible version {required_version}")
    return report


MODEL_VIEW_SOURCE = '''\
"""Spiral-audited Qwen3.5 text-only tuner seam."""
from mlx_lm.models.qwen3_5 import Model as _Qwen35Model, ModelArgs


class Model(_Qwen35Model):
    @property
    def layers(self):
        # MLX-LM's tuner requires model.layers; the upstream multimodal wrapper
        # exposes them one level lower even though inference here is text-only.
        return self.language_model.layers
'''


def create_text_training_view(model_path: Path, output_root: Path, base_receipt: Mapping[str, Any]) -> Path:
    view_identity = sha256_bytes(canonical_json({
        "revision": base_receipt["revision"],
        "config_sha256": base_receipt["config_sha256"],
        "weight_inventory_sha256": base_receipt["weight_inventory_sha256"],
        "source_sha256": sha256_bytes(MODEL_VIEW_SOURCE.encode("utf-8")),
    }))
    destination = output_root / ".model-views" / view_identity[:16]
    receipt_path = destination / "spiral_model_view_receipt.json"
    if destination.is_dir():
        receipt = _read_json(receipt_path, "model view receipt")
        if receipt.get("view_identity") != view_identity:
            raise HarnessError(f"existing model view has the wrong identity: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for source in model_path.iterdir():
            if source.name in {"config.json", ".spiral-base-revision.json", "base_revision.json"}:
                continue
            os.symlink(source.resolve(), staging / source.name, target_is_directory=source.is_dir())
        config = _read_json(model_path / "config.json", "model config")
        config["model_file"] = "spiral_qwen35_text_tuner.py"
        atomic_write_json(staging / "config.json", config)
        atomic_write_bytes(staging / "spiral_qwen35_text_tuner.py", MODEL_VIEW_SOURCE.encode("utf-8"))
        receipt = {
            "schema_version": "spiral.qwen35-text-training-view.v1",
            "view_identity": view_identity,
            "base_model": dict(base_receipt),
            "custom_module_sha256": sha256_bytes(MODEL_VIEW_SOURCE.encode("utf-8")),
            "purpose": "Expose the verified 64-layer text stack to MLX-LM LoRA without vision input.",
        }
        atomic_write_json(staging / receipt_path.name, receipt)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def yaml_training_config(
    config: Mapping[str, Any], model_view: Path, data_dir: Path, adapter_work_dir: Path,
    *, iterations: int | None = None, resume_adapter: Path | None = None,
) -> str:
    training = config["training"]
    lora = config["lora"]
    lines = [
        f"model: {json.dumps(str(model_view.resolve()))}",
        "train: true",
        f"data: {json.dumps(str(data_dir.resolve()))}",
        "fine_tune_type: lora",
        "optimizer: adamw",
        "mask_prompt: true",
        f"batch_size: {training['batch_size']}",
        f"grad_accumulation_steps: {training['grad_accumulation_steps']}",
        f"grad_checkpoint: {str(training['grad_checkpoint']).lower()}",
        f"num_layers: {training['num_layers']}",
        f"iters: {iterations if iterations is not None else training['iterations']}",
        f"learning_rate: {training['learning_rate']}",
        f"max_seq_length: {training['max_seq_length']}",
        f"val_batches: {training['val_batches']}",
        f"steps_per_report: {training['steps_per_report']}",
        f"steps_per_eval: {training['steps_per_eval']}",
        f"save_every: {training['save_every']}",
        f"seed: {training['seed']}",
        f"clear_cache_threshold: {json.dumps(str(training.get('clear_cache_threshold', '1GB')))}",
        f"adapter_path: {json.dumps(str(adapter_work_dir.resolve()))}",
        "trust_remote_code: true",
        "lora_parameters:",
        "  keys:",
    ]
    lines.extend(f"    - {json.dumps(key)}" for key in lora["keys"])
    lines.extend([
        f"  rank: {lora['rank']}",
        f"  scale: {lora['scale']}",
        f"  dropout: {lora['dropout']}",
    ])
    if resume_adapter is not None:
        lines.append(f"resume_adapter_file: {json.dumps(str(resume_adapter.resolve()))}")
    return "\n".join(lines) + "\n"


_OPERATIONAL_TRAINING_FIELDS = frozenset({
    "val_batches",
    "steps_per_report",
    "steps_per_eval",
    "save_every",
    "clear_cache_threshold",
})


def build_training_run_contract(
    config: Mapping[str, Any], base_receipt: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *, training_data_receipt: Mapping[str, Any] | None = None,
    parent_adapter: ParentAdapterInitialization | None = None,
) -> dict[str, Any]:
    """Build the stable semantic identity shared by every resume attempt.

    Evaluation/report/checkpoint cadence is deliberately recorded outside this
    identity: changing when we observe or persist identical optimizer updates must
    not make an otherwise compatible checkpoint unusable.  Parameters that affect
    those updates (including layer count, sequence length, seed, LR, and LoRA
    topology) remain identity-bound.
    """
    training = config["training"]
    semantic_training = {
        key: value for key, value in training.items()
        if key not in _OPERATIONAL_TRAINING_FIELDS
    }
    split_identity = {
        name: {
            "count": entry["count"],
            "sha256": entry["sha256"],
        }
        for name, entry in sorted(dataset_manifest["splits"].items())
    }
    identity_contract = {
        "schema_version": TRAINING_RUN_SCHEMA,
        "profile": dict(config["profile"]),
        "base_model": {
            "model_id": base_receipt["model_id"],
            "revision": base_receipt["revision"],
            "config_sha256": base_receipt["config_sha256"],
            "weight_index_sha256": base_receipt.get("weight_index_sha256"),
            "weight_inventory_sha256": base_receipt["weight_inventory_sha256"],
            "quantization": dict(base_receipt["quantization"]),
        },
        "training": semantic_training,
        "lora": dict(config["lora"]),
        "compatibility": dict(config["compatibility"]),
        "dataset": {
            "schema_version": dataset_manifest["schema_version"],
            "prompt_contract": dataset_manifest["prompt_contract"],
            "completion_only_loss": dataset_manifest["completion_only_loss"],
            "source_corpus_sha256": dataset_manifest["source_corpus_sha256"],
            "source_corpus_file_sha256": dataset_manifest["source_corpus_file_sha256"],
            "source_corpus_manifest_sha256": dataset_manifest[
                "source_corpus_manifest_sha256"],
            "splits": split_identity,
        },
    }
    if dataset_manifest.get("prompt_contract") == STRUCTURE_PROMPT_CONTRACT:
        identity_contract["dataset"].update({
            "profile_id": dataset_manifest.get("profile_id"),
            "corpus_schema_version": dataset_manifest.get("corpus_schema_version"),
            "source_corpus_manifest_schema": dataset_manifest.get(
                "source_corpus_manifest_schema"),
            "target_contract": dataset_manifest.get("target_contract"),
        })
    if training_data_receipt is not None:
        identity_contract["bounded_training_data"] = _bounded_training_data_attestation(
            training_data_receipt)
    if parent_adapter is not None:
        identity_contract["parent_adapter"] = dict(parent_adapter.identity)
    return {
        "schema_version": TRAINING_RUN_SCHEMA,
        "run_identity": sha256_bytes(canonical_json(identity_contract)),
        "identity_contract": identity_contract,
        "resume_semantics": {
            "state": "adapter_weights_only",
            "optimizer_moments_restored": False,
            "rng_state_restored": False,
            "bit_exact": False,
            "description": (
                "MLX-LM reloads adapter weights; optimizer moments and RNG stream "
                "restart for each attempt, so resumed optimization is approximate."
            ),
        },
    }


def _bounded_training_data_attestation(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the small, path-independent subset that binds a safe train view."""

    identity_contract = receipt.get("identity_contract")
    output = receipt.get("output")
    gate = receipt.get("gate")
    preservation = receipt.get("preservation")
    if not all(
        isinstance(value, Mapping)
        for value in (identity_contract, output, gate, preservation)
    ):
        raise HarnessError("bounded training-data receipt is incomplete")
    view_identity = receipt.get("view_identity")
    train_sha256 = output.get("train_sha256")
    mapping_sha256 = preservation.get("mapping_sha256")
    if not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value)
        for value in (view_identity, train_sha256, mapping_sha256)
    ):
        raise HarnessError("bounded training-data receipt has invalid content hashes")
    if preservation.get("all_source_completions_preserved") is not True:
        raise HarnessError("bounded training-data receipt does not preserve all completions")
    maximum = gate.get("maximum_total_tokens")
    limit = gate.get("max_sequence_length")
    count = output.get("train_count")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or maximum > limit
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
    ):
        raise HarnessError("bounded training-data token gate is invalid")
    return {
        "schema_version": receipt.get("schema_version"),
        "view_identity": view_identity,
        "identity_contract": dict(identity_contract),
        "train_sha256": train_sha256,
        "train_count": count,
        "max_sequence_length": limit,
        "maximum_total_tokens": maximum,
        "mapping_sha256": mapping_sha256,
        "all_source_completions_preserved": True,
        "partitioned_source_rows": output.get("partitioned_source_rows"),
        "derived_rows": output.get("derived_rows"),
    }


def ensure_training_run_contract(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Create one immutable run contract or reject reuse by another run."""
    expected = dict(contract)
    if path.exists():
        existing = _read_json(path, "training run contract")
        if existing != expected:
            raise HarnessError(
                "output directory belongs to a different training run identity; "
                "use the original config/model/dataset or a fresh --output")
        return existing
    atomic_write_json(path, expected)
    return expected


def create_training_only_dataset_view(
    data_dir: Path, output_root: Path, dataset_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Expose only the attested train split so MLX-LM cannot validate in-process.

    The pinned MLX-LM trainer unconditionally validates before iteration 1 and
    before its final update whenever ``valid.jsonl`` is present.  This view removes
    those extra forwards from the training process; it does not claim to reduce the
    memory required by any individual train-row backward pass.  The original
    held-out files remain intact for evaluation after the model has unloaded.
    """
    train_entry = dataset_manifest.get("splits", {}).get("train", {})
    source = data_dir / str(train_entry.get("path", ""))
    expected_sha256 = str(train_entry.get("sha256", ""))
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise HarnessError("cannot create safe trainer data view: train split hash mismatch")
    view_identity = sha256_bytes(canonical_json({
        "schema_version": TRAINING_DATA_VIEW_SCHEMA,
        "train_sha256": expected_sha256,
        "train_count": train_entry.get("count"),
        "validation_policy": "separate_process_after_training",
    }))
    parent = output_root / ".work" / "trainer-data"
    destination = parent / view_identity[:16]
    receipt = {
        "schema_version": TRAINING_DATA_VIEW_SCHEMA,
        "view_identity": view_identity,
        "train_source": str(source.resolve()),
        "train_sha256": expected_sha256,
        "train_count": train_entry.get("count"),
        "exposed_files": ["train.jsonl"],
        "omitted_files": ["valid.jsonl", "test.jsonl"],
        "validation_policy": "separate_process_after_training",
        "reason": (
            "Avoid MLX-LM's forced in-process validation before iteration 1 and "
            "the final update; this does not lower per-train-row backward memory, "
            "and held-out data remains in the prepared dataset."
        ),
    }
    if destination.exists():
        if not destination.is_dir():
            raise HarnessError(f"safe trainer data view is not a directory: {destination}")
        existing = _read_json(destination / "view.json", "safe trainer data view receipt")
        if existing != receipt:
            raise HarnessError(f"safe trainer data view has the wrong identity: {destination}")
        train_view = destination / "train.jsonl"
        if (not train_view.is_file() or sha256_file(train_view) != expected_sha256
                or (destination / "valid.jsonl").exists()
                or (destination / "test.jsonl").exists()):
            raise HarnessError(f"safe trainer data view failed integrity checks: {destination}")
        return destination, receipt

    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        os.symlink(source.resolve(), staging / "train.jsonl")
        atomic_write_json(staging / "view.json", receipt)
        os.replace(staging, destination)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination, receipt


def mlx_training_command(yaml_path: Path, *, python_executable: str | None = None) -> list[str]:
    return [python_executable or os.sys.executable, "-m", "mlx_lm", "lora", "--config", str(yaml_path)]


@dataclass(frozen=True)
class CapturedCheckpoint:
    step: int
    path: Path
    sha256: str
    size_bytes: int
    config_path: Path
    config_sha256: str
    config_size_bytes: int
    bundle_sha256: str
    receipt_sha256: str
    run_identity: str | None


@dataclass(frozen=True)
class TrainingRetryDecision:
    """A bounded recovery decision for one exited MLX trainer child."""

    retry: bool
    recovery_mode: str
    delay_seconds: float | None
    reason: str
    signal_name: str | None


def plan_training_retry(
    exit_status: int, *, retries_used: int, max_retries: int,
    base_delay_seconds: float, maximum_delay_seconds: float,
    checkpoint: CapturedCheckpoint | None,
) -> TrainingRetryDecision:
    """Retry signal-terminated children, never ordinary deterministic exits.

    macOS reports an MLX/Metal ``SIGABRT`` as ``-6`` through ``subprocess``.
    Other negative values likewise mean the child was terminated by a signal and
    are safe to retry within the explicit finite budget.  Positive exits are
    treated as deterministic trainer/configuration failures and fail closed.
    """

    if not isinstance(exit_status, int) or isinstance(exit_status, bool):
        raise HarnessError("trainer exit status must be an integer")
    if (
        not isinstance(retries_used, int) or isinstance(retries_used, bool)
        or retries_used < 0
        or not isinstance(max_retries, int) or isinstance(max_retries, bool)
        or max_retries < 0
    ):
        raise HarnessError("training retry counts must be non-negative integers")
    if max_retries > 20:
        raise HarnessError("maximum training retries must not exceed 20")
    for value, label in (
        (base_delay_seconds, "base retry delay"),
        (maximum_delay_seconds, "maximum retry delay"),
    ):
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) < 0
        ):
            raise HarnessError(f"{label} must be a finite non-negative number")
    if float(maximum_delay_seconds) > 60:
        raise HarnessError("maximum retry delay must not exceed 60 seconds")
    if float(base_delay_seconds) > float(maximum_delay_seconds):
        raise HarnessError("base retry delay must not exceed maximum retry delay")

    recovery_mode = "checkpoint_resume" if checkpoint is not None else "fresh_restart"
    if exit_status >= 0:
        return TrainingRetryDecision(
            retry=False,
            recovery_mode=recovery_mode,
            delay_seconds=None,
            reason="non_signal_exit",
            signal_name=None,
        )
    signal_number = -exit_status
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"SIGNAL_{signal_number}"
    if retries_used >= max_retries:
        return TrainingRetryDecision(
            retry=False,
            recovery_mode=recovery_mode,
            delay_seconds=None,
            reason="retry_budget_exhausted",
            signal_name=signal_name,
        )
    delay = min(
        float(maximum_delay_seconds),
        float(base_delay_seconds) * (2 ** retries_used),
    )
    return TrainingRetryDecision(
        retry=True,
        recovery_mode=recovery_mode,
        delay_seconds=delay,
        reason="signal_terminated",
        signal_name=signal_name,
    )


class TrainingRunLock:
    """One fail-fast supervisor per immutable output/run directory.

    The descriptor is deliberately inherited by the MLX child.  If the Python
    supervisor itself is killed, the orphan trainer continues to hold this lock
    until it exits, so a replacement supervisor cannot launch a duplicate child.
    """

    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def acquire(self, *, owner: Mapping[str, Any]) -> None:
        if fcntl is None:
            raise HarnessError("the training run lock requires fcntl")
        if self.descriptor is not None:
            raise HarnessError("training run lock is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(self.path, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HarnessError(
                    "this academic training run is already supervised; "
                    "a duplicate trainer was not started"
                ) from exc
            record = {
                "pid": os.getpid(),
                "acquired_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "spiral_academic_training_supervisor",
                **dict(owner),
            }
            raw = json.dumps(
                record, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")[:4096]
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, raw)
            os.fsync(descriptor)
            os.set_inheritable(descriptor, True)
            self.descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "TrainingRunLock":
        if self.descriptor is None:
            raise HarnessError("call acquire() before entering the training run lock")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


class TrainingSupervisorJournal:
    """Durable, identity-locked retry events plus an atomic current status."""

    def __init__(
        self, output: Path, *, run_identity: str, session_id: str,
        max_retries: int, base_delay_seconds: float,
        maximum_delay_seconds: float,
    ) -> None:
        if not SHA256_RE.fullmatch(run_identity):
            raise HarnessError("training supervisor run identity is invalid")
        # Reuse the policy validator without authorizing a retry.
        plan_training_retry(
            0, retries_used=0, max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            maximum_delay_seconds=maximum_delay_seconds,
            checkpoint=None,
        )
        if not isinstance(session_id, str) or not session_id:
            raise HarnessError("training supervisor session id is invalid")
        self.output = output
        self.events_path = output / "training-supervisor-events.jsonl"
        self.status_path = output / "training-supervisor-status.json"
        self.run_identity = run_identity
        self.session_id = session_id
        self.max_retries = max_retries
        self.base_delay_seconds = float(base_delay_seconds)
        self.maximum_delay_seconds = float(maximum_delay_seconds)
        self.sequence = 0
        output.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if self.events_path.exists() and not self.events_path.is_file():
            raise HarnessError("training supervisor event ledger is not a file")
        if self.events_path.is_file():
            for line_number, raw in enumerate(
                self.events_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise HarnessError(
                        "training supervisor event ledger is corrupt at line "
                        f"{line_number}: {exc}"
                    ) from exc
                if (
                    not isinstance(record, Mapping)
                    or record.get("schema_version")
                    != TRAINING_SUPERVISOR_EVENT_SCHEMA
                    or record.get("run_identity") != self.run_identity
                    or record.get("sequence") != line_number
                ):
                    raise HarnessError(
                        "training supervisor event ledger identity/sequence mismatch")
                self.sequence = line_number
        if self.status_path.exists() and not self.status_path.is_file():
            raise HarnessError("training supervisor status is not a file")
        if self.status_path.is_file():
            status = _read_json(
                self.status_path, "training supervisor status")
            if (
                status.get("schema_version")
                != TRAINING_SUPERVISOR_STATUS_SCHEMA
                or status.get("run_identity") != self.run_identity
            ):
                raise HarnessError(
                    "training supervisor status belongs to another run identity")

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        if not isinstance(event, str) or not event:
            raise HarnessError("training supervisor event name is invalid")
        self.sequence += 1
        record = {
            "schema_version": TRAINING_SUPERVISOR_EVENT_SCHEMA,
            "sequence": self.sequence,
            "recorded_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_identity": self.run_identity,
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        descriptor = os.open(
            self.events_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, canonical_json(record))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return record

    def write_status(self, *, state: str, **fields: Any) -> dict[str, Any]:
        status = {
            "schema_version": TRAINING_SUPERVISOR_STATUS_SCHEMA,
            "state": state,
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_identity": self.run_identity,
            "session_id": self.session_id,
            "pid": os.getpid(),
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "maximum_delay_seconds": self.maximum_delay_seconds,
            "last_event_sequence": self.sequence,
            "event_ledger": str(self.events_path),
            **fields,
        }
        atomic_write_json(self.status_path, status)
        return status


class CheckpointLedger:
    """Publish and verify immutable adapter-weight/config checkpoint bundles."""

    def __init__(self, root: Path, *, run_identity: str | None = None):
        self.root = root
        self.run_identity = run_identity
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, source: Path, step: int, *, adapter_config: Path | None = None) -> CapturedCheckpoint:
        safetensors_header(source)
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise HarnessError("checkpoint step must be a positive integer")
        if adapter_config is None or not adapter_config.is_file():
            raise HarnessError("checkpoint adapter_config.json is missing")
        _read_json(adapter_config, "checkpoint adapter config")
        source_before = source.stat()
        config_before = adapter_config.stat()
        destination = self.root / f"step-{step:07d}"
        if destination.exists():
            existing = self._validate_checkpoint(destination)
            if (existing.sha256 != sha256_file(source)
                    or existing.config_sha256 != sha256_file(adapter_config)):
                raise HarnessError(
                    f"existing checkpoint {destination} conflicts with new checkpoint bytes")
            self._write_pointer(existing)
            return existing
        staging = self.root / f".step-{step:07d}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            weights = staging / "adapters.safetensors"
            with source.open("rb") as reader, weights.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            config_copy = staging / "adapter_config.json"
            with adapter_config.open("rb") as reader, config_copy.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            source_after = source.stat()
            config_after = adapter_config.stat()
            if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
                raise HarnessError(f"checkpoint source changed while being captured: {source}")
            if ((config_before.st_size, config_before.st_mtime_ns)
                    != (config_after.st_size, config_after.st_mtime_ns)):
                raise HarnessError(
                    f"checkpoint adapter config changed while being captured: {adapter_config}")
            safetensors_header(weights)
            _read_json(config_copy, "captured adapter config")
            weights_entry = {
                "path": weights.name,
                "sha256": sha256_file(weights),
                "size_bytes": weights.stat().st_size,
            }
            config_entry = {
                "path": config_copy.name,
                "sha256": sha256_file(config_copy),
                "size_bytes": config_copy.stat().st_size,
            }
            bundle_sha256 = _checkpoint_bundle_digest(weights_entry, config_entry)
            metadata = {
                "schema_version": CHECKPOINT_SCHEMA,
                "step": step,
                "run_identity": self.run_identity,
                "weights": weights_entry,
                "adapter_config": config_entry,
                "bundle_sha256": bundle_sha256,
                "resume_semantics": {
                    "state": "adapter_weights_only",
                    "optimizer_moments_restored": False,
                    "rng_state_restored": False,
                    "bit_exact": False,
                },
            }
            atomic_write_json(staging / "checkpoint.json", metadata)
            os.replace(staging, destination)
            _fsync_directory(self.root)
            published = self._validate_checkpoint(destination)
            self._write_pointer(published)
            return published
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def latest(self) -> CapturedCheckpoint | None:
        committed: list[CapturedCheckpoint] = []
        pattern = re.compile(r"^step-(\d{7})$")
        for candidate in sorted(self.root.iterdir()):
            match = pattern.fullmatch(candidate.name)
            if match is None:
                continue
            if not candidate.is_dir():
                raise HarnessError(f"checkpoint entry is not a directory: {candidate}")
            committed.append(self._validate_checkpoint(candidate))

        pointer = self.root / "latest.json"
        if pointer.exists() and not pointer.is_file():
            raise HarnessError("latest checkpoint pointer is not a file")
        pointed: CapturedCheckpoint | None = None
        if pointer.is_file():
            metadata = _read_json(pointer, "checkpoint pointer")
            if metadata.get("schema_version") != CHECKPOINT_POINTER_SCHEMA:
                raise HarnessError("latest checkpoint pointer has an incompatible schema")
            if metadata.get("run_identity") != self.run_identity:
                raise HarnessError("latest checkpoint pointer run identity mismatch")
            path_name = str(metadata.get("path", ""))
            match = pattern.fullmatch(path_name)
            if match is None:
                raise HarnessError("latest checkpoint pointer path is invalid")
            matches = [item for item in committed if item.path.parent.name == path_name]
            if len(matches) != 1:
                raise HarnessError("latest checkpoint pointer references a missing checkpoint")
            pointed = matches[0]
            if (metadata.get("step") != pointed.step
                    or metadata.get("receipt_sha256") != pointed.receipt_sha256):
                raise HarnessError("latest checkpoint pointer receipt mismatch")
        if not committed:
            if pointed is not None:
                raise HarnessError("latest checkpoint pointer has no committed checkpoint")
            return None
        latest = max(committed, key=lambda item: item.step)
        if pointed != latest:
            # A crash can occur after the directory rename but before the pointer
            # replacement. Recover the newest fully verified committed receipt.
            self._write_pointer(latest)
        return latest

    def _validate_checkpoint(self, directory: Path) -> CapturedCheckpoint:
        match = re.fullmatch(r"step-(\d{7})", directory.name)
        if match is None:
            raise HarnessError(f"invalid checkpoint directory name: {directory}")
        metadata_path = directory / "checkpoint.json"
        metadata = _read_json(metadata_path, "checkpoint receipt")
        step = int(match.group(1))
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
            raise HarnessError(f"checkpoint {directory} has an incompatible receipt schema")
        if metadata.get("step") != step:
            raise HarnessError(f"checkpoint {directory} receipt step mismatch")
        if metadata.get("run_identity") != self.run_identity:
            raise HarnessError(f"checkpoint {directory} run identity mismatch")
        weights_entry = metadata.get("weights")
        config_entry = metadata.get("adapter_config")
        if not isinstance(weights_entry, Mapping) or not isinstance(config_entry, Mapping):
            raise HarnessError(f"checkpoint {directory} receipt is incomplete")
        weights = _verify_checkpoint_file(
            directory, weights_entry, expected_name="adapters.safetensors")
        config_path = _verify_checkpoint_file(
            directory, config_entry, expected_name="adapter_config.json")
        safetensors_header(weights)
        _read_json(config_path, "checkpoint adapter config")
        bundle_sha256 = _checkpoint_bundle_digest(weights_entry, config_entry)
        if metadata.get("bundle_sha256") != bundle_sha256:
            raise HarnessError(f"checkpoint {directory} bundle hash mismatch")
        semantics = metadata.get("resume_semantics")
        if (not isinstance(semantics, Mapping)
                or semantics.get("state") != "adapter_weights_only"
                or semantics.get("optimizer_moments_restored") is not False
                or semantics.get("rng_state_restored") is not False
                or semantics.get("bit_exact") is not False):
            raise HarnessError(f"checkpoint {directory} has invalid resume semantics")
        return CapturedCheckpoint(
            step=step,
            path=weights,
            sha256=str(weights_entry["sha256"]),
            size_bytes=int(weights_entry["size_bytes"]),
            config_path=config_path,
            config_sha256=str(config_entry["sha256"]),
            config_size_bytes=int(config_entry["size_bytes"]),
            bundle_sha256=bundle_sha256,
            receipt_sha256=sha256_file(metadata_path),
            run_identity=self.run_identity,
        )

    def _write_pointer(self, checkpoint: CapturedCheckpoint) -> None:
        atomic_write_json(self.root / "latest.json", {
            "schema_version": CHECKPOINT_POINTER_SCHEMA,
            "step": checkpoint.step,
            "path": checkpoint.path.parent.name,
            "receipt_sha256": checkpoint.receipt_sha256,
            "run_identity": self.run_identity,
        })


def _verify_checkpoint_file(
    directory: Path, entry: Mapping[str, Any], *, expected_name: str,
) -> Path:
    if entry.get("path") != expected_name:
        raise HarnessError(f"checkpoint {directory} has an invalid {expected_name} path")
    path = directory / expected_name
    size = entry.get("size_bytes")
    digest = entry.get("sha256")
    if (not path.is_file() or not isinstance(size, int) or isinstance(size, bool)
            or size <= 0 or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
        raise HarnessError(f"checkpoint {directory} has incomplete {expected_name} metadata")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise HarnessError(f"checkpoint {directory} failed {expected_name} integrity checks")
    return path


def _checkpoint_bundle_digest(
    weights_entry: Mapping[str, Any], config_entry: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for entry in sorted((weights_entry, config_entry), key=lambda item: str(item["path"])):
        digest.update(
            f"{entry['path']}\0{entry['size_bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def adapter_bundle_digest(adapter_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    required = ["adapter_config.json", "adapters.safetensors"]
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for relative in sorted(required):
        path = adapter_dir / relative
        if not path.is_file():
            raise HarnessError(f"adapter bundle is missing required file {relative}")
        if relative.endswith(".safetensors"):
            safetensors_header(path)
        file_digest = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode("utf-8"))
        entries.append({"path": relative, "size_bytes": size, "sha256": file_digest})
    return digest.hexdigest(), entries


@dataclass(frozen=True)
class ParentAdapterInitialization:
    """Verified immutable adapter used to initialize a distinct training run."""

    manifest_path: Path
    adapter_dir: Path
    adapter_config_path: Path
    weights_path: Path
    identity: Mapping[str, Any]


def _parent_adapter_path(manifest_path: Path, value: Any) -> Path:
    relative = Path(str(value or ""))
    if not str(value or "") or relative.is_absolute() or ".." in relative.parts:
        raise HarnessError("parent adapter manifest contains an unsafe adapter.path")
    destination = (manifest_path.parent / relative).resolve()
    try:
        destination.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise HarnessError(
            "parent adapter manifest escapes its containing directory") from exc
    if not destination.is_dir():
        raise HarnessError(f"parent adapter directory is missing: {destination}")
    return destination


def _expected_adapter_tensor_names(target_inventory: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    targets_per_layer = target_inventory.get("targets_per_layer")
    if not isinstance(targets_per_layer, Mapping) or not targets_per_layer:
        raise HarnessError("base preflight has no exact LoRA target inventory")
    for raw_layer, raw_targets in targets_per_layer.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise HarnessError("base LoRA inventory has an invalid layer") from exc
        if not isinstance(raw_targets, list) or not raw_targets:
            raise HarnessError("base LoRA inventory has an empty target layer")
        for raw_target in raw_targets:
            target = str(raw_target)
            prefix = f"language_model.model.layers.{layer}.{target}"
            result.update({f"{prefix}.lora_a", f"{prefix}.lora_b"})
    return result


def _validate_parent_tensor_inventory(
    weights_path: Path, target_inventory: Mapping[str, Any], *, rank: int,
) -> None:
    header = safetensors_header(weights_path)
    observed = {name for name in header if name != "__metadata__"}
    expected = _expected_adapter_tensor_names(target_inventory)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:4]))
        if extra:
            detail.append("unexpected " + ", ".join(extra[:4]))
        raise HarnessError(
            "parent adapter tensor inventory does not exactly match the configured "
            "LoRA topology" + (": " + "; ".join(detail) if detail else ""))
    for name in sorted(observed):
        spec = header[name]
        shape = spec.get("shape") if isinstance(spec, Mapping) else None
        dtype = spec.get("dtype") if isinstance(spec, Mapping) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape
            )
        ):
            raise HarnessError(f"parent adapter tensor {name!r} has an invalid shape")
        if name.endswith(".lora_a") and shape[-1] != rank:
            raise HarnessError(f"parent adapter tensor {name!r} has the wrong rank")
        if name.endswith(".lora_b") and shape[0] != rank:
            raise HarnessError(f"parent adapter tensor {name!r} has the wrong rank")
        if dtype != "F32":
            raise HarnessError(
                f"parent adapter tensor {name!r} must retain MLX-LM F32 LoRA weights")


def validate_parent_adapter_initialization(
    manifest_path: Path, config: Mapping[str, Any],
    base_receipt: Mapping[str, Any],
) -> ParentAdapterInitialization:
    """Authenticate a parent adapter without importing or loading MLX weights."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path, "parent academic adapter manifest")
    if manifest.get("schema_version") != ADAPTER_SCHEMA:
        raise HarnessError("parent adapter manifest has an incompatible schema")
    parent_profile = str(manifest.get("profile_id", ""))
    parent_prompt = str(manifest.get("prompt_contract", ""))
    if SUPPORTED_PROFILE_CONTRACTS.get(parent_profile) != parent_prompt:
        raise HarnessError("parent adapter profile/prompt contract is unsupported")
    base = manifest.get("base_model")
    if not isinstance(base, Mapping):
        raise HarnessError("parent adapter manifest has no base-model receipt")
    for field in (
        "model_id", "revision", "model_type", "architecture", "config_sha256",
        "weight_index_sha256", "weight_inventory_sha256", "weight_files",
        "quantization",
    ):
        if base.get(field) != base_receipt.get(field):
            raise HarnessError(
                f"parent adapter base {field} does not match the selected Qwen3.8 base")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, Mapping) or adapter.get("format") != "mlx_lm_lora":
        raise HarnessError("parent adapter manifest has no MLX-LM LoRA bundle")
    adapter_dir = _parent_adapter_path(manifest_path, adapter.get("path"))
    bundle_sha256, required_files = adapter_bundle_digest(adapter_dir)
    if adapter.get("sha256") != bundle_sha256:
        raise HarnessError("parent adapter bundle digest does not match its manifest")
    manifest_files = adapter.get("required_files")
    if not isinstance(manifest_files, list) or manifest_files != required_files:
        raise HarnessError(
            "parent adapter required-file inventory does not match its manifest")

    adapter_config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapters.safetensors"
    adapter_config = _read_json(adapter_config_path, "parent adapter config")
    if adapter_config.get("fine_tune_type") != "lora":
        raise HarnessError("parent adapter is not a LoRA adapter")
    if adapter_config.get("num_layers") != config["training"]["num_layers"]:
        raise HarnessError("parent adapter num_layers does not match stage-two topology")
    parent_lora = adapter_config.get("lora_parameters")
    if not isinstance(parent_lora, Mapping):
        raise HarnessError("parent adapter config has no LoRA parameters")
    expected_lora = config["lora"]
    try:
        parent_scale = float(parent_lora.get("scale"))
        parent_dropout = float(parent_lora.get("dropout"))
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "parent adapter LoRA scale/dropout must be finite numbers") from exc
    if (
        parent_lora.get("keys") != expected_lora["keys"]
        or parent_lora.get("rank") != expected_lora["rank"]
        or not math.isfinite(parent_scale)
        or not math.isfinite(parent_dropout)
        or parent_scale != float(expected_lora["scale"])
        or parent_dropout != float(expected_lora["dropout"])
    ):
        raise HarnessError(
            "parent adapter LoRA keys/rank/scale/dropout do not exactly match "
            "stage-two topology")
    target_inventory = base_receipt.get("target_inventory")
    if not isinstance(target_inventory, Mapping):
        raise HarnessError("selected base receipt has no LoRA target inventory")
    parent_training = manifest.get("training")
    if not isinstance(parent_training, Mapping) or (
        parent_training.get("trainable_layers")
        != target_inventory.get("selected_layers")
        or parent_training.get("trainable_target_paths") != expected_lora["keys"]
        or parent_training.get("target_path_counts")
        != target_inventory.get("target_path_counts")
        or parent_training.get("total_target_modules")
        != target_inventory.get("total_target_modules")
    ):
        raise HarnessError(
            "parent adapter manifest LoRA inventory does not match the selected base")
    _validate_parent_tensor_inventory(
        weights_path, target_inventory, rank=int(expected_lora["rank"]))

    config_entry = next(
        entry for entry in required_files if entry["path"] == "adapter_config.json")
    weights_entry = next(
        entry for entry in required_files if entry["path"] == "adapters.safetensors")
    parent_lineage = manifest.get("lineage")
    parent_generation = 0
    parent_lineage_sha256 = None
    if parent_lineage is not None:
        if not isinstance(parent_lineage, Mapping):
            raise HarnessError("parent adapter lineage must be an object")
        if set(parent_lineage) != {"schema_version", "generation", "parent"}:
            raise HarnessError("parent adapter lineage has unexpected fields")
        if parent_lineage.get("schema_version") != ADAPTER_LINEAGE_SCHEMA:
            raise HarnessError("parent adapter lineage has an incompatible schema")
        generation = parent_lineage.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
            raise HarnessError("parent adapter lineage generation is invalid")
        ancestor = parent_lineage.get("parent")
        if not isinstance(ancestor, Mapping):
            raise HarnessError("parent adapter lineage has no authenticated ancestor")
        allowed_ancestor_fields = {
            "schema_version", "generation", "manifest_sha256", "profile_id",
            "prompt_contract", "base_weight_inventory_sha256",
            "adapter_tree_sha256", "adapter_config_sha256",
            "adapter_weights_sha256", "lora_topology_sha256",
            "parent_lineage_sha256",
        }
        if set(ancestor) - allowed_ancestor_fields:
            raise HarnessError("parent adapter lineage ancestor has unexpected fields")
        if (
            ancestor.get("schema_version") != PARENT_ADAPTER_SCHEMA
            or ancestor.get("generation") != generation
        ):
            raise HarnessError("parent adapter lineage generation is inconsistent")
        if SUPPORTED_PROFILE_CONTRACTS.get(str(ancestor.get("profile_id", ""))) != str(
            ancestor.get("prompt_contract", "")
        ):
            raise HarnessError("parent adapter lineage profile is unsupported")
        for digest_field in (
            "manifest_sha256", "base_weight_inventory_sha256",
            "adapter_tree_sha256", "adapter_config_sha256",
            "adapter_weights_sha256", "lora_topology_sha256",
        ):
            if not SHA256_RE.fullmatch(str(ancestor.get(digest_field, ""))):
                raise HarnessError(
                    f"parent adapter lineage has invalid {digest_field}")
        optional_lineage_hash = ancestor.get("parent_lineage_sha256")
        if optional_lineage_hash is not None and not SHA256_RE.fullmatch(
            str(optional_lineage_hash)
        ):
            raise HarnessError(
                "parent adapter lineage has invalid parent_lineage_sha256")
        if ancestor.get("base_weight_inventory_sha256") != base.get(
            "weight_inventory_sha256"
        ):
            raise HarnessError(
                "parent adapter ancestor is based on a different weight inventory")
        parent_generation = generation
        parent_lineage_sha256 = sha256_bytes(canonical_json(parent_lineage))
    topology = {
        "num_layers": config["training"]["num_layers"],
        "keys": list(expected_lora["keys"]),
        "rank": expected_lora["rank"],
        "scale": expected_lora["scale"],
        "dropout": expected_lora["dropout"],
        "tensor_names": sorted(_expected_adapter_tensor_names(target_inventory)),
    }
    identity = {
        "schema_version": PARENT_ADAPTER_SCHEMA,
        "generation": parent_generation + 1,
        "manifest_sha256": sha256_file(manifest_path),
        "profile_id": parent_profile,
        "prompt_contract": parent_prompt,
        "base_weight_inventory_sha256": base["weight_inventory_sha256"],
        "adapter_tree_sha256": bundle_sha256,
        "adapter_config_sha256": config_entry["sha256"],
        "adapter_weights_sha256": weights_entry["sha256"],
        "lora_topology_sha256": sha256_bytes(canonical_json(topology)),
    }
    if parent_lineage_sha256 is not None:
        identity["parent_lineage_sha256"] = parent_lineage_sha256
    return ParentAdapterInitialization(
        manifest_path=manifest_path,
        adapter_dir=adapter_dir,
        adapter_config_path=adapter_config_path,
        weights_path=weights_path,
        identity=identity,
    )


def publish_adapter_bundle(work_dir: Path, output_root: Path) -> tuple[Path, str, list[dict[str, Any]]]:
    adapter_dir = output_root / "adapter"
    if adapter_dir.exists():
        raise HarnessError(f"refusing to overwrite an existing published adapter: {adapter_dir}")
    bundle_digest, entries = adapter_bundle_digest(work_dir)
    staging = output_root / f".adapter.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        for entry in entries:
            source = work_dir / entry["path"]
            destination = staging / entry["path"]
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        copied_digest, _ = adapter_bundle_digest(staging)
        if copied_digest != bundle_digest:
            raise HarnessError("adapter changed while being published")
        os.replace(staging, adapter_dir)
        _fsync_directory(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return adapter_dir, bundle_digest, entries


def build_adapter_manifest(
    *, config: Mapping[str, Any], base_receipt: Mapping[str, Any], dataset_manifest: Mapping[str, Any],
    dataset_manifest_path: Path, adapter_manifest_path: Path, adapter_dir: Path, bundle_digest: str,
    required_files: Sequence[Mapping[str, Any]], package_versions: Mapping[str, str | None],
    training_data_receipt: Mapping[str, Any] | None = None,
    parent_adapter: ParentAdapterInitialization | None = None,
) -> dict[str, Any]:
    training = config["training"]
    target_inventory = base_receipt["target_inventory"]
    source_manifest_path = Path(str(dataset_manifest["source_corpus_manifest"]))
    configured_profile = config.get("profile")
    if isinstance(configured_profile, Mapping):
        profile_id = str(configured_profile.get("id", PROFILE_ID))
        prompt_contract = str(
            configured_profile.get("prompt_contract", PROMPT_CONTRACT))
    else:
        # The low-level builder historically accepted minimal, already-audited
        # prose configs. Production configs still pass strict validation at load.
        profile_id = PROFILE_ID
        prompt_contract = PROMPT_CONTRACT
    structure_mode = prompt_contract == STRUCTURE_PROMPT_CONTRACT
    manifest = {
        "schema_version": ADAPTER_SCHEMA,
        "profile_id": profile_id,
        "prompt_contract": prompt_contract,
        "base_model": {
            "model_id": base_receipt["model_id"],
            "revision": base_receipt["revision"],
            "model_type": base_receipt["model_type"],
            "architecture": base_receipt["architecture"],
            "config_sha256": base_receipt["config_sha256"],
            "weight_index_sha256": base_receipt.get("weight_index_sha256"),
            "weight_inventory_sha256": base_receipt["weight_inventory_sha256"],
            "weight_files": list(base_receipt["weight_files"]),
            "quantization": base_receipt["quantization"],
        },
        "adapter": {
            "path": os.path.relpath(adapter_dir, adapter_manifest_path.parent),
            "format": "mlx_lm_lora",
            "sha256": bundle_digest,
            "sha256_semantics": "sha256(sorted(relative_path\\0size_bytes\\0file_sha256\\n))",
            "required_files": list(required_files),
        },
        "training": {
            "seed": training["seed"],
            "batch_size": training["batch_size"],
            "grad_accumulation_steps": training["grad_accumulation_steps"],
            "gradient_checkpointing": training["grad_checkpoint"],
            "completion_only_loss": training["mask_prompt"],
            "trainable_layers": target_inventory["selected_layers"],
            "trainable_target_paths": list(config["lora"]["keys"]),
            "target_path_counts": target_inventory["target_path_counts"],
            "total_target_modules": target_inventory["total_target_modules"],
            "corpus_manifest_path": os.path.relpath(
                source_manifest_path, adapter_manifest_path.parent),
            "corpus_manifest_sha256": dataset_manifest["source_corpus_manifest_sha256"],
        },
        "dataset": {
            "dataset_manifest_path": os.path.relpath(
                dataset_manifest_path, adapter_manifest_path.parent),
            "manifest_sha256": sha256_file(dataset_manifest_path),
            "source_corpus_sha256": dataset_manifest["source_corpus_sha256"],
            "source_corpus_file_sha256": dataset_manifest["source_corpus_file_sha256"],
            "split_sha256": {
                name: value["sha256"] for name, value in dataset_manifest["splits"].items()
            },
        },
        "environment": {
            "mlx": package_versions.get("mlx"),
            "mlx_lm": package_versions.get("mlx_lm"),
        },
        "runtime": {
            "provider": "mlx_lm",
            "model": (
                "qwen3.8-27b-academic-structure"
                if structure_mode else "qwen3.8-27b-academic"),
            "base_url": "http://127.0.0.1:8080/v1",
            "transport_adapter": "openai-compatible",
            # Default request multiplier around the trained adapter delta. The
            # server may select a validated per-request multiplier without
            # changing the scale stored in adapter_config.json.
            "default_adapter_strength": 1.0,
        },
    }
    if structure_mode:
        manifest["runtime"].update({
            "scope": "spiral_paper_planner_only",
            "spiralchat_eligible": False,
        })
    if training_data_receipt is not None:
        manifest["dataset"]["bounded_training_data"] = _bounded_training_data_attestation(
            training_data_receipt)
    if parent_adapter is not None:
        manifest["lineage"] = {
            "schema_version": ADAPTER_LINEAGE_SCHEMA,
            "generation": parent_adapter.identity["generation"],
            "parent": dict(parent_adapter.identity),
        }
    return manifest


def parse_training_metric(
    line: str, *, mode: str, cumulative_offset: int = 0,
) -> dict[str, Any] | None:
    """Parse only MLX-LM's audited metric lines; never infer from arbitrary text."""

    if mode not in {"production", "smoke_only", "feasibility_only"}:
        raise HarnessError(
            "training metric mode must be production, smoke_only, or feasibility_only")
    stripped = line.strip()
    match = TRAIN_METRIC_RE.fullmatch(stripped)
    event = "train"
    float_fields = (
        "train_loss", "learning_rate", "iterations_per_second",
        "tokens_per_second", "peak_memory_gb",
    )
    integer_fields = ("trained_tokens",)
    if match is None:
        match = VAL_METRIC_RE.fullmatch(stripped)
        event = "validation"
        float_fields = ("val_loss", "validation_seconds")
        integer_fields = ()
    if match is None:
        return None
    phase_iteration = int(match.group("iteration"))
    record: dict[str, Any] = {
        "schema_version": TRAINING_METRIC_SCHEMA,
        "mode": mode,
        "event": event,
        "iteration": cumulative_offset + phase_iteration,
        "phase_iteration": phase_iteration,
        "valid": True,
    }
    errors: list[str] = []
    for field_name in float_fields:
        raw_value = match.group(field_name)
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{field_name} is not numeric: {raw_value!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"{field_name} is nonfinite: {raw_value!r}")
            continue
        if field_name.endswith("loss") and value < 0:
            errors.append(f"{field_name} is negative: {raw_value!r}")
            continue
        record[field_name] = value
    for field_name in integer_fields:
        record[field_name] = int(match.group(field_name))
    if errors:
        record["valid"] = False
        record["errors"] = errors
        record["rejected_values"] = {
            field_name: match.group(field_name) for field_name in float_fields
            if field_name not in record
        }
    return record


def _metric_points(records: Sequence[Mapping[str, Any]], field_name: str) -> list[tuple[int, float]]:
    return [
        (int(record["iteration"]), float(record[field_name]))
        for record in records
        if record.get("valid") is True and field_name in record
    ]


def render_training_metrics_html(
    records: Sequence[Mapping[str, Any]], destination: Path, *, mode: str,
) -> None:
    """Atomically refresh a dependency-free localhost/file loss-curve view."""

    train = _metric_points(records, "train_loss")
    validation = _metric_points(records, "val_loss")
    all_points = train + validation
    width, height, padding = 960, 480, 54
    if all_points:
        minimum_x = min(point[0] for point in all_points)
        maximum_x = max(point[0] for point in all_points)
        minimum_y = min(point[1] for point in all_points)
        maximum_y = max(point[1] for point in all_points)
    else:
        minimum_x, maximum_x, minimum_y, maximum_y = 0, 1, 0.0, 1.0
    maximum_x = max(maximum_x, minimum_x + 1)
    maximum_y = max(maximum_y, minimum_y + 1e-9)

    def polyline(points: Sequence[tuple[int, float]]) -> str:
        values = []
        for x_value, y_value in points:
            x = padding + (x_value - minimum_x) / (maximum_x - minimum_x) * (width - 2 * padding)
            y = height - padding - (y_value - minimum_y) / (maximum_y - minimum_y) * (height - 2 * padding)
            values.append(f"{x:.2f},{y:.2f}")
        return " ".join(values)

    invalid = [record for record in records if record.get("valid") is False]
    last_train = train[-1][1] if train else None
    last_val = validation[-1][1] if validation else None
    status = (
        f"train {last_train:.3f}" if last_train is not None else "train pending")
    status += " · " + (
        f"validation {last_val:.3f}" if last_val is not None else "validation pending")
    if invalid:
        status += f" · {len(invalid)} rejected nonfinite/malformed metric(s)"
    safe_mode = html.escape(mode)
    safe_status = html.escape(status)
    payload = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="2">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Spiral academic training · {safe_mode}</title>
<style>body{{background:#171411;color:#e8ded2;font:16px ui-monospace,monospace;margin:32px}}
.card{{max-width:1040px;background:#211c18;border:1px solid #4b3b30;border-radius:16px;padding:24px}}
svg{{width:100%;height:auto;background:#15120f;border-radius:10px}}.train{{fill:none;stroke:#ef815f;stroke-width:3}}
.val{{fill:none;stroke:#77c990;stroke-width:3}}.muted{{color:#ae9b8b}}.bad{{color:#ff8b76}}</style></head>
<body><main class="card"><h1>academic adapter · {safe_mode}</h1><p>{safe_status}</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="training and validation loss">
<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#705f52"/>
<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#705f52"/>
<polyline class="train" points="{polyline(train)}"/><polyline class="val" points="{polyline(validation)}"/></svg>
<p><span style="color:#ef815f">train</span> · <span style="color:#77c990">validation</span></p>
<p class="muted">iteration {minimum_x}–{maximum_x} · loss {minimum_y:.3f}–{maximum_y:.3f} · refreshes every 2 seconds · no network assets</p>
{('<p class="bad">Invalid metrics were rejected; this run cannot publish until resolved.</p>' if invalid else '')}
</main></body></html>"""
    atomic_write_bytes(destination, payload.encode("utf-8"))


class TrainingMetricsJournal:
    """Append-only JSONL metrics with resume de-duplication and atomic HTML view."""

    def __init__(
        self, path: Path, *, mode: str, run_id: str, cumulative_offset: int = 0,
        html_path: Path | None = None,
    ) -> None:
        self.path = path
        self.mode = mode
        self.run_id = run_id
        self.cumulative_offset = cumulative_offset
        self.html_path = html_path or path.with_name("loss-curves.html")
        self.records: list[dict[str, Any]] = []
        self.keys: set[tuple[str, int]] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HarnessError(
                        f"training metrics journal is corrupt at line {line_number}: {exc}") from exc
                if record.get("schema_version") != TRAINING_METRIC_SCHEMA:
                    raise HarnessError("training metrics journal has an incompatible record")
                if record.get("mode") != mode:
                    raise HarnessError("training metrics journal belongs to a different run mode")
                key = (str(record.get("event")), int(record.get("iteration", -1)))
                if key in self.keys:
                    raise HarnessError(f"training metrics journal contains duplicate {key}")
                self.keys.add(key)
                self.records.append(record)
        render_training_metrics_html(self.records, self.html_path, mode=mode)

    def consume(self, line: str) -> dict[str, Any] | None:
        record = parse_training_metric(
            line, mode=self.mode, cumulative_offset=self.cumulative_offset)
        if record is None:
            return None
        key = (str(record["event"]), int(record["iteration"]))
        if key in self.keys:
            return None
        record["run_id"] = self.run_id
        record["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        raw = canonical_json(record)
        descriptor = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.keys.add(key)
        self.records.append(record)
        render_training_metrics_html(self.records, self.html_path, mode=self.mode)
        return record

    def require_valid(self) -> None:
        invalid = [record for record in self.records if record.get("valid") is False]
        if invalid:
            detail = "; ".join(
                f"{record.get('event')} iter {record.get('iteration')}: "
                + ", ".join(record.get("errors") or [])
                for record in invalid[:5]
            )
            raise HarnessError(f"trainer emitted rejected nonfinite/invalid metrics: {detail}")
        if not any(record.get("event") == "train" for record in self.records):
            raise HarnessError("trainer emitted no parseable training-loss metric")


def run_training_process(
    command: Sequence[str], work_adapter_dir: Path, ledger: CheckpointLedger,
    *, cumulative_offset: int = 0, poll_seconds: float = 0.5,
    inherited_lease_fd: int | None = None,
    inherited_run_lock_fd: int | None = None,
    metrics_path: Path | None = None, metrics_mode: str = "production",
    metrics_run_id: str = "", trainer_log_path: Path | None = None,
) -> int:
    """Run MLX-LM and atomically mirror each complete numbered checkpoint."""
    pass_fds = tuple(
        descriptor for descriptor in (inherited_lease_fd, inherited_run_lock_fd)
        if descriptor is not None
    )
    journal = (
        TrainingMetricsJournal(
            metrics_path, mode=metrics_mode,
            run_id=metrics_run_id or uuid.uuid4().hex,
            cumulative_offset=cumulative_offset)
        if metrics_path is not None else None
    )
    log_descriptor: int | None = None
    if trainer_log_path is not None:
        trainer_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_descriptor = os.open(
            trainer_log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        process = subprocess.Popen(
            list(command), pass_fds=pass_fds, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
    except BaseException:
        if log_descriptor is not None:
            os.close(log_descriptor)
        raise
    line_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for output_line in process.stdout:
                line_queue.put(output_line)
        finally:
            line_queue.put(None)

    reader = threading.Thread(target=read_output, name="academic-trainer-stdout", daemon=True)
    reader.start()
    captured: set[Path] = set()
    output_complete = False
    metric_failures: list[str] = []
    try:
        while process.poll() is None or not output_complete:
            while True:
                try:
                    output_line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if output_line is None:
                    output_complete = True
                    break
                sys.stdout.write(output_line)
                sys.stdout.flush()
                if log_descriptor is not None:
                    os.write(log_descriptor, output_line.encode("utf-8", errors="replace"))
                if journal is not None:
                    try:
                        journal.consume(output_line)
                    except (HarnessError, OSError) as exc:
                        metric_failures.append(str(exc))
            _capture_native_checkpoints(
                work_adapter_dir, ledger, captured, cumulative_offset, strict=False)
            time.sleep(poll_seconds)
        reader.join(timeout=2)
        _capture_native_checkpoints(
            work_adapter_dir, ledger, captured, cumulative_offset,
            # A signal-killed Metal child can leave its current native snapshot
            # incomplete. Never let that uncommitted file mask the negative
            # return code that the bounded supervisor must classify and retry.
            strict=process.returncode == 0)
        if log_descriptor is not None:
            os.fsync(log_descriptor)
        if metric_failures:
            raise HarnessError(
                "training completed but its loss stream failed closed: "
                + "; ".join(metric_failures[:3]))
        if process.returncode:
            return int(process.returncode)
        if journal is not None:
            journal.require_valid()
        return 0
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        # Preserve any checkpoint that became complete while the child was being
        # stopped. A partial native file remains uncommitted and cannot move the
        # ledger pointer.
        _capture_native_checkpoints(
            work_adapter_dir, ledger, captured, cumulative_offset, strict=False)
        raise
    finally:
        if log_descriptor is not None:
            os.close(log_descriptor)


def _capture_native_checkpoints(
    work_adapter_dir: Path, ledger: CheckpointLedger, captured: set[Path], cumulative_offset: int,
    *, strict: bool,
) -> None:
    pattern = re.compile(r"^(\d{7})_adapters\.safetensors$")
    for source in sorted(work_adapter_dir.glob("*_adapters.safetensors")):
        if source in captured:
            continue
        match = pattern.fullmatch(source.name)
        if not match:
            continue
        try:
            ledger.capture(
                source,
                cumulative_offset + int(match.group(1)),
                adapter_config=work_adapter_dir / "adapter_config.json",
            )
        except HarnessError:
            # The native writer may still have the file open. It will be retried.
            if strict:
                raise
            continue
        captured.add(source)


class TrainingComputeLease:
    """Fail-closed owner of SpiralChat's host-wide model flock.

    Training is a background, multi-hour operation and therefore never publishes an
    ``interactive-*`` priority ticket.  It checks that queue on both sides of the
    non-blocking flock acquisition, then holds the descriptor for the complete MLX
    subprocess lifetime.
    """

    def __init__(self, path: Path | None = None):
        self.path = (path or Path("~/.spiralchat/spiral-compute.lease")).expanduser()
        self.descriptor: int | None = None

    @property
    def priority_dir(self) -> Path:
        return self.path.parent / "spiral-compute.priority"

    def _live_interactive_tickets(self) -> list[Path]:
        try:
            tickets = list(self.priority_dir.iterdir())
        except OSError:
            return []
        now = time.time()
        live: list[Path] = []
        for ticket in tickets:
            if not ticket.name.startswith("interactive-") or ticket.suffix != ".json":
                continue
            stale = False
            try:
                stat = ticket.stat()
                payload = json.loads(ticket.read_text(encoding="utf-8")[:4096])
                pid = int(payload.get("pid") or 0)
                stale = pid <= 0 or now - stat.st_mtime > 300
                if not stale:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        pass
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                try:
                    stale = now - ticket.lstat().st_mtime > 300
                except OSError:
                    stale = False
            if stale:
                try:
                    ticket.unlink()
                except OSError:
                    pass
            else:
                live.append(ticket)
        return live

    def acquire(self, *, owner: Mapping[str, Any]) -> None:
        if fcntl is None:
            raise HarnessError("the shared compute lease requires fcntl")
        if self.descriptor is not None:
            raise HarnessError("training compute lease is already acquired")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._live_interactive_tickets():
            raise HarnessError("interactive SpiralChat work is queued; academic training will not cut in line")
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(self.path, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HarnessError("SpiralChat's shared compute lane is busy; training was not started") from exc
            if self._live_interactive_tickets():
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                raise HarnessError("interactive SpiralChat work arrived; training was not started")
            record = {
                "pid": os.getpid(),
                "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "spiral_academic_qlora",
                "operation": "training",
                **dict(owner),
            }
            raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")[:4096]
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, raw)
            os.fsync(descriptor)
            os.set_inheritable(descriptor, True)
            self.descriptor = descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "TrainingComputeLease":
        if self.descriptor is None:
            raise HarnessError("call acquire() before entering the training lease")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def verify_ollama_empty(base_url: str = "http://127.0.0.1:11434", *, timeout: float = 3.0) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + "/api/ps", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
        raise HarnessError(f"cannot verify Ollama residency at {base_url}/api/ps: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        raise HarnessError("Ollama /api/ps returned an invalid residency response")
    if models:
        names = [str(item.get("name") or item.get("model") or "unknown") for item in models if isinstance(item, Mapping)]
        raise HarnessError(
            "Ollama still has resident model(s); training will not evict them: " + ", ".join(names)
        )
    return {"base_url": base_url, "resident_models": [], "verified_empty": True}
