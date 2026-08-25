#!/usr/bin/env python3
"""Lease-scoped OpenAI-compatible runtime for Spiral's academic MLX adapter.

The HTTP parent never imports MLX or retains weights.  Each accepted completion
rehashes the manifest-bound assets, acquires Spiral's host-wide compute lease,
verifies that Ollama is empty, and runs generation in a child process.  Process
exit unloads every MLX object before the parent releases the lease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# The authenticated parent intentionally launches this file directly in the
# selected MLX runtime.  Direct execution places only this script's directory on
# sys.path, while both the worker and its helpers import the shared ``spiral``
# package from the repository root.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from spiral.academic_structure_contract import (
    STRUCTURE_REQUEST_SYSTEM_MARKER,
    StructurePromptError,
    parse_brief_to_blueprint_prompt,
)

try:
    from .training_support import (
        ADAPTER_SCHEMA,
        DATASET_SCHEMA,
        EXPECTED_ARCHITECTURE,
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_TYPE,
        EXPECTED_REVISION,
        MODEL_VIEW_SOURCE,
        PROFILE_ID,
        PROMPT_CONTRACT,
        REQUIRED_STRATA,
        STRUCTURE_PROFILE_ID,
        STRUCTURE_PROMPT_CONTRACT,
        HarnessError,
        TrainingComputeLease,
        atomic_write_json,
        canonical_json,
        python_package_versions,
        sha256_bytes,
        sha256_file,
        verify_ollama_empty,
    )
except ImportError:  # direct script execution
    from scripts.academic_finetune.training_support import (  # type: ignore
        ADAPTER_SCHEMA,
        DATASET_SCHEMA,
        EXPECTED_ARCHITECTURE,
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_TYPE,
        EXPECTED_REVISION,
        MODEL_VIEW_SOURCE,
        PROFILE_ID,
        PROMPT_CONTRACT,
        REQUIRED_STRATA,
        STRUCTURE_PROFILE_ID,
        STRUCTURE_PROMPT_CONTRACT,
        HarnessError,
        TrainingComputeLease,
        atomic_write_json,
        canonical_json,
        python_package_versions,
        sha256_bytes,
        sha256_file,
        verify_ollama_empty,
    )


IDENTITY_SCHEMA = "spiral.academic-runtime-identity.v1"
CORPUS_MANIFEST_SCHEMA = "spiral.academic-corpus-manifest.v1"
MODEL_VIEW_SCHEMA = "spiral.qwen35-text-training-view.v1"
EXPECTED_PROVIDER = "mlx_lm"
EXPECTED_RUNTIME_MODEL = "qwen3.8-27b-academic"
EXPECTED_STRUCTURE_RUNTIME_MODEL = "qwen3.8-27b-academic-structure"
EXPECTED_TRANSPORT = "openai-compatible"
STRUCTURE_RUNTIME_SCOPE = "spiral_paper_planner_only"
STRUCTURE_SPIRALCHAT_ELIGIBLE = False
DEFAULT_ADAPTER_STRENGTH = 1.0
MIN_ADAPTER_STRENGTH = 0.0
MAX_ADAPTER_STRENGTH = 2.0
ADAPTER_STRENGTH_STEP = 0.05
EXPECTED_REQUIRED_FILES = ("adapter_config.json", "adapters.safetensors")
ADAPTER_DIGEST_SEMANTICS = "sha256(sorted(relative_path\\0size_bytes\\0file_sha256\\n))"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 100_000
MAX_COMPLETION_TOKENS = 8192
SERVER_LIFECYCLE_IDENTITY = {
    "server_contract": "spiral.academic-one-request-server.v1",
    "weight_residency": "child-process-per-request",
    "compute_lease": "spiral-compute-flock-v1",
    "ollama_admission": "strict-empty-no-eviction",
    "unload_boundary": "child-exit-before-lease-release",
    "adapter_strength_supported": True,
    "adapter_strength_min": MIN_ADAPTER_STRENGTH,
    "adapter_strength_max": MAX_ADAPTER_STRENGTH,
    "adapter_strength_step": ADAPTER_STRENGTH_STEP,
    "adapter_strength_default": DEFAULT_ADAPTER_STRENGTH,
}


class RequestError(HarnessError):
    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class AcademicRuntimeContract:
    """Immutable serving identity selected by an adapter's training contract."""

    profile_id: str
    prompt_contract: str
    runtime_model: str
    strength_env: str
    scope: str | None = None
    spiralchat_eligible: bool | None = None


PROSE_RUNTIME_CONTRACT = AcademicRuntimeContract(
    profile_id=PROFILE_ID,
    prompt_contract=PROMPT_CONTRACT,
    runtime_model=EXPECTED_RUNTIME_MODEL,
    strength_env="SPIRAL_ACADEMIC_WRITER_STRENGTH",
)
STRUCTURE_RUNTIME_CONTRACT = AcademicRuntimeContract(
    profile_id=STRUCTURE_PROFILE_ID,
    prompt_contract=STRUCTURE_PROMPT_CONTRACT,
    runtime_model=EXPECTED_STRUCTURE_RUNTIME_MODEL,
    strength_env="SPIRAL_ACADEMIC_PLANNER_STRENGTH",
    scope=STRUCTURE_RUNTIME_SCOPE,
    spiralchat_eligible=STRUCTURE_SPIRALCHAT_ELIGIBLE,
)
RUNTIME_CONTRACTS = {
    (contract.profile_id, contract.prompt_contract): contract
    for contract in (PROSE_RUNTIME_CONTRACT, STRUCTURE_RUNTIME_CONTRACT)
}


def canonical_adapter_strength(value: Any, *, request: bool = False) -> float:
    """Validate and canonicalize the public LoRA contribution multiplier."""

    error_type = RequestError if request else HarnessError
    if request and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise error_type(
            "adapter_strength must be from 0.0 through 2.0 in 0.05 steps")
    try:
        strength = float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError):
        strength = float("nan")
    if not math.isfinite(strength) or not MIN_ADAPTER_STRENGTH <= strength <= MAX_ADAPTER_STRENGTH:
        raise error_type(
            "adapter_strength must be from 0.0 through 2.0 in 0.05 steps")
    steps = round(strength / ADAPTER_STRENGTH_STEP)
    canonical = round(steps * ADAPTER_STRENGTH_STEP, 10)
    if abs(strength - canonical) > 1e-9:
        raise error_type(
            "adapter_strength must be from 0.0 through 2.0 in 0.05 steps")
    return canonical


@dataclass(frozen=True)
class RuntimeAssets:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    adapter_dir: Path
    model_view: Path
    identity: dict[str, Any]


def _read_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} {path} must contain an object")
    return raw, value


def _section(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    if not isinstance(section, dict):
        raise HarnessError(f"academic adapter manifest is missing object {name!r}")
    return section


def _runtime_contract(manifest: Mapping[str, Any]) -> AcademicRuntimeContract:
    profile_id = manifest.get("profile_id")
    prompt_contract = manifest.get("prompt_contract")
    contract = RUNTIME_CONTRACTS.get((profile_id, prompt_contract))
    if contract is None:
        supported = ", ".join(
            f"{item.profile_id!r}/{item.prompt_contract!r}"
            for item in (PROSE_RUNTIME_CONTRACT, STRUCTURE_RUNTIME_CONTRACT)
        )
        raise HarnessError(
            "academic adapter profile/prompt contract is incompatible; "
            f"expected one of {supported}"
        )
    return contract


def _validate_runtime_contract(
    runtime: Mapping[str, Any], contract: AcademicRuntimeContract,
) -> None:
    expected_runtime = {
        "provider": EXPECTED_PROVIDER,
        "model": contract.runtime_model,
        "transport_adapter": EXPECTED_TRANSPORT,
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise HarnessError("academic adapter runtime provider identity is incompatible")
    if contract.scope is not None and (
        runtime.get("scope") != contract.scope
        or runtime.get("spiralchat_eligible") is not contract.spiralchat_eligible
    ):
        raise HarnessError(
            "academic structure runtime must be planner-only and SpiralChat-excluded"
        )


def _manifest_relative(manifest_path: Path, value: Any, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise HarnessError(f"academic adapter manifest is missing {label}")
    path = Path(text).expanduser()
    return (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _strict_adapter_digest(
    adapter_dir: Path, expected_rows: Any,
) -> tuple[str, list[dict[str, Any]]]:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise HarnessError("adapter.path must be a real immutable directory")
    if not isinstance(expected_rows, list) or [
        str(row.get("path") or "") for row in expected_rows if isinstance(row, dict)
    ] != list(EXPECTED_REQUIRED_FILES):
        raise HarnessError(
            "adapter.required_files must be adapter_config.json then adapters.safetensors")
    root = adapter_dir.resolve()
    lines: list[str] = []
    verified: list[dict[str, Any]] = []
    for expected, relative in zip(expected_rows, EXPECTED_REQUIRED_FILES, strict=True):
        if not isinstance(expected, Mapping):
            raise HarnessError("adapter.required_files entries must be objects")
        path = adapter_dir / relative
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise HarnessError(f"adapter file is missing, linked, or escapes its bundle: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected.get("size_bytes") or digest != str(expected.get("sha256") or ""):
            raise HarnessError(f"adapter file does not match manifest identity: {relative}")
        lines.append(f"{relative}\0{size}\0{digest}\n")
        verified.append({"path": relative, "size_bytes": size, "sha256": digest})
    return sha256_bytes("".join(sorted(lines)).encode("utf-8")), verified


def _validate_dataset_and_corpus(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    training = _section(manifest, "training")
    dataset_identity = _section(manifest, "dataset")
    corpus_path = _manifest_relative(
        manifest_path, training.get("corpus_manifest_path"),
        "training.corpus_manifest_path")
    corpus_raw, corpus = _read_object(corpus_path, "academic corpus manifest")
    if sha256_bytes(corpus_raw) != training.get("corpus_manifest_sha256"):
        raise HarnessError("corpus manifest bytes do not match the adapter receipt")
    if corpus.get("schema_version", corpus.get("schema")) != CORPUS_MANIFEST_SCHEMA:
        raise HarnessError("academic corpus manifest has the wrong schema")
    if corpus.get("trainable") is not True:
        raise HarnessError("academic corpus manifest is not marked trainable")
    if set(corpus.get("source_strata") or []) != REQUIRED_STRATA:
        raise HarnessError("academic corpus manifest does not attest the exact three strata")
    counts = corpus.get("counts")
    split_counts = counts.get("by_split") if isinstance(counts, Mapping) else None
    if not isinstance(split_counts, Mapping) or any(
        not isinstance(split_counts.get(split), int) or split_counts[split] <= 0
        for split in ("train", "validation", "test")
    ):
        raise HarnessError("academic corpus manifest does not attest nonempty train/validation/test splits")
    if not isinstance(corpus.get("split_policy"), str) or "document-author" not in corpus["split_policy"]:
        raise HarnessError("academic corpus manifest does not attest author-safe split policy")

    dataset_path = _manifest_relative(
        manifest_path, dataset_identity.get("dataset_manifest_path"),
        "dataset.dataset_manifest_path")
    dataset_raw, dataset = _read_object(dataset_path, "academic dataset manifest")
    if sha256_bytes(dataset_raw) != dataset_identity.get("manifest_sha256"):
        raise HarnessError("dataset manifest bytes do not match the adapter receipt")
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise HarnessError("academic dataset manifest has the wrong schema")
    if dataset.get("prompt_contract") != PROMPT_CONTRACT or dataset.get("completion_only_loss") is not True:
        raise HarnessError("academic dataset does not preserve the completion-only prompt contract")
    if dataset.get("source_corpus_manifest_sha256") != training.get("corpus_manifest_sha256"):
        raise HarnessError("dataset and adapter name different corpus manifests")
    for field in ("source_corpus_sha256", "source_corpus_file_sha256"):
        if dataset.get(field) != dataset_identity.get(field):
            raise HarnessError(f"dataset {field} does not match adapter identity")
    if corpus.get("corpus_sha256") != dataset_identity.get("source_corpus_file_sha256"):
        raise HarnessError("source corpus bytes do not match corpus and adapter receipts")
    source_path = Path(str(dataset.get("source_corpus") or "")).expanduser()
    if not source_path.is_absolute():
        source_path = (dataset_path.parent / source_path).resolve()
    if not source_path.is_file() or sha256_file(source_path) != dataset_identity.get("source_corpus_file_sha256"):
        raise HarnessError("source corpus is missing or changed after training")
    split_hashes = dataset_identity.get("split_sha256")
    if not isinstance(split_hashes, Mapping):
        raise HarnessError("adapter receipt has no split hash inventory")
    for split in ("train", "valid", "test"):
        entry = (dataset.get("splits") or {}).get(split)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("count"), int) or entry["count"] <= 0:
            raise HarnessError(f"prepared {split} split is absent or empty")
        split_path = (dataset_path.parent / str(entry.get("path") or "")).resolve()
        if not split_path.is_file() or sha256_file(split_path) != entry.get("sha256"):
            raise HarnessError(f"prepared {split} split changed after training")
        if split_hashes.get(split) != entry.get("sha256"):
            raise HarnessError(f"adapter receipt names the wrong {split} split")
    observed_strata = {
        str(name)
        for entry in (dataset.get("splits") or {}).values()
        if isinstance(entry, Mapping)
        for name, count in (entry.get("source_strata") or {}).items()
        if isinstance(count, int) and count > 0
    }
    if observed_strata != REQUIRED_STRATA:
        raise HarnessError("prepared dataset does not contain the exact three academic strata")


def _validate_training_identity_receipt(manifest: Mapping[str, Any]) -> None:
    """Validate immutable training provenance without reopening training storage.

    Corpus and prepared split bytes are inputs to training, not inference assets. The
    published adapter manifest already binds their SHA-256 identities and is itself
    rehashed at every inference admission. Requiring those historical paths to remain
    mounted made a fully local model deployment depend on a removable training disk and
    macOS privacy mediation, even though no training data is read by generation.
    """

    training = _section(manifest, "training")
    dataset = _section(manifest, "dataset")
    if training.get("completion_only_loss") is not True:
        raise HarnessError("academic training receipt must attest completion-only loss")
    corpus_path = str(training.get("corpus_manifest_path") or "")
    dataset_path = str(dataset.get("dataset_manifest_path") or "")
    if not corpus_path or not dataset_path:
        raise HarnessError("academic training receipt has no corpus/dataset identity")

    def digest(value: Any, label: str) -> str:
        selected = str(value or "").lower()
        if len(selected) != 64 or any(
            char not in "0123456789abcdef" for char in selected
        ):
            raise HarnessError(f"academic training receipt has invalid {label}")
        return selected

    digest(training.get("corpus_manifest_sha256"), "corpus manifest SHA-256")
    digest(dataset.get("manifest_sha256"), "dataset manifest SHA-256")
    digest(dataset.get("source_corpus_sha256"), "source corpus identity")
    digest(dataset.get("source_corpus_file_sha256"), "source corpus file SHA-256")
    split_hashes = dataset.get("split_sha256")
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != {
        "train", "valid", "test"
    }:
        raise HarnessError("academic training receipt has no exact split inventory")
    for split, value in split_hashes.items():
        digest(value, f"{split} split SHA-256")


def _validate_model_view(model_view: Path, base: Mapping[str, Any]) -> None:
    if model_view.is_symlink() or not model_view.is_dir():
        raise HarnessError("model view must be a real local directory")
    _, receipt = _read_object(model_view / "spiral_model_view_receipt.json", "model-view receipt")
    if receipt.get("schema_version") != MODEL_VIEW_SCHEMA:
        raise HarnessError("model view receipt has the wrong schema")
    receipt_base = receipt.get("base_model")
    if not isinstance(receipt_base, Mapping):
        raise HarnessError("model view receipt has no base identity")
    for field in (
        "model_id", "revision", "model_type", "architecture", "config_sha256",
        "weight_inventory_sha256",
    ):
        if receipt_base.get(field) != base.get(field):
            raise HarnessError(f"model view base {field} does not match adapter manifest")
    if receipt_base.get("quantization") != base.get("quantization"):
        raise HarnessError("model view quantization does not match adapter manifest")
    expected_weights = base.get("weight_files")
    if not isinstance(expected_weights, list) or not expected_weights:
        raise HarnessError("adapter manifest has no full base-weight inventory")
    lines: list[str] = []
    for row in expected_weights:
        if not isinstance(row, Mapping):
            raise HarnessError("invalid base weight inventory entry")
        relative = str(row.get("path") or "")
        if not relative or Path(relative).name != relative:
            raise HarnessError("base weight inventory contains an unsafe path")
        shard = model_view / relative
        if not shard.is_file():
            raise HarnessError(f"model view is missing base shard {relative}")
        size = shard.stat().st_size
        digest = sha256_file(shard)
        if size != row.get("size_bytes") or digest != row.get("sha256"):
            raise HarnessError(f"base shard changed after adapter training: {relative}")
        lines.append(f"{relative}\0{size}\0{digest}\n")
    if sha256_bytes("".join(lines).encode("utf-8")) != base.get("weight_inventory_sha256"):
        raise HarnessError("full base-weight inventory digest does not match adapter manifest")
    module_path = model_view / "spiral_qwen35_text_tuner.py"
    expected_module_sha = sha256_bytes(MODEL_VIEW_SOURCE.encode("utf-8"))
    if not module_path.is_file() or sha256_file(module_path) != expected_module_sha:
        raise HarnessError("model view custom text-only wrapper changed")
    _, view_config = _read_object(model_view / "config.json", "model-view config")
    if view_config.get("model_file") != module_path.name or view_config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise HarnessError("model view does not select the audited Qwen3.5 text-only wrapper")
    if EXPECTED_ARCHITECTURE not in (view_config.get("architectures") or []):
        raise HarnessError("model view has the wrong Qwen3.5 architecture")


def validate_runtime_assets(manifest_path: Path, model_view: Path) -> RuntimeAssets:
    manifest_path = manifest_path.expanduser().resolve()
    model_view = model_view.expanduser().resolve()
    manifest_raw, manifest = _read_object(manifest_path, "academic adapter manifest")
    if manifest.get("schema_version") != ADAPTER_SCHEMA:
        raise HarnessError(f"academic adapter manifest schema must be {ADAPTER_SCHEMA}")
    contract = _runtime_contract(manifest)
    base = _section(manifest, "base_model")
    adapter = _section(manifest, "adapter")
    runtime = _section(manifest, "runtime")
    if base.get("model_id") != EXPECTED_MODEL_ID or base.get("revision") != EXPECTED_REVISION:
        raise HarnessError("academic adapter is not for the pinned Qwen3.8 checkpoint")
    if base.get("model_type") != EXPECTED_MODEL_TYPE or base.get("architecture") != EXPECTED_ARCHITECTURE:
        raise HarnessError("academic adapter is not for the audited Qwen3.8/qwen3_5 architecture")
    quantization = base.get("quantization")
    if not isinstance(quantization, Mapping) or (
        quantization.get("bits") != 4
        or quantization.get("group_size") != 64
        or quantization.get("mode", "affine") != "affine"
    ):
        raise HarnessError("academic adapter base must be 4-bit/group-64 MLX")
    if adapter.get("format") != "mlx_lm_lora" or adapter.get("sha256_semantics") != ADAPTER_DIGEST_SEMANTICS:
        raise HarnessError("academic adapter bundle format/digest semantics are incompatible")
    _validate_runtime_contract(runtime, contract)
    # The published artifact pins only the default. Request strength is dynamic
    # and echoed separately from this immutable base/runtime identity.
    manifest_default = canonical_adapter_strength(runtime.get(
        "default_adapter_strength", DEFAULT_ADAPTER_STRENGTH))
    if manifest_default != DEFAULT_ADAPTER_STRENGTH:
        raise HarnessError("academic adapter manifest default strength must be 1.0")
    canonical_adapter_strength(os.environ.get(
        contract.strength_env, DEFAULT_ADAPTER_STRENGTH))
    adapter_dir = _manifest_relative(manifest_path, adapter.get("path"), "adapter.path")
    digest, _ = _strict_adapter_digest(adapter_dir, adapter.get("required_files"))
    if digest != adapter.get("sha256"):
        raise HarnessError("adapter tree SHA-256 does not match manifest")
    _validate_training_identity_receipt(manifest)
    _validate_model_view(model_view, base)
    manifest_sha = sha256_bytes(manifest_raw)
    identity = {
        "schema_version": IDENTITY_SCHEMA,
        "manifest_sha256": manifest_sha,
        "adapter_tree_sha256": digest,
        "base_model_id": str(base.get("model_id") or ""),
        "base_model_revision": str(base.get("revision") or ""),
        "base_weight_inventory_sha256": str(base.get("weight_inventory_sha256") or ""),
        "profile_id": str(manifest["profile_id"]),
        "provider": str(runtime["provider"]),
        "model": str(runtime["model"]),
        "transport_adapter": str(runtime["transport_adapter"]),
        **SERVER_LIFECYCLE_IDENTITY,
    }
    if contract.scope is not None:
        identity.update({
            "scope": contract.scope,
            "spiralchat_eligible": contract.spiralchat_eligible,
        })
    if any(value is None or value == "" for value in identity.values()):
        raise HarnessError("academic runtime identity contains an empty field")
    return RuntimeAssets(
        manifest_path=manifest_path, manifest_sha256=manifest_sha, manifest=manifest,
        adapter_dir=adapter_dir, model_view=model_view, identity=identity)


def validate_runtime_storage_sentinel(assets: RuntimeAssets) -> None:
    """Cheap live-presence check for the sidecar identity endpoint.

    Full inference admission rehashes every bound byte. Health only needs to detect
    that the already-attested deployment disappeared (notably after a removable-volume
    unmount) without rereading multi-gigabyte model shards. This function executes in
    the optional sidecar, so a stalled volume read remains bounded by the core host's
    loopback identity timeout.
    """

    try:
        manifest_raw = assets.manifest_path.read_bytes()
    except OSError as exc:
        raise HarnessError(
            f"academic adapter storage is unavailable: {assets.manifest_path}: {exc}"
        ) from exc
    if sha256_bytes(manifest_raw) != assets.manifest_sha256:
        raise HarnessError("academic adapter manifest changed after server startup")

    def require_file(path: Path, label: str, size: Any = None) -> None:
        try:
            present = path.is_file()
            observed_size = path.stat().st_size if present else None
        except OSError as exc:
            raise HarnessError(f"cannot inspect {label} {path}: {exc}") from exc
        if not present or (isinstance(size, int) and observed_size != size):
            raise HarnessError(f"{label} is missing or changed: {path}")

    adapter = _section(assets.manifest, "adapter")
    for row in adapter.get("required_files") or ():
        if not isinstance(row, Mapping):
            raise HarnessError("adapter storage receipt is invalid")
        relative = str(row.get("path") or "")
        if Path(relative).name != relative:
            raise HarnessError("adapter storage receipt has an unsafe path")
        require_file(
            assets.adapter_dir / relative, f"academic adapter file {relative}",
            row.get("size_bytes"),
        )

    base = _section(assets.manifest, "base_model")
    for row in base.get("weight_files") or ():
        if not isinstance(row, Mapping):
            raise HarnessError("base-weight storage receipt is invalid")
        relative = str(row.get("path") or "")
        if Path(relative).name != relative:
            raise HarnessError("base-weight storage receipt has an unsafe path")
        require_file(
            assets.model_view / relative, f"academic base shard {relative}",
            row.get("size_bytes"),
        )
    for relative in (
        "spiral_model_view_receipt.json", "spiral_qwen35_text_tuner.py", "config.json",
    ):
        require_file(assets.model_view / relative, f"academic model-view file {relative}")



def derive_model_view_path(manifest_path: Path, cache_root: Path) -> Path:
    """Derive the content-addressed view path emitted by create_text_training_view."""

    _, manifest = _read_object(manifest_path.expanduser().resolve(), "academic adapter manifest")
    base = _section(manifest, "base_model")
    view_identity = sha256_bytes(canonical_json({
        "revision": base.get("revision"),
        "config_sha256": base.get("config_sha256"),
        "weight_inventory_sha256": base.get("weight_inventory_sha256"),
        "source_sha256": sha256_bytes(MODEL_VIEW_SOURCE.encode("utf-8")),
    }))
    return cache_root.expanduser().resolve() / ".model-views" / view_identity[:16]


def validate_runtime_environment(assets: RuntimeAssets, python_executable: str) -> dict[str, str | None]:
    actual = python_package_versions(python_executable)
    expected = _section(assets.manifest, "environment")
    for key in ("mlx", "mlx_lm"):
        if not expected.get(key) or actual.get(key) != expected.get(key):
            raise HarnessError(
                f"selected runtime {key} {actual.get(key)!r} does not match adapter receipt {expected.get(key)!r}")
    return actual


def validate_bind_address(host: str, port: int, runtime_base_url: str) -> None:
    if host not in {"127.0.0.1", "::1"}:
        raise HarnessError("academic adapter server is loopback-only")
    if not 1 <= port <= 65535:
        raise HarnessError("server port must be in [1, 65535]")
    parsed = urllib.parse.urlparse(runtime_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise HarnessError("manifest runtime.base_url must be loopback HTTP")
    if parsed.port != port:
        raise HarnessError("server port must exactly match manifest runtime.base_url")
    if parsed.path.rstrip("/") != "/v1":
        raise HarnessError("manifest runtime.base_url must end in /v1")


def validate_chat_request(
    value: Any, *, expected_model: str, runtime_scope: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RequestError("request body must be a JSON object")
    if value.get("model") != expected_model:
        raise RequestError("model does not match the manifest-bound academic runtime")
    if value.get("stream", False) is not False:
        raise RequestError("streaming is not supported by this unload-after-response runtime")
    requested_strength = canonical_adapter_strength(
        value.get("adapter_strength", DEFAULT_ADAPTER_STRENGTH), request=True)
    planner_only = runtime_scope == STRUCTURE_RUNTIME_SCOPE
    response_format = value.get("response_format")
    if value.get("tools"):
        raise RequestError("academic runtime does not accept tools")
    if planner_only:
        if response_format != {"type": "json_object"}:
            raise RequestError(
                "academic structure runtime requires response_format json_object"
            )
    elif response_format:
        raise RequestError("academic prose runtime accepts text chat only")
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise RequestError(f"messages must contain 1–{MAX_MESSAGES} entries")
    clean_messages: list[dict[str, str]] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in {"system", "user", "assistant"}:
            raise RequestError("each message needs a system/user/assistant role")
        content = message.get("content")
        if not isinstance(content, str):
            raise RequestError("multimodal or structured message content is not supported")
        total_chars += len(content)
        clean_messages.append({"role": str(message["role"]), "content": content})
    model_messages = clean_messages
    if planner_only:
        if (
            len(clean_messages) != 2
            or clean_messages[0]["role"] != "system"
            or STRUCTURE_REQUEST_SYSTEM_MARKER not in clean_messages[0]["content"]
            or clean_messages[1]["role"] != "user"
        ):
            raise RequestError(
                "academic structure runtime accepts one authorized planner "
                "outline/budget request only"
            )
        try:
            parse_brief_to_blueprint_prompt(clean_messages[1]["content"])
        except StructurePromptError as exc:
            raise RequestError(
                f"academic structure runtime prompt contract mismatch: {exc}"
            ) from exc
        # The first message is admission metadata, not learned task context.
        # Stage-two completion rows were chat-templated as exactly one user
        # message containing the canonical brief_to_blueprint envelope.
        model_messages = [dict(clean_messages[1])]
    if total_chars > MAX_MESSAGE_CHARS:
        raise RequestError(f"message content exceeds {MAX_MESSAGE_CHARS:,} characters")
    max_tokens = value.get("max_completion_tokens", value.get("max_tokens", 1024))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_COMPLETION_TOKENS:
        raise RequestError(f"max_tokens must be an integer in [1, {MAX_COMPLETION_TOKENS}]")
    temperature = value.get("temperature", 0.0)
    top_p = value.get("top_p", 1.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(float(temperature)) or not 0 <= temperature <= 2:
        raise RequestError("temperature must be finite and in [0, 2]")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise RequestError("top_p must be finite and in (0, 1]")
    stop = value.get("stop")
    if stop is None:
        stops: list[str] = []
    elif isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, list) and len(stop) <= 8 and all(isinstance(item, str) for item in stop):
        stops = list(stop)
    else:
        raise RequestError("stop must be a string or up to eight strings")
    if any(not item or len(item) > 128 for item in stops):
        raise RequestError("stop strings must be nonempty and at most 128 characters")
    seed = value.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise RequestError("seed must be an integer")
    clean = {
        "model": expected_model,
        "messages": clean_messages,
        "model_messages": model_messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stop": stops,
        "seed": seed,
        "adapter_strength": requested_strength,
    }
    if planner_only:
        clean["response_format"] = {"type": "json_object"}
    return clean


def identity_health_response(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object": "spiral.runtime.identity",
        "spiral_runtime_identity": dict(identity),
    }


def openai_completion_response(
    result: Mapping[str, Any], identity: Mapping[str, Any], *,
    completion_id: str | None = None, created: int | None = None,
) -> dict[str, Any]:
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    completion_tokens = int(result.get("completion_tokens") or 0)
    return {
        "id": completion_id or ("chatcmpl-spiral-" + uuid.uuid4().hex),
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": str(identity["model"]),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": str(result.get("text") or "")},
            "finish_reason": str(result.get("finish_reason") or "stop"),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "spiral_runtime_identity": dict(identity),
        "spiral_adapter_strength": canonical_adapter_strength(
            result.get("spiral_adapter_strength", DEFAULT_ADAPTER_STRENGTH)),
    }


def apply_adapter_strength(
    model: Any, strength: Any, *, lora_types: tuple[type, ...] | None = None,
) -> dict[str, Any]:
    """Multiply every loaded MLX LoRA module's trained scale in memory.

    This is the actual inference control, not metadata: a trained scale of 32
    becomes ``32 * strength`` for the forward pass. The adapter files are never
    edited, so a later request always starts from the authenticated trained scale.
    """

    selected = canonical_adapter_strength(strength, request=True)
    if lora_types is None:  # imported only in the child that already owns MLX
        from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

        lora_types = (LoRALinear, LoRASwitchLinear, LoRAEmbedding)
    trained_scales: list[float] = []
    for _, module in model.named_modules():
        if not isinstance(module, lora_types):
            continue
        trained_scale = float(module.scale)
        module.scale = trained_scale * selected
        trained_scales.append(trained_scale)
    if not trained_scales:
        raise HarnessError("loaded academic adapter contains no LoRA modules")
    return {
        "adapter_strength": selected,
        "lora_module_count": len(trained_scales),
        "trained_scales": sorted(set(trained_scales)),
        "effective_scales": sorted({scale * selected for scale in trained_scales}),
    }


def _worker_generate(
    *, manifest_path: Path, model_view: Path, request_path: Path, output_path: Path,
) -> None:
    assets = validate_runtime_assets(manifest_path, model_view)
    validate_runtime_environment(assets, sys.executable)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    clean = validate_chat_request(
        request,
        expected_model=assets.identity["model"],
        runtime_scope=assets.identity.get("scope"),
    )
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:  # pragma: no cover - live pinned-runtime path
        raise HarnessError(f"selected server Python cannot import MLX-LM: {exc}") from exc
    derived_seed = int.from_bytes(
        hashlib.sha256(
            assets.manifest_sha256.encode("ascii") + canonical_json(clean)
        ).digest()[:4], "big")
    seed = int(clean["seed"] if clean["seed"] is not None else derived_seed) & 0xFFFFFFFF
    mx.random.seed(seed)
    model, tokenizer = load(
        str(assets.model_view), adapter_path=str(assets.adapter_dir),
        tokenizer_config={"trust_remote_code": True})
    strength_receipt = apply_adapter_strength(model, clean["adapter_strength"])
    try:
        prompt = tokenizer.apply_chat_template(
            clean["model_messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            clean["model_messages"], tokenize=False, add_generation_prompt=True)
    sampler = make_sampler(temp=clean["temperature"], top_p=clean["top_p"])
    segments: list[str] = []
    final = None
    for response in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=clean["max_tokens"], sampler=sampler):
        segments.append(response.text)
        final = response
    text = "".join(segments)
    finish_reason = getattr(final, "finish_reason", None) or "stop"
    for stop in clean["stop"]:
        if stop in text:
            text = text.split(stop, 1)[0]
            finish_reason = "stop"
    result = {
        "text": text,
        "prompt_tokens": int(getattr(final, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(final, "generation_tokens", 0) or 0),
        "finish_reason": finish_reason,
        "seed": seed,
        "spiral_adapter_strength": strength_receipt["adapter_strength"],
        "lora_module_count": strength_receipt["lora_module_count"],
        "spiral_runtime_identity": assets.identity,
    }
    atomic_write_json(output_path, result)
    del model, tokenizer
    mx.clear_cache()


class AcademicAdapterService:
    def __init__(
        self, *, manifest_path: Path, model_view: Path, python_executable: str,
        lease_path: Path, ollama_url: str, request_timeout: float,
    ) -> None:
        self.manifest_path = manifest_path
        self.model_view = model_view
        self.python_executable = python_executable
        self.lease_path = lease_path
        self.ollama_url = ollama_url
        self.request_timeout = request_timeout
        self._request_lock = threading.Lock()
        self.assets = validate_runtime_assets(manifest_path, model_view)
        validate_runtime_environment(self.assets, python_executable)

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self.assets.identity)

    def readiness_identity(self) -> dict[str, Any]:
        validate_runtime_storage_sentinel(self.assets)
        return self.identity

    def complete(self, request: Any) -> dict[str, Any]:
        clean = validate_chat_request(
            request,
            expected_model=self.identity["model"],
            runtime_scope=self.identity.get("scope"),
        )
        if not self._request_lock.acquire(blocking=False):
            raise RequestError(
                "academic runtime is serving another completion", status=503,
                code="server_busy")
        try:
            # Rehash every manifest-bound artifact immediately before model load.
            assets = validate_runtime_assets(self.manifest_path, self.model_view)
            validate_runtime_environment(assets, self.python_executable)
            if assets.identity != self.assets.identity:
                raise HarnessError("academic runtime assets changed after server startup")
            lease = TrainingComputeLease(self.lease_path)
            lease.acquire(owner={
                "operation": "academic_adapter_inference",
                "model": assets.identity["model"],
                "manifest_sha256": assets.manifest_sha256,
                "adapter_strength": clean["adapter_strength"],
            })
            with lease, tempfile.TemporaryDirectory(prefix="spiral-academic-request-") as raw_tmp:
                verify_ollama_empty(self.ollama_url)
                scratch = Path(raw_tmp)
                request_path = scratch / "request.json"
                output_path = scratch / "response.json"
                atomic_write_json(request_path, clean)
                command = [
                    self.python_executable, str(Path(__file__).resolve()), "_worker",
                    "--manifest", str(self.manifest_path), "--model-view", str(self.model_view),
                    "--request", str(request_path), "--output", str(output_path),
                ]
                completed = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=self.request_timeout,
                    pass_fds=(lease.descriptor,) if lease.descriptor is not None else (),
                    check=False,
                )
                if completed.returncode:
                    raise HarnessError(
                        "academic MLX worker failed before publishing a completion: "
                        + completed.stdout.strip()[-1200:])
                _, result = _read_object(output_path, "academic worker response")
            if result.get("spiral_runtime_identity") != assets.identity:
                raise HarnessError("academic worker runtime attestation does not match loaded assets")
            if result.get("spiral_adapter_strength") != clean["adapter_strength"]:
                raise HarnessError(
                    "academic worker adapter-strength attestation does not match the request")
            return openai_completion_response(result, assets.identity)
        except subprocess.TimeoutExpired as exc:
            raise RequestError(
                "academic MLX worker exceeded its bounded request timeout",
                status=504, code="timeout") from exc
        finally:
            self._request_lock.release()


class AcademicHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], service: AcademicAdapterService):
        super().__init__(address, AcademicRequestHandler)
        self.service = service


class AcademicRequestHandler(BaseHTTPRequestHandler):
    server: AcademicHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("academic-server · " + (fmt % args) + "\n")

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = canonical_json(dict(value))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, exc: BaseException) -> None:
        if isinstance(exc, RequestError):
            status, code = exc.status, exc.code
        elif isinstance(exc, HarnessError):
            status, code = 503, "runtime_unavailable"
        else:
            status, code = 500, "internal_error"
        self._json(status, {"error": {"message": str(exc), "type": code, "code": code}})

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/v1/spiral/identity":
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            self._json(
                200,
                identity_health_response(self.server.service.readiness_identity()),
            )
        except Exception as exc:  # noqa: BLE001 - serialize handler failures
            self._error(exc)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length_raw = self.headers.get("Content-Length")
            if length_raw is None:
                raise RequestError("Content-Length is required", status=411)
            length = int(length_raw)
            if not 1 <= length <= MAX_REQUEST_BYTES:
                raise RequestError(
                    f"request body must be 1–{MAX_REQUEST_BYTES} bytes", status=413)
            raw = self.rfile.read(length)
            try:
                request = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError("request body is not valid UTF-8 JSON") from exc
            self._json(200, self.server.service.complete(request))
        except Exception as exc:  # noqa: BLE001 - serialize handler failures
            self._error(exc)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Serve an exact Spiral academic MLX adapter without resident weights")
    subparsers = result.add_subparsers(dest="command")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--model-view", type=Path, required=True)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--model-view", type=Path)
    result.add_argument(
        "--model-view-cache", type=Path,
        default=Path("~/Library/Caches/SpiralAcademic").expanduser(),
        help="content-addressed view cache; used when --model-view is omitted")
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8080)
    result.add_argument(
        "--lease-path", type=Path,
        default=Path("~/.spiralchat/spiral-compute.lease").expanduser())
    result.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    result.add_argument("--request-timeout", type=float, default=1800.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "_worker":
            _worker_generate(
                manifest_path=arguments.manifest, model_view=arguments.model_view,
                request_path=arguments.request, output_path=arguments.output)
            return 0
        if arguments.manifest is None:
            raise HarnessError("--manifest is required")
        model_view = arguments.model_view or derive_model_view_path(
            arguments.manifest, arguments.model_view_cache)
        if not math.isfinite(arguments.request_timeout) or not 1 <= arguments.request_timeout <= 7200:
            raise HarnessError("--request-timeout must be between 1 and 7200 seconds")
        assets = validate_runtime_assets(arguments.manifest, model_view)
        validate_bind_address(
            arguments.host, arguments.port,
            str(_section(assets.manifest, "runtime").get("base_url") or ""))
        service = AcademicAdapterService(
            manifest_path=arguments.manifest, model_view=model_view,
            python_executable=arguments.python, lease_path=arguments.lease_path,
            ollama_url=arguments.ollama_url, request_timeout=arguments.request_timeout)
        server = AcademicHTTPServer((arguments.host, arguments.port), service)
        print(json.dumps({
            "listening": f"http://{arguments.host}:{arguments.port}/v1",
            "identity": service.identity,
            "weight_residency": "child-process-per-request; unloaded before lease release",
        }, sort_keys=True), flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
