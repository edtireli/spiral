#!/usr/bin/env python3
"""Authenticated, unload-after-request MLX-VLM endpoint for SpiralChat.

The trained adapter was produced by ``mlx_lm`` against Qwen3.8's language
submodule.  This server applies those same weights to the corresponding fully
qualified modules in the complete vision-language checkpoint.  Each request
therefore retains Qwen's image and tool capabilities while selecting a real
LoRA contribution multiplier.

The public surface is Ollama-compatible (``POST /api/chat``) because this lane
is consumed by the Spiral host, not by the paper writer's OpenAI-compatible
provider.  The HTTP parent never imports MLX or keeps weights resident.  A
worker process loads, generates, and exits for every request.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import hmac
import ipaddress
import itertools
import json
import math
import os
import re
import select
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    from .serve_adapter import (
        ADAPTER_DIGEST_SEMANTICS,
        ADAPTER_STRENGTH_STEP,
        DEFAULT_ADAPTER_STRENGTH,
        MAX_ADAPTER_STRENGTH,
        MIN_ADAPTER_STRENGTH,
        RequestError,
        _manifest_relative,
        _read_object,
        _section,
        _strict_adapter_digest,
        _validate_training_identity_receipt,
        canonical_adapter_strength,
    )
    from .training_support import (
        ADAPTER_SCHEMA,
        EXPECTED_ARCHITECTURE,
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_TYPE,
        EXPECTED_REVISION,
        PROFILE_ID,
        PROMPT_CONTRACT,
        HarnessError,
        TrainingComputeLease,
        canonical_json,
        sha256_bytes,
        sha256_file,
        verify_ollama_empty,
    )
except ImportError:  # direct script execution
    from serve_adapter import (  # type: ignore
        ADAPTER_DIGEST_SEMANTICS,
        ADAPTER_STRENGTH_STEP,
        DEFAULT_ADAPTER_STRENGTH,
        MAX_ADAPTER_STRENGTH,
        MIN_ADAPTER_STRENGTH,
        RequestError,
        _manifest_relative,
        _read_object,
        _section,
        _strict_adapter_digest,
        _validate_training_identity_receipt,
        canonical_adapter_strength,
    )
    from training_support import (  # type: ignore
        ADAPTER_SCHEMA,
        EXPECTED_ARCHITECTURE,
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_TYPE,
        EXPECTED_REVISION,
        PROFILE_ID,
        PROMPT_CONTRACT,
        HarnessError,
        TrainingComputeLease,
        canonical_json,
        sha256_bytes,
        sha256_file,
        verify_ollama_empty,
    )


IDENTITY_SCHEMA = "spiral.academic-vlm-runtime-identity.v1"
EXPECTED_PROVIDER = "mlx_vlm"
EXPECTED_RUNTIME_MODEL = "qwen3.8:27b"
EXPECTED_TRANSPORT = "ollama-compatible"
EXPECTED_MLX_VERSION = "0.32.1"
EXPECTED_MLX_VLM_VERSION = "0.6.16"
EXPECTED_TRANSFORMERS_VERSION = "5.15.1"
EXPECTED_LORA_MODULE_COUNT = 20
VLM_FRONTEND_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
DEFAULT_VLM_PYTHON = Path(
    "~/Library/Application Support/SpiralAcademic/vlm-runtime/bin/python3"
).expanduser()
DEFAULT_MODEL_ROOT = Path("/Volumes/T5_EVO_EDT/qwen38-mlx")
LEASE_AUTHORITY_HEADER = "X-Spiral-Lease-Authority"
LEASE_HANDOFF_CONTRACT = "spiral-host-held-flock-hmac-token-v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 100_000
MAX_IMAGES = 10
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 48 * 1024 * 1024
MAX_COMPLETION_TOKENS = 32768
MAX_TOOLS = 128
MAX_TOOL_SCHEMA_CHARS = 100_000
IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"RIFF": ".webp",
}
SERVER_LIFECYCLE_IDENTITY = {
    "server_contract": "spiral.academic-vlm-one-request-server.v1",
    "weight_residency": "child-process-per-request",
    "compute_lease": "spiral-compute-flock-v1",
    "lease_handoff": LEASE_HANDOFF_CONTRACT,
    "ollama_admission": "strict-empty-no-eviction",
    "unload_boundary": "child-exit-before-lease-release",
    "adapter_strength_supported": True,
    "adapter_strength_min": MIN_ADAPTER_STRENGTH,
    "adapter_strength_max": MAX_ADAPTER_STRENGTH,
    "adapter_strength_step": ADAPTER_STRENGTH_STEP,
    "adapter_strength_default": DEFAULT_ADAPTER_STRENGTH,
    "vision_supported": True,
    "tools_supported": True,
    "streaming_supported": True,
    "thinking_supported": True,
    "mlx_version": EXPECTED_MLX_VERSION,
    "mlx_vlm_version": EXPECTED_MLX_VLM_VERSION,
    "transformers_version": EXPECTED_TRANSFORMERS_VERSION,
}


@dataclass(frozen=True)
class VlmRuntimeAssets:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    adapter_dir: Path
    model_root: Path
    identity: dict[str, Any]
    base_file_snapshot: tuple[dict[str, Any], ...]


def _regular_file(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HarnessError(f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HarnessError(f"{label} must be a real regular file")
    return path


def _file_snapshot(
    path: Path, relative: str, *, digest: str | None = None,
) -> dict[str, Any]:
    metadata = path.stat()
    result = {
        "path": relative,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }
    if digest is not None:
        result["sha256"] = digest
    return result


def _validate_full_vlm_model(
    model_root: Path, base: Mapping[str, Any], *,
    expected_snapshot: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    if model_root.is_symlink() or not model_root.is_dir():
        raise HarnessError("VLM model root must be a real local directory")
    root = model_root.resolve()
    config_path = _regular_file(root / "config.json", "VLM config")
    index_path = _regular_file(
        root / "model.safetensors.index.json", "VLM weight index")
    if sha256_file(config_path) != base.get("config_sha256"):
        raise HarnessError("full VLM config does not match the trained base identity")
    if sha256_file(index_path) != base.get("weight_index_sha256"):
        raise HarnessError("full VLM weight index does not match the trained base identity")

    _, config = _read_object(config_path, "VLM config")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise HarnessError("full VLM checkpoint has the wrong model_type")
    if EXPECTED_ARCHITECTURE not in (config.get("architectures") or []):
        raise HarnessError("full VLM checkpoint has the wrong architecture")
    if not isinstance(config.get("vision_config"), Mapping) or not config["vision_config"]:
        raise HarnessError("full checkpoint does not expose a vision configuration")
    quantization = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quantization, Mapping) or (
        quantization.get("bits") != 4
        or quantization.get("group_size") != 64
        or quantization.get("mode", "affine") != "affine"
    ):
        raise HarnessError("full VLM checkpoint is not affine q4/group-64")

    expected_weights = base.get("weight_files")
    if not isinstance(expected_weights, list) or not expected_weights:
        raise HarnessError("adapter manifest has no full base-weight inventory")
    lines: list[str] = []
    expected_names: set[str] = set()
    observed_snapshot: list[dict[str, Any]] = []
    snapshot_by_path = {
        str(row.get("path") or ""): dict(row)
        for row in (expected_snapshot or ()) if isinstance(row, Mapping)
    }

    def attest_file(path: Path, relative: str, expected_digest: str | None = None) -> str:
        observed = _file_snapshot(path, relative)
        if expected_snapshot is None:
            digest = sha256_file(path)
            if expected_digest is not None and digest != expected_digest:
                raise HarnessError(f"VLM file changed after training: {relative}")
        else:
            startup = snapshot_by_path.get(relative)
            startup_metadata = {
                key: value for key, value in (startup or {}).items() if key != "sha256"}
            if startup_metadata != observed:
                raise HarnessError(
                    f"VLM file metadata changed after startup attestation: {relative}")
            digest = str((startup or {}).get("sha256") or "")
            if len(digest) != 64 or (expected_digest is not None and digest != expected_digest):
                raise HarnessError(f"VLM startup receipt has the wrong digest: {relative}")
        observed_snapshot.append(_file_snapshot(path, relative, digest=digest))
        return digest

    for row in expected_weights:
        if not isinstance(row, Mapping):
            raise HarnessError("invalid base-weight inventory entry")
        relative = str(row.get("path") or "")
        if not relative or Path(relative).name != relative:
            raise HarnessError("base-weight inventory contains an unsafe path")
        shard = _regular_file(root / relative, f"VLM base shard {relative}")
        if shard.stat().st_size != row.get("size_bytes"):
            raise HarnessError(f"full VLM base shard changed after training: {relative}")
        attest_file(shard, relative, str(row.get("sha256") or ""))
        lines.append(
            f"{relative}\0{row.get('size_bytes')}\0{row.get('sha256')}\n")
        expected_names.add(relative)
    if sha256_bytes("".join(lines).encode("utf-8")) != base.get(
        "weight_inventory_sha256"
    ):
        raise HarnessError("full VLM base-weight inventory digest is incompatible")
    _, index = _read_object(index_path, "VLM weight index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or set(weight_map.values()) != expected_names:
        raise HarnessError("VLM index does not reference the exact manifest shard set")

    template_path = _regular_file(root / "chat_template.jinja", "VLM chat template")
    template = template_path.read_text(encoding="utf-8")
    if "<tool_call>" not in template or "<function=" not in template:
        raise HarnessError("VLM chat template does not support the verified Qwen tool grammar")
    frontend_lines: list[str] = []
    frontend_hashes: dict[str, str] = {}
    for relative in VLM_FRONTEND_FILES:
        path = _regular_file(root / relative, f"VLM frontend file {relative}")
        digest = attest_file(path, relative)
        frontend_hashes[relative] = digest
        frontend_lines.append(f"{relative}\0{path.stat().st_size}\0{digest}\n")
    expected_names.update(VLM_FRONTEND_FILES)
    if expected_snapshot is not None and set(snapshot_by_path) != expected_names:
        raise HarnessError("startup base-file snapshot names the wrong VLM file set")
    return {
        "vlm_config_sha256": sha256_file(config_path),
        "vlm_weight_index_sha256": sha256_file(index_path),
        "vlm_chat_template_sha256": frontend_hashes["chat_template.jinja"],
        "vlm_frontend_inventory_sha256": sha256_bytes(
            "".join(sorted(frontend_lines)).encode("utf-8")),
    }, tuple(observed_snapshot)


def validate_vlm_runtime_assets(
    manifest_path: Path, model_root: Path, *,
    expected_base_snapshot: Sequence[Mapping[str, Any]] | None = None,
) -> VlmRuntimeAssets:
    manifest_path = manifest_path.expanduser().resolve()
    model_root = model_root.expanduser().resolve()
    manifest_raw, manifest = _read_object(manifest_path, "academic adapter manifest")
    if manifest.get("schema_version") != ADAPTER_SCHEMA:
        raise HarnessError(f"academic adapter manifest schema must be {ADAPTER_SCHEMA}")
    if manifest.get("profile_id") != PROFILE_ID or manifest.get(
        "prompt_contract"
    ) != PROMPT_CONTRACT:
        raise HarnessError("academic adapter profile/prompt contract is incompatible")
    base = _section(manifest, "base_model")
    adapter = _section(manifest, "adapter")
    runtime = _section(manifest, "runtime")
    if base.get("model_id") != EXPECTED_MODEL_ID or base.get(
        "revision"
    ) != EXPECTED_REVISION:
        raise HarnessError("academic adapter is not for the pinned Qwen3.8 checkpoint")
    if base.get("model_type") != EXPECTED_MODEL_TYPE or base.get(
        "architecture"
    ) != EXPECTED_ARCHITECTURE:
        raise HarnessError("academic adapter has the wrong Qwen3.8 architecture")
    quantization = base.get("quantization")
    if not isinstance(quantization, Mapping) or (
        quantization.get("bits") != 4
        or quantization.get("group_size") != 64
        or quantization.get("mode", "affine") != "affine"
    ):
        raise HarnessError("academic adapter base must be affine q4/group-64")
    if adapter.get("format") != "mlx_lm_lora" or adapter.get(
        "sha256_semantics"
    ) != ADAPTER_DIGEST_SEMANTICS:
        raise HarnessError("academic adapter bundle format/digest is incompatible")
    # The VLM transport is deliberately derived from, rather than substituted
    # into, the manifest-bound text runtime.
    if runtime.get("provider") != "mlx_lm" or runtime.get(
        "model"
    ) != "qwen3.8-27b-academic" or runtime.get(
        "transport_adapter"
    ) != "openai-compatible":
        raise HarnessError("source academic runtime identity is incompatible")
    if canonical_adapter_strength(
        runtime.get("default_adapter_strength", DEFAULT_ADAPTER_STRENGTH)
    ) != DEFAULT_ADAPTER_STRENGTH:
        raise HarnessError("academic adapter manifest default strength must be 1.0")

    adapter_dir = _manifest_relative(manifest_path, adapter.get("path"), "adapter.path")
    adapter_digest, _ = _strict_adapter_digest(
        adapter_dir, adapter.get("required_files"))
    if adapter_digest != adapter.get("sha256"):
        raise HarnessError("adapter tree SHA-256 does not match manifest")
    _validate_training_identity_receipt(manifest)
    vlm_hashes, base_file_snapshot = _validate_full_vlm_model(
        model_root, base, expected_snapshot=expected_base_snapshot)
    manifest_sha = sha256_bytes(manifest_raw)
    identity = {
        "schema_version": IDENTITY_SCHEMA,
        "manifest_sha256": manifest_sha,
        "adapter_tree_sha256": adapter_digest,
        "base_model_id": str(base.get("model_id") or ""),
        "base_model_revision": str(base.get("revision") or ""),
        "base_weight_inventory_sha256": str(
            base.get("weight_inventory_sha256") or ""),
        "profile_id": str(manifest.get("profile_id") or ""),
        "provider": EXPECTED_PROVIDER,
        "model": EXPECTED_RUNTIME_MODEL,
        "transport_adapter": EXPECTED_TRANSPORT,
        **vlm_hashes,
        **SERVER_LIFECYCLE_IDENTITY,
    }
    if any(value is None or value == "" for value in identity.values()):
        raise HarnessError("academic VLM identity contains an empty field")
    return VlmRuntimeAssets(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        adapter_dir=adapter_dir,
        model_root=model_root,
        identity=identity,
        base_file_snapshot=base_file_snapshot,
    )


def validate_vlm_storage_sentinel(assets: VlmRuntimeAssets) -> None:
    """Detect a vanished live VLM deployment without rehashing its large shards."""

    try:
        manifest_raw = assets.manifest_path.read_bytes()
    except OSError as exc:
        raise HarnessError(
            f"academic VLM storage is unavailable: {assets.manifest_path}: {exc}"
        ) from exc
    if sha256_bytes(manifest_raw) != assets.manifest_sha256:
        raise HarnessError("academic VLM manifest changed after server startup")

    adapter = _section(assets.manifest, "adapter")
    for row in adapter.get("required_files") or ():
        if not isinstance(row, Mapping):
            raise HarnessError("academic VLM adapter storage receipt is invalid")
        relative = str(row.get("path") or "")
        if Path(relative).name != relative:
            raise HarnessError("academic VLM adapter storage receipt has an unsafe path")
        path = assets.adapter_dir / relative
        try:
            metadata = path.stat()
        except OSError as exc:
            raise HarnessError(
                f"academic VLM adapter file is unavailable: {path}: {exc}"
            ) from exc
        if not path.is_file() or metadata.st_size != row.get("size_bytes"):
            raise HarnessError(f"academic VLM adapter file is missing or changed: {path}")

    for startup in assets.base_file_snapshot:
        relative = str(startup.get("path") or "")
        if not relative or Path(relative).name != relative:
            raise HarnessError("academic VLM startup snapshot has an unsafe path")
        try:
            observed = _file_snapshot(assets.model_root / relative, relative)
        except OSError as exc:
            raise HarnessError(
                f"academic VLM base file is unavailable: {relative}: {exc}"
            ) from exc
        expected = {key: value for key, value in startup.items() if key != "sha256"}
        if observed != expected:
            raise HarnessError(
                f"academic VLM base file changed after server startup: {relative}")


def validate_vlm_environment(python_executable: str) -> dict[str, str]:
    probe = (
        "import importlib.metadata,json;"
        "print(json.dumps({'mlx':importlib.metadata.version('mlx'),"
        "'mlx_vlm':importlib.metadata.version('mlx-vlm'),"
        "'transformers':importlib.metadata.version('transformers')},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [python_executable, "-c", probe], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(f"cannot run selected MLX-VLM Python: {exc}") from exc
    if completed.returncode:
        raise HarnessError(
            "selected MLX-VLM Python failed its package probe: "
            + completed.stderr.strip()[:300])
    try:
        versions = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HarnessError("selected MLX-VLM Python returned invalid package metadata") from exc
    expected = {
        "mlx": EXPECTED_MLX_VERSION,
        "mlx_vlm": EXPECTED_MLX_VLM_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
    }
    if versions != expected:
        raise HarnessError(
            f"selected VLM runtime {versions!r} does not match pinned {expected!r}")
    return versions


def validate_bind_address(host: str, port: int) -> None:
    if host not in {"127.0.0.1", "::1"}:
        raise HarnessError("academic VLM server is loopback-only")
    if not 1 <= port <= 65535:
        raise HarnessError("server port must be in [1, 65535]")


def load_lease_authority_token(
    token_file: Path | None, *, allow_env_for_tests: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bytes | None:
    if token_file is None:
        if not allow_env_for_tests:
            return None
        raw_env = (environ or os.environ).get("SPIRAL_VLM_LEASE_AUTHORITY_TOKEN", "")
        return _validate_authority_token(raw_env.encode("ascii", "strict")) if raw_env else None
    path = token_file.expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HarnessError(f"cannot securely open lease-authority token: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise HarnessError("lease-authority token must be a real regular file")
        if metadata.st_uid != os.getuid():
            raise HarnessError("lease-authority token must be owned by the serving user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise HarnessError("lease-authority token permissions must be owner-only (0600)")
        if metadata.st_size > 512:
            raise HarnessError("lease-authority token file is too large")
        raw = os.read(descriptor, 513).strip()
        if len(raw) > 512:
            raise HarnessError("lease-authority token file is too large")
    finally:
        os.close(descriptor)
    return _validate_authority_token(raw)


def _validate_authority_token(raw: bytes) -> bytes:
    if not 32 <= len(raw) <= 512 or any(byte < 0x21 or byte > 0x7E for byte in raw):
        raise HarnessError("lease-authority token must be 32–512 printable ASCII bytes")
    return raw


def trusted_lease_handoff(header: str | None, configured: bytes | None) -> bool:
    if header is None:
        return False
    try:
        supplied = header.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise RequestError("invalid lease-authority handoff", status=403, code="forbidden") from exc
    if configured is None or not hmac.compare_digest(supplied, configured):
        raise RequestError("invalid lease-authority handoff", status=403, code="forbidden")
    return True


def _normalized_tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_TOOLS:
        raise RequestError(f"tools must be a list of at most {MAX_TOOLS} functions")
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise RequestError("tools must be JSON serializable") from exc
    if len(encoded) > MAX_TOOL_SCHEMA_CHARS:
        raise RequestError("tool schemas are too large")
    clean: list[dict[str, Any]] = []
    names: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping) or row.get("type", "function") != "function":
            raise RequestError("only function tools are supported")
        function = row.get("function")
        if not isinstance(function, Mapping):
            raise RequestError("each tool needs a function object")
        name = str(function.get("name") or "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", name) or name in names:
            raise RequestError("tool names must be unique safe identifiers")
        names.add(name)
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, Mapping):
            raise RequestError("tool parameters must be a JSON-schema object")
        clean.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(function.get("description") or "")[:4096],
                "parameters": dict(parameters),
            },
        })
    return clean


def _image_decoded_size(value: str) -> int:
    payload = value.split(",", 1)[1] if value.startswith("data:image/") and "," in value else value
    if not payload or len(payload) % 4 == 1:
        raise RequestError("message image is not valid base64")
    return (len(payload.rstrip("=")) * 3) // 4


def validate_vlm_chat_request(value: Any, *, expected_model: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RequestError("request body must be a JSON object")
    if value.get("model") != expected_model:
        raise RequestError("model does not match the manifest-bound VLM runtime")
    stream = value.get("stream", True)
    if not isinstance(stream, bool):
        raise RequestError("stream must be boolean")
    think = value.get("think", False)
    if not isinstance(think, bool):
        raise RequestError("think must be boolean")
    strength = canonical_adapter_strength(
        value.get("adapter_strength", DEFAULT_ADAPTER_STRENGTH), request=True)
    messages = value.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise RequestError(f"messages must contain 1–{MAX_MESSAGES} entries")
    clean_messages: list[dict[str, Any]] = []
    total_chars = total_images = total_image_bytes = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in {
            "system", "user", "assistant", "tool"
        }:
            raise RequestError("each message needs a system/user/assistant/tool role")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RequestError("Ollama message content must be text")
        total_chars += len(content)
        images = message.get("images", [])
        if images is None:
            images = []
        if not isinstance(images, list) or not all(isinstance(item, str) for item in images):
            raise RequestError("message images must be a list of base64 strings")
        if images and message.get("role") != "user":
            raise RequestError("images are accepted only on user messages")
        sizes = [_image_decoded_size(item) for item in images]
        if any(size <= 0 or size > MAX_IMAGE_BYTES for size in sizes):
            raise RequestError(f"each decoded image must be at most {MAX_IMAGE_BYTES} bytes")
        total_images += len(images)
        total_image_bytes += sum(sizes)
        clean: dict[str, Any] = {
            "role": str(message["role"]), "content": content, "images": list(images)}
        if "tool_calls" in message:
            if not isinstance(message["tool_calls"], list):
                raise RequestError("message tool_calls must be a list")
            clean["tool_calls"] = message["tool_calls"]
        if "tool_name" in message:
            clean["name"] = str(message["tool_name"] or "")
        clean_messages.append(clean)
    if total_chars > MAX_MESSAGE_CHARS:
        raise RequestError(f"message content exceeds {MAX_MESSAGE_CHARS:,} characters")
    if total_images > MAX_IMAGES or total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
        raise RequestError("request contains too many or too-large images")
    if not total_chars and not total_images:
        raise RequestError("request must contain text or an image")

    options = value.get("options") or {}
    if not isinstance(options, Mapping):
        raise RequestError("options must be an object")
    max_tokens = options.get("num_predict", value.get("max_tokens", 1024))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not (
        1 <= max_tokens <= MAX_COMPLETION_TOKENS
    ):
        raise RequestError(f"num_predict must be in [1, {MAX_COMPLETION_TOKENS}]")
    temperature = options.get("temperature", 0.0)
    top_p = options.get("top_p", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not (
        math.isfinite(float(temperature)) and 0 <= float(temperature) <= 2
    ):
        raise RequestError("temperature must be finite and in [0, 2]")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not (
        math.isfinite(float(top_p)) and 0 < float(top_p) <= 1
    ):
        raise RequestError("top_p must be finite and in (0, 1]")
    seed = options.get("seed", value.get("seed"))
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RequestError("seed must be an integer")
    stop = options.get("stop", [])
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > 8 or not all(
        isinstance(item, str) and 0 < len(item) <= 128 for item in stop
    ):
        raise RequestError("stop must contain up to eight nonempty strings")
    tools = _normalized_tools(value.get("tools"))
    return {
        "model": expected_model,
        "messages": clean_messages,
        "tools": tools,
        "stream": stream,
        "think": think,
        "adapter_strength": strength,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "stop": list(stop),
    }


def decode_request_images(messages: Sequence[Mapping[str, Any]], scratch: Path) -> list[str]:
    paths: list[str] = []
    total_bytes = 0
    for message in messages:
        for encoded in message.get("images", []):
            payload = encoded
            declared = ""
            if encoded.startswith("data:image/") and "," in encoded:
                header, payload = encoded.split(",", 1)
                if ";base64" not in header:
                    raise HarnessError("image data URL must use base64 encoding")
                declared = header[11:].split(";", 1)[0].lower()
            try:
                raw = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HarnessError("message image is not valid base64") from exc
            total_bytes += len(raw)
            if not raw or len(raw) > MAX_IMAGE_BYTES or total_bytes > MAX_TOTAL_IMAGE_BYTES:
                raise HarnessError("decoded image payload exceeds the bounded request limits")
            suffix = ""
            for magic, candidate in IMAGE_MAGIC.items():
                if raw.startswith(magic):
                    suffix = candidate
                    break
            if suffix == ".webp" and raw[8:12] != b"WEBP":
                suffix = ""
            if not suffix or (declared and declared not in {
                "png", "jpeg", "jpg", "webp"
            }):
                raise HarnessError("image must be a PNG, JPEG, or WebP")
            path = scratch / f"image-{len(paths):02d}{suffix}"
            path.write_bytes(raw)
            paths.append(str(path))
    return paths


def lora_parameters_for_strength(config: Mapping[str, Any], strength: Any) -> dict[str, Any]:
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, Mapping):
        raise HarnessError("adapter_config.json has no lora_parameters")
    rank = parameters.get("rank")
    scale = parameters.get("scale")
    dropout = parameters.get("dropout")
    try:
        valid = rank == 16 and float(scale) == 32.0 and float(dropout) == 0.0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise HarnessError("adapter config no longer attests trained rank16/scale32/dropout0")
    selected = canonical_adapter_strength(strength, request=True)
    return {"rank": rank, "scale": float(scale) * selected, "dropout": float(dropout)}


def vlm_processor_messages(clean_messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve media-to-turn association while normalizing Ollama tool history."""

    messages: list[dict[str, Any]] = []
    for source in clean_messages:
        text = str(source.get("content") or "")
        image_count = len(source.get("images") or [])
        content: Any = text
        if image_count:
            content = ([{"type": "text", "text": text}] if text else []) + [
                {"type": "image"} for _ in range(image_count)]
        message: dict[str, Any] = {"role": source["role"], "content": content}
        if source.get("tool_calls") is not None:
            calls = []
            for row in source["tool_calls"]:
                if not isinstance(row, Mapping):
                    continue
                call = dict(row)
                function = call.get("function")
                if isinstance(function, Mapping):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            function["arguments"] = {}
                    call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        if source.get("name"):
            message["name"] = source["name"]
        messages.append(message)
    return messages


def _emit_event(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(canonical_json(dict(value)).decode("utf-8"))
    stream.flush()


def _worker_generate(
    *, manifest_path: Path, model_root: Path, request_path: Path,
    startup_receipt_path: Path, events_fd: int,
) -> None:
    worker_started = time.monotonic()
    _, startup_receipt = _read_object(startup_receipt_path, "VLM startup receipt")
    expected_snapshot = startup_receipt.get("base_file_snapshot")
    if not isinstance(expected_snapshot, list):
        raise HarnessError("VLM startup receipt has no base-file snapshot")
    assets = validate_vlm_runtime_assets(
        manifest_path, model_root, expected_base_snapshot=expected_snapshot)
    if startup_receipt.get("spiral_runtime_identity") != assets.identity:
        raise HarnessError("VLM startup receipt runtime identity mismatch")
    validate_vlm_environment(sys.executable)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    clean = validate_vlm_chat_request(request, expected_model=assets.identity["model"])
    scratch = request_path.parent / "images"
    scratch.mkdir(mode=0o700)
    image_paths = decode_request_images(clean["messages"], scratch)
    attestation_seconds = time.monotonic() - worker_started
    try:
        import mlx.core as mx
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.server import (
            make_response_stream_state, prompt_has_open_thinking,
            suppress_tool_call_content,
        )
        from mlx_vlm.server.responses_state import process_tool_calls
        from mlx_vlm.tool_parsers import _infer_tool_parser_from_processor, load_tool_module
        from mlx_vlm.trainer.utils import _to_lora, get_module_by_name, set_module_by_name
    except ImportError as exc:  # pragma: no cover - live pinned-runtime path
        raise HarnessError(f"selected worker cannot import pinned MLX-VLM: {exc}") from exc

    _, adapter_config = _read_object(
        assets.adapter_dir / "adapter_config.json", "adapter config")
    parameters = lora_parameters_for_strength(adapter_config, clean["adapter_strength"])
    model_load_started = time.monotonic()
    model, processor = load(str(assets.model_root), lazy=False, strict=True)
    weights = mx.load(str(assets.adapter_dir / "adapters.safetensors"))
    if not isinstance(weights, Mapping):
        raise HarnessError("adapter safetensors did not load as a parameter map")
    module_names = sorted({str(key).rsplit(".", 1)[0] for key in weights})
    if len(module_names) != EXPECTED_LORA_MODULE_COUNT:
        raise HarnessError(
            f"adapter must target exactly {EXPECTED_LORA_MODULE_COUNT} VLM modules")
    for name in module_names:
        try:
            wrapped = _to_lora(get_module_by_name(model, name), parameters)
            set_module_by_name(model, name, wrapped)
        except Exception as exc:
            raise HarnessError(f"cannot apply adapter to authenticated module {name}: {exc}") from exc
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model_load_seconds = time.monotonic() - model_load_started

    messages = vlm_processor_messages(clean["messages"])
    tools = clean["tools"]
    tool_module = None
    if tools:
        parser_type = _infer_tool_parser_from_processor(processor)
        if parser_type != "qwen3_coder":
            raise HarnessError("authenticated Qwen VLM tool parser is unavailable")
        tool_module = load_tool_module(parser_type)
    formatted = apply_chat_template(
        processor, model.config, messages, num_images=len(image_paths),
        tools=tools or None, enable_thinking=clean["think"])
    derived_seed = int.from_bytes(hashlib.sha256(
        assets.manifest_sha256.encode("ascii") + canonical_json(clean)
    ).digest()[:4], "big")
    seed = int(clean["seed"] if clean["seed"] is not None else derived_seed) & 0xFFFFFFFF
    mx.random.seed(seed)

    full_raw = ""
    full_content = ""
    full_thinking = ""
    pending = ""
    final = None
    stopped = False
    content_streamed = False
    thinking_streamed = False
    holdback = max((len(item) - 1 for item in clean["stop"]), default=0)
    generation_started = time.monotonic()
    first_token_at: float | None = None
    with os.fdopen(events_fd, "w", encoding="utf-8", buffering=1, closefd=True) as events:
        thinking_state = make_response_stream_state(
            processor,
            prompt_has_open_thinking(formatted, clean["think"]),
        )
        in_tool_call = False
        tool_call_start = tool_module.tool_call_start if tool_module else None

        def emit_raw(segment: str, *, last: bool = False) -> None:
            nonlocal full_raw, full_content, full_thinking
            nonlocal content_streamed, thinking_streamed, in_tool_call
            full_raw += segment
            split = thinking_state.feed(segment, last=last)
            content = split.content
            thinking = split.reasoning
            in_tool_call, content = suppress_tool_call_content(
                full_raw, in_tool_call, tool_call_start, content)
            if content:
                full_content += content
                content_streamed = True
            if thinking:
                full_thinking += thinking
                thinking_streamed = True
            if content or thinking:
                _emit_event(events, {
                    "type": "delta", "text": content or "",
                    "thinking": thinking or "",
                    "spiral_adapter_strength": clean["adapter_strength"],
                })

        for result in stream_generate(
            model, processor, formatted,
            image=image_paths or None,
            max_tokens=clean["max_tokens"],
            temperature=clean["temperature"],
            top_p=clean["top_p"],
            seed=seed,
            verbose=False,
            enable_thinking=clean["think"],
            skip_special_tokens=False if (tools or clean["think"]) else True,
        ):
            final = result
            if first_token_at is None:
                first_token_at = time.monotonic()
            pending += result.text
            stop_positions = [pending.find(item) for item in clean["stop"] if item in pending]
            if stop_positions:
                segment = pending[:min(stop_positions)]
                emit_raw(segment, last=True)
                pending = ""
                stopped = True
                break
            safe = len(pending) - holdback
            if safe > 0:
                segment, pending = pending[:safe], pending[safe:]
                emit_raw(segment)
        if not stopped:
            emit_raw(pending, last=True)

        tool_calls: list[dict[str, Any]] = []
        content = full_content
        if tool_module is not None:
            parsed = process_tool_calls(full_raw, tool_module, tools)
            for call in parsed["calls"]:
                function = dict(call.get("function") or {})
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_calls.append({
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": arguments,
                    }
                })
        finish_reason = "tool_calls" if tool_calls else (
            "stop" if stopped else (getattr(final, "finish_reason", None) or "stop"))
        _emit_event(events, {
            "type": "result",
            "text": content,
            "thinking": full_thinking,
            "content_streamed": content_streamed,
            "thinking_streamed": thinking_streamed,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "prompt_tokens": int(getattr(final, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(final, "generation_tokens", 0) or 0),
            "seed": seed,
            "lora_module_count": len(module_names),
            "spiral_adapter_strength": clean["adapter_strength"],
            "spiral_runtime_identity": assets.identity,
            "timings": {
                "asset_attestation_seconds": attestation_seconds,
                "model_load_seconds": model_load_seconds,
                "generation_to_first_token_seconds": (
                    (first_token_at - generation_started) if first_token_at is not None else 0.0),
                "load_to_first_token_seconds": (
                    (first_token_at - model_load_started) if first_token_at is not None else model_load_seconds),
            },
        })
    del model, processor, weights
    mx.clear_cache()


@contextlib.contextmanager
def compute_admission(
    *, trusted_handoff: bool, lease_path: Path, ollama_url: str,
    owner: Mapping[str, Any],
) -> Iterator[TrainingComputeLease | None]:
    """Avoid self-deadlock when the authenticated Spiral host holds the flock."""

    lease: TrainingComputeLease | None = None
    if not trusted_handoff:
        lease = TrainingComputeLease(lease_path)
        lease.acquire(owner=dict(owner))
    try:
        # A host-held handoff skips only a second flock acquisition.  It never
        # weakens the strict no-resident-Ollama admission check.
        verify_ollama_empty(ollama_url)
        yield lease
    finally:
        if lease is not None:
            lease.release()


def _iter_pipe_json(process: subprocess.Popen[Any], descriptor: int, timeout: float) -> Iterator[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    buffer = b""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        ready, _, _ = select.select([descriptor], [], [], min(remaining, 1.0))
        if ready:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise HarnessError("academic VLM worker emitted invalid event JSON") from exc
                if not isinstance(event, dict):
                    raise HarnessError("academic VLM worker event must be an object")
                yield event
        elif process.poll() is not None:
            # Drain the pipe after child exit.
            continue
    if buffer.strip():
        raise HarnessError("academic VLM worker emitted a truncated event")


def _attested_worker_events(
    *, process: subprocess.Popen[Any], descriptor: int, timeout: float,
    expected_identity: Mapping[str, Any], expected_strength: float,
    parent_attestation_seconds: float, log_path: Path,
) -> Iterator[dict[str, Any]]:
    """Yield deltas live but publish success only after a clean worker exit."""

    pending_result: dict[str, Any] | None = None
    for event in _iter_pipe_json(process, descriptor, timeout):
        event_type = event.get("type")
        if event_type not in {"delta", "result"}:
            raise HarnessError("academic VLM worker emitted an unknown event")
        if event.get("spiral_adapter_strength") != expected_strength:
            raise HarnessError("academic VLM strength attestation mismatch")
        if event_type == "result":
            if pending_result is not None:
                raise HarnessError("academic VLM worker emitted multiple results")
            if event.get("spiral_runtime_identity") != expected_identity:
                raise HarnessError("academic VLM runtime attestation mismatch")
            if event.get("lora_module_count") != EXPECTED_LORA_MODULE_COUNT:
                raise HarnessError("academic VLM adapter module-count mismatch")
            timings = event.get("timings")
            if not isinstance(timings, dict):
                raise HarnessError("academic VLM worker emitted no timing receipt")
            timings["parent_request_attestation_seconds"] = parent_attestation_seconds
            pending_result = event
            continue
        if pending_result is not None:
            raise HarnessError("academic VLM emitted data after its result")
        yield event

    return_code = process.wait(timeout=5)
    if return_code or pending_result is None:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-1600:]
        raise HarnessError(
            "academic MLX-VLM worker failed before an attested result: "
            + detail.strip())
    # This is the sole success terminal event. Reaching it proves the worker
    # closed its event pipe and exited zero after MLX cleanup.
    yield pending_result


@contextlib.contextmanager
def _worker_lifecycle(state: dict[str, Any]) -> Iterator[None]:
    """Unload the worker before enclosing compute-admission contexts unwind."""

    try:
        yield
    finally:
        for key in ("write_fd", "read_fd"):
            descriptor = state.pop(key, None)
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        process = state.get("process")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class AcademicVlmService:
    def __init__(
        self, *, manifest_path: Path, model_root: Path, python_executable: str,
        lease_path: Path, ollama_url: str, request_timeout: float,
        lease_authority_token: bytes | None,
    ) -> None:
        self.manifest_path = manifest_path
        self.model_root = model_root
        self.python_executable = python_executable
        self.lease_path = lease_path
        self.ollama_url = ollama_url
        self.request_timeout = request_timeout
        self.lease_authority_token = lease_authority_token
        self._request_lock = threading.Lock()
        self.assets = validate_vlm_runtime_assets(manifest_path, model_root)
        validate_vlm_environment(python_executable)

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self.assets.identity)

    def readiness_identity(self) -> dict[str, Any]:
        validate_vlm_storage_sentinel(self.assets)
        return self.identity

    def events(self, request: Any, *, lease_authority_header: str | None) -> Iterator[dict[str, Any]]:
        clean = validate_vlm_chat_request(request, expected_model=self.identity["model"])
        handoff = trusted_lease_handoff(
            lease_authority_header, self.lease_authority_token)
        if not self._request_lock.acquire(blocking=False):
            raise RequestError(
                "academic VLM runtime is serving another completion", status=503,
                code="server_busy")
        lifecycle: dict[str, Any] = {}
        try:
            request_attestation_started = time.monotonic()
            assets = validate_vlm_runtime_assets(
                self.manifest_path, self.model_root,
                expected_base_snapshot=self.assets.base_file_snapshot)
            request_attestation_seconds = time.monotonic() - request_attestation_started
            validate_vlm_environment(self.python_executable)
            if assets.identity != self.assets.identity:
                raise HarnessError("academic VLM assets changed after server startup")
            owner = {
                "operation": "academic_vlm_inference",
                "model": assets.identity["model"],
                "manifest_sha256": assets.manifest_sha256,
                "adapter_strength": clean["adapter_strength"],
            }
            with compute_admission(
                trusted_handoff=handoff, lease_path=self.lease_path,
                ollama_url=self.ollama_url, owner=owner,
            ) as lease, tempfile.TemporaryDirectory(
                prefix="spiral-academic-vlm-request-"
            ) as raw_tmp, _worker_lifecycle(lifecycle):
                scratch = Path(raw_tmp)
                request_path = scratch / "request.json"
                request_path.write_bytes(canonical_json(clean))
                startup_receipt_path = scratch / "startup-receipt.json"
                startup_receipt_path.write_bytes(canonical_json({
                    "spiral_runtime_identity": self.assets.identity,
                    "base_file_snapshot": list(self.assets.base_file_snapshot),
                }))
                read_fd, write_fd = os.pipe()
                lifecycle.update({"read_fd": read_fd, "write_fd": write_fd})
                log_path = scratch / "worker.log"
                with log_path.open("wb") as log:
                    pass_fds = [write_fd]
                    if lease is not None and lease.descriptor is not None:
                        pass_fds.append(lease.descriptor)
                    process = subprocess.Popen(
                        [
                            self.python_executable, str(Path(__file__).resolve()), "_worker",
                            "--manifest", str(self.manifest_path),
                            "--model-root", str(self.model_root),
                            "--request", str(request_path),
                            "--startup-receipt", str(startup_receipt_path),
                            "--events-fd", str(write_fd),
                        ],
                        stdout=log, stderr=subprocess.STDOUT,
                        pass_fds=tuple(pass_fds), close_fds=True,
                    )
                    lifecycle["process"] = process
                os.close(write_fd)
                lifecycle.pop("write_fd", None)
                try:
                    for event in _attested_worker_events(
                        process=process, descriptor=read_fd,
                        timeout=self.request_timeout,
                        expected_identity=assets.identity,
                        expected_strength=clean["adapter_strength"],
                        parent_attestation_seconds=request_attestation_seconds,
                        log_path=log_path,
                    ):
                        yield event
                finally:
                    os.close(read_fd)
                    lifecycle.pop("read_fd", None)
        except subprocess.TimeoutExpired as exc:
            raise RequestError(
                "academic MLX-VLM worker exceeded its bounded request timeout",
                status=504, code="timeout") from exc
        finally:
            self._request_lock.release()


def identity_health_response(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {"object": "spiral.runtime.identity", "spiral_runtime_identity": dict(identity)}


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ollama_delta(
    identity: Mapping[str, Any], text: str, thinking: str = "", *,
    adapter_strength: Any = DEFAULT_ADAPTER_STRENGTH,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if thinking:
        message["thinking"] = thinking
    return {
        "model": identity["model"], "created_at": _created_at(),
        "message": message, "done": False,
        "spiral_adapter_strength": canonical_adapter_strength(adapter_strength),
    }


def ollama_result(identity: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant", "content": str(event.get("text") or "")}
    if event.get("thinking"):
        message["thinking"] = str(event["thinking"])
    if event.get("tool_calls"):
        message["tool_calls"] = event["tool_calls"]
    return {
        "model": identity["model"], "created_at": _created_at(),
        "message": message,
        "done": True,
        "done_reason": str(event.get("finish_reason") or "stop"),
        "prompt_eval_count": int(event.get("prompt_tokens") or 0),
        "eval_count": int(event.get("completion_tokens") or 0),
        "spiral_runtime_identity": dict(identity),
        "spiral_adapter_strength": canonical_adapter_strength(
            event.get("spiral_adapter_strength", DEFAULT_ADAPTER_STRENGTH)),
        "spiral_timings": dict(event.get("timings") or {}),
    }


class AcademicVlmHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], service: AcademicVlmService):
        super().__init__(address, AcademicVlmRequestHandler)
        self.service = service


class AcademicVlmRequestHandler(BaseHTTPRequestHandler):
    server: AcademicVlmHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("academic-vlm-server · " + (fmt % args) + "\n")

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        payload = canonical_json(dict(value))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, exc: BaseException) -> None:
        if isinstance(exc, RequestError):
            status, code = exc.status, exc.code
        elif isinstance(exc, HarnessError):
            status, code = 503, "runtime_unavailable"
        else:
            status, code = 500, "internal_error"
        self.close_connection = True
        self._json(status, {"error": str(exc), "code": code})

    def _validate_local_http_request(self, *, require_json: bool) -> None:
        host_header = self.headers.get("Host", "")
        try:
            hostname = urllib.parse.urlsplit("//" + host_header).hostname
            is_loopback = hostname == "localhost" or (
                hostname is not None and ipaddress.ip_address(hostname).is_loopback)
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise RequestError("Host must name loopback", status=403, code="forbidden")
        if self.headers.get("Origin") is not None:
            raise RequestError("browser-origin requests are forbidden", status=403, code="forbidden")
        if self.headers.get("Sec-Fetch-Site", "").lower() in {"cross-site", "same-site"}:
            raise RequestError("cross-site requests are forbidden", status=403, code="forbidden")
        if require_json:
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                raise RequestError(
                    "Content-Type must be application/json", status=415,
                    code="unsupported_media_type")

    def _stream_error(
        self, exc: BaseException, *, adapter_strength: float,
    ) -> None:
        if isinstance(exc, RequestError):
            code = exc.code
        elif isinstance(exc, HarnessError):
            code = "runtime_unavailable"
        else:
            code = "internal_error"
        payload = {
            "model": self.server.service.identity["model"],
            "created_at": _created_at(),
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "error",
            "error": str(exc),
            "code": code,
            "spiral_runtime_identity": self.server.service.identity,
            "spiral_adapter_strength": adapter_strength,
        }
        self.wfile.write(canonical_json(payload))
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._validate_local_http_request(require_json=False)
            if urllib.parse.urlparse(self.path).path.rstrip("/") != "/v1/spiral/identity":
                self._json(404, {"error": "not found", "code": "not_found"})
                return
            self._json(
                200,
                identity_health_response(self.server.service.readiness_identity()),
            )
        except BaseException as exc:
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path.rstrip("/") != "/api/chat":
            self._json(404, {"error": "not found", "code": "not_found"})
            return
        iterator: Iterator[dict[str, Any]] | None = None
        committed = False
        requested_strength = DEFAULT_ADAPTER_STRENGTH
        try:
            self._validate_local_http_request(require_json=True)
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise RequestError("Content-Length is required", status=411)
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise RequestError("Content-Length must be an integer", status=411) from exc
            if not 1 <= length <= MAX_REQUEST_BYTES:
                raise RequestError(
                    f"request body must be 1–{MAX_REQUEST_BYTES} bytes", status=413)
            try:
                request = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RequestError("request body is not valid UTF-8 JSON") from exc
            clean = validate_vlm_chat_request(
                request, expected_model=self.server.service.identity["model"])
            requested_strength = clean["adapter_strength"]
            iterator = iter(self.server.service.events(
                request,
                lease_authority_header=self.headers.get(LEASE_AUTHORITY_HEADER)))
            first = next(iterator)
            if clean["stream"]:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                committed = True
                self.close_connection = True
                for event in itertools.chain((first,), iterator):
                    if event.get("type") == "delta":
                        payload = ollama_delta(
                            self.server.service.identity,
                            str(event.get("text") or ""),
                            str(event.get("thinking") or ""),
                            adapter_strength=event.get("spiral_adapter_strength"))
                    else:
                        streamed_result = dict(event)
                        # Any fields already emitted as deltas must be empty on
                        # Ollama's done frame or the APK accumulator duplicates them.
                        if streamed_result.get("content_streamed"):
                            streamed_result["text"] = ""
                        if streamed_result.get("thinking_streamed"):
                            streamed_result["thinking"] = ""
                        payload = ollama_result(
                            self.server.service.identity, streamed_result)
                    self.wfile.write(canonical_json(payload))
                    self.wfile.flush()
            else:
                events = [first, *iterator]
                result = next((event for event in reversed(events) if event.get("type") == "result"), None)
                if result is None:
                    raise HarnessError("academic VLM worker returned no result")
                self._json(200, ollama_result(self.server.service.identity, result))
        except StopIteration:
            self._error(HarnessError("academic VLM worker returned no events"))
        except (BrokenPipeError, ConnectionResetError):
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()  # unload child even if the phone disconnects
        except BaseException as exc:
            if committed:
                try:
                    self._stream_error(exc, adapter_strength=requested_strength)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._error(exc)
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Serve Spiral's exact academic adapter on full MLX Qwen vision")
    subparsers = result.add_subparsers(dest="command")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--model-root", type=Path, required=True)
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--startup-receipt", type=Path, required=True)
    worker.add_argument("--events-fd", type=int, required=True)
    result.add_argument("--manifest", type=Path)
    result.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    result.add_argument("--python", default=str(DEFAULT_VLM_PYTHON))
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8081)
    result.add_argument(
        "--lease-path", type=Path,
        default=Path("~/.spiralchat/spiral-compute.lease").expanduser())
    result.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    result.add_argument("--request-timeout", type=float, default=7200.0)
    result.add_argument(
        "--lease-authority-token-file", type=Path,
        help="owner-only token shared with Spiral host for host-held lease handoff")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "_worker":
            _worker_generate(
                manifest_path=arguments.manifest, model_root=arguments.model_root,
                request_path=arguments.request,
                startup_receipt_path=arguments.startup_receipt,
                events_fd=arguments.events_fd)
            return 0
        if arguments.manifest is None:
            raise HarnessError("--manifest is required")
        validate_bind_address(arguments.host, arguments.port)
        if not math.isfinite(arguments.request_timeout) or not 1 <= arguments.request_timeout <= 7200:
            raise HarnessError("--request-timeout must be between 1 and 7200 seconds")
        token = load_lease_authority_token(arguments.lease_authority_token_file)
        startup_attestation_started = time.monotonic()
        service = AcademicVlmService(
            manifest_path=arguments.manifest, model_root=arguments.model_root,
            python_executable=arguments.python, lease_path=arguments.lease_path,
            ollama_url=arguments.ollama_url, request_timeout=arguments.request_timeout,
            lease_authority_token=token)
        startup_attestation_seconds = time.monotonic() - startup_attestation_started
        server = AcademicVlmHTTPServer((arguments.host, arguments.port), service)
        print(json.dumps({
            "listening": f"http://{arguments.host}:{arguments.port}",
            "identity": service.identity,
            "weight_residency": "child-process-per-request",
            "startup_attestation_seconds": startup_attestation_seconds,
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
