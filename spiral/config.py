"""Configuration — backend is a swappable seam; local-first defaults.

Model strategy (32 GB unified memory, hardware-honest):
  - ONE resident qwen3.8:27b (dense, hybrid-attention, vision-capable) for
    planning, reading, writing and vision. Thinking is toggled by role — same
    weights, so plan and build never pay a model swap. The hybrid layout (16 of
    64 layers full attention) keeps KV tiny, so context is cheap on 32 GB.
  - ESCALATION is the same model with thinking ON. Thinking is a per-request
    flag, not another set of weights, so the lanes share one name and each
    attempt records the lane it ran in for the route ledger to read.
  - Different-family judges remain available, but are an explicit opt-in. The
    default loaded configuration reuses the resident 27B with different role
    prompts because correlated model opinions do not justify hidden RAM pressure.
    Deterministic evidence, not a second persona, is the normal judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    num_ctx: int = 16384
    think: bool = False


ACADEMIC_ADAPTER_SCHEMA = "spiral.academic-adapter.v1"
ACADEMIC_CORPUS_SCHEMA = "spiral.academic-corpus-manifest.v1"
ACADEMIC_DATASET_SCHEMA = "spiral.academic-mlx-dataset.v1"
ACADEMIC_PROMPT_CONTRACT = "spiral.academic-plan-prose.v1"
ACADEMIC_ADAPTER_SHA256_SEMANTICS = (
    "sha256(sorted(relative_path\\0size_bytes\\0file_sha256\\n))"
)
ACADEMIC_SOURCE_STRATA = (
    "arxiv:hep-th",
    "arxiv:hep-ph",
    "pubmed",
)
ACADEMIC_PROFILE_ID = "academic-hep-pubmed-v1"
ACADEMIC_RUNTIME_MODEL = "qwen3.8-27b-academic"
ACADEMIC_PROVIDER = "mlx_lm"
ACADEMIC_TRANSPORT_ADAPTER = "openai-compatible"
ACADEMIC_BASE_MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
ACADEMIC_BASE_MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
ACADEMIC_BASE_MODEL_TYPE = "qwen3_5"
ACADEMIC_BASE_MODEL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
ACADEMIC_BASE_MODEL_CONFIG_SHA256 = (
    "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"
)
ACADEMIC_BASE_WEIGHT_INDEX_SHA256 = (
    "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"
)
ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256 = (
    "8126a3fd4aef3346254965791eedc5a5468bf7fcf46bdd95ef29dd13266ed589"
)
ACADEMIC_BASE_WEIGHT_FILES = (
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
ACADEMIC_ADAPTER_FORMAT = "mlx_lm_lora"
ACADEMIC_RUNTIME_IDENTITY_SCHEMA = "spiral.academic-runtime-identity.v1"
ACADEMIC_SERVER_CONTRACT = "spiral.academic-one-request-server.v1"
ACADEMIC_WEIGHT_RESIDENCY = "child-process-per-request"
ACADEMIC_COMPUTE_LEASE = "spiral-compute-flock-v1"
ACADEMIC_OLLAMA_ADMISSION = "strict-empty-no-eviction"
ACADEMIC_UNLOAD_BOUNDARY = "child-exit-before-lease-release"
ACADEMIC_ADAPTER_STRENGTH = 1.0
ACADEMIC_ADAPTER_STRENGTH_MIN = 0.0
ACADEMIC_ADAPTER_STRENGTH_MAX = 2.0
ACADEMIC_ADAPTER_STRENGTH_STEP = 0.05
ACADEMIC_AUTHOR_SAFE_SPLIT_POLICY = (
    "connected document-author components; sha256 90/5/5 with deterministic "
    "nonempty pilot repair"
)


@dataclass
class AcademicWriterSpec(ModelSpec):
    """An opt-in, identity-pinned final-prose route.

    ``name`` is a private provider-map alias, not the model sent on the wire.  Keeping
    those names separate prevents an academic endpoint from silently capturing planner,
    tool, or Builder calls that happen to use the same base model id.
    """

    enabled: bool = False
    ready: bool = False
    profile_id: str = ""
    runtime_model: str = ""
    provider: str = ""
    base_url: str = ""
    transport_adapter: str = "openai-compatible"
    api_key_env: str = ""
    api_key_required: bool = False
    manifest_path: str = ""
    manifest_sha256: str = ""
    prompt_contract: str = ""
    base_model_id: str = ""
    base_model_revision: str = ""
    base_model_type: str = ""
    base_model_architecture: str = ""
    base_model_config_sha256: str = ""
    base_weight_index_sha256: str = ""
    base_weight_inventory_sha256: str = ""
    base_weight_files: tuple[dict, ...] = ()
    base_model_quantization: dict = field(default_factory=dict)
    adapter_path: str = ""
    adapter_format: str = ""
    adapter_sha256: str = ""
    adapter_required_files: tuple[dict, ...] = ()
    corpus_manifest_path: str = ""
    corpus_manifest_sha256: str = ""
    dataset_manifest_path: str = ""
    dataset_manifest_sha256: str = ""
    source_corpus_path: str = ""
    source_corpus_sha256: str = ""
    source_corpus_file_sha256: str = ""
    source_strata: tuple[str, ...] = ()
    adapter_strength: float = ACADEMIC_ADAPTER_STRENGTH
    runtime_identity: dict = field(default_factory=dict)
    overrides: tuple[str, ...] = ()
    error: str = ""

    def identity(self) -> dict:
        """Stable public receipt for every academic synthesis attempt."""

        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "route_name": self.name,
            "profile_id": self.profile_id,
            "runtime_model": self.runtime_model,
            "provider": self.provider,
            "base_url": self.base_url,
            "transport_adapter": self.transport_adapter,
            "adapter_strength": self.adapter_strength,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "prompt_contract": self.prompt_contract,
            "base_model": {
                "model_id": self.base_model_id,
                "revision": self.base_model_revision,
                "model_type": self.base_model_type,
                "architecture": self.base_model_architecture,
                "config_sha256": self.base_model_config_sha256,
                "weight_index_sha256": self.base_weight_index_sha256,
                "weight_inventory_sha256": self.base_weight_inventory_sha256,
                "weight_files": [dict(row) for row in self.base_weight_files],
                "quantization": dict(self.base_model_quantization),
            },
            "adapter": {
                "path": self.adapter_path,
                "format": self.adapter_format,
                "sha256": self.adapter_sha256,
                "required_files": [dict(row) for row in self.adapter_required_files],
            },
            "corpus_manifest_path": self.corpus_manifest_path,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "dataset_manifest_path": self.dataset_manifest_path,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "source_corpus_path": self.source_corpus_path,
            "source_corpus_sha256": self.source_corpus_sha256,
            "source_corpus_file_sha256": self.source_corpus_file_sha256,
            "source_strata": list(self.source_strata),
            "runtime_identity": dict(self.runtime_identity),
            "overrides": list(self.overrides),
            "error": self.error,
        }


def _academic_writer_config(cfg, overlay: dict, config_file) -> None:
    """Resolve and authenticate the optional academic route without model I/O."""

    import hashlib
    import json
    import os
    import re
    from datetime import date
    from pathlib import Path

    spec = cfg.academic_writer
    raw = overlay.get("academic_writer") or {}
    if not isinstance(raw, dict):
        spec.error = "academic_writer must be an object"
        return

    def env_or(name: str, key: str, default=""):
        return os.environ.get(name, raw.get(key, default))

    def truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    def resolved_path(value, *, relative_to: Path) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        return (relative_to / path).resolve() if not path.is_absolute() else path.resolve()

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def adapter_bundle_sha256(path: Path, required_files) -> tuple[str, tuple[dict, ...]]:
        """Authenticate the immutable MLX-LM adapter payload, not directory metadata."""

        if not isinstance(required_files, list):
            raise ValueError("adapter.required_files must be a list")
        expected_paths = ["adapter_config.json", "adapters.safetensors"]
        if [str(row.get("path") or "") for row in required_files
                if isinstance(row, dict)] != expected_paths:
            raise ValueError(
                "adapter.required_files must be adapter_config.json then adapters.safetensors")
        root = path.resolve()
        verified = []
        lines = []
        for row in required_files:
            if not isinstance(row, dict):
                raise ValueError("adapter.required_files entries must be objects")
            relative = str(row.get("path") or "")
            candidate = root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"missing immutable adapter file: {relative}")
            resolved = candidate.resolve()
            if resolved.parent != root:
                raise ValueError(f"adapter file escapes bundle: {relative}")
            size = resolved.stat().st_size
            digest = file_sha256(resolved)
            if size != int(row.get("size_bytes", -1)):
                raise ValueError(f"adapter file size mismatch: {relative}")
            if digest != str(row.get("sha256") or "").lower():
                raise ValueError(f"adapter file SHA-256 mismatch: {relative}")
            verified.append({
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
            })
            lines.append(f"{relative}\0{size}\0{digest}\n")
        bundle_digest = hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()
        return bundle_digest, tuple(verified)

    spec.enabled = truthy(env_or(
        "SPIRAL_ACADEMIC_WRITER_ENABLED", "enabled", False))
    manifest_value = env_or(
        "SPIRAL_ACADEMIC_WRITER_MANIFEST", "manifest_path",
        raw.get("manifest", ""))
    config_dir = Path(config_file).parent if config_file else Path.cwd()
    manifest_path = resolved_path(manifest_value, relative_to=config_dir)
    spec.manifest_path = str(manifest_path) if manifest_path else ""
    if not spec.enabled:
        return
    if manifest_path is None or not manifest_path.is_file():
        spec.error = "enabled academic writer requires a readable manifest_path"
        return

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError, TypeError) as exc:
        spec.error = f"invalid academic writer manifest: {type(exc).__name__}"
        return
    if not isinstance(manifest, dict):
        spec.error = "academic writer manifest must be an object"
        return
    if manifest.get("schema_version") != ACADEMIC_ADAPTER_SCHEMA:
        spec.error = f"academic writer manifest schema must be {ACADEMIC_ADAPTER_SCHEMA}"
        return
    if manifest.get("prompt_contract") != ACADEMIC_PROMPT_CONTRACT:
        spec.error = f"academic writer prompt contract must be {ACADEMIC_PROMPT_CONTRACT}"
        return

    spec.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    spec.prompt_contract = str(manifest.get("prompt_contract") or "")
    runtime = manifest.get("runtime") or {}
    base_model = manifest.get("base_model") or {}
    adapter = manifest.get("adapter") or {}
    training = manifest.get("training") or {}
    dataset = manifest.get("dataset") or {}
    if not all(isinstance(value, dict) for value in (
            runtime, base_model, adapter, training, dataset)):
        spec.error = "academic writer manifest identity sections must be objects"
        return

    override_fields = []

    def configured(env: str, key: str, manifest_value="") -> str:
        value = env_or(env, key, manifest_value)
        return str(value or "").strip()

    def immutable(env: str, key: str, manifest_value="") -> str:
        pinned = str(manifest_value or "").strip()
        requested = configured(env, key, pinned)
        if requested != pinned:
            raise ValueError(
                f"academic writer {key} is manifest-pinned and cannot be overridden")
        return pinned

    try:
        spec.profile_id = immutable(
            "SPIRAL_ACADEMIC_WRITER_PROFILE", "profile_id",
            manifest.get("profile_id"))
        spec.runtime_model = immutable(
            "SPIRAL_ACADEMIC_WRITER_MODEL", "model", runtime.get("model"))
        spec.provider = immutable(
            "SPIRAL_ACADEMIC_WRITER_PROVIDER", "provider", runtime.get("provider"))
        spec.transport_adapter = immutable(
            "SPIRAL_ACADEMIC_WRITER_TRANSPORT", "transport_adapter",
            runtime.get("transport_adapter"))
    except ValueError as exc:
        spec.error = str(exc)
        return
    manifest_base_url = str(runtime.get("base_url") or "").strip()
    spec.base_url = configured(
        "SPIRAL_ACADEMIC_WRITER_BASE_URL", "base_url", manifest_base_url)
    if spec.base_url != manifest_base_url:
        override_fields.append("base_url")
    spec.api_key_env = configured(
        "SPIRAL_ACADEMIC_WRITER_API_KEY_ENV", "api_key_env",
        runtime.get("api_key_env"))
    if spec.api_key_env != str(runtime.get("api_key_env") or "").strip():
        override_fields.append("api_key_env")
    spec.api_key_required = truthy(raw.get(
        "api_key_required", runtime.get("api_key_required", bool(spec.api_key_env))))
    # This multiplies the LoRA contribution around its trained MLX scale. It
    # never rewrites adapter_config.json: 0.0 selects the base contribution,
    # 1.0 preserves scale 32, and values above one amplify the learned delta.
    manifest_adapter_strength = runtime.get(
        "default_adapter_strength", ACADEMIC_ADAPTER_STRENGTH)
    configured_adapter_strength = os.environ.get(
        "SPIRAL_ACADEMIC_WRITER_STRENGTH",
        raw.get("adapter_strength", manifest_adapter_strength))

    def valid_adapter_strength(value) -> float | None:
        try:
            strength = float(value) if not isinstance(value, bool) else float("nan")
        except (TypeError, ValueError):
            return None
        if not ACADEMIC_ADAPTER_STRENGTH_MIN <= strength <= ACADEMIC_ADAPTER_STRENGTH_MAX:
            return None
        steps = round(strength / ACADEMIC_ADAPTER_STRENGTH_STEP)
        canonical = round(steps * ACADEMIC_ADAPTER_STRENGTH_STEP, 10)
        return canonical if abs(strength - canonical) <= 1e-9 else None

    manifest_default = valid_adapter_strength(manifest_adapter_strength)
    if manifest_default != ACADEMIC_ADAPTER_STRENGTH:
        spec.error = "academic writer manifest default adapter strength must be 1.0"
        return
    selected_strength = valid_adapter_strength(configured_adapter_strength)
    if selected_strength is None:
        spec.error = (
            "academic writer adapter_strength must be from 0.0 through 2.0 "
            "in 0.05 steps")
        return
    spec.adapter_strength = selected_strength
    if spec.adapter_strength != manifest_default:
        override_fields.append("adapter_strength")
    spec.num_ctx = int(raw.get("num_ctx", runtime.get("num_ctx", 24_576)))
    spec.think = truthy(raw.get("think", runtime.get("think", False)))
    spec.base_model_id = str(base_model.get("model_id") or "").strip()
    spec.base_model_revision = str(base_model.get("revision") or "").strip()
    spec.base_model_type = str(base_model.get("model_type") or "").strip()
    spec.base_model_architecture = str(base_model.get("architecture") or "").strip()
    spec.base_model_config_sha256 = str(
        base_model.get("config_sha256") or "").strip().lower()
    spec.base_weight_index_sha256 = str(
        base_model.get("weight_index_sha256") or "").strip().lower()
    spec.base_weight_inventory_sha256 = str(
        base_model.get("weight_inventory_sha256") or "").strip().lower()
    quantization = base_model.get("quantization") or {}
    spec.base_model_quantization = (
        dict(quantization) if isinstance(quantization, dict) else {})

    immutable_identity = {
        "profile_id": (spec.profile_id, ACADEMIC_PROFILE_ID),
        "runtime.model": (spec.runtime_model, ACADEMIC_RUNTIME_MODEL),
        "runtime.provider": (spec.provider, ACADEMIC_PROVIDER),
        "runtime.transport_adapter": (
            spec.transport_adapter, ACADEMIC_TRANSPORT_ADAPTER),
        "base_model.model_id": (spec.base_model_id, ACADEMIC_BASE_MODEL_ID),
        "base_model.revision": (
            spec.base_model_revision, ACADEMIC_BASE_MODEL_REVISION),
        "base_model.model_type": (
            spec.base_model_type, ACADEMIC_BASE_MODEL_TYPE),
        "base_model.architecture": (
            spec.base_model_architecture, ACADEMIC_BASE_MODEL_ARCHITECTURE),
    }
    for label, (actual, expected) in immutable_identity.items():
        if actual != expected:
            spec.error = f"academic writer {label} must be immutable {expected!r}"
            return
    if spec.base_model_quantization != {
            "bits": 4, "group_size": 64, "mode": "affine"}:
        spec.error = "academic writer base must be the pinned affine q4/group-64 model"
        return
    sha256_re = re.compile(r"^[0-9a-f]{64}$")
    if not sha256_re.fullmatch(spec.base_model_config_sha256):
        spec.error = "academic writer base config SHA-256 is missing or malformed"
        return
    if not sha256_re.fullmatch(spec.base_weight_index_sha256):
        spec.error = "academic writer base weight-index SHA-256 is missing or malformed"
        return
    if not sha256_re.fullmatch(spec.base_weight_inventory_sha256):
        spec.error = "academic writer base weight inventory SHA-256 is missing or malformed"
        return
    if spec.base_model_config_sha256 != ACADEMIC_BASE_MODEL_CONFIG_SHA256:
        spec.error = "academic writer base config SHA-256 is not the pinned Qwen3.8 base"
        return
    if spec.base_weight_index_sha256 != ACADEMIC_BASE_WEIGHT_INDEX_SHA256:
        spec.error = "academic writer base weight-index SHA-256 is not the pinned Qwen3.8 base"
        return
    if spec.base_weight_inventory_sha256 != ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256:
        spec.error = "academic writer base weight inventory SHA-256 is not the pinned Qwen3.8 base"
        return
    weight_files = base_model.get("weight_files")
    if not isinstance(weight_files, list) or not weight_files:
        spec.error = "academic writer base weight inventory is missing"
        return
    verified_weight_files = []
    inventory_lines = []
    seen_weight_paths = set()
    for row in weight_files:
        if not isinstance(row, dict):
            spec.error = "academic writer base weight inventory entry is malformed"
            return
        relative = str(row.get("path") or "")
        digest = str(row.get("sha256") or "").lower()
        try:
            size = int(row.get("size_bytes", -1))
        except (TypeError, ValueError):
            size = -1
        if (not relative or Path(relative).name != relative or relative in seen_weight_paths
                or size <= 0 or not sha256_re.fullmatch(digest)):
            spec.error = "academic writer base weight inventory entry is malformed"
            return
        seen_weight_paths.add(relative)
        verified_weight_files.append({
            "path": relative, "size_bytes": size, "sha256": digest})
        inventory_lines.append(f"{relative}\0{size}\0{digest}\n")
    actual_inventory_sha = hashlib.sha256(
        "".join(sorted(inventory_lines)).encode("utf-8")).hexdigest()
    if actual_inventory_sha != spec.base_weight_inventory_sha256:
        spec.error = "academic writer base weight inventory digest is inconsistent"
        return
    if tuple(verified_weight_files) != ACADEMIC_BASE_WEIGHT_FILES:
        spec.error = "academic writer base weight shards are not the pinned Qwen3.8 inventory"
        return
    spec.base_weight_files = tuple(verified_weight_files)

    adapter_value = env_or(
        "SPIRAL_ACADEMIC_WRITER_ADAPTER", "adapter_path", adapter.get("path"))
    if str(adapter_value or "").strip() != str(adapter.get("path") or "").strip():
        override_fields.append("adapter_path")
    adapter_path = resolved_path(adapter_value, relative_to=manifest_path.parent)
    spec.adapter_path = str(adapter_path) if adapter_path else ""
    spec.adapter_format = str(adapter.get("format") or "").strip()
    spec.adapter_sha256 = str(adapter.get("sha256") or "").strip().lower()
    if spec.adapter_format != ACADEMIC_ADAPTER_FORMAT:
        spec.error = f"academic writer adapter format must be {ACADEMIC_ADAPTER_FORMAT}"
        return
    if adapter.get("sha256_semantics") != ACADEMIC_ADAPTER_SHA256_SEMANTICS:
        spec.error = "academic writer adapter has unsupported SHA-256 semantics"
        return
    if adapter_path is None or not adapter_path.exists():
        spec.error = "academic writer adapter path is missing"
        return
    try:
        if not adapter_path.is_dir():
            raise ValueError("MLX-LM adapter path must be an authenticated bundle directory")
        actual_adapter_sha, required_files = adapter_bundle_sha256(
            adapter_path, adapter.get("required_files"))
        spec.adapter_required_files = required_files
    except (OSError, TypeError, ValueError) as exc:
        spec.error = f"invalid academic writer adapter bundle: {exc}"
        return
    if not spec.adapter_sha256 or actual_adapter_sha != spec.adapter_sha256:
        spec.error = "academic writer adapter SHA-256 does not match the manifest"
        return

    corpus_value = env_or(
        "SPIRAL_ACADEMIC_WRITER_CORPUS_MANIFEST", "corpus_manifest_path",
        raw.get("corpus_manifest", training.get("corpus_manifest_path", "")))
    corpus_path = resolved_path(corpus_value, relative_to=manifest_path.parent)
    spec.corpus_manifest_path = str(corpus_path) if corpus_path else ""
    if corpus_path is None or not corpus_path.is_file():
        spec.error = "academic writer requires its exact corpus_manifest_path"
        return
    try:
        corpus_bytes = corpus_path.read_bytes()
        corpus_manifest = json.loads(corpus_bytes)
    except (OSError, ValueError, TypeError) as exc:
        spec.error = f"invalid academic corpus manifest: {type(exc).__name__}"
        return
    if not isinstance(corpus_manifest, dict) or (
            corpus_manifest.get("schema_version", corpus_manifest.get("schema"))
            != ACADEMIC_CORPUS_SCHEMA):
        spec.error = f"academic corpus manifest schema must be {ACADEMIC_CORPUS_SCHEMA}"
        return
    spec.corpus_manifest_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    expected_corpus_sha = str(training.get("corpus_manifest_sha256") or "").lower()
    if not expected_corpus_sha or spec.corpus_manifest_sha256 != expected_corpus_sha:
        spec.error = "academic corpus manifest SHA-256 does not match training identity"
        return
    strata = tuple(sorted(str(value) for value in corpus_manifest.get("source_strata") or []))
    required_strata = tuple(sorted(ACADEMIC_SOURCE_STRATA))
    spec.source_strata = strata
    if strata != required_strata:
        spec.error = (
            "academic corpus source_strata must be exactly "
            + ", ".join(required_strata))
        return
    if corpus_manifest.get("corpus_schema_version") != ACADEMIC_PROMPT_CONTRACT:
        spec.error = "academic corpus examples do not match the plan-prose prompt contract"
        return
    if corpus_manifest.get("trainable") is not True:
        spec.error = "academic corpus manifest is not marked trainable"
        return
    if corpus_manifest.get("non_trainable_reasons") not in ([], None):
        spec.error = "academic corpus manifest has unresolved trainability failures"
        return

    def positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    try:
        corpus_cutoff = date.fromisoformat(str(corpus_manifest.get("cutoff") or ""))
    except ValueError:
        spec.error = "academic corpus cutoff must be an ISO date"
        return
    if corpus_cutoff > date(2021, 12, 31):
        spec.error = "academic corpus cutoff must not be later than 2021-12-31"
        return
    if corpus_manifest.get("split_policy") != ACADEMIC_AUTHOR_SAFE_SPLIT_POLICY:
        spec.error = "academic corpus must use the pinned author-safe split policy"
        return
    split_diagnostics = corpus_manifest.get("split_diagnostics")
    component_splits = (
        split_diagnostics.get("component_counts_by_split")
        if isinstance(split_diagnostics, dict) else None)
    if (not isinstance(split_diagnostics, dict)
            or not positive_int(split_diagnostics.get("components"))
            or not isinstance(component_splits, dict)
            or set(component_splits) != {"train", "validation", "test"}
            or not all(positive_int(component_splits.get(name))
                       for name in ("train", "validation", "test"))):
        spec.error = "academic corpus lacks author-component split diagnostics"
        return
    counts = corpus_manifest.get("counts")
    if not isinstance(counts, dict):
        spec.error = "academic corpus manifest counts are missing"
        return

    expected_corpus_splits = ("train", "validation", "test")
    by_split = counts.get("by_split")
    if (not isinstance(by_split, dict)
            or set(by_split) != set(expected_corpus_splits)
            or not all(positive_int(by_split.get(name)) for name in expected_corpus_splits)):
        spec.error = "academic corpus requires positive train/validation/test counts"
        return
    task_counts = counts.get("by_task_type")
    if (not isinstance(task_counts, dict)
            or set(task_counts) != {"sentence", "paragraph"}
            or not all(positive_int(task_counts.get(name))
                       for name in ("sentence", "paragraph"))):
        spec.error = "academic corpus must contain only sentence and paragraph tasks"
        return
    expected_stratum_splits = {
        f"{stratum}|{split}"
        for stratum in required_strata
        for split in expected_corpus_splits
    }
    for field_name in ("documents_by_stratum_split", "examples_by_stratum_split"):
        coverage = counts.get(field_name)
        if (not isinstance(coverage, dict)
                or set(coverage) != expected_stratum_splits
                or not all(positive_int(coverage.get(key))
                           for key in expected_stratum_splits)):
            spec.error = (
                "academic corpus requires every source stratum in every author-safe split")
            return
    feasibility = corpus_manifest.get("task_feasibility")
    if not isinstance(feasibility, dict) or set(feasibility) != {"sentence", "paragraph"}:
        spec.error = "academic corpus task feasibility exceeds plan-prose v1"
        return

    dataset_value = env_or(
        "SPIRAL_ACADEMIC_WRITER_DATASET_MANIFEST", "dataset_manifest_path",
        raw.get("dataset_manifest", dataset.get("dataset_manifest_path", "")))
    dataset_path = resolved_path(dataset_value, relative_to=manifest_path.parent)
    spec.dataset_manifest_path = str(dataset_path) if dataset_path else ""
    if dataset_path is None or not dataset_path.is_file():
        spec.error = "academic writer requires its exact dataset_manifest_path"
        return
    try:
        dataset_bytes = dataset_path.read_bytes()
        dataset_manifest = json.loads(dataset_bytes)
    except (OSError, ValueError, TypeError) as exc:
        spec.error = f"invalid academic dataset manifest: {type(exc).__name__}"
        return
    if not isinstance(dataset_manifest, dict) or (
            dataset_manifest.get("schema_version") != ACADEMIC_DATASET_SCHEMA):
        spec.error = f"academic dataset manifest schema must be {ACADEMIC_DATASET_SCHEMA}"
        return
    if dataset_manifest.get("prompt_contract") != ACADEMIC_PROMPT_CONTRACT:
        spec.error = "academic dataset prompt contract does not match the adapter"
        return
    if dataset_manifest.get("completion_only_loss") is not True:
        spec.error = "academic dataset does not guarantee completion-only training"
        return
    if training.get("completion_only_loss") is not True:
        spec.error = "academic adapter training does not agree on completion-only loss"
        return
    if dataset_manifest.get("format") != "mlx_lm.completions":
        spec.error = "academic dataset must use the MLX completion-only format"
        return
    spec.dataset_manifest_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    expected_dataset_sha = str(dataset.get("manifest_sha256") or "").lower()
    if not expected_dataset_sha or spec.dataset_manifest_sha256 != expected_dataset_sha:
        spec.error = "academic dataset manifest SHA-256 does not match adapter identity"
        return
    spec.source_corpus_sha256 = str(dataset.get("source_corpus_sha256") or "").lower()
    if (not spec.source_corpus_sha256 or spec.source_corpus_sha256
            != str(dataset_manifest.get("source_corpus_sha256") or "").lower()):
        spec.error = "academic source-corpus canonical digest does not match dataset identity"
        return
    spec.source_corpus_file_sha256 = str(
        dataset.get("source_corpus_file_sha256") or "").lower()
    if (not spec.source_corpus_file_sha256 or spec.source_corpus_file_sha256
            != str(corpus_manifest.get("corpus_sha256") or "").lower()):
        spec.error = "academic source-corpus file digest does not match corpus manifest"
        return
    if str(dataset_manifest.get("source_corpus_file_sha256") or "").lower() != (
            spec.source_corpus_file_sha256):
        spec.error = "academic dataset source-corpus bytes do not match adapter identity"
        return
    if str(dataset_manifest.get("source_corpus_manifest_sha256") or "").lower() != (
            spec.corpus_manifest_sha256):
        spec.error = "academic dataset corpus-manifest identity does not match training"
        return
    dataset_corpus_manifest = resolved_path(
        dataset_manifest.get("source_corpus_manifest"), relative_to=dataset_path.parent)
    if dataset_corpus_manifest != corpus_path:
        spec.error = "academic dataset points at a different corpus manifest"
        return
    source_path = resolved_path(
        dataset_manifest.get("source_corpus"), relative_to=dataset_path.parent)
    spec.source_corpus_path = str(source_path) if source_path else ""
    if source_path is None or not source_path.is_file():
        spec.error = "academic source corpus is missing"
        return
    if file_sha256(source_path) != spec.source_corpus_file_sha256:
        spec.error = "academic source corpus changed after its adapter was trained"
        return
    if str(corpus_manifest.get("output_filename") or "") != source_path.name:
        spec.error = "academic corpus manifest does not identify the trained source corpus"
        return
    prepared_splits = dataset_manifest.get("splits")
    expected_prepared_splits = ("train", "valid", "test")
    if not isinstance(prepared_splits, dict) or set(prepared_splits) != set(
            expected_prepared_splits):
        spec.error = "academic dataset requires exact train/valid/test splits"
        return
    seen_task_types = set()
    for split_name in expected_prepared_splits:
        split = prepared_splits.get(split_name)
        if not isinstance(split, dict) or not positive_int(split.get("count")):
            spec.error = f"academic dataset {split_name} split must be nonempty"
            return
        split_count = split["count"]
        split_strata = split.get("source_strata")
        if (not isinstance(split_strata, dict)
                or set(split_strata) != set(required_strata)
                or not all(positive_int(split_strata.get(name))
                           for name in required_strata)
                or sum(split_strata.values()) != split_count):
            spec.error = (
                f"academic dataset {split_name} must contain all three exact source strata")
            return
        split_tasks = split.get("task_types")
        if (not isinstance(split_tasks, dict) or not split_tasks
                or not set(split_tasks).issubset({"sentence", "paragraph"})
                or not all(positive_int(value) for value in split_tasks.values())
                or sum(split_tasks.values()) != split_count):
            spec.error = (
                f"academic dataset {split_name} exceeds sentence/paragraph plan-prose tasks")
            return
        seen_task_types.update(split_tasks)
    if seen_task_types != {"sentence", "paragraph"}:
        spec.error = "academic dataset must preserve both sentence and paragraph tasks"
        return

    if not all((spec.profile_id, spec.runtime_model, spec.provider, spec.base_url,
                spec.base_model_id, spec.base_model_revision, spec.base_model_type,
                spec.base_model_architecture, spec.base_model_config_sha256,
                spec.base_weight_index_sha256,
                spec.base_weight_inventory_sha256, spec.adapter_format)):
        spec.error = "academic writer manifest has incomplete model/provider identity"
        return
    if spec.transport_adapter != ACADEMIC_TRANSPORT_ADAPTER:
        spec.error = "only the explicit openai-compatible academic transport is supported"
        return
    expected_route_name = f"academic-writer::{spec.profile_id}"
    route_name = configured(
        "SPIRAL_ACADEMIC_WRITER_ROUTE_NAME", "route_name", expected_route_name)
    if route_name != expected_route_name:
        spec.error = "academic writer route_name is identity-pinned and cannot be overridden"
        return
    if route_name in {
            cfg.worker.name, cfg.planner.name, cfg.escalation.name,
            cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name}:
        spec.error = "academic writer route_name must not collide with an orchestration role"
        return
    spec.name = route_name
    spec.overrides = tuple(sorted(set(override_fields)))

    spec.runtime_identity = {
        "schema_version": ACADEMIC_RUNTIME_IDENTITY_SCHEMA,
        "manifest_sha256": spec.manifest_sha256,
        "adapter_tree_sha256": spec.adapter_sha256,
        "base_model_id": spec.base_model_id,
        "base_model_revision": spec.base_model_revision,
        "base_weight_inventory_sha256": spec.base_weight_inventory_sha256,
        "profile_id": spec.profile_id,
        "provider": spec.provider,
        "model": spec.runtime_model,
        "transport_adapter": spec.transport_adapter,
        "adapter_strength_supported": True,
        "adapter_strength_min": ACADEMIC_ADAPTER_STRENGTH_MIN,
        "adapter_strength_max": ACADEMIC_ADAPTER_STRENGTH_MAX,
        "adapter_strength_step": ACADEMIC_ADAPTER_STRENGTH_STEP,
        "adapter_strength_default": ACADEMIC_ADAPTER_STRENGTH,
        "server_contract": ACADEMIC_SERVER_CONTRACT,
        "weight_residency": ACADEMIC_WEIGHT_RESIDENCY,
        "compute_lease": ACADEMIC_COMPUTE_LEASE,
        "ollama_admission": ACADEMIC_OLLAMA_ADMISSION,
        "unload_boundary": ACADEMIC_UNLOAD_BOUNDARY,
    }

    providers = dict(cfg.providers) if isinstance(cfg.providers, dict) else {}
    provider_entry = {
        "base_url": spec.base_url,
        "model": spec.runtime_model,
        "provider": spec.provider,
        "transport_adapter": spec.transport_adapter,
        "adapter_strength": spec.adapter_strength,
        "api_key_env": spec.api_key_env,
        "api_key_required": spec.api_key_required,
        "academic_profile_id": spec.profile_id,
        "academic_manifest_sha256": spec.manifest_sha256,
        "academic_adapter_sha256": spec.adapter_sha256,
        "academic_corpus_manifest_sha256": spec.corpus_manifest_sha256,
        "academic_dataset_manifest_sha256": spec.dataset_manifest_sha256,
        "academic_source_corpus_sha256": spec.source_corpus_sha256,
        "academic_source_corpus_file_sha256": spec.source_corpus_file_sha256,
        "required_runtime_identity": dict(spec.runtime_identity),
        "academic_writer_only": True,
    }
    existing = providers.get(route_name)
    if existing is not None and existing != provider_entry:
        spec.error = "academic writer route_name already maps to a different provider identity"
        return
    providers[route_name] = provider_entry
    cfg.providers = providers
    spec.ready = True


@dataclass
class Config:
    # backend seam — local-first. "ollama" today; another provider could slot in.
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    # Complexity chooses a finite JOINT wall/token/call envelope. It is a policy
    # tier, not a promise to spend the allowance.
    complexity_tier: str = "standard"
    prefer_single_resident_model: bool = True

    # Conductor/worker: qwen3.8:27b = the 3.8-gen 27B dense VLM (Q4_K_M GGUF).
    # RAM: ~17GB weights + mmproj + small hybrid KV @24k → ~19GB footprint,
    # inside the Metal working set with headroom for gradle + macOS on 32GB.
    # Slower per token than the old 3.6 MoE (~3B active), but a full generation
    # stronger on coding benchmarks — verification, not speed, is the bottleneck.
    planner: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.8:27b", num_ctx=24576, think=True)
    )
    worker: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.8:27b", num_ctx=24576, think=False)
    )
    # The retry lane is the SAME model thinking harder, because thinking is a
    # per-request flag, not a different set of weights. A second ollama name for
    # it would be one model wearing two hats — the ledger tells the lanes apart
    # by the `lane` field it records, not by guessing from the model name.
    escalation: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.8:27b", num_ctx=16384, think=True)
    )
    # Raw alternative critic for users who explicitly set
    # prefer_single_resident_model=false. The normal Config.load path aliases local
    # judging roles to the worker model so a default run never loads both models.
    # The 2026-07-27 A/B showed
    # a same-family critic rubber-stamps (six passes on a zero-mapping plan);
    # gemma3:12b found 8 real defects on the identical input in less time.
    # Thinking stays ON — llm.py retries without `think` for the models that
    # reject the toggle, so the role keeps its intent and gemma still answers.
    critic: ModelSpec = field(
        default_factory=lambda: ModelSpec("gemma3:12b", num_ctx=24576, think=True)
    )
    # Independent research adjudicator. Its contract is independence from the
    # PROVIDER, not the family: CLI API tiers deliberately leave it local so a
    # frontier worker never grades its own proposals. It must also stay a
    # different model from `critic` — research_loop collapses the audit into
    # the critic lane when the two names match, which silently costs the
    # escalated re-check of basis and claim scope.
    research_auditor: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.8:27b", num_ctx=16384, think=True)
    )
    # Optional publication-only route. It never participates in planning, tools,
    # Builder, or judging, and remains inert until an authenticated manifest plus an
    # explicit serving endpoint are configured.
    academic_writer: AcademicWriterSpec = field(
        default_factory=lambda: AcademicWriterSpec("", num_ctx=24576, think=False)
    )

    # The abliterated seat, used only when a run asks for it with --uncensored.
    # Never a default: refusal removal costs some instruction-following precision,
    # and the judging seats stay stock so a run is not graded by the model whose
    # checks were removed.
    uncensored_model: str = "qwen3.8:27b-uncensored"

    def spec_for(self, model_name: str) -> ModelSpec:
        """The ModelSpec whose name matches — so per-model num_ctx follows the
        model wherever it's used (worker vs escalation lanes)."""
        for spec in (
            self.worker, self.escalation, self.planner, self.critic,
            self.research_auditor, self.academic_writer, self.janitor,
        ):
            if spec.name == model_name:
                return spec
        return self.worker
    # optional janitor for later phases (compaction / done-checks)
    janitor: ModelSpec = field(
        default_factory=lambda: ModelSpec("llama3.2:1b", num_ctx=8192, think=False)
    )

    # model residency: how long Ollama keeps a model loaded after a request.
    # Without this, the 5-min idle default unloads mid-gradle-verify and every
    # attempt pays a 30-60s reload.
    keep_alive: str = "45m"

    # budgets — the guardrails that keep an autonomous run bounded
    # output cap per reply (num_predict) — NOT the context window (that's num_ctx,
    # set by `tune`). A ceiling, not a target: a short reply still ends early, so
    # a higher cap only rescues replies that would otherwise truncate mid-block.
    worker_max_tokens: int = 8192
    planner_max_tokens: int = 16384    # thinking + a whole-app plan; thinking alone can eat 8k
    task_attempt_budget: int = 6       # edit→verify cycles before escalation
    escalation_attempts: int = 4       # extra cycles on the stronger model
    bootstrap_attempts: int = 12       # first-green repair gets a longer leash
    plan_rounds: int = 3               # lint→critic→repair cycles before execution
    validate_rounds: int = 8           # max validate→remediate cycles; stops early on a true plateau
    # Joint orchestration ceilings include every role and every provider. Local
    # inference consumes time, energy, and attention; it is never treated as free.
    run_wall_budget_seconds: int = 2 * 60 * 60
    run_token_budget: int = 350_000
    run_call_budget: int = 96
    # Optional Builder-specific stop, finite by default and never interpreted as
    # "unlimited" when set to zero by an old configuration.
    builder_token_budget: int = 350_000
    verify_timeout: int = 900          # seconds; real build gates (gradle) are slow
    # Best-of-N is opt-in. Deterministic evidence and a focused repair beat routine
    # brute-force sampling on a single-machine 27B deployment. 0 disables.
    diversity_samples: int = 0

    # Worker research: repo/file/web/browser ASKs do not consume edit attempts.
    # For all action-count limits in this section, zero means unlimited. Web is
    # GET-only through spiral.research and each result is persisted under
    # .spiral/research/ for audit.
    ask_budget: int = 32
    web_research: bool = True
    web_research_budget: int = 24
    web_research_k: int = 8
    builder_repo_auto: bool = True
    builder_repo_budget: int = 3
    builder_repo_max_mb: int = 500
    # Remote package code is acquired automatically, but arbitrary lifecycle/build
    # hooks stay off unless the user explicitly accepts that execution boundary.
    builder_allow_install_scripts: bool = False
    builder_tool_auto: bool = True
    builder_tool_install_budget: int = 6
    builder_shell_budget: int = 24
    builder_vision_budget: int = 8
    builder_browser_budget: int = 8
    builder_shell_timeout: int = 300
    builder_require_sandbox: bool = True
    # Exact, user-approved files/directories that Builder may inspect as
    # untrusted, read-only reference material.  This is deliberately runtime-only:
    # Config.load() never grants persistent paths from config.json.
    builder_reference_roots: list[str] = field(default_factory=list)
    # Full-access build: the model's shell reaches the whole machine — writes any
    # path, network on, may modify spiral's own installed source — with only the
    # catastrophic backstop plus the separately approved Git-action boundary left
    # standing.
    # Off by default and never implied; a run turns it on with --full-access.
    builder_full_access: bool = False
    vision_model: str = ""
    visual_review: bool = True
    visual_review_url: str = ""
    visual_review_rounds: int = 3
    visual_review_timeout: int = 45
    product_audit_rounds: int = 3
    finish_rounds: int = 4
    builder_remediation_batch: int = 6
    builder_remediation_attempts: int = 3
    builder_remediation_escalation_attempts: int = 2
    # Vision-capable thinking models can consume a 2k allowance before emitting
    # their JSON defect report; keep enough room for reasoning plus the verdict.
    visual_review_max_tokens: int = 8192
    research_repo_auto: bool = False
    research_repo_budget: int = 1
    research_repo_max_mb: int = 750
    research_cleanup_failed_repos: bool = True
    research_tool_auto: bool = True
    research_tool_install_budget: int = 4
    # Public scientific data is acquired only by the typed Research data broker.
    # The model never receives an unrestricted networked shell. The broker resolves
    # catalog metadata first, computes the exact selected byte total, keeps a disk
    # reserve, resumes partial files, hashes content, and records licences/versions.
    research_data_auto: bool = True
    research_data_catalog_limit: int = 18
    research_data_max_gb: float = 20.0
    research_data_reserve_gb: float = 8.0
    research_data_file_limit: int = 20_000
    research_data_timeout: int = 3600
    research_data_sources: list[str] = field(
        default_factory=lambda: ["openneuro", "allen", "neuromaps", "zenodo"])
    research_notes_model: str = ""
    research_search_results_per_query: int = 8
    research_reading_limit: int = 60
    research_deep_read_limit: int = 8
    research_deep_chunk_limit: int = 10
    research_min_grounded_notes: int = 6
    research_min_grounded_deep_reads: int = 2
    research_min_papers: int = 10
    research_min_usable_texts: int = 6
    research_min_relevant_papers: int = 5
    research_min_relevant_usable_primary_texts: int = 4
    research_min_unique_queries: int = 3
    research_min_healthy_searches: int = 2
    research_min_relevant_query_families: int = 2
    research_min_topic_term_coverage: float = 0.45
    research_min_graph_success_rate: float = 0.60
    # Epistemic kernel and discovery policy. These gates are strict by default for
    # original research and are bypassed only by explicit expository verification mode.
    research_obligation_graph: bool = True
    research_blind_replication: bool = True
    research_replication_attempts: int = 2
    research_counterfactuals: bool = True
    research_counterfactual_limit: int = 6
    research_information_scheduler: bool = True
    research_information_gain_floor: float = 0.04
    research_plateau_patience: int = 8
    # Stalled rounds (flat corpus, dead instruments, only instrument checks failing)
    # before discovery degrades explicitly instead of vetoing forever.
    research_stall_patience: int = 3
    research_taste_model: bool = True
    research_git: bool = True
    research_living_papers: bool = True
    research_living_recheck_days: int = 30

    # user-defined extra gate welded into every task's verify (your own linter,
    # tests, anything) — set "extra_gate" in ~/.config/spiral/config.json
    extra_gate: str = ""

    # remote OpenAI-compatible providers, keyed by model id. Any role set to one
    # of these model ids is dispatched to the endpoint instead of Ollama. API keys
    # live in env vars (api_key_env), never here. e.g.:
    #   "providers": {"kimi-k3": {"base_url": "https://api.moonshot.ai/v1",
    #                             "api_key_env": "MOONSHOT_API_KEY", "temperature": 1}}
    providers: dict = field(default_factory=dict)

    # theme — clay brand + a hacker triad mapped to verify-loop states
    clay: str = "#D97757"          # brand / prompt / the mark
    spiral_style: str = "spiral"   # banner shape: spiral · galaxy · uzumaki
    live_green: str = "#35f0a0"    # tests green / task committed
    working_amber: str = "#ffb000" # generating / verifying
    fail_red: str = "#ff5c57"      # verify failed / stuck

    @classmethod
    def load(cls) -> "Config":
        """Defaults → config-file overlay → env vars. Models are fully swappable
        without touching code:

          env:   SPIRAL_WORKER / SPIRAL_PLANNER / SPIRAL_ESCALATION /
                 SPIRAL_CRITIC / SPIRAL_JANITOR / SPIRAL_BASE_URL
                 SPIRAL_ACADEMIC_WRITER_ENABLED / SPIRAL_ACADEMIC_WRITER_MANIFEST /
                 SPIRAL_ACADEMIC_WRITER_BASE_URL
          file:  ~/.config/spiral/config.json →
                 {"models": {"worker": "...", ...}, "num_ctx": {...}, "hooks": {...}}
        """
        cfg = cls()
        overlay = {}
        config_file = None
        try:
            import json
            import os
            from pathlib import Path

            roles = {
                "planner": cfg.planner,
                "worker": cfg.worker,
                "escalation": cfg.escalation,
                "critic": cfg.critic,
                "research_auditor": cfg.research_auditor,
                "janitor": cfg.janitor,
            }

            config_file = Path.home() / ".config" / "spiral" / "config.json"
            overlay = json.loads(config_file.read_text()) if config_file.is_file() else {}
            for role, name in overlay.get("models", {}).items():
                if role in roles:
                    roles[role].name = str(name)
            for role, spec in roles.items():
                env = os.environ.get(f"SPIRAL_{role.upper()}")
                if env:
                    spec.name = env
                if spec.name in overlay.get("num_ctx", {}):
                    spec.num_ctx = int(overlay["num_ctx"][spec.name])
            cfg.base_url = os.environ.get("SPIRAL_BASE_URL", overlay.get("base_url", cfg.base_url))
            cfg.complexity_tier = str(os.environ.get(
                "SPIRAL_COMPLEXITY", overlay.get("complexity_tier", cfg.complexity_tier)
            )).lower()
            from spiral.execution import BudgetLimits

            tier_limits = BudgetLimits.for_tier(cfg.complexity_tier)
            cfg.run_wall_budget_seconds = int(os.environ.get(
                "SPIRAL_RUN_WALL_SECONDS",
                overlay.get("run_wall_budget_seconds", tier_limits.wall_seconds),
            ))
            cfg.run_token_budget = int(os.environ.get(
                "SPIRAL_RUN_TOKEN_BUDGET",
                overlay.get("run_token_budget", tier_limits.total_tokens),
            ))
            cfg.run_call_budget = int(os.environ.get(
                "SPIRAL_RUN_CALL_BUDGET",
                overlay.get("run_call_budget", tier_limits.model_calls),
            ))
            # Old zero/unlimited values are migrated to the joint finite ceiling.
            if cfg.run_wall_budget_seconds <= 0:
                cfg.run_wall_budget_seconds = tier_limits.wall_seconds
            if cfg.run_token_budget <= 0:
                cfg.run_token_budget = tier_limits.total_tokens
            if cfg.run_call_budget <= 0:
                cfg.run_call_budget = tier_limits.model_calls
            cfg.prefer_single_resident_model = bool(overlay.get(
                "prefer_single_resident_model", cfg.prefer_single_resident_model))
            cfg.uncensored_model = os.environ.get(
                "SPIRAL_UNCENSORED_MODEL",
                overlay.get("uncensored_model", cfg.uncensored_model))
            cfg.extra_gate = overlay.get("extra_gate", cfg.extra_gate)
            cfg.spiral_style = os.environ.get("SPIRAL_STYLE", overlay.get("style", cfg.spiral_style))
            cfg.worker_max_tokens = int(overlay.get("worker_max_tokens", cfg.worker_max_tokens))
            cfg.builder_token_budget = int(
                overlay.get("builder_token_budget", cfg.run_token_budget))
            if cfg.builder_token_budget <= 0:
                cfg.builder_token_budget = cfg.run_token_budget
            cfg.diversity_samples = int(overlay.get("diversity_samples", cfg.diversity_samples))
            cfg.ask_budget = int(overlay.get("ask_budget", cfg.ask_budget))
            cfg.web_research = bool(overlay.get("web_research", cfg.web_research))
            cfg.web_research_budget = int(overlay.get("web_research_budget", cfg.web_research_budget))
            cfg.web_research_k = int(overlay.get("web_research_k", cfg.web_research_k))
            cfg.builder_repo_auto = bool(
                overlay.get("builder_repo_auto", cfg.builder_repo_auto))
            cfg.builder_repo_budget = int(
                overlay.get("builder_repo_budget", cfg.builder_repo_budget))
            cfg.builder_repo_max_mb = int(
                overlay.get("builder_repo_max_mb", cfg.builder_repo_max_mb))
            cfg.builder_allow_install_scripts = bool(overlay.get(
                "builder_allow_install_scripts", cfg.builder_allow_install_scripts))
            cfg.builder_tool_auto = bool(overlay.get(
                "builder_tool_auto", cfg.builder_tool_auto))
            cfg.builder_tool_install_budget = int(overlay.get(
                "builder_tool_install_budget", cfg.builder_tool_install_budget))
            cfg.builder_shell_budget = int(overlay.get(
                "builder_shell_budget", cfg.builder_shell_budget))
            cfg.builder_vision_budget = int(overlay.get(
                "builder_vision_budget", cfg.builder_vision_budget))
            cfg.builder_browser_budget = int(overlay.get(
                "builder_browser_budget", cfg.builder_browser_budget))
            cfg.builder_shell_timeout = int(overlay.get(
                "builder_shell_timeout", cfg.builder_shell_timeout))
            cfg.builder_require_sandbox = bool(overlay.get(
                "builder_require_sandbox", cfg.builder_require_sandbox))
            cfg.builder_full_access = bool(overlay.get(
                "builder_full_access", cfg.builder_full_access))
            cfg.vision_model = os.environ.get("SPIRAL_VISION", overlay.get("vision_model", cfg.vision_model))
            cfg.visual_review = bool(overlay.get("visual_review", cfg.visual_review))
            cfg.visual_review_url = os.environ.get(
                "SPIRAL_VISUAL_URL", overlay.get("visual_review_url", cfg.visual_review_url))
            cfg.visual_review_rounds = int(overlay.get("visual_review_rounds", cfg.visual_review_rounds))
            cfg.visual_review_timeout = int(overlay.get("visual_review_timeout", cfg.visual_review_timeout))
            cfg.product_audit_rounds = int(
                overlay.get("product_audit_rounds", cfg.product_audit_rounds))
            cfg.finish_rounds = int(overlay.get("finish_rounds", cfg.finish_rounds))
            cfg.builder_remediation_batch = int(overlay.get(
                "builder_remediation_batch", cfg.builder_remediation_batch))
            cfg.builder_remediation_attempts = int(overlay.get(
                "builder_remediation_attempts", cfg.builder_remediation_attempts))
            cfg.builder_remediation_escalation_attempts = int(overlay.get(
                "builder_remediation_escalation_attempts",
                cfg.builder_remediation_escalation_attempts))
            cfg.visual_review_max_tokens = int(
                overlay.get("visual_review_max_tokens", cfg.visual_review_max_tokens))
            cfg.research_repo_auto = bool(overlay.get("research_repo_auto", cfg.research_repo_auto))
            cfg.research_repo_budget = int(overlay.get("research_repo_budget", cfg.research_repo_budget))
            cfg.research_repo_max_mb = int(overlay.get("research_repo_max_mb", cfg.research_repo_max_mb))
            cfg.research_cleanup_failed_repos = bool(
                overlay.get("research_cleanup_failed_repos", cfg.research_cleanup_failed_repos))
            cfg.research_tool_auto = bool(
                overlay.get("research_tool_auto", cfg.research_tool_auto))
            cfg.research_tool_install_budget = int(overlay.get(
                "research_tool_install_budget",
                cfg.research_tool_install_budget))
            cfg.research_data_auto = bool(overlay.get(
                "research_data_auto", cfg.research_data_auto))
            cfg.research_data_catalog_limit = int(overlay.get(
                "research_data_catalog_limit", cfg.research_data_catalog_limit))
            cfg.research_data_max_gb = float(overlay.get(
                "research_data_max_gb", cfg.research_data_max_gb))
            cfg.research_data_reserve_gb = float(overlay.get(
                "research_data_reserve_gb", cfg.research_data_reserve_gb))
            cfg.research_data_file_limit = int(overlay.get(
                "research_data_file_limit", cfg.research_data_file_limit))
            cfg.research_data_timeout = int(overlay.get(
                "research_data_timeout", cfg.research_data_timeout))
            configured_sources = overlay.get(
                "research_data_sources", cfg.research_data_sources)
            if isinstance(configured_sources, list):
                cfg.research_data_sources = [
                    str(source).strip().lower() for source in configured_sources
                    if str(source).strip()
                ]
            cfg.research_notes_model = os.environ.get(
                "SPIRAL_RESEARCH_NOTES_MODEL",
                overlay.get("research_notes_model", cfg.research_notes_model),
            )
            cfg.research_search_results_per_query = int(overlay.get(
                "research_search_results_per_query",
                cfg.research_search_results_per_query))
            cfg.research_reading_limit = int(
                overlay.get("research_reading_limit", cfg.research_reading_limit))
            cfg.research_deep_read_limit = int(
                overlay.get("research_deep_read_limit", cfg.research_deep_read_limit))
            cfg.research_deep_chunk_limit = int(
                overlay.get("research_deep_chunk_limit", cfg.research_deep_chunk_limit))
            cfg.research_min_grounded_notes = int(overlay.get(
                "research_min_grounded_notes", cfg.research_min_grounded_notes))
            cfg.research_min_grounded_deep_reads = int(overlay.get(
                "research_min_grounded_deep_reads", cfg.research_min_grounded_deep_reads))
            cfg.research_min_papers = int(
                overlay.get("research_min_papers", cfg.research_min_papers))
            cfg.research_min_usable_texts = int(
                overlay.get("research_min_usable_texts", cfg.research_min_usable_texts))
            cfg.research_min_relevant_papers = int(
                overlay.get("research_min_relevant_papers", cfg.research_min_relevant_papers))
            cfg.research_min_relevant_usable_primary_texts = int(overlay.get(
                "research_min_relevant_usable_primary_texts",
                cfg.research_min_relevant_usable_primary_texts))
            cfg.research_min_unique_queries = int(
                overlay.get("research_min_unique_queries", cfg.research_min_unique_queries))
            cfg.research_min_healthy_searches = int(
                overlay.get("research_min_healthy_searches", cfg.research_min_healthy_searches))
            cfg.research_min_relevant_query_families = int(overlay.get(
                "research_min_relevant_query_families",
                cfg.research_min_relevant_query_families))
            cfg.research_min_topic_term_coverage = float(overlay.get(
                "research_min_topic_term_coverage", cfg.research_min_topic_term_coverage))
            cfg.research_min_graph_success_rate = float(overlay.get(
                "research_min_graph_success_rate", cfg.research_min_graph_success_rate))
            cfg.research_obligation_graph = bool(overlay.get(
                "research_obligation_graph", cfg.research_obligation_graph))
            cfg.research_blind_replication = bool(overlay.get(
                "research_blind_replication", cfg.research_blind_replication))
            cfg.research_replication_attempts = int(overlay.get(
                "research_replication_attempts", cfg.research_replication_attempts))
            cfg.research_counterfactuals = bool(overlay.get(
                "research_counterfactuals", cfg.research_counterfactuals))
            cfg.research_counterfactual_limit = int(overlay.get(
                "research_counterfactual_limit", cfg.research_counterfactual_limit))
            cfg.research_information_scheduler = bool(overlay.get(
                "research_information_scheduler", cfg.research_information_scheduler))
            cfg.research_information_gain_floor = float(overlay.get(
                "research_information_gain_floor", cfg.research_information_gain_floor))
            cfg.research_plateau_patience = int(overlay.get(
                "research_plateau_patience", cfg.research_plateau_patience))
            cfg.research_stall_patience = int(overlay.get(
                "research_stall_patience", cfg.research_stall_patience))
            cfg.research_taste_model = bool(overlay.get(
                "research_taste_model", cfg.research_taste_model))
            cfg.research_git = bool(overlay.get("research_git", cfg.research_git))
            cfg.research_living_papers = bool(overlay.get(
                "research_living_papers", cfg.research_living_papers))
            cfg.research_living_recheck_days = int(overlay.get(
                "research_living_recheck_days", cfg.research_living_recheck_days))
            cfg.providers = overlay.get("providers", cfg.providers)
        except Exception:
            pass  # a broken overlay must never break spiral
        # Academic serving is a separate, fail-closed route: an invalid or incomplete
        # manifest disables only that optional writer and leaves the established writer
        # untouched. The resolver performs no model or network access.
        try:
            _academic_writer_config(cfg, overlay, config_file)
        except Exception as exc:
            cfg.academic_writer.ready = False
            cfg.academic_writer.error = (
                f"academic writer configuration failed: {type(exc).__name__}")
        # This safety policy deliberately runs *after* the broad compatibility
        # catch. A malformed config must fall back to one resident local model,
        # not silently restore the historical qwen+gemma+llama RAM pile-up.
        if cfg.prefer_single_resident_model:
            providers = cfg.providers if isinstance(cfg.providers, dict) else {}
            local_specs = [
                spec for spec in (
                    cfg.worker, cfg.planner, cfg.escalation, cfg.critic,
                    cfg.research_auditor, cfg.janitor,
                ) if spec.name not in providers
            ]
            if local_specs:
                preferred = (
                    cfg.worker.name if cfg.worker.name not in providers
                    else local_specs[0].name
                )
                for spec in local_specs:
                    spec.name = preferred
        return cfg
