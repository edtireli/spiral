"""Pure/offline contracts for the opt-in academic publication writer."""

from __future__ import annotations

import ast
import hashlib
import io
import inspect
import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest
from rich.console import Console

from spiral.cli import _apply_tier, _apply_uncensored
from spiral.config import (
    ACADEMIC_ADAPTER_FORMAT,
    ACADEMIC_ADAPTER_STRENGTH,
    ACADEMIC_ADAPTER_STRENGTH_MAX,
    ACADEMIC_ADAPTER_STRENGTH_MIN,
    ACADEMIC_ADAPTER_STRENGTH_STEP,
    ACADEMIC_ADAPTER_SCHEMA,
    ACADEMIC_AUTHOR_SAFE_SPLIT_POLICY,
    ACADEMIC_BASE_MODEL_ARCHITECTURE,
    ACADEMIC_BASE_MODEL_CONFIG_SHA256,
    ACADEMIC_BASE_MODEL_ID,
    ACADEMIC_BASE_MODEL_REVISION,
    ACADEMIC_BASE_MODEL_TYPE,
    ACADEMIC_BASE_WEIGHT_FILES,
    ACADEMIC_BASE_WEIGHT_INDEX_SHA256,
    ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256,
    ACADEMIC_COMPUTE_LEASE,
    ACADEMIC_CORPUS_SCHEMA,
    ACADEMIC_DATASET_SCHEMA,
    ACADEMIC_OLLAMA_ADMISSION,
    ACADEMIC_PROFILE_ID,
    ACADEMIC_PROMPT_CONTRACT,
    ACADEMIC_PROVIDER,
    ACADEMIC_RUNTIME_IDENTITY_SCHEMA,
    ACADEMIC_RUNTIME_MODEL,
    ACADEMIC_SERVER_CONTRACT,
    ACADEMIC_SOURCE_STRATA,
    ACADEMIC_TRANSPORT_ADAPTER,
    ACADEMIC_UNLOAD_BOUNDARY,
    ACADEMIC_WEIGHT_RESIDENCY,
    Config,
)
from spiral.llm import ChatResult, Ollama
from spiral.research_loop import ResearchLoop
from scripts.academic_finetune.serve_adapter import SERVER_LIFECYCLE_IDENTITY


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_route_fixture(
    root: Path, *, tamper_adapter: bool = False,
    source_strata=ACADEMIC_SOURCE_STRATA,
    corpus_variant: str = "",
    dataset_variant: str = "",
    manifest_variant: str = "",
    completion_only_training: bool = True,
) -> tuple[Path, Path]:
    bundle = root / "academic-adapter"
    bundle.mkdir(parents=True)
    payloads = {
        "adapter_config.json": b'{"rank":16,"alpha":32}\n',
        "adapters.safetensors": b"deterministic-test-weights",
    }
    required_files = []
    digest_lines = []
    for relative in ("adapter_config.json", "adapters.safetensors"):
        data = payloads[relative]
        (bundle / relative).write_bytes(data)
        digest = _sha256(data)
        required_files.append({
            "path": relative,
            "size_bytes": len(data),
            "sha256": digest,
        })
        digest_lines.append(f"{relative}\0{len(data)}\0{digest}\n")
    bundle_digest = _sha256("".join(sorted(digest_lines)).encode("utf-8"))

    source_corpus = root / "academic_corpus.jsonl"
    source_corpus.write_bytes(
        b'{"schema_version":"spiral.academic-plan-prose.v1","id":"one"}\n')
    source_corpus_file_sha = _sha256(source_corpus.read_bytes())
    source_corpus_sha = _sha256(b"canonical-record-digest")
    corpus_manifest = root / "academic_corpus.jsonl.manifest.json"
    corpus_splits = ("train", "validation", "test")
    coverage = {
        f"{name}|{split}": 1
        for name in source_strata
        for split in corpus_splits
    }
    corpus_payload = {
        "schema_version": ACADEMIC_CORPUS_SCHEMA,
        "corpus_schema_version": ACADEMIC_PROMPT_CONTRACT,
        "source_strata": sorted(source_strata),
        "balanced_sources": True,
        "trainable": True,
        "non_trainable_reasons": [],
        "counts": {
            "examples": 18,
            "documents": 9,
            "by_split": {name: 6 for name in corpus_splits},
            "by_stratum": {name: 6 for name in source_strata},
            "by_task_type": {"sentence": 9, "paragraph": 9},
            "documents_by_stratum_split": dict(coverage),
            "examples_by_stratum_split": {
                key: 2 for key in coverage},
        },
        "cutoff": "2021-12-31",
        "corpus_sha256": source_corpus_file_sha,
        "output_filename": source_corpus.name,
        "split_policy": ACADEMIC_AUTHOR_SAFE_SPLIT_POLICY,
        "split_diagnostics": {
            "components": 9,
            "largest_component_examples": 2,
            "component_counts_by_split": {
                "train": 3, "validation": 3, "test": 3},
        },
        "task_feasibility": {
            "sentence": "2-4 shallow semantic role slots",
            "paragraph": "one compact proposition slot per sentence",
        },
    }
    if corpus_variant == "not_trainable":
        corpus_payload["trainable"] = False
        corpus_payload["non_trainable_reasons"] = ["pilot only"]
    elif corpus_variant == "late_cutoff":
        corpus_payload["cutoff"] = "2022-01-01"
    elif corpus_variant == "unsafe_split":
        corpus_payload["split_policy"] = "random examples"
    elif corpus_variant == "missing_split_diagnostics":
        corpus_payload.pop("split_diagnostics")
    elif corpus_variant == "zero_validation":
        corpus_payload["counts"]["by_split"]["validation"] = 0
    elif corpus_variant == "missing_stratum_split":
        corpus_payload["counts"]["examples_by_stratum_split"].pop(
            "pubmed|test", None)
    corpus_manifest.write_text(
        json.dumps(corpus_payload, sort_keys=True) + "\n", encoding="utf-8")
    corpus_sha = _sha256(corpus_manifest.read_bytes())

    dataset_manifest = root / "dataset_manifest.json"
    dataset_payload = {
        "schema_version": ACADEMIC_DATASET_SCHEMA,
        "prompt_contract": ACADEMIC_PROMPT_CONTRACT,
        "source_corpus": str(source_corpus),
        "source_corpus_sha256": source_corpus_sha,
        "source_corpus_file_sha256": source_corpus_file_sha,
        "source_corpus_manifest": str(corpus_manifest),
        "source_corpus_manifest_sha256": corpus_sha,
        "format": "mlx_lm.completions",
        "completion_only_loss": True,
        "splits": {
            name: {
                "path": f"{name}.jsonl",
                "count": 6,
                "sha256": _sha256(name.encode()),
                "source_strata": {name: 2 for name in source_strata},
                "task_types": {"sentence": 3, "paragraph": 3},
            }
            for name in ("train", "valid", "test")
        },
    }
    if dataset_variant == "not_completion_only":
        dataset_payload["completion_only_loss"] = False
    elif dataset_variant == "zero_test":
        dataset_payload["splits"]["test"]["count"] = 0
    elif dataset_variant == "missing_valid_stratum":
        del dataset_payload["splits"]["valid"]["source_strata"]["pubmed"]
        dataset_payload["splits"]["valid"]["count"] = 4
        dataset_payload["splits"]["valid"]["task_types"] = {
            "sentence": 2, "paragraph": 2}
    dataset_manifest.write_text(
        json.dumps(dataset_payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest = root / "academic-adapter.manifest.json"
    from scripts.academic_finetune.training_support import build_adapter_manifest

    emitted = build_adapter_manifest(
        config={
            "training": {
                "seed": 17,
                "batch_size": 1,
                "grad_accumulation_steps": 1,
                "grad_checkpoint": True,
                "mask_prompt": completion_only_training,
            },
            "lora": {"keys": ["self_attn.q_proj"]},
        },
        base_receipt={
            "model_id": ACADEMIC_BASE_MODEL_ID,
            "revision": ACADEMIC_BASE_MODEL_REVISION,
            "model_type": ACADEMIC_BASE_MODEL_TYPE,
            "architecture": ACADEMIC_BASE_MODEL_ARCHITECTURE,
            "config_sha256": ACADEMIC_BASE_MODEL_CONFIG_SHA256,
            "weight_index_sha256": ACADEMIC_BASE_WEIGHT_INDEX_SHA256,
            "weight_inventory_sha256": ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256,
            "weight_files": [dict(row) for row in ACADEMIC_BASE_WEIGHT_FILES],
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            "target_inventory": {
                "selected_layers": 8,
                "target_path_counts": {"self_attn.q_proj": 8},
                "total_target_modules": 8,
            },
        },
        dataset_manifest=json.loads(dataset_manifest.read_text(encoding="utf-8")),
        dataset_manifest_path=dataset_manifest,
        adapter_manifest_path=manifest,
        adapter_dir=bundle,
        bundle_digest=bundle_digest,
        required_files=required_files,
        package_versions={"mlx": "test", "mlx_lm": "test"},
    )
    if manifest_variant:
        section, field, value = {
            "profile": (None, "profile_id", "different-profile"),
            "runtime_model": ("runtime", "model", "other-model"),
            "provider": ("runtime", "provider", "other-provider"),
            "transport": ("runtime", "transport_adapter", "other-transport"),
            "base_id": ("base_model", "model_id", "other/base"),
            "revision": ("base_model", "revision", "b" * 40),
            "model_type": ("base_model", "model_type", "other_type"),
            "architecture": ("base_model", "architecture", "OtherArchitecture"),
            "config_sha256": ("base_model", "config_sha256", "b" * 64),
            "weight_index_sha256": (
                "base_model", "weight_index_sha256", "b" * 64),
            "weight_inventory_sha256": (
                "base_model", "weight_inventory_sha256", "b" * 64),
            "quantization": (
                "base_model", "quantization", {"bits": 8, "group_size": 64}),
            "adapter_format": ("adapter", "format", "other_adapter"),
            "default_strength": ("runtime", "default_adapter_strength", 1.25),
        }[manifest_variant]
        target = emitted if section is None else emitted[section]
        target[field] = value
    manifest.write_text(json.dumps(emitted, sort_keys=True), encoding="utf-8")
    if tamper_adapter:
        (bundle / "adapters.safetensors").write_bytes(b"changed-after-manifest")
    return manifest, corpus_manifest


def _load_enabled_config(
    tmp_path: Path, monkeypatch, *, tamper_adapter: bool = False,
    source_strata=ACADEMIC_SOURCE_STRATA,
    corpus_variant: str = "", dataset_variant: str = "",
    manifest_variant: str = "", completion_only_training: bool = True,
    config_overrides: dict | None = None,
    env_overrides: dict | None = None,
) -> Config:
    manifest, _ = _write_route_fixture(
        tmp_path / "artifacts", tamper_adapter=tamper_adapter,
        source_strata=source_strata, corpus_variant=corpus_variant,
        dataset_variant=dataset_variant, manifest_variant=manifest_variant,
        completion_only_training=completion_only_training)
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    academic_config = {
            "enabled": True,
            "manifest_path": str(manifest),
            "base_url": "http://127.0.0.1:8080/v1",
    }
    academic_config.update(config_overrides or {})
    (config_dir / "config.json").write_text(json.dumps({
        "academic_writer": academic_config,
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    for name, value in (env_overrides or {}).items():
        monkeypatch.setenv(name, value)
    return Config.load()


def _load_manifest_config(
    root: Path, manifest: Path, monkeypatch,
) -> Config:
    config_dir = root / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "academic_writer": {
            "enabled": True,
            "manifest_path": str(manifest),
            "base_url": "http://127.0.0.1:8080/v1",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(root))
    return Config.load()


def _relocated_route_fixture(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "offline-volume"
    _write_route_fixture(origin)
    staged = tmp_path / "staged"
    shutil.copytree(origin, staged)
    shutil.rmtree(origin)
    # Inference does not consume training examples.  Leave only the two staged,
    # manifest-bound provenance receipts and the adapter bundle.
    (staged / "academic_corpus.jsonl").unlink()
    return staged / "academic-adapter.manifest.json", staged


def test_academic_writer_is_off_by_default_and_outside_residency_aliasing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config.load()

    assert not cfg.academic_writer.enabled
    assert not cfg.academic_writer.ready
    assert not any(name.startswith("academic-writer::") for name in cfg.providers)
    assert {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name,
        cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name,
    } == {cfg.worker.name}


def test_manifest_pins_exact_mlx_adapter_provider_and_corpus_identity(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    spec = cfg.academic_writer

    assert spec.enabled and spec.ready, spec.error
    assert spec.profile_id == ACADEMIC_PROFILE_ID
    assert spec.runtime_model == ACADEMIC_RUNTIME_MODEL
    assert spec.provider == ACADEMIC_PROVIDER
    assert spec.adapter_strength == ACADEMIC_ADAPTER_STRENGTH
    assert spec.base_url == "http://127.0.0.1:8080/v1"
    assert spec.source_strata == tuple(sorted(ACADEMIC_SOURCE_STRATA))
    assert [row["path"] for row in spec.adapter_required_files] == [
        "adapter_config.json", "adapters.safetensors"]
    provider = cfg.providers[spec.name]
    assert provider["model"] == spec.runtime_model
    assert provider["adapter_strength"] == ACADEMIC_ADAPTER_STRENGTH
    assert provider["base_url"] == spec.base_url
    assert provider["academic_adapter_sha256"] == spec.adapter_sha256
    assert provider["academic_corpus_manifest_sha256"] == spec.corpus_manifest_sha256
    assert provider["academic_dataset_manifest_sha256"] == spec.dataset_manifest_sha256
    assert provider["academic_source_corpus_file_sha256"] == spec.source_corpus_file_sha256
    assert spec.adapter_format == ACADEMIC_ADAPTER_FORMAT
    assert spec.transport_adapter == ACADEMIC_TRANSPORT_ADAPTER
    assert spec.base_model_config_sha256 == ACADEMIC_BASE_MODEL_CONFIG_SHA256
    assert spec.base_weight_index_sha256 == ACADEMIC_BASE_WEIGHT_INDEX_SHA256
    assert spec.base_weight_inventory_sha256 == ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256
    assert spec.base_weight_files == ACADEMIC_BASE_WEIGHT_FILES
    assert provider["required_runtime_identity"] == spec.runtime_identity
    assert spec.runtime_identity == {
        "schema_version": ACADEMIC_RUNTIME_IDENTITY_SCHEMA,
        "manifest_sha256": spec.manifest_sha256,
        "adapter_tree_sha256": spec.adapter_sha256,
        "base_model_id": ACADEMIC_BASE_MODEL_ID,
        "base_model_revision": ACADEMIC_BASE_MODEL_REVISION,
        "base_weight_inventory_sha256": spec.base_weight_inventory_sha256,
        "profile_id": ACADEMIC_PROFILE_ID,
        "provider": ACADEMIC_PROVIDER,
        "model": ACADEMIC_RUNTIME_MODEL,
        "transport_adapter": ACADEMIC_TRANSPORT_ADAPTER,
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
    assert {
        key: spec.runtime_identity[key]
        for key in SERVER_LIFECYCLE_IDENTITY
    } == SERVER_LIFECYCLE_IDENTITY
    assert spec.name not in {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name,
        cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name,
    }


def test_relocated_staged_provenance_does_not_require_training_volume(
        tmp_path, monkeypatch):
    manifest, staged = _relocated_route_fixture(tmp_path)

    cfg = _load_manifest_config(tmp_path / "home", manifest, monkeypatch)

    spec = cfg.academic_writer
    assert spec.ready, spec.error
    assert Path(spec.corpus_manifest_path) == (
        staged / "academic_corpus.jsonl.manifest.json")
    assert Path(spec.dataset_manifest_path) == staged / "dataset_manifest.json"
    assert spec.source_corpus_path.startswith(str(tmp_path / "offline-volume"))
    assert not Path(spec.source_corpus_path).exists()


def test_relocated_staged_dataset_tamper_fails_closed(tmp_path, monkeypatch):
    manifest, staged = _relocated_route_fixture(tmp_path)
    dataset_path = staged / "dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["source_corpus_sha256"] = "0" * 64
    dataset_path.write_text(json.dumps(dataset, sort_keys=True) + "\n", encoding="utf-8")

    cfg = _load_manifest_config(tmp_path / "home", manifest, monkeypatch)

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert "dataset manifest SHA-256" in cfg.academic_writer.error


def test_manifest_relative_corpus_filename_must_match_output_identity(
        tmp_path, monkeypatch):
    manifest, staged = _relocated_route_fixture(tmp_path)
    wrong_name = staged / "wrong-corpus.manifest.json"
    shutil.copyfile(staged / "academic_corpus.jsonl.manifest.json", wrong_name)
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    receipt["training"]["corpus_manifest_path"] = wrong_name.name
    manifest.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")

    cfg = _load_manifest_config(tmp_path / "home", manifest, monkeypatch)

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert "filename" in cfg.academic_writer.error


def test_mutated_adapter_bundle_fails_closed_to_existing_writer(tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch, tamper_adapter=True)

    assert cfg.academic_writer.enabled
    assert not cfg.academic_writer.ready
    assert "adapter" in cfg.academic_writer.error.lower()
    assert cfg.academic_writer.name not in cfg.providers


def test_corpus_profile_must_explicitly_cover_all_three_source_strata(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch,
        source_strata=("arxiv:hep-th", "arxiv:hep-ph"))

    assert cfg.academic_writer.enabled
    assert not cfg.academic_writer.ready
    assert "source_strata" in cfg.academic_writer.error
    assert cfg.academic_writer.name not in cfg.providers


@pytest.mark.parametrize(("variant", "error_fragment"), [
    ("not_trainable", "trainable"),
    ("late_cutoff", "cutoff"),
    ("unsafe_split", "author-safe"),
    ("missing_split_diagnostics", "author-component"),
    ("zero_validation", "train/validation/test"),
    ("missing_stratum_split", "every source stratum"),
])
def test_real_emitter_corpus_semantic_failures_are_rejected(
        tmp_path, monkeypatch, variant, error_fragment):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch, corpus_variant=variant)

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert error_fragment in cfg.academic_writer.error


@pytest.mark.parametrize(("dataset_variant", "completion_only", "error_fragment"), [
    ("not_completion_only", True, "completion-only"),
    ("zero_test", True, "test split"),
    ("missing_valid_stratum", True, "all three exact source strata"),
    ("", False, "agree on completion-only"),
])
def test_real_emitter_dataset_split_and_loss_contracts_are_rejected(
        tmp_path, monkeypatch, dataset_variant, completion_only, error_fragment):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch, dataset_variant=dataset_variant,
        completion_only_training=completion_only)

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert error_fragment in cfg.academic_writer.error


@pytest.mark.parametrize("manifest_variant", [
    "profile", "runtime_model", "provider", "transport", "base_id", "revision",
    "model_type", "architecture", "config_sha256", "weight_index_sha256",
    "weight_inventory_sha256", "quantization", "adapter_format",
    "default_strength",
])
def test_real_emitter_identity_fields_cannot_be_relabelled(
        tmp_path, monkeypatch, manifest_variant):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch, manifest_variant=manifest_variant)

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert cfg.academic_writer.error
    assert not any(
        entry.get("academic_writer_only")
        for entry in cfg.providers.values()
        if isinstance(entry, dict))


def test_config_and_environment_cannot_override_manifest_model_identity(
        tmp_path, monkeypatch):
    config_override = _load_enabled_config(
        tmp_path / "config", monkeypatch,
        config_overrides={"model": "silent-substitute"})
    assert not config_override.academic_writer.ready
    assert "cannot be overridden" in config_override.academic_writer.error

    env_override = _load_enabled_config(
        tmp_path / "env", monkeypatch,
        env_overrides={"SPIRAL_ACADEMIC_WRITER_PROVIDER": "silent-provider"})
    assert not env_override.academic_writer.ready
    assert "cannot be overridden" in env_override.academic_writer.error


def test_academic_strength_is_configurable_on_valid_grid_and_attested_per_provider(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch,
        env_overrides={"SPIRAL_ACADEMIC_WRITER_STRENGTH": "1.35"})

    assert cfg.academic_writer.ready, cfg.academic_writer.error
    assert cfg.academic_writer.adapter_strength == 1.35
    assert cfg.providers[cfg.academic_writer.name]["adapter_strength"] == 1.35
    assert "adapter_strength" in cfg.academic_writer.overrides
    assert "adapter_strength" not in cfg.academic_writer.runtime_identity


@pytest.mark.parametrize("strength", ["-0.05", "2.05", "0.333", "nan", "true"])
def test_academic_strength_rejects_out_of_range_or_off_grid_values(
        tmp_path, monkeypatch, strength):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch,
        env_overrides={"SPIRAL_ACADEMIC_WRITER_STRENGTH": strength})

    assert cfg.academic_writer.enabled and not cfg.academic_writer.ready
    assert "0.05 steps" in cfg.academic_writer.error


class _FakeModels:
    def __init__(
        self, cfg: Config, *, fail_academic: bool = False,
        fail_strict_eviction: bool = False,
        academic_reply: str = "academic prose",
        local_reply: str = "existing writer",
    ):
        self.providers = dict(cfg.providers)
        self.base_url = cfg.base_url
        self.fail_academic = fail_academic
        self.fail_strict_eviction = fail_strict_eviction
        self.academic_reply = academic_reply
        self.local_reply = local_reply
        self.calls = []
        self.evictions = []
        self.events = []

    def evict_owned_local_models_except(self, keep, log=None, *, strict=False):
        self.evictions.append(set(keep))
        self.events.append(("evict-owned", tuple(sorted(keep)), strict))
        if strict and self.fail_strict_eviction:
            raise RuntimeError("owned unload returned false")
        return []

    def chat(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        self.events.append(("chat", model))
        if self.fail_academic and model.startswith("academic-writer::"):
            return ChatResult(text="", raw={"error": "academic endpoint unavailable"})
        return ChatResult(
            text=(self.local_reply if model == "qwen3.8:27b" else self.academic_reply),
            prompt_tokens=2,
            completion_tokens=3,
            raw={"finish_reason": "stop"},
        )


def test_paper_prose_synthesis_uses_academic_route_but_ordinary_planning_and_repairs_do_not(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    models = _FakeModels(cfg)
    loop = ResearchLoop("routing test", workdir=tmp_path / "run", cfg=cfg, ol=models)

    prose, _ = loop._synthesize_prose(
        "write", "evidence", phase="abstract", fallback_role="worker")
    body, _ = loop._synthesize_prose(
        "write body", "evidence", phase="section-draft:Results",
        fallback_role="worker")
    compile_repair, _ = loop._synthesize_prose(
        "repair latex", "errors", phase="latex-compile-repair:1",
        fallback_role="worker")
    planned, _ = loop._think("plan", "task", role="planner")

    assert prose == "academic prose"
    assert body == "academic prose"
    assert compile_repair == "existing writer"
    assert planned == "existing writer"
    assert [call[0] for call in models.calls] == [
        cfg.academic_writer.name, cfg.academic_writer.name,
        cfg.worker.name, cfg.planner.name]
    assert models.calls[0][2]["num_predict"] == 8192
    assert models.calls[1][2]["num_predict"] == 8192
    assert models.evictions[:2] == [set(), set()]
    assert models.events[:2] == [
        ("evict-owned", (), True), ("chat", cfg.academic_writer.name)]


def test_paper_only_blueprint_and_outline_use_same_academic_writer(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    proposal = {
        "title": "A bounded paper",
        "sections": [
            {
                "name": "Introduction", "rhetorical_role": "introduction",
                "intent": "motivate the problem", "target_words": 300,
            },
            {
                "name": "Analysis", "rhetorical_role": "results",
                "intent": "establish the result", "target_words": 900,
            },
        ],
    }
    models = _FakeModels(cfg, academic_reply=json.dumps(proposal))
    loop = ResearchLoop("paper planning route", workdir=tmp_path / "run", cfg=cfg, ol=models)

    planned = loop._synthesize_paper_plan_json(
        "Plan this paper", "verified evidence",
        phase="paper-outline", required=("title", "sections"),
    )
    ordinary, _ = loop._think("Plan research", "ordinary task", role="planner")

    assert planned == proposal
    assert ordinary == "existing writer"
    assert [call[0] for call in models.calls] == [
        cfg.academic_writer.name, cfg.planner.name]
    assert models.calls[0][2]["num_predict"] == 8192
    assert models.evictions == [set(), {cfg.planner.name}]


def test_paper_outline_budget_preserves_adapter_emphasis_and_exact_total():
    proposal = {
        "sections": [
            {"name": "Introduction", "target_words": 200},
            {"name": "Core result", "target_words": 600},
            {"name": "Discussion", "target_words": 200},
        ],
    }
    outline = {
        "title": "T",
        "sections": [
            {"name": "Introduction", "rhetorical_role": "introduction"},
            {"name": "Core result", "rhetorical_role": "results"},
            {"name": "Discussion", "rhetorical_role": "discussion"},
        ],
    }

    budgeted = ResearchLoop._apply_paper_outline_budget(
        outline, proposal, target_words=1800)

    words = [row["target_words"] for row in budgeted["sections"]]
    assert sum(words) == 1800
    assert words[1] > words[0] == words[2]
    assert budgeted["paper_outline_budget"] == {
        "requested_words": 1800,
        "proposed_words": 1000,
        "adapter_supplied_weights": True,
        "normalization": "deterministic_largest_remainder",
    }


def test_malformed_academic_paper_plan_visibly_retries_ordinary_planner(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    fallback = {
        "title": "Conservative fallback",
        "sections": [{
            "name": "Results", "rhetorical_role": "results",
            "intent": "state the bounded result", "target_words": 900,
        }],
    }
    models = _FakeModels(
        cfg,
        academic_reply="not a JSON object",
        local_reply=json.dumps(fallback),
    )
    loop = ResearchLoop("paper plan fallback", workdir=tmp_path / "run", cfg=cfg, ol=models)

    planned = loop._synthesize_paper_plan_json(
        "Plan this paper", "verified evidence",
        phase="paper-outline", required=("title", "sections"),
    )

    assert planned == fallback
    assert [call[0] for call in models.calls] == [
        cfg.academic_writer.name, cfg.planner.name]
    thoughts = [json.loads(line) for line in (
        tmp_path / "run" / "thoughts.jsonl").read_text().splitlines()]
    assert thoughts[-1]["phase"] == "academic-writer-plan-fallback"


def test_explicit_uncensored_mode_marks_process_and_bypasses_academic_prose(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    models = _FakeModels(cfg)
    console = Console(file=io.StringIO(), force_terminal=False)

    class Installed:
        def models(self):
            return [cfg.uncensored_model]

    for env_name in (
            "SPIRAL_UNCENSORED_ACTIVE", "SPIRAL_WORKER",
            "SPIRAL_PLANNER", "SPIRAL_ESCALATION"):
        monkeypatch.setenv(env_name, "")
    monkeypatch.setattr(Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr("spiral.cli.Ollama", lambda *_args, **_kwargs: Installed())
    _apply_uncensored(console)
    loop = ResearchLoop("uncensored route", workdir=tmp_path / "run", cfg=cfg, ol=models)
    text, _ = loop._synthesize_prose(
        "write", "evidence", phase="section-draft:Results",
        fallback_role="worker")

    assert os.environ["SPIRAL_UNCENSORED_ACTIVE"] == "1"
    assert text == "existing writer"
    assert [model for model, _, _ in models.calls] == [cfg.worker.name]
    assert models.evictions == [{cfg.worker.name}]


def test_publication_pipeline_routes_only_paper_planning_content_and_semantic_revisions():
    tree = ast.parse(textwrap.dedent(inspect.getsource(ResearchLoop.write)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_synthesize_prose"
    ]
    assert len(calls) == 11
    phases = [
        next(keyword.value for keyword in call.keywords if keyword.arg == "phase")
        for call in calls
    ]
    phase_labels = {
        value.value if isinstance(value, ast.Constant)
        else value.values[0].value
        for value in phases
        if (isinstance(value, ast.Constant)
            or (isinstance(value, ast.JoinedStr)
                and value.values
                and isinstance(value.values[0], ast.Constant)))
    }
    assert phase_labels == {
        "section-draft:", "body-coherence-revision", "referee-revision:",
        "strong-referee-revision:", "citation-support-revision:",
        "evidence-regression-revision:", "claim-scope-revision:",
        "citation-regression-revision:", "final-referee-revision:",
        "abstract", "abstract-scope-repair:",
    }
    source = textwrap.dedent(inspect.getsource(ResearchLoop.write))
    assert source.count("self._synthesize_paper_plan_json(") == 2
    assert 'phase="paper-writing-blueprint"' in source
    assert 'phase="paper-outline"' in source
    assert "self._academic_structure_outline(" not in source
    assert "self._think_json(" in source
    assert 'phase="latex-compile' not in source


def test_api_tiers_never_promote_publication_only_route_into_build_roles(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    original_roles = (cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name)
    console = Console(file=io.StringIO(), force_terminal=False)

    with pytest.raises(SystemExit, match="academic routes do not count"):
        _apply_tier(cfg, console, "api")
    assert (cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name) == original_roles

    cfg.providers["general-api"] = {
        "base_url": "https://provider.invalid/v1",
        "api_key_env": "GENERAL_TEST_KEY",
    }
    _apply_tier(cfg, console, "api")
    assert {cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name} == {
        "general-api"}
    assert cfg.academic_writer.name != "general-api"


def test_academic_failure_is_audited_then_visibly_falls_back_without_hidden_swap(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    models = _FakeModels(cfg, fail_academic=True)
    loop = ResearchLoop("fallback test", workdir=tmp_path / "run", cfg=cfg, ol=models)
    messages = []
    monkeypatch.setattr(loop, "_say", messages.append)

    text, _ = loop._synthesize_prose(
        "write", "evidence", phase="abstract", fallback_role="worker")

    assert text == "existing writer"
    assert [model for model, _, _ in models.calls] == [
        cfg.academic_writer.name, cfg.academic_writer.name, cfg.worker.name]
    assert any("academic writer unavailable" in message for message in messages)
    audit = [json.loads(line) for line in (
        tmp_path / "run" / "model-calls.jsonl").read_text().splitlines()]
    assert audit[0]["role"] == "academic_writer"
    assert audit[0]["route_identity"]["runtime_model"] == "qwen3.8-27b-academic"
    assert audit[-1]["role"] == "worker"


def test_owned_unload_failure_blocks_academic_dispatch_and_visibly_falls_back(
        tmp_path, monkeypatch):
    cfg = _load_enabled_config(tmp_path, monkeypatch)
    models = _FakeModels(cfg, fail_strict_eviction=True)
    loop = ResearchLoop("handoff test", workdir=tmp_path / "run", cfg=cfg, ol=models)
    messages = []
    monkeypatch.setattr(loop, "_say", messages.append)

    text, _ = loop._synthesize_prose(
        "write", "evidence", phase="abstract", fallback_role="worker")

    assert text == "existing writer"
    assert [model for model, _, _ in models.calls] == [cfg.worker.name]
    assert not any(model == cfg.academic_writer.name for model, _, _ in models.calls)
    assert models.events[0] == ("evict-owned", (), True)
    assert any("academic writer unavailable" in message for message in messages)
    thought_rows = [json.loads(line) for line in (
        tmp_path / "run" / "thoughts.jsonl").read_text().splitlines()]
    assert thought_rows[-1]["phase"] == "academic-writer-fallback"
    assert "release run-owned" in thought_rows[-1]["text"]


def test_openai_compatible_local_route_can_be_explicitly_keyless(monkeypatch):
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    client = Ollama(providers={"academic-route": {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "exact-academic-model",
        "api_key_env": "",
        "api_key_required": False,
    }})
    fake = Client()
    client._client = fake

    result = client.chat(
        "academic-route", [{"role": "user", "content": "draft"}],
        think=False, num_predict=1024)

    assert result.text == "ok"
    body = fake.calls[0][1]["json"]
    headers = fake.calls[0][1]["headers"]
    assert body["model"] == "exact-academic-model"
    assert "Authorization" not in headers


@pytest.mark.parametrize(
    "identity_mode", ["exact", "missing", "mismatch", "stock-legacy", "extra"])
def test_academic_endpoint_output_requires_exact_runtime_attestation(
        tmp_path, monkeypatch, identity_mode):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch,
        config_overrides={"base_url": "http://127.0.0.1:18080/v1"})
    spec = cfg.academic_writer
    assert spec.ready, spec.error
    assert spec.overrides == ("base_url",)

    observed = dict(spec.runtime_identity)
    if identity_mode == "mismatch":
        observed["base_model_revision"] = "b" * 40
    elif identity_mode == "extra":
        observed["unreviewed_capability"] = "enabled"
    elif identity_mode == "stock-legacy":
        for field in (
            "server_contract", "weight_residency", "compute_lease",
            "ollama_admission", "unload_boundary",
        ):
            observed.pop(field)

    class Response:
        status_code = 200

        def json(self):
            payload = {
                "choices": [{
                    "message": {"content": "attested academic prose"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
            if identity_mode != "missing":
                payload["spiral_runtime_identity"] = observed
            payload["spiral_adapter_strength"] = spec.adapter_strength
            return payload

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    client = Ollama(providers={spec.name: cfg.providers[spec.name]})
    fake = Client()
    client._client = fake
    result = client.chat(
        spec.name, [{"role": "user", "content": "draft abstract"}],
        think=False, num_predict=1024)

    assert fake.calls[0][0] == "http://127.0.0.1:18080/v1/chat/completions"
    assert fake.calls[0][1]["json"]["adapter_strength"] == spec.adapter_strength
    if identity_mode == "exact":
        assert result.text == "attested academic prose"
        assert result.raw["spiral_runtime_identity"] == spec.runtime_identity
    else:
        assert result.text == ""
        assert result.raw["status"] == 409
        assert "identity" in str(result.raw["error"])


@pytest.mark.parametrize("observed_strength", [None, 0.95, 1.05, True])
def test_academic_endpoint_requires_exact_per_request_strength_echo(
        tmp_path, monkeypatch, observed_strength):
    cfg = _load_enabled_config(
        tmp_path, monkeypatch,
        env_overrides={"SPIRAL_ACADEMIC_WRITER_STRENGTH": "1.25"})
    spec = cfg.academic_writer
    assert spec.ready, spec.error

    class Response:
        status_code = 200

        def json(self):
            payload = {
                "choices": [{"message": {"content": "prose"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                "spiral_runtime_identity": spec.runtime_identity,
            }
            if observed_strength is not None:
                payload["spiral_adapter_strength"] = observed_strength
            return payload

    class Client:
        def post(self, *_args, **_kwargs):
            return Response()

    client = Ollama(providers={spec.name: cfg.providers[spec.name]})
    client._client = Client()
    result = client.chat(
        spec.name, [{"role": "user", "content": "draft"}],
        think=False, num_predict=1024)

    assert result.text == ""
    assert result.raw["status"] == 409
    assert "strength" in result.raw["error"]
