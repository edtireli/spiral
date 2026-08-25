"""Offline isolation and fallback contracts for the paper-structure planner."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from scripts.academic_finetune import serve_adapter
from spiral.cli import _apply_tier
from spiral.conductor import Conductor
from spiral.config import (
    ACADEMIC_ADAPTER_FORMAT,
    ACADEMIC_ADAPTER_LINEAGE_SCHEMA,
    ACADEMIC_ADAPTER_SCHEMA,
    ACADEMIC_ADAPTER_SHA256_SEMANTICS,
    ACADEMIC_BASE_MODEL_ARCHITECTURE,
    ACADEMIC_BASE_MODEL_CONFIG_SHA256,
    ACADEMIC_BASE_MODEL_ID,
    ACADEMIC_BASE_MODEL_REVISION,
    ACADEMIC_BASE_MODEL_TYPE,
    ACADEMIC_BASE_WEIGHT_FILES,
    ACADEMIC_BASE_WEIGHT_INDEX_SHA256,
    ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256,
    ACADEMIC_DATASET_SCHEMA,
    ACADEMIC_PARENT_ADAPTER_SCHEMA,
    ACADEMIC_PLANNER_PROFILE_ID,
    ACADEMIC_PLANNER_PROMPT_CONTRACT,
    ACADEMIC_PLANNER_RUNTIME_MODEL,
    ACADEMIC_PLANNER_SCOPE,
    ACADEMIC_PROVIDER,
    ACADEMIC_RUNTIME_IDENTITY_SCHEMA,
    ACADEMIC_SOURCE_STRATA,
    ACADEMIC_STRUCTURE_CORPUS_SCHEMA,
    ACADEMIC_STRUCTURE_LORA_KEYS,
    ACADEMIC_STRUCTURE_MANIFEST_SCHEMA,
    ACADEMIC_STRUCTURE_REPLAY_TASK,
    ACADEMIC_STRUCTURE_SPLIT_POLICY,
    ACADEMIC_STRUCTURE_TARGET_PATH_COUNTS,
    ACADEMIC_STRUCTURE_TASKS,
    ACADEMIC_STRUCTURE_TRAINABLE_LAYERS,
    ACADEMIC_TRANSPORT_ADAPTER,
    AcademicPlannerSpec,
    Config,
    general_api_providers,
)
from spiral.research_loop import ResearchLoop


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value) -> str:
    return _sha256((
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode())


def _lora_tensor_names() -> list[str]:
    targets_by_layer = {
        **{
            str(layer): [
                "linear_attn.in_proj_qkv", "linear_attn.out_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]
            for layer in (60, 61, 62)
        },
        "63": [
            "self_attn.q_proj", "self_attn.v_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
    }
    return sorted(
        f"language_model.model.layers.{layer}.{target}.{suffix}"
        for layer, targets in targets_by_layer.items()
        for target in targets
        for suffix in ("lora_a", "lora_b")
    )


def _write_lora_safetensors(path: Path, *, extra_tensor: bool = False) -> None:
    names = _lora_tensor_names()
    if extra_tensor:
        names.append("language_model.model.layers.0.mlp.up_proj.lora_a")
    header = {}
    offset = 0
    for name in sorted(names):
        shape = [2, 16] if name.endswith(".lora_a") else [16, 2]
        size = shape[0] * shape[1] * 4
        header[name] = {
            "dtype": "F32", "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _topology_sha256() -> str:
    return _canonical_sha256({
        "num_layers": 4,
        "keys": list(ACADEMIC_STRUCTURE_LORA_KEYS),
        "rank": 16,
        "scale": 32.0,
        "dropout": 0.0,
        "tensor_names": _lora_tensor_names(),
    })


def _write_structure_route(
    root: Path, *, corrupt_prompt: bool = False,
    corrupt_lineage: bool = False, missing_lineage: bool = False,
    corrupt_topology: bool = False, corrupt_tensor: bool = False,
) -> Path:
    root.mkdir(parents=True)
    adapter_dir = root / "structure-adapter"
    adapter_dir.mkdir()
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": 3 if corrupt_topology else 4,
        "lora_parameters": {
            "keys": list(ACADEMIC_STRUCTURE_LORA_KEYS),
            "rank": 16,
            "scale": 32.0,
            "dropout": 0.0,
        },
    }
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, sort_keys=True), encoding="utf-8")
    _write_lora_safetensors(
        adapter_dir / "adapters.safetensors", extra_tensor=corrupt_tensor)
    required_files = []
    digest_lines = []
    for name in ("adapter_config.json", "adapters.safetensors"):
        payload = (adapter_dir / name).read_bytes()
        digest = _sha256(payload)
        required_files.append({
            "path": name,
            "size_bytes": len(payload),
            "sha256": digest,
        })
        digest_lines.append(f"{name}\0{len(payload)}\0{digest}\n")
    adapter_sha = _sha256("".join(sorted(digest_lines)).encode())

    source = root / "paper-structure.jsonl"
    source.write_bytes(b'{"schema_version":"spiral.academic-paper-structure.v1"}\n')
    source_sha = _sha256(source.read_bytes())
    corpus_manifest = root / "paper-structure.jsonl.manifest.json"
    coverage = {
        f"{stratum}|{split}": 1
        for stratum in ACADEMIC_SOURCE_STRATA
        for split in ("train", "validation", "test")
    }
    task_counts = {name: 4 for name in ACADEMIC_STRUCTURE_TASKS}
    task_counts[ACADEMIC_STRUCTURE_REPLAY_TASK] = 6
    corpus = {
        "schema_version": ACADEMIC_STRUCTURE_MANIFEST_SCHEMA,
        "corpus_schema_version": ACADEMIC_STRUCTURE_CORPUS_SCHEMA,
        "prompt_contract": (
            "wrong.prompt" if corrupt_prompt else ACADEMIC_PLANNER_PROMPT_CONTRACT),
        "cutoff": "2021-12-31",
        "trainable": True,
        "non_trainable_reasons": [],
        "source_strata": sorted(ACADEMIC_SOURCE_STRATA),
        "corpus_sha256": source_sha,
        "output_filename": source.name,
        "intended_use": {
            "component": "spiral_research_paper_planner",
            "planner_only": True,
            "target_modality": "json_paper_architecture",
            "excluded_component": "spiralchat_general_conversation",
        },
        "split_policy": ACADEMIC_STRUCTURE_SPLIT_POLICY,
        "split_diagnostics": {
            "components": 9,
            "component_counts_by_split": {
                "train": 3, "validation": 3, "test": 3,
            },
        },
        "counts": {
            "examples": 30,
            "documents": 9,
            "prose_replay_ratio": 0.2,
            "by_task_type": task_counts,
            "documents_by_stratum_split": dict(coverage),
            "examples_by_stratum_split": dict(coverage),
        },
    }
    corpus_manifest.write_text(json.dumps(corpus, sort_keys=True), encoding="utf-8")
    corpus_manifest_sha = _sha256(corpus_manifest.read_bytes())

    split_tasks = {name: 1 for name in (*ACADEMIC_STRUCTURE_TASKS, ACADEMIC_STRUCTURE_REPLAY_TASK)}
    split_tasks["brief_to_blueprint"] = 4
    split_strata = {
        "arxiv:hep-th": 4,
        "arxiv:hep-ph": 3,
        "pubmed": 3,
    }
    splits = {}
    for split_name in ("train", "valid", "test"):
        split_path = root / f"{split_name}.jsonl"
        split_path.write_bytes(f"{split_name}-structure-data\n".encode())
        splits[split_name] = {
            "path": split_path.name,
            "count": 10,
            "sha256": _sha256(split_path.read_bytes()),
            "source_strata": dict(split_strata),
            "task_types": dict(split_tasks),
        }
    dataset_manifest = root / "dataset_manifest.json"
    canonical_source_sha = _sha256(b"canonical-structure-records")
    dataset = {
        "schema_version": ACADEMIC_DATASET_SCHEMA,
        "prompt_contract": ACADEMIC_PLANNER_PROMPT_CONTRACT,
        "profile_id": ACADEMIC_PLANNER_PROFILE_ID,
        "corpus_schema_version": ACADEMIC_STRUCTURE_CORPUS_SCHEMA,
        "source_corpus_manifest_schema": ACADEMIC_STRUCTURE_MANIFEST_SCHEMA,
        "source_corpus": str(source),
        "source_corpus_sha256": canonical_source_sha,
        "source_corpus_file_sha256": source_sha,
        "source_corpus_manifest": str(corpus_manifest),
        "source_corpus_manifest_sha256": corpus_manifest_sha,
        "format": "mlx_lm.completions",
        "completion_only_loss": True,
        "target_contract": {
            "structure_tasks": sorted(ACADEMIC_STRUCTURE_TASKS),
            "structure_target": "canonical_json_object",
            "prose_replay_task": ACADEMIC_STRUCTURE_REPLAY_TASK,
            "prose_replay_target": "nonempty_academic_prose",
        },
        "splits": splits,
    }
    dataset_manifest.write_text(json.dumps(dataset, sort_keys=True), encoding="utf-8")
    adapter_manifest = root / "structure-adapter.manifest.json"
    manifest = {
        "schema_version": ACADEMIC_ADAPTER_SCHEMA,
        "profile_id": ACADEMIC_PLANNER_PROFILE_ID,
        "prompt_contract": ACADEMIC_PLANNER_PROMPT_CONTRACT,
        "base_model": {
            "model_id": ACADEMIC_BASE_MODEL_ID,
            "revision": ACADEMIC_BASE_MODEL_REVISION,
            "model_type": ACADEMIC_BASE_MODEL_TYPE,
            "architecture": ACADEMIC_BASE_MODEL_ARCHITECTURE,
            "config_sha256": ACADEMIC_BASE_MODEL_CONFIG_SHA256,
            "weight_index_sha256": ACADEMIC_BASE_WEIGHT_INDEX_SHA256,
            "weight_inventory_sha256": ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256,
            "weight_files": [dict(row) for row in ACADEMIC_BASE_WEIGHT_FILES],
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
        "adapter": {
            "path": adapter_dir.name,
            "format": ACADEMIC_ADAPTER_FORMAT,
            "sha256": adapter_sha,
            "sha256_semantics": ACADEMIC_ADAPTER_SHA256_SEMANTICS,
            "required_files": required_files,
        },
        "training": {
            "completion_only_loss": True,
            "trainable_layers": list(ACADEMIC_STRUCTURE_TRAINABLE_LAYERS),
            "trainable_target_paths": list(ACADEMIC_STRUCTURE_LORA_KEYS),
            "target_path_counts": dict(ACADEMIC_STRUCTURE_TARGET_PATH_COUNTS),
            "total_target_modules": 20,
            "corpus_manifest_path": corpus_manifest.name,
            "corpus_manifest_sha256": corpus_manifest_sha,
        },
        "dataset": {
            "dataset_manifest_path": dataset_manifest.name,
            "manifest_sha256": _sha256(dataset_manifest.read_bytes()),
            "source_corpus_sha256": canonical_source_sha,
            "source_corpus_file_sha256": source_sha,
            "split_sha256": {
                name: row["sha256"] for name, row in splits.items()
            },
        },
        "environment": {"mlx": "test", "mlx_lm": "test"},
        "runtime": {
            "provider": ACADEMIC_PROVIDER,
            "model": ACADEMIC_PLANNER_RUNTIME_MODEL,
            "base_url": "http://127.0.0.1:8080/v1",
            "transport_adapter": ACADEMIC_TRANSPORT_ADAPTER,
            "default_adapter_strength": 1.0,
            "scope": ACADEMIC_PLANNER_SCOPE,
            "spiralchat_eligible": False,
        },
        "lineage": {
            "schema_version": ACADEMIC_ADAPTER_LINEAGE_SCHEMA,
            "generation": 1,
            "parent": {
                "schema_version": ACADEMIC_PARENT_ADAPTER_SCHEMA,
                "generation": 1,
                "manifest_sha256": "a" * 64,
                "profile_id": (
                    ACADEMIC_PLANNER_PROFILE_ID if corrupt_lineage
                    else "academic-hep-pubmed-v1"),
                "prompt_contract": (
                    ACADEMIC_PLANNER_PROMPT_CONTRACT if corrupt_lineage
                    else "spiral.academic-plan-prose.v1"),
                "base_weight_inventory_sha256": ACADEMIC_BASE_WEIGHT_INVENTORY_SHA256,
                "adapter_tree_sha256": "b" * 64,
                "adapter_config_sha256": "c" * 64,
                "adapter_weights_sha256": "d" * 64,
                "lora_topology_sha256": _topology_sha256(),
            },
        },
    }
    if missing_lineage:
        manifest.pop("lineage")
    adapter_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return adapter_manifest


def _load_planner_config(
    tmp_path: Path, monkeypatch, *, corrupt_prompt: bool = False,
    corrupt_lineage: bool = False, missing_lineage: bool = False,
    corrupt_topology: bool = False, corrupt_tensor: bool = False,
) -> Config:
    manifest = _write_structure_route(
        tmp_path / "artifacts", corrupt_prompt=corrupt_prompt,
        corrupt_lineage=corrupt_lineage, missing_lineage=missing_lineage,
        corrupt_topology=corrupt_topology, corrupt_tensor=corrupt_tensor)
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "academic_planner": {
            "enabled": True,
            "manifest_path": str(manifest),
            "base_url": "http://127.0.0.1:8181/v1",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    return Config.load()


def test_academic_planner_is_inert_by_default_and_has_a_separate_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    default = Config.load()
    assert not default.academic_planner.enabled
    assert not default.academic_planner.ready
    assert not any(name.startswith("academic-planner::") for name in default.providers)

    cfg = _load_planner_config(tmp_path / "configured", monkeypatch)
    spec = cfg.academic_planner
    assert spec.enabled and spec.ready, spec.error
    assert spec.name == f"academic-planner::{ACADEMIC_PLANNER_PROFILE_ID}"
    assert spec.name != cfg.academic_writer.name
    assert spec.runtime_model == ACADEMIC_PLANNER_RUNTIME_MODEL
    assert spec.prompt_contract == ACADEMIC_PLANNER_PROMPT_CONTRACT
    provider = cfg.providers[spec.name]
    assert provider["academic_planner_only"] is True
    assert provider["spiralchat_eligible"] is False
    assert provider["required_runtime_identity"]["scope"] == ACADEMIC_PLANNER_SCOPE
    assert provider["required_runtime_identity"]["schema_version"] == (
        ACADEMIC_RUNTIME_IDENTITY_SCHEMA)
    assert spec.name not in {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name,
        cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name,
    }


def test_config_identity_exactly_matches_planner_server_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load_planner_config(tmp_path, monkeypatch)
    monkeypatch.setattr(serve_adapter, "_validate_model_view", lambda *_args: None)

    assets = serve_adapter.validate_runtime_assets(
        Path(cfg.academic_planner.manifest_path), tmp_path / "unused-model-view")

    assert assets.identity == cfg.academic_planner.runtime_identity


def test_reserved_planner_aliases_are_scrubbed_even_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = f"academic-planner::{ACADEMIC_PLANNER_PROFILE_ID}"
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "models": {"worker": route},
        "providers": {
            route: {"academic_planner_only": True, "base_url": "https://stale.invalid"},
            "forged-planner": {
                "academic_planner_only": True, "base_url": "https://forged.invalid",
            },
            "general-api": {"base_url": "https://general.invalid"},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = Config.load()

    assert not cfg.academic_planner.enabled and not cfg.academic_planner.ready
    assert route not in cfg.providers
    assert "forged-planner" not in cfg.providers
    assert cfg.worker.name != route
    assert set(general_api_providers(cfg.providers)) == {"general-api"}


def test_reserved_planner_alias_is_scrubbed_when_manifest_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = f"academic-planner::{ACADEMIC_PLANNER_PROFILE_ID}"
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "providers": {
            route: {"academic_planner_only": True, "base_url": "https://stale.invalid"},
        },
        "academic_planner": {
            "enabled": True,
            "manifest_path": str(tmp_path / "missing-manifest.json"),
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = Config.load()

    assert cfg.academic_planner.enabled and not cfg.academic_planner.ready
    assert "readable manifest_path" in cfg.academic_planner.error
    assert route not in cfg.providers


def test_general_api_selection_excludes_planner_only_routes() -> None:
    cfg = Config()
    route = f"academic-planner::{ACADEMIC_PLANNER_PROFILE_ID}"
    cfg.providers = {route: {"academic_planner_only": True}}
    original = (cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name)
    console = Console(file=io.StringIO(), force_terminal=False)

    with pytest.raises(SystemExit, match="academic routes do not count"):
        _apply_tier(cfg, console, "api")
    assert (cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name) == original
    assert "general_api_providers" in inspect.getsource(Conductor.consult)

    cfg.providers["general-api"] = {"base_url": "https://general.invalid/v1"}
    _apply_tier(cfg, console, "api")
    assert {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name, cfg.critic.name,
    } == {"general-api"}


def test_invalid_structure_contract_fails_closed_without_mutating_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load_planner_config(tmp_path, monkeypatch, corrupt_prompt=True)
    assert cfg.academic_planner.enabled and not cfg.academic_planner.ready
    assert "contract" in cfg.academic_planner.error
    assert cfg.academic_planner.name not in cfg.providers
    assert not cfg.academic_writer.enabled and not cfg.academic_writer.ready


@pytest.mark.parametrize(
    ("option", "error_fragment"),
    [
        ("missing_lineage", "requires authenticated lineage"),
        ("corrupt_lineage", "parent ancestry"),
        ("corrupt_topology", "four-layer LoRA topology"),
        ("corrupt_tensor", "tensor inventory"),
    ],
)
def test_lineage_and_exact_lora_topology_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    option: str, error_fragment: str,
) -> None:
    cfg = _load_planner_config(tmp_path, monkeypatch, **{option: True})
    assert cfg.academic_planner.enabled and not cfg.academic_planner.ready
    assert error_fragment in cfg.academic_planner.error
    assert cfg.academic_planner.name not in cfg.providers


def _planner_provider(spec: AcademicPlannerSpec) -> dict:
    return {
        "base_url": spec.base_url,
        "model": spec.runtime_model,
        "provider": spec.provider,
        "transport_adapter": spec.transport_adapter,
        "adapter_strength": spec.adapter_strength,
        "api_key_env": spec.api_key_env,
        "api_key_required": spec.api_key_required,
        "academic_planner_profile_id": spec.profile_id,
        "academic_planner_manifest_sha256": spec.manifest_sha256,
        "academic_planner_adapter_sha256": spec.adapter_sha256,
        "academic_planner_lineage_sha256": spec.lineage_sha256,
        "academic_planner_lora_topology_sha256": spec.lora_topology_sha256,
        "academic_planner_corpus_manifest_sha256": spec.corpus_manifest_sha256,
        "academic_planner_dataset_manifest_sha256": spec.dataset_manifest_sha256,
        "academic_planner_source_corpus_sha256": spec.source_corpus_sha256,
        "academic_planner_source_corpus_file_sha256": spec.source_corpus_file_sha256,
        "required_runtime_identity": dict(spec.runtime_identity),
        "academic_planner_only": True,
        "spiralchat_eligible": False,
    }


class _Models:
    def __init__(self, providers: dict, replies: dict[str, object]) -> None:
        self.providers = providers
        self.base_url = "http://127.0.0.1:11434"
        self.replies = replies
        self.calls: list[str] = []
        self.evictions: list[tuple[set[str], bool]] = []

    def evict_owned_local_models_except(self, keep, log=None, strict=False):
        self.evictions.append((set(keep), bool(strict)))
        return []

    def chat(self, model, messages, **kwargs):
        self.calls.append(model)
        text = self.replies.get(model, "")
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(
            text=text,
            thinking="",
            prompt_tokens=3,
            completion_tokens=5,
            raw={"finish_reason": "stop"},
        )


def _loop(tmp_path: Path, *, planner_reply: str, fallback_reply: str = "{}"):
    cfg = Config()
    spec = AcademicPlannerSpec(
        name=f"academic-planner::{ACADEMIC_PLANNER_PROFILE_ID}",
        num_ctx=8192,
        think=False,
        enabled=True,
        ready=True,
        profile_id=ACADEMIC_PLANNER_PROFILE_ID,
        runtime_model=ACADEMIC_PLANNER_RUNTIME_MODEL,
        provider=ACADEMIC_PROVIDER,
        base_url="http://127.0.0.1:8181/v1",
        transport_adapter=ACADEMIC_TRANSPORT_ADAPTER,
        manifest_sha256="1" * 64,
        adapter_sha256="2" * 64,
        corpus_manifest_sha256="3" * 64,
        dataset_manifest_sha256="4" * 64,
        source_corpus_sha256="5" * 64,
        source_corpus_file_sha256="6" * 64,
        lineage_sha256="7" * 64,
        lora_topology_sha256="8" * 64,
        parent_adapter_identity={"generation": 1},
        runtime_identity={"scope": ACADEMIC_PLANNER_SCOPE},
    )
    cfg.academic_planner = spec
    models = _Models(
        {spec.name: _planner_provider(spec)},
        {spec.name: planner_reply, cfg.planner.name: fallback_reply},
    )
    loop = object.__new__(ResearchLoop)
    loop.cfg = cfg
    loop.ol = models
    loop.dir = tmp_path
    loop.state = SimpleNamespace(
        round=0, tokens=0, api_tokens=0, local_tokens=0,
    )
    loop._model_call_hash = ""
    loop._thought_hash = ""
    loop._say = lambda _message: None
    loop._prepare_owned_local_model = lambda _model, _label: []
    thoughts = []
    loop._log_thought = lambda phase, text, **extra: thoughts.append((phase, text, extra))
    return loop, models, thoughts


def _valid_reply() -> str:
    return json.dumps({
        "paper_counts": {
            "abstract_words": 180,
            "section_words": 1000,
            "section_paragraphs": 20,
            "unsectioned_words": 0,
            "figures": 2,
            "tables": 1,
        },
        "sections": [
            {"heading": "Introduction", "id": "s1", "role": "introduction", "words": 180},
            {"heading": "Formalism", "id": "s2", "role": "formalism", "words": 270},
            {"heading": "Analysis", "id": "s3", "role": "analysis", "words": 370},
            {"heading": "Conclusions", "id": "s4", "role": "conclusion", "words": 180},
        ],
    })


def test_structure_call_is_isolated_and_budget_conversion_is_deterministic(tmp_path: Path) -> None:
    loop, models, thoughts = _loop(tmp_path, planner_reply=_valid_reply())
    blueprint = {"section_template": []}
    receipt = {}
    first = loop._academic_structure_outline(
        "structure only", "paper brief",
        deterministic_blueprint=blueprint,
        target_words=1800,
        title_seed="A bounded test",
        route_receipt=receipt,
    )
    second = loop._academic_structure_outline(
        "structure only", "paper brief",
        deterministic_blueprint=blueprint,
        target_words=1800,
        title_seed="A bounded test",
    )
    assert first == second
    assert sum(row["target_words"] for row in first["sections"]) == 1800
    assert [row["rhetorical_role"] for row in first["sections"]] == [
        "introduction", "setup", "results", "conclusion",
    ]
    assert set(models.calls) == {loop.cfg.academic_planner.name}
    assert all(call != loop.cfg.planner.name for call in models.calls)
    assert models.evictions == [(set(), True), (set(), True)]
    assert not thoughts
    assert receipt == {
        "attempted": True, "status": "success",
        "fallback_reason": "", "fallback_used": False,
    }


def test_runtime_accepts_nine_section_targets_present_in_exact_gated_training() -> None:
    payload = json.loads(_valid_reply())
    learned_roles = [
        "introduction", "domain_development", "analysis", "formalism",
        "results", "method_or_setup", "discussion", "limitations", "conclusion",
    ]
    payload["sections"] = [
        {
            "heading": f"Section {index}",
            "id": f"s{index}",
            "role": learned_roles[index - 1],
            "words": 100,
        }
        for index in range(1, 10)
    ]
    payload["paper_counts"]["section_words"] = 900

    outline = ResearchLoop._normalise_academic_structure_outline(
        payload,
        deterministic_blueprint={},
        target_words=1800,
        title_seed="Nine-section training target",
    )

    assert len(outline["sections"]) == 9
    assert [section["name"] for section in outline["sections"]] == [
        f"Section {index}" for index in range(1, 10)
    ]
    assert sum(section["target_words"] for section in outline["sections"]) == 1800


def test_invalid_structure_output_is_audited_then_explicitly_falls_back(
    tmp_path: Path,
) -> None:
    invalid = json.loads(_valid_reply())
    invalid["paper_counts"]["section_words"] = 999
    loop, models, thoughts = _loop(
        tmp_path,
        planner_reply=json.dumps(invalid),
        fallback_reply=json.dumps({
            "title": "Fallback",
            "sections": [
                {"name": "Introduction", "rhetorical_role": "introduction"},
                {"name": "Setup", "rhetorical_role": "setup"},
                {"name": "Results", "rhetorical_role": "results"},
                {"name": "Discussion", "rhetorical_role": "discussion"},
            ],
        }),
    )
    receipt = {}
    specialized = loop._academic_structure_outline(
        "structure only", "paper brief",
        deterministic_blueprint={}, target_words=1800, title_seed="T",
        route_receipt=receipt,
    )
    assert specialized is None
    assert models.calls == [loop.cfg.academic_planner.name]
    assert thoughts and thoughts[-1][0] == "academic-planner-fallback"
    fallback = loop._think_json(
        "ordinary outline", "paper brief", role="planner", required=("sections",),
    )
    assert fallback["title"] == "Fallback"
    assert models.calls[-1] == loop.cfg.planner.name
    assert loop.cfg.academic_planner.name not in models.calls[1:]
    assert receipt["attempted"] is True
    assert receipt["status"] == "invalid_json"
    assert "section budgets" in receipt["fallback_reason"]
    assert receipt["fallback_used"] is True


def test_provider_identity_mismatch_never_calls_structure_or_hides_fallback(tmp_path: Path) -> None:
    loop, models, thoughts = _loop(tmp_path, planner_reply=_valid_reply())
    models.providers[loop.cfg.academic_planner.name]["spiralchat_eligible"] = True
    receipt = {}
    result = loop._academic_structure_outline(
        "structure only", "paper brief",
        deterministic_blueprint={}, target_words=1800, title_seed="T",
        route_receipt=receipt,
    )
    assert result is None
    assert models.calls == []
    assert thoughts[-1][0] == "academic-planner-fallback"
    assert "provider identity" in thoughts[-1][1]
    assert receipt["status"] == "identity_mismatch"
    assert receipt["attempted"] is False


@pytest.mark.parametrize(
    ("setup", "status", "attempted"),
    [
        ("disabled", "disabled", False),
        ("not_ready", "not_ready", False),
        ("server_error", "server_error", True),
    ],
)
def test_route_outcome_receipt_distinguishes_unavailable_states(
    tmp_path: Path, setup: str, status: str, attempted: bool,
) -> None:
    loop, models, thoughts = _loop(tmp_path, planner_reply=_valid_reply())
    if setup == "disabled":
        loop.cfg.academic_planner.enabled = False
    elif setup == "not_ready":
        loop.cfg.academic_planner.ready = False
        loop.cfg.academic_planner.error = "manifest endpoint is not configured"
    else:
        models.replies[loop.cfg.academic_planner.name] = RuntimeError("server offline")
    receipt = {}

    result = loop._academic_structure_outline(
        "structure only", "paper brief", deterministic_blueprint={},
        target_words=1800, title_seed="T", route_receipt=receipt,
    )

    assert result is None
    assert receipt["status"] == status
    assert receipt["attempted"] is attempted
    assert receipt["fallback_used"] is True
    assert receipt["fallback_reason"]
    if setup == "disabled":
        assert not thoughts and not models.calls
    else:
        assert thoughts[-1][2]["status"] == status


def test_ordinary_planning_and_academic_prose_never_use_structure_alias(tmp_path: Path) -> None:
    loop, models, _thoughts = _loop(
        tmp_path, planner_reply=_valid_reply(), fallback_reply="ordinary prose")
    planned, _ = loop._think("ordinary system", "ordinary user", role="planner")
    prose, _ = loop._synthesize_prose(
        "write prose", "section evidence",
        phase="section-draft:introduction", fallback_role="worker",
    )
    assert planned == "ordinary prose"
    assert prose == "ordinary prose"
    assert models.calls == [loop.cfg.planner.name, loop.cfg.worker.name]
    assert loop.cfg.academic_planner.name not in models.calls


def test_single_writer_receipt_scopes_paper_planning_away_from_ordinary_routes() -> None:
    source = inspect.getsource(ResearchLoop.write)
    assert "academic-writer-route.json" in source
    assert "paper-specific writing blueprint" in source
    assert "paper-specific section outline and word budget" in source
    assert "research planning outside write()" in source
    assert '"tools", "builder"' in source
    assert "academic-planner-route.json" not in source
    assert "self._academic_structure_outline(" not in source
