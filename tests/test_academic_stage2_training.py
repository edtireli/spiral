"""Model-free contracts for cumulative academic structure QLoRA."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.academic_finetune.train_qlora import (
    _require_unpartitioned_structure_view,
    _resume_command,
    _select_adapter_initialization,
)
from scripts.academic_finetune.training_support import (
    HarnessError,
    STRUCTURE_CORPUS_MANIFEST_SCHEMA,
    STRUCTURE_CORPUS_SCHEMA,
    STRUCTURE_PROFILE_ID,
    STRUCTURE_PROMPT_CONTRACT,
    adapter_bundle_digest,
    atomic_write_json,
    build_adapter_manifest,
    build_training_run_contract,
    canonical_json,
    load_dataset_manifest,
    load_toml_config,
    prepare_mlx_dataset,
    run_preflight,
    sha256_file,
    validate_corpus_records,
    validate_parent_adapter_initialization,
    validate_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_CONFIG = (
    ROOT / "scripts" / "academic_finetune" /
    "qwen38_27b_q4_structure_32gb.toml"
)
STRATA = ("arxiv:hep-th", "arxiv:hep-ph", "pubmed")


def _response_schema(*required: str) -> dict:
    return {
        "type": "object",
        "properties": {key: {} for key in required},
        "required": list(required),
    }


def _structure_record(index: int, split: str, stratum: str, task: str) -> dict:
    if task == "budget_structure":
        target = {
            "section_budgets": [
                {
                    "id": f"s{index}.1", "paragraphs": 3, "words": 120,
                    "figures": 0, "tables": 1,
                },
                {
                    "id": f"s{index}.2", "paragraphs": 4, "words": 180,
                    "figures": 1, "tables": 0,
                },
            ],
            "section_words": 300,
        }
        required = ("section_words", "section_budgets")
    elif task == "brief_to_blueprint":
        target = {
            "paper_counts": {
                "abstract_words": 80,
                "section_words": 300,
                "section_paragraphs": 7,
                "unsectioned_words": 0,
                "figures": 1,
                "tables": 1,
            },
            "sections": [
                {
                    "heading": "Introduction", "id": f"s{index}.1",
                    "role": "introduction", "words": 120,
                },
                {
                    "heading": "Analysis", "id": f"s{index}.2",
                    "role": "domain_development", "words": 180,
                },
            ],
        }
        required = ("paper_counts", "sections")
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(task)
    return {
        "schema_version": STRUCTURE_CORPUS_SCHEMA,
        "example_id": f"structure-{index}",
        "split": split,
        "task_type": task,
        "source": {
            "provider": "pubmed" if stratum == "pubmed" else "arxiv",
            "stratum": stratum,
        },
        "document": {
            "document_id": f"document-{index}",
            "authors": [f"Structure Author {index}"],
        },
        "input": {
            "instruction": "Recover the observed paper architecture.",
            "context": {"discipline": "theoretical_physics"},
            "constraints": {
                "evidence": "observed_paper_structure_only",
                "response": "json_object",
            },
            "response_schema": _response_schema(*required),
        },
        "target": target,
        "provenance": {
            "exact_training_tokens": 100,
            "exact_prompt_offset": 60,
            "exact_completion_tokens": 40,
        },
    }


def _replay_record(index: int = 3) -> dict:
    return {
        "schema_version": STRUCTURE_CORPUS_SCHEMA,
        "example_id": f"structure-{index}",
        "split": "train",
        "task_type": "prose_replay",
        "source": {"provider": "pubmed", "stratum": "pubmed"},
        "document": {
            "document_id": f"document-{index}",
            "authors": [f"Structure Author {index}"],
        },
        "input": {
            "unit": "sentence",
            "context": "The preceding analysis establishes the bounded regime.",
            "claims": ["the inferred relation remains conditional"],
            "rhetorical_relation": "qualification",
            "certainty": "bounded",
            "citation_count": 0,
            "citation_slots": [],
        },
        "target": (
            "Accordingly, the inferred relation remains conditional on the "
            "stated regime."
        ),
        "provenance": {
            "exact_training_tokens": 100,
            "exact_prompt_offset": 60,
            "exact_completion_tokens": 40,
        },
    }


def _structure_records() -> list[dict]:
    return [
        _structure_record(0, "train", STRATA[0], "budget_structure"),
        _structure_record(1, "validation", STRATA[1], "brief_to_blueprint"),
        _structure_record(2, "test", STRATA[2], "budget_structure"),
        _replay_record(),
    ]


def _write_structure_corpus(tmp_path: Path) -> tuple[Path, list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = _structure_records()
    corpus = tmp_path / "academic_structure.jsonl"
    corpus.write_bytes(b"".join(canonical_json(row) for row in records))
    atomic_write_json(corpus.with_name(f"{corpus.name}.manifest.json"), {
        "schema_version": STRUCTURE_CORPUS_MANIFEST_SCHEMA,
        "corpus_schema_version": STRUCTURE_CORPUS_SCHEMA,
        "prompt_contract": STRUCTURE_PROMPT_CONTRACT,
        "source_strata": sorted(STRATA),
        "trainable": True,
        "corpus_sha256": sha256_file(corpus),
        "gates": {
            "exact_training_token_gate": {
                "method": "mlx_lm.CompletionsDataset.apply_chat_template parity",
                "max_sequence_length": 448,
                "overflow_policy": "reject_candidate_never_truncate_or_partition",
                "derived_rows": 0,
                "tokenizer": {
                    "identity": "fixture-qwen-tokenizer-v1",
                    "loader": "injected-test-tokenizer",
                },
                "candidates_measured": len(records),
                "candidates_rejected": 0,
                "largest_accepted_tokens": 100,
            },
        },
    })
    return corpus, records


def _base_receipt(config: dict) -> dict:
    keys = (
        "model_id", "revision", "model_type", "architecture", "config_sha256",
        "weight_index_sha256", "weight_inventory_sha256", "weight_files",
        "quantization",
    )
    receipt = {key: copy.deepcopy(config["base_model"][key]) for key in keys}
    targets_per_layer: dict[str, list[str]] = {}
    for layer in (60, 61, 62):
        targets_per_layer[str(layer)] = [
            "linear_attn.in_proj_qkv", "linear_attn.out_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]
    targets_per_layer["63"] = [
        "self_attn.q_proj", "self_attn.v_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    receipt["target_inventory"] = {
        "selected_layers": [60, 61, 62, 63],
        "target_path_counts": {
            "self_attn.q_proj": 1,
            "self_attn.v_proj": 1,
            "linear_attn.in_proj_qkv": 3,
            "linear_attn.out_proj": 3,
            "mlp.gate_proj": 4,
            "mlp.up_proj": 4,
            "mlp.down_proj": 4,
        },
        "targets_per_layer": targets_per_layer,
        "total_target_modules": 20,
    }
    return receipt


def _write_lora_safetensors(
    path: Path, target_inventory: dict, *, rank: int, extra_tensor: bool = False,
) -> None:
    names: list[str] = []
    for layer, targets in target_inventory["targets_per_layer"].items():
        for target in targets:
            prefix = f"language_model.model.layers.{layer}.{target}"
            names.extend((f"{prefix}.lora_a", f"{prefix}.lora_b"))
    if extra_tensor:
        names.append("language_model.model.layers.0.mlp.up_proj.lora_a")
    header: dict[str, dict] = {}
    offset = 0
    for name in sorted(names):
        shape = [2, rank] if name.endswith(".lora_a") else [rank, 2]
        size = shape[0] * shape[1] * 4
        header[name] = {
            "dtype": "F32", "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + bytes(offset))


def _write_parent_manifest(
    tmp_path: Path, config: dict, base_receipt: dict, *,
    extra_tensor: bool = False,
) -> Path:
    release = tmp_path / ("parent-extra" if extra_tensor else "parent")
    adapter_dir = release / "adapter"
    adapter_dir.mkdir(parents=True)
    atomic_write_json(adapter_dir / "adapter_config.json", {
        "fine_tune_type": "lora",
        "num_layers": config["training"]["num_layers"],
        "lora_parameters": copy.deepcopy(config["lora"]),
    })
    _write_lora_safetensors(
        adapter_dir / "adapters.safetensors",
        base_receipt["target_inventory"], rank=config["lora"]["rank"],
        extra_tensor=extra_tensor)
    bundle, required = adapter_bundle_digest(adapter_dir)
    manifest = {
        "schema_version": "spiral.academic-adapter.v1",
        "profile_id": "academic-hep-pubmed-v1",
        "prompt_contract": "spiral.academic-plan-prose.v1",
        "base_model": {
            key: copy.deepcopy(base_receipt[key])
            for key in (
                "model_id", "revision", "model_type", "architecture",
                "config_sha256", "weight_index_sha256",
                "weight_inventory_sha256", "weight_files", "quantization",
            )
        },
        "adapter": {
            "path": "adapter", "format": "mlx_lm_lora", "sha256": bundle,
            "required_files": required,
        },
        "training": {
            "trainable_layers": base_receipt["target_inventory"]["selected_layers"],
            "trainable_target_paths": copy.deepcopy(config["lora"]["keys"]),
            "target_path_counts": base_receipt["target_inventory"]["target_path_counts"],
            "total_target_modules": base_receipt["target_inventory"]["total_target_modules"],
        },
    }
    manifest_path = release / "academic-adapter.manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def test_structure_config_is_exact_and_cumulative() -> None:
    config = load_toml_config(STRUCTURE_CONFIG)

    assert config["profile"] == {
        "id": STRUCTURE_PROFILE_ID,
        "prompt_contract": STRUCTURE_PROMPT_CONTRACT,
    }
    assert config["training"]["iterations"] == 1200
    assert config["training"]["learning_rate"] == pytest.approx(2e-6)
    assert config["training"]["max_seq_length"] == 448
    assert config["training"]["num_layers"] == 4
    assert config["training"]["seed"] == 25082026
    assert config["lora"] == {
        "rank": 16,
        "scale": 32.0,
        "dropout": 0.0,
        "keys": [
            "self_attn.q_proj", "self_attn.v_proj",
            "linear_attn.in_proj_qkv", "linear_attn.out_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ],
    }
    altered = copy.deepcopy(config)
    altered["training"]["learning_rate"] = 5e-6
    with pytest.raises(HarnessError, match="structure profile training.learning_rate"):
        validate_training_config(altered)


def test_structure_dataset_has_honest_dual_target_contract(tmp_path: Path) -> None:
    config = load_toml_config(STRUCTURE_CONFIG)
    corpus, source_rows = _write_structure_corpus(tmp_path / "corpus")
    data = tmp_path / "data"

    manifest = prepare_mlx_dataset(corpus, data, config=config)
    assert load_dataset_manifest(data, config=config) == manifest
    assert manifest["target_contract"] == {
        "structure_tasks": [
            "brief_to_blueprint", "budget_structure", "order_structure",
            "recognize_role", "repair_structure", "restore_section",
        ],
        "structure_target": "canonical_json_object",
        "prose_replay_task": "prose_replay",
        "prose_replay_target": "nonempty_academic_prose",
    }
    assert manifest["target_context_leakage_gate"]["scope"] == "prose_replay_only"
    assert manifest["source_exact_training_token_gate"]["tokenizer"]["identity"] == (
        "fixture-qwen-tokenizer-v1")
    rows = [
        json.loads(line)
        for split in ("train", "valid", "test")
        for line in (data / f"{split}.jsonl").read_text().splitlines()
    ]
    budget = next(row for row in rows if row["example_id"] == "structure-0")
    replay = next(row for row in rows if row["task_type"] == "prose_replay")
    assert budget["completion"] == json.dumps(
        source_rows[0]["target"], sort_keys=True, separators=(",", ":"))
    assert budget["completion"] not in budget["prompt"]
    assert budget["prompt"].endswith("\n")
    assert replay["completion"] == source_rows[-1]["target"]
    assert replay["prompt"].startswith("Reconstruct one missing unit")


def test_structure_dataset_rejects_missing_or_inconsistent_exact_token_receipts(
    tmp_path: Path,
) -> None:
    config = load_toml_config(STRUCTURE_CONFIG)
    corpus, records = _write_structure_corpus(tmp_path / "missing-gate")
    manifest_path = corpus.with_name(f"{corpus.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("gates")
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(HarnessError, match="no exact training-token gate"):
        prepare_mlx_dataset(corpus, tmp_path / "missing-gate-data", config=config)

    records[0]["provenance"]["exact_training_tokens"] = 449
    corpus = tmp_path / "overflow" / "academic_structure.jsonl"
    corpus.parent.mkdir()
    corpus.write_bytes(b"".join(canonical_json(row) for row in records))
    manifest["corpus_sha256"] = sha256_file(corpus)
    manifest["gates"] = {
        "exact_training_token_gate": {
            "method": "mlx_lm.CompletionsDataset.apply_chat_template parity",
            "max_sequence_length": 448,
            "overflow_policy": "reject_candidate_never_truncate_or_partition",
            "derived_rows": 0,
            "tokenizer": {"identity": "fixture-qwen-tokenizer-v1"},
            "candidates_measured": len(records),
            "candidates_rejected": 0,
            "largest_accepted_tokens": 448,
        },
    }
    atomic_write_json(corpus.with_name(f"{corpus.name}.manifest.json"), manifest)
    with pytest.raises(HarnessError, match="record 1.*exact training-token receipt"):
        prepare_mlx_dataset(corpus, tmp_path / "overflow-data", config=config)


def test_structure_preflight_loads_dataset_under_structure_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_toml_config(STRUCTURE_CONFIG)
    corpus, _ = _write_structure_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    prepare_mlx_dataset(corpus, data, config=config)
    base = _base_receipt(config)
    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.validate_local_base_model",
        lambda _path, _config: base,
    )
    report = run_preflight(
        config,
        tmp_path / "model",
        data,
        tmp_path / "output",
        resource_facts={
            "platform": "Darwin",
            "machine": "arm64",
            "memory_total_bytes": 64 * 1024**3,
            "memory_available_bytes": 48 * 1024**3,
            "disk_free_bytes": 128 * 1024**3,
        },
        package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"},
    )
    assert report.ok, report.errors
    assert report.facts["dataset"]["prompt_contract"] == STRUCTURE_PROMPT_CONTRACT


def test_structure_budget_and_blueprint_invariants_fail_closed() -> None:
    config = load_toml_config(STRUCTURE_CONFIG)

    wrong_sum = _structure_records()
    wrong_sum[0]["target"]["section_words"] = 301
    with pytest.raises(HarnessError, match="words sum to 300"):
        validate_corpus_records(wrong_sum, config=config)

    duplicate = _structure_records()
    duplicate[0]["target"]["section_budgets"][1]["id"] = (
        duplicate[0]["target"]["section_budgets"][0]["id"])
    with pytest.raises(HarnessError, match="duplicate section id"):
        validate_corpus_records(duplicate, config=config)

    negative = _structure_records()
    negative[1]["target"]["paper_counts"]["figures"] = -1
    with pytest.raises(HarnessError, match="figures: must be a non-negative integer"):
        validate_corpus_records(negative, config=config)

    unsupported_schema = _structure_records()
    unsupported_schema[0]["input"]["response_schema"]["oneOf"] = []
    with pytest.raises(HarnessError, match="unsupported keyword.*oneOf"):
        validate_corpus_records(unsupported_schema, config=config)


def test_parent_adapter_is_exactly_authenticated_and_inventory_bound(
    tmp_path: Path,
) -> None:
    config = load_toml_config(STRUCTURE_CONFIG)
    base = _base_receipt(config)
    manifest_path = _write_parent_manifest(tmp_path, config, base)

    parent = validate_parent_adapter_initialization(manifest_path, config, base)
    assert parent.identity["generation"] == 1
    assert parent.identity["manifest_sha256"] == sha256_file(manifest_path)
    assert parent.identity["base_weight_inventory_sha256"] == (
        base["weight_inventory_sha256"])
    assert parent.adapter_config_path.name == "adapter_config.json"
    assert parent.weights_path.name == "adapters.safetensors"

    wrong_base = copy.deepcopy(base)
    wrong_base["weight_inventory_sha256"] = "f" * 64
    with pytest.raises(HarnessError, match="weight_inventory_sha256"):
        validate_parent_adapter_initialization(manifest_path, config, wrong_base)

    extra_manifest = _write_parent_manifest(
        tmp_path, config, base, extra_tensor=True)
    with pytest.raises(HarnessError, match="tensor inventory"):
        validate_parent_adapter_initialization(extra_manifest, config, base)


def test_parent_lineage_binds_run_and_publishes_planner_only_identity(
    tmp_path: Path,
) -> None:
    config = load_toml_config(STRUCTURE_CONFIG)
    base = _base_receipt(config)
    parent_manifest = _write_parent_manifest(tmp_path, config, base)
    parent = validate_parent_adapter_initialization(parent_manifest, config, base)
    corpus, _ = _write_structure_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    dataset = prepare_mlx_dataset(corpus, data, config=config)

    unbound = build_training_run_contract(config, base, dataset)
    bound = build_training_run_contract(
        config, base, dataset, parent_adapter=parent)
    assert bound["run_identity"] != unbound["run_identity"]
    assert bound["identity_contract"]["parent_adapter"] == parent.identity

    adapter_dir = parent.adapter_dir
    bundle, required = adapter_bundle_digest(adapter_dir)
    child_manifest_path = tmp_path / "child" / "academic-adapter.manifest.json"
    child_manifest_path.parent.mkdir()
    child = build_adapter_manifest(
        config=config, base_receipt=base, dataset_manifest=dataset,
        dataset_manifest_path=data / "dataset_manifest.json",
        adapter_manifest_path=child_manifest_path, adapter_dir=adapter_dir,
        bundle_digest=bundle, required_files=required,
        package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"},
        parent_adapter=parent)
    assert child["lineage"]["parent"] == parent.identity
    assert child["runtime"] == {
        "provider": "mlx_lm",
        "model": "qwen3.8-27b-academic-structure",
        "base_url": "http://127.0.0.1:8080/v1",
        "transport_adapter": "openai-compatible",
        "default_adapter_strength": 1.0,
        "scope": "spiral_paper_planner_only",
        "spiralchat_eligible": False,
    }


def test_stage_two_resume_retains_parent_but_prefers_new_checkpoint(
    tmp_path: Path,
) -> None:
    initializer = tmp_path / "parent" / "academic-adapter.manifest.json"
    arguments = SimpleNamespace(
        python="/usr/bin/python3",
        config=STRUCTURE_CONFIG,
        data_dir=tmp_path / "data",
        model=tmp_path / "model",
        output=tmp_path / "run",
        model_view_cache=tmp_path / "view",
        lease_path=tmp_path / "lease",
        ollama_url="http://127.0.0.1:11434",
        safe_schedule=True,
        initialize_from_manifest=initializer,
    )
    command = _resume_command(arguments)
    assert "--initialize-from-manifest" in command
    assert str(initializer.resolve()) in command
    assert "--resume --execute" in command

    parent = SimpleNamespace(
        adapter_config_path=tmp_path / "parent-config.json",
        weights_path=tmp_path / "parent.safetensors",
    )
    checkpoint = SimpleNamespace(
        config_path=tmp_path / "checkpoint-config.json",
        path=tmp_path / "checkpoint.safetensors",
    )
    assert _select_adapter_initialization(None, parent) == (
        parent.adapter_config_path, parent.weights_path)
    assert _select_adapter_initialization(checkpoint, parent) == (
        checkpoint.config_path, checkpoint.path)


def test_structure_safe_view_rejects_partition_or_unknown_receipt() -> None:
    _require_unpartitioned_structure_view({
        "output": {"partitioned_source_rows": 0, "derived_rows": 0},
    })
    with pytest.raises(HarnessError, match="partitioning is forbidden"):
        _require_unpartitioned_structure_view({
            "output": {"partitioned_source_rows": 1, "derived_rows": 2},
        })
    with pytest.raises(HarnessError, match="partitioning is forbidden"):
        _require_unpartitioned_structure_view({})


def test_structure_safe_view_requires_compiler_tokenizer_identity_parity() -> None:
    receipt = {
        "output": {"partitioned_source_rows": 0, "derived_rows": 0},
        "tokenizer": {"identity": "runtime-tokenizer"},
        "gate": {"max_sequence_length": 448},
    }
    dataset = {
        "source_exact_training_token_gate": {
            "max_sequence_length": 448,
            "tokenizer": {"identity": "runtime-tokenizer"},
        },
    }
    _require_unpartitioned_structure_view(receipt, dataset)

    mismatched = copy.deepcopy(dataset)
    mismatched["source_exact_training_token_gate"]["tokenizer"]["identity"] = (
        "different-tokenizer")
    with pytest.raises(HarnessError, match="exact same tokenizer identity"):
        _require_unpartitioned_structure_view(receipt, mismatched)
