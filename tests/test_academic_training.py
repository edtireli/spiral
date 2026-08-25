"""Offline/model-free contracts for the academic MLX training harness."""

from __future__ import annotations

import copy
import base64
import hashlib
import http.client
import json
import os
import subprocess
import struct
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.academic_finetune import serve_vlm_adapter as vlm_server

from scripts.academic_finetune.evaluate import (
    evaluate_predictions,
    finalize_post_training_evaluation,
    mlx_nll_command,
    multiset_f1,
    run_mlx_predictions,
    run_mlx_nll,
    score_candidate,
    argument_score,
    claim_coverage,
    write_prediction_template,
)
from scripts.academic_finetune.train_qlora import main as train_main
from scripts.academic_finetune.serve_adapter import (
    ADAPTER_STRENGTH_STEP,
    DEFAULT_ADAPTER_STRENGTH,
    IDENTITY_SCHEMA,
    MAX_ADAPTER_STRENGTH,
    MIN_ADAPTER_STRENGTH,
    apply_adapter_strength,
    canonical_adapter_strength,
    identity_health_response,
    openai_completion_response,
    derive_model_view_path,
    validate_bind_address,
    validate_chat_request,
    validate_runtime_assets,
    validate_runtime_storage_sentinel,
)
from scripts.academic_finetune.serve_vlm_adapter import (
    AcademicVlmHTTPServer,
    EXPECTED_RUNTIME_MODEL as EXPECTED_VLM_RUNTIME_MODEL,
    IDENTITY_SCHEMA as VLM_IDENTITY_SCHEMA,
    LEASE_HANDOFF_CONTRACT,
    SERVER_LIFECYCLE_IDENTITY as VLM_SERVER_LIFECYCLE_IDENTITY,
    compute_admission,
    load_lease_authority_token,
    lora_parameters_for_strength,
    ollama_delta,
    ollama_result,
    trusted_lease_handoff,
    validate_bind_address as validate_vlm_bind_address,
    validate_vlm_chat_request,
    validate_vlm_runtime_assets,
    validate_vlm_storage_sentinel,
    vlm_processor_messages,
)
from scripts.academic_finetune.training_support import (
    ADAPTER_SCHEMA,
    CORPUS_SCHEMA,
    EXPECTED_ARCHITECTURE,
    EXPECTED_MODEL_TYPE,
    HarnessError,
    CheckpointLedger,
    TrainingComputeLease,
    TrainingMetricsJournal,
    adapter_bundle_digest,
    atomic_write_json,
    build_adapter_manifest,
    build_training_run_contract,
    create_training_only_dataset_view,
    create_text_training_view,
    detect_local_revision,
    load_toml_config,
    prepare_mlx_dataset,
    parse_training_metric,
    publish_adapter_bundle,
    run_preflight,
    run_training_process,
    safetensors_header,
    selected_target_inventory,
    sha256_file,
    system_resources,
    validate_training_config,
    validate_local_base_model,
    yaml_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "academic_finetune" / "qwen38_27b_q4.toml"
REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
STRATA = ("arxiv:hep-th", "arxiv:hep-ph", "pubmed")


def _record(index: int, split: str, stratum: str) -> dict:
    citation = index == 2
    target = (
        "However, the measured spectrum may constrain the effective interaction [12]."
        if citation else
        f"Consequently, observable {index} supports the stated physical mechanism."
    )
    return {
        "schema_version": CORPUS_SCHEMA,
        "example_id": f"example-{index}",
        "split": split,
        "task_type": "sentence",
        "source": {
            "provider": "pubmed" if stratum == "pubmed" else "arxiv",
            "source_id": f"source-{index}",
            "stratum": stratum,
            "landing_url": "https://example.invalid/paper",
            "artifact_url": "https://example.invalid/paper.pdf",
            "query": "test",
        },
        "document": {
            "document_id": f"document-{index}",
            "title": f"Paper {index}",
            "authors": [f"Author {index}"],
            "published": "2021-01-01",
            "latest_version": "v1",
            "content_sha256": f"{index:064x}",
        },
        "input": {
            "context": f"Prior context {index} with a distinct premise.",
            "claims": [f"observable {index} supports the physical mechanism"],
            "rhetorical_relation": "limitation" if citation else "causal",
            "certainty": "tentative" if citation else "moderate",
            "citation_count": 1 if citation else 0,
            "citation_slots": ["[12]"] if citation else [],
            "construction_method": "heuristic_keyword_proposition_v1",
        },
        "target": target,
        "provenance": {
            "metadata_endpoint": "fixture",
            "content_endpoint": "fixture",
            "extraction": "fixture",
            "locator": f"sentence:{index}",
            "text_unit_sha256": f"{index + 10:064x}",
            "raw_record_sha256": f"{index + 20:064x}",
            "cutoff": "2021-12-31",
        },
    }


def _write_corpus(tmp_path: Path) -> tuple[Path, list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = [
        _record(0, "train", STRATA[0]),
        _record(1, "validation", STRATA[1]),
        _record(2, "test", STRATA[2]),
    ]
    corpus = tmp_path / "academic_corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "spiral.academic-corpus-manifest.v1",
        "source_strata": sorted(STRATA),
        "counts": {
            "total": len(records),
            "by_split": {"train": 1, "validation": 1, "test": 1},
        },
        "cutoff": "2021-12-31",
        "trainable": True,
        "corpus_sha256": sha256_file(corpus),
        "output_filename": corpus.name,
        "split_policy": (
            "connected document-author components; sha256 90/5/5 with "
            "deterministic nonempty pilot repair"),
    }
    atomic_write_json(corpus.with_name(f"{corpus.name}.manifest.json"), manifest)
    return corpus, records


def _config(*, tiny_model: bool = False) -> dict:
    config = load_toml_config(CONFIG_PATH)
    if tiny_model:
        config = copy.deepcopy(config)
        config["resources"].update({
            "minimum_model_bytes": 1,
            "minimum_total_memory_gib": 1,
            "minimum_available_memory_gib": 1,
            "minimum_free_disk_gib": 1,
        })
    return config


def _fake_qwen38_model(tmp_path: Path, config: dict) -> Path:
    model = tmp_path / "snapshots" / REVISION
    model.mkdir(parents=True)
    layer_types = ["linear_attention" if (index + 1) % 4 else "full_attention" for index in range(64)]
    model_config = {
        "architectures": [EXPECTED_ARCHITECTURE],
        "model_type": EXPECTED_MODEL_TYPE,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "text_config": {"num_hidden_layers": 64, "layer_types": layer_types},
        "vision_config": {"model_type": "qwen3_5_vision", "depth": 2},
    }
    (model / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    weight_map = {}
    for layer in range(56, 64):
        prefix = "linear_attn" if layer_types[layer] == "linear_attention" else "self_attn"
        keys = (
            ("in_proj_qkv", "out_proj") if prefix == "linear_attn"
            else ("q_proj", "v_proj")
        )
        for key in keys:
            weight_map[f"language_model.model.layers.{layer}.{prefix}.{key}.weight"] = "model.safetensors"
        for key in ("gate_proj", "up_proj", "down_proj"):
            weight_map[f"language_model.model.layers.{layer}.mlp.{key}.weight"] = "model.safetensors"
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake-indexed-shard")
    (model / "chat_template.jinja").write_text(
        "{% if tools %}<tool_call><function=example></function></tool_call>{% endif %}",
        encoding="utf-8")
    for frontend in (
        "generation_config.json", "preprocessor_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
        "vocab.json",
    ):
        (model / frontend).write_text("{}\n", encoding="utf-8")
    shard = model / "model.safetensors"
    shard_row = {
        "path": shard.name,
        "size_bytes": shard.stat().st_size,
        "sha256": sha256_file(shard),
    }
    inventory_line = (
        f"{shard_row['path']}\0{shard_row['size_bytes']}\0{shard_row['sha256']}\n")
    config["base_model"].update({
        "config_sha256": sha256_file(model / "config.json"),
        "weight_index_sha256": sha256_file(model / "model.safetensors.index.json"),
        "weight_inventory_sha256": hashlib.sha256(
            inventory_line.encode("utf-8")).hexdigest(),
        "weight_bytes": shard_row["size_bytes"],
        "weight_files": [shard_row],
    })
    return model


def _tiny_safetensors(path: Path, payload: bytes = b"1234") -> None:
    header = json.dumps({
        "weight": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }, separators=(",", ":")).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_local_dir_huggingface_metadata_attests_revision_without_snapshot_name(tmp_path):
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for name in ("config.json.metadata", "model.safetensors.index.json.metadata"):
        (metadata / name).write_text(f"{REVISION}\netag\ntimestamp\n", encoding="utf-8")

    assert detect_local_revision(tmp_path) == REVISION

    (metadata / "model.safetensors.index.json.metadata").write_text(
        f"{'b' * 40}\netag\ntimestamp\n", encoding="utf-8")
    assert detect_local_revision(tmp_path) is None


def _runtime_fixture(tmp_path: Path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path / "model", config)
    corpus, _ = _write_corpus(tmp_path / "corpus")
    release = tmp_path / "release"
    release.mkdir()
    data = release / "data"
    dataset = prepare_mlx_dataset(corpus, data)
    base = validate_local_base_model(model, config)
    view = create_text_training_view(model, tmp_path / "view-cache", base)
    work = tmp_path / "work-adapter"
    work.mkdir()
    (work / "adapter_config.json").write_text(json.dumps({
        "lora_parameters": {"rank": 16, "scale": 32.0, "dropout": 0.0},
    }) + "\n")
    _tiny_safetensors(work / "adapters.safetensors")
    adapter, digest, required = publish_adapter_bundle(work, release)
    manifest_path = release / "academic-adapter.manifest.json"
    manifest = build_adapter_manifest(
        config=config, base_receipt=base, dataset_manifest=dataset,
        dataset_manifest_path=data / "dataset_manifest.json",
        adapter_manifest_path=manifest_path, adapter_dir=adapter,
        bundle_digest=digest, required_files=required,
        package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"})
    atomic_write_json(manifest_path, manifest)
    return manifest_path, view, adapter


def test_default_config_pins_exact_qwen38_hybrid_qlora_and_base_inventory():
    config = load_toml_config(CONFIG_PATH)

    assert config["base_model"] == {
        "model_id": "mlx-community/Qwen3.8-27B-4bit",
        "revision": REVISION,
        "model_type": EXPECTED_MODEL_TYPE,
        "architecture": EXPECTED_ARCHITECTURE,
        "config_sha256": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
        "weight_index_sha256": "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
        "weight_inventory_sha256": "8126a3fd4aef3346254965791eedc5a5468bf7fcf46bdd95ef29dd13266ed589",
        "weight_bytes": 16_054_541_349,
        "weight_files": [
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
        ],
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    assert config["training"]["batch_size"] == 1
    assert config["training"]["grad_accumulation_steps"] > 1
    assert config["training"]["grad_checkpoint"] is True
    assert config["training"]["mask_prompt"] is True
    assert {"self_attn.q_proj", "linear_attn.in_proj_qkv"} <= set(config["lora"]["keys"])


def test_config_rejects_missing_linear_attention_targets():
    config = copy.deepcopy(load_toml_config(CONFIG_PATH))
    config["lora"]["keys"] = ["self_attn.q_proj", "self_attn.v_proj"]

    with pytest.raises(HarnessError, match="linear-attention"):
        validate_training_config(config)


def test_prepare_dataset_is_completion_only_and_binds_exact_citation_slots(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    data = tmp_path / "data"

    manifest = prepare_mlx_dataset(corpus, data)

    assert manifest["completion_only_loss"] is True
    assert manifest["source_corpus_manifest_sha256"] == sha256_file(
        corpus.with_name(f"{corpus.name}.manifest.json"))
    test_row = json.loads((data / "test.jsonl").read_text())
    assert test_row["completion"] == records[-1]["target"]
    assert records[-1]["target"] not in test_row["prompt"]
    assert "Citation markers required (1): [12]" in test_row["prompt"]
    assert set(manifest["splits"]) == {"train", "valid", "test"}


def test_prepare_dataset_rejects_document_leakage_across_splits(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    records[1]["document"]["document_id"] = records[0]["document"]["document_id"]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest_path = corpus.with_name(f"{corpus.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_sha256"] = sha256_file(corpus)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(HarnessError, match="leaks across"):
        prepare_mlx_dataset(corpus, tmp_path / "data")


def test_prepare_dataset_rejects_target_copied_into_context_even_with_punctuation_changes(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    records[0]["input"]["context"] = (
        "Earlier context. " + records[0]["target"].upper().replace(".", "!")
    )
    corpus.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest_path = corpus.with_name(f"{corpus.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_sha256"] = sha256_file(corpus)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(HarnessError, match="target leaks into input.context"):
        prepare_mlx_dataset(corpus, tmp_path / "data")


def test_training_boundary_rejects_same_author_across_different_documents_and_splits(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    records[1]["document"]["authors"] = ["AUTHOR 0"]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest_path = corpus.with_name(f"{corpus.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["corpus_sha256"] = sha256_file(corpus)
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(HarnessError, match="author .* leaks across"):
        prepare_mlx_dataset(corpus, tmp_path / "data")


def test_selected_targets_cover_every_full_and_linear_layer(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path, config)
    model_config = json.loads((model / "config.json").read_text())
    weight_keys = set(json.loads((model / "model.safetensors.index.json").read_text())["weight_map"])

    inventory = selected_target_inventory(
        model_config, config["lora"]["keys"], config["training"]["num_layers"],
        weight_keys=weight_keys)

    assert inventory["selected_layers"] == list(range(56, 64))
    assert inventory["layer_type_counts"] == {"full_attention": 2, "linear_attention": 6}
    assert inventory["target_path_counts"]["self_attn.q_proj"] == 2
    assert inventory["target_path_counts"]["linear_attn.in_proj_qkv"] == 6
    assert len(inventory["targets_per_layer"]) == 8


def test_preflight_is_no_load_and_reports_target_counts_and_resources(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path, config)
    corpus, _ = _write_corpus(tmp_path)
    data = tmp_path / "data"
    prepare_mlx_dataset(corpus, data)
    resources = {
        "platform": "Darwin", "machine": "arm64",
        "memory_total_bytes": 64 * 1024 ** 3,
        "memory_available_bytes": 48 * 1024 ** 3,
        "disk_free_bytes": 100 * 1024 ** 3,
    }

    report = run_preflight(
        config, model, data, tmp_path / "output",
        resource_facts=resources, package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"})

    assert report.ok, report.errors
    assert report.facts["base_model"]["revision"] == REVISION
    assert report.facts["base_model"]["target_inventory"]["total_target_modules"] > 8
    assert len(report.facts["base_model"]["weight_files"]) == 1
    assert len(report.facts["base_model"]["weight_inventory_sha256"]) == 64
    assert report.facts["packages"] == {"mlx": "0.31.2", "mlx_lm": "0.31.3"}


def test_preflight_fails_closed_for_low_available_ram_and_missing_mlx(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path, config)
    corpus, _ = _write_corpus(tmp_path)
    data = tmp_path / "data"
    prepare_mlx_dataset(corpus, data)

    report = run_preflight(
        config, model, data, tmp_path / "output",
        resource_facts={
            "platform": "Darwin", "machine": "arm64",
            "memory_total_bytes": 64 * 1024 ** 3,
            "memory_available_bytes": 0,
            "disk_free_bytes": 100 * 1024 ** 3,
        },
        package_versions={"mlx": None, "mlx_lm": None},
    )

    assert not report.ok
    assert any("available unified memory" in error for error in report.errors)
    assert any("mlx is not installed" in error for error in report.errors)


def test_darwin_resources_prefer_memory_pressure_available_percentage(monkeypatch, tmp_path):
    total = 32 * 1024 ** 3
    calls: list[tuple[str, ...]] = []

    def fake_check_output(command, *, text):
        assert text is True
        calls.append(tuple(command))
        if command == ["sysctl", "-n", "hw.memsize"]:
            return f"{total}\n"
        if command == ["/usr/bin/memory_pressure", "-Q"]:
            return (
                "The system has 34359738368 (2097152 pages with a page size of 16384).\n"
                "System-wide memory free percentage: 82%\n"
            )
        raise AssertionError(f"unexpected resource probe: {command}")

    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.subprocess.check_output",
        fake_check_output,
    )

    resources = system_resources(tmp_path)

    assert resources["memory_available_bytes"] == total * 82 // 100
    assert resources["memory_available_source"] == "memory_pressure"
    assert resources["memory_available_percentage"] == 82.0
    assert ("vm_stat",) not in calls


def test_darwin_resources_fall_back_to_vm_stat_when_memory_pressure_unavailable(
        monkeypatch, tmp_path):
    total = 16 * 1024 ** 3
    page_size = 4096
    available_pages = 1024 + 2048 + 512 + 256

    def fake_check_output(command, *, text):
        assert text is True
        if command == ["sysctl", "-n", "hw.memsize"]:
            return f"{total}\n"
        if command == ["/usr/bin/memory_pressure", "-Q"]:
            raise subprocess.CalledProcessError(1, command)
        if command == ["vm_stat"]:
            return (
                f"Mach Virtual Memory Statistics: (page size of {page_size} bytes)\n"
                "Pages free: 1024.\n"
                "Pages inactive: 2048.\n"
                "Pages speculative: 512.\n"
                "Pages purgeable: 256.\n"
            )
        raise AssertionError(f"unexpected resource probe: {command}")

    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.subprocess.check_output",
        fake_check_output,
    )

    resources = system_resources(tmp_path)

    expected = available_pages * page_size
    assert resources["memory_available_bytes"] == expected
    assert resources["memory_available_source"] == "vm_stat"
    assert resources["memory_available_percentage"] == expected * 100.0 / total


def test_darwin_resources_fail_closed_on_malformed_memory_pressure(
        monkeypatch, tmp_path):
    total = 32 * 1024 ** 3
    calls: list[tuple[str, ...]] = []

    def fake_check_output(command, *, text):
        assert text is True
        calls.append(tuple(command))
        if command == ["sysctl", "-n", "hw.memsize"]:
            return f"{total}\n"
        if command == ["/usr/bin/memory_pressure", "-Q"]:
            return "System-wide free memory: eighty-two percent\n"
        if command == ["vm_stat"]:
            raise AssertionError("malformed primary output must not use a weaker fallback")
        raise AssertionError(f"unexpected resource probe: {command}")

    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "scripts.academic_finetune.training_support.subprocess.check_output",
        fake_check_output,
    )

    resources = system_resources(tmp_path)

    assert resources["memory_available_bytes"] == 0
    assert resources["memory_available_source"] == "unavailable"
    assert resources["memory_available_percentage"] is None
    assert "malformed" in resources["memory_available_probe_error"]
    assert ("vm_stat",) not in calls


def test_text_training_view_exposes_wrapper_layers_without_loading_mlx(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path, config)
    corpus, _ = _write_corpus(tmp_path)
    data = tmp_path / "data"
    prepare_mlx_dataset(corpus, data)
    report = run_preflight(
        config, model, data, tmp_path / "output",
        resource_facts={
            "platform": "Darwin", "machine": "arm64",
            "memory_total_bytes": 64 * 1024 ** 3,
            "memory_available_bytes": 48 * 1024 ** 3,
            "disk_free_bytes": 100 * 1024 ** 3,
        }, package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"})

    view = create_text_training_view(model, tmp_path / "internal-apfs-cache", report.facts["base_model"])

    assert (view / "model.safetensors").is_symlink()
    assert "return self.language_model.layers" in (view / "spiral_qwen35_text_tuner.py").read_text()
    assert json.loads((view / "config.json").read_text())["model_file"] == "spiral_qwen35_text_tuner.py"


def test_weight_inventory_mutation_is_rejected_under_same_revision(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path, config)
    first = validate_local_base_model(model, config)
    (model / "model.safetensors").write_bytes(b"different-shard-bytes")

    assert first["revision"] == REVISION
    with pytest.raises(HarnessError, match="pinned Qwen3.8 base"):
        validate_local_base_model(model, config)


def test_yaml_enforces_completion_mask_batch_accum_checkpoint_and_hybrid_keys(tmp_path):
    config = load_toml_config(CONFIG_PATH)
    yaml = yaml_training_config(
        config, tmp_path / "view", tmp_path / "data", tmp_path / "work")

    assert "mask_prompt: true" in yaml
    assert "batch_size: 1" in yaml
    assert "grad_accumulation_steps: 8" in yaml
    assert "grad_checkpoint: true" in yaml
    assert '    - "self_attn.q_proj"' in yaml
    assert '    - "linear_attn.in_proj_qkv"' in yaml
    assert "trust_remote_code: true" in yaml


def test_run_identity_binds_semantics_but_allows_safe_operational_schedule(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path / "model", config)
    base = validate_local_base_model(model, config)
    corpus, _ = _write_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    dataset = prepare_mlx_dataset(corpus, data)

    original = build_training_run_contract(config, base, dataset)
    safe_schedule = copy.deepcopy(config)
    safe_schedule["training"].update({
        "val_batches": 1,
        "steps_per_report": 16,
        "steps_per_eval": 1201,
        "save_every": 40,
        "clear_cache_threshold": "512MB",
    })
    rescheduled = build_training_run_contract(safe_schedule, base, dataset)
    changed_model = copy.deepcopy(config)
    changed_model["training"]["num_layers"] = 4
    changed = build_training_run_contract(changed_model, base, dataset)

    assert rescheduled["run_identity"] == original["run_identity"]
    assert changed["run_identity"] != original["run_identity"]
    assert original["resume_semantics"] == {
        "state": "adapter_weights_only",
        "optimizer_moments_restored": False,
        "rng_state_restored": False,
        "bit_exact": False,
        "description": (
            "MLX-LM reloads adapter weights; optimizer moments and RNG stream "
            "restart for each attempt, so resumed optimization is approximate."
        ),
    }


def test_run_identity_binds_token_bounded_training_view(tmp_path):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path / "model", config)
    base = validate_local_base_model(model, config)
    corpus, _ = _write_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    dataset = prepare_mlx_dataset(corpus, data)
    digest = "a" * 64
    receipt = {
        "schema_version": "spiral.academic-bounded-training-data.v1",
        "view_identity": digest,
        "identity_contract": {"output_train_sha256": digest},
        "output": {
            "train_sha256": digest,
            "train_count": 12,
            "partitioned_source_rows": 2,
            "derived_rows": 5,
        },
        "gate": {"max_sequence_length": 512, "maximum_total_tokens": 509},
        "preservation": {
            "mapping_sha256": digest,
            "all_source_completions_preserved": True,
        },
    }

    direct = build_training_run_contract(config, base, dataset)
    bounded = build_training_run_contract(
        config, base, dataset, training_data_receipt=receipt)

    assert bounded["run_identity"] != direct["run_identity"]
    attestation = bounded["identity_contract"]["bounded_training_data"]
    assert attestation["maximum_total_tokens"] == 509
    assert attestation["all_source_completions_preserved"] is True

    receipt["gate"]["maximum_total_tokens"] = 513
    with pytest.raises(HarnessError, match="token gate"):
        build_training_run_contract(
            config, base, dataset, training_data_receipt=receipt)


def test_training_only_dataset_view_is_attested_and_hides_held_out_splits(tmp_path):
    corpus, _ = _write_corpus(tmp_path / "corpus")
    data = tmp_path / "data"
    dataset = prepare_mlx_dataset(corpus, data)

    view, receipt = create_training_only_dataset_view(data, tmp_path / "run", dataset)

    assert (view / "train.jsonl").is_symlink()
    assert sha256_file(view / "train.jsonl") == dataset["splits"]["train"]["sha256"]
    assert not (view / "valid.jsonl").exists()
    assert not (view / "test.jsonl").exists()
    assert receipt["validation_policy"] == "separate_process_after_training"
    assert json.loads((view / "view.json").read_text()) == receipt


def test_checkpoint_ledger_rejects_partial_files_and_resumes_hashed_snapshot(tmp_path):
    ledger = CheckpointLedger(tmp_path / "checkpoints")
    source = tmp_path / "adapters.safetensors"
    source.write_bytes(b"partial")
    with pytest.raises(HarnessError, match="safetensors"):
        ledger.capture(source, 80)
    _tiny_safetensors(source)
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text('{"rank":16}\n')

    captured = ledger.capture(source, 80, adapter_config=adapter_config)
    latest = ledger.latest()

    assert latest == captured
    assert latest.path.parent.name == "step-0000080"
    assert latest.sha256 == sha256_file(latest.path)
    assert latest.config_sha256 == sha256_file(latest.config_path)
    assert len(latest.bundle_sha256) == 64
    assert len(latest.receipt_sha256) == 64
    assert json.loads((ledger.root / "latest.json").read_text())["step"] == 80


def test_checkpoint_recovers_pointerless_commit_and_rejects_corruption_and_identity(tmp_path):
    source = tmp_path / "adapters.safetensors"
    config = tmp_path / "adapter_config.json"
    _tiny_safetensors(source)
    config.write_text('{"rank":16}\n')
    identity = "a" * 64
    ledger = CheckpointLedger(tmp_path / "checkpoints", run_identity=identity)
    captured = ledger.capture(source, 40, adapter_config=config)

    (ledger.root / "latest.json").unlink()
    assert ledger.latest() == captured
    assert (ledger.root / "latest.json").is_file()

    with pytest.raises(HarnessError, match="run identity mismatch"):
        CheckpointLedger(ledger.root, run_identity="b" * 64).latest()

    captured.config_path.write_text('{"rank":8}\n')
    with pytest.raises(HarnessError, match="adapter_config.json integrity"):
        ledger.latest()


def test_checkpoint_rejects_partial_committed_directory(tmp_path):
    ledger = CheckpointLedger(tmp_path / "checkpoints", run_identity="a" * 64)
    (ledger.root / "step-0000040").mkdir()

    with pytest.raises(HarnessError, match="checkpoint receipt"):
        ledger.latest()


def test_adapter_bundle_digest_and_manifest_receipt_are_consumer_exact(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "adapter_config.json").write_text('{"rank":16}\n')
    _tiny_safetensors(work / "adapters.safetensors")
    output = tmp_path / "release"
    output.mkdir()
    adapter, digest, required = publish_adapter_bundle(work, output)
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path / "model-root", config)
    corpus, _ = _write_corpus(tmp_path / "corpus-root")
    data = tmp_path / "release" / "data"
    dataset = prepare_mlx_dataset(corpus, data)
    report = run_preflight(
        config, model, data, output,
        resource_facts={
            "platform": "Darwin", "machine": "arm64",
            "memory_total_bytes": 64 * 1024 ** 3,
            "memory_available_bytes": 48 * 1024 ** 3,
            "disk_free_bytes": 100 * 1024 ** 3,
        }, package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"})
    manifest_path = output / "academic-adapter.manifest.json"

    manifest = build_adapter_manifest(
        config=config, base_receipt=report.facts["base_model"], dataset_manifest=dataset,
        dataset_manifest_path=data / "dataset_manifest.json",
        adapter_manifest_path=manifest_path, adapter_dir=adapter, bundle_digest=digest,
        required_files=required, package_versions={"mlx": "0.31.2", "mlx_lm": "0.31.3"})

    assert manifest["schema_version"] == ADAPTER_SCHEMA
    assert manifest["adapter"]["path"] == "adapter"
    assert manifest["adapter"]["sha256"] == adapter_bundle_digest(adapter)[0]
    assert [row["path"] for row in manifest["adapter"]["required_files"]] == [
        "adapter_config.json", "adapters.safetensors"]
    assert manifest["training"]["corpus_manifest_sha256"] == sha256_file(
        corpus.with_name(f"{corpus.name}.manifest.json"))
    assert manifest["dataset"]["dataset_manifest_path"] == "data/dataset_manifest.json"
    assert manifest["runtime"]["transport_adapter"] == "openai-compatible"
    assert manifest["runtime"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert manifest["runtime"]["default_adapter_strength"] == 1.0


def test_shared_compute_lease_fails_closed_when_another_owner_holds_it(tmp_path):
    first = TrainingComputeLease(tmp_path / "spiral-compute.lease")
    second = TrainingComputeLease(tmp_path / "spiral-compute.lease")
    first.acquire(owner={"model": "one"})
    try:
        with pytest.raises(HarnessError, match="busy"):
            second.acquire(owner={"model": "two"})
        owner = json.loads((tmp_path / "spiral-compute.lease").read_text())
        assert owner["model"] == "one"
        assert owner["type"] == "spiral_academic_qlora"
    finally:
        first.release()


def test_server_rehashes_full_runtime_and_emits_frozen_identity(tmp_path):
    manifest, view, _ = _runtime_fixture(tmp_path)

    assets = validate_runtime_assets(manifest, view)

    assert derive_model_view_path(manifest, tmp_path / "view-cache") == view
    assert assets.identity == {
        "schema_version": IDENTITY_SCHEMA,
        "manifest_sha256": sha256_file(manifest),
        "adapter_tree_sha256": assets.manifest["adapter"]["sha256"],
        "base_model_id": "mlx-community/Qwen3.8-27B-4bit",
        "base_model_revision": REVISION,
        "base_weight_inventory_sha256": assets.manifest["base_model"]["weight_inventory_sha256"],
        "profile_id": "academic-hep-pubmed-v1",
        "provider": "mlx_lm",
        "model": "qwen3.8-27b-academic",
        "transport_adapter": "openai-compatible",
        "adapter_strength_supported": True,
        "adapter_strength_min": MIN_ADAPTER_STRENGTH,
        "adapter_strength_max": MAX_ADAPTER_STRENGTH,
        "adapter_strength_step": ADAPTER_STRENGTH_STEP,
        "adapter_strength_default": DEFAULT_ADAPTER_STRENGTH,
        "server_contract": "spiral.academic-one-request-server.v1",
        "weight_residency": "child-process-per-request",
        "compute_lease": "spiral-compute-flock-v1",
        "ollama_admission": "strict-empty-no-eviction",
        "unload_boundary": "child-exit-before-lease-release",
    }
    health = identity_health_response(assets.identity)
    completion = openai_completion_response(
        {"text": "academic prose", "prompt_tokens": 4,
         "completion_tokens": 2, "finish_reason": "stop"},
        assets.identity, completion_id="chatcmpl-test", created=1)
    assert health["spiral_runtime_identity"] == assets.identity
    assert completion["spiral_runtime_identity"] == assets.identity
    assert completion["spiral_adapter_strength"] == 1.0
    assert set(health["spiral_runtime_identity"]) == set(assets.identity)
    assert set(completion["spiral_runtime_identity"]) == set(assets.identity)


def test_sidecar_readiness_fails_after_live_storage_disappears(tmp_path):
    manifest, view, _ = _runtime_fixture(tmp_path)
    text_assets = validate_runtime_assets(manifest, view)
    validate_runtime_storage_sentinel(text_assets)

    parked_manifest = manifest.with_suffix(".offline")
    manifest.rename(parked_manifest)
    with pytest.raises(HarnessError, match="storage is unavailable"):
        validate_runtime_storage_sentinel(text_assets)
    parked_manifest.rename(manifest)

    model_root = tmp_path / "model" / "snapshots" / REVISION
    vlm_assets = validate_vlm_runtime_assets(manifest, model_root)
    validate_vlm_storage_sentinel(vlm_assets)
    base_file = model_root / vlm_assets.base_file_snapshot[0]["path"]
    parked_base = base_file.with_suffix(base_file.suffix + ".offline")
    base_file.rename(parked_base)
    with pytest.raises(HarnessError, match="base file is unavailable"):
        validate_vlm_storage_sentinel(vlm_assets)


def test_inference_deployment_does_not_reopen_historical_training_data(tmp_path):
    manifest, view, _ = _runtime_fixture(tmp_path)
    receipt = json.loads(manifest.read_text())
    corpus_manifest = (
        manifest.parent / receipt["training"]["corpus_manifest_path"]
    ).resolve()
    dataset_manifest = (
        manifest.parent / receipt["dataset"]["dataset_manifest_path"]
    ).resolve()
    dataset = json.loads(dataset_manifest.read_text())
    source = Path(dataset["source_corpus"])
    for entry in dataset["splits"].values():
        (dataset_manifest.parent / entry["path"]).unlink()
    source.unlink()
    dataset_manifest.unlink()
    corpus_manifest.unlink()

    # The signed adapter manifest retains the exact corpus/dataset SHA-256 receipt,
    # while generation remains deployable using only its actual runtime assets.
    text_assets = validate_runtime_assets(manifest, view)
    model_root = tmp_path / "model" / "snapshots" / REVISION
    vlm_assets = validate_vlm_runtime_assets(manifest, model_root)
    validate_runtime_storage_sentinel(text_assets)
    validate_vlm_storage_sentinel(vlm_assets)


def test_server_rejects_mutated_adapter_before_model_load(tmp_path):
    manifest, view, adapter = _runtime_fixture(tmp_path)
    validate_runtime_assets(manifest, view)
    (adapter / "adapter_config.json").write_text('{"rank":8}\n')

    with pytest.raises(HarnessError, match="adapter file does not match"):
        validate_runtime_assets(manifest, view)


def test_server_is_loopback_only_and_bounds_text_chat_requests():
    validate_bind_address("127.0.0.1", 8080, "http://127.0.0.1:8080/v1")
    with pytest.raises(HarnessError, match="loopback-only"):
        validate_bind_address("0.0.0.0", 8080, "http://127.0.0.1:8080/v1")

    clean = validate_chat_request({
        "model": "qwen3.8-27b-academic",
        "messages": [{"role": "user", "content": "Draft one sentence."}],
        "temperature": 0.4,
        "max_tokens": 256,
        "adapter_strength": 1.35,
    }, expected_model="qwen3.8-27b-academic")
    assert clean["max_tokens"] == 256
    assert clean["adapter_strength"] == 1.35
    with pytest.raises(HarnessError, match="streaming"):
        validate_chat_request({
            "model": "qwen3.8-27b-academic", "stream": True,
            "messages": [{"role": "user", "content": "draft"}],
        }, expected_model="qwen3.8-27b-academic")

    for invalid_strength in (-0.05, 2.05, 0.333, True, "1.0"):
        with pytest.raises(HarnessError, match="adapter_strength"):
            validate_chat_request({
                "model": "qwen3.8-27b-academic",
                "messages": [{"role": "user", "content": "draft"}],
                "adapter_strength": invalid_strength,
            }, expected_model="qwen3.8-27b-academic")


def test_adapter_strength_changes_loaded_lora_forward_scales_without_mutating_bundle():
    class FakeLoRA:
        def __init__(self, scale):
            self.scale = scale

    class OtherModule:
        scale = 99.0

    class FakeModel:
        def __init__(self):
            self.modules = [FakeLoRA(32.0), FakeLoRA(20.0), OtherModule()]

        def named_modules(self):
            return [(str(index), module) for index, module in enumerate(self.modules)]

    model = FakeModel()
    receipt = apply_adapter_strength(model, 1.25, lora_types=(FakeLoRA,))

    assert [module.scale for module in model.modules] == [40.0, 25.0, 99.0]
    assert receipt == {
        "adapter_strength": 1.25,
        "lora_module_count": 2,
        "trained_scales": [20.0, 32.0],
        "effective_scales": [25.0, 40.0],
    }
    base_model = FakeModel()
    base_receipt = apply_adapter_strength(base_model, 0.0, lora_types=(FakeLoRA,))
    assert [module.scale for module in base_model.modules[:2]] == [0.0, 0.0]
    assert base_receipt["effective_scales"] == [0.0]
    full_model = FakeModel()
    apply_adapter_strength(full_model, 1.0, lora_types=(FakeLoRA,))
    assert [module.scale for module in full_model.modules[:2]] == [32.0, 20.0]


@pytest.mark.parametrize("value", [0.0, 0.05, 1, 1.35, 2.0])
def test_adapter_strength_grid_accepts_full_supported_range(value):
    assert canonical_adapter_strength(value, request=True) == float(value)


def test_full_vlm_runtime_rehashes_manifest_base_adapter_vision_and_tool_template(tmp_path):
    manifest, _, adapter = _runtime_fixture(tmp_path)
    model_root = tmp_path / "model" / "snapshots" / REVISION

    assets = validate_vlm_runtime_assets(manifest, model_root)

    assert assets.adapter_dir == adapter.resolve()
    assert assets.model_root == model_root.resolve()
    assert assets.identity["schema_version"] == VLM_IDENTITY_SCHEMA
    assert assets.identity["model"] == EXPECTED_VLM_RUNTIME_MODEL
    assert assets.identity["provider"] == "mlx_vlm"
    assert assets.identity["transport_adapter"] == "ollama-compatible"
    assert assets.identity["manifest_sha256"] == sha256_file(manifest)
    assert assets.identity["adapter_tree_sha256"] == assets.manifest["adapter"]["sha256"]
    assert assets.identity["base_weight_inventory_sha256"] == (
        assets.manifest["base_model"]["weight_inventory_sha256"])
    for key, expected in VLM_SERVER_LIFECYCLE_IDENTITY.items():
        assert assets.identity[key] == expected
    assert assets.identity["lease_handoff"] == LEASE_HANDOFF_CONTRACT
    assert assets.identity["vision_supported"] is True
    assert assets.identity["tools_supported"] is True
    assert assets.identity["streaming_supported"] is True
    assert assets.identity["thinking_supported"] is True
    assert len(assets.identity["vlm_frontend_inventory_sha256"]) == 64
    fast = validate_vlm_runtime_assets(
        manifest, model_root, expected_base_snapshot=assets.base_file_snapshot)
    assert fast.identity == assets.identity

    (model_root / "tokenizer.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="metadata changed"):
        validate_vlm_runtime_assets(
            manifest, model_root, expected_base_snapshot=assets.base_file_snapshot)
    (model_root / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    (model_root / "chat_template.jinja").write_text("no tool grammar", encoding="utf-8")
    with pytest.raises(HarnessError, match="tool grammar"):
        validate_vlm_runtime_assets(manifest, model_root)


def test_vlm_request_preserves_images_tools_stream_and_true_strength():
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nsmall-test").decode("ascii")
    request = {
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "Read it", "images": [png]}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "web_search", "description": "Search",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}}},
            },
        }],
        "stream": True,
        "think": True,
        "adapter_strength": 1.35,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 96, "seed": 7},
    }

    clean = validate_vlm_chat_request(request, expected_model=EXPECTED_VLM_RUNTIME_MODEL)

    assert clean["messages"][0]["images"] == [png]
    assert clean["tools"][0]["function"]["name"] == "web_search"
    assert clean["stream"] is True
    assert clean["think"] is True
    assert clean["adapter_strength"] == 1.35
    assert clean["max_tokens"] == 96
    assert clean["seed"] == 7
    for invalid in (-0.05, 2.05, 0.333, "1.0", True):
        rejected = copy.deepcopy(request)
        rejected["adapter_strength"] = invalid
        with pytest.raises(HarnessError, match="adapter_strength"):
            validate_vlm_chat_request(
                rejected, expected_model=EXPECTED_VLM_RUNTIME_MODEL)

    second_round = copy.deepcopy(request)
    second_round["messages"] = [
        {"role": "user", "content": "Find the paper"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": "web_search", "arguments": {"query": "paper"}},
        }]},
        {"role": "tool", "tool_name": "web_search", "content": "one result"},
    ]
    resumed = validate_vlm_chat_request(
        second_round, expected_model=EXPECTED_VLM_RUNTIME_MODEL)
    assert resumed["messages"][1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert resumed["messages"][2]["role"] == "tool"
    assert resumed["messages"][2]["name"] == "web_search"


def test_vlm_multiturn_images_keep_explicit_turn_association():
    png_a = base64.b64encode(b"\x89PNG\r\n\x1a\nfirst").decode("ascii")
    png_b = base64.b64encode(b"\x89PNG\r\n\x1a\nsecond").decode("ascii")
    clean = validate_vlm_chat_request({
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [
            {"role": "user", "content": "first image", "images": [png_a]},
            {"role": "assistant", "content": "I saw the first."},
            {"role": "user", "content": "now compare this", "images": [png_b]},
        ],
    }, expected_model=EXPECTED_VLM_RUNTIME_MODEL)

    processor_messages = vlm_processor_messages(clean["messages"])

    assert processor_messages[0]["content"] == [
        {"type": "text", "text": "first image"}, {"type": "image"}]
    assert processor_messages[1]["content"] == "I saw the first."
    assert processor_messages[2]["content"] == [
        {"type": "text", "text": "now compare this"}, {"type": "image"}]


def test_vlm_strength_multiplies_trained_scale32_without_rewriting_it():
    config = {"lora_parameters": {"rank": 16, "scale": 32.0, "dropout": 0.0}}

    assert lora_parameters_for_strength(config, 0.0)["scale"] == 0.0
    assert lora_parameters_for_strength(config, 1.0)["scale"] == 32.0
    assert lora_parameters_for_strength(config, 1.25)["scale"] == 40.0
    assert config["lora_parameters"]["scale"] == 32.0
    with pytest.raises(HarnessError, match="scale32"):
        lora_parameters_for_strength(
            {"lora_parameters": {"rank": 16, "scale": 16.0, "dropout": 0.0}}, 1.0)


@pytest.mark.parametrize("tier", [8192, 16384, 32768])
def test_vlm_accepts_every_apk_effort_token_tier(tier):
    clean = validate_vlm_chat_request({
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "continue"}],
        "options": {"num_predict": tier},
    }, expected_model=EXPECTED_VLM_RUNTIME_MODEL)
    assert clean["max_tokens"] == tier

    if tier == 32768:
        with pytest.raises(HarnessError, match="32768"):
            validate_vlm_chat_request({
                "model": EXPECTED_VLM_RUNTIME_MODEL,
                "messages": [{"role": "user", "content": "continue"}],
                "options": {"num_predict": 32769},
            }, expected_model=EXPECTED_VLM_RUNTIME_MODEL)


def test_vlm_default_timeout_preserves_max_effort_generation_window():
    assert vlm_server.parser().parse_args([]).request_timeout == 7200.0


def test_owner_only_lease_token_and_constant_time_handoff(tmp_path):
    token_path = tmp_path / "lease-authority.token"
    token_path.write_text("a" * 48, encoding="ascii")
    token_path.chmod(0o600)
    configured = load_lease_authority_token(token_path)

    assert configured == b"a" * 48
    assert trusted_lease_handoff(None, configured) is False
    assert trusted_lease_handoff("a" * 48, configured) is True
    with pytest.raises(HarnessError, match="handoff") as error:
        trusted_lease_handoff("b" * 48, configured)
    assert getattr(error.value, "status", None) == 403

    token_path.chmod(0o644)
    with pytest.raises(HarnessError, match="0600"):
        load_lease_authority_token(token_path)
    token_path.chmod(0o600)
    symlink = tmp_path / "linked-token"
    symlink.symlink_to(token_path)
    with pytest.raises(HarnessError, match="securely open"):
        load_lease_authority_token(symlink)


def test_trusted_host_handoff_never_reacquires_its_own_compute_flock(monkeypatch, tmp_path):
    calls = []

    class DeadlockingLease:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("trusted handoff attempted a second flock acquisition")

    monkeypatch.setattr(vlm_server, "TrainingComputeLease", DeadlockingLease)
    monkeypatch.setattr(vlm_server, "verify_ollama_empty", lambda url: calls.append(url))

    with compute_admission(
        trusted_handoff=True, lease_path=tmp_path / "compute.lease",
        ollama_url="http://127.0.0.1:11434", owner={"model": "qwen3.8:27b"},
    ) as lease:
        assert lease is None

    assert calls == ["http://127.0.0.1:11434"]


def test_standalone_worker_exits_before_compute_flock_release(monkeypatch, tmp_path):
    order = []

    class Lease:
        descriptor = None

        def __init__(self, _path):
            pass

        def acquire(self, *, owner):
            order.append("lease-acquire")

        def release(self):
            order.append("lease-release")

    class Worker:
        alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            order.append("worker-terminate")

        def wait(self, timeout):
            self.alive = False
            order.append("worker-exit")
            return 0

    monkeypatch.setattr(vlm_server, "TrainingComputeLease", Lease)
    monkeypatch.setattr(vlm_server, "verify_ollama_empty", lambda _url: None)
    worker = Worker()

    with pytest.raises(RuntimeError, match="cancel"):
        with compute_admission(
            trusted_handoff=False, lease_path=tmp_path / "compute.lease",
            ollama_url="http://127.0.0.1:11434", owner={"model": "qwen"},
        ):
            with vlm_server._worker_lifecycle({"process": worker}):
                raise RuntimeError("cancel")

    assert order == [
        "lease-acquire", "worker-terminate", "worker-exit", "lease-release"]


def test_ollama_ndjson_flushes_first_delta_before_worker_finishes():
    framed = vlm_server.canonical_json({"frame": 1})
    assert framed.endswith(b"\n") and not framed.endswith(b"\n\n")
    release = threading.Event()
    worker_finished = threading.Event()
    identity = {
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "adapter_strength_default": 1.0,
    }

    class StreamingService:
        @property
        def identity(self):
            return identity

        def events(self, request, *, lease_authority_header):
            assert request["adapter_strength"] == 1.0
            assert request["tools"][0]["function"]["name"] == "web_search"
            assert lease_authority_header is None
            yield {
                "type": "delta", "text": "first ", "thinking": "reason ",
                "spiral_adapter_strength": 1.0,
            }
            assert release.wait(2), "test client never released the worker"
            worker_finished.set()
            yield {
                "type": "result", "text": "first final", "tool_calls": [],
                "thinking": "reason ", "content_streamed": True,
                "thinking_streamed": True,
                "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 2,
                "spiral_adapter_strength": 1.0,
            }

    server = AcademicVlmHTTPServer(("127.0.0.1", 0), StreamingService())
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    body = json.dumps({
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "look"}],
        "tools": [{"type": "function", "function": {
            "name": "web_search", "parameters": {"type": "object"}}}],
        "stream": True, "adapter_strength": 1.0,
    })
    started = time.monotonic()
    connection.request("POST", "/api/chat", body=body, headers={
        "Content-Type": "application/json", "Content-Length": str(len(body)),
    })
    response = connection.getresponse()
    first = json.loads(response.readline())

    assert response.status == 200
    assert time.monotonic() - started < 1.0
    assert first["message"]["content"] == "first "
    assert first["message"]["thinking"] == "reason "
    assert first["done"] is False
    assert first["spiral_adapter_strength"] == 1.0
    assert not worker_finished.is_set()

    release.set()
    final = json.loads(response.readline())
    assert final["done"] is True
    assert final["message"]["content"] == ""
    assert "thinking" not in final["message"]
    assert final["spiral_adapter_strength"] == 1.0
    thread.join(timeout=2)
    connection.close()
    server.server_close()


def test_committed_ndjson_failure_emits_terminal_frame_not_second_http_status():
    identity = {"model": EXPECTED_VLM_RUNTIME_MODEL}

    class FailingService:
        @property
        def identity(self):
            return identity

        def events(self, request, *, lease_authority_header):
            yield {
                "type": "delta", "text": "partial", "thinking": "",
                "spiral_adapter_strength": 1.0,
            }
            raise HarnessError("worker vanished")

    server = AcademicVlmHTTPServer(("127.0.0.1", 0), FailingService())
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    body = json.dumps({
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "go"}], "stream": True,
    })
    connection.request("POST", "/api/chat", body=body, headers={
        "Content-Type": "application/json", "Content-Length": str(len(body)),
    })
    response = connection.getresponse()
    first = json.loads(response.readline())
    terminal = json.loads(response.readline())

    assert response.status == 200 and first["done"] is False
    assert terminal["done"] is True
    assert terminal["done_reason"] == "error"
    assert terminal["error"] == "worker vanished"
    assert terminal["spiral_adapter_strength"] == 1.0
    remainder = response.read()
    assert b"HTTP/1.1" not in remainder
    assert remainder == b""
    thread.join(timeout=2)
    connection.close()
    server.server_close()


@pytest.mark.parametrize("with_delta", [False, True])
def test_worker_result_is_withheld_when_post_result_cleanup_exits_nonzero(
    tmp_path, with_delta,
):
    identity = {"model": EXPECTED_VLM_RUNTIME_MODEL, "receipt": "exact"}
    strength = 1.25
    result = {
        "type": "result",
        "text": "must not be published",
        "spiral_adapter_strength": strength,
        "spiral_runtime_identity": identity,
        "lora_module_count": 20,
        "timings": {},
    }
    read_fd, write_fd = os.pipe()
    try:
        if with_delta:
            os.write(write_fd, vlm_server.canonical_json({
                "type": "delta", "text": "partial",
                "spiral_adapter_strength": strength,
            }))
        os.write(write_fd, vlm_server.canonical_json(result))
    finally:
        os.close(write_fd)
    log_path = tmp_path / "worker.log"
    log_path.write_text("post-result cleanup crashed", encoding="utf-8")

    class FailedAfterResult:
        args = ["fake-worker"]

        def poll(self):
            return 7

        def wait(self, timeout):
            return 7

    events = vlm_server._attested_worker_events(
        process=FailedAfterResult(), descriptor=read_fd, timeout=2,
        expected_identity=identity, expected_strength=strength,
        parent_attestation_seconds=0.01, log_path=log_path,
    )
    try:
        if with_delta:
            assert next(events)["type"] == "delta"
        with pytest.raises(HarnessError, match="post-result cleanup crashed"):
            next(events)
    finally:
        os.close(read_fd)


@pytest.mark.parametrize(("headers", "status"), [
    ({"Content-Type": "text/plain"}, 415),
    ({"Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
    ({"Content-Type": "application/json", "Host": "evil.example"}, 403),
])
def test_vlm_http_rejects_browser_cross_site_non_json_and_nonloopback_host(headers, status):
    class NeverRuns:
        identity = {"model": EXPECTED_VLM_RUNTIME_MODEL}

        def events(self, *_args, **_kwargs):
            raise AssertionError("rejected HTTP request reached the model")

    server = AcademicVlmHTTPServer(("127.0.0.1", 0), NeverRuns())
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    body = json.dumps({
        "model": EXPECTED_VLM_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "go"}],
    })
    request_headers = {"Content-Length": str(len(body)), **headers}
    connection.request("POST", "/api/chat", body=body, headers=request_headers)
    response = connection.getresponse()
    payload = json.loads(response.read())

    assert response.status == status
    assert payload["code"] in {"forbidden", "unsupported_media_type"}
    thread.join(timeout=2)
    connection.close()
    server.server_close()


def test_vlm_bind_and_ollama_receipts_are_exact():
    validate_vlm_bind_address("127.0.0.1", 8081)
    with pytest.raises(HarnessError, match="loopback-only"):
        validate_vlm_bind_address("0.0.0.0", 8081)
    identity = {"model": EXPECTED_VLM_RUNTIME_MODEL}
    delta = ollama_delta(identity, "piece", "reason", adapter_strength=1.25)
    assert delta["done"] is False
    assert delta["message"]["thinking"] == "reason"
    assert delta["spiral_adapter_strength"] == 1.25
    result = ollama_result(identity, {
        "text": "answer", "finish_reason": "stop",
        "prompt_tokens": 3, "completion_tokens": 2,
        "spiral_adapter_strength": 1.25,
    })
    assert result["message"]["content"] == "answer"
    assert result["spiral_adapter_strength"] == 1.25
    tool_result = ollama_result(identity, {
        "text": "", "tool_calls": [{"function": {
            "name": "web_search", "arguments": {"query": "paper"}}}],
        "finish_reason": "tool_calls", "spiral_adapter_strength": 1.0,
    })
    assert tool_result["message"]["tool_calls"][0]["function"]["name"] == "web_search"


def test_smoke_override_is_irrevocably_marked_non_deployable(tmp_path):
    corpus, _ = _write_corpus(tmp_path)
    data = tmp_path / "data"
    smoke_model = tmp_path / "qwen25-smoke"
    smoke_model.mkdir()
    (smoke_model / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
        "quantization": {"bits": 4, "group_size": 64},
    }))
    (smoke_model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.layers.0.self_attn.q_proj.weight": "model.safetensors"}
    }))
    (smoke_model / "model.safetensors").write_bytes(b"fake")
    output = tmp_path / "smoke-output"

    exit_code = train_main([
        "--corpus", str(corpus), "--data-dir", str(data), "--output", str(output),
        "--smoke-model", str(smoke_model),
    ])

    assert exit_code == 0
    marker = json.loads((output / "SMOKE_ONLY" / "SMOKE_ONLY.json").read_text())
    assert marker["deployable"] is False
    assert "max_seq_length: 640" in (
        output / "SMOKE_ONLY" / "mlx_lora.yaml").read_text()
    assert not (output / "academic-adapter.manifest.json").exists()


def test_training_metric_parser_is_exact_and_rejects_nonfinite_loss():
    train = parse_training_metric(
        "Iter 8: Train loss 2.367, Learning Rate 1.000e-05, It/sec 1.250, "
        "Tokens/sec 321.500, Trained Tokens 2048, Peak mem 0.725 GB",
        mode="smoke_only")
    invalid = parse_training_metric(
        "Iter 8: Val loss nan, Val took 0.420s", mode="smoke_only")

    assert train == {
        "schema_version": "spiral.academic-training-metric.v1",
        "mode": "smoke_only", "event": "train", "iteration": 8,
        "phase_iteration": 8, "valid": True, "train_loss": 2.367,
        "learning_rate": 1e-5, "iterations_per_second": 1.25,
        "tokens_per_second": 321.5, "peak_memory_gb": 0.725,
        "trained_tokens": 2048,
    }
    assert invalid["valid"] is False
    assert "val_loss" not in invalid
    assert invalid["rejected_values"]["val_loss"] == "nan"
    assert parse_training_metric("unrelated trainer output", mode="smoke_only") is None


def test_27b_feasibility_dry_run_is_exact_bounded_and_non_deployable(tmp_path, monkeypatch):
    config = _config(tiny_model=True)
    model = _fake_qwen38_model(tmp_path / "model", config)
    base = validate_local_base_model(model, config)
    corpus, _ = _write_corpus(tmp_path / "corpus")
    output = tmp_path / "feasibility-run"

    report = SimpleNamespace(
        facts={"base_model": base}, ok=True,
        as_dict=lambda: {"schema_version": "test", "ok": True},
        require_ok=lambda: None)
    monkeypatch.setattr(
        "scripts.academic_finetune.train_qlora.python_package_versions",
        lambda _python: {"mlx": "0.31.2", "mlx_lm": "0.31.3"})
    monkeypatch.setattr(
        "scripts.academic_finetune.train_qlora.run_preflight",
        lambda *_args, **_kwargs: report)

    result = train_main([
        "--corpus", str(corpus), "--data-dir", str(tmp_path / "data"),
        "--model", str(model), "--output", str(output),
        "--model-view-cache", str(tmp_path / "view-cache"),
        "--feasibility-iters", "1",
    ])

    assert result == 0
    marker = json.loads((
        output / "FEASIBILITY_ONLY" / "FEASIBILITY_ONLY.json").read_text())
    assert marker["deployable"] is False
    assert marker["iterations"] == 1
    assert marker["training_contract"]["exact_production_config"] is True
    yaml = (output / "FEASIBILITY_ONLY" / "mlx_lora.yaml").read_text()
    assert "iters: 1" in yaml
    assert "grad_accumulation_steps: 8" in yaml
    assert '    - "linear_attn.in_proj_qkv"' in yaml
    assert not (output / "academic-adapter.manifest.json").exists()


def test_metrics_journal_is_append_only_resume_deduplicated_and_refreshes_html(tmp_path):
    metrics = tmp_path / "training-metrics.jsonl"
    view = tmp_path / "loss-curves.html"
    journal = TrainingMetricsJournal(
        metrics, mode="production", run_id="first", cumulative_offset=80,
        html_path=view)
    line = (
        "Iter 1: Train loss 1.750, Learning Rate 1.000e-05, It/sec 1.000, "
        "Tokens/sec 100.000, Trained Tokens 512, Peak mem 20.000 GB")
    assert journal.consume(line)["iteration"] == 81
    assert journal.consume(line) is None
    resumed = TrainingMetricsJournal(
        metrics, mode="production", run_id="second", cumulative_offset=80,
        html_path=view)
    assert resumed.consume(line) is None

    rows = [json.loads(value) for value in metrics.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "first"
    assert "train 1.750" in view.read_text()
    resumed.require_valid()


def test_training_process_streams_metrics_without_blocking_and_fails_nan_after_exit(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ledger = CheckpointLedger(tmp_path / "checkpoints")
    train_line = (
        "Iter 1: Train loss 3.125, Learning Rate 5.000e-06, It/sec 2.000, "
        "Tokens/sec 200.000, Trained Tokens 128, Peak mem 0.750 GB")
    result = run_training_process(
        [sys.executable, "-c", f"print({train_line!r}, flush=True)"],
        work, ledger, poll_seconds=0.01,
        metrics_path=tmp_path / "metrics.jsonl", metrics_mode="smoke_only",
        metrics_run_id="integration", trainer_log_path=tmp_path / "trainer.log")
    assert result == 0
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["train_loss"] == 3.125
    assert train_line in (tmp_path / "trainer.log").read_text()
    assert (tmp_path / "loss-curves.html").is_file()

    invalid_script = f"print({train_line!r}); print('Iter 1: Val loss nan, Val took 0.1s')"
    with pytest.raises(HarnessError, match="nonfinite"):
        run_training_process(
            [sys.executable, "-c", invalid_script], work,
            CheckpointLedger(tmp_path / "bad-checkpoints"), poll_seconds=0.01,
            metrics_path=tmp_path / "bad-metrics.jsonl", metrics_mode="smoke_only",
            metrics_run_id="invalid")


def test_fake_trainer_crash_after_validation_keeps_checkpoint_and_resumes(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    ledger = CheckpointLedger(checkpoints, run_identity="a" * 64)
    metrics = tmp_path / "training-metrics.jsonl"
    first_work = tmp_path / "attempt-1"
    first_work.mkdir()
    train_40 = (
        "Iter 40: Train loss 2.500, Learning Rate 5.000e-06, It/sec 1.000, "
        "Tokens/sec 100.000, Trained Tokens 4000, Peak mem 20.000 GB")
    fake_crash = f"""
import json
import struct
from pathlib import Path

work = Path({str(first_work)!r})
(work / "adapter_config.json").write_text(json.dumps({{"rank": 16}}) + "\\n")
header = json.dumps({{
    "weight": {{"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
}}, separators=(",", ":")).encode()
header += b" " * ((8 - len(header) % 8) % 8)
(work / "0000040_adapters.safetensors").write_bytes(
    struct.pack("<Q", len(header)) + header + b"1234")
print({train_40!r}, flush=True)
print("Iter 41: Val loss 2.250, Val took 0.100s", flush=True)
raise SystemExit(80)
"""

    first_result = run_training_process(
        [sys.executable, "-c", fake_crash], first_work, ledger,
        poll_seconds=0.01, metrics_path=metrics, metrics_mode="production",
        metrics_run_id="attempt-1")

    assert first_result == 80
    assert ledger.latest().step == 40
    assert ledger.latest().config_path.is_file()

    second_work = tmp_path / "attempt-2"
    second_work.mkdir()
    train_40_resumed = (
        "Iter 40: Train loss 2.125, Learning Rate 5.000e-06, It/sec 1.000, "
        "Tokens/sec 100.000, Trained Tokens 4000, Peak mem 20.000 GB")
    fake_resume = f"""
import json
import struct
from pathlib import Path

work = Path({str(second_work)!r})
(work / "adapter_config.json").write_text(json.dumps({{"rank": 16}}) + "\\n")
header = json.dumps({{
    "weight": {{"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
}}, separators=(",", ":")).encode()
header += b" " * ((8 - len(header) % 8) % 8)
(work / "0000040_adapters.safetensors").write_bytes(
    struct.pack("<Q", len(header)) + header + b"5678")
print({train_40_resumed!r}, flush=True)
"""

    second_result = run_training_process(
        [sys.executable, "-c", fake_resume], second_work, ledger,
        cumulative_offset=40, poll_seconds=0.01, metrics_path=metrics,
        metrics_mode="production", metrics_run_id="attempt-2")

    assert second_result == 0
    assert ledger.latest().step == 80
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert [(row["event"], row["iteration"]) for row in rows] == [
        ("train", 40), ("validation", 41), ("train", 80)]


def test_evaluation_uses_nll_semantic_argument_and_citation_not_exact_match_only(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    held_out = records[-1]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "schema_version": "spiral.academic-predictions.v1",
        "example_id": held_out["example_id"],
        "base": {
            "text": "This result is obviously conclusive [99].",
            "target_nll": 3.2,
        },
        "adapter": {
            "text": "However, the measured spectrum might constrain the effective interaction [12].",
            "target_nll": 1.4,
        },
    }) + "\n", encoding="utf-8")

    summary = evaluate_predictions(corpus, predictions, tmp_path / "evaluation", seed=17)

    assert summary["arms"]["adapter"]["mean_target_nll"] < summary["arms"]["base"]["mean_target_nll"]
    assert summary["arms"]["adapter"]["semantic_content_f1"] > summary["arms"]["base"]["semantic_content_f1"]
    assert summary["arms"]["adapter"]["citation_fidelity"] > summary["arms"]["base"]["citation_fidelity"]
    assert "exact_match_rate" not in summary["arms"]["adapter"]
    assert "exact_match_rate" in summary["diagnostics_not_selection_metrics"]["adapter"]
    packet = json.loads((tmp_path / "evaluation" / "blind_ab_packets.jsonl").read_text())
    answer = json.loads((tmp_path / "evaluation" / "blind_ab_answer_key.jsonl").read_text())
    assert "held_out_target" not in packet
    assert answer["held_out_target"] == held_out["target"]


def test_evaluation_requires_held_out_target_nll(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "example_id": records[-1]["example_id"],
        "base": {"text": "base"},
        "adapter": {"text": "adapter"},
    }) + "\n")

    with pytest.raises(HarnessError, match="target NLL is mandatory"):
        evaluate_predictions(corpus, predictions, tmp_path / "evaluation")


def test_base_nll_command_explicitly_disables_default_adapter_directory(tmp_path):
    base = mlx_nll_command(
        python_executable="python", model_view=tmp_path / "model",
        data_dir=tmp_path / "data", adapter_dir=None, max_sequence_length=512)
    adapted = mlx_nll_command(
        python_executable="python", model_view=tmp_path / "model",
        data_dir=tmp_path / "data", adapter_dir=tmp_path / "adapter",
        max_sequence_length=512)

    assert base[base.index("--adapter-path") + 1] == ""
    assert "--trust-remote-code" not in base
    assert adapted[adapted.index("--adapter-path") + 1] == str(tmp_path / "adapter")


def test_nll_runner_persists_failed_arm_log_before_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.academic_finetune.evaluate.verify_ollama_empty",
        lambda *_args, **_kwargs: {"verified_empty": True})
    monkeypatch.setattr(
        "scripts.academic_finetune.evaluate.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="mlx_lm: error: unrecognized arguments: --bad-option\n"))
    output = tmp_path / "target_nll.json"

    with pytest.raises(HarnessError, match="status 2.*target_nll.base.log"):
        run_mlx_nll(
            python_executable=sys.executable,
            model_view=tmp_path / "model-view",
            adapter_dir=tmp_path / "adapter",
            data_dir=tmp_path / "data",
            output_path=output,
            lease_path=tmp_path / "spiral-compute.lease",
            ollama_url="http://ollama.invalid",
        )

    assert output.with_suffix(".base.log").read_text() == (
        "mlx_lm: error: unrecognized arguments: --bad-option\n")
    assert not output.with_suffix(".adapter.log").exists()
    assert not output.exists()


def test_prediction_runner_uses_exact_coverage_and_identical_greedy_receipt(tmp_path, monkeypatch):
    corpus, records = _write_corpus(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"rank":16}\n')
    _tiny_safetensors(adapter / "adapters.safetensors")

    monkeypatch.setattr(
        "scripts.academic_finetune.evaluate.verify_ollama_empty",
        lambda *_args, **_kwargs: {"verified_empty": True})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        arm = command[command.index("--arm") + 1]
        requests = Path(command[command.index("--requests") + 1])
        rows = [json.loads(line) for line in requests.read_text().splitlines()]
        output.write_text("".join(json.dumps({
            "schema_version": "spiral.academic-prediction-arm.v1",
            "example_id": row["example_id"], "arm": arm,
            "text": f"{arm} generated sentence", "seed": 1,
            "decode": "greedy_argmax",
        }) + "\n" for row in rows))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("scripts.academic_finetune.evaluate.subprocess.run", fake_run)
    output = tmp_path / "predictions.jsonl"
    receipt_path = tmp_path / "prediction-receipt.json"

    receipt = run_mlx_predictions(
        corpus_path=corpus, python_executable=sys.executable,
        model_view=tmp_path / "model-view", adapter_dir=adapter,
        output_path=output, receipt_path=receipt_path,
        lease_path=tmp_path / "spiral-compute.lease",
        ollama_url="http://ollama.invalid", max_examples=1, seed=123)

    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["example_id"] for row in predictions] == [records[-1]["example_id"]]
    assert receipt["held_out_count"] == 1
    assert receipt["decode"]["strategy"] == "greedy_argmax"
    assert receipt["decode"]["identical_between_arms"] is True
    assert receipt["predictions_sha256"] == sha256_file(output)


def test_prediction_template_is_held_out_only_and_byte_stable(tmp_path):
    corpus, records = _write_corpus(tmp_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_prediction_template(corpus, first)
    write_prediction_template(corpus, second)

    assert first.read_bytes() == second.read_bytes()
    template = json.loads(first.read_text())
    assert template["example_id"] == records[-1]["example_id"]
    assert template["base"]["target_nll"] is None
    assert records[-1]["target"] not in template["prompt"]


def test_post_training_evaluation_finalization_is_durable_and_idempotent(tmp_path):
    manifest_path, model_view, adapter = _runtime_fixture(tmp_path)
    run = manifest_path.parent
    data = run / "data"
    corpus = tmp_path / "corpus" / "academic_corpus.jsonl"
    evaluation = run / "evaluation"
    evaluation.mkdir()
    bundle_digest, required_files = adapter_bundle_digest(adapter)
    run_identity = "a" * 64
    status_path = run / "training-status.json"
    atomic_write_json(status_path, {
        "schema_version": "spiral.academic-training-status.v1",
        "state": "completed",
        "updated_at": "2026-08-24T00:00:00Z",
        "run_identity": run_identity,
        "total_steps": 1200,
        "completed_steps": 1200,
        "remaining_steps": 0,
        "latest_checkpoint": {
            "step": 1200,
            "bundle_sha256": bundle_digest,
            "receipt_sha256": "b" * 64,
        },
        "post_training_validation_required": True,
    })
    held_out = _record(2, "test", STRATA[2])
    predictions = evaluation / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "schema_version": "spiral.academic-predictions.v1",
        "example_id": held_out["example_id"],
        "base": {"text": "The base sentence.", "target_nll": None},
        "adapter": {
            "text": "However, the measured spectrum may constrain the effective interaction [12].",
            "target_nll": None,
        },
    }) + "\n", encoding="utf-8")
    nll_path = evaluation / "target_nll.json"
    atomic_write_json(nll_path, {
        "schema_version": "spiral.academic-target-nll.v1",
        "completion_only": True,
        "test_split_sha256": sha256_file(data / "test.jsonl"),
        "model_view": str(model_view.resolve()),
        "adapter_path": str(adapter.resolve()),
        "base": {"mean_target_nll": 3.0},
        "adapter": {"mean_target_nll": 1.0},
    })
    (evaluation / "target_nll.base.log").write_text("Test loss 3.0\n")
    (evaluation / "target_nll.adapter.log").write_text("Test loss 1.0\n")
    evaluate_predictions(
        corpus, predictions, evaluation, nll_report_path=nll_path)
    atomic_write_json(evaluation / "prediction_receipt.json", {
        "schema_version": "spiral.academic-prediction-receipt.v1",
        "corpus_sha256": sha256_file(corpus),
        "held_out_count": 1,
        "predictions_sha256": sha256_file(predictions),
        "base_model_view": str(model_view.resolve()),
        "adapter": {
            "path": str(adapter.resolve()),
            "sha256": bundle_digest,
            "required_files": required_files,
        },
        "decode": {
            "strategy": "greedy_argmax", "temperature": 0.0,
            "seed": 24082026, "max_tokens": 256,
            "identical_between_arms": True,
        },
    })

    receipt = finalize_post_training_evaluation(
        training_run=run, evaluation_dir=evaluation, corpus_path=corpus,
        model_view=model_view, adapter_dir=adapter, data_dir=data)
    receipt_path = run / "post-training-evaluation.json"
    receipt_bytes = receipt_path.read_bytes()
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)

    assert receipt["run_identity"] == run_identity
    assert receipt["adapter"]["sha256"] == bundle_digest
    assert status["post_training_validation_required"] is False
    assert status["post_training_validation"]["state"] == "completed"
    assert status["post_training_validation"]["receipt_sha256"] == sha256_file(receipt_path)

    repeated = finalize_post_training_evaluation(
        training_run=run, evaluation_dir=evaluation, corpus_path=corpus,
        model_view=model_view, adapter_dir=adapter, data_dir=data)
    assert repeated == receipt
    assert receipt_path.read_bytes() == receipt_bytes
    assert status_path.read_bytes() == status_bytes


def test_post_training_evaluation_rejects_mutated_artifact_without_clearing_gate(tmp_path):
    manifest_path, model_view, adapter = _runtime_fixture(tmp_path)
    run = manifest_path.parent
    data = run / "data"
    corpus = tmp_path / "corpus" / "academic_corpus.jsonl"
    evaluation = run / "evaluation"
    evaluation.mkdir()
    bundle_digest, required_files = adapter_bundle_digest(adapter)
    status_path = run / "training-status.json"
    atomic_write_json(status_path, {
        "schema_version": "spiral.academic-training-status.v1",
        "state": "completed", "run_identity": "a" * 64,
        "total_steps": 1, "completed_steps": 1,
        "latest_checkpoint": {"step": 1, "bundle_sha256": bundle_digest},
        "post_training_validation_required": True,
    })
    predictions = evaluation / "predictions.jsonl"
    predictions.write_text("{}\n")
    atomic_write_json(evaluation / "prediction_receipt.json", {
        "schema_version": "spiral.academic-prediction-receipt.v1",
        "corpus_sha256": sha256_file(corpus),
        "held_out_count": 1,
        "predictions_sha256": "0" * 64,
        "base_model_view": str(model_view.resolve()),
        "adapter": {"path": str(adapter.resolve()), "sha256": bundle_digest,
                    "required_files": required_files},
    })
    for name in (
        "target_nll.json", "evaluation_summary.json",
    ):
        atomic_write_json(evaluation / name, {})
    for name in (
        "target_nll.base.log", "target_nll.adapter.log", "evaluation_examples.jsonl",
        "blind_ab_packets.jsonl", "blind_ab_answer_key.jsonl",
    ):
        (evaluation / name).write_text("")

    with pytest.raises(HarnessError, match="NLL report has the wrong schema"):
        finalize_post_training_evaluation(
            training_run=run, evaluation_dir=evaluation, corpus_path=corpus,
            model_view=model_view, adapter_dir=adapter, data_dir=data)
    assert json.loads(status_path.read_text())["post_training_validation_required"] is True
    assert not (run / "post-training-evaluation.json").exists()


def test_semantic_metric_allows_valid_nonidentical_realizations():
    target = "These observations suggest that the interaction is scale dependent."
    paraphrase = "The observations therefore suggest a scale-dependent interaction."

    assert paraphrase != target
    assert multiset_f1(target, paraphrase) > 0.7


def test_claim_coverage_ignores_structural_role_before_em_dash():
    claims = ["causal antecedent — curvature changes the effective coupling"]

    assert claim_coverage(claims, "Curvature changes the effective coupling.") == 1.0
    assert claim_coverage(claims, "The causal antecedent is discussed.") == 0.0


def test_argument_metric_covers_evidence_and_definition_relations():
    assert argument_score(
        "evidence", "Measurements support the hypothesis.",
        "The observed data support the hypothesis.") == 1.0
    assert argument_score(
        "definition", "We define the effective scale as the pole mass.",
        "The effective scale is defined as the pole mass.") == 1.0
