"""Model-free serving contracts for the academic prose and structure adapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.academic_finetune import serve_adapter as server
from scripts.academic_finetune.training_support import (
    ADAPTER_SCHEMA,
    EXPECTED_ARCHITECTURE,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_TYPE,
    EXPECTED_REVISION,
    PROFILE_ID,
    PROMPT_CONTRACT,
    STRUCTURE_PROFILE_ID,
    STRUCTURE_PROMPT_CONTRACT,
    HarnessError,
    atomic_write_json,
    sha256_bytes,
    sha256_file,
)
from spiral.academic_structure_contract import (
    BRIEF_TO_BLUEPRINT_TASK,
    brief_to_blueprint_input,
    format_structure_prompt,
)


def _runtime_manifest(tmp_path: Path, *, structure: bool) -> tuple[Path, Path]:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True)
    required_files = []
    digest_lines = []
    for name, payload in (
        ("adapter_config.json", b'{"rank":16,"scale":32.0}\n'),
        ("adapters.safetensors", b"model-free-structure-serving-fixture"),
    ):
        path = adapter_dir / name
        path.write_bytes(payload)
        digest = sha256_file(path)
        required_files.append({
            "path": name,
            "size_bytes": len(payload),
            "sha256": digest,
        })
        digest_lines.append(f"{name}\0{len(payload)}\0{digest}\n")
    adapter_digest = sha256_bytes(
        "".join(sorted(digest_lines)).encode("utf-8")
    )
    digest = "a" * 64
    runtime = {
        "provider": server.EXPECTED_PROVIDER,
        "model": (
            server.EXPECTED_STRUCTURE_RUNTIME_MODEL
            if structure
            else server.EXPECTED_RUNTIME_MODEL
        ),
        "base_url": "http://127.0.0.1:8080/v1",
        "transport_adapter": server.EXPECTED_TRANSPORT,
        "default_adapter_strength": 1.0,
    }
    if structure:
        runtime.update({
            "scope": server.STRUCTURE_RUNTIME_SCOPE,
            "spiralchat_eligible": False,
        })
    manifest = {
        "schema_version": ADAPTER_SCHEMA,
        "profile_id": STRUCTURE_PROFILE_ID if structure else PROFILE_ID,
        "prompt_contract": (
            STRUCTURE_PROMPT_CONTRACT if structure else PROMPT_CONTRACT
        ),
        "base_model": {
            "model_id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_REVISION,
            "model_type": EXPECTED_MODEL_TYPE,
            "architecture": EXPECTED_ARCHITECTURE,
            "config_sha256": digest,
            "weight_inventory_sha256": "b" * 64,
            "weight_files": [{
                "path": "model.safetensors",
                "size_bytes": 1,
                "sha256": "c" * 64,
            }],
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        },
        "adapter": {
            "path": "adapter",
            "format": "mlx_lm_lora",
            "sha256": adapter_digest,
            "sha256_semantics": server.ADAPTER_DIGEST_SEMANTICS,
            "required_files": required_files,
        },
        "training": {
            "completion_only_loss": True,
            "corpus_manifest_path": "training/structure-corpus.manifest.json",
            "corpus_manifest_sha256": digest,
        },
        "dataset": {
            "dataset_manifest_path": "training/dataset_manifest.json",
            "manifest_sha256": digest,
            "source_corpus_sha256": digest,
            "source_corpus_file_sha256": digest,
            "split_sha256": {
                "train": digest,
                "valid": digest,
                "test": digest,
            },
        },
        "runtime": runtime,
    }
    manifest_path = tmp_path / "academic-adapter.manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path, tmp_path / "model-view"


def _skip_model_view_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    # This suite exercises admission contracts only; it must never open or load a model.
    monkeypatch.setattr(server, "_validate_model_view", lambda *_args: None)


def _planner_request() -> dict:
    prompt_input = brief_to_blueprint_input(
        title="A bounded academic paper",
        abstract_brief=(
            "We derive a constrained result and organize its assumptions, method, "
            "evidence, limitations, and conclusion."
        ),
        discipline="theoretical_physics",
        genre="theory_article",
    )
    return {
        "model": server.EXPECTED_STRUCTURE_RUNTIME_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one JSON object. "
                    + server.STRUCTURE_REQUEST_SYSTEM_MARKER
                ),
            },
            {
                "role": "user",
                "content": format_structure_prompt(
                    BRIEF_TO_BLUEPRINT_TASK, prompt_input),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "adapter_strength": 1.0,
    }


def test_structure_manifest_admits_exact_planner_only_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, model_view = _runtime_manifest(tmp_path, structure=True)
    _skip_model_view_validation(monkeypatch)

    assets = server.validate_runtime_assets(manifest_path, model_view)

    assert assets.identity["profile_id"] == STRUCTURE_PROFILE_ID
    assert assets.identity["model"] == server.EXPECTED_STRUCTURE_RUNTIME_MODEL
    assert assets.identity["scope"] == server.STRUCTURE_RUNTIME_SCOPE
    assert assets.identity["spiralchat_eligible"] is False
    assert assets.identity["manifest_sha256"] == sha256_file(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", server.EXPECTED_RUNTIME_MODEL),
        ("scope", "general_chat"),
        ("spiralchat_eligible", True),
    ],
)
def test_structure_manifest_rejects_non_planner_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    manifest_path, model_view = _runtime_manifest(tmp_path, structure=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"][field] = value
    atomic_write_json(manifest_path, manifest)
    _skip_model_view_validation(monkeypatch)

    with pytest.raises(HarnessError, match="runtime|planner-only"):
        server.validate_runtime_assets(manifest_path, model_view)


def test_profile_and_prompt_contract_cannot_be_cross_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, model_view = _runtime_manifest(tmp_path, structure=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_contract"] = PROMPT_CONTRACT
    atomic_write_json(manifest_path, manifest)
    _skip_model_view_validation(monkeypatch)

    with pytest.raises(HarnessError, match="profile/prompt contract"):
        server.validate_runtime_assets(manifest_path, model_view)


def test_structure_requests_require_json_mode_and_planner_scope() -> None:
    request = _planner_request()
    clean = server.validate_chat_request(
        request,
        expected_model=server.EXPECTED_STRUCTURE_RUNTIME_MODEL,
        runtime_scope=server.STRUCTURE_RUNTIME_SCOPE,
    )
    assert clean["response_format"] == {"type": "json_object"}
    assert clean["messages"] == request["messages"]
    assert clean["model_messages"] == [request["messages"][1]]
    assert all(message["role"] != "system" for message in clean["model_messages"])

    missing_json = copy.deepcopy(request)
    missing_json.pop("response_format")
    with pytest.raises(HarnessError, match="response_format"):
        server.validate_chat_request(
            missing_json,
            expected_model=server.EXPECTED_STRUCTURE_RUNTIME_MODEL,
            runtime_scope=server.STRUCTURE_RUNTIME_SCOPE,
        )

    generic_chat = copy.deepcopy(request)
    generic_chat["messages"][1]["content"] = "Budget and outline this paper."
    with pytest.raises(HarnessError, match="prompt contract"):
        server.validate_chat_request(
            generic_chat,
            expected_model=server.EXPECTED_STRUCTURE_RUNTIME_MODEL,
            runtime_scope=server.STRUCTURE_RUNTIME_SCOPE,
        )


def test_structure_training_and_runtime_use_the_same_model_visible_prompt() -> None:
    from scripts.academic_finetune.training_support import format_structure_prompt as training_prompt

    prompt_input = brief_to_blueprint_input(
        title="Matched prompt contract",
        abstract_brief="A short scientific brief with a bounded evidential claim.",
        discipline="particle_phenomenology",
        genre="phenomenology_article",
    )
    runtime_prompt = format_structure_prompt(
        BRIEF_TO_BLUEPRINT_TASK, prompt_input)
    assert training_prompt({
        "task_type": BRIEF_TO_BLUEPRINT_TASK,
        "input": prompt_input,
    }) == runtime_prompt

    request = _planner_request()
    request["messages"][1]["content"] = runtime_prompt
    clean = server.validate_chat_request(
        request,
        expected_model=server.EXPECTED_STRUCTURE_RUNTIME_MODEL,
        runtime_scope=server.STRUCTURE_RUNTIME_SCOPE,
    )
    assert clean["model_messages"] == [{"role": "user", "content": runtime_prompt}]


def test_prose_serving_contract_remains_backward_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, model_view = _runtime_manifest(tmp_path, structure=False)
    _skip_model_view_validation(monkeypatch)

    assets = server.validate_runtime_assets(manifest_path, model_view)
    assert assets.identity["profile_id"] == PROFILE_ID
    assert assets.identity["model"] == server.EXPECTED_RUNTIME_MODEL
    assert "scope" not in assets.identity
    assert "spiralchat_eligible" not in assets.identity

    clean = server.validate_chat_request({
        "model": server.EXPECTED_RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "Draft one paragraph."}],
    }, expected_model=server.EXPECTED_RUNTIME_MODEL)
    assert clean["messages"][0]["content"] == "Draft one paragraph."
    with pytest.raises(HarnessError, match="prose runtime accepts text chat only"):
        server.validate_chat_request({
            "model": server.EXPECTED_RUNTIME_MODEL,
            "messages": [{"role": "user", "content": "Draft one paragraph."}],
            "response_format": {"type": "json_object"},
        }, expected_model=server.EXPECTED_RUNTIME_MODEL)
