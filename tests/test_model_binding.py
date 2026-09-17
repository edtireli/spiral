import os
from unittest import mock

import pytest

from spiral.config import Config
from spiral.model_binding import ModelBindingError, require_selected_local_model, selected_local_model


def test_bound_model_controls_every_config_load_and_local_seat(tmp_path):
    with mock.patch("pathlib.Path.home", return_value=tmp_path), mock.patch.dict(os.environ, {
        "SPIRAL_REQUIRED_LOCAL_MODEL": "chosen:27b", "SPIRAL_WORKER": "different:latest",
        "SPIRAL_PLANNER": "another:latest", "SPIRAL_UNCENSORED_MODEL": "old-heretic:latest",
    }):
        for _ in range(2):
            cfg = Config.load()
            assert {spec.name for spec in (cfg.worker, cfg.planner, cfg.escalation,
                    cfg.critic, cfg.research_auditor, cfg.janitor)} == {"chosen:27b"}
            assert cfg.research_notes_model == cfg.uncensored_model == "chosen:27b"
            assert not cfg.academic_writer.enabled and not cfg.academic_planner.enabled


def test_no_binding_preserves_standalone_configuration():
    with mock.patch.dict(os.environ, {"SPIRAL_REQUIRED_LOCAL_MODEL": ""}):
        assert selected_local_model() == ""
        require_selected_local_model("anything", {"anything": {}})


def test_explicit_run_context_binds_every_seat_without_writing_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    settings = tmp_path / '.config/spiral/config.json'
    settings.parent.mkdir(parents=True)
    original = '{"models":{"worker":"chosen:27b"},"num_ctx":{"chosen:27b":86016}}'
    settings.write_text(original)
    monkeypatch.setenv('SPIRAL_REQUIRED_LOCAL_MODEL', 'chosen:27b')
    monkeypatch.delenv('SPIRAL_RUN_CONTEXT_TOKENS', raising=False)
    assert Config.load().worker.num_ctx == 86016
    monkeypatch.setenv('SPIRAL_RUN_CONTEXT_TOKENS', '32768')
    cfg = Config.load()
    assert {seat.num_ctx for seat in (cfg.worker, cfg.planner, cfg.escalation,
            cfg.critic, cfg.research_auditor, cfg.janitor)} == {32768}
    assert settings.read_text() == original
    monkeypatch.delenv('SPIRAL_RUN_CONTEXT_TOKENS')
    assert Config.load().worker.num_ctx == 86016


@pytest.mark.parametrize('value', ['garbage', '3.5', '0', '1023', '1048577'])
def test_invalid_run_context_is_not_silently_ignored(value, monkeypatch):
    monkeypatch.setenv('SPIRAL_REQUIRED_LOCAL_MODEL', 'chosen:27b')
    monkeypatch.setenv('SPIRAL_RUN_CONTEXT_TOKENS', value)
    with pytest.raises(ValueError):
        Config.load()


def test_inference_rejects_wrong_model_before_any_transport_or_budget_access():
    from spiral.llm import Ollama
    # The guard must run before any HTTP or resource admission. A deliberately
    # incomplete client detects accidental movement of that boundary.
    client = object.__new__(Ollama)
    client.providers = {}
    with mock.patch.dict(os.environ, {"SPIRAL_REQUIRED_LOCAL_MODEL": "chosen:27b"}):
        with pytest.raises(ModelBindingError):
            client._chat_impl("wrong:27b", [])
        with pytest.raises(ModelBindingError):
            next(client.chat_stream("wrong:27b", []))


def test_binding_rejects_wrong_model_api_redirection_and_malformed_name():
    with mock.patch.dict(os.environ, {"SPIRAL_REQUIRED_LOCAL_MODEL": "chosen:latest"}):
        require_selected_local_model("registry.ollama.ai/library/chosen:latest", {})
        with pytest.raises(ModelBindingError):
            require_selected_local_model("different", {})
        with pytest.raises(ModelBindingError):
            require_selected_local_model("chosen:latest", {"chosen:latest": {"base_url": "https://example.invalid"}})
    with mock.patch.dict(os.environ, {"SPIRAL_REQUIRED_LOCAL_MODEL": "chosen --other"}):
        with pytest.raises(ModelBindingError):
            selected_local_model()
