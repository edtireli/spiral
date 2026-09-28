import sys
from types import SimpleNamespace

import pytest

from spiral.execution import BudgetLimits, RunBudget
from spiral.llm import Ollama, OfflineModelAccess


def client(monkeypatch, bridge, backend="owned_llama"):
    monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")
    monkeypatch.setenv("SPIRAL_ENGINE_INFERENCE_BACKEND", backend)
    model = Ollama(providers={})  # Physical no-network transport for this fixture.
    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS")
    monkeypatch.setitem(sys.modules, "spiral_engine_owned", SimpleNamespace(OwnedEngineTransport=bridge))
    model.budget = RunBudget(BudgetLimits(30, 10000, 1))
    return model


@pytest.mark.parametrize("backend", ["owned_llama", "slotstream"])
def test_owned_seam_preserves_budget_selected_request_and_live_callback(monkeypatch, backend):
    seen = []
    class Bridge:
        def __init__(self, client):
            self.client = client
        def chat(self, payload, *, checkpoint, on_delta):
            checkpoint()
            seen.append(payload)
            on_delta("text", "first")
            assert seen[-1] == "first"  # Callback ran inside the transport.
            return {"text": "first", "prompt_tokens": 7, "completion_tokens": 1}
        def close(self):
            seen.append("closed")
    with client(monkeypatch, Bridge, backend) as model:
        result = model.chat("selected:exact", [{"role": "user", "content": "keep constraints"}],
            num_predict=64, num_ctx=8192, fmt={"type": "object"}, stop=["STOP"],
            on_delta=lambda kind, text: seen.append(text))
        assert result.text == "first" and model.budget.calls == 1 and model.budget.total_tokens == 8
        assert seen[0]["model"] == "selected:exact"
        assert seen[0]["format"] == {"type": "object"}
        assert seen[0]["options"]["stop"] == ["STOP"]
    assert seen[-1] == "closed"


def test_slotstream_discovery_uses_verified_catalog_without_ollama_or_gguf(monkeypatch):
    monkeypatch.setenv("SPIRAL_ENGINE_INFERENCE_BACKEND", "slotstream")
    monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")
    monkeypatch.setitem(sys.modules, "spiral_slotstream", SimpleNamespace(
        catalog_entry=lambda: {"name": "qwen3.8-flash-next:4bit"}))
    with Ollama(providers={}) as model:
        assert model.models() == ["qwen3.8-flash-next:4bit"]
        assert model.source_tokenizer("qwen3.8-flash-next:4bit") is None


def test_consumer_stop_is_not_retried_as_a_transport_error(monkeypatch):
    calls = []
    class Bridge:
        def __init__(self, client): pass
        def chat(self, payload, *, checkpoint, on_delta):
            calls.append(payload)
            on_delta("text", "partial")
        def close(self): pass
    def stop(*args):
        raise KeyboardInterrupt()
    with client(monkeypatch, Bridge) as model:
        with pytest.raises(KeyboardInterrupt):
            model.chat("selected:exact", [{"role": "user", "content": "work"}],
                       num_predict=64, num_ctx=8192, on_delta=stop)
        assert len(calls) == 1 and model.budget.calls == 1
        assert model.budget.total_tokens >= 64


def test_offline_and_unsupported_stream_route_cannot_start_shared_inference(monkeypatch):
    monkeypatch.setenv("SPIRAL_ENGINE_INFERENCE_BACKEND", "owned_llama")
    monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")
    with Ollama(providers={}) as model:
        with pytest.raises(OfflineModelAccess):
            model.chat("selected:exact", [{"role": "user", "content": "work"}])
        with pytest.raises(ValueError, match="generator transport"):
            next(model.chat_stream("selected:exact", []))
