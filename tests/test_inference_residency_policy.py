"""Request-scoped residency, never an out-of-band shared-model eviction."""
import pytest

from spiral.llm import Ollama


def payload(client, keep_alive):
    return client._payload("selected:27b", [{"role": "user", "content": "fixture"}],
                           think=False, num_predict=8, temperature=0, stop=None,
                           fmt=None, keep_alive=keep_alive)


@pytest.mark.parametrize("requested", [None, "45m", 1200, 0])
def test_owned_parent_handoff_releases_only_its_childs_request(monkeypatch, tmp_path, requested):
    monkeypatch.setenv("SPIRAL_MODEL_LEASE_PATH", str(tmp_path / "compute.lease"))
    monkeypatch.setenv("SPIRAL_MODEL_RESIDENCY_POLICY", "release_after_request")
    client = Ollama(providers={})
    try:
        result = payload(client, requested)
        assert result["keep_alive"] == 0
        assert result["model"] == "selected:27b"
        assert result["messages"] == [{"role": "user", "content": "fixture"}]
    finally:
        client._client.close()


def test_ordinary_ollama_retention_is_unchanged(monkeypatch):
    monkeypatch.delenv("SPIRAL_MODEL_RESIDENCY_POLICY", raising=False)
    client = Ollama(providers={})
    try:
        assert payload(client, "45m")["keep_alive"] == "45m"
        assert "keep_alive" not in payload(client, None)
    finally:
        client._client.close()


@pytest.mark.parametrize("policy", ["unknown", "", "release_after_request"])
def test_invalid_or_unleased_release_policy_is_rejected(monkeypatch, policy):
    monkeypatch.delenv("SPIRAL_MODEL_LEASE_PATH", raising=False)
    monkeypatch.setenv("SPIRAL_MODEL_RESIDENCY_POLICY", policy)
    with pytest.raises(ValueError, match="residency"):
        client = Ollama(providers={})
        client._client.close()
