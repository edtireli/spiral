"""Tests default to offline inference and a private empirical tool registry."""
import os
import pytest

from spiral.toolsmith import Toolsmith


@pytest.fixture(autouse=True)
def offline_models_unless_explicitly_requested(monkeypatch):
    # A bare pytest command must not load/evict models or compete with real work.
    # Individual transport tests may still explicitly replace this inside their
    # own mocked boundary; real-model tests require the existing explicit opt-in.
    if os.environ.get("SPIRAL_LIVE_OLLAMA_TESTS") != "1":
        monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")


@pytest.fixture(autouse=True)
def isolated_tool_registry(tmp_path, monkeypatch):
    original = Toolsmith.__init__

    def isolated(self, workspace=None, *, registry_path=None):
        original(self, workspace, registry_path=(
            registry_path if registry_path is not None else tmp_path / "toolsmith.json"))

    monkeypatch.setattr(Toolsmith, "__init__", isolated)
