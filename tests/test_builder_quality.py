from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _finished_product(root: Path) -> None:
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text(
        "def run(value: int) -> int:\n    if value < 0:\n        raise ValueError('positive required')\n    return value + 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "from src.app import run\n\ndef test_run():\n    assert run(1) == 2\n")
    (root / "README.md").write_text(
        "# Complete tool\n\n## Setup\nInstall Python 3.11.\n\n## Run\n"
        "Use `python -m src.app`.\n\n## Test\nUse `python -m pytest -q`.\n\n"
        "## Build\nPackage from a clean checkout using the declared project metadata.\n")


def test_product_audit_accepts_substantive_small_product(tmp_path):
    from spiral.product_audit import audit_product

    _finished_product(tmp_path)
    report = audit_product(tmp_path, "Build a command-line application", "other")

    assert report["applicable"] is True
    assert report["issues"] == []


def test_product_audit_rejects_placeholder_and_weak_simulation(tmp_path):
    from spiral.product_audit import audit_product

    _finished_product(tmp_path)
    (tmp_path / "src" / "app.py").write_text(
        "import random\n\ndef simulate():\n    # TODO: replace dummy data\n    return random.random()\n")
    report = audit_product(tmp_path, "Build a numerical simulation program", "other")
    ids = {issue["id"] for issue in report["issues"]}

    assert {"product-placeholders", "simulation-seed", "simulation-reference-check"} <= ids


def test_node_dependency_sync_disables_hooks_and_scrubs_credentials(tmp_path, monkeypatch):
    from spiral import builder_tools

    (tmp_path / "package.json").write_text(json.dumps({
        "name": "demo", "dependencies": {"lucide": "1.0.0"},
    }))
    calls = []

    def fake_which(name):
        return "/usr/local/bin/npm" if name == "npm" else None

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        (tmp_path / "node_modules").mkdir(exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr(builder_tools.shutil, "which", fake_which)
    monkeypatch.setattr(builder_tools.subprocess, "run", fake_run)
    monkeypatch.setenv("MOONSHOT_API_KEY", "must-not-leak")

    result = builder_tools.ensure_node_dependencies(tmp_path)

    assert result["ok"] and result["changed"]
    argv, kwargs = calls[0]
    assert "--ignore-scripts" in argv
    assert "MOONSHOT_API_KEY" not in kwargs["env"]
    assert json.loads((tmp_path / ".spiral/dependency-cache/state.json").read_text())["ok"]


def test_python_requirement_reader_rejects_direct_urls(tmp_path):
    from spiral.builder_tools import _python_requirements

    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\ndependencies=['numpy>=2']\n"
        "[project.optional-dependencies]\ntest=['pytest>=8']\n")
    (tmp_path / "requirements-extra.txt").write_text(
        "requests>=2\nunsafe @ https://example.test/pkg.whl\n")

    requirements, inputs, errors = _python_requirements(tmp_path)

    assert {"numpy>=2", "pytest>=8", "requests>=2"} <= set(requirements)
    assert inputs and any("manual review" in error for error in errors)


def test_python_requirement_reader_names_merge_conflict(tmp_path):
    from spiral.builder_tools import _python_requirements

    (tmp_path / "requirements.txt").write_text("numpy>=2\n=======\n")

    _, _, errors = _python_requirements(tmp_path)

    assert any("merge-conflict marker" in error for error in errors)


def test_public_repo_acquisition_rejects_non_github_url(tmp_path):
    from spiral.builder_tools import acquire_public_repo

    result = acquire_public_repo("https://example.test/owner/repo", tmp_path)

    assert "rejected" in result
    assert not (tmp_path / ".spiral" / "tools" / "owner-repo").exists()


def test_build_cli_reaches_config_without_local_scope_crash(tmp_path, monkeypatch):
    from spiral import cli, conductor

    seen = {}

    class FakeConductor:
        def __init__(self, workspace, cfg):
            seen["workspace"] = workspace

        def build(self, goal, resume=False, approve=False):
            seen["goal"] = goal

    monkeypatch.setattr(conductor, "Conductor", FakeConductor)
    monkeypatch.setattr(cli, "print_banner", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_info_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["spiral", "build", "make a tool", "--dir", str(tmp_path)])

    cli.main()

    assert seen == {"workspace": str(tmp_path), "goal": "make a tool"}


def test_detect_gate_covers_full_node_and_native_checks(tmp_path):
    from spiral.conductor import detect_gate

    (tmp_path / "package.json").write_text(json.dumps({
        "scripts": {"test": "vitest run", "build": "vite build", "lint": "eslint ."},
    }))
    node_gate = detect_gate(tmp_path)
    assert "npm run test" in node_gate and "npm run build" in node_gate and "npm run lint" in node_gate

    (tmp_path / "package.json").unlink()
    (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n")
    assert "ctest" in detect_gate(tmp_path)


def test_detect_gate_and_dependencies_follow_nested_product(tmp_path, monkeypatch):
    from spiral import builder_tools
    from spiral.conductor import detect_gate

    (tmp_path / "requirements.txt").write_text("=======\n")
    product = tmp_path / "project"
    product.mkdir()
    (product / "Makefile").write_text("test:\n\t@echo ok\n")
    (product / "tests").mkdir()
    seen = []
    monkeypatch.setattr(
        builder_tools, "ensure_node_dependencies",
        lambda root, **kwargs: seen.append(Path(root)) or {"applicable": False, "ok": True},
    )
    monkeypatch.setattr(
        builder_tools, "ensure_python_dependencies",
        lambda root, **kwargs: seen.append(Path(root)) or {"applicable": False, "ok": True},
    )

    gate = detect_gate(tmp_path)
    result = builder_tools.ensure_builder_dependencies(tmp_path)

    assert "cd project &&" in gate and "make test" in gate
    assert seen == [product, product]
    assert result["project_root"] == str(product)


def test_advertisement_goal_enters_visual_pipeline(tmp_path):
    from spiral.conductor import Conductor

    conductor = object.__new__(Conductor)
    conductor.ws = tmp_path

    assert conductor._project_kind(
        "Create a polished product advertisement with generated assets") == "image"


def test_validator_outage_retains_prior_verdict_without_inventing_gap(
        tmp_path, monkeypatch):
    from spiral import conductor as conductor_module
    from spiral.conductor import Conductor

    runner = Conductor(tmp_path)
    spec = [{"id": "R1", "text": "Render the finished product", "kind": "feature"}]
    (tmp_path / ".spiral").mkdir(exist_ok=True)
    (tmp_path / ".spiral" / "validation.json").write_text(json.dumps([{
        "id": "R1", "status": "implemented", "evidence": "visible in app", "fresh": True,
    }]))
    monkeypatch.setattr(runner, "_load_spec", lambda goal: spec)
    monkeypatch.setattr(conductor_module, "build_repomap", lambda *args, **kwargs: "repo")
    monkeypatch.setattr(
        conductor_module, "validate_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    verdicts = runner.validate_only("build it")

    assert verdicts[0]["status"] == "implemented"
    assert verdicts[0]["fresh"] is False
    assert "retained prior" in verdicts[0]["evidence"]
