"""The selected verifier must exist before spending a model's edit budget."""
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral import builder_tools
from spiral.harness_check import HarnessFault, require_verifier_started
from spiral.ladder import python_ladder


def fake_installer(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:3] == ["-m", "venv"]:
            python = Path(argv[3]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(builder_tools.subprocess, "run", run)
    return calls


def test_stdlib_ladder_provisions_its_runner_without_changing_project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="demo"\nversion="1"\n')
    (tmp_path / "test_demo.py").write_text("def test_fail():\n    assert False\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    command = python_ladder(tmp_path)
    calls = fake_installer(monkeypatch)
    report = builder_tools.ensure_builder_dependencies(tmp_path, verification_command=command)
    assert report["ok"] and report["changed"]
    install = next(argv for argv, _ in calls if "install" in argv)
    assert "--only-binary=:all:" in install and install[-1] == "pytest"
    assert str(tmp_path / ".spiral/dependency-cache/python/venv/bin/python") == install[0]
    assert {name: (tmp_path / name).read_bytes() for name in before} == before
    assert not (tmp_path / "requirements.txt").exists()
    state = json.loads((tmp_path / ".spiral/dependency-cache/python/state.json").read_text())
    assert state["verification_requirements"] == ["pytest"]
    assert state["source_builds"] == "binary-wheels-only"
    count = len(calls)
    again = builder_tools.ensure_builder_dependencies(tmp_path, verification_command=command)
    assert again["ok"] and not again["changed"] and len(calls) == count


def test_project_version_constraint_is_preserved_with_harness_requirement(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("pytest==8.3.5\n")
    calls = fake_installer(monkeypatch)
    report = builder_tools.ensure_python_dependencies(tmp_path, verification_requirements=("pytest",))
    assert report["ok"]
    install = next(argv for argv, _ in calls if "install" in argv)
    assert install[-2:] == ["pytest==8.3.5", "pytest"]


def test_no_selected_ladder_does_not_acquire_test_runner(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="demo"\n')
    python_ladder(tmp_path)  # Old materialized files alone do not grant a request.
    calls = fake_installer(monkeypatch)
    result = builder_tools.ensure_builder_dependencies(tmp_path, verification_command="true")
    assert result["ok"] and not result["applicable"] and calls == []


def test_harness_dependency_cannot_be_an_arbitrary_model_package(tmp_path, monkeypatch):
    calls = fake_installer(monkeypatch)
    with pytest.raises(ValueError, match="unsupported verification"):
        builder_tools.ensure_python_dependencies(tmp_path, verification_requirements=("arbitrary",))
    assert calls == []


def test_absent_runner_fails_closed_even_without_project_tests(tmp_path):
    (tmp_path / "example.py").write_text("VALUE = 1\n")
    command = python_ladder(tmp_path).replace("python ", shlex.quote(sys.executable) + " -S ")
    result = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "SPIRAL_VERIFIER_UNAVAILABLE: pytest is absent from" in output
    with pytest.raises(HarnessFault, match="no source-edit attempt"):
        require_verifier_started(output, result.returncode)


def test_actual_assertion_failure_remains_a_code_failure(tmp_path):
    (tmp_path / "test_demo.py").write_text("def test_fail():\n    assert 1 == 2\n")
    command = python_ladder(tmp_path).replace("python ", shlex.quote(sys.executable) + " ")
    result = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "1 failed" in result.stdout
    require_verifier_started(result.stdout + result.stderr, result.returncode)


@pytest.mark.parametrize("entry", ["worker", "conductor"])
def test_gate_startup_failure_never_returns_an_edit_verdict(tmp_path, monkeypatch, entry):
    from spiral.agent import Atom
    from spiral.conductor import Conductor
    from spiral.tools import RunResult

    monkeypatch.setattr(builder_tools, "ensure_builder_dependencies", lambda *a, **k: {"ok": True})
    runner = object.__new__(Atom if entry == "worker" else Conductor)
    runner.ws = tmp_path
    runner.cfg = SimpleNamespace(verify_timeout=5)
    runner.command_broker = SimpleNamespace(environment={}, run=lambda *a, **k: SimpleNamespace(
        result=RunResult("verifier", 1, "SPIRAL_VERIFIER_UNAVAILABLE: test instrument absent")))
    runner._tree_hash = lambda: "tree"
    runner._effective_gate_command = lambda c: c
    ui = SimpleNamespace(print=lambda *a: None, detail=lambda *a: None)
    with pytest.raises(HarnessFault, match="test instrument absent"):
        if entry == "worker":
            runner._run_gate("gate", ui)
        else:
            runner._run_verified_command("gate")
    assert not getattr(runner, "_gate_memo", None)
