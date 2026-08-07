"""The build gate must (a) be re-detected as a greenfield project materialises —
spiral usually starts on an empty repo and creates the project mid-run, so a gate
detected once at construction stays empty forever and every task runs unverified —
and (b) treat pytest's "no tests collected" (exit 5) as green, or an early project
with a pyproject but no tests yet becomes a permanently-red gate that thrashes.

Runs standalone (`python tests/test_gate_detect.py`) or under pytest.
"""
# Run with pytest. There was once a hand-rolled runner below this point, which
# collected globals() from where it sat mid-file, called each test with no
# arguments, and caught only AssertionError. So it silently skipped every test
# defined after it and every test taking a fixture, then printed "N/N passed".
# A runner that reports a pass count over a subset it chose is the vacuous green
# this suite exists to catch.
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral import tools  # noqa: E402
from spiral.conductor import Conductor, detect_gate  # noqa: E402


def _repo() -> Path:
    d = Path(tempfile.mkdtemp())
    subprocess.run("git init -q", shell=True, cwd=d)
    return d


def test_gate_redetected_when_project_materialises():
    d = _repo()
    c = Conductor(d)
    assert c.gate == "" and c._refresh_gate() is False        # empty repo, nothing to detect
    (d / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    assert c._refresh_gate() is True                          # gate appears mid-run
    assert "pytest" in c.gate and "footguns" in c.gate_disp
    assert c._refresh_gate() is False                         # idempotent — no false 'changed'


# These exercise the raw detected gate (``_base_gate``); the footguns half of the
# composed gate needs ``spiral`` importable by sys.executable, which is an orthogonal
# install concern, not what the exit-5 fix is about.
def test_no_tests_collected_is_green():
    d = _repo()
    (d / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    c = Conductor(d)
    r = tools.run(c._base_gate, d)
    assert r.ok, "a pyproject with no tests yet (pytest exit 5) must read green, not red"


def test_real_failure_still_red():
    d = _repo()
    (d / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    (d / "tests").mkdir()
    (d / "tests" / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    c = Conductor(d)
    assert not tools.run(c._base_gate, d).ok                  # exit 1 stays red
    (d / "tests" / "test_bad.py").write_text("def test_ok():\n    assert True\n")
    assert tools.run(c._base_gate, d).ok                      # passing suite is green


def test_tests_dir_alone_triggers_gate():
    d = _repo()
    (d / "tests").mkdir()
    assert "pytest" in detect_gate(d)




def test_composed_gate_is_not_double_parenthesised():
    """A single detected gate + footguns must compose to `(a) && (b)`, never
    `((a)) && (b)` — the double paren is arithmetic syntax and zsh/sh fail it with
    'bad math expression', which broke `spiral build` on every task."""
    d = Path(tempfile.mkdtemp()) / "dir with spaces"
    d.mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>x</title>")
    c = Conductor(workspace=d)
    assert "+footguns" in c.gate_disp
    assert not c.gate.lstrip().startswith("(("), c.gate
    assert "(( " not in c.gate and "((/" not in c.gate and "((python" not in c.gate
    # prove it actually parses (syntax, not exit code) under both shells
    for sh in ("/bin/sh", "/bin/zsh"):
        if not Path(sh).exists():
            continue
        r = subprocess.run([sh, "-c", c.gate], cwd=d, capture_output=True, text=True)
        assert "bad math expression" not in r.stderr, (sh, r.stderr)
        assert "syntax error in expression" not in r.stderr, (sh, r.stderr)


def test_an_unparseable_composed_gate_is_a_named_harness_fault(tmp_path, monkeypatch):
    """`((…))` read as shell arithmetic once made every run abort at bootstrap
    while the loop "fixed" healthy code. A gate that cannot parse must fail at
    composition time, attributed to spiral, never to the project."""
    import subprocess

    import spiral.conductor as conductor_module
    from spiral.conductor import Conductor

    subprocess.run("git init -q .", shell=True, cwd=tmp_path, check=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")

    original = conductor_module._compose_gates

    def sabotage(ws, gates):
        composed = original(ws, gates)
        return f"(({composed}" if composed else composed

    monkeypatch.setattr(conductor_module, "_compose_gates", sabotage)
    try:
        Conductor(tmp_path)
    except RuntimeError as exc:
        assert "spiral bug" in str(exc)
    else:
        raise AssertionError("a malformed gate must not be accepted silently")
