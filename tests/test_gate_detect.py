"""The build gate must (a) be re-detected as a greenfield project materialises —
spiral usually starts on an empty repo and creates the project mid-run, so a gate
detected once at construction stays empty forever and every task runs unverified —
and (b) treat pytest's "no tests collected" (exit 5) as green, or an early project
with a pyproject but no tests yet becomes a permanently-red gate that thrashes.

Runs standalone (`python tests/test_gate_detect.py`) or under pytest.
"""
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


if __name__ == "__main__":
    for fn in (test_gate_redetected_when_project_materialises, test_no_tests_collected_is_green,
               test_real_failure_still_red, test_tests_dir_alone_triggers_gate):
        fn()
        print(f"ok  {fn.__name__}")
    print("all gate-detection tests passed")
