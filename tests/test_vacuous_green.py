"""Green must not be vacuous — a test suite nothing runs is not evidence.

Grounded in an observed real build: spiral generated `test/calc.test.js` containing a
self-contradicting assertion (asserting a call equals BOTH -40 and -60), committed it,
and declared the project complete. The project had no manifest, so no test runner was
ever detected, and the artifact gate only checks that the file *parses* — which a broken
assertion does perfectly.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from spiral.conductor import _orphan_test_gate, detect_gate, detect_gates

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def _ws() -> Path:
    return Path(tempfile.mkdtemp())


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


PASSING = """
const { test } = require('node:test');
const assert = require('node:assert/strict');
test('adds', () => { assert.strictEqual(1 + 1, 2); });
"""

# the actual defect shape that shipped: two contradictory assertions about one call
CONTRADICTORY = """
const { test } = require('node:test');
const assert = require('node:assert/strict');
const total = (b, p) => b + b * (p / 100);
test('negative bill', () => {
  assert.strictEqual(total(-50, 20), -40);
  assert.strictEqual(total(-50, 20), -60);
});
"""


def test_orphan_test_suite_becomes_a_gate():
    ws = _ws()
    _write(ws, "index.html", "<!doctype html><title>x</title>")
    _write(ws, "test/calc.test.js", PASSING)
    ecosystems = {g.ecosystem for g in detect_gates(ws)}
    assert "tests" in ecosystems, "a test suite with no runner must become a gate"
    assert "artifact-integrity" in ecosystems, "structural check must still apply"


def test_a_failing_orphan_suite_makes_the_gate_red():
    """The property that was missing: a suite that fails is a RED gate, not an absent
    one. Without this the loop commits broken tests as done."""
    ws = _ws()
    _write(ws, "index.html", "<!doctype html><title>x</title>")
    _write(ws, "test/calc.test.js", CONTRADICTORY)
    gate = detect_gate(ws)
    assert gate, "no gate composed"
    r = subprocess.run(["/bin/sh", "-c", gate], cwd=ws, capture_output=True, text=True)
    assert r.returncode != 0, "a self-contradicting test suite still passed the gate"


def test_a_passing_orphan_suite_is_green():
    """The tests rung alone — the artifact rung shells out to sys.executable, which
    under pytest is whichever interpreter runs the suite and need not have spiral
    importable. Isolating the rung keeps this test about the property it names."""
    ws = _ws()
    _write(ws, "index.html", "<!doctype html><title>x</title>")
    _write(ws, "test/calc.test.js", PASSING)
    gate = _orphan_test_gate(ws)
    r = subprocess.run(["/bin/sh", "-c", gate], cwd=ws, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")


def test_no_test_files_invents_no_test_gate():
    """A gate that cannot fail is not a check — but neither should spiral fabricate a
    test gate where the project has no tests."""
    ws = _ws()
    _write(ws, "index.html", "<!doctype html><title>x</title>")
    assert _orphan_test_gate(ws) == ""
    assert "tests" not in {g.ecosystem for g in detect_gates(ws)}


def test_manifest_that_already_runs_tests_is_not_duplicated():
    ws = _ws()
    _write(ws, "package.json",
           '{"name":"x","version":"1.0.0","scripts":{"test":"node --test test/"}}')
    _write(ws, "test/calc.test.js", PASSING)
    ecosystems = [g.ecosystem for g in detect_gates(ws)]
    assert ecosystems.count("tests") == 0, "npm test already runs the suite"


def test_detection_prunes_heavy_directories():
    """node_modules must be pruned, not walked — detection runs on every gate refresh."""
    ws = _ws()
    _write(ws, "test/calc.test.js", PASSING)
    for i in range(30):
        _write(ws, f"node_modules/pkg{i}/index.test.js", "throw new Error('never run')")
    gate = _orphan_test_gate(ws)
    assert "node_modules" not in gate
    assert "test/calc.test.js" in gate
