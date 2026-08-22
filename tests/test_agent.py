"""Exercises the atom's pure pieces — the diversity round's candidate
fingerprint, which decides whether the gate re-judges a sampled edit set.
Runs standalone (`python tests/test_agent.py`) or under pytest.
"""
# Run with pytest. There was once a hand-rolled runner below this point, which
# collected globals() from where it sat mid-file, called each test with no
# arguments, and caught only AssertionError. So it silently skipped every test
# defined after it and every test taking a fixture, then printed "N/N passed".
# A runner that reports a pass count over a subset it chose is the vacuous green
# this suite exists to catch.
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.agent import SYSTEM, Atom, TaskSpec, _blocks_key  # noqa: E402
from spiral.edits import EditBlock  # noqa: E402


def test_identical_edit_sets_share_a_key():
    a = [EditBlock("m.py", "x = 1", "x = 2")]
    b = [EditBlock("m.py", "x = 1", "x = 2")]
    assert _blocks_key(a) == _blocks_key(b)


def test_surrounding_whitespace_does_not_split_keys():
    a = [EditBlock("m.py", "  x = 1\n", "x = 2")]
    b = [EditBlock("m.py", "x = 1", "  x = 2  ")]
    assert _blocks_key(a) == _blocks_key(b)


def test_different_content_or_path_differs():
    a = [EditBlock("m.py", "x = 1", "x = 2")]
    b = [EditBlock("m.py", "x = 1", "x = 3")]
    c = [EditBlock("n.py", "x = 1", "x = 2")]
    assert len({_blocks_key(a), _blocks_key(b), _blocks_key(c)}) == 3


def test_block_order_matters():
    one = EditBlock("m.py", "a", "b")
    two = EditBlock("m.py", "c", "d")
    assert _blocks_key([one, two]) != _blocks_key([two, one])


def test_worker_protocol_allows_web_ask():
    assert "ASK: web <focused search query>" in SYSTEM
    assert "ASK: repo <public GitHub URL>" in SYSTEM


def test_worker_runtime_prompt_marks_repository_content_as_untrusted(tmp_path):
    atom = Atom(tmp_path)
    prompt = atom._worker_system()
    assert "FILES" in prompt
    assert "untrusted data" in prompt
    assert "WORKSPACE" in prompt


def test_full_access_worker_receives_the_real_runtime_contract(tmp_path):
    from spiral.config import Config

    cfg = Config()
    cfg.builder_full_access = True
    prompt = Atom(tmp_path, cfg)._worker_system()
    assert "FULL ACCESS" in prompt
    assert "unsandboxed" in prompt
    assert "network access" in prompt
    assert "Do not inspect credentials" in prompt


def test_worker_prompt_lists_exact_references_as_untrusted_read_only_data(tmp_path):
    from spiral.config import Config

    reference = tmp_path.parent / f"{tmp_path.name}-reference.pdf"
    reference.write_bytes(b"%PDF")
    cfg = Config()
    cfg.builder_reference_roots = [str(reference.resolve())]

    prompt = Atom(tmp_path, cfg)._worker_system()

    assert str(reference.resolve()) in prompt
    assert "APPROVED READ-ONLY REFERENCES" in prompt
    assert "untrusted data" in prompt
    assert "Never edit, rename, delete, execute" in prompt


def test_web_research_fetches_and_persists():
    from spiral import research

    orig_search, orig_fetch = research.search, research.fetch
    try:
        research.search = lambda q, k=5: [
            research.Hit("Library migration guide", "https://example.test/guide", "upgrade snippet")
        ]
        research.fetch = lambda url: "Use the new frobnicate(options={}) API and update imports."
        d = Path(tempfile.mkdtemp())
        atom = Atom(d)
        txt = atom._web_research(
            "frobnicate TypeError official docs",
            task=TaskSpec("fix frobnicate", "python -m pytest -q"),
            verify_out="TypeError: frobnicate() got an unexpected keyword",
        )
        assert "WEB RESEARCH" in txt and "frobnicate(options={})" in txt
        saved = list((d / ".spiral" / "research").glob("*.md"))
        assert saved and "Library migration guide" in saved[0].read_text()
    finally:
        research.search, research.fetch = orig_search, orig_fetch






def test_gate_verdicts_are_memoized_on_tree_content(tmp_path):
    """Every audit attempt that landed no edit re-ran the full ladder against an
    unchanged tree — paying a subprocess for what a hash already knew."""
    import subprocess

    from spiral.agent import Atom
    from spiral.config import Config

    subprocess.run("git init -q .", shell=True, cwd=tmp_path, check=True)
    (tmp_path / "thing.py").write_text("X = 1\n")
    atom = Atom(tmp_path, Config())

    class Quiet:
        def print(self, *a, **k): pass
        def __getattr__(self, _): return lambda *a, **k: None

    first = atom._run_gate("echo once && true", Quiet())
    assert first.code == 0
    marker = atom._gate_memo
    second = atom._run_gate("echo once && true", Quiet())
    assert second.code == 0
    assert atom._gate_memo is marker, "unchanged tree must reuse the verdict"

    (tmp_path / "thing.py").write_text("X = 2\n")
    atom._run_gate("echo once && true", Quiet())
    assert atom._gate_memo is not marker, "a changed tree must re-run the gate"


def test_a_changed_command_also_busts_the_memo(tmp_path):
    import subprocess

    from spiral.agent import Atom
    from spiral.config import Config

    subprocess.run("git init -q .", shell=True, cwd=tmp_path, check=True)
    atom = Atom(tmp_path, Config())

    class Quiet:
        def print(self, *a, **k): pass
        def __getattr__(self, _): return lambda *a, **k: None

    ok = atom._run_gate("true", Quiet())
    assert ok.code == 0
    bad = atom._run_gate("false", Quiet())
    assert bad.code != 0, "a different command must not reuse the old verdict"
