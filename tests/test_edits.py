"""Exercises the edit engine against the ways a local model actually errs.
Runs standalone (`python tests/test_edits.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.edits import apply_edits, parse_edits, EditBlock  # noqa: E402


def test_exact():
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "m.py"
        fp.write_text("def f():\n    return 1\n")
        [r] = apply_edits(d, [EditBlock("m.py", "    return 1", "    return 2")])
        assert r.ok and r.how == "exact", r
        assert fp.read_text() == "def f():\n    return 2\n"


def test_elastic_wrong_indentation():
    # model returns the right lines but with 2-space indent; file uses 4.
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "m.py"
        fp.write_text("def f():\n    x = 1\n    return x\n")
        search = "  x = 1\n  return x"
        replace = "  x = 2\n  return x + 1"
        [r] = apply_edits(d, [EditBlock("m.py", search, replace)])
        assert r.ok and r.how == "elastic", r
        out = fp.read_text()
        assert "    x = 2" in out and "    return x + 1" in out, out  # reindented to 4


def test_fuzzy_typo_in_context():
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "m.py"
        fp.write_text("def score(items):\n    # compute the total score for all items\n    return sum(items)\n")
        search = "def score(items):\n    # compute total score for all items\n    return sum(items)"
        replace = "def score(items):\n    # compute the total score\n    return sum(i for i in items)"
        [r] = apply_edits(d, [EditBlock("m.py", search, replace)])
        assert r.ok and r.how == "fuzzy", r
        assert "sum(i for i in items)" in fp.read_text()


def test_create_new_file():
    with tempfile.TemporaryDirectory() as d:
        [r] = apply_edits(d, [EditBlock("pkg/new.py", "", "print('hi')\n")])
        assert r.ok and r.how == "created", r
        assert (Path(d) / "pkg" / "new.py").read_text() == "print('hi')\n"


def test_miss_reports_failure():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "m.py").write_text("a = 1\n")
        [r] = apply_edits(d, [EditBlock("m.py", "totally not here\nnope", "x")])
        assert not r.ok and "not found" in r.reason, r


def test_parse_fenced_multiblock():
    blob = (
        "Here you go.\n\n"
        "`math_utils.py`\n"
        "```python\n"
        "<<<<<<< SEARCH\n"
        "def add(a, b):\n    return a + b\n"
        "=======\n"
        "def add(a, b):\n    return a + b + 0\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    blocks = parse_edits(blob)
    assert len(blocks) == 1, blocks
    assert blocks[0].path == "math_utils.py", blocks[0].path
    assert "return a + b + 0" in blocks[0].replace


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  \033[32mPASS\033[0m {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  \033[31mFAIL\033[0m {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
