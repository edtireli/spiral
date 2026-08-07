"""Exercises the edit engine against the ways a local model actually errs.
Runs standalone (`python tests/test_edits.py`) or under pytest.
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






def test_an_edit_that_breaks_a_parsing_file_is_rejected_atomically(tmp_path):
    """A fuzzy application landing in the wrong region beheaded a working
    app/database.py; the damage surfaced a full gate run later and the repair
    model flip-flopped between ')' and '}' because it could not see the imbalance.
    The parser can, at application time, before anything lands."""
    from spiral.edits import EditBlock, apply_edits

    target = tmp_path / "logic.py"
    target.write_text("def add(a, b):\n    return a + b\n")
    result = apply_edits(tmp_path, [EditBlock(
        "logic.py", "    return a + b", "    return (a + b\n")])[0]
    assert result.ok is False
    assert "REJECTED" in result.reason and "stop parsing" in result.reason
    assert target.read_text() == "def add(a, b):\n    return a + b\n", (
        "the file on disk must be untouched")

    good = apply_edits(tmp_path, [EditBlock(
        "logic.py", "    return a + b", "    return (a + b)\n")])[0]
    assert good.ok is True


def test_a_broken_file_may_still_be_edited(tmp_path):
    """The rule is monotone: never make a parsing file unparsable — but a file
    that already fails to parse must accept edits, or it could never be fixed."""
    from spiral.edits import EditBlock, apply_edits

    target = tmp_path / "broken.py"
    target.write_text("def f(:\n    pass\n")
    step = apply_edits(tmp_path, [EditBlock(
        "broken.py", "def f(:", "def f(x:")])[0]
    assert step.ok is True, "an intermediate edit on a broken file must land"


def test_inline_html_scripts_are_held_to_the_same_bar(tmp_path):
    import shutil as _sh

    from spiral.edits import EditBlock, apply_edits

    if not _sh.which("node"):
        return
    page = tmp_path / "index.html"
    page.write_text("<button>Go</button><script>const x = 1;</script>\n")
    bad = apply_edits(tmp_path, [EditBlock(
        "index.html", "const x = 1;", "const x = 1;;)\n")])[0]
    assert bad.ok is False and "stop parsing" in bad.reason
    assert "const x = 1;</script>" in page.read_text()


def test_a_new_file_that_does_not_parse_is_never_created(tmp_path):
    from spiral.edits import EditBlock, apply_edits

    result = apply_edits(tmp_path, [EditBlock("fresh.py", "", "def broken(:\n")])[0]
    assert result.ok is False and "would not parse" in result.reason
    assert not (tmp_path / "fresh.py").exists()
