"""Edit-interface robustness — the shapes models actually emit.

Written against a reproduced failure: 11 of 13 realistic reply shapes from a real
local-model build produced ZERO edit blocks, so the worker had no way to change a file
and burned its whole budget. These tests pin the tolerant parser AND the safety guards
that keep tolerance from becoming recklessness.
"""
import tempfile
from pathlib import Path

import pytest

from spiral.edits import EditBlock, apply_edits, parse_any, parse_edits, _plausible_path

S, D, T = "<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"
BODY = f"{S}\nold line\n{D}\nnew line\n{T}"


@pytest.mark.parametrize("name,text,fmt", [
    ("bare path",           f"src/app.py\n{BODY}", "sr"),
    ("bold path",           f"**src/app.py**\n{BODY}", "sr"),
    ("file label",          f"File: src/app.py\n{BODY}", "sr"),
    ("trailing colon",      f"src/app.py:\n{BODY}", "sr"),
    ("markdown heading",    f"### src/app.py\n{BODY}", "sr"),
    ("dashed header",       f"--- src/app.py ---\n{BODY}", "sr"),
    ("fence lang:path",     f"```python:src/app.py\n{BODY}\n```", "sr"),
    ("prose lead-in",       f"I'll update the file src/app.py now.\n{BODY}", "sr"),
    ("labelled divider",    f"src/app.py\n{S}\nold\n======= REPLACE\nnew\n{T}", "sr"),
    ("short markers",       "src/app.py\n<<<<< SEARCH\nold\n=====\nnew\n>>>>> REPLACE", "sr"),
    ("think wrapper",       f"<think>plan</think>\nsrc/app.py\n{BODY}", "sr"),
    ("unified diff",        "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-old line\n+new line", "udiff"),
    ("whole file fenced",   "src/app.py\n```python\nprint('hi')\n```", "whole"),
])
def test_realistic_reply_shapes_parse(name, text, fmt):
    out = parse_any(text)
    assert out.blocks, f"{name}: produced no blocks"
    assert out.blocks[0].path == "src/app.py", f"{name}: wrong path {out.blocks[0].path!r}"
    assert out.fmt == fmt, f"{name}: expected {fmt}, got {out.fmt}"


def test_consecutive_blocks_inherit_one_header():
    """A second block under a single path header used to be silently dropped."""
    out = parse_any(f"src/app.py\n{BODY}\n{BODY}")
    assert len(out.blocks) == 2 and all(b.path == "src/app.py" for b in out.blocks)


def test_framework_paths_are_editable():
    """Next.js dynamic routes were rejected by the path regex — whole frameworks
    were unbuildable."""
    assert _plausible_path("src/[id]/page.tsx")
    assert _plausible_path("app/(marketing)/layout.tsx")
    out = parse_any(f"src/[id]/page.tsx\n{BODY}")
    assert out.blocks and out.blocks[0].path == "src/[id]/page.tsx"


def test_unparseable_reply_explains_itself():
    """'no edits parsed' teaches a model nothing; the reason must name the problem."""
    out = parse_any(f"{S}\nold\n{D}\nnew\n{T}")          # no path anywhere
    assert not out.blocks and out.dropped
    assert "path" in out.dropped[0].lower()


def test_truncated_block_is_reported_not_silently_dropped():
    out = parse_any(f"src/app.py\n{S}\nold line without a divider")
    assert not out.blocks and out.dropped and "cut off" in out.dropped[0]


def test_parse_edits_wrapper_keeps_list_contract():
    assert isinstance(parse_edits(f"src/app.py\n{BODY}"), list)


# ------------------------------------------------------------------ safety guards

def _root():
    return Path(tempfile.mkdtemp())


def test_ambiguous_search_is_refused_not_applied_to_first_match():
    """Was: applied to the first match and reported success — silently editing the
    wrong function."""
    root = _root()
    (root / "Card.tsx").write_text(
        "function Left(){\n  return <div/>\n}\nfunction Right(){\n  return <div/>\n}\n")
    r = apply_edits(root, [EditBlock("Card.tsx", "  return <div/>", "  return <span/>")])[0]
    assert not r.ok and "ambiguous" in r.reason
    assert (root / "Card.tsx").read_text().count("<div/>") == 2   # untouched


def test_whole_file_cannot_shrink_a_real_file_into_a_stub():
    root = _root()
    (root / "big.py").write_text("\n".join(f"x{i} = {i}" for i in range(40)) + "\n")
    r = apply_edits(root, [EditBlock("big.py", "", "x = 1\n", mode="whole")])[0]
    assert not r.ok and "shrink" in r.reason
    assert len((root / "big.py").read_text().splitlines()) == 40


def test_whole_file_must_parse():
    root = _root()
    (root / "ok.py").write_text("a = 1\n")
    r = apply_edits(root, [EditBlock("ok.py", "", "def broken(:\n", mode="whole")])[0]
    assert not r.ok and "not parse" in r.reason
    assert (root / "ok.py").read_text() == "a = 1\n"


def test_whole_file_creates_and_rewrites_legitimately():
    root = _root()
    r = apply_edits(root, [EditBlock("pkg/mod.py", "", "def f():\n    return 2\n", mode="whole")])[0]
    assert r.ok and (root / "pkg/mod.py").exists()
    big = "\n".join(f"line{i} = {i}" for i in range(30)) + "\n"
    (root / "keep.py").write_text(big)
    grown = big + "extra = 1\n"
    r = apply_edits(root, [EditBlock("keep.py", "", grown, mode="whole")])[0]
    assert r.ok and "extra = 1" in (root / "keep.py").read_text()


def test_format_example_placeholder_is_rejected_in_every_mode():
    """A real run created path/spiral/__init__.py — the model copied the 'path/' prefix
    out of the format example and the guard was too narrow to catch it."""
    root = _root()
    r = apply_edits(root, [EditBlock("path/to/file.ext", "", "x = 1\n", mode="whole")])[0]
    assert not r.ok and "placeholder" in r.reason
