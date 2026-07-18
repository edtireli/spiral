"""Exercises the atom's pure pieces — the diversity round's candidate
fingerprint, which decides whether the gate re-judges a sampled edit set.
Runs standalone (`python tests/test_agent.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.agent import _blocks_key  # noqa: E402
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


def _run() -> int:
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
