"""Exercises the runtime-footgun linter — patterns that compile fine and crash
at runtime. It is welded into every gate, so a silent regression here would
let those crashes ship again.
Runs standalone (`python tests/test_footguns.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.footguns import main, scan  # noqa: E402

BAD_KT = (
    'val c = Color.parseColor("red")\n'
    "val h = Handler()\n"
    'val e = intent.getSerializableExtra("k")\n'
    "Thread.sleep(1000)\n"
)

CLEAN_KT = (
    'val c = Color.parseColor("#FF0000")\n'
    "val h = Handler(Looper.getMainLooper())\n"
    "handler.postDelayed({ tick() }, 1000)\n"
)


def test_each_pattern_fires():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Bad.kt").write_text(BAD_KT)
        hits = scan(d)
        assert len(hits) == 4, hits
        for want in ("parseColor", "Looper", "getSerializableExtra", "Thread.sleep"):
            assert any(want in h for h in hits), f"no hit mentions {want}"


def test_clean_file_is_silent():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Clean.kt").write_text(CLEAN_KT)
        assert scan(d) == []


def test_output_matches_error_machinery_format():
    """The gate harvests these as compile errors: `error: <file>:<line>: ...`."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Bad.kt").write_text("x\ny\nval h = Handler()\n")
        [hit] = scan(d)
        assert hit.startswith("error: ")
        assert ":3: [footgun]" in hit


def test_only_kotlin_and_java_are_scanned():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "notes.txt").write_text(BAD_KT)
        (Path(d) / "gen.py").write_text(BAD_KT)
        assert scan(d) == []


def test_build_dirs_are_skipped():
    with tempfile.TemporaryDirectory() as d:
        gen = Path(d) / "build" / "Gen.kt"
        gen.parent.mkdir()
        gen.write_text(BAD_KT)
        assert scan(d) == []


def test_main_exit_codes_drive_the_gate():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Clean.kt").write_text(CLEAN_KT)
        argv = sys.argv
        try:
            sys.argv = ["footguns", d]
            assert main() == 0
            (Path(d) / "Bad.kt").write_text(BAD_KT)
            assert main() == 1
        finally:
            sys.argv = argv


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
