"""Deterministic runtime-footgun linter — crash patterns the compiler can't see.

The build gate proves code compiles; these patterns compile FINE and crash at
runtime (the eye view shipped two of them). Zero tokens, instant, composable
into the gate:  ... && python -m spiral.footguns <dir>

Output lines are formatted so the existing error machinery (harvest, sigs,
gate-says) picks them up unchanged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHECKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'parseColor\(\s*"(?!#)'),
     "Color.parseColor without '#' — IllegalArgumentException at runtime"),
    (re.compile(r"\bHandler\(\s*\)"),
     "Handler() without a Looper — deprecated, crashes on threads without one"),
    (re.compile(r"\.getSerializableExtra\("),
     "getSerializableExtra is deprecated/unsafe — prefer typed extras"),
    (re.compile(r"Thread\.sleep\("),
     "Thread.sleep on what may be the main thread — ANR risk; use postDelayed"),
]

SCAN_EXT = {".kt", ".java"}


def scan(root: str | Path) -> list[str]:
    root = Path(root)
    hits: list[str] = []
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for f in files:
        if f.suffix not in SCAN_EXT or not f.is_file() or set(f.parts) & {"build", ".git", ".venv", "venv", "node_modules"}:
            continue
        try:
            lines = f.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            for rx, msg in CHECKS:
                if rx.search(line):
                    hits.append(f"error: {f}:{i}: [footgun] {msg}")
    return hits


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    hits = scan(root)
    for h in hits:
        print(h)
    if hits:
        print(f"FOOTGUNS: {len(hits)} runtime-crash pattern(s) — fix them like compile errors")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
