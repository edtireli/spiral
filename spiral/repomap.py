"""Compact repo snapshot — the context spiral hands its conductor.

A file tree plus truncated contents of the small text files, skipping junk and
hidden dirs (.git, .spiral, .aider*, node_modules, build output). This is how the
planner learns what already exists before it decomposes the goal.
"""
from __future__ import annotations

from pathlib import Path

from spiral import tools

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".txt", ".md", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".c", ".h", ".cpp", ".go", ".rs", ".java", ".kt", ".kts", ".gradle",
    ".xml", ".html", ".css", ".yaml", ".yml", ".properties", ".pro",
}


def _skip(rel: Path) -> bool:
    return any(part in tools._SKIP_DIRS or part.startswith(".") for part in rel.parts)


def list_files(root: str | Path) -> list[str]:
    """Relative paths of all non-junk files — the ground-truth file set for plan lint."""
    root = Path(root)
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not _skip(p.relative_to(root))
    )


def build_repomap(root: str | Path, max_file_bytes: int = 1800, max_total: int = 22_000) -> str:
    root = Path(root)
    rels = sorted(
        p.relative_to(root) for p in root.rglob("*") if p.is_file() and not _skip(p.relative_to(root))
    )

    out = ["# REPO TREE", *[str(r) for r in rels], "", "# FILE CONTENTS (truncated)"]
    total = 0
    for rel in rels:
        p = root / rel
        if p.suffix.lower() not in TEXT_EXT:
            continue
        if total >= max_total:
            out.append(f"\n--- {rel} --- (omitted — context budget reached)")
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        snippet = text[:max_file_bytes]
        total += len(snippet)
        trailer = "\n…(truncated)" if len(text) > max_file_bytes else ""
        out.append(f"\n--- {rel} ({len(text)} bytes) ---\n{snippet}{trailer}")
    return "\n".join(out)
