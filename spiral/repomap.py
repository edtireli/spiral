"""Compact repo snapshot — the context spiral hands its conductor.

A file tree plus truncated contents of the small text files, skipping junk and
hidden dirs (.git, .spiral, .aider*, node_modules, build output). This is how the
planner learns what already exists before it decomposes the goal.
"""
from __future__ import annotations

import re
from pathlib import Path

from spiral import tools

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".txt", ".md", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".c", ".h", ".cpp", ".go", ".rs", ".java", ".kt", ".kts", ".gradle",
    ".xml", ".html", ".css", ".yaml", ".yml", ".properties", ".pro", ".mjs",
    ".cjs", ".vue", ".svelte", ".swift", ".cs", ".fs", ".fsx", ".rb", ".php",
    ".scala", ".dart", ".lua", ".r", ".jl", ".ex", ".exs", ".sol", ".tex",
    ".lean", ".sql", ".graphql", ".gql", ".proto", ".dockerfile", ".csv", ".tsv",
}
_TEXT_NAMES = {
    "Dockerfile", "Makefile", "Procfile", "Gemfile", "Rakefile", "Justfile",
    "CMakeLists.txt", "LICENSE", "README", "CHANGELOG",
}
_CONTEXT_NAMES = {
    "package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml",
    "Package.swift", "CMakeLists.txt", "Makefile", "Dockerfile", "README.md",
    "README.rst", "requirements.txt", "build.gradle", "settings.gradle",
}
_QUERY_STOP = {
    "about", "after", "again", "against", "also", "been", "before", "being",
    "between", "build", "could", "every", "from", "have", "into", "make",
    "must", "only", "project", "should", "that", "their", "then", "there",
    "these", "this", "through", "using", "what", "when", "where", "which",
    "with", "without", "would",
}


def _skip(rel: Path) -> bool:
    return any(part in tools._SKIP_DIRS or part.startswith(".") for part in rel.parts)


def is_text_file(path: str | Path, *, max_bytes: int | None = None) -> bool:
    """Recognize source/config text without relying on a closed extension list."""

    path = Path(path)
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return False
        if path.suffix.lower() in TEXT_EXT or path.name in _TEXT_NAMES:
            return True
        if path.suffix:
            return False
        sample = path.read_bytes()[:4096]
        if b"\x00" in sample:
            return False
        sample.decode("utf-8")
        return bool(sample.strip())
    except (OSError, UnicodeDecodeError):
        return False


def text_files(root: str | Path, *, max_bytes: int | None = None) -> list[Path]:
    root = Path(root)
    return [
        path for path in sorted(root.rglob("*"))
        if path.is_file()
        and not _skip(path.relative_to(root))
        and is_text_file(path, max_bytes=max_bytes)
    ]


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
        if not is_text_file(p):
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


def build_relevant_repomap(
    root: str | Path, queries: list[dict] | list[str] | str,
    *, max_file_bytes: int = 24_000, max_total: int = 120_000,
) -> tuple[str, list[str]]:
    """Build requirement-conditioned evidence context and report selected files.

    The final validator receives different code for each requirement batch. Files
    are ranked by exact declared hints, path terms, content terms, tests, and project
    metadata; a single alphabetically-truncated global dump can no longer hide the
    implementation while inviting a false "missing" verdict.
    """

    root = Path(root).resolve()
    rows = queries if isinstance(queries, list) else [queries]
    texts: list[str] = []
    hinted: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            texts.extend(str(row.get(key) or "") for key in (
                "id", "text", "evidence", "description",
            ))
            fix = row.get("fix") or {}
            if isinstance(fix, dict):
                texts.extend(str(fix.get(key) or "") for key in (
                    "title", "description",
                ))
                hinted.update(str(value) for value in (fix.get("files") or []))
            hinted.update(str(value) for value in (row.get("files") or []))
        else:
            texts.append(str(row))
    query = " ".join(texts).lower()
    terms = {
        token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_-]{2,}", query)
        if token not in _QUERY_STOP
    }
    candidates: list[tuple[int, str, Path, str]] = []
    for path in text_files(root, max_bytes=max(max_file_bytes * 8, 1_000_000)):
        rel = str(path.relative_to(root))
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        low_path = rel.lower()
        low_content = content[:250_000].lower()
        score = 0
        if rel in hinted or any(
                rel == hint or rel.endswith("/" + hint) for hint in hinted):
            score += 10_000
        for term in terms:
            if term in low_path:
                score += 30
            occurrences = low_content.count(term)
            score += min(occurrences, 8)
        if path.name in _CONTEXT_NAMES:
            score += 14
        if re.search(r"(?:^|/)(?:tests?|specs?)(?:/|$)", low_path):
            score += 10
        if score:
            candidates.append((score, rel, path, content))

    candidates.sort(key=lambda row: (-row[0], row[1]))
    if not candidates:
        # No term hit is honest but unhelpful; include project metadata and the
        # smallest source files, clearly labeled as a fallback.
        for path in text_files(root, max_bytes=max_file_bytes):
            rel = str(path.relative_to(root))
            try:
                content = path.read_text(errors="replace")
            except OSError:
                continue
            score = 1 + (20 if path.name in _CONTEXT_NAMES else 0)
            candidates.append((score, rel, path, content))
        candidates.sort(key=lambda row: (-row[0], len(row[3]), row[1]))

    tree = list_files(root)
    out = [
        "# REQUIREMENT-CONDITIONED REPOSITORY EVIDENCE",
        f"# Query terms: {', '.join(sorted(terms)[:80]) or '(none)'}",
        "# Repository tree",
        *tree[:2000],
        "",
        "# Ranked file contents",
    ]
    selected: list[str] = []
    total = 0
    for score, rel, _path, content in candidates:
        if total >= max_total:
            break
        budget = min(max_file_bytes, max_total - total)
        if budget <= 0:
            break
        snippet = content[:budget]
        total += len(snippet)
        selected.append(rel)
        trailer = "\n...(truncated)" if len(content) > len(snippet) else ""
        out.append(
            f"\n--- {rel} ({len(content)} bytes; relevance {score}) ---\n"
            f"{snippet}{trailer}"
        )
    return "\n".join(out), selected
