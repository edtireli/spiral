"""Bounded source navigation and receipts for the text in a worker's prompt.

An index locates evidence; it never proves behavior or model comprehension.
Rebuild on lookup and compare content hashes on use, including after a restart.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from spiral import tools
from spiral.runtime_control import checkpoint

MAX_FILE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    text: str
    kind: str = "workspace"

    def page(self, start: int = 0, size: int = 3000) -> "SourcePage":
        start = min(len(self.text), max(0, start))
        return SourcePage(self.path, self.sha256, start, len(self.text),
                          self.text[start:start + max(1, size)], self.kind)


@dataclass(frozen=True)
class SourcePage:
    path: str
    sha256: str
    start: int
    total: int
    text: str
    kind: str = "workspace"

    @property
    def end(self) -> int:
        return self.start + len(self.text)

    def render(self) -> str:
        label = ("Historical saved context; check against current TASK and source. "
                 if self.kind == "saved_context" else "")
        return (label + f"--- {self.path}; sha256 {self.sha256}; characters "
                f"{self.start}:{self.end} of {self.total} ---\n{self.text}\n"
                f"[Source text only. Continue: ASK: file {self.path} :: offset {self.end}]\n")


def read_source(root: Path, relative: str) -> Source:
    path = tools.resolve_workspace_path(root, relative)
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("source unavailable or exceeds 2 MiB read bound")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
        raise ValueError("source exceeds read bound or is binary")
    return Source(relative, hashlib.sha256(raw).hexdigest(), raw.decode("utf-8"))


def verification_files(root: Path, command: str) -> list[str]:
    """Literal existing file arguments are evidence dependencies, never commands.

    No shell expansion or execution. A gate's own source can explain a terse
    assertion report that omits the expected condition.
    """
    try:
        arguments = shlex.split(command)[:128]
    except ValueError:
        return []
    found = []
    for argument in arguments:
        if len(argument) > 1024:
            continue
        path = Path(argument)
        if path.is_absolute():
            try:
                argument = path.relative_to(root).as_posix()
            except ValueError:
                continue
        try:
            read_source(root, argument)
        except (OSError, ValueError, UnicodeError):
            continue
        if argument not in found:
            found.append(argument)
        if len(found) >= 4:
            break
    return found


def covered(source: Source, start: int, end: int, pages: list[SourcePage]) -> bool:
    """Only current-version ranges actually rendered in this prompt count."""
    cursor = start
    for page in sorted(pages, key=lambda p: p.start):
        if page.path != source.path or page.sha256 != source.sha256:
            continue
        if page.start <= cursor:
            cursor = max(cursor, page.end)
        if cursor >= end:
            return True
    return False


def edit_gap(root: Path, blocks, pages: list[SourcePage]):
    """Return the first missing source region before applying any proposed block.

Creation and invalid/protected paths remain subject to the existing edit broker.
Missing SEARCH text in a partial view requires fresh evidence and a new exact
SEARCH; a guessed approximate match must not write into an unseen region.
    """
    for block in blocks:
        try:
            path = tools.resolve_workspace_path(root, block.path)
        except ValueError:
            continue  # existing edit validation refuses invalid paths
        if not path.exists():
            continue  # creation/missing-file semantics remain in edits.py
        try:
            source = read_source(root, block.path)
        except (OSError, ValueError, UnicodeError) as exc:
            return f"Cannot obtain bounded source evidence for {block.path}: {exc}. No edit is admitted.", None
        if not block.search.strip() and block.mode != "whole":
            continue  # existing-file empty SEARCH is already refused
        if covered(source, 0, len(source.text), pages):
            continue
        if block.mode == "whole":
            start, end = 0, len(source.text)
            reason = "Whole-file replacement includes source outside the current view. Use a small SEARCH block."
        elif block.search and source.text.count(block.search) == 1:
            start = source.text.index(block.search)
            end = start + len(block.search)
            if covered(source, start, end, pages):
                continue
            reason = "The proposed edit targets source absent from the current prompt or changed since it was read."
        else:
            # Exact ambiguity is refused by edits.py. Missing/approximate text
            # must first be grounded; choose a literal line as a navigation hint.
            if block.search and source.text.count(block.search) > 1:
                continue
            lines = sorted((line.strip() for line in block.search.splitlines()
                            if len(line.strip()) >= 8), key=len, reverse=True)
            start = next((source.text.index(line) for line in lines
                          if line in source.text), 0)
            end = start
            reason = "SEARCH is not exact and the full current file is not visible. Read the target and use exact current text."
        page = source.page(max(0, start - 500), min(5000, max(3000, end - start + 1000)))
        return reason, page
    return None


def diagnostic_gap(root: Path, output: str, pages: list[SourcePage]):
    """A structured diagnostic address can reveal a gap before another LM call.

    Only actual workspace paths are read. Free prose is not interpreted as a
    command or authorization. The last traceback frame is the nearest failure.
    """
    matches = list(re.finditer(
        r'File ["\']([^"\'\n]+)["\'], line (\d+)|'
        r'(?m:^|\s)(/?(?:[\w.@+ -]+/)*[\w.@+-]+\.[A-Za-z0-9]+):(\d+)(?::\d+)?',
        output[-12000:]))
    for match in reversed(matches[-30:]):
        path, number = (match[1], match[2]) if match[1] else (match[3], match[4])
        path = path.strip()
        if Path(path).is_absolute():
            try:
                path = Path(path).relative_to(root).as_posix()
            except ValueError:
                continue
        try:
            source = read_source(root, path)
        except (OSError, ValueError, UnicodeError):
            continue
        lines = source.text.splitlines(keepends=True)
        line = int(number) - 1
        if line < 0 or line >= len(lines):
            continue
        start = sum(len(value) for value in lines[:line])
        end = start + len(lines[line])
        if not covered(source, start, end, pages):
            return source.page(max(0, start - 500), 3000)
    return None


class ContextMap:
    """Fresh, bounded file/symbol/import map. No model or code execution."""

    def __init__(self, root: Path, *, max_files: int = 4000,
                 max_bytes: int = 32 * 1024 * 1024, max_entries: int = 20000,
                 extra_roots: tuple[str, ...] = ()):
        self.root = root.resolve()
        self.sources: dict[str, Source] = {}
        self.symbols: dict[str, list[dict]] = {}
        self.imports: dict[str, list[str]] = {}
        self.omitted = 0
        self.limited = False
        consumed = visited = 0
        roots = [self.root]
        for relative in extra_roots[:4]:
            try:
                path = tools.resolve_workspace_path(self.root, relative)
                if path.is_dir() and path not in roots:
                    roots.append(path)
            except (OSError, ValueError):
                continue
        def walk():
            for root in roots:
                yield from os.walk(root, followlinks=False)
        for directory, dirs, names in walk():
            checkpoint()
            dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                             and d not in tools._SKIP_DIRS
                             and not (Path(directory) / d).is_symlink())
            visited += len(dirs) + len(names)
            if visited > max_entries:
                self.limited = True
                break
            for name in sorted(names):
                checkpoint()
                path = Path(directory) / name
                if name.startswith(".") or path.is_symlink():
                    self.omitted += 1
                    continue
                if len(self.sources) >= max_files or consumed >= max_bytes:
                    self.limited = True
                    break
                rel = path.relative_to(self.root).as_posix()
                if rel in self.sources:
                    continue
                try:
                    size = path.stat().st_size
                    if size > min(MAX_FILE_BYTES, max_bytes - consumed):
                        self.omitted += 1
                        continue
                    consumed += size  # invalid/binary reads also spend the I/O allowance
                    source = read_source(self.root, rel)
                except (OSError, ValueError, UnicodeError):
                    self.omitted += 1
                    continue
                self.sources[rel] = source
                self.symbols[rel], self.imports[rel] = self._python_map(source)
            if self.limited:
                break
        # Exact originals already written by Atom survive prompt compaction.
        # Index only those content-addressed records, never recursive map
        # manifests, tooling, caches, credentials or arbitrary hidden files.
        try:
            context_dir = tools.resolve_workspace_path(self.root, ".spiral/context")
            if not self.limited and context_dir.is_dir():
                with os.scandir(context_dir) as entries:
                    for entry in entries:
                        checkpoint()
                        visited += 1
                        if (visited > max_entries or len(self.sources) >= max_files
                                or consumed >= max_bytes):
                            self.limited = True
                            break
                        match = re.fullmatch(
                            r"(?:project|task|attempts(?:-view)?|verification-output|"
                            r"worker-(?:scope|reply|verification|feedback|tool-result)(?:-part)?|"
                            r"tool-(?:output|history|working-set))-([a-f0-9]{64})\.txt", entry.name)
                        if not match or entry.is_symlink() or not entry.is_file():
                            continue
                        size = entry.stat().st_size
                        if size > min(MAX_FILE_BYTES, max_bytes - consumed):
                            self.omitted += 1
                            continue
                        consumed += size
                        rel = ".spiral/context/" + entry.name
                        try:
                            source = read_source(self.root, rel)
                        except (OSError, ValueError, UnicodeError):
                            self.omitted += 1
                            continue
                        if source.sha256 != match[1]:
                            self.omitted += 1
                            continue
                        self.sources[rel] = Source(source.path, source.sha256, source.text, "saved_context")
                        self.symbols[rel], self.imports[rel] = [], []
        except (OSError, ValueError):
            pass

    @staticmethod
    def _python_map(source: Source):
        if not source.path.endswith(".py") or len(source.text) > 256000:
            return [], []
        try:
            tree = ast.parse(source.text)
        except (SyntaxError, RecursionError, ValueError):
            return [], []
        symbols, imports = [], []
        offsets = [0]
        for line in source.text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        for node in ast.walk(tree):
            checkpoint()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(dict(name=node.name, line=node.lineno,
                                    offset=offsets[node.lineno - 1]))
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                imports.append(base)
                imports.extend(base + ("." if node.module else "") + alias.name
                               for alias in node.names if alias.name != "*")
        return symbols, sorted(set(imports))

    def dependencies(self, path: str) -> list[str]:
        found = set()
        for module in self.imports.get(path, []):
            level = len(module) - len(module.lstrip("."))
            name = module.lstrip(".").replace(".", "/")
            base = Path(path).parent
            for _ in range(max(0, level - 1)):
                base = base.parent
            stems = [(base / name).as_posix()] if level else [name, (base / name).as_posix()]
            for stem in stems:
                for candidate in (stem + ".py", stem + "/__init__.py"):
                    if candidate in self.sources:
                        found.add(candidate)
        return sorted(found)

    def manifest(self) -> dict:
        return {"version": 1, "limited": self.limited, "omitted": self.omitted,
                "scope": "bounded workspace text; hidden, binary and oversized sources omitted",
                "files": [{"path": path, "sha256": source.sha256,
                           "characters": len(source.text), "kind": source.kind,
                           "symbols": self.symbols[path],
                           "imports": self.imports[path], "local_dependencies": self.dependencies(path)}
                          for path, source in self.sources.items()]}

    def lookup(self, query: str) -> tuple[str, list[SourcePage]]:
        query = query.strip()[:500]
        terms = list(dict.fromkeys(re.findall(r"[\w./-]+", query.lower())))[:16]
        patterns = [re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)") for term in terms]
        matches = []
        frequencies = [0] * len(terms)
        for path, source in self.sources.items():
            checkpoint()
            low = source.text.lower()
            counts, offsets = [], []
            for i, pattern in enumerate(patterns):
                # A few occurrences suffice for ranking; do not materialize all
                # matches in generated source or long repeated tool output.
                hits = []
                for match in pattern.finditer(low):
                    hits.append(match.start())
                    if len(hits) == 3:
                        break
                counts.append(len(hits))
                offsets.append(hits[0] if hits else None)
                frequencies[i] += bool(hits)
            matches.append((path, counts, offsets))
        ranked = []
        for path, counts, offsets in matches:
            score = sum(count * (1 + math.log((len(self.sources) + 1) / (frequencies[i] + 1)))
                        for i, count in enumerate(counts))
            score += sum(8 for term in terms if term in path.lower())
            score += sum(30 for symbol in self.symbols[path] if symbol["name"].lower() in terms)
            if score:
                at = next((symbol["offset"] for symbol in self.symbols[path]
                           if symbol["name"].lower() in terms),
                          next((offset for offset in offsets if offset is not None), 0))
                ranked.append((-score, path, at))
        ranked.sort()
        header = (f"CONTEXT MAP: {len(self.sources)} indexed text files; omitted={self.omitted}; "
                  f"scan limit reached={self.limited}. Python definitions/import edges are static; "
                  "other languages use text lookup. Matches are navigation evidence, not proof.\n")
        if not ranked:
            return header + "No match in this bounded indexed scope. Refine the query or ASK: file <known path>.", []
        pages, rows = [], []
        for _, path, at in ranked[:3]:
            page = self.sources[path].page(max(0, at - 150), 1000)
            pages.append(page)
            deps = self.dependencies(path)
            callers = [p for p in self.sources if path in self.dependencies(p)][:8]
            rows.append(f"{path}: static local import candidates {deps}; import references from {callers}\n" + page.render())
        return header + "\n".join(rows), pages
