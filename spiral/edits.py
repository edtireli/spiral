"""SEARCH/REPLACE edit blocks — the primitive the worker uses to change code.

The worker emits blocks in plaintext (no JSON tool-calling overhead):

    path/to/file.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE

Local models get whitespace and surrounding context subtly wrong, so application
is layered: exact match → whitespace-elastic (strip per line) → difflib near-match.
Every block reports whether it applied and how; a failed block becomes feedback the
loop hands straight back to the model.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

HEAD = "<<<<<<<"
DIVIDER = "======="
TAIL = ">>>>>>>"


@dataclass
class EditBlock:
    path: str
    search: str
    replace: str
    mode: str = "sr"   # sr | whole  (whole = replace the entire file body)


@dataclass
class ParseOutcome:
    """What a reply yielded, and — when it yielded nothing — WHY. The 'why' is the
    difference between telling a model 'no edits parsed' (it retries the same mistake)
    and 'your path line was bold' (it fixes it)."""
    blocks: list
    fmt: str = "none"          # sr | udiff | whole | none
    dropped: list = None       # human-readable reasons, fed back to the model

    def __post_init__(self):
        if self.dropped is None:
            self.dropped = []


@dataclass
class EditResult:
    path: str
    ok: bool
    how: str = ""      # exact | elastic | fuzzy | created
    reason: str = ""   # populated on failure
    hint: str = ""     # on failure: the ACTUAL file text closest to the search —
                       # feed it back so the model corrects toward reality


# ----------------------------------------------------------------------------- parse

def _is_divider(s: str) -> bool:
    t = s.strip()
    return len(t) >= 5 and set(t) == {"="}


def _is_fence(s: str) -> bool:
    return s.strip().startswith("```")


# Models emit markers with the wrong repeat count and label the divider ("======= REPLACE").
# Normalising these BEFORE parsing rescues replies that were otherwise perfect.
_THINK_RX = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_HEAD_RX = re.compile(r"^\s*<{3,}\s*SEARCH\b.*$", re.I)
_TAIL_RX = re.compile(r"^\s*>{3,}\s*REPLACE\b.*$", re.I)
_DIV_RX = re.compile(r"^\s*={3,}\s*(?:REPLACE)?\s*$", re.I)

# brackets/parens are legal in real trees: Next.js routes (src/[id]/page.tsx),
# Xcode groups, generated files. Rejecting them made whole frameworks unbuildable.
_PATH_RX = re.compile(r"^[\w~@./+\-\[\]()]+$")
_LABEL_RX = re.compile(
    r"^(?:file|filename|filepath|path|in|edit|update|modify|create|new file)\s*[:\-]\s*", re.I)
# a token that looks like a real file: has an extension or a directory separator
_PATHISH_RX = re.compile(r"[\w~@.+\-\[\]()]+(?:/[\w~@.+\-\[\]()]+)+|[\w~@.+\-\[\]()]+\.\w{1,8}")


def _strip_reasoning(text: str) -> str:
    """Thinking models wrap chain-of-thought in <think> tags; a stray marker inside it
    would otherwise open a phantom block."""
    return _THINK_RX.sub("", text or "")


def _normalize_markers(text: str) -> str:
    """Rewrite near-miss markers to canonical form, tracking block state so a legitimate
    '=======' inside a REPLACE payload (git conflict text, markdown rule) is left alone."""
    out, in_block, in_search = [], False, False
    for ln in (text or "").splitlines():
        if not in_block and _HEAD_RX.match(ln):
            out.append(f"{HEAD} SEARCH")
            in_block = in_search = True
            continue
        if in_block and in_search and _DIV_RX.match(ln):
            out.append(DIVIDER)
            in_search = False
            continue
        if in_block and _TAIL_RX.match(ln):
            out.append(f"{TAIL} REPLACE")
            in_block = in_search = False
            continue
        out.append(ln)
    return "\n".join(out)


def _clean_path(s: str) -> str:
    """Strip the decoration models wrap a path in: **bold**, `code`, ### heading,
    'File:' labels, --- rules ---, list bullets, quotes, trailing colons, diff a/ b/."""
    t = (s or "").strip()
    t = t.strip("`").strip()
    t = re.sub(r"^#{1,6}\s*", "", t)                     # ### heading
    t = re.sub(r"^[-*+]\s+", "", t)                      # - list bullet
    m = re.match(r"^-{2,}\s*(.+?)\s*-{2,}$", t)          # --- path --- (our FILES header)
    if m:
        t = m.group(1)
    t = re.sub(r"^(?:-{2,}|\+{2,})\s*", "", t)           # --- path / +++ path (diff)
    t = re.sub(r"^[*_]{1,3}(.+?)[*_]{1,3}$", r"\1", t).strip()   # **bold**
    t = _LABEL_RX.sub("", t).strip()                     # File: path
    t = t.strip("`").strip().strip('"\'').strip()
    t = t.rstrip(":").strip()                            # path:
    t = re.sub(r"^[ab]/", "", t)                         # a/path b/path
    return t.strip("`").strip()


def _path_from_line(s: str) -> str:
    """Best-effort path from a header line, then from prose ('I'll update src/app.py now')."""
    cand = _clean_path(s)
    if _plausible_path(cand):
        return cand
    # fenced header carrying the path: ```python:src/app.py
    if _is_fence(s):
        tail = s.strip().lstrip("`").strip()
        if ":" in tail:
            cand = _clean_path(tail.split(":", 1)[1])
            if _plausible_path(cand):
                return cand
        return ""
    for tok in _PATHISH_RX.findall(s or ""):
        tok = _clean_path(tok)
        if _plausible_path(tok) and ("/" in tok or "." in tok):
            return tok
    return ""


def _plausible_path(s: str) -> bool:
    """A filename line must LOOK like a path — models sometimes write prose
    above a block, and a sentence swallowed as a path OSErrors on .exists()."""
    return 0 < len(s) < 180 and _PATH_RX.match(s) is not None and ("/" in s or "." in s)


def parse_edits(text: str) -> list[EditBlock]:
    """Extract every well-formed SEARCH/REPLACE block from model output."""
    return parse_any(text).blocks


def parse_any(text: str) -> ParseOutcome:
    """Parse a model reply into edit blocks, tolerating how models actually write.

    Tries SEARCH/REPLACE first; if that yields nothing, falls back to unified diff and
    then whole-file. Records a reason for every block it had to drop, so the loop can
    tell the model what was wrong instead of the useless 'no edits parsed'."""
    raw = _strip_reasoning(text or "")
    normalized = _normalize_markers(raw)
    outcome = _parse_search_replace(normalized)
    if outcome.blocks:
        return outcome
    for fallback in (_parse_udiff, _parse_whole_file):
        alt = fallback(raw)
        if alt.blocks:
            alt.dropped = outcome.dropped + alt.dropped
            return alt
    return outcome


def _parse_search_replace(text: str) -> ParseOutcome:
    blocks: list[EditBlock] = []
    dropped: list[str] = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    last_path = ""
    while i < n:
        if lines[i].strip().startswith(HEAD):
            # filename = nearest previous non-empty line, decoration tolerated
            path = ""
            j = i - 1
            while j >= 0:
                cand = lines[j].strip()
                if cand:
                    path = _path_from_line(cand)
                    if not path and _is_fence(cand):
                        j -= 1          # bare fence: the path may sit above it
                        continue
                    break
                j -= 1
            if not path and last_path:
                # consecutive blocks under one header — inherit it
                path = last_path
            if not path:
                near = lines[max(0, i - 1)].strip()[:80]
                dropped.append(
                    "a SEARCH/REPLACE block had no usable file path above it"
                    + (f" (saw {near!r})" if near else "")
                    + " — put the bare path on its own line directly above the block")
            last_path = path or last_path
            i += 1
            search: list[str] = []
            saw_divider = False
            while i < n:
                if _is_divider(lines[i]):
                    saw_divider = True
                    break
                search.append(lines[i])
                i += 1
            i += 1  # skip divider
            replace: list[str] = []
            while i < n and not lines[i].strip().startswith(TAIL):
                replace.append(lines[i])
                i += 1
            i += 1  # skip tail
            # missing tail at EOF = truncated reply; the block is still usable.
            # missing DIVIDER = truncated mid-SEARCH → an empty replace would
            # DELETE the matched code. Drop it.
            if path and saw_divider:
                blocks.append(EditBlock(path, "\n".join(search), "\n".join(replace)))
            elif path and not saw_divider:
                dropped.append(
                    f"{path}: block was cut off before the '{DIVIDER}' divider "
                    "(reply likely hit the token cap) — send a SHORTER block")
        else:
            i += 1
    return ParseOutcome(blocks, "sr" if blocks else "none", dropped)


_HUNK_RX = re.compile(r"^@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@")


def _parse_udiff(text: str) -> ParseOutcome:
    """Unified diff → ordinary EditBlocks, so diffs reuse the whole existing matcher
    and syntax gate rather than opening a second write path."""
    blocks: list[EditBlock] = []
    path = ""
    search: list[str] = []
    replace: list[str] = []
    in_hunk = False

    def flush():
        nonlocal search, replace, in_hunk
        if in_hunk and path and (search or replace):
            blocks.append(EditBlock(path, "\n".join(search), "\n".join(replace)))
        search, replace, in_hunk = [], [], False

    for ln in (text or "").splitlines():
        if ln.startswith("--- "):
            flush()
            continue
        if ln.startswith("+++ "):
            flush()
            cand = _clean_path(ln[4:])
            path = cand if _plausible_path(cand) else path
            continue
        if _HUNK_RX.match(ln):
            flush()
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if ln.startswith("-"):
            search.append(ln[1:])
        elif ln.startswith("+"):
            replace.append(ln[1:])
        elif ln.startswith(" ") or not ln:
            body = ln[1:] if ln.startswith(" ") else ln
            search.append(body)
            replace.append(body)
        else:
            flush()
    flush()
    return ParseOutcome(blocks, "udiff" if blocks else "none", [])


_FENCE_RX = re.compile(r"^```+\s*([\w+#.:/\-\[\]()]*)\s*$")


def _parse_whole_file(text: str) -> ParseOutcome:
    """Last-resort format: a path plus a fenced body = the file's entire new contents.
    This is the shape every model is most fluent in, and without it a model that cannot
    produce SEARCH/REPLACE has no way at all to succeed."""
    lines = (text or "").splitlines()
    blocks: list[EditBlock] = []
    i, n = 0, len(lines)
    while i < n:
        m = _FENCE_RX.match(lines[i].strip())
        if not m:
            i += 1
            continue
        path = ""
        info = m.group(1) or ""
        if ":" in info:                       # ```python:src/app.py
            path = _path_from_line(info.split(":", 1)[1])
        if not path:                          # else the nearest line above
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                path = _path_from_line(lines[j])
        i += 1
        body: list[str] = []
        while i < n and not lines[i].strip().startswith("```"):
            body.append(lines[i])
            i += 1
        i += 1
        if path and body:
            blocks.append(EditBlock(path, "", "\n".join(body), mode="whole"))
    return ParseOutcome(blocks, "whole" if blocks else "none", [])


# ----------------------------------------------------------------------------- apply

def _leading(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _reindent(rlines: list[str], from_indent: str, to_indent: str) -> list[str]:
    out: list[str] = []
    delta = len(to_indent) - len(from_indent)
    for line in rlines:
        if not line.strip():
            out.append("")
            continue
        if from_indent and line.startswith(from_indent):
            out.append(to_indent + line[len(from_indent):])
        elif delta > 0:
            out.append(" " * delta + line)
        elif delta < 0:
            cur = _leading(line)
            out.append(line[min(-delta, len(cur)):])
        else:
            out.append(line)
    return out


def _stitch(tlines: list[str], i: int, size: int, replace: str,
            slines: list[str], had_final_nl: bool) -> str:
    indent = _leading(tlines[i])
    s_indent = _leading(slines[0]) if slines and slines[0].strip() else ""
    rlines = _reindent(replace.splitlines(), s_indent, indent)
    new = tlines[:i] + rlines + tlines[i + size:]
    return "\n".join(new) + ("\n" if had_final_nl else "")


def _elastic(text: str, search: str, replace: str) -> str | None:
    """Match ignoring per-line leading/trailing whitespace, then reindent."""
    tlines = text.splitlines()
    slines = search.splitlines()
    m = len(slines)
    if m == 0:
        return None
    target = [s.strip() for s in slines]
    for i in range(0, len(tlines) - m + 1):
        if [tlines[i + k].strip() for k in range(m)] == target:
            return _stitch(tlines, i, m, replace, slines, text.endswith("\n"))
    return None


def _closest_window(text: str, search: str) -> tuple[int, int, float]:
    """Best-matching contiguous line window for `search` — (start, size, ratio)."""
    tlines = text.splitlines()
    slines = search.splitlines()
    m = max(len(slines), 1)
    best = (0, min(m, len(tlines)))
    best_r = 0.0
    for size in {m, m - 1, m + 1}:
        if size < 1 or size > len(tlines):
            continue
        for i in range(0, len(tlines) - size + 1):
            window = "\n".join(tlines[i:i + size])
            r = difflib.SequenceMatcher(None, window, search).ratio()
            if r > best_r:
                best_r, best = r, (i, size)
    return best[0], best[1], best_r


def _fuzzy(text: str, search: str, replace: str, threshold: float = 0.90) -> str | None:
    """Last resort: closest contiguous line-window by difflib ratio."""
    slines = search.splitlines()
    if not slines:
        return None
    i, size, r = _closest_window(text, search)
    if r >= threshold:
        tlines = text.splitlines()
        return _stitch(tlines, i, size, replace, slines, text.endswith("\n"))
    return None


def _apply_one(root: Path, b: EditBlock) -> EditResult:
    if b.search == b.replace:
        return EditResult(b.path, False, reason="SEARCH and REPLACE are identical — a no-op; make a real change")
    try:
        return _apply_one_inner(root, b)
    except OSError as e:
        return EditResult(b.path, False, reason=f"invalid path: {e}")


def _apply_whole(fp: Path, b: EditBlock) -> EditResult:
    """Replace a file's entire contents. The permissive fallback format, so it carries
    the strictest guards: it must parse, and it may not silently shrink a real file into
    a stub (the classic 'model rewrote my 400-line module as 12 lines' failure)."""
    body = b.replace if b.replace.endswith("\n") else b.replace + "\n"
    if "path/to" in b.path or b.path.endswith("file.ext"):
        return EditResult(b.path, False,
                          reason="placeholder path copied from the format example — "
                                 "use the real file path")
    existed = fp.exists()
    old = fp.read_text(errors="replace") if existed else ""
    if existed and old.strip() == body.strip():
        return EditResult(b.path, False, reason="whole-file body is identical — a no-op")
    error = syntax_error(fp, body)
    if error is not None:
        return EditResult(b.path, False,
                          reason=f"REJECTED — the file would not parse ({error}); nothing written")
    if existed:
        old_lines, new_lines = len(old.splitlines()), len(body.splitlines())
        if old_lines >= 20 and new_lines < old_lines * 0.5:
            return EditResult(
                b.path, False,
                reason=(f"whole-file rewrite would shrink {b.path} from {old_lines} to "
                        f"{new_lines} lines — that deletes working code; send a targeted "
                        "SEARCH/REPLACE block instead"),
                hint=old[:400])
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(body)
    return EditResult(b.path, True, "created" if not existed else "whole")


def _apply_one_inner(root: Path, b: EditBlock) -> EditResult:
    fp = root / b.path
    if b.mode == "whole":
        return _apply_whole(fp, b)
    # empty search = create a NEW file — never a silent overwrite
    if b.search.strip() == "":
        if "path/to" in b.path or b.path.endswith("file.ext"):
            return EditResult(
                b.path, False,
                reason="placeholder path copied from the format example — use the real file path",
            )
        if fp.exists():
            return EditResult(
                b.path, False,
                reason="file already EXISTS — empty SEARCH is only for NEW files; "
                       "modify it with a real SEARCH/REPLACE block",
                hint=fp.read_text(errors="replace")[:400],
            )
        fp.parent.mkdir(parents=True, exist_ok=True)
        body = b.replace if b.replace.endswith("\n") else b.replace + "\n"
        error = syntax_error(fp, body)
        if error is not None:
            return EditResult(
                b.path, False,
                reason=f"REJECTED — the new file would not parse ({error}); "
                       "nothing was written")
        fp.write_text(body)
        return EditResult(b.path, True, "created")
    if not fp.exists():
        return EditResult(b.path, False, reason="file does not exist")
    text = fp.read_text()

    if b.search in text:
        # An ambiguous SEARCH used to apply to the FIRST match and report success —
        # silently editing the wrong region. Refuse instead, and show both sites.
        hits = text.count(b.search)
        if hits > 1 and b.search.strip():
            spots = []
            start = 0
            for _ in range(min(hits, 3)):
                k = text.find(b.search, start)
                line_no = text.count("\n", 0, k) + 1
                spots.append(f"line {line_no}")
                start = k + 1
            return EditResult(
                b.path, False,
                reason=(f"SEARCH matches {hits} places ({', '.join(spots)}) — it is ambiguous; "
                        "include surrounding lines so it identifies exactly ONE region"),
                hint=b.search[:400])
        return _write_checked(
            fp, text, text.replace(b.search, b.replace, 1), "exact", b.path)
    for how, fn in (("elastic", _elastic), ("fuzzy", _fuzzy)):
        new = fn(text, b.search, b.replace)
        if new is not None:
            return _write_checked(fp, text, new, how, b.path)
    # failure → hand back reality: the actual file text nearest the search,
    # so the model's next attempt copies the file instead of its imagination
    i, size, r = _closest_window(text, b.search)
    tlines = text.splitlines()
    lo, hi = max(0, i - 1), min(len(tlines), i + size + 1)
    hint = "\n".join(tlines[lo:hi])[:700]
    return EditResult(
        b.path, False,
        reason=f"search block not found in file (closest region only {r:.0%} similar)",
        hint=hint,
    )



# ---- syntax preservation -------------------------------------------------------
# The one thing an edit must never do is turn a parsing file into a non-parsing
# one. When it happens anyway (fuzzy application landing in the wrong region, an
# unbalanced brace in a SEARCH/REPLACE), the damage is only discovered a full gate
# run later, attributed vaguely, and repaired by a model that flip-flops between
# ')' and '}' because it cannot see the imbalance. The parser can. Checking at
# APPLICATION time makes the rejection instant, atomic, and carries the parser's
# own line-numbered error back as the hint.
#
# The rule is monotone on purpose: an edit to a file that ALREADY fails to parse is
# always allowed — otherwise a broken file could never be repaired incrementally.

def _js_syntax_error(source: str, label: str) -> str | None:
    node = shutil.which("node")
    if not node:
        return None                      # no checker on this machine — allow
    probe = (
        "const vm=require('vm');let s='';process.stdin.on('data',d=>s+=d);"
        "process.stdin.on('end',()=>{try{new vm.Script(s,{filename:process.argv[1]});"
        "process.exit(0)}catch(e){if(e instanceof SyntaxError)"
        "{console.error(e.message);process.exit(1)}process.exit(0)}});"
    )
    try:
        done = subprocess.run([node, "-e", probe, label], input=source,
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stderr.strip()[:200] if done.returncode == 1 else None


_SCRIPT_TAG = re.compile(
    r"<script\b([^>]*)>([\s\S]*?)</script>", re.I)
_EXTERNAL_OR_DATA = re.compile(
    r"\bsrc\s*=|type\s*=\s*[\"']?(?:application|text)/(?:json|template)", re.I)


def syntax_error(path: Path, text: str) -> str | None:
    """The parser's verdict on this text, or None when it parses (or is uncheckable)."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            ast.parse(text, filename=path.name)
            return None
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"
    if suffix in {".js", ".mjs", ".cjs"}:
        return _js_syntax_error(text, path.name)
    if suffix in {".html", ".htm"}:
        for index, match in enumerate(_SCRIPT_TAG.finditer(text), 1):
            if _EXTERNAL_OR_DATA.search(match.group(1) or ""):
                continue
            error = _js_syntax_error(match.group(2), f"{path.name} <script> #{index}")
            if error:
                return f"<script> #{index}: {error}"
        return None
    if suffix == ".json":
        try:
            json.loads(text)
            return None
        except ValueError as exc:
            return str(exc)[:200]
    return None


def _write_checked(fp: Path, before: str, after: str, how: str,
                   block_path: str) -> EditResult:
    """Write only if the edit does not break a file that parsed before it."""
    if syntax_error(fp, before) is None:
        error = syntax_error(fp, after)
        if error is not None:
            return EditResult(
                block_path, False,
                reason=f"REJECTED — this edit makes {fp.name} stop parsing "
                       f"({error}). The file on disk is unchanged; re-read the "
                       "region and balance the block before trying again.",
                hint=_around(after, _error_line(error)),
            )
    fp.write_text(after)
    return EditResult(block_path, True, how)


def _around(text: str, line: int, radius: int = 3) -> str:
    """The rejected text near the parse error, so the retry sees the imbalance."""
    lines = text.splitlines()
    if not line:
        return ""
    lo, hi = max(0, line - 1 - radius), min(len(lines), line + radius)
    return "\n".join(lines[lo:hi])[:400]


def _error_line(error: str) -> int:
    found = re.search(r"line (\d+)", error or "")
    return int(found.group(1)) if found else 0

def apply_edits(root: str | Path, blocks: list[EditBlock]) -> list[EditResult]:
    root = Path(root)
    return [_apply_one(root, b) for b in blocks]
