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


_PATH_RX = re.compile(r"^[\w~@./+-]+$")


def _plausible_path(s: str) -> bool:
    """A filename line must LOOK like a path — models sometimes write prose
    above a block, and a sentence swallowed as a path OSErrors on .exists()."""
    return 0 < len(s) < 180 and _PATH_RX.match(s) is not None and ("/" in s or "." in s)


def parse_edits(text: str) -> list[EditBlock]:
    """Extract every well-formed SEARCH/REPLACE block from model output."""
    blocks: list[EditBlock] = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].strip().startswith(HEAD):
            # filename = nearest previous non-empty, non-fence line
            path = ""
            j = i - 1
            while j >= 0:
                cand = lines[j].strip()
                if cand and not _is_fence(cand):
                    cand = cand.strip("`").strip()
                    if _plausible_path(cand):
                        path = cand
                    break  # nearest non-empty line only — prose means no path
                j -= 1
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
        else:
            i += 1
    return blocks


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


def _apply_one_inner(root: Path, b: EditBlock) -> EditResult:
    fp = root / b.path
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
        fp.write_text(body)
        return EditResult(b.path, True, "created")
    if not fp.exists():
        return EditResult(b.path, False, reason="file does not exist")
    text = fp.read_text()

    if b.search in text:
        fp.write_text(text.replace(b.search, b.replace, 1))
        return EditResult(b.path, True, "exact")
    for how, fn in (("elastic", _elastic), ("fuzzy", _fuzzy)):
        new = fn(text, b.search, b.replace)
        if new is not None:
            fp.write_text(new)
            return EditResult(b.path, True, how)
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


def apply_edits(root: str | Path, blocks: list[EditBlock]) -> list[EditResult]:
    root = Path(root)
    return [_apply_one(root, b) for b in blocks]
