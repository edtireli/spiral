"""Mine the AI-writing tells from their source instead of transcribing them by hand.

Wikipedia maintains *Signs of AI writing* as a living catalogue: each tell is a section,
and most carry an explicit ``Words to watch:`` box listing the exact phrases. Hand-copying
that list is how you end up shipping a detector that finds "not only X but Y" and misses
"not X, but Y" — the author transcribes what they remember rather than what is written.

So the phrase list is *data*, mined from the page, cached as JSON, and compiled into
patterns deterministically. Re-mining picks up whatever the editors have added since.

Notation used by the page, and how it compiles:
    ``stands/serves as``   → ``(?:stands|serves)\\s+as``      (slash = alternatives)
    ``highlighting ...``   → ``highlighting[^.!?]{0,40}``     (ellipsis = wildcard)
"""
from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

SOURCE_PAGE = "Wikipedia:Signs_of_AI_writing"
SOURCE_URL = (
    "https://en.wikipedia.org/w/api.php?action=parse&page="
    f"{SOURCE_PAGE}&prop=wikitext%7Crevid&format=json&formatversion=2")
CACHE = Path(__file__).with_name("ai_tells.json")

_SECTION = re.compile(r"^(={2,5})\s*(.+?)\s*\1\s*$", re.M)
_ITALIC = re.compile(r"''([^']+?)''")
_ANCHOR = re.compile(r"<span[^>]*></span>|\[\[([^\]|]+\|)?|\]\]|'''|''")
_TQ = re.compile(r"\{\{tq\|([^{}]{6,160})\}\}")


def _words_boxes(text: str) -> list[str]:
    """Return balanced ``{{strong|...}}`` payloads after ``Words to watch``.

    A regex ending at the first ``}}`` silently truncated the live vocabulary box as
    soon as an inline citation template appeared. The 2026 page has nested ``{{cite
    web|...}}`` and ``{{citation needed|...}}`` templates inside that box, so braces
    must be balanced rather than matched non-greedily.
    """

    out: list[str] = []
    for marker in re.finditer(r"Words to watch:\s*", text or "", re.I):
        start = (text or "").find("{{strong|", marker.end(), marker.end() + 400)
        if start < 0:
            continue
        cursor = start + len("{{strong|")
        payload_start = cursor
        depth = 1
        while cursor < len(text) - 1:
            if text.startswith("{{", cursor):
                depth += 1
                cursor += 2
                continue
            if text.startswith("}}", cursor):
                depth -= 1
                if depth == 0:
                    out.append(text[payload_start:cursor])
                    break
                cursor += 2
                continue
            cursor += 1
    return out


def _fetch_snapshot(url: str = SOURCE_URL, *, timeout: float = 30.0) -> tuple[str, int | None]:
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": (
                          "spiral-coder/0.3.1 "
                          "(https://github.com/edtireli/spiral; AI-writing style cache)")},
                      trust_env=False) as cl:
        r = cl.get(url)
        r.raise_for_status()
        parsed = r.json()["parse"]
        revision = parsed.get("revid")
        return parsed["wikitext"], (int(revision) if revision is not None else None)


def fetch_wikitext(url: str = SOURCE_URL, *, timeout: float = 30.0) -> str:
    """Fetch just the source text (kept as the small public API used by callers)."""
    return _fetch_snapshot(url, timeout=timeout)[0]


def _clean_heading(raw: str) -> str:
    return re.sub(r"\s+", " ", _ANCHOR.sub("", raw)).strip(" \"'")


def mine_wikitext(wikitext: str) -> dict:
    """Section → {words_to_watch, examples}. Purely structural; no model, no guessing."""
    text = wikitext or ""
    marks = [(m.start(), len(m.group(1)), _clean_heading(m.group(2)))
             for m in _SECTION.finditer(text)]
    out: dict = {}
    for i, (start, _level, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        phrases: list[str] = []
        for box in _words_boxes(body):
            for phrase in _ITALIC.findall(box):
                phrase = re.sub(r"\s+", " ", phrase).strip().strip(",")
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
        examples = []
        for ex in _TQ.findall(body):
            ex = re.sub(r"\s+", " ", _ANCHOR.sub("", ex)).strip()
            if 8 < len(ex) < 160:
                examples.append(ex)
        if phrases or examples:
            out[name] = {"words_to_watch": phrases, "examples": examples[:6]}
    return out


def _phrase_to_regex(phrase: str) -> str | None:
    """Compile one 'words to watch' entry into a pattern.

    Slash groups are alternatives that bind to their own token, so
    ``a crucial/pivotal role/moment`` matches "a pivotal moment"."""
    p = phrase.strip()
    if not p or len(p) < 3:
        return None
    placeholders = {
        "[country name]": "\x00COUNTRY\x00",
        "[date]": "\x00DATE\x00",
        "[a]": "\x00ARTICLE\x00",
    }
    for label, sentinel in placeholders.items():
        p = p.replace(label, sentinel)
    # Wikipedia writes wildcards both as standalone tokens (``its ... role``) and
    # attached to words (``its...``, ``...in``).  Tokenise all three forms alike.
    p = re.sub(r"(?:\.\.\.|…)", " \x00ELLIPSIS\x00 ", p)
    tokens = p.split()
    parts: list[str] = []
    for tok in tokens:
        if tok == "\x00ELLIPSIS\x00":
            parts.append(r"[^.!?]{0,40}?")
            continue
        tok = tok.strip(",;:")
        if not tok:
            continue
        if tok == "\x00COUNTRY\x00":
            parts.append(r"\w+")
            continue
        if tok == "\x00DATE\x00":
            parts.append(
                r"(?:\d{4}|(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)(?:\s+\d{1,2},?)?\s+\d{4})"
            )
            continue
        if tok == "\x00ARTICLE\x00":
            parts.append(r"(?:a|an|the)")
            continue
        if "/" in tok:
            alts = [re.escape(a) for a in tok.split("/") if a]
            if not alts:
                continue
            parts.append("(?:" + "|".join(alts) + ")")
        else:
            parts.append(re.escape(tok))
    if not parts:
        return None
    # A wildcard already spans arbitrary whitespace, so do not require another space
    # beside it.  Other tokens retain a normal flexible-space boundary.
    body = ""
    for part in parts:
        if body and not body.endswith("?") and not part.startswith("[^.!?"):
            body += r"\s+"
        body += part
    first_literal = next((part for part in parts if not part.startswith("[^.!?")), "")
    last_literal = next((part for part in reversed(parts)
                         if not part.startswith("[^.!?")), "")
    lead = r"(?<!\w)" if first_literal and parts[0] == first_literal else ""
    tail = r"(?!\w)" if last_literal and parts[-1] == last_literal else ""
    return f"{lead}{body}{tail}"


def compile_tells(mined: dict) -> list[tuple[str, str, re.Pattern]]:
    """Section list → compiled (id, explanation, pattern). One pattern per section,
    alternating over its phrases, so a hit reports which *kind* of tell it is."""
    compiled: list[tuple[str, str, re.Pattern]] = []
    for name, data in (mined or {}).items():
        alts = [r for r in (_phrase_to_regex(p) for p in data.get("words_to_watch", []))
                if r]
        if not alts:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
        try:
            pattern = re.compile("|".join(f"(?:{a})" for a in alts), re.I)
        except re.error:
            continue
        compiled.append((slug, name, pattern))
    return compiled


def refresh(cache: Path = CACHE) -> dict:
    """Re-mine from Wikipedia and update the cached JSON."""
    source, revision = _fetch_snapshot()
    mined = mine_wikitext(source)
    payload = {"source": f"https://en.wikipedia.org/wiki/{SOURCE_PAGE}",
               "revision": revision,
               "revision_url": (
                   f"https://en.wikipedia.org/w/index.php?oldid={revision}&title="
                   f"{SOURCE_PAGE}" if revision else None),
               "fetched_at": datetime.now(timezone.utc).isoformat(),
               "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
               "sections": mined}
    cache.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return mined


def load(cache: Path = CACHE) -> dict:
    """Cached mined tells; empty dict when the cache is absent (the hand-written
    patterns in ``writing_style`` still apply, so detection degrades, never breaks)."""
    try:
        return json.loads(cache.read_text()).get("sections") or {}
    except Exception:
        return {}
