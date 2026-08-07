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
import re
from pathlib import Path

SOURCE_PAGE = "Wikipedia:Signs_of_AI_writing"
SOURCE_URL = (
    "https://en.wikipedia.org/w/api.php?action=parse&page="
    f"{SOURCE_PAGE}&prop=wikitext&format=json&formatversion=2")
CACHE = Path(__file__).with_name("ai_tells.json")

_SECTION = re.compile(r"^(={2,5})\s*(.+?)\s*\1\s*$", re.M)
_WORDS_BOX = re.compile(r"Words to watch:\s*\{\{strong\|(.+?)\}\}", re.S)
_ITALIC = re.compile(r"''([^']+?)''")
_ANCHOR = re.compile(r"<span[^>]*></span>|\[\[([^\]|]+\|)?|\]\]|'''|''")
_TQ = re.compile(r"\{\{tq\|([^{}]{6,160})\}\}")


def fetch_wikitext(url: str = SOURCE_URL, *, timeout: float = 30.0) -> str:
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "spiral-research/0.4"}) as cl:
        r = cl.get(url)
        r.raise_for_status()
        return r.json()["parse"]["wikitext"]


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
        for box in _WORDS_BOX.findall(body):
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
    p = p.replace("[country name]", "\x00COUNTRY\x00")
    tokens = p.split()
    parts: list[str] = []
    for tok in tokens:
        if tok in {"...", "…"}:
            parts.append(r"[^.!?]{0,40}?")
            continue
        tok = tok.strip(",;:")
        if not tok:
            continue
        if tok == "\x00COUNTRY\x00":
            parts.append(r"\w+")
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
    body = r"\s+".join(parts)
    # word-boundary only when the pattern starts/ends with a word character
    lead = r"\b" if re.match(r"[\w(]", parts[0][0] if parts[0] else "") or \
        parts[0].startswith("(?:") else ""
    return f"{lead}{body}"


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
    mined = mine_wikitext(fetch_wikitext())
    payload = {"source": f"https://en.wikipedia.org/wiki/{SOURCE_PAGE}",
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
