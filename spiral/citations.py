"""Novelty & prior-art search — the check that a "new" result is actually new.

A research loop that only reads what it downloaded will happily rediscover 1962. This
module is the antidote: before a result or question is accepted as novel it is searched
against the literature graph — who worked on this, when, and how far did they get. That
turns "I think this is new" into evidence, the same way ``verify_math`` turns "I think
this is true" into a check.

Two graphs, whichever answers:
  * **INSPIRE-HEP** — the high-energy-physics literature DB, with real citation/reference
    edges (ideal for "this was 80 %-done in the '60s; what's left").
  * **Semantic Scholar** — everything else (maths, CS, all of arXiv), with citation counts.

Network isolated; JSON parsing is pure and unit-tested against fixtures. Results are
DATA the reasoning model weighs — this module reports prior art, it does not judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prior:
    """One piece of prior art found in the literature graph."""
    title: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    url: str = ""
    citations: int = 0
    source: str = ""            # inspire | semantic_scholar
    abstract: str = ""


def _get_json(url: str, timeout: float = 25.0):
    try:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "spiral-research/0.3"}) as cl:
            r = cl.get(url)
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


# ── parsers (pure — fixture-tested) ──────────────────────────────────────────
def parse_inspire(payload: dict) -> list[Prior]:
    out: list[Prior] = []
    for hit in (payload or {}).get("hits", {}).get("hits", []):
        md = hit.get("metadata", {})
        title = (md.get("titles") or [{}])[0].get("title", "")
        authors = [a.get("full_name", "") for a in md.get("authors", [])][:6]
        year = md.get("earliest_date", "") or md.get("preprint_date", "")
        yr = int(year[:4]) if year[:4].isdigit() else None
        out.append(Prior(
            title=title, year=yr, authors=authors,
            url=f"https://inspirehep.net/literature/{hit.get('id','')}",
            citations=int(md.get("citation_count", 0) or 0),
            source="inspire", abstract=(md.get("abstracts") or [{}])[0].get("value", "")[:800],
        ))
    return out


def parse_semantic_scholar(payload: dict) -> list[Prior]:
    out: list[Prior] = []
    for p in (payload or {}).get("data", []):
        out.append(Prior(
            title=p.get("title", ""), year=p.get("year"),
            authors=[a.get("name", "") for a in (p.get("authors") or [])][:6],
            url=p.get("url", "") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
            citations=int(p.get("citationCount", 0) or 0),
            source="semantic_scholar", abstract=(p.get("abstract") or "")[:800],
        ))
    return out


# ── live queries ─────────────────────────────────────────────────────────────
def inspire(query: str, k: int = 8) -> list[Prior]:
    """High-energy-physics prior art, most-cited first."""
    import urllib.parse
    q = urllib.parse.quote(query)
    url = (f"https://inspirehep.net/api/literature?q={q}&size={k}"
           "&sort=mostcited&fields=titles,authors,earliest_date,citation_count,abstracts")
    return parse_inspire(_get_json(url) or {})


def semantic_scholar(query: str, k: int = 8) -> list[Prior]:
    """General prior art (maths/CS/all arXiv)."""
    import urllib.parse
    q = urllib.parse.quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
           f"&limit={k}&fields=title,year,authors,citationCount,url,abstract")
    return parse_semantic_scholar(_get_json(url) or {})


def prior_art(query: str, k: int = 8, physics: bool = True) -> list[Prior]:
    """Best-available prior art: INSPIRE for physics, Semantic Scholar otherwise/also.
    De-duplicated by lowercased title, most-cited first."""
    hits: list[Prior] = []
    if physics:
        hits += inspire(query, k=k)
    hits += semantic_scholar(query, k=k)
    seen, uniq = set(), []
    for h in sorted(hits, key=lambda p: -p.citations):
        key = h.title.lower().strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(h)
    return uniq[:k]


def novelty_digest(priors: list[Prior]) -> str:
    """Compact prior-art digest for a reasoning prompt (the model judges novelty)."""
    if not priors:
        return "No prior art found in the searched databases (weak evidence of novelty — widen the query before trusting it)."
    lines = [f"- {p.title} ({p.year or '?'}, {p.citations} cites) {p.url}" for p in priors]
    return "PRIOR ART:\n" + "\n".join(lines)
