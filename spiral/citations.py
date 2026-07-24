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
    identifier: str = ""


def _get_json(url: str, timeout: float = 25.0, report: dict | None = None):
    try:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "spiral-research/0.3"}) as cl:
            r = cl.get(url)
            r.raise_for_status()
            payload = r.json()
            if report is not None:
                report.update({"source_ok": True, "status": r.status_code, "error": ""})
            return payload
    except Exception as exc:
        if report is not None:
            report.update({"source_ok": False, "status": 0,
                           "error": f"{type(exc).__name__}: {exc}"})
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
        eprints = md.get("arxiv_eprints") or []
        identifier = str((eprints[0] if eprints else {}).get("value") or "")
        out.append(Prior(
            title=title, year=yr, authors=authors,
            url=f"https://inspirehep.net/literature/{hit.get('id','')}",
            citations=int(md.get("citation_count", 0) or 0),
            source="inspire", abstract=(md.get("abstracts") or [{}])[0].get("value", "")[:800],
            identifier=identifier,
        ))
    return out


def parse_semantic_scholar(payload: dict) -> list[Prior]:
    out: list[Prior] = []
    for p in (payload or {}).get("data", []):
        identifier = str((p.get("externalIds") or {}).get("ArXiv") or "")
        out.append(Prior(
            title=p.get("title", ""), year=p.get("year"),
            authors=[a.get("name", "") for a in (p.get("authors") or [])][:6],
            url=p.get("url", "") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}",
            citations=int(p.get("citationCount", 0) or 0),
            source="semantic_scholar", abstract=(p.get("abstract") or "")[:800],
            identifier=identifier,
        ))
    return out


# ── live queries ─────────────────────────────────────────────────────────────
def inspire(query: str, k: int = 8, report: dict | None = None) -> list[Prior]:
    """High-energy-physics prior art, most-cited first."""
    import urllib.parse
    q = urllib.parse.quote(query)
    url = (f"https://inspirehep.net/api/literature?q={q}&size={k}"
           "&sort=mostcited&fields=titles,authors,earliest_date,citation_count,abstracts,arxiv_eprints")
    payload = _get_json(url, report=report)
    hits = parse_inspire(payload or {})
    if report is not None:
        report.update({"source": "inspire", "query": query, "url": url,
                       "result_count": len(hits)})
    return hits


def semantic_scholar(query: str, k: int = 8, report: dict | None = None) -> list[Prior]:
    """General prior art (maths/CS/all arXiv)."""
    import urllib.parse
    q = urllib.parse.quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}"
           f"&limit={k}&fields=title,year,authors,citationCount,url,abstract,externalIds")
    payload = _get_json(url, report=report)
    hits = parse_semantic_scholar(payload or {})
    if report is not None:
        report.update({"source": "semantic_scholar", "query": query, "url": url,
                       "result_count": len(hits)})
    return hits


def arxiv_prior(query: str, k: int = 8, report: dict | None = None) -> list[Prior]:
    """Independent arXiv metadata search for source-health and query coverage."""

    from spiral.research import arxiv

    local_report: dict = {}
    hits = arxiv(query, k=k, report=local_report)
    out = []
    for hit in hits:
        aid = hit.url.rstrip("/").split("/")[-1]
        year = None
        if getattr(hit, "published", "")[:4].isdigit():
            year = int(hit.published[:4])
        out.append(Prior(
            title=hit.title, year=year,
            authors=[a.strip() for a in hit.snippet.split(",") if a.strip()][:8],
            url=hit.url, source="arxiv", abstract=hit.text[:800], identifier=aid,
        ))
    if report is not None:
        report.update(local_report)
        report.update({"source": "arxiv", "query": query, "result_count": len(out)})
    return out


def prior_art(query: str, k: int = 8, physics: bool = True,
              report: dict | None = None) -> list[Prior]:
    """Best-available prior art: INSPIRE for physics, Semantic Scholar otherwise/also.
    De-duplicated by lowercased title, most-cited first."""
    hits: list[Prior] = []
    source_reports: dict[str, dict] = {}
    if physics:
        source_reports["inspire"] = {}
        hits += inspire(query, k=k, report=source_reports["inspire"])
    source_reports["semantic_scholar"] = {}
    hits += semantic_scholar(query, k=k, report=source_reports["semantic_scholar"])
    source_reports["arxiv"] = {}
    hits += arxiv_prior(query, k=k, report=source_reports["arxiv"])
    seen, uniq = {}, []
    for h in sorted(hits, key=lambda p: -p.citations):
        key = h.title.lower().strip()
        if key and key not in seen:
            seen[key] = h
            uniq.append(h)
        elif key:
            held = seen[key]
            # Preserve the most-cited record but enrich it with an arXiv identifier and
            # abstract found by another database. This allows a nearby prior to be fetched
            # and read instead of remaining a title-only search result.
            if not held.identifier and h.identifier:
                held.identifier = h.identifier
            if not held.abstract and h.abstract:
                held.abstract = h.abstract
    result = uniq[:k]
    if report is not None:
        ok_sources = [name for name, r in source_reports.items() if r.get("source_ok") is True]
        report.update({
            "query": query,
            "sources": source_reports,
            "sources_attempted": list(source_reports),
            "sources_ok": ok_sources,
            "healthy_source_count": len(ok_sources),
            "result_count": len(result),
            "ready": len(ok_sources) >= 2,
        })
    return result


def novelty_digest(priors: list[Prior]) -> str:
    """Compact prior-art digest for a reasoning prompt (the model judges novelty)."""
    if not priors:
        return "No prior art found in the searched databases (weak evidence of novelty — widen the query before trusting it)."
    lines = []
    for p in priors:
        line = f"- {p.title} ({p.year or '?'}, {p.citations} cites) {p.url}"
        abstract = " ".join((p.abstract or "").split())[:600]
        if abstract:
            line += f"\n  abstract: {abstract}"
        if p.identifier:
            line += f"\n  arXiv: {p.identifier}"
        lines.append(line)
    return "PRIOR ART:\n" + "\n".join(lines)
