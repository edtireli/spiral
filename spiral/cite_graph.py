"""Citation-graph corpus expansion — depth without the model knowing its blind spots.

The unknown-unknowns problem: you cannot ask a model "is this corpus deep enough?",
because the paper it is missing is missing *precisely because it never saw it* — its
absence is invisible to introspection. So we don't ask. Instead we make depth an
**observable property of the citation graph**, which requires only counting, never
knowing-what-you-don't-know:

* **Snowball, especially backward.** Following a paper's *references* is terminology-
  and era-independent: a 2020 paper → a 2005 review → the 1969 origin, even though the
  1969 authors used words no keyword search would guess. This is how the corpus reaches
  the foundational work that keyword search structurally cannot.
* **Co-citation holes.** Count how many corpus papers cite each *external* paper. One
  cited by many, yet absent, is a load-bearing reference we are mechanically missing —
  detected from reference lists alone, with zero knowledge of its content.
* **Saturation.** "Deep enough" = a round adds few new papers and the top co-cited
  externals are already in hand. That is arithmetic, not a vibe.

Edges come from Semantic Scholar's graph API (covers maths/CS/physics; arXiv itself has
no edges). Network isolated; parsers are pure and fixture-tested. Coverage is patchy for
very old/new papers and rate-limited without a key, so saturation is "good enough", not
"provably complete" — but it is night-and-day better than one-shot keyword search.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Edge:
    """A neighbour in the citation graph (a referenced or citing paper)."""
    arxiv_id: str = ""
    title: str = ""
    year: int | None = None
    citations: int = 0
    authors: list[str] = field(default_factory=list)
    s2_id: str = ""


def _bare(arxiv_id: str) -> str:
    return (arxiv_id or "").split("v")[0]


# ── S2 graph API (network isolated) ──────────────────────────────────────────
def _get_json(url: str, timeout: float = 25.0):
    try:
        import httpx
        headers = {"User-Agent": "spiral-research/0.4"}
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
        if key:
            headers["x-api-key"] = key
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers=headers) as cl:
            r = cl.get(url)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None


def parse_edges(payload: dict, direction: str) -> list[Edge]:
    """Parse an S2 references/citations response. ``direction`` selects which side of
    each edge to read: ``references`` → the cited paper, ``citations`` → the citing one.
    Only edges that carry an arXiv id are kept (we can only fetch those)."""
    key = "citedPaper" if direction == "references" else "citingPaper"
    out: list[Edge] = []
    for row in (payload or {}).get("data", []):
        p = row.get(key) or {}
        aid = (p.get("externalIds") or {}).get("ArXiv")
        if not aid:
            continue
        out.append(Edge(
            arxiv_id=_bare(aid), title=p.get("title", "") or "",
            year=p.get("year"), citations=int(p.get("citationCount", 0) or 0),
            authors=[a.get("name", "") for a in (p.get("authors") or [])][:6],
            s2_id=p.get("paperId", "") or "",
        ))
    return out


def paper_edges(arxiv_id: str, direction: str, *, limit: int = 100,
                health: dict | None = None) -> list[Edge]:
    """Live references/citations plus explicit request-health telemetry.

    An empty successful response is different from a timeout/rate limit.  Callers need
    that distinction or a dead API can be misreported as a saturated graph.
    """
    fields = "externalIds,title,year,authors,citationCount"
    url = (f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{_bare(arxiv_id)}/"
           f"{direction}?fields={fields}&limit={min(limit, 1000)}")
    payload = _get_json(url)
    if health is not None:
        health["requests"] = int(health.get("requests", 0)) + 1
        directions = health.setdefault("directions", {})
        d = directions.setdefault(direction, {"requests": 0, "successful_requests": 0,
                                               "failed_requests": 0})
        d["requests"] += 1
        if payload is None:
            health["failed_requests"] = int(health.get("failed_requests", 0)) + 1
            d["failed_requests"] += 1
        else:
            health["successful_requests"] = int(health.get("successful_requests", 0)) + 1
            d["successful_requests"] += 1
            seeds = health.setdefault("successful_seeds", [])
            if _bare(arxiv_id) not in seeds:
                seeds.append(_bare(arxiv_id))
    return parse_edges(payload or {}, direction)


# ── snowball + hole detection (the coverage engine) ──────────────────────────
def snowball(seed_ids: list[str], have: set[str], *, backward: bool = True,
             forward: bool = True, limit_per: int = 80, max_seeds: int = 30,
             pause: float = 0.2, on=None, trace: list[dict] | None = None,
             health: dict | None = None,
             ) -> tuple[Counter, set, dict]:
    """Expand one graph step from the seeds. Returns:

    * ``back``  — Counter: external arXiv id → how many seed papers *reference* it
                  (its value is the co-citation strength that flags a hole),
    * ``fwd``   — set of external arXiv ids that *cite* the seeds (recent builders),
    * ``meta``  — external arXiv id → best :class:`Edge` metadata seen.

    ``have`` (ids already in the corpus) is excluded from all three."""
    back: Counter = Counter()
    fwd: set = set()
    meta: dict = {}
    seeds = list(seed_ids)[:max_seeds]
    if health is not None:
        health.setdefault("requests", 0)
        health.setdefault("successful_requests", 0)
        health.setdefault("failed_requests", 0)
        health.setdefault("successful_seeds", [])
        health["seeds_attempted"] = len(seeds)

    def _edges(sid: str, direction: str) -> list[Edge]:
        try:
            return paper_edges(sid, direction, limit=limit_per, health=health)
        except TypeError:
            # Compatibility for fixture/third-party paper_edges callables with the older
            # signature.  A returned fixture is a successful request for test telemetry.
            rows = paper_edges(sid, direction, limit=limit_per)
            if health is not None:
                health["requests"] += 1
                health["successful_requests"] += 1
                if sid not in health["successful_seeds"]:
                    health["successful_seeds"].append(sid)
            return rows

    for sid in seeds:
        if backward:
            for e in _edges(sid, "references"):
                if e.arxiv_id and e.arxiv_id not in have:
                    back[e.arxiv_id] += 1
                    meta[e.arxiv_id] = e
                    if trace is not None:
                        trace.append({"source": sid, "target": e.arxiv_id,
                                      "direction": "references", "title": e.title,
                                      "citations": e.citations, "year": e.year})
            if pause:
                time.sleep(pause)
        if forward:
            for e in _edges(sid, "citations"):
                if e.arxiv_id and e.arxiv_id not in have:
                    fwd.add(e.arxiv_id)
                    meta.setdefault(e.arxiv_id, e)
                    if trace is not None:
                        trace.append({"source": sid, "target": e.arxiv_id,
                                      "direction": "citations", "title": e.title,
                                      "citations": e.citations, "year": e.year})
            if pause:
                time.sleep(pause)
        if on:
            on(f"graph · {sid} (holes so far: {sum(1 for c in back.values() if c >= 2)})")
    return back, fwd, meta


def rank_holes(back: Counter, *, min_cocite: int = 2) -> list[str]:
    """Load-bearing missing papers: external ids co-cited by ≥ ``min_cocite`` corpus
    papers, strongest first. These are the holes the model could never have named."""
    return [aid for aid, _ in back.most_common() if back[aid] >= min_cocite]
