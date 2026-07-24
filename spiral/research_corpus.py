"""The paper corpus — download, extract, store, dedup the source material.

`spiral research` reasons over *primary sources*, not just abstracts: for each arXiv
paper it fetches the PDF **and the TeX e-print source**, because the source carries the
equations, definitions and derivations the loop needs to verify (a PDF's math is baked
into glyphs; the .tex has ``\\begin{equation}`` you can read). Everything lands in a
local store keyed by arXiv id, so a multi-round loop never re-downloads and a run is
reproducible.

Network is a tool, isolated to ``_download`` — GET-only, https, size-capped. Fetched
content is UNTRUSTED DATA: corpus material for the reasoning model, never instructions
and never executed.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_DOWNLOAD = 30_000_000       # 30 MB cap per file
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?|([a-z-]+/\d{7})(v\d+)?", re.I)


@dataclass
class Paper:
    """One source paper. ``text`` is the extracted body (TeX preferred, else PDF)."""
    arxiv_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)
    url: str = ""
    pdf_path: str = ""
    tex_path: str = ""
    text: str = ""
    body_source: str = ""       # tex | pdf | abstract | missing
    fetch_errors: list[str] = field(default_factory=list)
    content_hash: str = ""

    @property
    def bare_id(self) -> str:
        """Version-stripped id (``2401.01234v2`` → ``2401.01234``) — the dedup key."""
        return self.arxiv_id.split("v")[0]


def parse_arxiv_id(s: str) -> str | None:
    """Pull an arXiv id out of a url, abs page, ``arXiv:...`` string, or bare id."""
    m = _ARXIV_ID.search(str(s))
    if not m:
        return None
    return (m.group(1) or m.group(3)) + (m.group(2) or m.group(4) or "")


# ── network (isolated) ───────────────────────────────────────────────────────
def _download(url: str, timeout: float = 40.0, report: dict | None = None) -> bytes | None:
    """GET one url as bytes (https, size-capped). None on any failure — a missing
    download degrades a paper to abstract-only, it never aborts a run."""
    if not url.startswith(("http://", "https://")):
        return None
    try:
        import httpx
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "spiral-research/0.3"}) as cl:
            r = cl.get(url)
            r.raise_for_status()
            declared = int(r.headers.get("content-length") or 0)
            if declared > MAX_DOWNLOAD or len(r.content) > MAX_DOWNLOAD:
                if report is not None:
                    report.update({"ok": False, "error": "download exceeds size limit",
                                   "bytes": max(declared, len(r.content))})
                return None
            if report is not None:
                report.update({"ok": True, "error": "", "bytes": len(r.content)})
            return r.content
    except Exception as exc:
        if report is not None:
            report.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "bytes": 0})
        return None


# ── TeX extraction ───────────────────────────────────────────────────────────
def _clean_tex(tex: str) -> str:
    """Strip TeX comments and collapse whitespace — keep the math and prose."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)                     # line comments (not \%)
    return re.sub(r"\n{3,}", "\n\n", tex).strip()


def extract_tex_source(raw: bytes) -> str:
    """arXiv e-prints are a gzipped tar of .tex/.bbl (or a single gzipped .tex, or a
    bare .tex). Concatenate every .tex/.bbl found — the main file plus its inputs —
    with the main document FIRST and bibliographies LAST, so ``text[:n]`` excerpts
    show the paper's actual body (equations, derivations) rather than a reference
    list or preamble. A run once fed its critic nothing but ``\\begin{thebibliography}``
    blocks because the .bbl happened to sort first in the tar."""
    if not raw:
        return ""
    compressed = raw.startswith(b"\x1f\x8b")

    def _part_rank(name: str, content: str) -> int:
        if "\\begin{document}" in content:
            return 0                                    # the main file
        if name.lower().endswith(".bbl") or "\\begin{thebibliography}" in content:
            return 2                                    # reference lists last
        return 1                                        # included sections between

    # try tar.gz (the common multi-file case)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            parts = []
            for m in tar.getmembers():
                if m.isfile() and m.name.lower().endswith((".tex", ".bbl")):
                    f = tar.extractfile(m)
                    if f:
                        content = f.read().decode("utf-8", "ignore")
                        parts.append((_part_rank(m.name, content), m.name, content))
            if parts:
                parts.sort(key=lambda item: (item[0], item[1]))
                return _clean_tex("\n\n".join(content for _, _, content in parts))
    except (tarfile.TarError, OSError, EOFError, gzip.BadGzipFile):
        pass
    # single gzipped file
    try:
        return _clean_tex(gzip.decompress(raw).decode("utf-8", "ignore"))
    except (OSError, EOFError, gzip.BadGzipFile):
        pass
    if compressed:
        return ""
    # bare text
    try:
        s = raw.decode("utf-8", "ignore")
        if "\\" in s or "\\begin" in s:
            return _clean_tex(s)
    except Exception:
        pass
    return ""


_BIB_ENV = re.compile(r"\\begin\{thebibliography\}.*?(?:\\end\{thebibliography\}|\Z)", re.S)


def display_body(text: str, chars: int = 1400) -> str:
    """The readable slice of a stored paper text for a reasoning prompt: start at the
    document body, drop reference lists and macro-definition noise, then truncate.
    Corpora stored before body-first extraction may hold bibliography-first text —
    this fixes the *view* without refetching anything."""
    if not text:
        return ""
    body = text
    m = re.search(r"\\begin\{document\}", body)
    if m:
        body = body[m.end():]
    body = _BIB_ENV.sub(" ", body)
    body = re.sub(
        r"\\(documentclass|usepackage|providecommand|newcommand|renewcommand"
        r"|def|makeatletter|makeatother)\b[^\n]*", " ", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 40 <= len(text):
        body = text.strip()          # degenerate extraction — show the raw text honestly
    return body[:chars]


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF (best-effort; TeX source is always preferred)."""
    import logging

    logger = logging.getLogger("pypdf")
    previous_level = logger.level
    try:
        from pypdf import PdfReader
        # Truncated arXiv/CDN responses can make pypdf print "EOF marker not found"
        # even though this best-effort fallback catches the exception. Source-health
        # metadata is the canonical place for that failure, not the live status UI.
        logger.setLevel(logging.CRITICAL)
        return "\n".join(
            (p.extract_text() or "") for p in PdfReader(str(path)).pages
        ).strip()
    except Exception:
        return ""
    finally:
        logger.setLevel(previous_level)


# ── the store ────────────────────────────────────────────────────────────────
@dataclass
class Corpus:
    """A local store of papers under ``root``, deduplicated by bare arXiv id."""
    root: Path
    papers: dict[str, Paper] = field(default_factory=dict)
    last_build_report: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.root = Path(self.root)
        (self.root / "papers").mkdir(parents=True, exist_ok=True)
        self._load()

    def _manifest(self) -> Path:
        return self.root / "corpus.json"

    def _load(self):
        f = self._manifest()
        if f.is_file():
            for d in json.loads(f.read_text()).get("papers", []):
                self.papers[Paper(**d).bare_id] = Paper(**d)

    def save(self):
        target = self._manifest()
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"papers": [asdict(p) for p in self.papers.values()]},
                indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)

    def has(self, arxiv_id: str) -> bool:
        return Paper(arxiv_id=arxiv_id).bare_id in self.papers

    def add(self, paper: Paper, *, fetch: bool = True) -> Paper:
        """Add a paper, downloading its PDF + TeX source (unless already present)."""
        if self.has(paper.arxiv_id):
            existing = self.papers[paper.bare_id]
            self._merge_metadata(existing, paper)
            return existing
        if fetch:
            self._fetch_bodies(paper)
        self.papers[paper.bare_id] = paper
        return paper

    @staticmethod
    def _merge_metadata(existing: Paper, incoming: Paper) -> None:
        """Enrich graph-discovered stubs when a later search returns full metadata."""

        for name in ("title", "abstract", "published", "url"):
            old = getattr(existing, name, "") or ""
            new = getattr(incoming, name, "") or ""
            if new and (not old or len(new) > len(old)):
                setattr(existing, name, new)
        if incoming.authors:
            existing.authors = list(dict.fromkeys(existing.authors + incoming.authors))
        if incoming.categories:
            existing.categories = list(dict.fromkeys(existing.categories + incoming.categories))

    def _fetch_bodies(self, paper: Paper):
        bid = paper.bare_id
        pdir = self.root / "papers" / bid.replace("/", "_")
        pdir.mkdir(parents=True, exist_ok=True)
        # TeX source — the version the loop reads for equations
        src_report: dict = {}
        src = _download(f"https://arxiv.org/e-print/{bid}", report=src_report)
        tex = extract_tex_source(src) if src else ""
        if tex:
            tp = pdir / "source.tex"
            tp.write_text(tex, encoding="utf-8")
            paper.tex_path, paper.text, paper.body_source = str(tp), tex, "tex"
        elif src_report.get("error"):
            paper.fetch_errors.append(f"source: {src_report['error']}")
        elif src:
            paper.fetch_errors.append("source: downloaded archive contained no readable TeX")
        # PDF — for record + a text fallback when there is no usable source
        pdf_report: dict = {}
        pdf = _download(f"https://arxiv.org/pdf/{bid}.pdf", report=pdf_report)
        if pdf and pdf.startswith(b"%PDF"):
            pp = pdir / "paper.pdf"
            pp.write_bytes(pdf)
            paper.pdf_path = str(pp)
            if not paper.text:
                paper.text = extract_pdf_text(pp)
                if paper.text:
                    paper.body_source = "pdf"
        elif pdf:
            paper.fetch_errors.append("pdf: response did not contain a PDF")
        elif pdf_report.get("error"):
            paper.fetch_errors.append(f"pdf: {pdf_report['error']}")
        if not paper.text:                                   # last resort: the abstract
            paper.text = paper.abstract
            paper.body_source = "abstract" if paper.abstract else "missing"
        paper.fetch_errors = list(dict.fromkeys(paper.fetch_errors))
        paper.content_hash = hashlib.sha256(
            (paper.text or "").encode("utf-8", "ignore")).hexdigest() if paper.text else ""

    def build(self, query: str, k: int = 8, *, categories=None, on=None) -> list[Paper]:
        """Search arXiv (optionally restricted to ``categories``) and ingest the top
        ``k`` into the store."""
        from spiral.research import arxiv as arxiv_search
        added = []
        retrieval: dict = {}
        try:
            hits = arxiv_search(query, k=k, categories=categories, report=retrieval)
        except TypeError:
            # Compatibility for tests/plugins providing the older arxiv callable.
            hits = arxiv_search(query, k=k, categories=categories)
            retrieval = {"source": "arxiv", "source_ok": None,
                         "result_count": len(hits), "error": "source health unavailable"}
        updated = 0
        result_ids = []
        for hit in hits:
            aid = parse_arxiv_id(hit.url)
            if not aid:
                continue
            result_ids.append(Paper(arxiv_id=aid).bare_id)
            existed = self.has(aid)
            if on and not existed:
                on(aid)
            p = Paper(arxiv_id=aid, title=hit.title, url=hit.url,
                      abstract=hit.text, published=getattr(hit, "published", ""),
                      categories=list(getattr(hit, "categories", None) or []),
                      authors=[a.strip() for a in hit.snippet.split(",") if a.strip()])
            stored = self.add(p, fetch=not existed)
            if existed:
                updated += 1
                if not stored.tex_path and not stored.pdf_path and stored.body_source in {"", "abstract", "missing"}:
                    self._fetch_bodies(stored)
            else:
                added.append(stored)
        retrieval.update({
            "query": query,
            "categories": list(categories or []),
            "result_count": int(retrieval.get("result_count") or len(hits)),
            "result_ids": list(dict.fromkeys(result_ids)),
            "added_count": len(added),
            "updated_count": updated,
        })
        self.last_build_report = retrieval
        self.save()
        return added

    def graph_deepen(self, *, rounds: int = 2, min_cocite: int = 2, cap: int = 30,
                     seed_ids: list[str] | None = None, on=None) -> dict:
        """Deepen the corpus along the CITATION GRAPH until it saturates — the answer to
        'the model can't know what it's missing'. Each round snowballs from the current
        papers, pulls in the **co-citation holes** (foundational works many corpus papers
        cite but we lack — often decades old, unreachable by keyword search) plus a few
        recent papers that *cite* the corpus, then checks growth. Stops when a round adds
        little (<10 %) — saturation as an observable, not the model's guess."""
        from spiral.cite_graph import Edge, rank_holes, snowball
        report = {"rounds": 0, "added": 0, "holes": [], "saturated": False,
                  "round_reports": [], "errors": [],
                  "health": {"requests": 0, "successful_requests": 0,
                             "failed_requests": 0, "coverage_valid": False}}
        for _ in range(max(1, rounds)):
            have = set(self.papers)
            requested_seeds = [
                str(seed).replace("arXiv:", "").split("v")[0]
                for seed in (seed_ids or sorted(have))
            ]
            requested_seeds = [seed for seed in dict.fromkeys(requested_seeds) if seed in have]
            before = len(self.papers)
            trace: list[dict] = []
            health: dict = {}
            back, fwd, meta = snowball(
                requested_seeds, have, on=on, trace=trace, health=health)
            requests = int(health.get("requests") or 0)
            successes = int(health.get("successful_requests") or 0)
            attempted_seeds = int(health.get("seeds_attempted") or min(len(have), 30))
            successful_seeds = len(health.get("successful_seeds") or [])
            success_rate = successes / requests if requests else 0.0
            min_seed_coverage = min(3, attempted_seeds)
            health["success_rate"] = round(success_rate, 4)
            health["coverage_valid"] = bool(
                requests and success_rate >= 0.60 and successful_seeds >= min_seed_coverage)
            holes = rank_holes(back, min_cocite=min_cocite)
            round_report = {
                "seeds": requested_seeds,
                "eligible_seed_count": len(have),
                "edge_count": len(trace),
                "edges": trace,
                "holes": [{"id": h, "count": back[h],
                           "title": (meta.get(h) or Edge(arxiv_id=h)).title,
                           "citations": (meta.get(h) or Edge(arxiv_id=h)).citations}
                          for h in holes[:20]],
                "recent": [],
                "added": [],
                "errors": [],
                "saturated": False,
                "batch_frontier_closed": False,
                "health": health,
            }
            recent = sorted((i for i in fwd if i not in have),
                            key=lambda i: -((meta.get(i) or Edge(arxiv_id=i)).citations or 0))[:max(3, cap // 4)]
            round_report["recent"] = [
                {"id": i, "title": (meta.get(i) or Edge(arxiv_id=i)).title,
                 "citations": (meta.get(i) or Edge(arxiv_id=i)).citations}
                for i in recent
            ]
            candidates = []
            seen = set()
            for aid in holes + recent:
                if aid not in seen:
                    seen.add(aid)
                    candidates.append(aid)
            round_report["candidate_count"] = len(candidates)
            round_report["candidate_cap"] = cap
            round_report["frontier_truncated"] = len(candidates) > cap
            for aid in candidates[:cap]:
                e: Edge = meta.get(aid) or Edge(arxiv_id=aid)
                try:
                    p = self.add(Paper(arxiv_id=aid, title=e.title, authors=e.authors,
                                       url=f"https://arxiv.org/abs/{aid}"), fetch=True)
                except Exception as exc:
                    err = {"id": aid, "error": f"{type(exc).__name__}: {exc}"}
                    round_report["errors"].append(err)
                    report["errors"].append(err)
                    continue
                if on and p.bare_id not in have:
                    tag = f"hole ×{back.get(aid, 0)}" if aid in holes else "recent"
                    on(f"  + {aid} ({tag}) {e.title[:40]}")
                if p.bare_id not in have:
                    round_report["added"].append(p.bare_id)
            added = len(self.papers) - before
            report["rounds"] += 1
            report["added"] += added
            report["holes"] = [(h, back[h]) for h in holes[:10]]
            self.save()
            # Saturation is meaningful only when the graph source actually answered.  A
            # timeout/rate limit that yields no edges is an unavailable frontier, not a dry one.
            unresolved_holes = [aid for aid in holes if aid not in self.papers]
            round_report["unresolved_holes_after_round"] = unresolved_holes[:20]
            # This closes only the requested seed batch. Global closure is computed by
            # ``research_quality`` from the union of successfully closed batches. That
            # distinction prevents a 300-paper corpus from being called saturated after
            # Semantic Scholar happened to answer for the first 30 seeds.
            if (health["coverage_valid"] and not round_report["frontier_truncated"]
                    and not unresolved_holes):
                report["saturated"] = True
                round_report["saturated"] = True
                round_report["batch_frontier_closed"] = True
            if not health["coverage_valid"]:
                err = {
                    "error": "citation graph coverage invalid; saturation not evaluated",
                    "requests": requests,
                    "successful_requests": successes,
                    "successful_seeds": successful_seeds,
                }
                round_report["errors"].append(err)
                report["errors"].append(err)
            for key in ("requests", "successful_requests", "failed_requests"):
                report["health"][key] += int(health.get(key) or 0)
            report["health"]["coverage_valid"] = (
                report["health"]["coverage_valid"] or health["coverage_valid"])
            report["round_reports"].append(round_report)
            if round_report["saturated"]:
                break
        return report

    def summaries(self, limit: int = 40, chars: int = 1400,
                  ids: list[str] | None = None) -> str:
        """A compact, numbered digest of the corpus for a reasoning prompt.

        ``ids`` selects and ORDERS the papers shown (pass a topic-relevance ranking).
        Without it the first ``limit`` papers in insertion order are shown — which for
        a seeded, growing corpus means the seeds forever, so any caller judging
        coverage must pass a ranking rather than accept the default window."""
        if ids is not None:
            chosen = [self.papers[i] for i in ids if i in self.papers][:limit]
        else:
            chosen = list(self.papers.values())[:limit]
        out = []
        for i, p in enumerate(chosen, 1):
            body = display_body(p.text or p.abstract, chars)
            out.append(f"[{i}] {p.title} ({p.arxiv_id})\n{', '.join(p.authors[:4])}\n{body}\n")
        return "\n".join(out)

    def index(self, *, title_chars: int = 90) -> str:
        """One ``id · title`` line for EVERY paper in the store — the cheap complement
        to :meth:`summaries`. A judge deciding what is MISSING must be able to see what
        is PRESENT; a windowed digest alone reads as the whole corpus and produces
        confident 'X is absent' verdicts about papers sitting just outside the window."""
        return "\n".join(
            f"{p.bare_id} · {(p.title or '(untitled)').strip()[:title_chars]}"
            for p in self.papers.values())
