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
import io
import json
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
def _download(url: str, timeout: float = 40.0) -> bytes | None:
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
            return r.content[:MAX_DOWNLOAD]
    except Exception:
        return None


# ── TeX extraction ───────────────────────────────────────────────────────────
def _clean_tex(tex: str) -> str:
    """Strip TeX comments and collapse whitespace — keep the math and prose."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)                     # line comments (not \%)
    return re.sub(r"\n{3,}", "\n\n", tex).strip()


def extract_tex_source(raw: bytes) -> str:
    """arXiv e-prints are a gzipped tar of .tex/.bbl (or a single gzipped .tex, or a
    bare .tex). Concatenate every .tex/.bbl found — the main file plus its inputs."""
    if not raw:
        return ""
    # try tar.gz (the common multi-file case)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            parts = []
            for m in tar.getmembers():
                if m.isfile() and m.name.lower().endswith((".tex", ".bbl")):
                    f = tar.extractfile(m)
                    if f:
                        parts.append(f.read().decode("utf-8", "ignore"))
            if parts:
                return _clean_tex("\n\n".join(parts))
    except (tarfile.TarError, OSError):
        pass
    # single gzipped file
    try:
        return _clean_tex(gzip.decompress(raw).decode("utf-8", "ignore"))
    except (OSError, gzip.BadGzipFile):
        pass
    # bare text
    try:
        s = raw.decode("utf-8", "ignore")
        if "\\" in s or "\\begin" in s:
            return _clean_tex(s)
    except Exception:
        pass
    return ""


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF (best-effort; TeX source is always preferred)."""
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages).strip()
    except Exception:
        return ""


# ── the store ────────────────────────────────────────────────────────────────
@dataclass
class Corpus:
    """A local store of papers under ``root``, deduplicated by bare arXiv id."""
    root: Path
    papers: dict[str, Paper] = field(default_factory=dict)

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
        self._manifest().write_text(json.dumps(
            {"papers": [asdict(p) for p in self.papers.values()]}, indent=2))

    def has(self, arxiv_id: str) -> bool:
        return Paper(arxiv_id=arxiv_id).bare_id in self.papers

    def add(self, paper: Paper, *, fetch: bool = True) -> Paper:
        """Add a paper, downloading its PDF + TeX source (unless already present)."""
        if self.has(paper.arxiv_id):
            return self.papers[paper.bare_id]
        if fetch:
            self._fetch_bodies(paper)
        self.papers[paper.bare_id] = paper
        return paper

    def _fetch_bodies(self, paper: Paper):
        bid = paper.bare_id
        pdir = self.root / "papers" / bid.replace("/", "_")
        pdir.mkdir(parents=True, exist_ok=True)
        # TeX source — the version the loop reads for equations
        src = _download(f"https://arxiv.org/e-print/{bid}")
        tex = extract_tex_source(src) if src else ""
        if tex:
            tp = pdir / "source.tex"
            tp.write_text(tex, encoding="utf-8")
            paper.tex_path, paper.text = str(tp), tex
        # PDF — for record + a text fallback when there is no usable source
        pdf = _download(f"https://arxiv.org/pdf/{bid}.pdf")
        if pdf:
            pp = pdir / "paper.pdf"
            pp.write_bytes(pdf)
            paper.pdf_path = str(pp)
            if not paper.text:
                paper.text = extract_pdf_text(pp)
        if not paper.text:                                   # last resort: the abstract
            paper.text = paper.abstract

    def build(self, query: str, k: int = 8, *, on=None) -> list[Paper]:
        """Search arXiv for ``query`` and ingest the top ``k`` into the store."""
        from spiral.research import arxiv as arxiv_search
        added = []
        for hit in arxiv_search(query, k=k):
            aid = parse_arxiv_id(hit.url)
            if not aid or self.has(aid):
                continue
            if on:
                on(aid)
            p = Paper(arxiv_id=aid, title=hit.title, url=hit.url,
                      abstract=hit.text, authors=[a.strip() for a in hit.snippet.split(",") if a.strip()])
            added.append(self.add(p))
        self.save()
        return added

    def summaries(self, limit: int = 40, chars: int = 1400) -> str:
        """A compact, numbered digest of the corpus for a reasoning prompt."""
        out = []
        for i, p in enumerate(list(self.papers.values())[:limit], 1):
            body = (p.text or p.abstract)[:chars]
            out.append(f"[{i}] {p.title} ({p.arxiv_id})\n{', '.join(p.authors[:4])}\n{body}\n")
        return "\n".join(out)
