"""Research tools — live web + scientific knowledge for a local-first agent.

Two depths:
  search()  — fast: ranked results (web, optionally arXiv), no fetching.
  research()— deep: gather a corpus across web + arXiv + PubMed, follow a level of
              links, then synthesize a cited answer with a local thinking model.

The network is a TOOL, entered only through this module: GET-only, http(s)-only,
size-capped, tag-stripped. Fetched content is UNTRUSTED DATA — it becomes source
material for synthesis, never instructions, never executed.
"""
from __future__ import annotations

import html as htmllib
import re
import urllib.parse
from dataclasses import dataclass

import httpx

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh) spiral-research/0.2"}
MAX_BYTES = 800_000
MAX_TEXT = 12_000


@dataclass
class Hit:
    title: str
    url: str
    snippet: str = ""
    text: str = ""
    source: str = "web"   # web | arxiv | pubmed


# ---------------------------------------------------------------- primitives
def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav|form).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = htmllib.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def _get(url: str, timeout: float = 20.0) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as cl:
            r = cl.get(url)
            r.raise_for_status()
            return r.text[:MAX_BYTES]
    except Exception:
        return ""


def fetch(url: str, timeout: float = 20.0) -> str:
    """GET one page, return readable text (capped)."""
    raw = _get(url, timeout)
    return _strip_html(raw)[:MAX_TEXT] if raw else f"(fetch failed: {url})"


def _links(raw_html: str, want: set[str]) -> list[str]:
    """Outbound links whose href contains a query term — the 'follow' corpus."""
    out: list[str] = []
    for m in re.finditer(r'href="(https?://[^"#]+)"', raw_html):
        u = m.group(1)
        low = u.lower()
        if any(t in low for t in want) and "duckduckgo" not in low:
            out.append(u)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- sources
def search(query: str, k: int = 8, timeout: float = 20.0) -> list[Hit]:
    """DuckDuckGo HTML endpoint — no API key."""
    q = urllib.parse.quote_plus(query)
    body = _get(f"https://html.duckduckgo.com/html/?q={q}", timeout)
    if not body:
        return [Hit(title="(search failed)", url="")]
    hits: list[Hit] = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body, re.S):
        href, title = m.group(1), _strip_html(m.group(2))
        if "uddg=" in href:
            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
        if href.startswith("http"):
            hits.append(Hit(title=title, url=href))
        if len(hits) >= k:
            break
    for i, m in enumerate(re.finditer(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.S)):
        if i < len(hits):
            hits[i].snippet = _strip_html(m.group(1))[:300]
    return hits


def arxiv(query: str, k: int = 6, categories: list[str] | None = None,
          timeout: float = 25.0) -> list[Hit]:
    """arXiv Atom API — titles, authors, abstracts, no key.

    ``categories`` restricts the search to arXiv subject classes (``["math.NT"]``,
    ``["hep-th","hep-ph"]``, …). This matters: an unrestricted ``all:`` query for a
    term like *Ramanujan* returns mostly string-theory papers that merely cite it, so
    searching the RIGHT category is what keeps the corpus on-topic."""
    terms = f"all:{urllib.parse.quote_plus(query)}"
    if categories:
        cats = "+OR+".join(f"cat:{urllib.parse.quote_plus(c)}" for c in categories)
        sq = f"%28{cats}%29+AND+{terms}" if len(categories) > 1 else f"{cats}+AND+{terms}"
    else:
        sq = terms
    body = _get(f"http://export.arxiv.org/api/query?search_query={sq}&start=0&max_results={k}&sortBy=relevance", timeout)
    hits: list[Hit] = []
    for m in re.finditer(r"<entry>(.*?)</entry>", body, re.S):
        e = m.group(1)
        tm = re.search(r"<title>(.*?)</title>", e, re.S)
        sm = re.search(r"<summary>(.*?)</summary>", e, re.S)
        im = re.search(r"<id>(.*?)</id>", e)
        authors = re.findall(r"<name>(.*?)</name>", e)
        title = _strip_html(tm.group(1)) if tm else "(untitled)"
        hits.append(Hit(
            title=f"{title}", url=(im.group(1).strip() if im else ""),
            snippet=", ".join(authors[:5]), text=_strip_html(sm.group(1)) if sm else "", source="arxiv",
        ))
    return hits


def pubmed(query: str, k: int = 6, timeout: float = 25.0) -> list[Hit]:
    """PubMed via NCBI E-utilities — esearch for ids, efetch for abstracts."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    q = urllib.parse.quote_plus(query)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as cl:
            ids = cl.get(f"{base}/esearch.fcgi?db=pubmed&term={q}&retmax={k}&retmode=json").json()
            idlist = ids.get("esearchresult", {}).get("idlist", [])
            if not idlist:
                return []
            txt = cl.get(f"{base}/efetch.fcgi?db=pubmed&id={','.join(idlist)}&rettype=abstract&retmode=text").text
    except Exception:
        return []
    hits: list[Hit] = []
    for rec, pmid in zip(re.split(r"\n\n\n+", txt.strip()), idlist):
        rec = rec.strip()
        if not rec:
            continue
        title = next((ln.strip() for ln in rec.splitlines() if len(ln.strip()) > 30), rec[:100])
        hits.append(Hit(title=title[:160], url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        text=rec[:MAX_TEXT], source="pubmed"))
    return hits


# ---------------------------------------------------------------- corpus + synthesis
def gather(question: str, k: int = 6, sci: bool = False, web: bool = True,
           follow: int = 0, on=None) -> list[Hit]:
    """Collect sources, fetch web bodies, optionally follow a level of links."""
    hits: list[Hit] = []
    if web:
        hits += search(question, k=k)
    if sci:
        hits += arxiv(question, k=max(4, k // 2))
        hits += pubmed(question, k=max(4, k // 2))
    for h in hits:
        if h.source == "web" and h.url and not h.text:
            if on:
                on(h.url)
            h.text = fetch(h.url)
    if follow and web:
        want = {t for t in re.findall(r"[a-z]{4,}", question.lower())}
        extra: list[str] = []
        for h in hits[:2]:
            raw = _get(h.url)
            for u in _links(raw, want)[:2]:
                if u not in {x.url for x in hits} and u not in extra:
                    extra.append(u)
        for u in extra[:3]:
            if on:
                on(u)
            hits.append(Hit(title=u, url=u, text=fetch(u), source="web"))
    return [h for h in hits if h.text and not h.text.startswith("(fetch failed")]


def synthesize(question: str, hits: list[Hit], cfg=None, ol=None, deep: bool = False, on=None):
    """Write a cited answer from the numbered corpus using a local thinking model."""
    from spiral.config import Config
    from spiral.llm import Ollama
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url, providers=cfg.providers)

    per = 6000 if deep else 2500
    budget = 42_000 if deep else 15_000
    corpus, used = [], []
    for i, h in enumerate(hits, 1):
        block = f"[{i}] {h.title} ({h.source})\n{h.url}\n{h.text[:per]}\n"
        if sum(len(c) for c in corpus) + len(block) > budget:
            break
        corpus.append(block)
        used.append((i, h))
    system = (
        "You are a rigorous research assistant. Write an accurate, well-structured answer to the "
        "QUESTION grounded in the numbered SOURCES, citing them inline as [n]. You may use "
        "well-established textbook knowledge for standard facts (equations, definitions) even if "
        "not in the sources, but flag anything genuinely uncertain or contested. If the question "
        "asks for math, output correct LaTeX. End with a 'Sources' section listing the [n] you cited."
    )
    user = f"QUESTION: {question}\n\nSOURCES:\n" + "\n".join(corpus)
    res = ol.chat(
        cfg.planner.name, [{"role": "system", "content": system}, {"role": "user", "content": user}],
        think=deep, num_predict=cfg.planner_max_tokens, num_ctx=cfg.planner.num_ctx,
        keep_alive=cfg.keep_alive, temperature=0.3,
        on_delta=(lambda kind, piece: on()) if on else None,
    )
    return res.text, [h for _, h in used], res


def research(query: str, k: int = 3) -> list[Hit]:
    """Back-compat: search then read top-k pages (used by the worker's tools)."""
    hits = search(query, k=k)
    for h in hits:
        if h.url:
            h.text = fetch(h.url)
    return hits
