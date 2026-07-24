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
import ipaddress
import re
import socket
import urllib.parse
import xml.etree.ElementTree as ET
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
    published: str = ""
    categories: list[str] | None = None


# ---------------------------------------------------------------- primitives
def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav|form).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = htmllib.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def _public_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        if parsed.hostname.lower() in {"localhost", "localhost.localdomain"}:
            return False
        addresses = {
            item[4][0] for item in socket.getaddrinfo(
                parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
        if not addresses:
            return False
        return all(
            not (
                (ip := ipaddress.ip_address(address)).is_private
                or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified
            )
            for address in addresses
        )
    except Exception:
        return False


def _get(url: str, timeout: float = 20.0) -> str:
    if not _public_url(url):
        return ""
    try:
        with httpx.Client(
                timeout=timeout, follow_redirects=False, headers=_UA,
                trust_env=False) as cl:
            current = url
            for _redirect in range(6):
                if not _public_url(current):
                    return ""
                with cl.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return ""
                        current = urllib.parse.urljoin(current, location)
                        continue
                    response.raise_for_status()
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        remaining = MAX_BYTES - size
                        if remaining <= 0:
                            break
                        chunks.append(chunk[:remaining])
                        size += min(len(chunk), remaining)
                    encoding = response.encoding or "utf-8"
                    return b"".join(chunks).decode(encoding, errors="replace")
            return ""
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


def arxiv_terms(query: str) -> str:
    """Build the ``all:`` clause for an arXiv API query.

    A short query stays an exact phrase (``all:"gregory laflamme"`` — precision for
    names). A longer keyword query becomes an AND of individual terms: arXiv treats
    ``all:"kodama ishibashi master equations higher dimensional black holes"`` as a
    verbatim 8-word phrase, which matches essentially nothing — a whole research run
    once stalled because every multi-word search silently returned zero results."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", query)
    if len(words) <= 2:
        return f'all:"{query}"'
    return " AND ".join(f"all:{w}" for w in words[:8])


def arxiv(query: str, k: int = 6, categories: list[str] | None = None,
          timeout: float = 25.0, report: dict | None = None) -> list[Hit]:
    """arXiv Atom API — titles, authors, abstracts, no key.

    ``categories`` restricts the search to arXiv subject classes (``["math.NT"]``,
    ``["hep-th","hep-ph"]``, …). This matters: an unrestricted ``all:`` query for a
    term like *Ramanujan* returns mostly string-theory papers that merely cite it, so
    searching the RIGHT category is what keeps the corpus on-topic."""
    terms = arxiv_terms(query)
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        cats = f"({cats})" if len(categories) > 1 else cats
        sq = f"{cats} AND ({terms})" if " AND " in terms else f"{cats} AND {terms}"
    else:
        sq = terms
    params = urllib.parse.urlencode({
        "search_query": sq,
        "start": 0,
        "max_results": k,
        "sortBy": "relevance",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    body = _get(url, timeout)
    if report is not None:
        report.update({
            "source": "arxiv",
            "source_ok": bool(body),
            "query": query,
            "categories": list(categories or []),
            "url": url,
            "error": "" if body else "arXiv API returned no response",
        })
    hits: list[Hit] = []
    if body:
        try:
            root = ET.fromstring(body)
            atom = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", atom):
                title = " ".join((entry.findtext("a:title", default="", namespaces=atom) or "").split())
                abstract = " ".join((entry.findtext("a:summary", default="", namespaces=atom) or "").split())
                identifier = (entry.findtext("a:id", default="", namespaces=atom) or "").strip()
                published = (entry.findtext("a:published", default="", namespaces=atom) or "").strip()
                authors = [
                    " ".join((node.findtext("a:name", default="", namespaces=atom) or "").split())
                    for node in entry.findall("a:author", atom)
                ]
                cats = [node.attrib.get("term", "") for node in entry.findall("a:category", atom)]
                hits.append(Hit(
                    title=title or "(untitled)", url=identifier,
                    snippet=", ".join(a for a in authors[:8] if a), text=abstract,
                    source="arxiv", published=published,
                    categories=[c for c in cats if c],
                ))
        except ET.ParseError as exc:
            if report is not None:
                report["source_ok"] = False
                report["error"] = f"invalid arXiv XML: {exc}"
    if report is not None:
        report["result_count"] = len(hits)
    return hits


def pubmed(query: str, k: int = 6, timeout: float = 25.0) -> list[Hit]:
    """PubMed via NCBI E-utilities — esearch for ids, efetch for abstracts."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    q = urllib.parse.quote_plus(query)
    try:
        with httpx.Client(
                timeout=timeout, follow_redirects=True, headers=_UA,
                trust_env=False) as cl:
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
