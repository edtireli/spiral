"""The researcher's hands — live web knowledge for a local-first agent.

Skills are frozen knowledge; the web is live knowledge (current versions, API
docs, unfamiliar errors). The models stay local — the network is a TOOL, entered
only through this module: GET-only, http(s)-only, size-capped, tag-stripped.
The worker's shell denylist still blocks curl/wget; this is the one door.

Safety: fetched content is UNTRUSTED DATA. It is summarized into briefs under
.spiral/research/ and fed to models as reference material — never treated as
instructions and never executed.
"""
from __future__ import annotations

import html as htmllib
import re
import urllib.parse
from dataclasses import dataclass

import httpx

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh) spiral-research/0.1"}
MAX_BYTES = 500_000
MAX_TEXT = 8_000


@dataclass
class Hit:
    title: str
    url: str
    snippet: str = ""
    text: str = ""


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|header|footer|nav).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = htmllib.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


def fetch(url: str, timeout: float = 20.0) -> str:
    """GET one page, return readable text (capped). https/http only."""
    if not url.startswith(("http://", "https://")):
        return f"(refused non-http url: {url})"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as cl:
            r = cl.get(url)
            r.raise_for_status()
            return _strip_html(r.text[:MAX_BYTES])[:MAX_TEXT]
    except Exception as e:
        return f"(fetch failed: {e})"


def search(query: str, k: int = 5, timeout: float = 20.0) -> list[Hit]:
    """DuckDuckGo HTML endpoint — no API key, parsed with regex."""
    q = urllib.parse.quote_plus(query)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA) as cl:
            r = cl.get(f"https://html.duckduckgo.com/html/?q={q}")
            r.raise_for_status()
            body = r.text
    except Exception as e:
        return [Hit(title="(search failed)", url="", snippet=str(e))]

    hits: list[Hit] = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body, re.S
    ):
        href, title = m.group(1), _strip_html(m.group(2))
        if "uddg=" in href:  # ddg redirect wrapper
            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
        if href.startswith("http"):
            hits.append(Hit(title=title, url=href))
        if len(hits) >= k:
            break
    for i, m in enumerate(re.finditer(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.S)):
        if i < len(hits):
            hits[i].snippet = _strip_html(m.group(1))[:300]
    return hits


def research(query: str, k: int = 3) -> list[Hit]:
    """Search, then read the top-k pages. Returns hits with page text attached."""
    hits = search(query, k=k)
    for h in hits:
        if h.url:
            h.text = fetch(h.url)
    return hits
