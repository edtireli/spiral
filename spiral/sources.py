"""Literature source adapters — the corpus's channels beyond arXiv.

Physics/math live on arXiv; neuroscience, biology and medicine live on bioRxiv,
medRxiv, PubMed, PMC and the wider DOI ecosystem. Each adapter normalises its
provider into one :class:`Record` shape so the corpus never learns provider quirks.

Design rules, learned the hard way from the arXiv-only era:

* Every record carries a **namespaced uid** (``arxiv:2401.001``, ``doi:10.1101/…``,
  ``pmid:34567890``, ``pmc:PMC123``) so dedup and citation-graph keys stay unambiguous
  across providers. A bioRxiv preprint and its published version share a DOI, so the
  DOI is the join key whenever present.
* Adapters accept an injected ``fetch_json`` / ``fetch_text`` so the whole module is
  unit-testable offline — no network in tests, ever.
* A provider being down degrades to an empty list plus a health ``report`` (mirroring
  ``research.arxiv(report=)``); it never raises into the loop.
* Europe PMC is the workhorse: one query surfaces bioRxiv, medRxiv, PMC and MEDLINE
  with DOIs and open-access full-text links, so it covers most of "all of the above".
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Any

MAX_TEXT = 40_000
_UA = {"User-Agent": "spiral-research/0.4 (research corpus; mailto:research@spiral.local)"}
# Unpaywall and Crossref politely ask for a contact; a mailbox that need not receive
# mail satisfies the etiquette without leaking a real address.
_CONTACT = "research@spiral.local"


@dataclass
class Record:
    """One normalised source record. ``uid`` is the namespaced dedup key; ``doi`` is
    the cross-provider join key when present; ``full_text_url`` (+ ``full_text_kind``)
    is how the corpus later fetches a body."""
    uid: str
    title: str
    source: str                       # arxiv | biorxiv | medrxiv | pmc | pubmed | crossref
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    doi: str = ""
    published: str = ""
    url: str = ""
    full_text_url: str = ""
    full_text_kind: str = ""          # jats | pdf | tex | ""
    subjects: list[str] = field(default_factory=list)
    venue: str = ""


# ── injectable transport (real by default, faked in tests) ────────────────────
def _real_json(url: str, timeout: float) -> Any:
    import httpx
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA,
                      trust_env=False) as cl:
        r = cl.get(url)
        r.raise_for_status()
        return r.json()


def _real_text(url: str, timeout: float) -> str:
    import httpx
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=_UA,
                      trust_env=False) as cl:
        r = cl.get(url)
        r.raise_for_status()
        return r.text


def _norm_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def _clean(s: str, limit: int = MAX_TEXT) -> str:
    return " ".join((s or "").split())[:limit]


# ── Europe PMC: bioRxiv + medRxiv + PMC + MEDLINE in one query ────────────────
def europepmc(query: str, k: int = 6, *, sources: list[str] | None = None,
              open_access_only: bool = False, timeout: float = 25.0,
              report: dict | None = None,
              fetch_json: Callable[[str, float], Any] | None = None) -> list[Record]:
    """Europe PMC REST search. ``sources`` filters by EPMC source code:
    ``PPR`` (preprints incl. bioRxiv/medRxiv), ``MED`` (MEDLINE/PubMed), ``PMC``
    (full-text OA), ``AGR``, ``CBA``… Default: all. Returns records with DOIs and,
    where OA, a JATS full-text URL."""
    fetch_json = fetch_json or _real_json
    q = query
    if sources:
        q = f"({query}) AND (" + " OR ".join(f"SRC:{s}" for s in sources) + ")"
    if open_access_only:
        q = f"({q}) AND OPEN_ACCESS:Y"
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           + urllib.parse.urlencode({
               "query": q, "format": "json", "pageSize": k,
               "resultType": "core"}))
    rep = {"source": "europepmc", "query": query, "url": url,
           "sources": list(sources or []), "source_ok": False, "error": ""}
    try:
        data = fetch_json(url, timeout)
    except Exception as exc:
        rep["error"] = f"{type(exc).__name__}: {exc}"
        if report is not None:
            report.update(rep)
        return []
    results = ((data or {}).get("resultList") or {}).get("result") or []
    out: list[Record] = []
    for r in results:
        doi = _norm_doi(r.get("doi") or "")
        pmid = str(r.get("pmid") or "").strip()
        pmcid = str(r.get("pmcid") or "").strip()
        src = (r.get("source") or "").upper()
        # preprint publisher tells bioRxiv from medRxiv apart within SRC:PPR
        pub = (r.get("publisher") or "") + " " + (r.get("bookOrReportDetails") or {}).get("publisher", "") \
            if isinstance(r.get("bookOrReportDetails"), dict) else (r.get("publisher") or "")
        low = pub.lower()
        if src == "PPR" and "medrxiv" in (doi + low):
            source = "medrxiv"
        elif src == "PPR" and ("biorxiv" in (doi + low) or "cold spring harbor" in low):
            source = "biorxiv"
        elif src == "PPR":
            source = "preprint"
        elif src == "PMC" or pmcid:
            source = "pmc"
        else:
            source = "pubmed"
        uid = (f"doi:{doi}" if doi else f"pmc:{pmcid}" if pmcid
               else f"pmid:{pmid}" if pmid else f"epmc:{src}:{r.get('id')}")
        # OA full text: PMC open-access articles are served as JATS XML by NCBI's
        # efetch (db=pmc). The Europe PMC /{src}/{id}/fullTextXML endpoint 404s across
        # every era, and the advertised PDF links are interstitials/time out — verified
        # against the live APIs — so efetch is the one reliable full-text route.
        ft_url, ft_kind = "", ""
        if pmcid and (r.get("inEPMC") == "Y" or r.get("isOpenAccess") == "Y"):
            num = re.sub(r"\D", "", pmcid)
            if num:
                ft_url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                          f"db=pmc&id={num}&rettype=full&retmode=xml")
                ft_kind = "jats"
        authors = []
        for a in ((r.get("authorList") or {}).get("author") or []):
            name = a.get("fullName") or " ".join(
                x for x in (a.get("firstName"), a.get("lastName")) if x)
            if name:
                authors.append(name)
        out.append(Record(
            uid=uid, title=_clean(r.get("title") or "", 400) or "(untitled)",
            source=source, authors=authors[:20],
            abstract=_clean(r.get("abstractText") or ""),
            doi=doi, published=str(r.get("firstPublicationDate")
                                   or r.get("pubYear") or ""),
            url=(f"https://doi.org/{doi}" if doi
                 else f"https://europepmc.org/article/{src}/{r.get('id')}"),
            full_text_url=ft_url, full_text_kind=ft_kind,
            subjects=[t for t in [(r.get("pubTypeList") or {}).get("pubType")]
                      if isinstance(t, str)],
            venue=_clean(r.get("journalTitle") or pub, 160)))
    rep.update({"source_ok": True, "result_count": len(out),
                "full_text_available": sum(1 for r in out if r.full_text_kind)})
    if report is not None:
        report.update(rep)
    return out


def biorxiv(query: str, k: int = 6, **kw) -> list[Record]:
    """bioRxiv preprints via Europe PMC (SRC:PPR filtered to bioRxiv)."""
    kw.setdefault("sources", ["PPR"])
    recs = europepmc(query, k=k * 3, **kw)
    return [r for r in recs if r.source == "biorxiv"][:k]


def medrxiv(query: str, k: int = 6, **kw) -> list[Record]:
    """medRxiv clinical preprints via Europe PMC (SRC:PPR filtered to medRxiv)."""
    kw.setdefault("sources", ["PPR"])
    recs = europepmc(query, k=k * 3, **kw)
    return [r for r in recs if r.source == "medrxiv"][:k]


# ── NCBI PubMed (E-utilities) ─────────────────────────────────────────────────
def pubmed(query: str, k: int = 6, *, timeout: float = 25.0,
           report: dict | None = None,
           fetch_json: Callable[[str, float], Any] | None = None,
           fetch_text: Callable[[str, float], str] | None = None) -> list[Record]:
    """PubMed via NCBI E-utilities: esearch → esummary (metadata) + efetch (abstracts).
    Returns records keyed ``pmid:…`` with DOIs when the summary carries one."""
    fetch_json = fetch_json or _real_json
    fetch_text = fetch_text or _real_text
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    q = urllib.parse.quote_plus(query)
    rep = {"source": "pubmed", "query": query, "source_ok": False, "error": ""}
    try:
        ids = fetch_json(f"{base}/esearch.fcgi?db=pubmed&term={q}"
                         f"&retmax={k}&retmode=json", timeout)
        idlist = (((ids or {}).get("esearchresult") or {}).get("idlist")) or []
        if not idlist:
            rep.update({"source_ok": True, "result_count": 0})
            if report is not None:
                report.update(rep)
            return []
        summ = fetch_json(f"{base}/esummary.fcgi?db=pubmed&id={','.join(idlist)}"
                          "&retmode=json", timeout)
        abstracts = fetch_text(f"{base}/efetch.fcgi?db=pubmed&id={','.join(idlist)}"
                               "&rettype=abstract&retmode=text", timeout)
    except Exception as exc:
        rep["error"] = f"{type(exc).__name__}: {exc}"
        if report is not None:
            report.update(rep)
        return []
    abs_by_order = [a.strip() for a in re.split(r"\n\n\n+", (abstracts or "").strip())]
    result = (summ or {}).get("result") or {}
    out: list[Record] = []
    for i, pmid in enumerate(idlist):
        meta = result.get(pmid) or {}
        doi = ""
        for aid in (meta.get("articleids") or []):
            if aid.get("idtype") == "doi":
                doi = _norm_doi(aid.get("value") or "")
        authors = [a.get("name") for a in (meta.get("authors") or []) if a.get("name")]
        abstract = abs_by_order[i] if i < len(abs_by_order) else ""
        title = _clean(meta.get("title") or "", 400) or (
            next((ln.strip() for ln in abstract.splitlines()
                  if len(ln.strip()) > 30), abstract[:100]))
        out.append(Record(
            uid=f"doi:{doi}" if doi else f"pmid:{pmid}", title=title or "(untitled)",
            source="pubmed", authors=authors[:20], abstract=_clean(abstract),
            doi=doi, published=str(meta.get("pubdate") or ""),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            venue=_clean(meta.get("fulljournalname") or meta.get("source") or "", 160)))
    rep.update({"source_ok": True, "result_count": len(out)})
    if report is not None:
        report.update(rep)
    return out


# ── Crossref (DOI metadata across every publisher) + Unpaywall (OA PDF) ────────
def crossref(query: str, k: int = 6, *, timeout: float = 25.0,
             report: dict | None = None,
             fetch_json: Callable[[str, float], Any] | None = None) -> list[Record]:
    """Crossref works search — DOI-native metadata across all publishers (chemistry,
    clinical journals, anything with a DOI). Full text is resolved lazily via
    :func:`unpaywall` only when a body is actually needed."""
    fetch_json = fetch_json or _real_json
    url = ("https://api.crossref.org/works?"
           + urllib.parse.urlencode({"query": query, "rows": k, "mailto": _CONTACT}))
    rep = {"source": "crossref", "query": query, "url": url,
           "source_ok": False, "error": ""}
    try:
        data = fetch_json(url, timeout)
    except Exception as exc:
        rep["error"] = f"{type(exc).__name__}: {exc}"
        if report is not None:
            report.update(rep)
        return []
    items = ((data or {}).get("message") or {}).get("items") or []
    out: list[Record] = []
    for it in items:
        doi = _norm_doi(it.get("DOI") or "")
        if not doi:
            continue
        authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
                   for a in (it.get("author") or [])]
        title = _clean(" ".join(it.get("title") or []) or "", 400)
        parts = ((it.get("published") or {}).get("date-parts") or [[None]])[0]
        year = str(parts[0]) if parts and parts[0] else ""
        # Crossref sometimes carries a direct full-text link
        ft_url, ft_kind = "", ""
        for link in (it.get("link") or []):
            if link.get("content-type") in ("application/pdf", "unspecified"):
                ft_url, ft_kind = link.get("URL", ""), "pdf"
                break
        out.append(Record(
            uid=f"doi:{doi}", title=title or "(untitled)", source="crossref",
            authors=[a for a in authors if a][:20],
            abstract=_clean(re.sub(r"<[^>]+>", " ", it.get("abstract") or "")),
            doi=doi, published=year,
            url=f"https://doi.org/{doi}", full_text_url=ft_url, full_text_kind=ft_kind,
            subjects=list(it.get("subject") or []),
            venue=_clean(" ".join(it.get("container-title") or []), 160)))
    rep.update({"source_ok": True, "result_count": len(out)})
    if report is not None:
        report.update(rep)
    return out


def unpaywall(doi: str, *, timeout: float = 25.0,
              fetch_json: Callable[[str, float], Any] | None = None) -> tuple[str, str]:
    """Resolve an open-access PDF for a DOI via Unpaywall. Returns ``(url, kind)`` or
    ``("", "")``. No key required — only a contact email."""
    fetch_json = fetch_json or _real_json
    doi = _norm_doi(doi)
    if not doi:
        return "", ""
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={_CONTACT}"
    try:
        data = fetch_json(url, timeout)
    except Exception:
        return "", ""
    loc = (data or {}).get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf") or loc.get("url")
    return (pdf, "pdf") if pdf else ("", "")


# ── JATS full text → plain text ───────────────────────────────────────────────
def jats_to_text(xml: str, *, limit: int = MAX_TEXT) -> str:
    """Flatten a JATS/NLM full-text XML body to readable text: title, abstract, then
    each <body> section with its heading. Math (<tex-math>, <mml:*>) is preserved
    inline so quantitative claims stay checkable. References are dropped (a corpus was
    once fed nothing but bibliographies)."""
    import xml.etree.ElementTree as ET

    def _localname(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    try:
        root = ET.fromstring(xml)
    except Exception:
        # last resort: strip tags
        return _clean(re.sub(r"<[^>]+>", " ", xml or ""), limit)

    parts: list[str] = []

    def walk(el, in_body: bool) -> None:
        name = _localname(el.tag)
        if name in ("ref-list", "back", "fn-group", "table-wrap", "fig"):
            return
        if name == "tex-math" and (el.text or "").strip():
            parts.append(f" ${el.text.strip()}$ ")
            return
        if name in ("title", "label") and (el.text or "").strip():
            parts.append("\n\n" + el.text.strip() + " ")   # heading, then children
        elif el.text and el.text.strip():
            parts.append(el.text)
        for child in el:
            walk(child, in_body)
            if child.tail and child.tail.strip():
                parts.append(child.tail)

    # abstract first, then the article body
    for el in root.iter():
        if _localname(el.tag) in ("abstract", "body"):
            walk(el, True)
    text = re.sub(r"[ \t]+", " ", "".join(parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit]
