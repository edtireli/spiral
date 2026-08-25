"""Official, rate-limited academic metadata and prose sources.

Only the arXiv export API/download hosts and NCBI E-utilities are contacted.
Parsing functions are intentionally public and side-effect free so releases can
be tested against checked-in XML fixtures without touching the network.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Protocol

import httpx

from scripts.academic_finetune.text import clean_pdf_bytes, clean_pmc_xml, clean_tex_archive


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/"
ARXIV_EPRINT = "https://export.arxiv.org/e-print/"
ARXIV_PDF = "https://arxiv.org/pdf/"
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iso_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        match = re.search(r"(?:19|20)\d{2}", value)
        return date(int(match.group(0)), 1, 1) if match else None


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


@dataclass(frozen=True)
class SourceDocument:
    provider: str
    stratum: str
    source_id: str
    title: str
    authors: tuple[str, ...]
    published: str
    latest_version: str
    abstract: str
    body: str
    landing_url: str
    artifact_url: str
    metadata_endpoint: str
    content_endpoint: str
    query: str
    extraction: str
    license: str = ""
    metadata_revised: str = ""

    @property
    def document_id(self) -> str:
        return f"{self.provider}:{self.source_id}"

    @property
    def prose(self) -> str:
        parts = [part.strip() for part in (self.abstract, self.body) if part.strip()]
        return "\n\n".join(parts)

    @property
    def content_sha256(self) -> str:
        return sha256_text(self.prose)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authors"] = list(self.authors)
        value["content_sha256"] = self.content_sha256
        value["raw_record_sha256"] = sha256_text(canonical_json(value))
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceDocument":
        fields = {
            key: value[key]
            for key in cls.__dataclass_fields__
            if key in value
        }
        fields["authors"] = tuple(fields.get("authors", ()))
        return cls(**fields)


@dataclass(frozen=True)
class SourcePage:
    documents: tuple[SourceDocument, ...]
    next_offset: int
    exhausted: bool
    scanned: int


class AcademicSource(Protocol):
    stratum: str
    checkpoint_key: str

    def descriptor(self) -> Mapping[str, Any]: ...

    def fetch_page(self, offset: int, page_size: int) -> SourcePage: ...

    def hydrate(self, document: SourceDocument) -> SourceDocument: ...


class PoliteFetcher:
    """Small deterministic HTTP client with bounded retries and response sizes."""

    def __init__(
        self,
        *,
        user_agent: str,
        min_interval_seconds: float,
        timeout_seconds: float = 45.0,
        max_attempts: int = 4,
        max_bytes: int = 64 * 1024 * 1024,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("a descriptive User-Agent is required")
        self.user_agent = user_agent
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.max_bytes = max_bytes
        self._client = client or httpx.Client(follow_redirects=True)
        self._owns_client = client is None
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_started: float | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "PoliteFetcher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> bytes:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        allowed = {
            "export.arxiv.org",
            "arxiv.org",
            "eutils.ncbi.nlm.nih.gov",
        }
        if host not in allowed:
            raise ValueError(f"refusing non-official academic source host: {host}")

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            if self._last_started is not None:
                remaining = self.min_interval_seconds - (self._monotonic() - self._last_started)
                if remaining > 0:
                    self._sleep(remaining)
            self._last_started = self._monotonic()
            try:
                with self._client.stream(
                    "GET",
                    url,
                    params=params,
                    headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"},
                    timeout=self.timeout_seconds,
                ) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"temporary upstream status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError(f"academic source response exceeded {self.max_bytes} bytes")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                self._sleep(min(2 ** attempt, 8))
        assert last_error is not None
        raise last_error


def parse_arxiv_feed(
    xml_bytes: bytes,
    *,
    category: str,
    cutoff: date,
    query: str,
) -> tuple[tuple[SourceDocument, ...], int, int]:
    """Parse an Atom page and reject any record revised after the cutoff."""

    root = ET.fromstring(xml_bytes)
    total_text = root.findtext("opensearch:totalResults", default="0", namespaces=_ARXIV_NS)
    start_text = root.findtext("opensearch:startIndex", default="0", namespaces=_ARXIV_NS)
    total = int((total_text or "0").strip())
    start = int((start_text or "0").strip())
    documents: list[SourceDocument] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        updated = _iso_date(entry.findtext("atom:updated", default="", namespaces=_ARXIV_NS))
        published = _iso_date(entry.findtext("atom:published", default="", namespaces=_ARXIV_NS))
        # Using `updated`, not only first publication, prevents a 2022+ revision
        # from leaking through a pre-2022 paper identifier.
        if not updated or not published or updated > cutoff:
            continue
        raw_id = entry.findtext("atom:id", default="", namespaces=_ARXIV_NS).rstrip("/").split("/")[-1]
        if not raw_id:
            continue
        title = _text(entry.find("atom:title", _ARXIV_NS))
        abstract = _text(entry.find("atom:summary", _ARXIV_NS))
        if not title or not abstract:
            continue
        authors = tuple(
            name
            for author in entry.findall("atom:author", _ARXIV_NS)
            if (name := _text(author.find("atom:name", _ARXIV_NS)))
        )
        license_url = entry.findtext("arxiv:license", default="", namespaces=_ARXIV_NS)
        documents.append(
            SourceDocument(
                provider="arxiv",
                stratum=f"arxiv:{category}",
                source_id=raw_id,
                title=title,
                authors=authors,
                published=published.isoformat(),
                latest_version=updated.isoformat(),
                abstract=abstract,
                body="",
                landing_url=f"{ARXIV_ABS}{raw_id}",
                artifact_url="",
                metadata_endpoint=ARXIV_API,
                content_endpoint="",
                query=query,
                extraction="atom_abstract",
                license=(license_url or "").strip(),
            )
        )
    return tuple(documents), total, start


class ArxivSource:
    def __init__(
        self,
        category: str,
        *,
        cutoff: date,
        fetcher: PoliteFetcher,
        body_mode: str = "none",
        year: int | None = None,
    ) -> None:
        if category not in {"hep-th", "hep-ph"}:
            raise ValueError("the academic profile only permits arXiv hep-th and hep-ph")
        if body_mode not in {"none", "source", "pdf"}:
            raise ValueError("arXiv body mode must be none, source, or pdf")
        self.category = category
        self.stratum = f"arxiv:{category}"
        self.cutoff = cutoff
        self.fetcher = fetcher
        self.body_mode = body_mode
        self.year = year
        if year is not None and not 1991 <= year <= cutoff.year:
            raise ValueError("arXiv year must be between 1991 and the cutoff year")
        first = f"{year}01010000" if year is not None else "199101010000"
        last_year = year if year is not None else cutoff.year
        last_day = cutoff.strftime("%m%d") if last_year == cutoff.year else "1231"
        last = f"{last_year}{last_day}2359"
        self.query = f"cat:{category} AND submittedDate:[{first} TO {last}]"
        self.checkpoint_key = f"{self.stratum}:{year}" if year is not None else self.stratum

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "kind": "arxiv",
            "stratum": self.stratum,
            "category": self.category,
            "cutoff": self.cutoff.isoformat(),
            "body_mode": self.body_mode,
            "query": self.query,
            "year_quota": self.year,
        }

    def fetch_page(self, offset: int, page_size: int) -> SourcePage:
        payload = self.fetcher.get(
            ARXIV_API,
            {
                "search_query": self.query,
                "start": offset,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        documents, total, returned_start = parse_arxiv_feed(
            payload,
            category=self.category,
            cutoff=self.cutoff,
            query=self.query,
        )
        if returned_start != offset:
            raise ValueError(f"arXiv returned start index {returned_start}, expected {offset}")
        scanned = min(page_size, max(0, total - offset))
        next_offset = offset + scanned
        return SourcePage(documents, next_offset, next_offset >= total or scanned == 0, scanned)

    def hydrate(self, document: SourceDocument) -> SourceDocument:
        if self.body_mode == "none":
            return document
        if self.body_mode == "source":
            url = f"{ARXIV_EPRINT}{document.source_id}"
            body = clean_tex_archive(self.fetcher.get(url))
            return replace(
                document,
                body=body,
                artifact_url=url,
                content_endpoint=url,
                extraction="atom_abstract+arxiv_tex",
            )
        url = f"{ARXIV_PDF}{document.source_id}"
        body = clean_pdf_bytes(self.fetcher.get(url))
        return replace(
            document,
            body=body,
            artifact_url=url,
            content_endpoint=url,
            extraction="atom_abstract+pypdf",
        )


def _pubmed_date(article: ET.Element) -> tuple[date | None, date | None]:
    article_date = article.find(".//ArticleDate")
    pub_date = article_date or article.find(".//JournalIssue/PubDate")
    published: date | None = None
    if pub_date is not None:
        year_text = _text(pub_date.find("Year")) or _text(pub_date.find("MedlineDate"))
        year_match = re.search(r"(?:19|20)\d{2}", year_text)
        if year_match:
            year = int(year_match.group(0))
            month_text = _text(pub_date.find("Month"))
            month_lookup = {
                name: index
                for index, name in enumerate(
                    ("", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
                )
            }
            try:
                month = int(month_text) if month_text.isdigit() else month_lookup.get(month_text[:3].lower(), 1)
                day = int(_text(pub_date.find("Day")) or "1")
                published = date(year, month or 1, day)
            except ValueError:
                published = date(year, 1, 1)
    revised_node = article.find(".//DateRevised")
    revised: date | None = None
    if revised_node is not None:
        try:
            revised = date(
                int(_text(revised_node.find("Year"))),
                int(_text(revised_node.find("Month")) or "1"),
                int(_text(revised_node.find("Day")) or "1"),
            )
        except (TypeError, ValueError):
            revised = None
    return published, revised or published


def parse_pubmed_xml(
    xml_bytes: bytes,
    *,
    cutoff: date,
    query: str,
) -> tuple[SourceDocument, ...]:
    """Parse EFetch or an official PubMed baseline XML fragment."""

    root = ET.fromstring(xml_bytes)
    documents: list[SourceDocument] = []
    for record in root.findall(".//PubmedArticle"):
        citation = record.find("MedlineCitation")
        if citation is None:
            continue
        pmid = _text(citation.find("PMID"))
        article = citation.find("Article")
        if not pmid or article is None:
            continue
        published, metadata_revised = _pubmed_date(record)
        # MEDLINE DateRevised is an indexing/metadata maintenance timestamp, not
        # a new version of the authors' abstract. Applying the prose cutoff to it
        # excludes nearly every older article fetched from today's live index.
        if not published or published > cutoff:
            continue
        publication_types = {
            _text(node).casefold()
            for node in article.findall(".//PublicationTypeList/PublicationType")
        }
        excluded_types = {
            "published erratum",
            "retracted publication",
            "retraction of publication",
            "corrected and republished article",
            "expression of concern",
        }
        if publication_types & excluded_types:
            continue
        title = _text(article.find("ArticleTitle"))
        abstract_parts: list[str] = []
        for abstract in article.findall(".//Abstract/AbstractText"):
            value = _text(abstract)
            label = (abstract.attrib.get("Label") or "").strip()
            if value:
                abstract_parts.append(f"{label}: {value}" if label and not value.startswith(label) else value)
        abstract = "\n\n".join(abstract_parts)
        if not title or not abstract:
            continue
        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            personal = " ".join(
                part for part in (_text(author.find("ForeName")), _text(author.find("LastName"))) if part
            )
            if collective or personal:
                authors.append(collective or personal)
        pmc_id = ""
        for identifier in record.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if (identifier.attrib.get("IdType") or "").lower() == "pmc":
                pmc_id = _text(identifier)
                break
        documents.append(
            SourceDocument(
                provider="pubmed",
                stratum="pubmed",
                source_id=pmid,
                title=title,
                authors=tuple(authors),
                published=published.isoformat(),
                # PubMed exposes no abstract version history analogous to arXiv;
                # the electronic/article publication date is the content date.
                latest_version=published.isoformat(),
                abstract=abstract,
                body="",
                landing_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                artifact_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else "",
                metadata_endpoint=NCBI_EFETCH,
                content_endpoint="",
                query=query,
                extraction="pubmed_abstract",
                license="",
                metadata_revised=metadata_revised.isoformat() if metadata_revised else "",
            )
        )
    return tuple(documents)


def parse_pubmed_esearch(xml_bytes: bytes) -> tuple[tuple[str, ...], int, int]:
    root = ET.fromstring(xml_bytes)
    ids = tuple(_text(node) for node in root.findall("./IdList/Id") if _text(node))
    count = int(root.findtext("Count", default="0"))
    start = int(root.findtext("RetStart", default="0"))
    return ids, count, start


class PubMedSource:
    def __init__(
        self,
        query: str,
        *,
        cutoff: date,
        email: str,
        fetcher: PoliteFetcher,
        api_key: str = "",
        body_mode: str = "abstract",
        year: int | None = None,
    ) -> None:
        if not query.strip():
            raise ValueError("PubMed query cannot be empty")
        if "@" not in email:
            raise ValueError("NCBI requests require a contact email")
        if body_mode not in {"abstract", "pmc"}:
            raise ValueError("PubMed body mode must be abstract or pmc")
        self.stratum = "pubmed"
        self.query = query
        self.cutoff = cutoff
        self.email = email
        self.fetcher = fetcher
        self.api_key = api_key
        self.body_mode = body_mode
        self.year = year
        if year is not None and not 1800 <= year <= cutoff.year:
            raise ValueError("PubMed year must not exceed the cutoff year")
        self.checkpoint_key = f"{self.stratum}:{year}" if year is not None else self.stratum

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "kind": "pubmed",
            "stratum": self.stratum,
            "query": self.query,
            "cutoff": self.cutoff.isoformat(),
            "body_mode": self.body_mode,
            # Contact details are sent to NCBI but never persisted in a cache or
            # derived corpus artifact.
            "contact_sha256": sha256_text(self.email.strip().casefold()),
            "api_key_present": bool(self.api_key),
            "year_quota": self.year,
            "parser_revision": 2,
        }

    def _common(self) -> dict[str, str]:
        params = {"tool": "spiral_academic_corpus", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def fetch_page(self, offset: int, page_size: int) -> SourcePage:
        first_date = f"{self.year}/01/01" if self.year is not None else "1900/01/01"
        last_date = (
            f"{self.year}/12/31"
            if self.year is not None and self.year < self.cutoff.year
            else self.cutoff.isoformat().replace("-", "/")
        )
        dated_query = f"({self.query}) AND ({first_date}:{last_date}[Date - Publication])"
        search_params = self._common()
        search_params.update(
            {
                "db": "pubmed",
                "term": dated_query,
                "retmode": "xml",
                "retstart": str(offset),
                "retmax": str(page_size),
                "sort": "pub date",
            }
        )
        ids, total, returned_start = parse_pubmed_esearch(self.fetcher.get(NCBI_ESEARCH, search_params))
        if returned_start != offset:
            raise ValueError(f"PubMed returned start index {returned_start}, expected {offset}")
        documents: tuple[SourceDocument, ...] = ()
        if ids:
            fetch_params = self._common()
            fetch_params.update({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
            documents = parse_pubmed_xml(
                self.fetcher.get(NCBI_EFETCH, fetch_params),
                cutoff=self.cutoff,
                query=dated_query,
            )
        next_offset = offset + len(ids)
        return SourcePage(documents, next_offset, next_offset >= total or not ids, len(ids))

    def hydrate(self, document: SourceDocument) -> SourceDocument:
        if self.body_mode != "pmc" or not document.artifact_url:
            return document
        match = re.search(r"/articles/(PMC\d+)/", document.artifact_url, re.IGNORECASE)
        if not match:
            return document
        pmc_id = match.group(1).upper()
        params = self._common()
        params.update({"db": "pmc", "id": pmc_id, "retmode": "xml"})
        payload = self.fetcher.get(NCBI_EFETCH, params)
        body = clean_pmc_xml(payload)
        return replace(
            document,
            body=body,
            content_endpoint=f"{NCBI_EFETCH}?db=pmc&id={pmc_id}&retmode=xml",
            extraction="pubmed_abstract+pmc_xml",
        )


class PubMedBaselineSource:
    """Bounded local adapter for NLM's official PubMed baseline XML(.gz)."""

    stratum = "pubmed"
    checkpoint_key = "pubmed:baseline"

    def __init__(
        self,
        path: Path,
        *,
        cutoff: date,
        query_label: str = "pubmed-baseline",
        maximum_compressed_bytes: int = 2 * 1024 * 1024 * 1024,
        maximum_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self.path = path
        self.cutoff = cutoff
        self.query = query_label
        self.maximum_compressed_bytes = maximum_compressed_bytes
        self.maximum_uncompressed_bytes = maximum_uncompressed_bytes

    def _compressed_digest(self) -> str:
        size = self.path.stat().st_size
        if size > self.maximum_compressed_bytes:
            raise ValueError("PubMed baseline shard exceeds the compressed size limit")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def descriptor(self) -> Mapping[str, Any]:
        return {
            "kind": "pubmed-baseline",
            "stratum": self.stratum,
            "path_sha256": self._compressed_digest(),
            "cutoff": self.cutoff.isoformat(),
            "query": self.query,
            "parser_revision": 2,
        }

    def _page(self, offset: int, page_size: int) -> tuple[tuple[SourceDocument, ...], bool]:
        """Stream a baseline shard; never materialize the decompressed file."""

        self._compressed_digest()  # validates the on-disk bound before opening gzip
        raw: BinaryIO = self.path.open("rb")
        stream: BinaryIO = gzip.GzipFile(fileobj=raw) if self.path.suffix == ".gz" else raw
        bounded = _BoundedReader(stream, self.maximum_uncompressed_bytes)
        selected: list[SourceDocument] = []
        valid_index = 0
        exhausted = True
        try:
            for _event, element in ET.iterparse(bounded, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] != "PubmedArticle":
                    continue
                wrapped = b"<PubmedArticleSet>" + ET.tostring(element, encoding="utf-8") + b"</PubmedArticleSet>"
                parsed = parse_pubmed_xml(wrapped, cutoff=self.cutoff, query=self.query)
                element.clear()
                for document in parsed:
                    if valid_index >= offset and len(selected) < page_size:
                        selected.append(document)
                    elif valid_index >= offset + page_size:
                        exhausted = False
                        return tuple(selected), exhausted
                    valid_index += 1
        finally:
            bounded.close()
            if stream is not raw:
                stream.close()
            raw.close()
        return tuple(selected), exhausted

    def fetch_page(self, offset: int, page_size: int) -> SourcePage:
        page, exhausted = self._page(offset, page_size)
        next_offset = offset + len(page)
        return SourcePage(page, next_offset, exhausted, len(page))

    def hydrate(self, document: SourceDocument) -> SourceDocument:
        return document


class _BoundedReader:
    """Binary wrapper enforcing a decompressed-byte budget during iterparse."""

    def __init__(self, handle: BinaryIO, maximum_bytes: int) -> None:
        self.handle = handle
        self.maximum_bytes = maximum_bytes
        self.consumed = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.consumed += len(chunk)
        if self.consumed > self.maximum_bytes:
            raise ValueError("PubMed baseline shard exceeds the uncompressed size limit")
        return chunk

    def close(self) -> None:
        # Ownership remains with the caller, but ElementTree may call close.
        return None


def raw_record_sha256(document: SourceDocument) -> str:
    value = document.to_dict()
    return str(value["raw_record_sha256"])


def load_source_documents(paths: Iterable[Path]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = value.pop("raw_record_sha256", "")
        value.pop("content_sha256", None)
        document = SourceDocument.from_dict(value)
        if expected and raw_record_sha256(document) != expected:
            raise ValueError(f"cached source record hash mismatch: {path}")
        documents.append(document)
    return documents
