"""Typed public scientific-data discovery and acquisition for Research.

The research model may describe *what* public data it needs, but it does not get a
networked shell. This broker owns the network boundary and makes acquisition an
auditable scientific operation:

* search catalog metadata before transferring data;
* accept only known public hosts and credential-free GET/HEAD/GraphQL requests;
* resolve the complete selected file list and byte count before downloading;
* enforce a per-run cap, a free-disk reserve, resumable ``.part`` files and hashes;
* pin source id, release/version, licence, citation and retrieval URLs;
* lock an analysis plan and spatial/cross-modal alignment contract before execution;
* expose immutable cached data to certificates through hard links.

It is intentionally not a general downloader. Unsupported repositories can still be
added as typed adapters without weakening the network or provenance boundary.
"""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"
ALLEN_PRODUCTS = (
    "https://api.brain-map.org/api/v2/data/query.json?"
    "criteria=model::Product,rma::options[num_rows$eq'all']"
)
ZENODO_API = "https://zenodo.org/api/records"
PYPI_NEUROMAPS = "https://pypi.org/pypi/neuromaps/json"

_SOURCE_HOSTS = {
    "openneuro": {
        "openneuro.org", "s3.amazonaws.com", "openneuro.org.s3.amazonaws.com",
    },
    "allen": {
        "api.brain-map.org", "download.alleninstitute.org",
        "alleninstitute.org", "visualcodingdata.s3-us-west-2.amazonaws.com",
        "allen-brain-observatory.s3.us-west-2.amazonaws.com",
    },
    "zenodo": {"zenodo.org"},
    "neuromaps": {
        "pypi.org", "files.pythonhosted.org", "files.osf.io", "osf.io",
    },
    "neurovault": {"neurovault.org"},
    "osf": {"osf.io", "api.osf.io", "files.osf.io"},
    "url": set(),
}
_DATA_EXTENSIONS = (
    ".nii", ".nii.gz", ".gii", ".dscalar.nii", ".dtseries.nii", ".nrrd",
    ".nwb", ".h5", ".hdf5", ".mat", ".csv", ".tsv", ".json", ".parquet",
    ".feather", ".zarr", ".ome.tif", ".ome.tiff", ".tif", ".tiff", ".mgh",
    ".mgz", ".annot", ".label", ".graphml", ".xlsx",
)
_DEFAULT_METADATA = (
    "dataset_description.json", "participants.tsv", "participants.json",
    "README", "CHANGES", "LICENSE",
)
_STOP = {
    "about", "after", "again", "against", "analysis", "atlas", "browse",
    "combined", "could", "data", "dataset", "datasets", "download", "find",
    "idea", "imaging", "maybe", "novel", "open", "previously", "resource",
    "resources", "run", "something", "using", "various", "want", "with",
}
_CATALOG_ALIASES = {
    "psychedelic": [
        "psilocybin", "dmt", "lsd", "ayahuasca", "harmine", "5-ht2a", "serotonin",
    ],
    "psychedelics": [
        "psychedelic", "psilocybin", "dmt", "lsd", "ayahuasca", "5-ht2a", "serotonin",
    ],
    "receptor": ["neurotransmitter", "binding", "pet"],
    "receptors": ["receptor", "neurotransmitter", "binding", "pet"],
    "microscopy": ["histology", "two-photon", "ish"],
    "transcriptomic": ["gene-expression", "microarray", "rna"],
}
_PLAN_FIELDS = {
    "hypothesis", "primary_outcome", "unit_of_analysis", "inclusion_exclusion",
    "confounds", "missing_data", "multiple_testing", "validation",
    "stopping_rule", "causal_scope",
}
_SPATIAL_TERMS = {
    "atlas", "brain", "connectome", "cortex", "fmri", "imaging", "map", "mri",
    "pet", "receptor", "spatial", "voxel",
}
_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_OPENNEURO_ID = re.compile(r"^ds\d{6}$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class CatalogRecord:
    source: str
    dataset_id: str
    title: str
    url: str
    description: str = ""
    version: str = ""
    doi: str = ""
    license: str = ""
    species: str = ""
    modalities: list[str] = field(default_factory=list)
    access: str = "public"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DataFile:
    path: str
    url: str
    bytes: int
    etag: str = ""
    expected_sha256: str = ""


class DataBrokerError(RuntimeError):
    """An acquisition or scientific-contract gate rejected the request."""


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str, default: str = "dataset") -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:100]
    return result or default


def _query_terms(query: str, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", query.lower()):
        if word not in _STOP and word not in terms:
            terms.append(word)
    return terms[:limit]


def _catalog_terms(query: str, limit: int = 12) -> list[str]:
    base = _query_terms(query, limit)
    expanded: list[str] = []
    for term in base:
        for candidate in [term, *_CATALOG_ALIASES.get(term, [])]:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded[:limit]


def _score(query: str, *values: str) -> float:
    haystack = " ".join(values).lower()
    terms = _catalog_terms(query, 18)
    if not terms:
        return 0.0
    hits = sum(2.0 if term in haystack else 0.0 for term in terms)
    phrase = " ".join(terms[:3])
    return hits + (2.5 if phrase and phrase in haystack else 0.0)


def _host_allowed(source: str, url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower().rstrip(".")
        allowed = _SOURCE_HOSTS.get(source, set())
        if not any(host == item or host.endswith("." + item) for item in allowed):
            return False
        addresses = {
            item[4][0] for item in socket.getaddrinfo(
                host, parsed.port or 443, type=socket.SOCK_STREAM)
        }
        if not addresses:
            return False
        import ipaddress

        return all(
            not (
                (address := ipaddress.ip_address(raw)).is_private
                or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_reserved
                or address.is_unspecified
            )
            for raw in addresses
        )
    except Exception:
        return False


def validate_analysis_contract(
    plan: dict | None, alignment: dict | None, requests: list[dict],
) -> dict:
    """Validate the pre-execution statistical and cross-modal analysis contract."""

    plan = plan if isinstance(plan, dict) else {}
    alignment = alignment if isinstance(alignment, dict) else {}
    missing = sorted(field for field in _PLAN_FIELDS if not str(plan.get(field) or "").strip())
    exploratory = str(plan.get("mode") or "confirmatory").lower() == "exploratory"
    text = json.dumps({"plan": plan, "requests": requests}).lower()
    spatial = any(term in text for term in _SPATIAL_TERMS)
    spatial_null = str(plan.get("spatial_null") or "").strip()
    replication = str(plan.get("replication") or plan.get("validation") or "").strip()
    alignment_missing: list[str] = []
    if spatial and len(requests) > 1:
        for field_name in ("target_space", "registration", "resolution_policy"):
            if not str(alignment.get(field_name) or "").strip():
                alignment_missing.append(field_name)
    species = {
        str(item.get("species") or "").strip().lower()
        for item in requests if str(item.get("species") or "").strip()
    }
    species_mismatch = len(species) > 1
    species_bridge = str(alignment.get("species_bridge") or "").strip()
    participant_linkage = str(
        alignment.get("participant_linkage") or "unmatched group-level sources").strip()
    warnings: list[str] = []
    if species_mismatch and not species_bridge:
        alignment_missing.append("species_bridge")
    if "unmatched" in participant_linkage.lower():
        warnings.append(
            "unmatched sources support ecological/spatial association, not "
            "participant-level or causal inference")
    if spatial and not spatial_null:
        missing.append("spatial_null")
    if not replication:
        missing.append("replication")
    if not requests:
        missing.append("datasets")
    missing = sorted(set(missing))
    confirmatory_ready = bool(
        not exploratory and not missing and not alignment_missing)
    return {
        "passes": bool(not missing and not alignment_missing),
        "confirmatory_ready": confirmatory_ready,
        "mode": "exploratory" if exploratory else "confirmatory",
        "missing_plan_fields": missing,
        "missing_alignment_fields": sorted(set(alignment_missing)),
        "spatial_analysis": spatial,
        "species": sorted(species),
        "species_mismatch": species_mismatch,
        "participant_linkage": participant_linkage,
        "warnings": warnings,
    }


class ScientificDataBroker:
    """Discover and acquire public scientific datasets under a strict policy."""

    def __init__(self, root: str | Path, cfg=None, *, client=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._client = client
        self.audit = self.root / "acquisition-log.jsonl"

    @property
    def max_bytes(self) -> int:
        return int(float(getattr(self.cfg, "research_data_max_gb", 20.0)) * 1024**3)

    @property
    def reserve_bytes(self) -> int:
        return int(float(getattr(self.cfg, "research_data_reserve_gb", 8.0)) * 1024**3)

    @property
    def file_limit(self) -> int:
        return int(getattr(self.cfg, "research_data_file_limit", 20_000))

    @property
    def timeout(self) -> float:
        return float(getattr(self.cfg, "research_data_timeout", 3600))

    def _record(self, payload: dict) -> None:
        row = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        with self.audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _request(self, method: str, source: str, url: str, **kwargs) -> httpx.Response:
        if not _host_allowed(source, url):
            raise DataBrokerError(f"untrusted {source} URL: {url}")
        owned = self._client is None
        client = self._client or httpx.Client(
            timeout=httpx.Timeout(30.0, read=max(30.0, self.timeout)),
            follow_redirects=False,
            headers={"User-Agent": "spiral-research-data/1.0"},
            trust_env=False,
        )
        current = url
        try:
            for _ in range(6):
                if not _host_allowed(source, current):
                    raise DataBrokerError(f"redirect escaped trusted {source} hosts")
                response = client.request(method, current, **kwargs)
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise DataBrokerError("redirect had no location")
                    current = urllib.parse.urljoin(current, location)
                    continue
                response.raise_for_status()
                return response
            raise DataBrokerError("too many redirects")
        finally:
            if owned:
                client.close()

    def _graphql(self, query: str, variables: dict | None = None) -> dict:
        response = self._request(
            "POST", "openneuro", OPENNEURO_GRAPHQL,
            json={"query": query, "variables": variables or {}})
        payload = response.json()
        if payload.get("errors"):
            raise DataBrokerError(
                "OpenNeuro GraphQL error: "
                + str(payload["errors"][0].get("message") or payload["errors"][0]))
        return payload.get("data") or {}

    def discover(
        self, query: str, *, sources: list[str] | None = None, limit: int = 18,
    ) -> dict:
        sources = [
            str(source).lower() for source in (
                sources or getattr(
                    self.cfg, "research_data_sources",
                    ["openneuro", "allen", "neuromaps", "zenodo"]))
        ]
        records: list[CatalogRecord] = []
        errors: dict[str, str] = {}
        for source in sources:
            try:
                if source == "openneuro":
                    records.extend(self._discover_openneuro(query, max(4, limit)))
                elif source == "allen":
                    records.extend(self._discover_allen(query, max(4, limit)))
                elif source == "zenodo":
                    records.extend(self._discover_zenodo(query, max(4, limit)))
                elif source == "neuromaps":
                    records.extend(self._discover_neuromaps(query, max(4, limit)))
            except Exception as exc:
                errors[source] = f"{type(exc).__name__}: {exc}"
        records.sort(key=lambda row: (-row.score, row.source, row.title.lower()))
        # Preserve source diversity. A permissive repository with many text records
        # must not push all imaging/atlas catalogs out of the model's context.
        grouped = {
            source: [row for row in records if row.source == source]
            for source in sources
        }
        diversified: list[CatalogRecord] = []
        while len(diversified) < max(1, limit):
            advanced = False
            for source in sources:
                bucket = grouped.get(source) or []
                if bucket:
                    diversified.append(bucket.pop(0))
                    advanced = True
                    if len(diversified) >= max(1, limit):
                        break
            if not advanced:
                break
        records = diversified
        report = {
            "query": query,
            "sources": sources,
            "records": [asdict(record) for record in records],
            "errors": errors,
            "healthy_sources": sorted(set(sources) - set(errors)),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(self.root / "catalog.json", report)
        self._record({
            "kind": "catalog-search", "query": query, "sources": sources,
            "results": len(records), "errors": errors,
        })
        return report

    def _discover_openneuro(self, query: str, limit: int) -> list[CatalogRecord]:
        graph = """
        query Search($keywords: [String!], $first: Int) {
          advancedSearch(
            query: {keywords: $keywords, publicOnly: true}, first: $first
          ) {
            edges {
              node {
                id name public publishDate
                metadata {
                  modalities species datasetUrl associatedPaperDOI
                  studyDesign studyDomain tasksCompleted
                }
                latestSnapshot {
                  tag
                  description { Name DatasetDOI License Authors }
                }
              }
            }
          }
        }
        """
        terms = _catalog_terms(query, 10)
        if not terms:
            terms = ["brain"]
        # Query terms independently. Catalog keyword semantics can tighten over time;
        # independent routes keep a compound prompt from requiring one dataset to carry
        # every user keyword.
        edges = []
        seen_ids = set()
        for term in terms:
            data = self._graphql(
                graph, {"keywords": [term], "first": min(30, limit * 2)})
            for edge in ((data.get("advancedSearch") or {}).get("edges") or []):
                dataset_id = str((edge.get("node") or {}).get("id") or "")
                if dataset_id and dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    edges.append(edge)
        records: list[CatalogRecord] = []
        for edge in edges:
            node = edge.get("node") or {}
            snapshot = node.get("latestSnapshot") or {}
            description = snapshot.get("description") or {}
            metadata = node.get("metadata") or {}
            dataset_id = str(node.get("id") or "")
            title = str(description.get("Name") or node.get("name") or dataset_id)
            details = " ".join(str(metadata.get(key) or "") for key in (
                "studyDesign", "studyDomain", "tasksCompleted"))
            records.append(CatalogRecord(
                source="openneuro", dataset_id=dataset_id, title=title,
                url=f"https://openneuro.org/datasets/{dataset_id}/versions/{snapshot.get('tag')}",
                description=details[:1500],
                version=str(snapshot.get("tag") or ""),
                doi=str(description.get("DatasetDOI") or "").removeprefix("doi:"),
                license=str(description.get("License") or ""),
                species=str(metadata.get("species") or ""),
                modalities=[str(x) for x in (metadata.get("modalities") or [])],
                score=_score(query, title, details),
                metadata={
                    "published": node.get("publishDate"),
                    "authors": description.get("Authors") or [],
                    "associated_paper_doi": metadata.get("associatedPaperDOI"),
                },
            ))
        return records

    def _discover_allen(self, query: str, limit: int) -> list[CatalogRecord]:
        response = self._request("GET", "allen", ALLEN_PRODUCTS)
        products = response.json().get("msg") or []
        records: list[CatalogRecord] = []
        for product in products:
            title = str(product.get("name") or "")
            description = str(product.get("description") or "")
            resource = str(product.get("resource") or "")
            score = _score(query, title, description, resource)
            # Atlas/expression/microscopy queries need reference resources even when
            # the user's disease/drug keyword is absent from Allen's product title.
            lowered = query.lower()
            if any(term in lowered for term in ("atlas", "receptor", "expression", "microscopy")):
                if any(term in (title + " " + description + " " + resource).lower()
                       for term in ("human", "atlas", "ish", "microarray", "histology")):
                    score += 2.0
            records.append(CatalogRecord(
                source="allen",
                dataset_id=f"product-{product.get('id')}",
                title=title or f"Allen product {product.get('id')}",
                url=(
                    "https://api.brain-map.org/api/v2/data/query.json?"
                    f"criteria=model::Product,rma::criteria,[id$eq{product.get('id')}]"
                ),
                description=description[:1500],
                species=str(product.get("species") or ""),
                modalities=[
                    value for value in ("microscopy" if "image" in description.lower() else "",
                                        "transcriptomics" if any(
                                            term in description.lower()
                                            for term in ("gene expression", "transcript")) else "")
                    if value
                ],
                score=score,
                metadata={
                    "resource": resource,
                    "abbreviation": product.get("abbreviation"),
                    "access_note": (
                        "Use the product-specific AllenSDK/API adapter named in the "
                        "analysis plan; this catalog record is not itself a data file."
                    ),
                },
            ))
        records.sort(key=lambda row: -row.score)
        return records[:limit]

    def _discover_zenodo(self, query: str, limit: int) -> list[CatalogRecord]:
        search_text = " ".join(_catalog_terms(query, 10)) or query[:200]
        url = ZENODO_API + "?" + urllib.parse.urlencode({
            "q": search_text, "size": min(50, limit * 2), "sort": "bestmatch"})
        response = self._request("GET", "zenodo", url)
        hits = ((response.json().get("hits") or {}).get("hits") or [])
        records: list[CatalogRecord] = []
        for hit in hits:
            metadata = hit.get("metadata") or {}
            if str(metadata.get("access_right") or "open").lower() not in {
                "open", "embargoed",
            }:
                continue
            resource_type = str(
                (metadata.get("resource_type") or {}).get("type")
                or metadata.get("upload_type") or "").lower()
            if resource_type in {
                "publication", "presentation", "poster", "lesson", "physicalobject",
            }:
                continue
            title = str(metadata.get("title") or hit.get("id") or "")
            description = re.sub(
                r"<[^>]+>", " ", str(metadata.get("description") or ""))
            files = hit.get("files") or []
            records.append(CatalogRecord(
                source="zenodo", dataset_id=str(hit.get("id") or ""),
                title=title,
                url=str((hit.get("links") or {}).get("html") or ""),
                description=re.sub(r"\s+", " ", description).strip()[:1500],
                version=str(metadata.get("version") or ""),
                doi=str(hit.get("doi") or ""),
                license=str((metadata.get("license") or {}).get("id") or ""),
                score=_score(query, title, description, " ".join(
                    str(keyword) for keyword in metadata.get("keywords") or [])),
                metadata={
                    "published": metadata.get("publication_date"),
                    "file_count": len(files),
                    "bytes": sum(int(item.get("size") or 0) for item in files),
                    "creators": metadata.get("creators") or [],
                },
            ))
        return records

    def _neuromaps_registry(self) -> tuple[str, list[dict], list[dict]]:
        """Read the signed registry from the latest pure-Python neuromaps wheel.

        The wheel is treated as a zip archive, never imported or executed. Its PyPI
        SHA-256 is verified before the two registry JSON files are parsed.
        """

        package = self._request("GET", "neuromaps", PYPI_NEUROMAPS).json()
        version = str((package.get("info") or {}).get("version") or "")
        release = package.get("releases", {}).get(version) or []
        artifact = next((
            item for item in release
            if str(item.get("filename") or "").endswith("py3-none-any.whl")
        ), None)
        if not artifact:
            raise DataBrokerError("neuromaps has no pure-Python release wheel")
        url = str(artifact.get("url") or "")
        sha256 = str((artifact.get("digests") or {}).get("sha256") or "")
        size = int(artifact.get("size") or 0)
        if not url or not _HEX_SHA256.fullmatch(sha256) or size <= 0:
            raise DataBrokerError("neuromaps PyPI artifact metadata is incomplete")
        wheel = self.root / "catalog-cache" / "neuromaps" / str(
            artifact.get("filename"))
        self._download(
            "neuromaps",
            DataFile(
                path=wheel.name, url=url, bytes=size,
                expected_sha256=sha256),
            wheel,
        )
        try:
            with zipfile.ZipFile(wheel) as archive:
                osf = json.loads(archive.read(
                    "neuromaps/datasets/data/osf.json"))
                metadata = json.loads(archive.read(
                    "neuromaps/datasets/data/meta.json"))
        except Exception as exc:
            raise DataBrokerError(
                f"invalid neuromaps registry wheel: {type(exc).__name__}: {exc}") from exc
        annotations = [
            item for item in (osf.get("annotations") or [])
            if (
                isinstance(item, dict)
                and isinstance(item.get("url"), list)
                and len(item["url"]) == 2
                and item["url"][0] != "grh4d"
            )
        ]
        return version, annotations, list(metadata.get("annotations") or [])

    @staticmethod
    def _neuromaps_key(item: dict) -> tuple[str, str, str, str]:
        return (
            str(item.get("source") or ""),
            str(item.get("desc") or ""),
            str(item.get("space") or ""),
            str(item.get("den") or item.get("res") or ""),
        )

    def _neuromaps_groups(self) -> tuple[str, dict[tuple, dict]]:
        version, files, metadata = self._neuromaps_registry()
        meta_by_key = {
            self._neuromaps_key(item.get("annot") or {}): item
            for item in metadata if isinstance(item, dict)
        }
        groups: dict[tuple, dict] = {}
        for item in files:
            key = self._neuromaps_key(item)
            group = groups.setdefault(key, {
                "key": key, "files": [], "metadata": meta_by_key.get(key) or {}})
            group["files"].append(item)
        return version, groups

    def _discover_neuromaps(self, query: str, limit: int) -> list[CatalogRecord]:
        version, groups = self._neuromaps_groups()
        records = []
        for key, group in groups.items():
            source_name, desc, space, density = key
            metadata = group.get("metadata") or {}
            description = str(metadata.get("full_desc") or "")
            tags = sorted({
                str(tag) for item in group["files"]
                for tag in (item.get("tags") or [])
            })
            references = [
                str(item.get("citation") or "")
                for category in ("primary", "secondary")
                for item in ((metadata.get("refs") or {}).get(category) or [])
                if str(item.get("citation") or "").strip()
            ]
            license_data = metadata.get("license") or {}
            dataset_id = "-".join(filter(None, key))
            title = description or f"{source_name} {desc} {space} {density}"
            records.append(CatalogRecord(
                source="neuromaps", dataset_id=dataset_id, title=title,
                url=(
                    "https://netneurolab.github.io/neuromaps/listofmaps.html"
                    f"#{dataset_id.lower()}"
                ),
                description=("; ".join(tags) + ". " + description).strip()[:1500],
                version=version,
                license=str(license_data.get("type") or ""),
                species="human",
                modalities=["PET"] if "pet" in " ".join(tags).lower() else ["brain map"],
                score=_score(
                    query, source_name, desc, space, " ".join(tags),
                    description, " ".join(references)),
                metadata={
                    "annotation": {
                        "source": source_name, "desc": desc, "space": space,
                        ("den" if group["files"][0].get("den") else "res"): density,
                    },
                    "tags": tags,
                    "demographics": metadata.get("demographics") or {},
                    "references": references,
                    "warning": metadata.get("warning") or "",
                    "file_count": len(group["files"]),
                },
            ))
        records.sort(key=lambda row: -row.score)
        return records[:limit]

    def _openneuro_metadata(self, dataset_id: str) -> dict:
        graph = """
        query Dataset($id: ID!) {
          dataset(id: $id) {
            id name public metadata { modalities species associatedPaperDOI }
            latestSnapshot {
              tag description { Name DatasetDOI License Authors }
            }
          }
        }
        """
        node = self._graphql(graph, {"id": dataset_id}).get("dataset")
        if not node or not node.get("public"):
            raise DataBrokerError(f"OpenNeuro dataset is missing or not public: {dataset_id}")
        return node

    def _list_openneuro(
        self, dataset_id: str, includes: list[str],
    ) -> list[DataFile]:
        rows: dict[str, DataFile] = {}
        wildcard_patterns: list[str] = []
        for pattern in includes:
            if not any(char in pattern for char in "*?["):
                key = dataset_id + "/" + pattern
                url = OPENNEURO_S3 + "/" + urllib.parse.quote(key, safe="/")
                try:
                    response = self._request("HEAD", "openneuro", url)
                except httpx.HTTPStatusError as exc:
                    if (
                        exc.response.status_code == 404
                        and pattern in _DEFAULT_METADATA
                    ):
                        continue
                    raise
                size = int(response.headers.get("content-length") or 0)
                rows[pattern] = DataFile(
                    path=pattern, url=url, bytes=size,
                    etag=response.headers.get("etag", "").strip('"'),
                )
            else:
                wildcard_patterns.append(pattern)

        # S3 prefix scans begin at the literal part before the first wildcard. This
        # makes a request such as ``derivatives/maps/*.nii.gz`` inspect that subtree
        # rather than every raw image in a large OpenNeuro dataset.
        prefixes: set[str] = set()
        for pattern in wildcard_patterns:
            first = min(
                (pattern.find(char) for char in "*?[" if char in pattern),
                default=0,
            )
            prefixes.add(dataset_id + "/" + pattern[:first])
        examined = 0
        inventory_cap = max(250_000, self.file_limit * 20)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for prefix in sorted(prefixes):
            token = ""
            while True:
                params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
                if token:
                    params["continuation-token"] = token
                url = OPENNEURO_S3 + "?" + urllib.parse.urlencode(params)
                response = self._request("GET", "openneuro", url)
                root = ET.fromstring(response.content)
                for item in root.findall("s3:Contents", namespace):
                    examined += 1
                    if examined > inventory_cap:
                        raise DataBrokerError(
                            f"OpenNeuro inventory scan exceeded {inventory_cap} objects; "
                            "use a narrower include prefix")
                    key = item.findtext("s3:Key", default="", namespaces=namespace)
                    relative = key.removeprefix(dataset_id + "/")
                    if not relative or not any(
                        fnmatch.fnmatch(relative, pattern)
                        for pattern in wildcard_patterns
                    ):
                        continue
                    size = int(item.findtext(
                        "s3:Size", default="0", namespaces=namespace))
                    etag = item.findtext(
                        "s3:ETag", default="", namespaces=namespace).strip('"')
                    rows[relative] = DataFile(
                        path=relative,
                        url=OPENNEURO_S3 + "/" + urllib.parse.quote(key, safe="/"),
                        bytes=size, etag=etag,
                    )
                    if len(rows) > self.file_limit:
                        raise DataBrokerError(
                            f"selection exceeds {self.file_limit} files")
                truncated = (
                    root.findtext(
                        "s3:IsTruncated", default="false", namespaces=namespace)
                    .lower() == "true"
                )
                token = root.findtext(
                    "s3:NextContinuationToken", default="", namespaces=namespace)
                if not truncated or not token:
                    break
        return list(rows.values())

    def _zenodo_metadata(self, dataset_id: str) -> tuple[dict, list[DataFile]]:
        url = f"{ZENODO_API}/{urllib.parse.quote(dataset_id, safe='')}"
        payload = self._request("GET", "zenodo", url).json()
        access = str(
            (payload.get("metadata") or {}).get("access_right") or "open").lower()
        if access not in {"open", "embargoed"}:
            raise DataBrokerError(f"Zenodo record is not publicly downloadable: {access}")
        files: list[DataFile] = []
        for item in payload.get("files") or []:
            links = item.get("links") or {}
            download = str(links.get("content") or links.get("self") or "")
            checksum = str(item.get("checksum") or "")
            files.append(DataFile(
                path=str(item.get("key") or Path(urllib.parse.urlparse(download).path).name),
                url=download, bytes=int(item.get("size") or 0),
                expected_sha256=(
                    checksum.split(":", 1)[1]
                    if checksum.startswith("sha256:") else ""),
                etag=checksum,
            ))
        return payload, files

    def _resolve(self, request: dict) -> tuple[dict, list[DataFile]]:
        source = str(request.get("source") or "").lower()
        dataset_id = str(request.get("id") or request.get("dataset_id") or "").strip()
        includes = request.get("include") or request.get("files")
        if not includes:
            includes = ["*"] if source == "neuromaps" else list(_DEFAULT_METADATA)
        if not isinstance(includes, list) or not includes:
            raise DataBrokerError("dataset include must be a non-empty list of glob patterns")
        includes = [str(pattern) for pattern in includes]
        if any(
            not pattern or pattern.startswith(("/", "..")) or "\x00" in pattern
            for pattern in includes
        ):
            raise DataBrokerError("unsafe dataset include pattern")
        metadata: dict[str, Any]
        files: list[DataFile]
        if source == "openneuro":
            if not _OPENNEURO_ID.fullmatch(dataset_id):
                raise DataBrokerError(f"invalid OpenNeuro accession: {dataset_id}")
            metadata = self._openneuro_metadata(dataset_id)
            latest = metadata.get("latestSnapshot") or {}
            requested_version = str(request.get("version") or "")
            if requested_version and requested_version != str(latest.get("tag") or ""):
                raise DataBrokerError(
                    f"{dataset_id} requested snapshot {requested_version}, but public S3 "
                    f"currently exposes latest snapshot {latest.get('tag')}; use the pinned latest "
                    "or add a historical-snapshot adapter")
            files = self._list_openneuro(dataset_id, includes)
            description = latest.get("description") or {}
            normalized = {
                "source": source, "dataset_id": dataset_id,
                "title": description.get("Name") or metadata.get("name"),
                "version": latest.get("tag") or "",
                "doi": str(description.get("DatasetDOI") or "").removeprefix("doi:"),
                "license": description.get("License") or "",
                "citation": request.get("citation") or description.get("DatasetDOI") or "",
                "species": (metadata.get("metadata") or {}).get("species") or "",
                "modalities": (metadata.get("metadata") or {}).get("modalities") or [],
                "associated_paper_doi": (
                    metadata.get("metadata") or {}).get("associatedPaperDOI") or "",
            }
        elif source == "zenodo":
            metadata, available = self._zenodo_metadata(dataset_id)
            files = [
                item for item in available
                if any(fnmatch.fnmatch(item.path, pattern) for pattern in includes)
            ]
            md = metadata.get("metadata") or {}
            normalized = {
                "source": source, "dataset_id": dataset_id,
                "title": md.get("title") or dataset_id,
                "version": md.get("version") or request.get("version") or "",
                "doi": metadata.get("doi") or "",
                "license": (md.get("license") or {}).get("id") or "",
                "citation": request.get("citation") or metadata.get("doi") or "",
                "species": request.get("species") or "",
                "modalities": request.get("modalities") or [],
            }
        elif source == "neuromaps":
            version, groups = self._neuromaps_groups()
            annotation = request.get("annotation")
            if isinstance(annotation, dict):
                key = self._neuromaps_key(annotation)
            else:
                key = next((
                    candidate for candidate in groups
                    if "-".join(filter(None, candidate)) == dataset_id
                ), ("", "", "", ""))
            group = groups.get(key)
            if not group:
                raise DataBrokerError(
                    f"unknown public neuromaps annotation: {dataset_id or annotation}")
            requested_version = str(request.get("version") or "")
            if requested_version and requested_version != version:
                raise DataBrokerError(
                    f"neuromaps registry release is {version}, not {requested_version}")
            files = []
            for item in group["files"]:
                relative = str(Path(str(item.get("rel_path") or "")) / str(
                    item.get("fname") or ""))
                if not any(
                    fnmatch.fnmatch(relative, pattern)
                    or fnmatch.fnmatch(str(item.get("fname") or ""), pattern)
                    for pattern in includes
                ):
                    continue
                project, file_id = item["url"]
                url = (
                    f"https://files.osf.io/v1/resources/{project}"
                    f"/providers/osfstorage/{file_id}"
                )
                response = self._request("HEAD", "neuromaps", url)
                size = int(response.headers.get("content-length") or 0)
                waterbutler = {}
                try:
                    waterbutler = json.loads(
                        response.headers.get("x-waterbutler-metadata") or "{}")
                except Exception:
                    pass
                hashes = (
                    ((waterbutler.get("attributes") or {}).get("extra") or {})
                    .get("hashes") or {}
                )
                expected_sha256 = str(hashes.get("sha256") or "")
                files.append(DataFile(
                    path=relative, url=url, bytes=size,
                    etag=response.headers.get("etag", "").strip('"'),
                    expected_sha256=(
                        expected_sha256
                        if _HEX_SHA256.fullmatch(expected_sha256) else ""),
                ))
            metadata = group.get("metadata") or {}
            license_data = metadata.get("license") or {}
            references = [
                str(item.get("citation") or "")
                for category in ("primary", "secondary")
                for item in ((metadata.get("refs") or {}).get(category) or [])
                if str(item.get("citation") or "").strip()
            ]
            normalized = {
                "source": source,
                "dataset_id": "-".join(filter(None, key)),
                "title": metadata.get("full_desc") or " ".join(key),
                "version": version,
                "doi": request.get("doi") or "",
                "license": license_data.get("type") or "",
                "citation": request.get("citation") or (
                    references[0] if references else ""),
                "species": "human",
                "modalities": ["PET"] if any(
                    "PET" in (item.get("tags") or []) for item in group["files"]
                ) else ["brain map"],
                "annotation": {
                    "source": key[0], "desc": key[1], "space": key[2],
                    ("den" if group["files"][0].get("den") else "res"): key[3],
                },
                "references": references,
                "warning": metadata.get("warning") or "",
            }
        else:
            url = str(request.get("url") or "")
            if source not in _SOURCE_HOSTS or source in {"url", "openneuro", "zenodo"}:
                raise DataBrokerError(f"unsupported typed data source: {source}")
            if not _host_allowed(source, url):
                raise DataBrokerError(f"untrusted {source} data URL")
            response = self._request("HEAD", source, url)
            size = int(response.headers.get("content-length") or request.get("bytes") or 0)
            if size <= 0:
                raise DataBrokerError("direct data URL has no trustworthy Content-Length")
            filename = str(
                request.get("filename")
                or Path(urllib.parse.urlparse(url).path).name
                or dataset_id)
            files = [DataFile(
                path=filename, url=url, bytes=size,
                etag=response.headers.get("etag", "").strip('"'),
                expected_sha256=str(request.get("sha256") or ""),
            )]
            normalized = {
                "source": source, "dataset_id": dataset_id or filename,
                "title": request.get("title") or dataset_id or filename,
                "version": request.get("version") or response.headers.get("last-modified", ""),
                "doi": request.get("doi") or "",
                "license": request.get("license") or "",
                "citation": request.get("citation") or request.get("doi") or "",
                "species": request.get("species") or "",
                "modalities": request.get("modalities") or [],
            }
        if not files:
            raise DataBrokerError(
                f"no files matched include patterns for {source}:{dataset_id}")
        if len(files) > self.file_limit:
            raise DataBrokerError(f"selection exceeds {self.file_limit} files")
        for item in files:
            if item.bytes < 0:
                raise DataBrokerError(f"invalid file size for {item.path}")
        normalized["include"] = includes
        normalized["file_count"] = len(files)
        normalized["bytes"] = sum(item.bytes for item in files)
        return normalized, files

    def _check_capacity(self, total: int) -> None:
        if total > self.max_bytes:
            raise DataBrokerError(
                f"selected data are {total / 1024**3:.2f} GiB; configured cap is "
                f"{self.max_bytes / 1024**3:.2f} GiB")
        free = shutil.disk_usage(self.root).free
        if free - total < self.reserve_bytes:
            raise DataBrokerError(
                f"download would leave {(free - total) / 1024**3:.2f} GiB free; "
                f"configured reserve is {self.reserve_bytes / 1024**3:.2f} GiB")

    def _download(self, source: str, item: DataFile, destination: Path) -> dict:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == item.bytes:
            digest = _sha256(destination)
            if not item.expected_sha256 or digest == item.expected_sha256.lower():
                return {
                    "path": str(destination), "bytes": item.bytes, "sha256": digest,
                    "etag": item.etag, "url": item.url, "cache_hit": True,
                }
        part = destination.with_name(destination.name + ".part")
        offset = part.stat().st_size if part.is_file() else 0
        if offset > item.bytes:
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        if not _host_allowed(source, item.url):
            raise DataBrokerError(f"untrusted resolved data URL: {item.url}")
        with httpx.Client(
            timeout=httpx.Timeout(30.0, read=max(30.0, self.timeout)),
            follow_redirects=False,
            headers={"User-Agent": "spiral-research-data/1.0"},
            trust_env=False,
        ) as client:
            current = item.url
            for _ in range(6):
                if not _host_allowed(source, current):
                    raise DataBrokerError("download redirect escaped trusted hosts")
                with client.stream("GET", current, headers=headers) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise DataBrokerError("download redirect had no location")
                        current = urllib.parse.urljoin(current, location)
                        continue
                    response.raise_for_status()
                    response_etag = response.headers.get("etag", "").strip('"')
                    if item.etag and response_etag and response_etag != item.etag:
                        raise DataBrokerError(
                            f"source object changed after inventory for {item.path}")
                    if offset and response.status_code != 206:
                        part.unlink(missing_ok=True)
                        offset = 0
                        return self._download(source, item, destination)
                    mode = "ab" if offset else "wb"
                    with part.open(mode) as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                    break
            else:
                raise DataBrokerError("too many download redirects")
        size = part.stat().st_size
        if size != item.bytes:
            raise DataBrokerError(
                f"incomplete download for {item.path}: {size} of {item.bytes} bytes")
        digest = _sha256(part)
        if item.expected_sha256 and digest != item.expected_sha256.lower():
            raise DataBrokerError(f"SHA-256 mismatch for {item.path}")
        part.replace(destination)
        return {
            "path": str(destination), "bytes": size, "sha256": digest,
            "etag": item.etag, "url": item.url, "cache_hit": False,
            "source_identity_verified": bool(item.etag),
        }

    @staticmethod
    def _format_inventory(files: list[dict]) -> dict:
        formats: dict[str, int] = {}
        unknown = 0
        for item in files:
            name = str(item.get("relative_path") or item.get("path") or "").lower()
            suffix = next((ext for ext in _DATA_EXTENSIONS if name.endswith(ext)), "")
            if suffix:
                formats[suffix] = formats.get(suffix, 0) + 1
            else:
                unknown += 1
        return {
            "formats": dict(sorted(formats.items())),
            "unknown_or_metadata_files": unknown,
            "bids_description_present": any(
                str(item.get("relative_path") or "").endswith("dataset_description.json")
                for item in files),
            "tabular_present": any(key in formats for key in (".csv", ".tsv", ".parquet")),
            "neuroimage_present": any(
                key in formats for key in (
                    ".nii", ".nii.gz", ".gii", ".dscalar.nii", ".dtseries.nii",
                    ".nrrd", ".mgh", ".mgz")),
        }

    @staticmethod
    def _signature_checks(files: list[dict]) -> dict:
        """Cheap structural checks before analysis libraries parse full arrays."""

        checks = []
        for item in files:
            path = Path(str(item.get("path") or ""))
            name = str(item.get("relative_path") or path.name).lower()
            kind = ""
            valid = True
            reason = ""
            try:
                def head(size: int) -> bytes:
                    with path.open("rb") as handle:
                        return handle.read(size)

                if name.endswith(".nii.gz"):
                    kind = "nifti-gzip"
                    with gzip.open(path, "rb") as handle:
                        header = handle.read(4)
                    valid = int.from_bytes(header, "little") == 348 or int.from_bytes(
                        header, "big") == 348
                elif name.endswith((".nii", ".dscalar.nii", ".dtseries.nii")):
                    kind = "nifti"
                    header = head(4)
                    valid = int.from_bytes(header, "little") == 348 or int.from_bytes(
                        header, "big") == 348
                elif name.endswith(".nrrd"):
                    kind = "nrrd"
                    valid = head(8).startswith(b"NRRD")
                elif name.endswith((".nwb", ".h5", ".hdf5")):
                    kind = "hdf5"
                    valid = head(8) == b"\x89HDF\r\n\x1a\n"
                elif name.endswith(".parquet"):
                    kind = "parquet"
                    with path.open("rb") as handle:
                        start = handle.read(4)
                        handle.seek(max(0, path.stat().st_size - 4))
                        end = handle.read(4)
                    valid = start == b"PAR1" and end == b"PAR1"
                elif name.endswith((".gii",)):
                    kind = "gifti"
                    valid = b"<GIFTI" in head(4096)
                elif name.endswith(".json"):
                    kind = "json"
                    if path.stat().st_size <= 20 * 1024 * 1024:
                        json.loads(path.read_text(encoding="utf-8"))
                    else:
                        valid = head(4096).lstrip()[:1] in {b"{", b"["}
                elif name.endswith((".csv", ".tsv")):
                    kind = "tabular-text"
                    valid = bool(head(4096).strip())
                elif name.endswith(".xlsx"):
                    kind = "xlsx"
                    valid = head(4) == b"PK\x03\x04"
                elif name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
                    kind = "tiff"
                    valid = head(4) in {b"II*\x00", b"MM\x00*"}
            except Exception as exc:
                valid = False
                reason = f"{type(exc).__name__}: {exc}"
            if kind:
                checks.append({
                    "path": str(item.get("relative_path") or path.name),
                    "kind": kind, "valid": bool(valid),
                    "reason": reason or ("" if valid else "invalid file signature"),
                })
        return {
            "checks": checks,
            "checked_file_count": len(checks),
            "invalid": [item for item in checks if not item["valid"]],
            "passes": all(item["valid"] for item in checks),
        }

    def acquire_many(
        self, requests: list[dict], *, plan: dict | None = None,
        alignment: dict | None = None, materialize_to: str | Path | None = None,
    ) -> dict:
        if not isinstance(requests, list) or not requests:
            raise DataBrokerError("datasets must be a non-empty list")
        if not bool(getattr(self.cfg, "research_data_auto", True)):
            raise DataBrokerError("automatic scientific-data acquisition is disabled")
        aliases: set[str] = set()
        clean_requests: list[dict] = []
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise DataBrokerError("each dataset request must be an object")
            request = dict(request)
            alias = str(request.get("alias") or f"dataset_{index + 1}")
            if not _ALIAS.fullmatch(alias) or alias in aliases:
                raise DataBrokerError(f"unsafe or duplicate dataset alias: {alias!r}")
            aliases.add(alias)
            request["alias"] = alias
            clean_requests.append(request)

        contract = validate_analysis_contract(plan, alignment, clean_requests)
        plan_payload = {
            "analysis_plan": plan or {}, "alignment": alignment or {},
            "datasets": clean_requests, "contract": contract,
        }
        plan_hash = hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        plan_path = self.root / "plans" / f"{plan_hash}.json"
        if not plan_path.exists():
            _atomic_json(plan_path, {
                **plan_payload, "plan_hash": plan_hash,
                "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        resolved: list[tuple[dict, dict, list[DataFile]]] = []
        for request in clean_requests:
            metadata, files = self._resolve(request)
            resolved.append((request, metadata, files))
        # Catalog metadata, rather than model prose, is authoritative for species and
        # modality compatibility. Re-evaluate the contract after resolution and before
        # any bytes move.
        resolved_requests = []
        for request, metadata, _ in resolved:
            resolved_requests.append({
                **request,
                "species": metadata.get("species") or request.get("species") or "",
                "modalities": metadata.get("modalities") or request.get("modalities") or [],
            })
        resolved_contract = validate_analysis_contract(
            plan, alignment, resolved_requests)
        resolved_contract["pre_resolution_contract"] = contract
        contract = resolved_contract
        total = sum(metadata["bytes"] for _, metadata, _ in resolved)
        self._check_capacity(total)

        records: list[dict] = []
        downloaded_total = 0
        try:
            for request, metadata, files in resolved:
                alias = request["alias"]
                version = _slug(str(metadata.get("version") or "unversioned"))
                cache = (
                    self.root / "cache" / _slug(metadata["source"])
                    / _slug(metadata["dataset_id"]) / version)
                cache.mkdir(parents=True, exist_ok=True)
                acquired: list[dict] = []
                for item in files:
                    relative = Path(item.path)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise DataBrokerError(f"unsafe resolved file path: {item.path}")
                    result = self._download(
                        metadata["source"], item, cache / relative)
                    result["relative_path"] = str(relative)
                    result["alias"] = alias
                    acquired.append(result)
                    if not result["cache_hit"]:
                        downloaded_total += result["bytes"]
                formats = self._format_inventory(acquired)
                signatures = self._signature_checks(acquired)
                complete_provenance = bool(
                    metadata.get("dataset_id") and metadata.get("version")
                    and metadata.get("license") and metadata.get("citation")
                    and all(item.get("sha256") for item in acquired)
                    and signatures.get("passes")
                )
                record = {
                    **metadata, "alias": alias, "cache_path": str(cache),
                    "files": acquired, "format_inventory": formats,
                    "signature_validation": signatures,
                    "provenance_complete": complete_provenance,
                    "purpose": request.get("purpose") or "",
                }
                _atomic_json(cache / "spiral-data-manifest.json", record)
                records.append(record)
            target = Path(materialize_to).resolve() if materialize_to else None
            if target is not None:
                target.mkdir(parents=True, exist_ok=True)
                for record in records:
                    destination = target / record["alias"]
                    destination.mkdir(parents=True, exist_ok=True)
                    for item in record["files"]:
                        source_path = Path(item["path"])
                        relative = Path(item["relative_path"])
                        output = destination / relative
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if output.exists():
                            if output.stat().st_size == source_path.stat().st_size:
                                continue
                            output.unlink()
                        try:
                            os.link(source_path, output)
                        except OSError:
                            shutil.copy2(source_path, output)
                    _atomic_json(
                        destination / "spiral-data-manifest.json", record)
            provenance_ready = all(
                record.get("provenance_complete") for record in records)
            report = {
                "ok": True,
                "plan_hash": plan_hash,
                "plan_manifest": str(plan_path),
                "contract": contract,
                "records": records,
                "selected_bytes": total,
                "downloaded_bytes": downloaded_total,
                "provenance_complete": provenance_ready,
                "confirmatory_ready": bool(
                    contract.get("confirmatory_ready") and provenance_ready),
                "materialized_path": str(target) if target else "",
            }
            report_path = self.root / "runs" / f"{plan_hash}.json"
            _atomic_json(report_path, report)
            report["manifest"] = str(report_path)
            self._record({
                "kind": "acquisition", "ok": True, "plan_hash": plan_hash,
                "datasets": [record["dataset_id"] for record in records],
                "selected_bytes": total, "downloaded_bytes": downloaded_total,
                "confirmatory_ready": report["confirmatory_ready"],
            })
            return report
        except Exception as exc:
            self._record({
                "kind": "acquisition", "ok": False, "plan_hash": plan_hash,
                "error": f"{type(exc).__name__}: {exc}",
                "downloaded_bytes": downloaded_total,
            })
            raise
