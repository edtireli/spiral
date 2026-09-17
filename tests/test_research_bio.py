"""Bio/med expansion: domain routing, channel fan-out, hybrid evidence verification.
All offline — adapters and models are faked, no network."""
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from spiral.config import Config
from spiral.research_evidence import evidence_support
from spiral.research_loop import (
    ResearchLoop, _QUALIFYING_EMPIRICAL, _QUALIFYING_MATH,
)
from spiral.sources import Record


def _paper(uid, text):
    return NS(bare_id=uid, arxiv_id=uid, title=uid, text=text, abstract="")


# ---------------------------------------------------------------- evidence grounding
def test_evidence_requires_independent_anchors():
    stmt = "TDP-43 aggregation increases with age in cortical neurons"
    papers = [
        _paper("doi:a", "Here TDP-43 aggregation increases with age in cortical neurons of mice."),
        _paper("doi:b", "We find TDP-43 aggregation rises with age across cortical neurons."),
        _paper("doi:c", "This is about photosynthesis and chlorophyll only."),
    ]
    ev = evidence_support(stmt, papers, min_support=2)
    assert ev.supported and ev.support_count == 2 and len(ev.anchors) == 2

    # one supporter is not evidence
    assert not evidence_support(stmt, papers[:1] + papers[2:], min_support=2).supported


def test_evidence_surfaces_dissent_not_hides_it():
    stmt = "TDP-43 aggregation increases with age in cortical neurons"
    papers = [
        _paper("doi:a", "TDP-43 aggregation increases with age in cortical neurons here."),
        _paper("doi:b", "TDP-43 aggregation rises with age in cortical neurons in patients."),
        _paper("doi:d", "Contrary to that, TDP-43 aggregation did not increase with age in cortical neurons."),
    ]
    ev = evidence_support(stmt, papers, min_support=2)
    assert ev.supported and len(ev.dissent) == 1 and ev.dissent[0].uid == "doi:d"
    assert "disagreement" in ev.detail


def test_evidence_rejects_vague_claims():
    assert not evidence_support("it works well", [_paper("x", "it works well indeed")]).supported


def test_grounded_qualifies_only_for_empirical():
    assert "grounded" in _QUALIFYING_EMPIRICAL and "grounded" not in _QUALIFYING_MATH


# ---------------------------------------------------------------- domain routing
def _loop(topic):
    loop = ResearchLoop(topic, workdir=Path(tempfile.mkdtemp()), cfg=Config.load())
    loop._think_json = lambda *a, **k: {}          # force deterministic fallback
    return loop


def test_search_plan_routes_bio_to_bio_channels():
    loop = _loop("TDP-43 aggregation drives ALS progression in cortical neurons")
    cats, qs = loop.search_plan()
    assert loop.state.field_domain == "bio-med"
    assert "europepmc" in loop.state.channels and "arxiv" not in loop.state.channels
    assert qs and cats == []


def test_search_plan_routes_physics_to_arxiv():
    loop = _loop("gauge anomaly cancellation in higher dimensional Yang-Mills theory")
    loop.search_plan()
    assert loop.state.field_domain == "physics-math" and loop.state.channels == ["arxiv"]


def test_search_plan_honours_model_channel_choice():
    loop = ResearchLoop("some ambiguous topic about networks",
                        workdir=Path(tempfile.mkdtemp()), cfg=Config.load())
    loop._think_json = lambda *a, **k: {
        "domain": "bio-med", "channels": ["pubmed", "crossref"],
        "categories": [], "queries": ["neural network plasticity"]}
    cats, qs = loop.search_plan()
    assert loop.state.channels == ["pubmed", "crossref"] and qs == ["neural network plasticity"]


# ---------------------------------------------------------------- channel fan-out
def test_gather_fans_out_across_channels_and_ingests(monkeypatch):
    import spiral.sources as S

    loop = _loop("TDP-43 aggregation in ALS")
    loop.search_plan()                             # sets bio channels
    loop.state.channels = ["europepmc", "pubmed"]

    def fake_epmc(query, k=6, report=None, **kw):
        if report is not None:
            report.update({"source_ok": True, "result_count": 1})
        return [Record(uid="doi:10.1101/x", title="EPMC hit", source="biorxiv",
                       doi="10.1101/x", abstract="TDP-43 aggregates in ALS neurons.")]

    def fake_pubmed(query, k=6, report=None, **kw):
        if report is not None:
            report.update({"source_ok": True, "result_count": 1})
        return [Record(uid="pmid:99", title="PubMed hit", source="pubmed",
                       abstract="A review of ALS pathology.")]

    monkeypatch.setattr(S, "europepmc", fake_epmc)
    monkeypatch.setattr(S, "pubmed", fake_pubmed)
    # no body fetch network: abstracts become the body
    n = loop.gather("TDP-43 ALS", k=4)
    assert n == 2
    ids = set(loop.corpus.papers)
    assert "doi:10.1101/x" in ids and "pmid:99" in ids
    # health recorded per channel in the last search report
    last = loop.corpus.last_build_report if hasattr(loop.corpus, "last_build_report") else {}
    # the search record carries channel health
    rec = loop.state.history[-1] if loop.state.history else {}
    assert loop.corpus.papers["pmid:99"].source == "pubmed"


def test_gather_survives_a_dead_channel(monkeypatch):
    import spiral.sources as S
    loop = _loop("TDP-43 aggregation in ALS")
    loop.state.channels = ["europepmc"]

    def boom(query, k=6, report=None, **kw):
        raise RuntimeError("EBI down")

    monkeypatch.setattr(S, "europepmc", boom)
    n = loop.gather("q", k=4)          # must not raise
    assert n == 0


def test_distinct_providers_fetch_concurrently_but_ingest_on_caller(monkeypatch):
    import threading
    import spiral.sources as S

    loop = _loop("neuronal protein expression")
    loop.state.channels = ["europepmc", "pubmed", "crossref", "pubmed"]
    rendezvous = threading.Barrier(3, timeout=3)
    caller = threading.get_ident()
    fetched, ingested = [], []

    def adapter(name):
        def fetch(query, k, report):
            fetched.append(name)
            assert threading.get_ident() != caller
            rendezvous.wait()  # all three must start before any can complete
            report.update(source_ok=True, result_count=1)
            return [name]
        return fetch

    def ingest(records, on):
        assert threading.get_ident() == caller
        ingested.extend(records)
        return records

    for channel in ("europepmc", "pubmed", "crossref"):
        monkeypatch.setattr(S, channel, adapter(channel))
    monkeypatch.setattr(loop.corpus, "ingest", ingest)
    added, health = loop._gather_channels("query", 3, on=lambda _: None)
    assert sorted(fetched) == ["crossref", "europepmc", "pubmed"]
    assert added == ingested == ["europepmc", "pubmed", "crossref"]
    assert all(row["source_ok"] for row in health.values())


def test_europe_pmc_channels_remain_serial_and_failure_does_not_hide_other_sources(monkeypatch):
    import threading
    import spiral.sources as S

    loop = _loop("neuronal protein expression")
    loop.state.channels = ["biorxiv", "pubmed", "europepmc", "medrxiv"]
    rendezvous = threading.Barrier(2, timeout=3)
    epmc_lock = threading.Lock()
    epmc_calls = []

    def epmc(name):
        def fetch(query, k, report):
            assert epmc_lock.acquire(blocking=False), "same provider was queried concurrently"
            try:
                epmc_calls.append(name)
                if name == "biorxiv":
                    rendezvous.wait()
                    raise RuntimeError("preprint query unavailable")
                report.update(source_ok=True, result_count=1)
                return [name]
            finally:
                epmc_lock.release()
        return fetch

    def pubmed(query, k, report):
        rendezvous.wait()
        report.update(source_ok=True, result_count=1)
        return ["pubmed"]

    for channel in ("biorxiv", "europepmc", "medrxiv"):
        monkeypatch.setattr(S, channel, epmc(channel))
    monkeypatch.setattr(S, "pubmed", pubmed)
    monkeypatch.setattr(loop.corpus, "ingest", lambda records, on: records)
    added, health = loop._gather_channels("query", 3, on=lambda _: None)
    assert epmc_calls == ["biorxiv", "europepmc", "medrxiv"]
    assert added == ["pubmed", "europepmc", "medrxiv"]
    assert health["biorxiv"]["source_ok"] is False
    assert "preprint query unavailable" in health["biorxiv"]["error"]
    assert all(health[channel]["source_ok"] for channel in added)


@pytest.mark.parametrize("arxiv_ok, pubmed_ok", [(False, True), (True, False), (False, False)])
def test_mixed_source_health_preserves_any_successful_provider(arxiv_ok, pubmed_ok, monkeypatch):
    loop = _loop("mixed literature")
    loop.state.channels = ["arxiv", "pubmed"]
    loop.corpus.last_build_report = {"source_ok": arxiv_ok, "result_count": int(arxiv_ok)}
    monkeypatch.setattr(loop.corpus, "build", lambda *_, **__: [])
    monkeypatch.setattr(loop, "_gather_channels", lambda *_, **__: (
        [], {"pubmed": {"source_ok": pubmed_ok, "result_count": int(pubmed_ok)}}))
    reports = []
    monkeypatch.setattr(loop, "_record_search", lambda *args: reports.append(args[-1]))
    loop.gather("query")
    assert reports[0]["source_ok"] == (arxiv_ok or pubmed_ok)
    assert reports[0]["result_count"] == int(arxiv_ok) + int(pubmed_ok)
