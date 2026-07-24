"""Liveness + grounded-critic fixes for spiralʳᵉˢᵉᵃʳᶜʰ.

These tests replay the failure modes of a real 20-round live-locked run (frozen
corpus, dead arXiv + Semantic Scholar, one relevant query family, a critic shown the
same 20 seed bibliographies forever) and pin the fixed behaviour: multi-word arXiv
queries that can match, body-first excerpts, a relevance-ranked + indexed critic
window with cross-round state, the citation graph counting as a relevant route, and
a stall that degrades the discovery gate explicitly instead of vetoing forever.
Runs standalone (`python tests/test_research_liveness.py`) or under pytest.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── arXiv query construction ────────────────────────────────────────────────
def test_arxiv_terms_short_query_stays_exact_phrase():
    from spiral.research import arxiv_terms
    assert arxiv_terms("gregory laflamme") == 'all:"gregory laflamme"'


def test_arxiv_terms_long_query_becomes_and_of_terms():
    """An 8-word exact phrase matches essentially nothing on arXiv — the query that
    live-locked a run must decompose into AND terms."""
    from spiral.research import arxiv_terms
    out = arxiv_terms("kodama ishibashi master equations higher dimensional black holes")
    assert out.count(" AND ") == 7
    assert '"' not in out and out.startswith("all:kodama")


def test_arxiv_search_query_parenthesizes_and_terms_with_categories(monkeypatch):
    from spiral import research
    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        return ""

    monkeypatch.setattr(research, "_get", fake_get)
    research.arxiv("black string instability master equation", k=3,
                   categories=["gr-qc", "hep-th"])
    from urllib.parse import parse_qs, urlparse
    sq = parse_qs(urlparse(seen["url"]).query)["search_query"][0]
    assert sq.startswith("(cat:gr-qc OR cat:hep-th) AND (all:black")
    assert sq.endswith(")")


# ── body-first TeX extraction and display ───────────────────────────────────
def _targz(files: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_extract_tex_orders_document_body_before_bibliography():
    """A .bbl that happens to sort first in the tar must not become text[:600] —
    that is how a critic ends up judging physics from reference lists."""
    from spiral.research_corpus import extract_tex_source
    raw = _targz([
        ("aaa.bbl", b"\\begin{thebibliography}{9}\\bibitem{x} X.\\end{thebibliography}"),
        ("main.tex", b"\\documentclass{a}\\begin{document}\\section{Master equations}"
                     b" \\begin{equation} L\\Phi=0 \\end{equation}\\end{document}"),
    ])
    out = extract_tex_source(raw)
    assert out.index("Master equations") < out.index("thebibliography")


def test_display_body_skips_preamble_and_reference_lists():
    from spiral.research_corpus import display_body
    text = (
        "\\documentclass{revtex4}\n\\usepackage{amsmath}\n"
        "\\newcommand{\\be}{\\begin{equation}}\n"
        "\\begin{document}\n\\section{Static response}\nThe master equation reads"
        " $L\\Phi=0$ with potential $V(r)$.\n"
        "\\begin{thebibliography}{99}\n\\bibitem{a} Someone, somewhere.\n"
        "\\end{thebibliography}\n\\end{document}"
    )
    body = display_body(text, chars=400)
    assert "master equation" in body
    assert "documentclass" not in body and "thebibliography" not in body


def test_display_body_falls_back_to_raw_when_extraction_degenerates():
    from spiral.research_corpus import display_body
    bib_only = ("\\begin{thebibliography}{99}\n" + "\\bibitem{x} Ref.\n" * 40
                + "\\end{thebibliography}")
    assert display_body(bib_only, chars=200)          # honest raw fallback, not ""


# ── corpus window and index ─────────────────────────────────────────────────
def _corpus(tmp: Path):
    from spiral.research_corpus import Corpus, Paper
    c = Corpus(tmp)
    c.add(Paper(arxiv_id="1000.00001", title="Seed A", abstract="unrelated seed"),
          fetch=False)
    c.add(Paper(arxiv_id="1000.00002", title="Seed B", abstract="another seed"),
          fetch=False)
    c.add(Paper(arxiv_id="2000.00001", title="Master equations for tidal response",
                abstract="master equations tidal response black holes"), fetch=False)
    return c


def test_summaries_honours_explicit_id_order():
    d = Path(tempfile.mkdtemp())
    c = _corpus(d)
    digest = c.summaries(limit=2, ids=["2000.00001", "1000.00002"])
    assert digest.index("Master equations") < digest.index("Seed B")
    assert "Seed A" not in digest


def test_index_lists_every_paper():
    d = Path(tempfile.mkdtemp())
    c = _corpus(d)
    index = c.index()
    assert index.count("\n") == 2                     # 3 papers, one line each
    for pid in ("1000.00001", "1000.00002", "2000.00001"):
        assert pid in index


# ── citation graph counts as a relevant route ───────────────────────────────
def _relevant_paper(i: int):
    return SimpleNamespace(
        bare_id=f"24{i:02d}.0000{i}", arxiv_id=f"24{i:02d}.0000{i}",
        title="Black hole tidal Love numbers and master equations",
        abstract="tidal love numbers master equations black holes static response",
        text="tidal love numbers master equations black holes static response " * 40,
        tex_path="x.tex", pdf_path="", body_source="tex",
    )


def test_graph_delivery_counts_as_relevant_route():
    from spiral.research_quality import CoveragePolicy, corpus_quality_report
    papers = [_relevant_paper(i) for i in range(1, 9)]
    ids = [p.bare_id for p in papers]
    topic = "classify black hole tidal love numbers via master equations static response"
    # One healthy keyword family delivered two relevant ids; the graph delivered the rest.
    research_map = {
        "searches": [
            {"query": "black hole tidal love numbers", "retrieval": {
                "source_ok": True, "result_count": 2, "result_ids": ids[:2]}},
            {"query": "tidal response master equations", "retrieval": {
                "source_ok": True, "result_count": 0, "result_ids": []}},
            {"query": "love numbers static response", "retrieval": {
                "source_ok": True, "result_count": 0, "result_ids": []}},
        ],
        "graph_rounds": [
            {"added": ids[2:], "health": {
                "requests": 10, "successful_requests": 9, "failed_requests": 1,
                "coverage_valid": True, "successful_seeds": ids[:3]}},
        ],
    }
    report = corpus_quality_report(topic, papers, research_map,
                                   policy=CoveragePolicy(min_papers=5))
    assert report["graph"]["relevant_delivered_count"] >= 2
    assert report["relevant_route_count"] >= 2
    assert report["discovery_checks"]["relevant_query_routes"] is True


def test_keyword_only_single_family_still_blocks_routes():
    """The generalized route check must not weaken the case it was written for:
    one keyword family and no graph delivery is still one route."""
    from spiral.research_quality import CoveragePolicy, corpus_quality_report
    papers = [_relevant_paper(i) for i in range(1, 9)]
    ids = [p.bare_id for p in papers]
    research_map = {
        "searches": [{"query": "black hole tidal love numbers", "retrieval": {
            "source_ok": True, "result_count": 2, "result_ids": ids[:2]}}],
        "graph_rounds": [],
    }
    report = corpus_quality_report(
        "classify black hole tidal love numbers via master equations static response",
        papers, research_map, policy=CoveragePolicy(min_papers=5))
    assert report["relevant_route_count"] == 1
    assert report["discovery_checks"]["relevant_query_routes"] is False


# ── stall override ──────────────────────────────────────────────────────────
def _blocked_report(blocking: list[str]) -> dict:
    checks = {name: True for name in (
        "paper_count_or_saturated_small_field", "usable_primary_texts",
        "topically_relevant_papers", "relevant_usable_primary_texts",
        "query_diversity", "retrieval_health", "relevant_query_routes",
        "topic_term_coverage")}
    for name in blocking:
        checks[name] = False
    return {
        "discovery_ready": not blocking,
        "novelty_ready": False,
        "discovery_checks": checks,
        "blocking_reasons": list(blocking),
        "warnings": [],
    }


def test_stall_override_opens_instrument_blocked_discovery_and_records_it():
    from spiral.research_quality import apply_stall_override
    report = apply_stall_override(
        _blocked_report(["relevant_query_routes"]),
        stalled_rounds=4, patience=3, instruments_dead=True,
        evidence={"recent_search_failures": 6})
    assert report["discovery_ready"] is True
    assert report["blocking_reasons"] == []
    assert report["stall_override"]["overridden_checks"] == ["relevant_query_routes"]
    assert report["novelty_ready"] is False           # never overridden
    assert any("stall override" in w for w in report["warnings"])


def test_stall_override_refuses_content_blockers_and_live_instruments():
    from spiral.research_quality import apply_stall_override
    # A failing CONTENT check is a real gap — no override.
    content = apply_stall_override(
        _blocked_report(["topically_relevant_papers", "relevant_query_routes"]),
        stalled_rounds=9, patience=3, instruments_dead=True)
    assert content["discovery_ready"] is False and "stall_override" not in content
    # Live instruments mean the routes can still change — no override.
    alive = apply_stall_override(
        _blocked_report(["relevant_query_routes"]),
        stalled_rounds=9, patience=3, instruments_dead=False)
    assert alive["discovery_ready"] is False and "stall_override" not in alive
    # Not stalled long enough — no override.
    early = apply_stall_override(
        _blocked_report(["relevant_query_routes"]),
        stalled_rounds=2, patience=3, instruments_dead=True)
    assert early["discovery_ready"] is False and "stall_override" not in early


# ── loop wiring: paraphrase filter, instrument health, grounded critic ──────
def _loop(tmp_path, replies=None):
    from spiral.config import Config
    from spiral.research_loop import ResearchLoop

    class FakeOl:
        providers = {}

        def __init__(self):
            self.calls = []

        def chat(self, model, messages, **kwargs):
            self.calls.append(messages)
            payload = (replies or [{}]).pop(0) if replies else {}
            return SimpleNamespace(
                text=json.dumps(payload), completion_tokens=3, raw={})

    return ResearchLoop("black hole tidal love numbers master equations",
                        workdir=tmp_path, cfg=Config(), ol=FakeOl())


def test_query_novelty_rejects_paraphrases_of_tried_searches(tmp_path):
    loop = _loop(tmp_path)
    loop.map["searches"] = [
        {"query": "kodama ishibashi master equations higher dimensional black holes"}]
    assert not loop._query_is_novel(
        "kodama ishibashi master equations higher-dimensional black holes")
    assert not loop._query_is_novel(
        "master equations kodama ishibashi black holes higher dimensional")
    assert loop._query_is_novel("confluent heun connection coefficients")


def test_instrument_health_flags_dead_retrieval(tmp_path):
    loop = _loop(tmp_path)
    loop.map["searches"] = [
        {"query": f"q{i}", "added": [],
         "retrieval": {"source_ok": False, "result_count": 0}}
        for i in range(6)
    ]
    loop.map["graph_rounds"] = [
        {"added": [], "health": {"successful_requests": 0, "failed_requests": 30}}]
    health = loop._instrument_health()
    assert health["instruments_dead"] is True
    live = _loop(tmp_path / "live")
    live.map["searches"] = [
        {"query": "q", "added": ["2401.00001"],
         "retrieval": {"source_ok": True, "result_count": 3}}]
    assert live._instrument_health()["instruments_dead"] is False


def test_assess_corpus_shows_relevant_window_full_index_and_state(tmp_path):
    from spiral.research_corpus import Paper
    replies = [{"sufficient": False,
                "missing": ["confluent Heun connection coefficients"],
                "resolved": [{"item": "master equations", "ids": ["2405.00005"]}],
                "searches": []}]
    loop = _loop(tmp_path, replies=replies)
    for i in range(1, 4):                              # seeds, inserted first
        loop.corpus.add(Paper(arxiv_id=f"1000.0000{i}", title=f"Unrelated seed {i}",
                              abstract="dark matter axion detection"), fetch=False)
    loop.corpus.add(Paper(                             # the relevant paper, inserted LAST
        arxiv_id="2405.00005",
        title="Master equations for black hole tidal Love numbers",
        abstract="master equations black hole tidal love numbers",
        text="\\begin{document} master equations for black hole tidal love numbers "
             * 20), fetch=False)
    loop.map["searches"] = [{"query": "black hole tidal love numbers"}]
    loop.map["corpus_assessments"] = [{
        "round": 1, "sufficient": False,
        "missing": ["Kodama-Ishibashi master equations"],
        "searches": [], "paper_count": 3,
        "known_ids": ["1000.00001", "1000.00002", "1000.00003"],
    }]

    data = loop.assess_corpus()

    user = loop.ol.calls[-1][-1]["content"]
    # Full index: every paper visible, including the one outside any window.
    assert "FULL INDEX (4 papers)" in user and "2405.00005" in user
    # The excerpt window is relevance-ranked: the relevant paper appears as [1].
    assert "[1] Master equations for black hole tidal Love numbers" in user
    # Cross-round state: previous gaps and the newly added paper are both shown.
    assert "PREVIOUSLY FLAGGED MISSING" in user and "Kodama-Ishibashi" in user
    assert "ADDED SINCE LAST ASSESSMENT (1 papers)" in user
    assert "ALREADY-TRIED QUERIES" in user
    # The verdict and the assessment record both persist.
    assert data["missing"] == ["confluent Heun connection coefficients"]
    record = loop.map["corpus_assessments"][-1]
    assert record["round"] == loop.state.round and "2405.00005" in record["known_ids"]


def test_evaluate_corpus_quality_applies_sticky_stall_override(tmp_path):
    """Mirror the real frozen run: a rich, on-topic corpus, healthy-but-zero-yield
    searches, a dead citation graph — and ONLY relevant_query_routes blocking."""
    from spiral.research_corpus import Paper
    loop = _loop(tmp_path)
    body = ("\\begin{document} black hole tidal love numbers master equations "
            "static response " * 60)
    for i in range(1, 13):
        loop.corpus.add(Paper(
            arxiv_id=f"24{i:02d}.{10000 + i}",
            title="Black hole tidal Love numbers and master equations",
            abstract="tidal love numbers master equations black holes",
            text=body, tex_path="x.tex", body_source="tex"), fetch=False)
    ids = list(loop.corpus.papers)
    healthy_dead_ends = [                              # distinct healthy families, 0 hits
        "confluent heun connection coefficients",
        "gregory laflamme instability spectrum",
        "lovelock perturbation stability criteria",
    ]
    loop.map["searches"] = (
        [{"query": "black hole tidal love numbers", "added": [],
          "retrieval": {"source_ok": True, "result_count": 2, "result_ids": ids[:2]}}]
        + [{"query": q, "added": [],
            "retrieval": {"source_ok": True, "result_count": 0, "result_ids": []}}
           for q in healthy_dead_ends]
        + [{"query": f"rate limited query {i}", "added": [],
            "retrieval": {"source_ok": False, "result_count": 0}} for i in range(6)]
    )
    loop.map["graph_rounds"] = [
        {"added": [], "health": {"requests": 30, "successful_requests": 0,
                                 "failed_requests": 30}}]

    blocked = loop.evaluate_corpus_quality(stalled_rounds=0)
    assert blocked["discovery_ready"] is False
    assert "relevant_query_routes" in blocked["blocking_reasons"]

    opened = loop.evaluate_corpus_quality(stalled_rounds=4)
    assert opened["discovery_ready"] is True
    assert opened["stall_override"]["overridden_checks"] == ["relevant_query_routes"]
    assert opened["novelty_ready"] is False

    # Sticky: the counter reset after the gate flip must not close the gate again
    # while the instruments are still dead.
    sticky = loop.evaluate_corpus_quality(stalled_rounds=1)
    assert sticky["discovery_ready"] is True and sticky["stall_override"]

    # The recorded artifact carries the limitation.
    saved = json.loads((tmp_path / "coverage-latest.json").read_text())
    assert saved["stall_override"]["overridden_checks"] == ["relevant_query_routes"]


if __name__ == "__main__":
    import inspect
    failed = 0
    module = sys.modules["__main__"]
    tmp_root = Path(tempfile.mkdtemp())
    for name, fn in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        kwargs = {}
        params = inspect.signature(fn).parameters
        if "tmp_path" in params:
            kwargs["tmp_path"] = tmp_root / name
            kwargs["tmp_path"].mkdir(parents=True, exist_ok=True)
        if "monkeypatch" in params:
            print(f"SKIP {name} (needs pytest monkeypatch)")
            continue
        try:
            fn(**kwargs)
            print(f"ok   {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
