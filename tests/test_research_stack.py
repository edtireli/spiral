"""spiralʳᵉˢᵉᵃʳᶜʰ stack — corpus, citations parsers, numeric lab, writer, and the
loop's wiring. Network + LLM are mocked; the deterministic verifiers run for real.
Runs standalone (`python tests/test_research_stack.py`) or under pytest.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── corpus ───────────────────────────────────────────────────────────────────
def test_parse_arxiv_id():
    from spiral.research_corpus import parse_arxiv_id
    assert parse_arxiv_id("http://arxiv.org/abs/2401.01234v2") == "2401.01234v2"
    assert parse_arxiv_id("arXiv:2312.09876") == "2312.09876"
    assert parse_arxiv_id("hep-th/9711200") == "hep-th/9711200"
    assert parse_arxiv_id("no id here") is None


def test_extract_tex_from_targz_strips_comments():
    from spiral.research_corpus import extract_tex_source
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = b"\\begin{equation} E=mc^2 \\end{equation}\n% a secret comment\nkeep this"
        ti = tarfile.TarInfo("main.tex"); ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    out = extract_tex_source(buf.getvalue())
    assert "E=mc^2" in out and "keep this" in out and "secret comment" not in out


def test_extract_tex_from_single_gzip():
    from spiral.research_corpus import extract_tex_source
    assert "section" in extract_tex_source(gzip.compress(b"\\section{Intro} body"))


def test_corpus_dedup_and_persist():
    from spiral.research_corpus import Corpus, Paper
    d = Path(tempfile.mkdtemp())
    c = Corpus(d)
    c.add(Paper(arxiv_id="2401.01234v1", title="A", authors=["Ann Lee"]), fetch=False)
    c.add(Paper(arxiv_id="2401.01234v2", title="A2"), fetch=False)   # same bare id → dedup
    assert len(c.papers) == 1
    c.save()
    assert Corpus(d).has("2401.01234")                               # reloads from disk
    assert "[1]" in Corpus(d).summaries()


# ── citations (pure parsers) ─────────────────────────────────────────────────
def test_parse_inspire_and_semantic_scholar():
    from spiral.citations import parse_inspire, parse_semantic_scholar, prior_art
    ins = parse_inspire({"hits": {"hits": [
        {"id": "451647", "metadata": {"titles": [{"title": "Large N"}],
         "authors": [{"full_name": "Maldacena, J."}], "earliest_date": "1997-11-27",
         "citation_count": 20000, "abstracts": [{"value": "the correspondence"}]}}]}})
    assert ins[0].title == "Large N" and ins[0].year == 1997 and ins[0].citations == 20000
    ss = parse_semantic_scholar({"data": [
        {"title": "A Theorem", "year": 2019, "authors": [{"name": "X Y"}],
         "citationCount": 5, "url": "u", "abstract": "a"}]})
    assert ss[0].title == "A Theorem" and ss[0].citations == 5
    assert parse_inspire({}) == [] and parse_semantic_scholar(None if False else {}) == []


# ── numeric lab ──────────────────────────────────────────────────────────────
def test_numeric_lab_runs_screens_and_checks():
    from spiral.numeric_lab import check_numeric_claim, run_python
    assert run_python("print(6*7)").stdout == "42"
    assert not run_python("import shutil; shutil.rmtree('/')").ok          # denylisted
    assert run_python("while True: pass", timeout=1).timed_out             # contained
    assert check_numeric_claim("print(sum(range(10))==45)").ok             # True on last line
    assert not check_numeric_claim("print(1==2)").ok


# ── writer ───────────────────────────────────────────────────────────────────
def test_writer_resolves_real_citations_drops_dangling():
    from spiral.research_corpus import Paper
    from spiral.research_writer import bibtex_from_corpus, build_document
    papers = [Paper(arxiv_id="2401.01234", title="Real Paper", authors=["Jane Roe"], published="2024-01-01")]
    bib, keymap = bibtex_from_corpus(papers)
    assert "@article" in bib and "2401.01234" in bib
    d = Path(tempfile.mkdtemp())
    body = "As shown \\cite{arXiv:2401.01234} and \\cite{arXiv:9999.99999} (absent)."
    tex = build_document("T", "abstract", body, papers, d).read_text()
    assert f"\\cite{{{keymap['2401.01234']}}}" in tex                      # real → resolved
    assert "9999.99999" not in tex and "\\cite{}" not in tex               # dangling → dropped


# ── loop wiring (mocked LLM + net, REAL verification) ────────────────────────
def test_loop_round_verifies_and_decides(monkeypatch):
    from spiral import citations, research_loop
    d = Path(tempfile.mkdtemp())
    loop = research_loop.ResearchLoop("test topic", workdir=d)
    monkeypatch.setattr(loop.corpus, "build", lambda q, k=8, on=None: [])
    monkeypatch.setattr(citations, "prior_art", lambda *a, **k: [])

    # a vetted proposal with one TRUE + one FALSE checkable claim (verification is REAL)
    monkeypatch.setattr(loop, "propose", lambda refine_rounds=2: {
        "question": "Is (x+1)^2 = x^2+2x+1?", "claims": [
            {"kind": "identity", "lhs": "(x+1)**2", "rhs": "x**2+2*x+1", "note": "true one"},
            {"kind": "identity", "lhs": "(x+1)**2", "rhs": "x**2+3*x+1", "note": "false one"},
        ]})
    replies = iter([
        json.dumps({"assessment": "verified", "novel": True, "action": "solved", "reason": "checked"}),
        "\\section{Result} It holds.",     # write() body
    ])
    monkeypatch.setattr(loop, "_think", lambda system, user, think=True: (next(replies), 5))

    state = loop.run(max_rounds=1)
    oks = [f for f in state.findings if f["ok"]]
    assert len(oks) == 1 and len([f for f in state.findings if not f["ok"]]) == 1
    assert oks[0]["backend"] == "sympy" and state.status == "solved"
    assert (d / "state.json").is_file()
    assert Path(loop.write()["tex"]).is_file()


def test_proposal_iterates_against_prior_art(monkeypatch):
    """A first-guess proposal that duplicates prior art is revised until the referee
    accepts it — so what reaches verification is vetted for novelty, not the first draft."""
    from spiral import citations, research_loop
    loop = research_loop.ResearchLoop("test topic", workdir=Path(tempfile.mkdtemp()))
    monkeypatch.setattr(citations, "prior_art", lambda *a, **k: [])
    replies = iter([
        json.dumps({"question": "old already-solved Q", "claims": []}),           # draft
        json.dumps({"verdict": "revise", "novelty": "duplicates prior art"}),     # referee: revise
        json.dumps({"question": "sharper novel Q", "claims": [{"kind": "zero", "expr": "x-x"}]}),  # refined
        json.dumps({"verdict": "accept", "novelty": "distinct"}),                 # referee: accept
    ])
    monkeypatch.setattr(loop, "_think", lambda system, user, think=True: (next(replies), 5))
    prop = loop.propose(refine_rounds=2)
    assert prop["question"] == "sharper novel Q" and prop.get("_vetted") is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
