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


def test_extract_tex_tolerates_truncated_arxiv_source():
    """A partial/corrupt arXiv e-print must degrade to abstract-only, not crash
    graph deepening with EOFError from tarfile/gzip."""
    from spiral.research_corpus import extract_tex_source
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = b"\\section{Intro} body"
        ti = tarfile.TarInfo("main.tex"); ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    assert extract_tex_source(buf.getvalue()[:20]) == ""


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


def test_corpus_enriches_graph_stub_with_search_metadata():
    from spiral.research_corpus import Corpus, Paper

    d = Path(tempfile.mkdtemp())
    c = Corpus(d)
    c.add(Paper(arxiv_id="2401.01234", title="stub"), fetch=False)
    stored = c.add(Paper(
        arxiv_id="2401.01234v2", title="A Full Descriptive Title",
        authors=["A. Author"], abstract="A complete abstract.",
        categories=["gr-qc"], published="2024-01-02"), fetch=False)
    assert len(c.papers) == 1
    assert stored.title == "A Full Descriptive Title"
    assert stored.abstract == "A complete abstract." and stored.categories == ["gr-qc"]


def test_malformed_pdf_fallback_is_quiet(tmp_path, capsys):
    from spiral.research_corpus import extract_pdf_text

    broken = tmp_path / "truncated.pdf"
    broken.write_bytes(b"%PDF-1.7\ntruncated without an EOF marker")

    assert extract_pdf_text(broken) == ""
    captured = capsys.readouterr()
    assert "EOF marker" not in captured.err


def test_research_reading_notes_are_cached_and_logged(tmp_path):
    from types import SimpleNamespace

    from spiral.config import Config
    from spiral.research_corpus import Paper
    from spiral.research_loop import ResearchLoop

    class FakeOl:
        providers = {}

        def __init__(self):
            self.models = []

        def chat(self, model, messages, **kwargs):
            self.models.append(model)
            return SimpleNamespace(text=json.dumps({
                "role_in_corpus": "master-equation source",
                "main_results": ["derives static perturbation equations"],
                "methods": ["gauge-invariant reduction"],
                "objects_equations": ["master equation"],
                "gaps_or_openings": ["tidal response not classified"],
                "confidence": "medium",
            }), completion_tokens=7, raw={})

    cfg = Config()
    cfg.research_notes_model = "notes-model"
    loop = ResearchLoop("black hole Love numbers", workdir=tmp_path, cfg=cfg, ol=FakeOl())
    loop.corpus.add(Paper(
        arxiv_id="2401.00001v1",
        title="Static Perturbations",
        abstract="We derive master equations.",
        text="\\section{Intro} We derive master equations. \\begin{equation} L\\Phi=0 \\end{equation}",
    ), fetch=False)

    notes = loop._ensure_reading_notes()
    notes_again = loop._ensure_reading_notes()

    assert notes[0]["role_in_corpus"] == "master-equation source"
    assert loop.ol.models == ["notes-model"]            # second call hit cache
    assert (tmp_path / "notes" / "papers" / "2401.00001.json").is_file()
    assert "paper-note" in (tmp_path / "thoughts.jsonl").read_text()
    assert notes_again[0]["source_hash"] == notes[0]["source_hash"]
    assert notes[0]["grounded"] is False and notes[0]["confidence"] == "low"

    from spiral.research_quality import verify_jsonl_hash_chain
    assert verify_jsonl_hash_chain(tmp_path / "thoughts.jsonl")["ok"] is True


def test_reading_note_accepts_only_exact_source_anchors(tmp_path):
    from types import SimpleNamespace

    from spiral.config import Config
    from spiral.research_corpus import Paper
    from spiral.research_loop import ResearchLoop

    class FakeOl:
        providers = {}

        def chat(self, model, messages, **kwargs):
            return SimpleNamespace(text=json.dumps({
                "role_in_corpus": "derivation",
                "main_results": ["derives the master equation"],
                "evidence": [
                    {"supports": "master equation", "anchor": "We derive the master equation"},
                    {"supports": "invented theorem", "anchor": "This phrase is not in the source"},
                ],
            }), prompt_tokens=10, completion_tokens=7, raw={})

    cfg = Config()
    cfg.research_notes_model = "notes-model"
    loop = ResearchLoop("master equations", workdir=tmp_path, cfg=cfg, ol=FakeOl())
    loop.corpus.add(Paper(
        arxiv_id="2401.99999", title="A source",
        text="\\section{Introduction} We derive the master equation from the action. " * 30,
        tex_path=str(tmp_path / "source.tex"), body_source="tex"), fetch=False)
    note = loop._ensure_reading_notes()[0]
    assert note["grounded"] is True and note["confidence"] == "medium"
    assert len(note["evidence"]) == 1 and note["rejected_evidence_count"] == 1


# ── citations (pure parsers) ─────────────────────────────────────────────────
def test_parse_inspire_and_semantic_scholar():
    from spiral.citations import novelty_digest, parse_inspire, parse_semantic_scholar, prior_art
    ins = parse_inspire({"hits": {"hits": [
        {"id": "451647", "metadata": {"titles": [{"title": "Large N"}],
         "authors": [{"full_name": "Maldacena, J."}], "earliest_date": "1997-11-27",
         "citation_count": 20000, "abstracts": [{"value": "the correspondence"}],
         "arxiv_eprints": [{"value": "hep-th/9711200"}]}}]}})
    assert ins[0].title == "Large N" and ins[0].year == 1997 and ins[0].citations == 20000
    assert ins[0].identifier == "hep-th/9711200"
    ss = parse_semantic_scholar({"data": [
        {"title": "A Theorem", "year": 2019, "authors": [{"name": "X Y"}],
         "citationCount": 5, "url": "u", "abstract": "a",
         "externalIds": {"ArXiv": "1901.00001"}}]})
    assert ss[0].title == "A Theorem" and ss[0].citations == 5
    assert ss[0].identifier == "1901.00001"
    assert "abstract: the correspondence" in novelty_digest(ins)
    assert parse_inspire({}) == [] and parse_semantic_scholar(None if False else {}) == []


def test_angle_audit_blocks_unread_identifiable_prior(monkeypatch, tmp_path):
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("novel classification", workdir=tmp_path)
    called = []
    monkeypatch.setattr(loop, "_think", lambda *a, **k: called.append(True) or ('{"verdict":"pursue"}', 1))
    audit = loop._audit_angle(
        {"question": "Is this classified?", "check_plan": "exact elimination"},
        [], {"ready": True},
        {"identifiable_prior_count": 2, "grounded_deep_reads": 0},
    )
    assert audit["verdict"] == "thin"
    assert called == []


def test_prior_art_report_distinguishes_healthy_no_hits_from_source_failure(monkeypatch):
    from spiral import citations

    def healthy(query, k=8, report=None):
        if report is not None:
            report.update({"source_ok": True, "result_count": 0})
        return []

    monkeypatch.setattr(citations, "inspire", healthy)
    monkeypatch.setattr(citations, "semantic_scholar", healthy)
    monkeypatch.setattr(citations, "arxiv_prior", healthy)
    report = {}
    assert citations.prior_art("unseen object", report=report) == []
    assert report["ready"] is True and report["healthy_source_count"] == 3

    def failed(query, k=8, report=None):
        if report is not None:
            report.update({"source_ok": False, "result_count": 0, "error": "offline"})
        return []

    monkeypatch.setattr(citations, "inspire", failed)
    monkeypatch.setattr(citations, "semantic_scholar", failed)
    monkeypatch.setattr(citations, "arxiv_prior", failed)
    failed_report = {}
    citations.prior_art("unseen object", report=failed_report)
    assert failed_report["ready"] is False and failed_report["healthy_source_count"] == 0


# ── numeric lab ──────────────────────────────────────────────────────────────
def test_numeric_lab_runs_screens_and_checks():
    from spiral.numeric_lab import check_numeric_claim, run_python
    assert run_python("print(6*7)").stdout == "42"
    assert not run_python("import shutil; shutil.rmtree('/')").ok          # denylisted
    assert run_python("while True: pass", timeout=1).timed_out             # contained
    assert check_numeric_claim("print(sum(range(10))==45)").ok             # True on last line
    assert not check_numeric_claim("print(1==2)").ok
    assert not check_numeric_claim("print(True)").ok                       # self-certification blocked


def test_research_workbench_runs_and_records_manifest():
    from spiral.research_workbench import run_workbench_claim
    root = Path(tempfile.mkdtemp())
    claim = {
        "kind": "workbench",
        "note": "tiny exact certificate",
        "files": {"check.py": "from sympy import expand, symbols\nx=symbols('x')\nassert expand((x+1)**2 - (x**2+2*x+1)) == 0\nprint('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
        "expect": "CERTIFICATE_OK",
    }
    res = run_workbench_claim(claim, root)
    assert res.ok and Path(res.manifest).is_file()
    data = json.loads(Path(res.manifest).read_text())
    assert data["ok"] is True and data["files"][0]["sha256"]


def test_research_workbench_treats_certificate_marker_as_success():
    from spiral.research_workbench import run_workbench_claim
    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "note": "marker with overstrict transcript",
        "files": {"check.py": (
            "cases = [i * i for i in range(6)]\n"
            "assert len(cases) == 6\n"
            "print('computed: 6 cases')\n"
            "print('CERTIFICATE_OK')\n"
        )},
        "cmd": "python check.py",
        "expect": "computed: 7 cases\nCERTIFICATE_OK",
    }, root)
    data = json.loads(Path(res.manifest).read_text())
    assert res.ok and data["marker_matched"] is True and data["expect_matched"] is False


def test_research_workbench_rejects_vacuous_success_marker():
    from spiral.research_workbench import run_workbench_claim

    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "note": "vacuous marker",
        "files": {"check.py": "print('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
    }, root)
    assert not res.ok and "vacuous" in res.detail


def test_research_workbench_runs_multistep_cpp_when_available():
    import shutil

    from spiral.research_workbench import run_workbench_claim
    if not shutil.which("c++"):
        import pytest
        pytest.skip("c++ compiler not installed")
    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "note": "tiny c plus plus certificate",
        "files": {"check.cpp": "#include <iostream>\nint main(){ if (2+2 != 4) return 1; std::cout << \"CERTIFICATE_OK\\n\"; }\n"},
        "steps": ["c++ -std=c++17 check.cpp -o check", "./check"],
        "expect": "CERTIFICATE_OK",
    }, root)
    data = json.loads(Path(res.manifest).read_text())
    assert res.ok and len(data["steps_run"]) == 2


def test_workbench_computational_grade_requires_observed_distinct_methods():
    from spiral.research_loop import ResearchLoop
    from spiral.research_workbench import run_workbench_claim

    root = Path(tempfile.mkdtemp())
    declared_only = {
        "kind": "workbench",
        "note": "metadata is not evidence",
        "files": {"check.py": "assert 2 + 2 == 4\nprint('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
        "validation": {
            "independent_methods": ["symbolic", "numeric"],
            "acceptance_criteria": ["residual is zero"],
        },
    }
    first = run_workbench_claim(declared_only, root)
    declared_only["manifest"] = first.manifest
    assert first.ok
    assert ResearchLoop._workbench_strength(declared_only, True) == "executable"

    authenticated = {
        "kind": "workbench",
        "note": "two observed methods",
        "files": {
            "symbolic.py": (
                "assert (2 + 2) == 4\n"
                "print('METHOD_OK:symbolic')\n"
                "print('CRITERION_OK:exact')\n"
            ),
            "enumerate.py": (
                "assert sum([1, 1, 1, 1]) == 4\n"
                "print('METHOD_OK:enumeration')\n"
                "print('CERTIFICATE_OK')\n"
            ),
        },
        "steps": ["python symbolic.py", "python enumerate.py"],
        "validation": {
            "independent_methods": [
                {"name": "symbolic", "step": 0, "marker": "METHOD_OK:symbolic"},
                {"name": "enumeration", "step": 1,
                 "marker": "METHOD_OK:enumeration"},
            ],
            "acceptance_criteria": [
                {"name": "exact", "step": 0, "marker": "CRITERION_OK:exact"},
            ],
        },
    }
    second = run_workbench_claim(authenticated, root)
    authenticated["manifest"] = second.manifest
    manifest = json.loads(Path(second.manifest).read_text())

    assert second.ok
    assert manifest["validation_evidence"]["computationally_reproduced"] is True
    assert ResearchLoop._workbench_strength(authenticated, True) == "computational"


def test_research_workbench_blocks_repo_clone_without_auto():
    from spiral.research_workbench import run_workbench_claim
    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "note": "repo needs approval",
        "files": {"check.py": (
            "values = [i + 1 for i in range(4)]\n"
            "assert sum(values) == 10\n"
            "print('CERTIFICATE_OK')\n"
        )},
        "cmd": "python check.py",
        "repos": [{"url": "https://github.com/example/example"}],
    }, root, allow_repos=False)
    assert not res.ok and "repo acquisition requires" in res.detail


def test_repository_acquisition_environment_excludes_credentials(monkeypatch, tmp_path):
    from spiral.research_workbench import _acquisition_env

    monkeypatch.setenv("MOONSHOT_API_KEY", "private")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/agent.sock")
    env = _acquisition_env(tmp_path)

    assert "MOONSHOT_API_KEY" not in env and "SSH_AUTH_SOCK" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"


def test_research_workbench_rejects_unsafe_files():
    from spiral.research_workbench import run_workbench_claim
    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "files": {"../escape.py": "print('CERTIFICATE_OK')"},
        "cmd": "python escape.py",
    }, root)
    assert not res.ok and "unsafe certificate path" in res.detail


def test_research_workbench_rejects_subprocess_code():
    from spiral.research_workbench import run_workbench_claim
    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "files": {"check.py": "import subprocess\nsubprocess.run(['python', '-V'])\nprint('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
    }, root)
    assert not res.ok and "blocked text" in res.detail


def test_research_workbench_rejects_unapproved_python_requirement():
    from spiral.research_workbench import run_workbench_claim

    root = Path(tempfile.mkdtemp())
    res = run_workbench_claim({
        "kind": "workbench",
        "files": {"check.py": "assert 2 + 2 == 4\nprint('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
        "requirements": ["definitely-not-an-approved-research-package"],
    }, root)

    assert not res.ok and "approved research package set" in res.detail


# ── writer ───────────────────────────────────────────────────────────────────
def test_research_progress_plan_shape():
    from spiral.research_ui import research_plan
    plan = research_plan()
    assert plan.task_count == 12
    assert [m.title for m in plan.milestones] == ["corpus coverage", "claim loop", "paper"]


def test_cli_research_resume_loads_saved_topic():
    from argparse import Namespace

    from rich.console import Console

    from spiral.cli import _load_research_topic
    root = Path(tempfile.mkdtemp())
    state_dir = root / "spiral-research"
    state_dir.mkdir()
    (state_dir / "state.json").write_text(json.dumps({"topic": "saved topic", "round": 1}))
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)

    assert _load_research_topic(Namespace(query=None, resume=True, dir=str(root)), console) == "saved topic"
    assert _load_research_topic(Namespace(query="different topic", resume=True, dir=str(root)), console) == "saved topic"
    assert "ignoring supplied query" in buf.getvalue()


def test_writer_resolves_real_citations_drops_dangling():
    from spiral.research_corpus import Paper
    from spiral.research_writer import (
        audit_body,
        bibtex_from_corpus,
        blueprint_markdown,
        build_document,
        corpus_style_profile,
        corpus_style_guide,
        corpus_writing_blueprint,
        normalise_outline,
        normalise_section_fragment,
        validate_model_blueprint,
    )
    papers = [Paper(arxiv_id="2401.01234", title="Real Paper", authors=["Jane Roe"], published="2024-01-01")]
    bib, keymap = bibtex_from_corpus(papers)
    assert "@article" in bib and "2401.01234" in bib
    d = Path(tempfile.mkdtemp())
    body = "```latex\n\\begin{document}\\section{Result} As shown \\cite{arXiv:2401.01234} and \\cite{arXiv:9999.99999} (absent).\\end{document}\n```"
    tex = build_document("T", "abstract", body, papers, d).read_text()
    assert f"\\cite{{{keymap['2401.01234']}}}" in tex                      # real → resolved
    assert "9999.99999" not in tex and "\\cite{}" not in tex               # dangling → dropped
    assert tex.count("\\begin{document}") == 1 and "```" not in tex         # wrappers stripped
    tex = build_document("T", "abstract", body, papers, d,
                         author="Edis Devin Tireli",
                         association="Department of Neuroscience, University of Copenhagen, Denmark").read_text()
    assert "Edis Devin Tireli" in tex and "University of Copenhagen" in tex
    tex = build_document("U(1)^2 & sigma_models", "A&B in U(1)^2", body, papers, d).read_text()
    assert "U(1)\\textasciicircum{}2 \\& sigma\\_models" in tex
    tex = build_document("Identity $x^2$", "The identity $x^2=x\\cdot x$ is exact.",
                         body, papers, d).read_text()
    assert "Identity $x^2$" in tex and "$x^2=x\\cdot x$" in tex
    styled = [Paper(arxiv_id="1", title="S", text=(
        "\\section{Introduction}\\section{Preliminaries}\\begin{definition}X\\end{definition}"
        "\\begin{theorem}T\\end{theorem}\\begin{proof}P\\end{proof} Throughout we prove this."))]
    prof = corpus_style_profile(styled)
    guide = corpus_style_guide(styled)
    blueprint = corpus_writing_blueprint(styled)
    bp_md = blueprint_markdown(blueprint)
    assert "Introduction" in guide and prof["environments"]["theorem"] == 1
    assert "Notation And Conventions" in bp_md and "theorem" in guide.lower()
    assert blueprint["section_template"] and "macros" in blueprint["notation_ledger"]
    assert "missing section structure" in audit_body("plain text", papers, [{"ok": True}])
    outline = normalise_outline({"title": "T", "sections": [{"name": "Results"}]}, blueprint)
    assert [s["rhetorical_role"] for s in outline["sections"]] == [
        "introduction", "setup", "results", "discussion"]
    fragment = normalise_section_fragment(
        "\\section{Wrong}Text\\section{Extra}More", "Setup")
    assert fragment.startswith("\\section{Setup}") and "\\subsection{Extra}" in fragment
    outline = normalise_outline({"title": "T", "sections": [
        {"name": "Algebraic Setup"}, {"name": "Appendix"}, {"name": "Results"}]}, blueprint)
    assert [s["rhetorical_role"] for s in outline["sections"]] == [
        "introduction", "setup", "results", "discussion"]
    validated, issues = validate_model_blueprint({
        "notation_plan": [
            {"concept": "metric", "chosen_symbol": "g", "definition": "target metric"},
            {"concept": "coupling", "chosen_symbol": "g", "definition": "coupling"},
        ],
        "citation_plan": [{"paper": "9999.99999", "use": "invented"}],
    }, blueprint)
    assert len(validated["notation_plan"]) == 1
    assert any("duplicate symbol" in issue for issue in issues)
    assert any("unknown paper" in issue for issue in issues)
    setup_body = (
        "\\section{Introduction}We state the problem. "
        "\\section{Setup}Let $R$ denote a ring and let $x\\in R$. "
        "\\section{Results}The encoded residual is verified exactly. "
        "\\section{Discussion}The evidence concerns the stated expression. "
        "\\section{Conclusion}The verified statement follows. "
        + ("Additional precise exposition of the definitions and calculation. " * 55)
        + "\\cite{arXiv:1}"
    )
    assert "does not state notation/conventions" not in " ".join(
        audit_body(setup_body, styled, [{"ok": True, "strength": "exact"}], blueprint))


def test_writer_detects_repeated_sections_and_never_accepts_stale_pdf(tmp_path):
    from spiral.research_corpus import Paper
    from spiral.research_writer import audit_body, build_document, compile_pdf

    repeated = ("This exact derivation expands the polynomial and checks its residual. " * 20)
    body = (
        "\\section{Introduction}" + repeated
        + "\\section{Results}" + repeated
        + "\\section{Discussion}" + ("The limitations concern only the encoded statement. " * 20)
        + "\\section{Conclusion}" + ("The verified result is recorded conservatively. " * 20)
    )
    assert any("substantial repetition" in issue
               for issue in audit_body(body, [], [{"ok": True, "strength": "exact"}]))
    bypass = body.replace("\\section{Introduction}", "\\section{Introduction} See [1]. ")
    assert any("literal numeric citation" in issue
               for issue in audit_body(bypass, [], [{"ok": True, "strength": "exact"}]))

    clean_body = (
        "\\section{Introduction}We motivate an exact algebraic calculation. "
        "\\section{Results}The verified residual vanishes identically. "
        "\\section{Discussion}The certificate covers only the encoded expression. "
        "\\section{Conclusion}The exact check is reproducible."
    )
    tex = build_document("Compile gate", "A valid abstract.", clean_body, [], tmp_path)
    pdf, error = compile_pdf(tex)
    assert pdf and not error
    tex.write_text("\\documentclass{article}\\begin{document}\\badcommand")
    pdf, error = compile_pdf(tex)
    assert pdf is None and error and not (tmp_path / "paper.pdf").exists()

    cited_dir = tmp_path / "cited"
    paper = Paper(arxiv_id="2401.00001", title="Held source", authors=["A Author"],
                  published="2024-01-01", abstract="An exact held source.")
    cited_body = clean_body.replace(
        "exact algebraic calculation", "exact algebraic calculation \\cite{arXiv:2401.00001}")
    cited_tex = build_document("Cited compile gate", "A valid abstract with $x^2$.",
                               cited_body, [paper], cited_dir)
    cited_pdf, cited_error = compile_pdf(cited_tex)
    assert cited_pdf and not cited_error and (cited_dir / "paper.bbl").is_file()


def test_citation_support_audit_requires_exact_held_anchor():
    from spiral.research_corpus import Paper
    from spiral.research_writer import citation_evidence_packet, validate_citation_audit

    paper = Paper(
        arxiv_id="2401.01234",
        title="Spectral reductions",
        abstract="We derive a rational Lax representation for the isotropic coupling locus.",
        text="We derive a rational Lax representation for the isotropic coupling locus.",
    )
    body = (
        "\\section{Introduction}\n"
        "A rational Lax representation is known on the isotropic locus "
        "\\cite{arXiv:2401.01234}."
    )
    packet = citation_evidence_packet(body, [paper])
    context = packet["contexts"][0]
    anchor = packet["sources"][0]["anchors"][0]["source_anchor"]
    audit = {"citations": [{
        "context_id": context["context_id"], "paper": paper.bare_id,
        "supported": True, "source_anchor": anchor,
    }]}

    assert validate_citation_audit(packet, audit, [paper]) == []
    audit["citations"][0]["source_anchor"] = "An invented sentence absent from the source."
    assert "invented or unverified" in validate_citation_audit(packet, audit, [paper])[0]
    audit["citations"][0].update({"source_anchor": anchor, "supported": False})
    assert "not supported" in validate_citation_audit(packet, audit, [paper])[0]

    empty = citation_evidence_packet("\\section{Introduction}No citation yet.", [paper])
    assert empty["contexts"] == []
    assert empty["sources"][0]["paper"] == paper.bare_id
    assert empty["sources"][0]["anchors"]


def test_claim_scope_audit_requires_disposition_and_valid_evidence():
    from spiral.research_writer import (
        claim_scope_packet,
        claims_requiring_escalation,
        merge_claim_scope_audits,
        validate_claim_scope_audit,
    )

    finding = {
        "claim_id": "exact-1", "strength": "exact", "backend": "sympy",
        "claim": {"kind": "identity", "lhs": "(x+1)**2", "rhs": "x**2+2*x+1"},
        "detail": "symbolic residual is zero",
    }
    packet = claim_scope_packet(
        "The identity fails in every noncommutative unital ring because the unit need not commute.",
        [finding], {})
    claim_id = packet["claims"][0]["claim_id"]
    assert packet["claims"][0]["escalate"] is True
    unsupported = {"claims": [{"claim_id": claim_id, "status": "unsupported",
                                "evidence_id": "", "reason": "not established"}]}
    assert "unsupported" in validate_claim_scope_audit(packet, unsupported)[0]
    hidden = {"claims": [{"claim_id": claim_id, "status": "nonclaim",
                           "evidence_id": "", "reason": ""}]}
    assert "high-risk" in validate_claim_scope_audit(packet, hidden)[0]
    valid = {"claims": [{"claim_id": claim_id, "status": "verified",
                          "evidence_id": "finding:exact-1", "reason": "fixture"}]}
    assert validate_claim_scope_audit(packet, valid) == []
    assert claims_requiring_escalation(packet, valid) == packet["claims"]

    # An escalated sentence cannot inherit a convenient fast verdict when the
    # strongest reviewer omitted it.
    merged = merge_claim_scope_audits(valid, {"claims": []}, {claim_id})
    assert "was not audited" in validate_claim_scope_audit(packet, merged)[0]

    ordinary = claim_scope_packet(
        "The residual $x-x=0$ is verified by direct symbolic reduction.", [finding], {})
    ordinary_id = ordinary["claims"][0]["claim_id"]
    ordinary_audit = {"claims": [{
        "claim_id": ordinary_id, "status": "verified", "evidence_id": "finding:exact-1",
    }]}
    assert ordinary["claims"][0]["escalate"] is False
    assert claims_requiring_escalation(ordinary, ordinary_audit) == ordinary["claims"]

    empirical_finding = {
        **finding, "claim_id": "empirical-1", "strength": "empirical",
    }
    empirical = claim_scope_packet(
        "The response coefficient vanishes for all tested dimensions.",
        [empirical_finding], {})
    empirical_id = empirical["claims"][0]["claim_id"]
    overstated = {"claims": [{
        "claim_id": empirical_id, "status": "verified",
        "evidence_id": "finding:empirical-1",
    }]}
    assert "presents empirical evidence as proof" in validate_claim_scope_audit(
        empirical, overstated)[0]


def test_claim_scope_protocol_evidence_only_supports_bounded_search_statements():
    from spiral.research_writer import claim_scope_packet, validate_claim_scope_audit

    protocol = [{
        "evidence_id": "protocol:prior-art",
        "kind": "bounded prior-art search",
        "scope": "three queries over two healthy databases",
        "queries": ["one", "two", "three"],
        "sources": ["arxiv", "semantic_scholar"],
        "result_count": 20,
    }]
    bounded = claim_scope_packet(
        "Our documented search did not locate a complete classification of this ansatz.",
        [], {}, protocol)
    claim_id = bounded["claims"][0]["claim_id"]
    audit = {"claims": [{
        "claim_id": claim_id, "status": "protocol_supported",
        "evidence_id": "protocol:prior-art",
    }]}
    assert validate_claim_scope_audit(bounded, audit) == []

    absolute = claim_scope_packet(
        "This is the first classification and is novel in the published literature.",
        [], {}, protocol)
    absolute_id = absolute["claims"][0]["claim_id"]
    bad = {"claims": [{
        "claim_id": absolute_id, "status": "protocol_supported",
        "evidence_id": "protocol:prior-art",
    }]}
    issues = validate_claim_scope_audit(absolute, bad)
    assert any("unbounded protocol conclusion" in issue for issue in issues)


def test_claim_scope_distinguishes_performative_setup_from_guarantees():
    from spiral.research_writer import (
        claim_scope_packet, claims_requiring_escalation, validate_claim_scope_audit)

    packet = claim_scope_packet(
        "We work exclusively inside the integer polynomial ring $\\mathbb{Z}[x]$ for this note.",
        [], {})
    row = packet["claims"][0]
    assert row["performative"] is True
    audit = {"claims": [{
        "claim_id": row["claim_id"], "status": "nonclaim", "evidence_id": "",
    }]}
    assert validate_claim_scope_audit(packet, audit) == []
    assert claims_requiring_escalation(packet, audit) == []

    guarantee = claim_scope_packet(
        "We work inside the integer polynomial ring $\\mathbb{Z}[x]$, ensuring that every "
        "implementation yields the same result.",
        [], {})
    risky = guarantee["claims"][0]
    assert risky["performative"] is False
    bad = {"claims": [{
        "claim_id": risky["claim_id"], "status": "nonclaim", "evidence_id": "",
    }]}
    assert "labelled nonclaim" in validate_claim_scope_audit(guarantee, bad)[0]


def test_deterministic_structure_gate_overrides_only_static_referee_noise():
    from spiral.research_writer import reconcile_referee_audit

    noisy = {"verdict": "revise", "issues": [
        "Section heading Setup should be called Preliminaries.",
        "The claimed theorem is stronger than the verified evidence.",
    ]}
    reconciled = reconcile_referee_audit(noisy, [])
    assert reconciled["verdict"] == "revise"
    assert reconciled["issues"] == [
        "The claimed theorem is stronger than the verified evidence."]
    assert len(reconciled["ignored_deterministic_issues"]) == 1

    static_only = reconcile_referee_audit({
        "verdict": "revise", "issues": ["Section count mismatch: six sections."]}, [])
    assert static_only["verdict"] == "accept" and static_only["issues"] == []

    genuinely_broken = reconcile_referee_audit({
        "verdict": "revise", "issues": ["Missing section: Results."]},
        ["missing results section"])
    assert genuinely_broken["verdict"] == "revise"

    expository = reconcile_referee_audit({
        "verdict": "revise",
        "issues": [
            "This trivial high-school identity is inappropriate for an arXiv research contribution."
        ],
        "instructions": "invent novelty",
    }, [], expository=True)
    assert expository["verdict"] == "accept"
    assert expository["ignored_deterministic_issues"]

    held_source = reconcile_referee_audit({
        "verdict": "revise",
        "issues": [
            "Citation arXiv:2401.00001 appears to be a non-existent placeholder.",
            "The Results heading might be better as Main Result because it contains a proof.",
        ],
    }, [], held_citation_ids={"2401.00001"})
    assert held_source["verdict"] == "accept"
    assert len(held_source["ignored_deterministic_issues"]) == 2

    unknown_source = reconcile_referee_audit({
        "verdict": "revise",
        "issues": ["Citation arXiv:9999.99999 appears to be non-existent."],
    }, [], held_citation_ids={"2401.00001"})
    assert unknown_source["verdict"] == "revise"


def test_confirmed_source_overlap_is_removed_without_losing_section_heading():
    from spiral.research_corpus import Paper
    from spiral.research_writer import (
        remove_suspicious_overlap_sentences,
        suspicious_phrase_overlap,
    )

    copied = (
        "Elementary polynomial identities follow from the algebraic laws defining a "
        "polynomial ring. We record the assumptions and make each reduction explicit."
    )
    source = Paper(arxiv_id="2401.00001", text=copied)
    body = (
        "\\section{Introduction}\n" + copied
        + " A separately worded sentence explains the present computation.\n"
        "\\section{Results}\nThe encoded residual reduces exactly to zero."
    )
    assert suspicious_phrase_overlap(body, [source])

    cleaned, removed = remove_suspicious_overlap_sentences(body, [source])

    assert removed
    assert "\\section{Introduction}" in cleaned
    assert "\\section{Results}" in cleaned
    assert not suspicious_phrase_overlap(cleaned, [source])
    assert "separately worded sentence" in cleaned


def test_unsupported_scope_sentences_are_deleted_without_losing_heading():
    from spiral.research_writer import (
        claim_scope_packet,
        remove_unsupported_claim_sentences,
    )

    body = (
        "\\section{Results}\n"
        "The encoded residual reduces exactly to zero. "
        "This procedure ensures universal auditability for every possible implementation.\n"
        "\\section{Discussion}\n"
        "The verified equation is retained."
    )
    packet = claim_scope_packet(body, [])
    rejected = next(
        row for row in packet["claims"] if "universal auditability" in row["sentence"])
    audit = {
        "claims": [{
            "claim_id": rejected["claim_id"],
            "status": "unsupported",
            "evidence_id": "",
        }]
    }

    cleaned, removed = remove_unsupported_claim_sentences(body, packet, audit)

    assert len(removed) == 1
    assert "universal auditability" not in cleaned
    assert "encoded residual" in cleaned
    assert "\\section{Results}" in cleaned
    assert "\\section{Discussion}" in cleaned


def test_research_graph_export_builds_browser_view():
    from spiral.research_corpus import Corpus, Paper
    from spiral.research_graph import build_graph_data, write_graph_view
    root = Path(tempfile.mkdtemp())
    corpus = Corpus(root / "corpus")
    corpus.add(Paper(arxiv_id="2401.00001", title="Seed Paper", authors=["A B"]), fetch=False)
    map_data = {
        "topic": "graph topic",
        "searches": [{"round": 1, "query": "seed query", "categories": ["math.NT"], "added": ["2401.00001"]}],
        "graph_rounds": [{
            "research_round": 1,
            "seeds": ["2401.00001"],
            "edges": [{"source": "2401.00001", "target": "2401.00002", "direction": "references", "title": "Missing Foundation"}],
            "holes": [{"id": "2401.00002", "count": 2, "title": "Missing Foundation"}],
            "recent": [],
            "added": [],
        }],
    }
    data = build_graph_data(map_data, corpus)
    out = write_graph_view(map_data, corpus, root)
    page = Path(out["html"]).read_text()
    assert data["counts"]["search"] == 1 and data["counts"]["hole"] == 1
    assert page.startswith("<!doctype html>")
    assert "ResizeObserver" in page and "minimumZoom" in page
    assert "height: 100dvh" in page and "radial-gradient" not in page
    assert Path(out["data"]).is_file()


# ── loop wiring (mocked LLM + net, REAL verification) ────────────────────────
def test_research_loop_resume_is_explicit():
    from spiral.research_corpus import Corpus, Paper
    from spiral.research_loop import ResearchLoop
    d = Path(tempfile.mkdtemp())
    old_corpus = Corpus(d / "corpus")
    old_corpus.add(Paper(arxiv_id="2401.00001", title="old paper"), fetch=False)
    old_corpus.save()
    (d / "state.json").write_text(json.dumps({
        "topic": "old topic",
        "question": "old question",
        "round": 2,
        "status": "open",
        "findings": [],
        "corpus_ids": ["2401.00001"],
        "history": [],
        "tokens": 10,
    }))
    (d / "research-map.json").write_text(json.dumps({
        "topic": "old topic",
        "searches": [{"query": "old search"}],
        "graph_rounds": [],
    }))

    fresh = ResearchLoop("new topic", workdir=d, resume=False)
    assert fresh.state.topic == "new topic" and fresh.state.round == 0
    assert fresh.corpus.papers == {} and fresh.map["searches"] == []

    resumed = ResearchLoop("ignored topic", workdir=d, resume=True)
    assert resumed.state.topic == "old topic" and resumed.state.round == 2
    assert "2401.00001" in resumed.corpus.papers and resumed.map["searches"][0]["query"] == "old search"


def test_research_loop_migrates_null_legacy_resume_fields(tmp_path):
    from spiral.research_loop import ResearchLoop

    (tmp_path / "state.json").write_text(json.dumps({
        "topic": "legacy topic",
        "round": "2",
        "findings": None,
        "history": None,
        "corpus_ids": None,
        "active_proposal": None,
        "coverage": None,
        "completion": None,
        "tokens": None,
    }))
    (tmp_path / "research-map.json").write_text(json.dumps({
        "topic": "legacy topic", "searches": None, "graph_rounds": None,
    }))

    resumed = ResearchLoop("ignored", workdir=tmp_path, resume=True)

    assert resumed.state.round == 2
    assert resumed.state.findings == [] and resumed.state.history == []
    assert resumed.state.corpus_ids == []
    assert resumed.state.active_proposal == {}
    assert resumed.state.coverage == {} and resumed.state.completion == {}
    assert resumed.state.tokens == 0
    assert resumed.map["searches"] == [] and resumed.map["graph_rounds"] == []


def test_resume_with_green_completion_skips_new_research_round(monkeypatch, tmp_path):
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("finished topic", workdir=tmp_path, resume=False)
    loop.state.status = "solved"
    loop.state.round = 3
    loop.state.completion = {"ready": True, "checks": {"all": True}}
    loop._save()

    resumed = ResearchLoop("ignored", workdir=tmp_path, resume=True)
    monkeypatch.setattr(
        resumed, "search_plan",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not search again")),
    )
    state = resumed.run()

    assert state.status == "solved"
    assert state.round == 3


def test_think_recovers_from_empty_model_with_compact_prompt():
    from types import SimpleNamespace

    from spiral import research_loop

    messages = []

    class FakeOl:
        providers = {}

        def __init__(self):
            self.calls = []

        def chat(self, model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return SimpleNamespace(text="", completion_tokens=0, raw={})
            return SimpleNamespace(text='{"ok": true}', completion_tokens=3, raw={})

    fake = FakeOl()
    loop = research_loop.ResearchLoop("test topic", workdir=Path(tempfile.mkdtemp()),
                                      ol=fake, ui=messages.append)
    big_corpus = "\n".join(f"[{i}] Paper {i}\n" + ("x" * 1600) for i in range(30))
    text, toks = loop._think("Return JSON.", "QUESTION: q\n\nCORPUS:\n" + big_corpus)

    assert text == '{"ok": true}' and toks == 3
    assert len(fake.calls) == 2
    assert len(fake.calls[1][0][1]["content"]) < len(fake.calls[0][0][1]["content"])
    assert any("recovered with compact prompt" in m for m in messages)


def test_extract_json_repairs_one_missing_closing_brace():
    from spiral.research_loop import _extract_json
    text = '{"kind":"workbench","files":{"check.py":"print(1)","cmd":"python check.py"}'
    data = _extract_json(text)
    assert data["kind"] == "workbench" and data["files"]["cmd"] == "python check.py"


def test_extract_json_escapes_raw_newline_inside_code_string():
    from spiral.research_loop import _extract_json
    text = '{"kind":"workbench","files":{"check.py":"print(1)\nprint(2)","cmd":"python check.py"}'
    data = _extract_json(text)
    assert data["files"]["check.py"] == "print(1)\nprint(2)"


def test_think_uses_role_default_thinking():
    from types import SimpleNamespace

    from spiral import research_loop

    class FakeOl:
        providers = {}

        def __init__(self):
            self.kwargs = []

        def chat(self, model, messages, **kwargs):
            self.kwargs.append(kwargs)
            return SimpleNamespace(text="ok", completion_tokens=1, raw={})

    fake = FakeOl()
    loop = research_loop.ResearchLoop("test topic", workdir=Path(tempfile.mkdtemp()), ol=fake)
    loop._think("system", "user", role="critic")

    assert fake.kwargs[0]["think"] is True


def test_structured_scientific_decision_can_request_deep_reasoning(tmp_path):
    from types import SimpleNamespace

    from spiral.research_loop import ResearchLoop

    class FakeOl:
        providers = {}

        def __init__(self):
            self.kwargs = []

        def chat(self, model, messages, **kwargs):
            self.kwargs.append(kwargs)
            return SimpleNamespace(
                text='{"decision":"continue"}', thinking="private trace",
                prompt_tokens=4, completion_tokens=5, raw={})

    fake = FakeOl()
    loop = ResearchLoop("test topic", workdir=tmp_path, ol=fake)

    result = loop._think_json(
        "Make a scientific decision.", "VISIBLE EVIDENCE", role="critic",
        max_tokens=512, reasoning=True)

    assert result == {"decision": "continue"}
    assert fake.kwargs[0]["think"] is True
    assert fake.kwargs[0]["num_predict"] >= 8192
    row = json.loads((tmp_path / "model-calls.jsonl").read_text())
    assert row["reasoning_requested"] is True
    assert row["private_reasoning_chars"] == len("private trace")
    assert len(row["private_reasoning_sha256"]) == 64


def test_kimi_k3_provider_uses_reasoning_and_completion_controls(monkeypatch):
    from spiral.llm import Ollama

    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"ok":true}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response()

    ol = Ollama(providers={"kimi-k3": {
        "base_url": "https://api.moonshot.ai/v1", "api_key_env": "TEST_KIMI_KEY",
    }})
    client = Client()
    ol._client = client
    monkeypatch.setenv("TEST_KIMI_KEY", "secret")
    result = ol.chat("kimi-k3", [{"role": "user", "content": "json"}],
                     think=False, num_predict=8192, fmt="json")
    ol.chat("kimi-k3", [{"role": "user", "content": "research"}],
            think=True, num_predict=16384)

    structured = client.calls[0][1]["json"]
    research = client.calls[1][1]["json"]
    assert result.text == '{"ok":true}'
    assert structured["reasoning_effort"] == "low"
    assert structured["max_completion_tokens"] == 32768
    assert "max_tokens" not in structured
    assert research["reasoning_effort"] == "max"
    assert research["max_completion_tokens"] == 131072


def test_api_tier_preserves_independent_local_research_auditor():
    import io

    from rich.console import Console

    from spiral.cli import _apply_tier
    from spiral.config import Config

    cfg = Config()
    local_auditor = cfg.research_auditor.name
    cfg.providers = {"api-reasoner": {
        "base_url": "https://example.invalid/v1",
        "api_key_env": "UNUSED_TEST_KEY",
    }}
    _apply_tier(cfg, Console(file=io.StringIO(), force_terminal=False), "api")

    assert {cfg.worker.name, cfg.planner.name, cfg.critic.name, cfg.escalation.name} == {
        "api-reasoner"
    }
    assert cfg.research_auditor.name == local_auditor


def test_loop_round_verifies_and_decides(monkeypatch):
    from types import SimpleNamespace

    from spiral import citations, research_loop
    d = Path(tempfile.mkdtemp())
    loop = research_loop.ResearchLoop("test topic", workdir=d, mode="expository")
    loop.cfg.research_min_grounded_notes = 0
    loop.cfg.research_min_grounded_deep_reads = 0
    loop.cfg.visual_review = False
    monkeypatch.setattr(loop.corpus, "build", lambda q, k=8, categories=None, on=None: [])
    monkeypatch.setattr(loop.corpus, "graph_deepen", lambda **k: {"added": 0, "saturated": True, "holes": []})
    monkeypatch.setattr(loop, "search_plan", lambda n=3: (["math.NT"], ["ramanujan series"]))
    monkeypatch.setattr(loop, "assess_corpus", lambda: {"sufficient": True})
    monkeypatch.setattr(loop, "evaluate_corpus_quality", lambda **kwargs: (
        setattr(loop.state, "coverage", {"discovery_ready": True, "novelty_ready": True})
        or loop.state.coverage))
    def ready_novelty(q):
        loop._last_novelty_report = {"ready": True}
        return []
    monkeypatch.setattr(loop, "novelty", ready_novelty)

    # a proposal with one TRUE + one FALSE checkable claim (verification is REAL sympy)
    proposal = {
        "question": "Is (x+1)^2 = x^2+2x+1?", "claims": [
            {"kind": "identity", "lhs": "(x+1)**2", "rhs": "x**2+2*x+1",
             "note": "true one", "statement": "The polynomial identity holds",
             "assumptions": ["x is a commutative symbol"],
             "falsifier": "a nonzero expanded residual", "method_family": "exact symbolic algebra",
             "required": True},
            {"kind": "identity", "lhs": "(x+1)**2", "rhs": "x**2+3*x+1",
             "note": "false one", "statement": "The altered polynomial identity holds",
             "assumptions": ["x is a commutative symbol"],
             "falsifier": "a nonzero expanded residual", "method_family": "exact symbolic algebra",
             "required": False},
        ], "_vetted": True,
    }
    def fixture_proposal(refine_rounds=2):
        loop._register_proposal_obligations(proposal)
        return proposal
    monkeypatch.setattr(loop, "propose", fixture_proposal)
    loop._audit_model_call(
        model="fixture", role="planner", variant="unit-test",
        system="fixture", user="fixture",
        result=SimpleNamespace(text="fixture", thinking="", prompt_tokens=1,
                               completion_tokens=1, raw={}),
    )
    # the referee says solved — but the loop must only accept it BECAUSE a claim verified
    monkeypatch.setattr(loop, "reflect", lambda findings, priors:
                        {"action": "solved", "novel": True, "reason": "checked"})

    state = loop.run(max_rounds=1)
    oks = [f for f in state.findings if f["ok"]]
    assert len(oks) == 1 and len([f for f in state.findings if not f["ok"]]) == 1
    assert oks[0]["backend"] == "sympy" and state.status == "solved"
    assert (d / "state.json").is_file() and (d / "journal.md").is_file()   # persisted + journalled
    paper_body = (
        "\\section{Introduction}" + (
            "Polynomial identities motivate a precise comparison between algebraic syntax "
            "and exact normalization. The introduction states the question and its scope. " * 16)
        + "\\section{Setup}" + (
            "Let the symbolic variable range over the commutative expression domain. "
            "Notation and assumptions are fixed before evaluating the polynomial residual. " * 16)
        + "\\section{Results}" + (
            "Expansion of the square gives three terms after collecting like monomials. "
            "The symbolic residual vanishes, so the encoded identity is exactly verified. " * 16)
        + "\\section{Discussion}" + (
            "The certificate concerns the supplied commutative symbolic expression only. "
            "No broader structural classification or numerical conjecture is asserted here. " * 16)
        + "\\section{Conclusion}" + (
            "The calculation is reproducible from the recorded expression and backend. "
            "This conclusion summarizes the established result without adding a new claim. " * 16)
    )
    def fake_writer(system, user, think=True, **kwargs):
        if "claim-scope referee" in system.lower():
            packet = research_loop._extract_json(user)
            evidence = (packet.get("verified_evidence") or [{"evidence_id": ""}])[0]["evidence_id"]
            return (json.dumps({"claims": [
                {"claim_id": claim["claim_id"], "status": "verified",
                 "evidence_id": evidence, "reason": "test fixture maps prose to exact finding"}
                for claim in (packet.get("claims") or [])
            ]}), 5)
        if "arxiv referee" in system.lower():
            return (json.dumps({"verdict": "accept", "issues": [],
                                "instructions": "publication gate passed"}), 5)
        if "abstract" in system.lower():
            return ("We verify an exact polynomial identity by symbolic reduction and record "
                    "the machine certificate. The result is elementary but fully reproducible.", 5)
        return paper_body, 5
    monkeypatch.setattr(loop, "_think", fake_writer)
    assert Path(loop.write()["tex"]).is_file()
    assert (d / "research-map.md").is_file() and (d / "research-map.json").is_file()


def test_loop_verifies_workbench_claim(monkeypatch):
    from spiral import research_loop
    loop = research_loop.ResearchLoop("certificate topic", workdir=Path(tempfile.mkdtemp()))
    loop.cfg.research_blind_replication = False
    claims = [{
        "kind": "workbench",
        "note": "case split certificate",
        "files": {"check.py": "assert 2 + 2 == 4\nprint('CERTIFICATE_OK')\n"},
        "cmd": "python check.py",
    }]

    findings = loop.verify_claims(claims)

    assert findings[0].ok and findings[0].backend == "workbench"
    assert "manifest" in findings[0].claim and Path(findings[0].claim["manifest"]).is_file()


def test_workbench_macos_execution_is_offline_read_and_write_confined():
    import shutil
    import sys

    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        return
    from spiral.research_workbench import run_workbench_claim

    root = Path(tempfile.mkdtemp())
    outside = Path("/tmp") / f"spiral-outside-{root.name}.txt"
    secret = Path.home() / f".spiral-secret-{root.name}.txt"
    outside.unlink(missing_ok=True)
    secret.write_text("must remain unreadable")
    code = f'''from pathlib import Path
write_denied = False
read_denied = False
network_denied = False
try:
    Path({str(outside)!r}).write_text("escape")
except PermissionError:
    write_denied = True
try:
    private_path = Path("/") / ("Us" + "ers") / ("e" + "dt") / {secret.name!r}
    private_path.read_text()
except PermissionError:
    read_denied = True
socket_module = __import__("socket")
try:
    connection = getattr(socket_module, "socket")()
    getattr(connection, "connect")(("127.0.0.1", 9))
except PermissionError:
    network_denied = True
assert write_denied and read_denied and network_denied
Path("inside.txt").write_text("ok")
print("CERTIFICATE_OK")
'''
    result = run_workbench_claim({
        "kind": "workbench",
        "note": "sandbox isolation",
        "files": {"check.py": code},
        "cmd": "python check.py",
    }, root)
    secret.unlink(missing_ok=True)

    manifest = json.loads(Path(result.manifest).read_text())
    assert result.ok
    assert not outside.exists()
    assert manifest["execution_isolation"]["mode"] == "macos-sandbox-exec"
    assert manifest["execution_isolation"]["network"] == "denied"
    assert manifest["execution_isolation"]["denied_read_roots"] == [
        "/Users", "/Volumes", "/Network"]


def test_workbench_failure_output_is_repaired_only_by_a_local_model(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("private certificate", workdir=tmp_path)
    loop.ol.providers = {"remote-model": {}}
    for role in ("worker", "planner", "critic", "escalation"):
        getattr(loop.cfg, role).name = "remote-model"
    loop.cfg.research_auditor.name = "local-auditor"
    seen = []

    def fake_json(system, user, **kwargs):
        seen.append((user, kwargs))
        return {
            "files": {"check.py": "assert 2 + 2 == 4\nprint('CERTIFICATE_OK')\n"},
            "cmd": "python check.py",
        }

    monkeypatch.setattr(loop, "_think_json", fake_json)
    fixed = loop._repair_workbench_claim(
        {"kind": "workbench", "note": "failed"},
        SimpleNamespace(detail="failed", stdout="PRIVATE OUTPUT", stderr="", manifest="m"),
        1,
    )

    assert fixed and seen[0][1]["role"] == "research_auditor"
    assert "PRIVATE OUTPUT" in seen[0][0]


def test_loop_repairs_failed_workbench_claim(monkeypatch):
    from spiral import research_loop
    loop = research_loop.ResearchLoop("repair certificate", workdir=Path(tempfile.mkdtemp()))
    loop.cfg.research_blind_replication = False
    repairs = []

    def fake_repair(system, user, think=None, **kwargs):
        repairs.append((system, user, kwargs))
        return (json.dumps({
            "kind": "workbench",
            "note": "fixed certificate",
            "files": {"check.py": "assert 2 + 2 == 4\nprint('CERTIFICATE_OK')\n"},
            "cmd": "python check.py",
            "expect": "CERTIFICATE_OK",
        }), 5)

    monkeypatch.setattr(loop, "_think", fake_repair)
    findings = loop.verify_claims([{
        "kind": "workbench",
        "note": "broken generated certificate",
        "files": {"check.py": "raise AssertionError('not yet')\n"},
        "cmd": "python check.py",
    }])

    assert findings[0].ok and findings[0].claim["_repair_attempt"] == 1
    assert repairs and repairs[0][2]["role"] == "research_auditor"


def test_loop_normalises_nested_workbench_metadata():
    from spiral import research_loop
    loop = research_loop.ResearchLoop("nested workbench fields", workdir=Path(tempfile.mkdtemp()))
    loop.cfg.research_blind_replication = False
    findings = loop.verify_claims([{
        "kind": "workbench",
        "note": "nested fields from local model",
        "files": {
            "check.py": "value = sum(i * i for i in range(4))\nassert value == 14\nprint('CERTIFICATE_OK')\n",
            "cmd": "python check.py",
            "expect": "CERTIFICATE_OK",
        },
    }])
    assert findings[0].ok
    assert "cmd" not in findings[0].claim["files"] and findings[0].claim["cmd"] == "python check.py"


def test_loop_does_not_give_up_without_a_verified_result(monkeypatch):
    """The regression that started this: an empty question / referee 'new_question' with
    zero verified findings must NOT terminate the loop — it keeps working to max_rounds."""
    from spiral import research_loop
    loop = research_loop.ResearchLoop("some hard topic", workdir=Path(tempfile.mkdtemp()))
    monkeypatch.setattr(loop, "search_plan", lambda n=3: ([], ["q"]))
    monkeypatch.setattr(loop, "assess_corpus", lambda: {"sufficient": True})
    monkeypatch.setattr(loop.corpus, "build", lambda q, k=8, categories=None, on=None: [])
    monkeypatch.setattr(loop.corpus, "graph_deepen", lambda **k: {"added": 0, "saturated": True, "holes": []})
    monkeypatch.setattr(loop, "novelty", lambda q: [])
    # model never produces a question and the referee tries to bail with 'new_question'
    monkeypatch.setattr(loop, "propose", lambda refine_rounds=2: {"question": "", "claims": []})
    monkeypatch.setattr(loop, "reflect", lambda findings, priors:
                        {"action": "new_question", "reason": "please supply a question"})
    state = loop.run(max_rounds=3)
    assert state.round == 3                       # ran all rounds — did NOT give up on round 1
    assert state.status == "exhausted"            # honest, not a false 'new_question'
    assert state.question == ""                     # no fake fallback question was forced
    assert all(h["action"] == "continue" for h in state.history)


def test_loop_exhausts_after_repeated_blank_rounds(monkeypatch):
    """When the provider returns parseable nothing forever, the loop should stop with
    an honest exhausted status after the configured observable-plateau fallback."""
    from spiral import research_loop
    loop = research_loop.ResearchLoop("some hard topic", workdir=Path(tempfile.mkdtemp()))
    loop.cfg.research_plateau_patience = 4
    monkeypatch.setattr(loop, "search_plan", lambda n=3: ([], ["q"]))
    monkeypatch.setattr(loop, "assess_corpus", lambda: {"sufficient": True})
    monkeypatch.setattr(loop.corpus, "build", lambda q, k=8, categories=None, on=None: [])
    monkeypatch.setattr(loop.corpus, "graph_deepen", lambda **k: {"added": 0, "saturated": True, "holes": [], "round_reports": []})
    monkeypatch.setattr(loop, "novelty", lambda q: [])
    monkeypatch.setattr(loop, "propose", lambda refine_rounds=2: {"question": "", "claims": []})
    monkeypatch.setattr(loop, "reflect", lambda findings, priors:
                        {"action": "continue", "reason": "no parseable decision"})
    state = loop.run()
    assert state.status == "exhausted" and state.round == 8
    assert "no observable progress" in state.history[-1]["reason"]


def test_expository_identity_prompt_stays_literal(monkeypatch):
    """A verify/write note prompt is not an invitation to invent a new research problem.
    The literal identity should become the first machine-checkable claim."""
    from spiral import research_loop
    messages = []
    topic = ("Verify and write a short mathematical note about the identity "
             "(x+1)^2 = x^2 + 2x + 1, using corpus citations only as background.")
    loop = research_loop.ResearchLoop(
        topic, workdir=Path(tempfile.mkdtemp()), ui=messages.append,
        mode="expository")
    monkeypatch.setattr(loop.corpus, "build", lambda q, k=8, categories=None, on=None: [])
    monkeypatch.setattr(loop.corpus, "graph_deepen", lambda **k: {"added": 0, "saturated": True, "holes": [], "round_reports": []})

    state = loop.run(max_rounds=1)
    assert state.status == "solved"
    assert state.question == "Verify the identity (x+1)^2 = x^2 + 2x + 1."
    assert state.findings[0]["ok"] and state.findings[0]["backend"] == "sympy"
    assert state.findings[0]["claim"]["lhs"] == "(x+1)^2"
    assert "nilpotent" not in (loop.dir / "journal.md").read_text().lower()
    assert any("novelty · skipped for expository note" in m for m in messages)


def test_task_mode_defaults_to_research_but_can_force_verification():
    from spiral import research_loop
    topic = ("Verify and write a short mathematical note about the identity "
             "(x+1)^2 = x^2 + 2x + 1, using corpus citations only as background.")
    assert research_loop.ResearchLoop("Verify possible binomial generalizations",
                                      workdir=Path(tempfile.mkdtemp()))._task_mode() == "research"
    assert research_loop.ResearchLoop(topic, workdir=Path(tempfile.mkdtemp()),
                                      mode="expository")._task_mode() == "expository"


def test_search_plan_picks_category_and_keywords(monkeypatch):
    """The loop must find the RIGHT arXiv category and short keyword queries — not dump
    the whole prompt (noise) nor search all of arXiv (a math identity drowns in hep-th)."""
    from spiral import research_loop
    topic = ("Discover a previously-unknown exact Ramanujan-type series identity for a "
             "mathematical constant and formally prove it in Lean")
    loop = research_loop.ResearchLoop(topic, workdir=Path(tempfile.mkdtemp()))
    roles = []

    def fake_plan(s, u, think=None, **k):
        roles.append(k.get("role", "planner"))
        return (json.dumps({"categories": ["math.NT", "math.CO"],
                            "queries": ["ramanujan series identity", "hypergeometric summation"]}), 3)

    monkeypatch.setattr(loop, "_think", fake_plan)
    cats, qs = loop.search_plan()
    assert cats == ["math.NT", "math.CO"] and qs == ["ramanujan series identity", "hypergeometric summation"]
    assert all(len(q) < 60 for q in qs)
    assert roles == ["planner"]
    # fallback: junk LLM → salient keywords, never the raw prompt
    monkeypatch.setattr(loop, "_think", lambda s, u, think=None, **k: ("not json", 1))
    fcats, fq = loop.search_plan()
    assert fq and fq[0] != topic and "previously" not in fq[0].lower()


def test_arxiv_query_restricts_to_categories():
    """The category filter must land in the arXiv search_query (cat: AND all:), not be
    stuffed into the all: field where it matches nothing."""
    import urllib.parse

    from spiral import research
    captured = {}
    orig = research._get
    research._get = lambda url, timeout=25.0: captured.setdefault("url", url) or ""
    try:
        research.arxiv("ramanujan series", categories=["math.NT", "math.CO"])
    finally:
        research._get = orig
    u = urllib.parse.parse_qs(urllib.parse.urlsplit(captured["url"]).query)["search_query"][0]
    assert ("cat:math.NT" in u and "cat:math.CO" in u and "AND" in u
            and 'all:"ramanujan series"' in u)


def test_proposal_iterates_against_prior_art(monkeypatch):
    """A first-guess proposal that duplicates prior art is revised until the referee
    accepts it and the basis audit finds named evidence — so what reaches verification is
    vetted for novelty and corpus grounding, not the first draft."""
    from spiral import citations, research_loop
    loop = research_loop.ResearchLoop("test topic", workdir=Path(tempfile.mkdtemp()))
    from spiral.research_corpus import Paper
    held = loop.corpus.add(Paper(
        arxiv_id="paper-1", title="Supported construction",
        text="The exact held source phrase establishes the same construction."), fetch=False)
    note_root = loop.dir / "notes" / "papers"
    note_root.mkdir(parents=True)
    (note_root / "paper-1.json").write_text(json.dumps({
        "arxiv_id": "paper-1", "grounded": True, "schema_version": 2,
        "source_hash": loop._paper_source_hash(held),
        "evidence": [{"supports": "same construction", "anchor": "exact held source phrase"}],
    }))
    def healthy_prior(*a, **k):
        if k.get("report") is not None:
            k["report"].update({"ready": True, "sources_ok": ["arxiv", "semantic_scholar"]})
        return []
    monkeypatch.setattr(citations, "prior_art", healthy_prior)
    monkeypatch.setattr(loop, "_discover_angles", lambda n=5: [{
        "question": "candidate gap Q",
        "search_queries": ["spectral obstruction", "algebraic classification"],
    }])
    monkeypatch.setattr(loop, "_audit_angle", lambda angle, priors, report, grounded_priors=None:
                        {"verdict": "pursue", "novelty": "not obviously classified"})
    monkeypatch.setattr(loop, "_proposal_from_angle", lambda angle, priors:
                        {"question": "old already-solved Q",
                         "claims": [{"kind": "zero", "expr": "x-x"}]})
    roles = []
    replies = iter([
        json.dumps({"verdict": "revise", "novelty": "duplicates prior art"}),     # referee: revise
        json.dumps({"question": "sharper novel Q", "claims": [{"kind": "zero", "expr": "x-x"}]}),  # refined
        json.dumps({"verdict": "accept", "novelty": "distinct"}),                 # referee: accept
        json.dumps({"verdict": "grounded", "basis": "supported by corpus method",
                    "evidence": [{"source": "corpus", "id": "paper-1",
                                  "supports": "same construction",
                                  "anchor": "exact held source phrase"}],
                    "missing": [], "searches": []}),                              # basis: accept
    ])

    def fake_think(system, user, think=None, **k):
        roles.append(k.get("role", "planner"))
        return next(replies), 5

    monkeypatch.setattr(loop, "_think", fake_think)
    prop = loop.propose(refine_rounds=2)
    assert prop["question"] == "sharper novel Q" and prop.get("_vetted") is True
    assert prop["_basis_audit"]["grounded"] is True
    assert roles == ["critic", "planner", "critic", "research_auditor"]


def test_prior_art_protocol_rejects_paraphrased_query_routes(monkeypatch, tmp_path):
    from spiral import citations
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("integrable sigma model classification", workdir=tmp_path)

    def healthy_prior(query, *args, **kwargs):
        report = kwargs.get("report")
        if report is not None:
            report.update({
                "query": query,
                "ready": True,
                "sources_ok": ["arxiv", "semantic_scholar"],
                "result_count": 0,
            })
        return []

    monkeypatch.setattr(citations, "prior_art", healthy_prior)
    _, repeated = loop._prior_art_bundle([
        "integrable sigma model Lax",
        "Lax integrable sigma model",
        "integrable sigma model Lax paper",
    ])
    assert repeated["healthy_queries"] == 3
    assert repeated["healthy_query_families"] == 1
    assert repeated["checks"]["independent_query_families"] is False
    assert repeated["ready"] is False

    _, independent = loop._prior_art_bundle([
        "integrable sigma model Lax",
        "Lie algebra automorphism coupling quotient",
        "renormalization group invariant metric flow",
    ])
    assert independent["healthy_query_families"] == 3
    assert independent["ready"] is True


def test_basis_audit_rejects_invented_or_unread_corpus_sources(monkeypatch, tmp_path):
    from spiral.research_corpus import Paper
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("grounded proposal", workdir=tmp_path)
    held = loop.corpus.add(Paper(
        arxiv_id="2401.00001", title="Held construction",
        text="An exact held source phrase gives the bridge used here."), fetch=False)
    proposal = {"question": "Does the held construction extend?", "claims": []}
    replies = iter([
        json.dumps({
            "verdict": "grounded", "basis": "named source", "missing": [], "searches": [],
            "evidence": [{"source": "corpus", "id": "9999.99999", "supports": "bridge",
                          "anchor": "exact held source phrase"}],
        }),
        json.dumps({
            "verdict": "grounded", "basis": "named source", "missing": [], "searches": [],
            "evidence": [{"source": "corpus", "id": "2401.00001", "supports": "bridge",
                          "anchor": "exact held source phrase"}],
        }),
        json.dumps({
            "verdict": "grounded", "basis": "named source", "missing": [], "searches": [],
            "evidence": [{"source": "corpus", "id": "2401.00001", "supports": "bridge",
                          "anchor": "exact held source phrase"}],
        }),
    ])
    monkeypatch.setattr(loop, "_think", lambda *a, **k: (next(replies), 1))

    invented = loop._basis_audit(proposal, [])
    assert invented["grounded"] is False
    assert invented["rejected_evidence"][0]["reason"] == "corpus reference is not held"

    unread = loop._basis_audit(proposal, [])
    assert unread["grounded"] is False
    assert "no source-grounded reading note" in unread["rejected_evidence"][0]["reason"]

    notes = tmp_path / "notes" / "papers"
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "2401.00001.json").write_text(json.dumps({
        "arxiv_id": "2401.00001", "grounded": True, "schema_version": 2,
        "source_hash": loop._paper_source_hash(held),
        "evidence": [{"supports": "bridge", "anchor": "exact held source phrase"}],
    }))
    grounded = loop._basis_audit(proposal, [])
    assert grounded["grounded"] is True
    assert grounded["evidence"][0]["resolved_id"] == "2401.00001"


def test_thin_basis_proposal_does_not_reach_verification(monkeypatch):
    """A fluent novelty proposal with no named corpus/prior basis must be treated as a
    search direction, not as a claim ready for Lean/SymPy."""
    from spiral import citations, research_loop
    loop = research_loop.ResearchLoop("Discover a novel binomial generalization",
                                      workdir=Path(tempfile.mkdtemp()))
    seen = []
    def healthy_prior(*a, **k):
        if k.get("report") is not None:
            k["report"].update({"ready": True, "sources_ok": ["arxiv", "semantic_scholar"]})
        return []
    monkeypatch.setattr(citations, "prior_art", healthy_prior)
    monkeypatch.setattr(loop, "_discover_angles", lambda n=5: [{
        "question": "nilpotent perturbation",
        "search_queries": ["dual numbers deformation", "square zero extension"],
    }])
    monkeypatch.setattr(loop, "_audit_angle", lambda angle, priors, report, grounded_priors=None:
                        {"verdict": "pursue", "novelty": "not found in prior art"})
    monkeypatch.setattr(loop, "_proposal_from_angle", lambda angle, priors:
                        {"question": "nilpotent perturbation",
                         "claims": [{"kind": "zero", "expr": "x-x"}]})
    replies = iter([
        json.dumps({"verdict": "accept", "novelty": "not found in prior art"}),
        json.dumps({"verdict": "thin", "basis": "only keyword overlap",
                    "evidence": [], "missing": ["no corpus bridge"],
                    "searches": ["dual numbers binomial theorem"]}),
        json.dumps({"question": "ground the algebra first", "claims": [{"kind": "zero", "expr": "x-x"}]}),
    ])
    monkeypatch.setattr(loop, "_think", lambda system, user, think=True, **k: (next(replies), 5))
    loop.ui = seen.append

    prop = loop.propose(refine_rounds=1)
    assert prop.get("_vetted") is not True
    assert prop["claims"] == []
    assert any("basis · thin" in m for m in seen)


def test_proposal_rejects_known_angles_and_deepens_search(monkeypatch):
    """If every corpus-mined angle is already known or too thin, the loop should
    ask for a better search, not force a theorem into the verifier."""
    from spiral import citations, research_loop

    loop = research_loop.ResearchLoop("classify a hard physics system",
                                      workdir=Path(tempfile.mkdtemp()))
    def healthy_prior(*a, **k):
        if k.get("report") is not None:
            k["report"].update({"ready": True, "sources_ok": ["arxiv", "semantic_scholar"]})
        return []
    monkeypatch.setattr(citations, "prior_art", healthy_prior)
    monkeypatch.setattr(loop, "_discover_angles", lambda n=5: [
        {"question": "known classification", "search_queries": ["known classification"]},
        {"question": "thin speculative bridge", "search_queries": ["thin bridge"]},
    ])
    audits = iter([
        {"verdict": "known", "novelty": "prior art already answers it",
         "next_query": "less classified target"},
        {"verdict": "thin", "basis": "no corpus bridge", "next_query": "master equations love numbers"},
    ])
    monkeypatch.setattr(loop, "_audit_angle", lambda angle, priors, report: next(audits))
    monkeypatch.setattr(loop, "_compact_proposal_retry", lambda hint:
                        {"no_proposal": True, "reasoning": "need deeper corpus",
                         "missing": ["fixed target with master equations"],
                         "next_query": "black brane static Love numbers"})

    prop = loop.propose(refine_rounds=1)

    assert prop["_no_proposal"] is True
    assert prop["claims"] == []
    assert prop["next_query"] == "black brane static Love numbers"
    assert len(prop["rejected_angles"]) == 2


# ── deterministic readiness gates ──────────────────────────────────────────
def test_corpus_quality_requires_healthy_saturated_retrieval():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "Lovelock black hole tidal perturbation master equations Love response"
    papers = [
        Paper(
            arxiv_id=f"2401.{i:05d}",
            title=f"Lovelock black hole tidal perturbation master equations {i}",
            abstract="Static Love response in higher dimensional black holes.",
            text=("Lovelock black hole tidal perturbation master equations Love response. " * 40),
            tex_path=f"/tmp/{i}.tex", body_source="tex",
        )
        for i in range(12)
    ]
    paper_ids = [paper.bare_id for paper in papers]
    searches = [
        {"query": q, "retrieval": {
            "source_ok": True, "result_count": 6,
            "result_ids": paper_ids[offset:offset + 6],
        }}
        for offset, q in enumerate(
            ("lovelock love", "tidal perturbations", "master equations"))
    ]
    healthy_graph = {
        "saturated": True,
        "holes": [],
        "health": {"requests": 20, "successful_requests": 20,
                   "failed_requests": 0, "coverage_valid": True,
                   "successful_seeds": paper_ids},
    }
    report = corpus_quality_report(topic, papers, {
        "searches": searches, "graph_rounds": [healthy_graph],
    })
    assert report["discovery_ready"] is True and report["novelty_ready"] is True

    dead_graph = {
        **healthy_graph,
        "health": {"requests": 20, "successful_requests": 0,
                   "failed_requests": 20, "coverage_valid": False},
    }
    failed = corpus_quality_report(topic, papers, {
        "searches": searches, "graph_rounds": [dead_graph],
    })
    assert failed["discovery_ready"] is True
    assert failed["novelty_ready"] is False and failed["graph"]["saturated"] is False

    small = corpus_quality_report(topic, papers[:6], {
        "searches": searches, "graph_rounds": [healthy_graph],
    })
    assert small["small_field_exception"] is True and small["discovery_ready"] is True

    failed_diversity = corpus_quality_report(topic, papers, {
        "searches": [
            {"query": "lovelock love", "retrieval": {"source_ok": True, "result_count": 6}},
            {"query": "tidal perturbations", "retrieval": {"source_ok": True, "result_count": 6}},
            {"query": "master equations", "retrieval": {"source_ok": False, "result_count": 0}},
        ],
        "graph_rounds": [healthy_graph],
    })
    assert failed_diversity["search"]["unique_queries"] == 3
    assert failed_diversity["search"]["healthy_unique_queries"] == 2
    assert failed_diversity["discovery_checks"]["query_diversity"] is False


def test_corpus_quality_rejects_duplicate_queries_and_stale_graph_closure():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "integrable sigma model Lax classification spectral parameter"
    papers = [
        Paper(arxiv_id=f"2402.{i:05d}", title=topic, abstract=topic,
              text=(topic + " ") * 200, tex_path=f"/tmp/{i}.tex", body_source="tex")
        for i in range(10)
    ]
    ids = [paper.bare_id for paper in papers]
    searches = [
        {"query": query, "retrieval": {
            "source_ok": True, "result_count": 5, "result_ids": ids[:5],
        }}
        for query in (
            "integrable sigma model Lax",
            "Lax integrable sigma model",
            "integrable sigma model Lax paper",
        )
    ]
    old_closed = {"saturated": True, "corpus_size": 9, "holes": [], "health": {
        "requests": 10, "successful_requests": 10,
        "failed_requests": 0, "coverage_valid": True}}
    report = corpus_quality_report(topic, papers, {
        "searches": searches, "graph_rounds": [old_closed]})

    assert report["search"]["healthy_unique_queries"] == 3
    assert report["search"]["healthy_query_families"] == 1
    assert report["discovery_checks"]["query_diversity"] is False
    assert report["graph"]["latest_matches_corpus"] is False
    assert report["graph"]["saturated"] is False


def test_corpus_quality_requires_relevance_and_primary_text_to_overlap():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "integrable sigma model Lax spectral classification"
    relevant = [
        Paper(arxiv_id=f"2403.{i:05d}", title=topic, abstract=topic,
              text=(topic + " ") * 200, body_source="abstract")
        for i in range(5)
    ]
    unrelated_primary = [
        Paper(arxiv_id=f"2404.{i:05d}", title="combinatorial number theory",
              abstract="partitions and primes", text=("partitions and primes " * 200),
              tex_path=f"/tmp/unrelated-{i}.tex", body_source="tex")
        for i in range(6)
    ]
    papers = relevant + unrelated_primary
    relevant_ids = [paper.bare_id for paper in relevant]
    searches = [
        {"query": query, "retrieval": {
            "source_ok": True, "result_count": len(relevant_ids),
            "result_ids": relevant_ids,
        }}
        for query in ("integrable sigma", "Lax spectral", "sigma classification")
    ]

    report = corpus_quality_report(topic, papers, {"searches": searches})

    assert report["usable_primary_text_count"] == 6
    assert report["relevant_paper_count"] == 5
    assert report["relevant_usable_primary_text_count"] == 0
    assert report["discovery_checks"]["relevant_usable_primary_texts"] is False
    assert report["discovery_ready"] is False


def test_citation_saturation_requires_every_current_seed_batch():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "integrable sigma model Lax spectral classification"
    papers = [
        Paper(arxiv_id=f"2405.{i:05d}", title=topic, abstract=topic,
              text=(topic + " ") * 200, tex_path=f"/tmp/{i}.tex", body_source="tex")
        for i in range(10)
    ]
    ids = [paper.bare_id for paper in papers]
    searches = [
        {"query": query, "retrieval": {
            "source_ok": True, "result_count": 6, "result_ids": ids[:6],
        }}
        for query in ("integrable sigma", "Lax spectral", "sigma classification")
    ]
    partial_graph = {"saturated": True, "batch_frontier_closed": True,
                     "holes": [], "health": {
                         "requests": 6, "successful_requests": 6,
                         "failed_requests": 0, "coverage_valid": True,
                         "successful_seeds": ids[:3],
                     }}

    report = corpus_quality_report(topic, papers, {
        "searches": searches, "graph_rounds": [partial_graph]})

    assert report["graph"]["healthy"] is True
    assert report["graph"]["successful_current_seed_count"] == 3
    assert report["graph"]["closed_current_seed_count"] == 3
    assert report["graph"]["saturated"] is False
    assert report["novelty_ready"] is False


def test_search_incidence_estimate_uses_overlap_but_never_gates():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "spectral deformation integrable sigma model classification"
    papers = [
        Paper(arxiv_id=f"2401.{i:05d}", title=topic, abstract=topic,
              text=(topic + " ") * 200, tex_path=f"/tmp/{i}.tex", body_source="tex")
        for i in range(10)
    ]
    searches = [
        {"query": "spectral deformation", "retrieval": {
            "source_ok": True, "result_count": 4,
            "result_ids": ["a", "b", "c", "d"]}},
        {"query": "integrable sigma", "retrieval": {
            "source_ok": True, "result_count": 4,
            "result_ids": ["b", "c", "e", "f"]}},
        {"query": "lax classification", "retrieval": {
            "source_ok": True, "result_count": 4,
            "restricted": {"result_ids": ["c", "e"]},
            "fallback": {"result_ids": ["g", "h"]}}},
    ]
    graph = {"saturated": True, "holes": [], "health": {
        "requests": 10, "successful_requests": 10,
        "failed_requests": 0, "coverage_valid": True,
        "successful_seeds": [paper.bare_id for paper in papers]}}
    report = corpus_quality_report(topic, papers, {
        "searches": searches, "graph_rounds": [graph]})
    estimate = report["search"]["incidence_coverage"]

    assert estimate["diagnostic_valid"] is True
    assert estimate["observed_unique_records"] == 8
    assert estimate["singletons"] == 5
    assert estimate["doubletons"] == 2
    assert estimate["estimated_observed_fraction"] < 1
    assert estimate["used_as_gate"] is False


def test_long_abstracts_do_not_satisfy_primary_text_gate():
    from spiral.research_corpus import Paper
    from spiral.research_quality import corpus_quality_report

    topic = "Lovelock tidal response master equation"
    papers = [
        Paper(arxiv_id=f"2402.{i:05d}", title=topic,
              abstract=(topic + " ") * 100, text=(topic + " ") * 100,
              body_source="abstract")
        for i in range(12)
    ]
    searches = [{"query": q, "retrieval": {"source_ok": True, "result_count": 5}}
                for q in ("lovelock", "tidal response", "master equation")]
    report = corpus_quality_report(topic, papers, {"searches": searches})

    assert report["usable_text_count"] == 12
    assert report["usable_primary_text_count"] == 0
    assert report["discovery_checks"]["usable_primary_texts"] is False
    assert report["discovery_ready"] is False


def test_completion_gate_rejects_missing_and_self_checked_claims(tmp_path, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    from spiral.research_loop import Finding, ResearchLoop
    from spiral.research_provenance import NoveltyBoundaryCertificate

    loop = ResearchLoop("novel exact identity", workdir=tmp_path)
    loop.cfg.research_blind_replication = False
    loop.cfg.research_obligation_graph = False
    loop.cfg.research_min_grounded_notes = 0
    loop.cfg.research_min_grounded_deep_reads = 0
    loop.state.question = "Does the exact identity hold?"
    loop.state.coverage = {"discovery_ready": True, "novelty_ready": True}
    loop._last_novelty_report = {"ready": True}
    monkeypatch.setattr(
        NoveltyBoundaryCertificate, "validate",
        staticmethod(lambda value: {"valid": True, "issues": []}))
    loop._log_thought("fixture", "completion evidence fixture")
    loop._audit_model_call(
        model="fixture", role="critic", variant="unit-test", system="fixture", user="fixture",
        result=SimpleNamespace(text="fixture", thinking="", prompt_tokens=1,
                               completion_tokens=1, raw={}),
    )
    contract = {
        "assumptions": ["x is a commutative symbol"],
        "falsifier": "a nonzero exact residual",
        "method_family": "exact symbolic algebra",
    }
    exact = {
        "kind": "zero", "expr": "x-x", "note": "exact core",
        "statement": "The exact residual vanishes", "required": True, **contract,
    }
    self_check = {
        "kind": "workbench", "note": "classification script",
        "statement": "The classification cases are exhausted", "required": True,
        **contract,
    }
    proposal = {"question": loop.state.question, "claims": [exact, self_check], "_vetted": True}
    loop.state.findings = [
        asdict(Finding(exact, True, "sympy", "zero", 1, loop.state.question,
                       loop._claim_id(exact), "exact", True)),
        asdict(Finding(self_check, True, "workbench", "ran", 1, loop.state.question,
                       loop._claim_id(self_check), "executable", True)),
    ]
    report = loop.completion_gate(proposal, {"action": "solved"})
    assert report["ready"] is False
    assert report["checks"]["required_evidence_is_independent"] is False

    loop.state.findings[1]["strength"] = "computational"
    report = loop.completion_gate(proposal, {"action": "solved"})
    assert report["ready"] is True


def test_completion_gate_requires_grounded_broad_and_deep_reading(tmp_path, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    from spiral.research_loop import Finding, ResearchLoop
    from spiral.research_provenance import NoveltyBoundaryCertificate

    loop = ResearchLoop("novel spectral classification", workdir=tmp_path)
    loop.cfg.research_blind_replication = False
    loop.cfg.research_obligation_graph = False
    loop.cfg.research_min_grounded_notes = 2
    loop.cfg.research_min_grounded_deep_reads = 1
    loop.state.question = "Which spectral locus is integrable?"
    loop.state.coverage = {
        "discovery_ready": True, "novelty_ready": True, "relevant_paper_count": 4,
    }
    loop._last_novelty_report = {"ready": True}
    monkeypatch.setattr(
        NoveltyBoundaryCertificate, "validate",
        staticmethod(lambda value: {"valid": True, "issues": []}))
    loop._log_thought("fixture", "grounded reading fixture")
    loop._audit_model_call(
        model="fixture", role="critic", variant="unit-test", system="fixture", user="fixture",
        result=SimpleNamespace(text="fixture", thinking="", prompt_tokens=1,
                               completion_tokens=1, raw={}),
    )
    claim = {
        "kind": "zero", "expr": "x-x", "note": "exact locus",
        "statement": "The exact spectral locus has zero residual",
        "assumptions": ["x is a commutative symbol"],
        "falsifier": "a nonzero exact residual",
        "method_family": "exact symbolic algebra",
        "required": True,
    }
    proposal = {"question": loop.state.question, "claims": [claim], "_vetted": True}
    loop.state.findings = [asdict(Finding(
        claim, True, "sympy", "zero", 1, loop.state.question,
        loop._claim_id(claim), "exact", True))]

    missing = loop.completion_gate(proposal, {"action": "solved"})
    assert missing["checks"]["source_grounded_reading_ready"] is False
    assert missing["checks"]["grounded_deep_reading_ready"] is False

    paper_root = tmp_path / "notes" / "papers"
    deep_root = tmp_path / "notes" / "deep"
    paper_root.mkdir(parents=True)
    deep_root.mkdir(parents=True)
    for index in range(2):
        (paper_root / f"p{index}.json").write_text(json.dumps({
            "grounded": True,
            "evidence": [{"supports": "claim", "anchor": "exact source anchor"}],
        }))
    (deep_root / "p0.json").write_text(json.dumps({
        "grounded": True,
        "grounded_evidence": [{"supports": "derivation", "anchor": "exact equation"}],
    }))

    ready = loop.completion_gate(proposal, {"action": "solved"})
    assert ready["ready"] is True

    # Re-reading the same paper under another round/family must not fake breadth.
    (deep_root / "duplicate.json").write_text(json.dumps({
        "arxiv_id": "p0", "grounded": True,
        "grounded_evidence": [{"supports": "same", "anchor": "same exact equation"}],
    }))
    from spiral.research_quality import reading_metrics
    assert reading_metrics(tmp_path / "notes")["grounded_deep_notes"] == 1


def test_thought_log_hash_chain_detects_tampering(tmp_path):
    from spiral.research_loop import ResearchLoop
    from spiral.research_quality import verify_jsonl_hash_chain

    loop = ResearchLoop("trace integrity", workdir=tmp_path)
    loop._log_thought("one", "first decision")
    loop._log_thought("two", "second decision")
    path = tmp_path / "thoughts.jsonl"
    assert verify_jsonl_hash_chain(path)["ok"] is True
    path.write_text(path.read_text().replace("first decision", "altered decision"), encoding="utf-8")
    assert verify_jsonl_hash_chain(path)["ok"] is False


def test_research_model_call_log_is_replayable_but_excludes_private_reasoning(tmp_path):
    from types import SimpleNamespace

    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("audit calls", workdir=tmp_path)
    result = SimpleNamespace(
        text='{"decision":"continue"}',
        thinking="private backend reasoning",
        prompt_tokens=12,
        completion_tokens=7,
        raw={"finish_reason": "stop"},
    )
    loop._audit_model_call(
        model="local-test", role="critic", variant="full",
        system="SYSTEM RULES", user="VISIBLE EVIDENCE", result=result,
        reasoning_requested=True)
    row = json.loads((tmp_path / "model-calls.jsonl").read_text())

    assert row["system_prompt"] == "SYSTEM RULES"
    assert row["user_prompt"] == "VISIBLE EVIDENCE"
    assert row["output"] == '{"decision":"continue"}'
    assert row["reasoning_requested"] is True
    assert row["private_reasoning_returned"] is True
    assert row["private_reasoning_chars"] == len("private backend reasoning")
    assert len(row["private_reasoning_sha256"]) == 64
    assert row["private_reasoning_recorded"] is False
    assert "private backend reasoning" not in json.dumps(row)
    from spiral.research_quality import verify_jsonl_hash_chain
    assert verify_jsonl_hash_chain(tmp_path / "model-calls.jsonl")["ok"] is True


# ── citation-graph corpus expansion ─────────────────────────────────────────
def test_parse_edges_keeps_arxiv_drops_others():
    from spiral.cite_graph import parse_edges
    payload = {"data": [
        {"citedPaper": {"externalIds": {"ArXiv": "hep-th/9711200"}, "title": "AdS/CFT",
                        "year": 1997, "citationCount": 20000, "authors": [{"name": "Maldacena"}]}},
        {"citedPaper": {"externalIds": {"DOI": "10.x"}, "title": "no arxiv id"}},   # dropped
    ]}
    edges = parse_edges(payload, "references")
    assert len(edges) == 1 and edges[0].arxiv_id == "hep-th/9711200" and edges[0].citations == 20000
    cit = parse_edges({"data": [{"citingPaper": {"externalIds": {"ArXiv": "2401.00001"},
                                                 "title": "builds on it"}}]}, "citations")
    assert cit[0].arxiv_id == "2401.00001"


def test_snowball_counts_cocitations_and_excludes_have(monkeypatch):
    from spiral import cite_graph
    from spiral.cite_graph import Edge
    # two seeds both reference the SAME foundational paper F (co-citation ×2) and one each
    graph = {
        "seedA": {"references": [Edge("found.0001", "Foundational", 1990, 500),
                                 Edge("only.A", "A-only")],
                  "citations": [Edge("recent.1", "cites A", 2024, 3)]},
        "seedB": {"references": [Edge("found.0001", "Foundational", 1990, 500),
                                 Edge("only.B", "B-only")],
                  "citations": []},
    }
    monkeypatch.setattr(cite_graph, "paper_edges",
                        lambda sid, direction, limit=100: graph.get(sid, {}).get(direction, []))
    back, fwd, meta = cite_graph.snowball(["seedA", "seedB"], have={"seedA", "seedB"}, pause=0)
    assert back["found.0001"] == 2 and back["only.A"] == 1     # co-citation strength
    assert "recent.1" in fwd and "found.0001" in meta
    holes = cite_graph.rank_holes(back, min_cocite=2)
    assert holes == ["found.0001"]                             # the load-bearing missing paper


def test_graph_deepen_fills_holes_and_saturates(monkeypatch):
    from collections import Counter

    from spiral import cite_graph
    from spiral.cite_graph import Edge
    from spiral.research_corpus import Corpus, Paper
    c = Corpus(Path(tempfile.mkdtemp()))
    c.add(Paper(arxiv_id="2401.00001", title="seed"), fetch=False)
    monkeypatch.setattr(Corpus, "_fetch_bodies", lambda self, p: None)   # no network
    calls = {"n": 0}

    def fake_snowball(seeds, have, **k):
        calls["n"] += 1
        health = k.get("health")
        if health is not None:
            health.update({
                "requests": max(2, 2 * len(seeds)),
                "successful_requests": max(2, 2 * len(seeds)),
                "failed_requests": 0,
                "seeds_attempted": len(seeds),
                "successful_seeds": list(seeds),
            })
        if calls["n"] == 1:                     # first round surfaces a co-citation hole
            return Counter({"found.9999": 3}), set(), {"found.9999": Edge("found.9999", "Foundational")}
        return Counter(), set(), {}             # second round: nothing new → saturate
    monkeypatch.setattr(cite_graph, "snowball", fake_snowball)
    rep = c.graph_deepen(rounds=3, min_cocite=2)
    assert c.has("found.9999") and rep["added"] == 1 and rep["saturated"]


def test_graph_api_failure_is_not_reported_as_saturation(monkeypatch):
    from collections import Counter

    from spiral import cite_graph
    from spiral.research_corpus import Corpus, Paper

    c = Corpus(Path(tempfile.mkdtemp()))
    c.add(Paper(arxiv_id="2401.00001", title="seed"), fetch=False)

    def failed_snowball(seeds, have, **kwargs):
        kwargs["health"].update({
            "requests": 2, "successful_requests": 0, "failed_requests": 2,
            "seeds_attempted": 1, "successful_seeds": [],
        })
        return Counter(), set(), {}

    monkeypatch.setattr(cite_graph, "snowball", failed_snowball)
    report = c.graph_deepen(rounds=1)
    assert report["saturated"] is False
    assert report["round_reports"][0]["health"]["coverage_valid"] is False
    assert report["errors"]


def test_graph_candidate_cap_is_not_reported_as_saturation(monkeypatch):
    from collections import Counter

    from spiral import cite_graph
    from spiral.cite_graph import Edge
    from spiral.research_corpus import Corpus, Paper

    c = Corpus(Path(tempfile.mkdtemp()))
    for index in range(40):
        c.add(Paper(arxiv_id=f"2401.{index:05d}", title=f"seed {index}"), fetch=False)
    monkeypatch.setattr(Corpus, "_fetch_bodies", lambda self, p: None)

    def capped_frontier(seeds, have, **kwargs):
        kwargs["health"].update({
            "requests": 60, "successful_requests": 60, "failed_requests": 0,
            "seeds_attempted": 30, "successful_seeds": list(seeds)[:30],
        })
        holes = Counter({f"2501.{index:05d}": 3 for index in range(35)})
        meta = {identifier: Edge(identifier, f"candidate {identifier}") for identifier in holes}
        return holes, set(), meta

    monkeypatch.setattr(cite_graph, "snowball", capped_frontier)
    report = c.graph_deepen(rounds=1, cap=30)

    round_report = report["round_reports"][0]
    assert round_report["frontier_truncated"] is True
    assert round_report["unresolved_holes_after_round"]
    assert report["saturated"] is False


def test_research_loop_rotates_across_closed_graph_seed_batches(tmp_path):
    from spiral.research_corpus import Paper
    from spiral.research_loop import ResearchLoop

    loop = ResearchLoop("integrable sigma model", workdir=tmp_path)
    for index in range(35):
        loop.corpus.add(Paper(
            arxiv_id=f"2406.{index:05d}",
            title=f"integrable sigma model {index}",
            abstract="Lax spectral classification",
        ), fetch=False)
    ids = sorted(loop.corpus.papers)
    loop.map["graph_rounds"] = [{
        "saturated": True,
        "batch_frontier_closed": True,
        "frontier_truncated": False,
        "unresolved_holes_after_round": [],
        "health": {
            "coverage_valid": True,
            "successful_seeds": ids[:30],
        },
    }]

    assert set(loop._graph_seed_batch()) == set(ids[30:])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
