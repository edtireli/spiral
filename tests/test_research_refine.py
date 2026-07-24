"""research --refine: the gates decide, the original is never touched."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral.research_refine import (
    RefineError,
    RefineRun,
    anchored_in,
    display_math,
    find_main_tex,
    flatten_tex,
    numeric_tokens,
    rebuild_violations,
    resolve_figures,
    split_document,
    survey_project,
)


# ------------------------------------------------------------------ pure helpers

def test_numeric_tokens_ignores_reference_arguments():
    tex = r"We find $p=0.03$~\cite{smith2021} (Fig.~\ref{fig:12}, \label{sec:9})."
    assert numeric_tokens(tex) == {"0.03"}


def test_display_math_is_order_insensitive_but_content_strict():
    a = r"\begin{equation}E=mc^2\end{equation}\[ a+b \]"
    b = r"\[a + b\] text \begin{equation} E = m c^2 \end{equation}"
    assert display_math(a) == display_math(b)
    c = r"\begin{equation}E=mc^3\end{equation}\[ a+b \]"
    assert display_math(a) != display_math(c)


def test_rebuild_violations_catch_the_sins():
    orig = (r"We measured 42 samples. \begin{equation}x=1\end{equation} "
            r"\label{sec:m} \cite{known}")
    ok = (r"Across 42 samples we measured the effect. "
          r"\begin{equation}x=1\end{equation} \label{sec:m} \cite{known}")
    assert rebuild_violations(orig, ok, allowed_numbers={"42", "1"},
                              allowed_cites={"known"}) == []
    invented = ok.replace("42 samples", "42 samples (p=0.001)")
    v = rebuild_violations(orig, invented, allowed_numbers={"42", "1"},
                           allowed_cites={"known"})
    assert any("invented numbers" in p for p in v)
    dropped_math = ok.replace(r"\begin{equation}x=1\end{equation}", "")
    v = rebuild_violations(orig, dropped_math, allowed_numbers={"42", "1"},
                           allowed_cites={"known"})
    assert any("display math" in p for p in v)
    lost_label = ok.replace(r"\label{sec:m}", "")
    v = rebuild_violations(orig, lost_label, allowed_numbers={"42", "1"},
                           allowed_cites={"known"})
    assert any("label" in p for p in v)
    stray_cite = ok.replace(r"\cite{known}", r"\cite{fabricated2029}")
    v = rebuild_violations(orig, stray_cite, allowed_numbers={"42", "1"},
                           allowed_cites={"known"})
    assert any("unknown keys" in p for p in v)


def test_anchor_requires_a_real_quote():
    paper = SimpleNamespace(title="T", abstract="The master equation reduces to "
                            "hypergeometric form in four dimensions.", text="")
    assert anchored_in(paper, "master equation reduces to hypergeometric form")
    assert not anchored_in(paper, "the Love numbers vanish identically")
    assert not anchored_in(paper, "short")   # length floor


def test_flatten_and_survey(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "intro.tex").write_text("Hello intro. % \\input{ghost}\n")
    (tmp_path / "fig1.png").write_bytes(b"\x89PNG fake")
    main = tmp_path / "main.tex"
    main.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\section{One}\nBody \\includegraphics{fig1}\n"
        "\\end{document}\n")
    assert find_main_tex(tmp_path) == main
    flat = flatten_tex(main)
    assert "Hello intro." in flat and "\\input{sections/intro}" not in flat
    s = survey_project(tmp_path)
    assert s.main_tex == main
    assert [f.name for f in s.figures] == ["fig1.png"]


def test_survey_refuses_non_projects(tmp_path):
    (tmp_path / "notes.tex").write_text("just a fragment, no documentclass")
    with pytest.raises(RefineError):
        survey_project(tmp_path)


def test_split_document_boundaries():
    flat = ("\\documentclass{article}\\begin{document}\\title{X}\\maketitle"
            "\\begin{abstract}A.\\end{abstract}"
            "\\section{Intro}alpha\\section{Results}beta"
            "\\bibliography{refs}\\end{document}")
    d = split_document(flat)
    assert "abstract" in d["opening"]
    assert [s["head"] for s in d["sections"]] == ["\\section{Intro}",
                                                  "\\section{Results}"]
    assert "\\bibliography{refs}" in d["closing"]
    assert "\\end{document}" in d["closing"]


# ------------------------------------------------------------- end-to-end pipeline

_ORIGINAL = r"""\documentclass{article}
\begin{document}
\title{A Modest Result}
\maketitle
\begin{abstract}
We study a toy system and find the constant equals 42.
\end{abstract}
\section{Introduction}
alpha This work concerns a toy system with exactly 42 states.
\begin{equation}Z = 42\end{equation}
\section{Discussion}
beta We conclude that the constant of the toy system equals 42 in every
configuration we examined across the analysis.
\end{document}
"""


class ScriptedOl:
    """Answers by stage: understanding, then rewrites (one honest, one that keeps
    inventing a number), then enrichment (one anchored, one fabricated)."""
    providers = {}

    def __init__(self):
        self.calls = []

    def chat(self, model, messages, **kw):
        system = messages[0]["content"]
        user = next((m["content"] for m in messages[1:] if m["role"] == "user"), "")
        self.calls.append(system[:40])
        if "reading a LaTeX research manuscript" in system:
            out = {"title": "A Modest Result", "field": "toy physics",
                   "summary": "s", "contributions": ["c"], "key_results": ["42"],
                   "terminology": ["toy system"],
                   "arxiv_categories": ["math-ph"], "queries": ["toy system constant"]}
        elif "repair LaTeX compile errors" in system:
            out = {"tex": ""}
        elif "connect a manuscript to its literature" in system:
            out = {"connections": [
                {"sentence": "Our constant matches the exactly solvable family",
                 "cite_id": "2401.11111",
                 "anchor": "the exactly solvable family with forty-two states"},
                {"sentence": "This proves the grand conjecture",
                 "cite_id": "2401.11111",
                 "anchor": "a quote that appears nowhere in the paper at all"}],
                "stronger_angles": [
                    {"idea": "Push toward the solvable-family classification",
                     "cite_id": "2401.11111",
                     "anchor": "the exactly solvable family with forty-two states"}]}
        elif "Rewrite the LaTeX fragment" in system:
            if "alpha" in user:
                out = {"tex": "In this toy system with exactly 42 states, we develop "
                              "the following.\n\\begin{equation}Z = 42\\end{equation}\n"}
            elif "beta" in user:
                out = {"tex": "beta We conclude the constant is 42, with p=0.001 "
                              "significance."}     # invents 0.001 — must be rejected
            else:
                out = {"tex": user.split("FRAGMENT", 1)[-1].split(":\n", 1)[-1]}
        else:
            out = {}
        return SimpleNamespace(text=json.dumps(out), prompt_tokens=5,
                               completion_tokens=7, raw={})


def _tree_hashes(root: Path, skip: str) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and skip not in str(p)}


def test_refine_end_to_end(tmp_path, monkeypatch):
    from spiral.config import Config
    from spiral.research_corpus import Paper

    proj = tmp_path / "paper"
    proj.mkdir()
    (proj / "main.tex").write_text(_ORIGINAL)
    before = _tree_hashes(proj, skip="spiral-refined")

    cfg = Config()
    run = RefineRun(proj, cfg=cfg, ol=ScriptedOl())

    def fake_build(query, k=8, categories=None, on=None):
        p = Paper(arxiv_id="2401.11111v1", title="Solvable Families",
                  abstract="We classify the exactly solvable family with forty-two "
                           "states and its constants.",
                  text="Body: the exactly solvable family with forty-two states.",
                  authors=["A. Author"], published="2024-01-01")
        run.corpus.papers[p.bare_id] = p
        return [p]

    monkeypatch.setattr(run.corpus, "build", fake_build)
    monkeypatch.setattr(run.corpus, "graph_deepen",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("offline")))
    art = run.run()

    # 1. the original tree is byte-identical
    assert _tree_hashes(proj, skip="spiral-refined") == before

    # 2. outputs land in the new folder only
    refined = proj / "spiral-refined" / "refined" / "main.tex"
    assert refined.is_file()
    tex = refined.read_text()

    # 3. the honest rewrite landed; the inventing rewrite was rejected (verbatim kept)
    assert "we develop the following" in tex
    assert "p=0.001" not in tex
    assert any(k["why"] == "gates" for k in run.report["kept_verbatim"])

    # 4. anchored enrichment woven with a real \cite; fabricated one dropped
    assert "exactly solvable family" in tex and "\\cite{" in tex
    assert len(run.report["enriched"]) == 1
    sugg = (proj / "spiral-refined" / "suggestions.md").read_text()
    assert "grand conjecture" in sugg           # dropped connection is reported
    assert "stronger-angle" in sugg             # angles are never auto-inserted
    assert "grand conjecture" not in tex

    # 5. merged bibliography exists and closing was rewired to it
    assert (proj / "spiral-refined" / "refined" / "references.bib").is_file()
    assert "\\bibliography{references}" in tex

    # 6. report written
    assert (proj / "spiral-refined" / "refine-report.md").is_file()

    # 7. with a TeX toolchain present, both PDFs must exist
    if shutil.which("pdflatex") or shutil.which("latexmk"):
        assert art["pdf"], f"submittable pdf missing: {art['compile_log']}"
        if shutil.which("latexdiff"):
            assert art["diff_pdf"], "blue-edit diff pdf missing"
            assert Path(art["diff_pdf"]).stat().st_size > 1000


def test_api_flag_requires_and_injects_key(monkeypatch, capsys):
    import os
    from spiral.cli import _apply_tier
    from spiral.config import Config

    class Spec:
        def __init__(self):
            self.name = "local"

    class Console:
        def print(self, *a, **k):
            pass

    cfg = Config()
    cfg.providers = {"kimi-k3": {"base_url": "https://api.moonshot.ai/v1",
                                 "api_key_env": "SPIRAL_TEST_KEY_ENV"}}
    cfg.worker, cfg.planner = Spec(), Spec()
    cfg.escalation, cfg.critic = Spec(), Spec()
    monkeypatch.delenv("SPIRAL_TEST_KEY_ENV", raising=False)
    _apply_tier(cfg, Console(), "api", api_key="sk-refine-test")
    assert os.environ.get("SPIRAL_TEST_KEY_ENV") == "sk-refine-test"
    assert cfg.critic.name == "kimi-k3"
