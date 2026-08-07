"""Documents keep their structure through an edit.

The failure this guards against: flattening a paper to plain text, rewriting the lot,
and saving something that is no longer a paper — preamble gone, equations reworded,
Word styles reset. Editing happens per paragraph, through the document's own structure.
"""
import tempfile
from pathlib import Path

import pytest

from spiral.documents import (
    default_output_path, read_document, write_document,
)

TEX = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
\section{Introduction}
Additionally, this framework stands as a testament to the enduring interplay.

\begin{equation}
E = mc^2
\end{equation}

\subsection{Method}
% an internal note
We measured 42 samples \cite{smith2020}.
\end{document}"""


def _tmp(name: str, body: str = "") -> Path:
    p = Path(tempfile.mkdtemp()) / name
    if body:
        p.write_text(body)
    return p


def test_tex_protects_math_preamble_and_sectioning():
    doc = read_document(_tmp("p.tex", TEX))
    protected = "\n".join(s.text for s in doc.segments if not s.editable)
    editable = "\n".join(s.text for s in doc.editable_segments)
    for must_protect in (r"\documentclass", r"\usepackage", r"\begin{document}",
                         r"\section{Introduction}", r"E = mc^2", r"\subsection{Method}",
                         "% an internal note", r"\end{document}"):
        assert must_protect in protected, f"{must_protect} was left editable"
        assert must_protect not in editable
    assert "stands as a testament" in editable


def test_tex_edit_replaces_only_the_targeted_paragraph():
    src = _tmp("p.tex", TEX)
    doc = read_document(src)
    target = doc.editable_segments[0]
    out = write_document(doc, {target.index: "This framework links theory and practice."},
                         src.with_name("p.edited.tex"))
    edited = out.read_text()
    assert "This framework links theory and practice." in edited
    assert "stands as a testament" not in edited
    assert r"E = mc^2" in edited and r"\documentclass{article}" in edited


def test_original_is_never_modified():
    src = _tmp("p.tex", TEX)
    before = src.read_text()
    doc = read_document(src)
    write_document(doc, {doc.editable_segments[0].index: "replaced"},
                   src.with_name("p.edited.tex"))
    assert src.read_text() == before


def test_markdown_headings_are_not_rewritten():
    doc = read_document(_tmp("n.md", "# Title\n\nSome prose here that can change.\n"))
    assert any(s.kind == "heading" and not s.editable for s in doc.segments)
    assert any("Some prose" in s.text for s in doc.editable_segments)


def test_default_output_never_overwrites_the_source():
    assert default_output_path(Path("a/paper.tex")).name == "paper.edited.tex"
    assert default_output_path(Path("a/report.docx")).name == "report.edited.docx"
    # a PDF cannot be faithfully rewritten, so its edit is emitted as markdown
    assert default_output_path(Path("a/scan.pdf")).name == "scan.edited.md"


def test_with_replacements_does_not_mutate_the_document():
    doc = read_document(_tmp("p.tex", TEX))
    idx = doc.editable_segments[0].index
    once = doc.with_replacements({idx: "NEW"})
    assert "NEW" in once
    assert "NEW" not in doc.text()          # pure function


# ------------------------------------------------------------------ docx
docx = pytest.importorskip("docx")


def test_docx_keeps_headings_and_styles_through_an_edit():
    path = _tmp("r.docx")
    d = docx.Document()
    d.add_heading("Results", level=1)
    d.add_paragraph("Additionally, this stands as a testament to the robust interplay.")
    d.save(str(path))

    doc = read_document(path)
    assert not doc.segments[0].editable, "a heading must not be rewritten"
    assert doc.segments[1].editable

    out = write_document(doc, {1: "This connects several ideas."},
                         path.with_name("r.edited.docx"))
    back = docx.Document(str(out))
    assert back.paragraphs[0].text == "Results"
    assert back.paragraphs[0].style.name.lower().startswith("heading")
    assert "testament" not in back.paragraphs[1].text
    assert "This connects several ideas." == back.paragraphs[1].text
