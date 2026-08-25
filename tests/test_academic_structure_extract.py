from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import pytest

from scripts.academic_finetune.structure_extract import (
    PaperStructure,
    StructureLimits,
    parse_jats_structure,
    parse_tex_structure_archive,
)


def _tex_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_tex_archive_preserves_hierarchy_counts_placements_and_back_matter() -> None:
    payload = _tex_archive(
        {
            "paper/main.tex": rb"""\documentclass{article}
\title{A Bounded Study of Two Fields}
\begin{document}
\maketitle
\begin{abstract}
We test a bounded structural extraction method on a controlled example.
\end{abstract}
\input{sections/introduction}
\include{sections/results}
\section*{Acknowledgments}
We thank the colleagues who discussed this example.
\appendix
\section{Auxiliary checks}
The appendix records a secondary check that must not enter the main count.
\subsection{Proof details}
The auxiliary proof closes the remaining special case.
\bibliography{paper}
\end{document}
""",
            "paper/sections/introduction.tex": rb"""\section{Introduction}
The first paragraph motivates the controlled physical question.

\begin{figure}
  \includegraphics{diagram.pdf}
  \caption{A diagram that is not counted as body prose.}
  \label{fig:overview}
\end{figure}

The second paragraph states the scope of the argument.
\subsection{Motivation}
This subsection explains why the comparison is useful.
""",
            "paper/sections/results.tex": rb"""\section{Results}
The measured response agrees with the bounded prediction.

\begin{table}
  \caption{Values that are not counted as body prose.}
  \label{tab:result}
\end{table}

The comparison remains stable under the stated perturbation.
""",
            "paper/diagram.pdf": b"not read or decoded",
        }
    )

    paper = parse_tex_structure_archive(payload)

    assert paper.source_format == "arxiv_tex"
    assert paper.title == "A Bounded Study of Two Fields"
    assert [node.title for node in paper.sections] == [
        "Introduction",
        "Results",
        "Acknowledgments",
        "References",
    ]
    introduction, results, acknowledgments, references = paper.sections
    assert introduction.path == (1,) and introduction.order == 1
    assert [child.title for child in introduction.children] == ["Motivation"]
    assert introduction.children[0].path == (1, 1)
    assert introduction.direct_paragraph_count == 2
    assert introduction.direct_figure_count == 1
    assert introduction.placements[0].after_paragraph == 1
    assert introduction.placements[0].identifier == "fig:overview"
    assert results.direct_paragraph_count == 2
    assert results.direct_table_count == 1
    assert results.placements[0].after_paragraph == 1
    assert results.placements[0].identifier == "tab:result"
    assert acknowledgments.included_in_main is False
    assert acknowledgments.exclusion_reason == "acknowledgments"
    assert references.included_in_main is False
    assert references.exclusion_reason == "references"
    assert [node.title for node in paper.appendices] == ["Auxiliary checks"]
    assert paper.appendices[0].exclusion_reason == "appendix"
    assert paper.appendices[0].children[0].title == "Proof details"
    assert paper.main_figure_count == 1 and paper.main_table_count == 1
    assert paper.main_word_count < sum(node.word_count for node in paper.sections + paper.appendices)
    assert paper.abstract_word_count == 11 and paper.abstract_paragraph_count == 1
    assert paper.provenance.root_member == "paper/main.tex"
    assert paper.provenance.included_members == (
        "paper/main.tex",
        "paper/sections/introduction.tex",
        "paper/sections/results.tex",
    )
    assert PaperStructure.from_dict(paper.to_dict()) == paper


def test_tex_single_gzip_is_supported_and_section_stars_are_classified() -> None:
    source = rb"""\documentclass{article}
\begin{document}
\section{Main}
One substantive paragraph remains in the main body.
\section*{References}
This citation prose is excluded from the main body.
\end{document}
"""
    paper = parse_tex_structure_archive(gzip.compress(source))
    assert [node.title for node in paper.sections] == ["Main", "References"]
    assert paper.sections[1].exclusion_reason == "references"
    assert paper.main_word_count == paper.sections[0].word_count


def test_tex_include_cycles_escapes_and_depth_are_rejected() -> None:
    cycle = _tex_archive(
        {
            "paper/main.tex": rb"\begin{document}\input{parts/a}\end{document}",
            "paper/parts/a.tex": rb"\input{../main}",
        }
    )
    with pytest.raises(ValueError, match="include cycle"):
        parse_tex_structure_archive(cycle)

    escape = _tex_archive(
        {"paper/main.tex": rb"\begin{document}\input{../../../etc/passwd}\end{document}"}
    )
    with pytest.raises(ValueError, match="escapes the source archive"):
        parse_tex_structure_archive(escape)

    deep = _tex_archive(
        {
            "main.tex": rb"\begin{document}\input{one}\end{document}",
            "one.tex": rb"\input{two}",
            "two.tex": rb"\section{Too deep}Body prose.",
        }
    )
    with pytest.raises(ValueError, match="include depth"):
        parse_tex_structure_archive(deep, limits=StructureLimits(max_include_depth=1))


def test_tex_archive_member_and_unpacked_limits_are_enforced() -> None:
    unsafe = _tex_archive({"../outside.tex": rb"\begin{document}unsafe\end{document}"})
    with pytest.raises(ValueError, match="unsafe path"):
        parse_tex_structure_archive(unsafe)

    payload = _tex_archive({"main.tex": b"x" * 128})
    with pytest.raises(ValueError, match="per-file size limit"):
        parse_tex_structure_archive(payload, limits=StructureLimits(max_member_bytes=64))


def test_jats_preserves_nested_sections_and_separates_back_matter() -> None:
    payload = b"""<article xmlns:mml="http://www.w3.org/1998/Math/MathML">
  <front><article-meta>
    <article-id pub-id-type="pmc">PMC123</article-id>
    <article-id pub-id-type="doi">10.1000/example</article-id>
    <title-group><article-title>A Structured Biomedical Study</article-title></title-group>
    <abstract><p>We summarize the controlled biomedical comparison.</p></abstract>
  </article-meta></front>
  <body>
    <p>This unsectioned prelude introduces the clinical setting.</p>
    <sec id="s1"><title>Introduction</title>
      <p>The first paragraph defines the study population <xref ref-type="bibr">[1]</xref>.</p>
      <fig id="fig1"><caption><p>Caption prose is not body prose.</p></caption></fig>
      <sec id="s1.1"><title>Prior evidence</title>
        <p>The nested section describes the relevant prior evidence.</p>
        <table-wrap id="table1"><caption><p>Excluded caption.</p></caption><table/></table-wrap>
      </sec>
    </sec>
    <sec id="s2"><title>Results</title>
      <p>The observed response remains stable at follow-up.</p>
    </sec>
    <sec sec-type="appendix"><title>Appendix A</title>
      <p>This sensitivity analysis is retained outside the main text.</p>
    </sec>
  </body>
  <back>
    <ack><p>We thank the study participants and clinical staff.</p></ack>
    <ref-list><title>References</title><ref><mixed-citation>Never count this citation.</mixed-citation></ref></ref-list>
    <app-group><app><title>Supplementary protocol</title>
      <p>This supplementary protocol remains separately available.</p>
      <sec><title>Additional cohort</title><p>The cohort details are retained.</p></sec>
    </app></app-group>
  </back>
</article>"""

    paper = parse_jats_structure(payload)

    assert paper.source_format == "pmc_jats"
    assert paper.title == "A Structured Biomedical Study"
    assert paper.abstract_paragraph_count == 1
    assert paper.unsectioned_paragraph_count == 1
    assert [node.title for node in paper.sections] == [
        "Introduction",
        "Results",
        "Acknowledgments",
        "References",
    ]
    introduction = paper.sections[0]
    assert introduction.direct_paragraph_count == 1
    assert introduction.placements[0].identifier == "fig1"
    assert introduction.placements[0].after_paragraph == 1
    assert introduction.children[0].title == "Prior evidence"
    assert introduction.children[0].path == (1, 1)
    assert introduction.children[0].direct_table_count == 1
    assert paper.sections[2].exclusion_reason == "acknowledgments"
    assert paper.sections[3].exclusion_reason == "references"
    assert [node.title for node in paper.appendices] == [
        "Appendix A",
        "Supplementary protocol",
    ]
    assert paper.appendices[1].children[0].title == "Additional cohort"
    assert paper.main_figure_count == 1 and paper.main_table_count == 1
    assert paper.provenance.identifiers == (
        ("doi", "10.1000/example"),
        ("pmc", "PMC123"),
    )
    assert PaperStructure.from_dict(paper.to_dict()) == paper


def test_jats_rejects_declarations_and_obeys_depth_limit() -> None:
    declared = b'<!DOCTYPE article [<!ENTITY x "expanded">]><article><body/></article>'
    with pytest.raises(ValueError, match="declarations and entities"):
        parse_jats_structure(declared)

    external = b'''<!DOCTYPE article PUBLIC "-//NLM//DTD JATS 1.3//EN" "JATS.dtd">
      <article><body><sec><title>Main</title><p>External DTDs are not resolved.</p></sec></body></article>'''
    parsed = parse_jats_structure(external)
    assert parsed.sections[0].title == "Main"
    assert parsed.provenance.warnings == ("external_doctype_ignored",)

    deep = b"<article><body><boxed-text><list><list-item><p>Text.</p></list-item></list></boxed-text></body></article>"
    with pytest.raises(ValueError, match="XML-depth limit"):
        parse_jats_structure(deep, limits=StructureLimits(max_xml_depth=4))


def test_jats_untitled_reference_wrapper_is_explicitly_excluded() -> None:
    payload = b"""<article><body>
      <sec><title>Discussion</title><p>Main discussion prose remains here.</p></sec>
      <sec><ref-list><p>Reference-list prose must never enter main counts.</p></ref-list></sec>
    </body></article>"""
    paper = parse_jats_structure(payload)
    assert [node.title for node in paper.sections] == ["Discussion", "References"]
    assert paper.sections[1].included_in_main is False
    assert paper.sections[1].exclusion_reason == "references"
    assert paper.sections[1].word_count == 0


def test_jats_body_level_back_matter_never_leaks_into_main_counts() -> None:
    payload = b"""<article><body>
      <sec><title>Main</title><p>Only this sentence belongs to the main article.</p></sec>
      <ack><p>Acknowledgment words stay outside the main article.</p></ack>
      <ref-list><p>Reference words stay outside the main article.</p></ref-list>
      <app><title>Appendix A</title><p>Appendix words stay outside the main article.</p></app>
    </body></article>"""
    paper = parse_jats_structure(payload)
    assert [(node.title, node.exclusion_reason) for node in paper.sections] == [
        ("Main", ""),
        ("Acknowledgments", "acknowledgments"),
        ("References", "references"),
    ]
    assert [node.title for node in paper.appendices] == ["Appendix A"]
    assert paper.main_word_count == paper.sections[0].word_count


def test_parser_module_does_not_write_extracted_archive_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    payload = _tex_archive(
        {"main.tex": rb"\begin{document}\section{Main}A short body paragraph.\end{document}"}
    )
    parse_tex_structure_archive(payload)
    assert list(tmp_path.iterdir()) == []
