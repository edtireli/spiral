"""LaTeX write-up with real citations — the loop's output artefact.

Turns a completed research state (question, verified findings, corpus) into an arXiv-style
LaTeX article with a BibTeX bibliography built from the actual corpus papers, and compiles
it to PDF. Citations are DETERMINISTIC — every ``\\cite{key}`` resolves to a real paper in
the store, generated from metadata, not invented by the model (which is exactly how
fabricated references get into papers). The model writes prose; the machine writes the
bibliography and wires the keys.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def _bibkey(paper) -> str:
    """Stable citation key from first author + year + arXiv id."""
    a = (paper.authors[0].split()[-1] if paper.authors else "arxiv").lower()
    a = re.sub(r"[^a-z]", "", a) or "ref"
    yr = (paper.published[:4] if paper.published else "") or paper.bare_id[:2]
    return f"{a}{yr}_{paper.bare_id.replace('.', '').replace('/', '')}"


def bibtex_from_corpus(papers) -> tuple[str, dict[str, str]]:
    """A BibTeX string + ``{arxiv_id: citekey}`` map from real corpus papers."""
    entries, keymap = [], {}
    for p in papers:
        key = _bibkey(p)
        keymap[p.bare_id] = key
        authors = " and ".join(p.authors) or "Unknown"
        title = (p.title or "Untitled").replace("{", "").replace("}", "")
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{(p.published[:4] if p.published else 'n.d.')}}},\n"
            f"  eprint = {{{p.bare_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  url = {{{p.url or f'https://arxiv.org/abs/{p.bare_id}'}}},\n"
            f"}}"
        )
    return "\n\n".join(entries), keymap


_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\title{%(title)s}
\author{spiral\textsuperscript{research}}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
%(abstract)s
\end{abstract}

%(body)s

\bibliographystyle{plain}
\bibliography{refs}
\end{document}
"""


def build_document(title: str, abstract: str, body_tex: str, papers, out_dir: str | Path
                   ) -> Path:
    """Write ``paper.tex`` + ``refs.bib`` into ``out_dir``; return the .tex path.

    ``body_tex`` is the model's prose/derivations, already citing papers by their
    **arXiv id** in ``\\cite{arXiv:ID}`` form; those are rewritten to the real BibTeX
    keys here, and any citation to a paper not in the corpus is dropped rather than left
    dangling — a citation must point at a source we actually hold."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bib, keymap = bibtex_from_corpus(papers)
    (out_dir / "refs.bib").write_text(bib, encoding="utf-8")

    def _resolve(m):
        ids = [i.strip().replace("arXiv:", "").split("v")[0] for i in m.group(1).split(",")]
        keys = [keymap[i] for i in ids if i in keymap]
        return f"\\cite{{{','.join(keys)}}}" if keys else ""
    body = re.sub(r"\\cite\{([^}]*)\}", _resolve, body_tex)

    tex = _TEMPLATE % {"title": title or "Untitled", "abstract": abstract.strip(),
                       "body": body.strip()}
    tp = out_dir / "paper.tex"
    tp.write_text(tex, encoding="utf-8")
    return tp


def compile_pdf(tex_path: str | Path) -> Path | None:
    """Compile to PDF with latexmk (or a pdflatex/bibtex/pdflatex×2 fallback). Returns
    the PDF path, or None if no TeX toolchain is installed (the .tex is still written)."""
    tex_path = Path(tex_path)
    d, stem = tex_path.parent, tex_path.stem
    if shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-silent", tex_path.name]
        seq = [cmd]
    elif shutil.which("pdflatex"):
        seq = [["pdflatex", "-interaction=nonstopmode", tex_path.name],
               ["bibtex", stem], ["pdflatex", "-interaction=nonstopmode", tex_path.name],
               ["pdflatex", "-interaction=nonstopmode", tex_path.name]]
    else:
        return None
    for cmd in seq:
        try:
            subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=180)
        except Exception:
            break
    pdf = d / f"{stem}.pdf"
    return pdf if pdf.is_file() else None
