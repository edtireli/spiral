"""Read and write real documents — .tex, .docx, .md/.txt, .pdf — without flattening them.

The naive way to "edit a document with a model" is to extract plain text, rewrite the
whole thing, and save the result. That destroys everything a document actually is:
LaTeX structure, Word styles, headings, tables, equations. It also makes the edit
unreviewable, because the output shares no structure with the input.

So this module works in **segments**. A document is read as an ordered list of editable
paragraphs plus the scaffolding between them. A rewrite touches one paragraph at a time,
each is independently accepted or rejected, and the document is written back through the
same structure it came from — a .docx stays a .docx with its styles, a .tex keeps its
preamble, macros and math.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# LaTeX constructs whose contents must never be handed to a prose rewriter
_TEX_PROTECTED = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray|figure|table|tabular|"
    r"lstlisting|verbatim|algorithm|thebibliography)\*?\}.*?"
    r"\\end\{\1\*?\}|"
    r"\$\$.*?\$\$|"
    r"\\\[.*?\\\]|"
    # NOTE: [^\n]* not .* — this pattern is compiled with DOTALL for the environment
    # alternatives above, and a greedy .* here swallowed the entire document.
    r"^[ \t]*\\(?:documentclass|usepackage|newcommand|renewcommand|def|input|include|"
    r"bibliography|bibliographystyle)\b[^\n]*$",
    re.S | re.M)


@dataclass
class Segment:
    """One editable unit. ``editable`` is False for scaffolding that must pass through
    byte-identical (preamble, equations, figures, code)."""
    index: int
    text: str
    editable: bool = True
    kind: str = "paragraph"


@dataclass
class Document:
    path: Path
    kind: str                      # tex | docx | markdown | pdf
    segments: list = field(default_factory=list)
    _backing: object = None        # docx Document, when applicable

    @property
    def editable_segments(self) -> list:
        return [s for s in self.segments if s.editable and s.text.strip()]

    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)

    def with_replacements(self, replacements: dict) -> str:
        """Full text with ``{index: new_text}`` applied — used for measuring the result
        before anything is written to disk."""
        out = []
        for s in self.segments:
            out.append(replacements.get(s.index, s.text))
        return "\n\n".join(out)


# ── reading ──────────────────────────────────────────────────────────────────
def _read_tex(path: Path) -> Document:
    raw = path.read_text(errors="replace")
    segments: list[Segment] = []
    cursor = 0
    idx = 0

    # a line that is nothing but a LaTeX command is structure, not prose — it must not
    # be handed to a rewriter even when it sits inside a paragraph block
    command_line = re.compile(
        r"^[ \t]*\\(?:begin|end|section|subsection|subsubsection|chapter|part|"
        r"paragraph|title|author|date|maketitle|label|caption|item|bibliography\w*|"
        r"appendix|tableofcontents|newpage|clearpage|noindent|centering)\b[^\n]*$|"
        r"^[ \t]*%[^\n]*$")

    def add_prose(chunk: str) -> None:
        nonlocal idx
        for para in re.split(r"\n\s*\n", chunk):
            if not para.strip():
                continue
            buf: list[str] = []

            def flush_buf() -> None:
                nonlocal idx, buf
                if buf and "".join(buf).strip():
                    segments.append(Segment(idx, "\n".join(buf), editable=True,
                                            kind="paragraph"))
                    idx += 1
                buf = []

            for line in para.split("\n"):
                if command_line.match(line):
                    flush_buf()
                    segments.append(Segment(idx, line, editable=False, kind="command"))
                    idx += 1
                else:
                    buf.append(line)
            flush_buf()

    for m in _TEX_PROTECTED.finditer(raw):
        add_prose(raw[cursor:m.start()])
        segments.append(Segment(idx, m.group(0), editable=False, kind="protected"))
        idx += 1
        cursor = m.end()
    add_prose(raw[cursor:])
    return Document(path, "tex", segments)


def _read_docx(path: Path) -> Document:
    try:
        import docx
    except ImportError as exc:                       # pragma: no cover - env dependent
        raise RuntimeError(
            "reading .docx needs python-docx: pip install python-docx") from exc
    doc = docx.Document(str(path))
    segments: list[Segment] = []
    for i, para in enumerate(doc.paragraphs):
        style = (para.style.name if para.style is not None else "") or ""
        # headings and captions carry meaning in few words; rewriting them adds risk
        # without benefit, so they pass through untouched
        editable = not re.match(r"(?i)heading|title|caption|toc|quote", style)
        segments.append(Segment(i, para.text, editable=editable,
                                kind=style.lower() or "paragraph"))
    return Document(path, "docx", segments, _backing=doc)


def _read_flat(path: Path, kind: str) -> Document:
    raw = (path.read_text(errors="replace") if kind != "pdf"
           else _read_pdf_text(path))
    segments = []
    for i, para in enumerate(re.split(r"\n\s*\n", raw)):
        heading = bool(re.match(r"\s*#{1,6}\s", para))
        segments.append(Segment(i, para, editable=not heading,
                                kind="heading" if heading else "paragraph"))
    return Document(path, kind, segments)


def _read_pdf_text(path: Path) -> str:
    from spiral.research_corpus import extract_pdf_text

    return extract_pdf_text(path)


def read_document(path: str | Path) -> Document:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".tex":
        return _read_tex(p)
    if suffix == ".docx":
        return _read_docx(p)
    if suffix == ".pdf":
        return _read_flat(p, "pdf")
    return _read_flat(p, "markdown")


# ── writing ──────────────────────────────────────────────────────────────────
def write_document(doc: Document, replacements: dict, out_path: str | Path) -> Path:
    """Write the edited document, preserving its native structure.

    A .docx is written back through python-docx so styles, headings and everything the
    rewriter never touched survive; a .tex is reassembled with its preamble and math
    exactly as they were. A PDF cannot be faithfully rewritten, so its edit is emitted
    as markdown and that is stated rather than pretended otherwise."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if doc.kind == "docx" and doc._backing is not None:
        backing = doc._backing
        for seg_index, new_text in replacements.items():
            if seg_index >= len(backing.paragraphs):
                continue
            para = backing.paragraphs[seg_index]
            if not para.runs:
                para.text = new_text
                continue
            # keep the first run's formatting, drop the rest — the paragraph keeps its
            # font/style instead of reverting to document defaults
            para.runs[0].text = new_text
            for run in para.runs[1:]:
                run.text = ""
        backing.save(str(out))
        return out

    out.write_text(doc.with_replacements(replacements))
    return out


def default_output_path(path: str | Path) -> Path:
    """Where an edit lands by default: alongside the original, never over it."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return p.with_suffix(".edited.md")       # a PDF edit is emitted as markdown
    return p.with_name(f"{p.stem}.edited{p.suffix or '.txt'}")
