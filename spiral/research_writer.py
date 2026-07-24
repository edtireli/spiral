"""LaTeX write-up with real citations — the loop's output artefact.

Turns a completed research state (question, verified findings, corpus) into an arXiv-style
LaTeX article with a BibTeX bibliography built from the actual corpus papers, and compiles
it to PDF. Citations are DETERMINISTIC — every ``\\cite{key}`` resolves to a real paper in
the store, generated from metadata, not invented by the model (which is exactly how
fabricated references get into papers). The model writes prose; the machine writes the
bibliography and wires the keys.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
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
        # Corpus metadata is data, not executable TeX. Escaping it also prevents an
        # upstream title from smuggling file-reading primitives into compilation.
        authors = " and ".join(_tex_escape(a) for a in p.authors) or "Unknown"
        title = _tex_escape(p.title or "Untitled")
        year = re.sub(
            r"[^0-9]", "", (p.published[:4] if p.published else ""))[:4] or "n.d."
        arxiv_id = re.sub(r"[^A-Za-z0-9./-]", "", str(p.bare_id))[:80]
        entries.append(
            f"@article{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            f"  eprint = {{{arxiv_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  url = {{https://arxiv.org/abs/{arxiv_id}}},\n"
            f"}}"
        )
    return "\n\n".join(entries), keymap


_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}
\usepackage{amsmath,amssymb,amsthm}
\IfFileExists{mathtools.sty}{\usepackage{mathtools}}{}
\IfFileExists{microtype.sty}{\usepackage{microtype}}{}
\IfFileExists{booktabs.sty}{\usepackage{booktabs}}{}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{}
\usepackage{graphicx}
\usepackage[margin=1in,headheight=14pt]{geometry}
\usepackage[dvipsnames]{xcolor}
\usepackage[colorlinks=true,linkcolor=MidnightBlue,citecolor=MidnightBlue,
            urlcolor=BrickRed,pdfborder={0 0 0}]{hyperref}
\usepackage{listings}
\numberwithin{equation}{section}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\newtheorem{example}{Example}[section]
\theoremstyle{remark}
\newtheorem{remark}{Remark}[section]
\allowdisplaybreaks[2]
\setlength{\emergencystretch}{3em}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,frame=single,
        columns=fullflexible,keepspaces=true}
\title{%(title)s}
\author{%(author)s%(affil)s}
\date{\today}
\begin{document}
\maketitle
\begin{abstract}
%(abstract)s
\end{abstract}

%(body)s

%(appendix)s

\bibliographystyle{amsplain}
\bibliography{refs}
\end{document}
"""


def _tex_escape(s: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
        "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
        "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return "".join(mapping.get(char, char) for char in str(s))


def _tex_escape_plain(s: str) -> str:
    """Escape a title/abstract while preserving explicitly delimited inline math."""

    s = re.sub(r"\\\((.*?)\\\)", lambda m: f"${m.group(1)}$", str(s), flags=re.S)
    parts = re.split(r"(\$[^$\n]+\$)", s)
    return "".join(part if part.startswith("$") and part.endswith("$") else _tex_escape(part)
                   for part in parts)


def strip_latex_wrappers(tex: str) -> str:
    """Remove model-added fences/preamble/document wrappers from a body fragment."""
    tex = (tex or "").strip()
    tex = re.sub(r"^```(?:latex|tex)?\s*", "", tex, flags=re.I)
    tex = re.sub(r"\s*```$", "", tex)
    tex = re.sub(r"\\documentclass(?:.|\n)*?\\begin\{document\}", "", tex, flags=re.S)
    tex = re.sub(r"\\begin\{document\}", "", tex)
    tex = re.sub(r"\\end\{document\}", "", tex)
    tex = re.sub(r"\\begin\{abstract\}(?:.|\n)*?\\end\{abstract\}", "", tex, flags=re.S)
    tex = re.sub(r"\\maketitle", "", tex)
    return tex.strip()


def normalise_section_fragment(tex: str, expected_name: str) -> str:
    """Make a one-section model response obey the outline's heading contract."""

    tex = strip_latex_wrappers(tex)
    expected = re.sub(r"[{}\\]", "", str(expected_name or "Section")).strip() or "Section"
    matches = list(_SECTION_RE.finditer(tex))
    if not matches:
        return f"\\section{{{expected}}}\n{tex}"
    first = matches[0]
    tex = tex[:first.start()] + f"\\section{{{expected}}}" + tex[first.end():]
    # A section worker occasionally writes a miniature whole paper. Preserve its prose,
    # but demote extra top-level headings so it cannot mutate the global outline.
    first_end = tex.find("}", tex.find("\\section{")) + 1
    return tex[:first_end] + re.sub(r"\\section\*?\{", r"\\subsection{", tex[first_end:])


_SECTION_RE = re.compile(r"\\section\*?\{([^{}]{1,100})\}")
_ANY_SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^{}]{1,100})\}")
_ENV_RE = re.compile(r"\\begin\{(theorem|proposition|lemma|definition|corollary|proof|remark|example)\}")
_EQUATION_RE = re.compile(
    r"\\begin\{(?:equation|align|gather|multline)\*?\}(.{1,1400}?)\\end\{(?:equation|align|gather|multline)\*?\}"
    r"|\$\$(.{1,1000}?)\$\$|\\\[(.{1,1000}?)\\\]",
    re.S,
)
_MACRO_RE = re.compile(r"\\(?:newcommand|renewcommand|DeclareMathOperator)\s*\{?\\([A-Za-z]+)\}?")
_SYMBOL_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z](?:_[A-Za-z0-9{}]+|\^[A-Za-z0-9{}]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "been", "being",
    "between", "both", "cannot", "could", "does", "from", "have", "into", "more",
    "most", "only", "other", "paper", "proof", "result", "results", "section", "show",
    "some", "such", "than", "that", "their", "there", "these", "this", "those", "through",
    "using", "where", "which", "while", "with", "within", "would",
}


def _clean_heading(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", s)
    s = re.sub(r"[$\\{}_^]", "", s)
    return " ".join(s.split()).strip()


def _heading_role(name: str) -> str:
    """Map a corpus heading to its rhetorical role without replacing its wording."""

    low = _clean_heading(name).lower()
    role_cues = (
        ("introduction", ("introduction", "overview", "motivation")),
        ("setup", (
            "preliminar", "background", "setup", "notation", "convention",
            "definitions", "geometr", "kinematic", "model and", "formulation",
        )),
        ("methods", (
            "method", "construction", "formalism", "algorithm", "computation",
            "numerical", "perturbation", "reduction", "equations of motion",
        )),
        ("results", (
            "main result", "results", "classification", "solution", "integrab",
            "spectrum", "response", "theorem",
        )),
        ("proof", ("proof", "derivation", "analysis of", "calculation")),
        ("discussion", (
            "discussion", "limitation", "comparison", "renormal", "outlook",
            "open problem", "physical interpretation",
        )),
        ("conclusion", ("conclusion", "summary", "concluding")),
        ("appendix", ("appendix", "supplement")),
    )
    for role, cues in role_cues:
        if any(cue in low for cue in cues):
            return role
    return "other"


def _role_intent(role: str) -> str:
    return {
        "introduction": "motivate the problem, delimit the literature boundary, and state the contribution",
        "setup": "fix assumptions, definitions, notation, and normalisations before calculations",
        "methods": "derive or explain the methods needed to obtain the stated results",
        "results": "state established results precisely with their evidence grades",
        "proof": "derive the claims and connect each step to exact or computational certificates",
        "discussion": "compare prior art and separate consequences, limitations, and open questions",
        "conclusion": "synthesise only what has been established and delimit what remains open",
        "appendix": "record reproducibility material and machine certificates",
        "other": "serve the corpus-derived argumentative arc without adding unsupported claims",
    }.get(role, "serve the paper's argumentative arc")


def _clean_latex_snippet(s: str, *, limit: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    s = re.sub(r"\\label\{[^{}]*\}", "", s)
    return s[:limit].strip()


def _normalise_tex_equation(value: str) -> str:
    """Normalize harmless TeX presentation differences for observable matching."""

    text = str(value or "")
    text = re.sub(
        r"\\begin\{(?:equation|align|alignat|gather|multline|displaymath|flalign|eqnarray)\*?\}",
        "", text)
    text = re.sub(
        r"\\end\{(?:equation|align|alignat|gather|multline|displaymath|flalign|eqnarray)\*?\}",
        "", text)
    text = re.sub(r"\\(?:label|tag)\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle|nonumber)\b", "", text)
    text = re.sub(r"\\(?:,|;|!|quad\b|qquad\b)", "", text)
    text = text.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    text = text.replace("$$", "").replace("$", "").replace("&", "")
    text = re.sub(r"\s+", "", text)
    return text.strip(".,;:")


def _sentences(text: str, *, limit: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text or "")
    return [s.strip() for s in _SENTENCE_RE.split(text) if 35 <= len(s.strip()) <= limit]


def _convention_sentences(text: str) -> list[str]:
    cues = (
        "throughout", "we denote", "we write", "we use the convention", "our convention",
        "let ", "fix ", "assume", "where ", "normalization", "normalisation",
        "metric signature", "indices", "summation convention",
    )
    out = []
    for s in _sentences(text, limit=220):
        low = s.lower()
        if any(c in low for c in cues):
            cleaned = _clean_latex_snippet(s, limit=260)
            if cleaned and cleaned not in out:
                out.append(cleaned)
    return out


def _equations(text: str, *, limit: int = 24) -> list[str]:
    out = []
    for m in _EQUATION_RE.finditer(text or ""):
        frag = next((g for g in m.groups() if g), "")
        frag = _clean_latex_snippet(frag, limit=260)
        if frag and frag not in out:
            out.append(frag)
        if len(out) >= limit:
            break
    return out


def _vocabulary(text: str, *, top: int = 24) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", (text or "").lower())
    words = [w.strip("-") for w in words if w.strip("-") not in _STOPWORDS and not w.startswith("arxiv")]
    counts = Counter(words)
    phrases = Counter()
    for a, b in zip(words, words[1:]):
        if a not in _STOPWORDS and b not in _STOPWORDS:
            phrases[f"{a} {b}"] += 1
    merged = []
    for k, _ in (phrases + counts).most_common(top * 2):
        if len(merged) >= top:
            break
        if k not in merged:
            merged.append(k)
    return merged


def corpus_writing_blueprint(papers, *, max_papers: int = 14) -> dict:
    """A deterministic scaffold the model must write against.

    The output is intentionally not prose. It is a field notebook: observed section
    arcs, notation/convention sentences, frequent vocabulary, equation idioms, and
    citation anchors. The writer can imitate structure and register without copying
    source sentences.
    """
    prof = corpus_style_profile(papers, max_papers=max_papers)
    macro_counter: Counter = Counter()
    symbol_counter: Counter = Counter()
    symbol_sources: dict[str, set[str]] = {}
    vocab_counter: Counter = Counter()
    convention_examples: list[dict] = []
    equation_examples: list[dict] = []
    citation_anchors: list[dict] = []

    for p in list(papers)[:max_papers]:
        text = p.text or p.abstract or ""
        if not text:
            continue
        macro_counter.update(_MACRO_RE.findall(text))
        eqs = _equations(text, limit=8)
        for eq in eqs:
            symbols = _SYMBOL_RE.findall(eq)
            symbol_counter.update(symbols)
            for symbol in symbols:
                symbol_sources.setdefault(symbol, set()).add(p.bare_id)
        for term in _vocabulary((p.title or "") + " " + (p.abstract or "") + " " + text[:5000], top=18):
            vocab_counter[term] += 1
        for s in _convention_sentences(text)[:4]:
            convention_examples.append({"paper": p.bare_id, "sentence": s})
        for eq in eqs[:4]:
            equation_examples.append({"paper": p.bare_id, "equation": eq})
        citation_anchors.append({
            "id": p.bare_id,
            "title": p.title,
            "use_for": _clean_latex_snippet((p.abstract or p.text or "")[:420], limit=260),
        })

    sections = prof.get("preferred_sections") or [
        "Introduction", "Setup", "Results", "Discussion and limitations",
    ]
    if "Appendix" not in sections:
        sections = list(sections) + ["Appendix"]
    return {
        "style_profile": prof,
        "section_template": [
            {
                "name": s,
                "rhetorical_role": _heading_role(s),
                "role": _role_intent(_heading_role(s)),
            }
            for s in sections[:8]
        ],
        "required_rhetorical_roles": [
            "introduction", "setup", "results", "scope",
        ],
        "notation_ledger": {
            "macros": [f"\\{m}" for m, _ in macro_counter.most_common(20)],
            "symbols": [s for s, _ in symbol_counter.most_common(28)],
            "symbol_entries": [
                {
                    "symbol": symbol,
                    "count": count,
                    "papers": sorted(symbol_sources.get(symbol, set()))[:12],
                }
                for symbol, count in symbol_counter.most_common(28)
            ],
            "convention_examples": convention_examples[:18],
        },
        "vocabulary": [t for t, _ in vocab_counter.most_common(32)],
        "equation_map": equation_examples[:24],
        "citation_anchors": citation_anchors[:max_papers],
        "audit_requirements": [
            "state conventions before using notation-heavy equations",
            "keep one notation system throughout the paper",
            "cite corpus papers for background claims and method lineage",
            "mark every machine-verified result as verified/proved/certified",
            "do not present unverified experiments as theorems",
        ],
    }


def blueprint_markdown(blueprint: dict) -> str:
    lines = ["# Corpus writing blueprint", ""]
    tmpl = blueprint.get("section_template") or []
    lines += ["## Section Template"]
    for s in tmpl:
        lines.append(f"- {s.get('name')}: {s.get('role')}")
    lines += ["", "## Notation And Conventions"]
    ledger = blueprint.get("notation_ledger") or {}
    lines.append("- Macros: " + (", ".join(ledger.get("macros") or []) or "none observed"))
    lines.append("- Symbols: " + (", ".join(ledger.get("symbols") or []) or "none observed"))
    for e in (ledger.get("convention_examples") or [])[:8]:
        lines.append(f"- {e.get('paper')}: {e.get('sentence')}")
    lines += ["", "## Vocabulary"]
    lines.append(", ".join(blueprint.get("vocabulary") or []) or "none observed")
    lines += ["", "## Equation Map"]
    for e in (blueprint.get("equation_map") or [])[:10]:
        lines.append(f"- {e.get('paper')}: `{e.get('equation')}`")
    lines += ["", "## Citation Anchors"]
    for c in (blueprint.get("citation_anchors") or [])[:10]:
        lines.append(f"- {c.get('id')}: {c.get('title')}")
    lines += ["", "## Audit Requirements"]
    for r in blueprint.get("audit_requirements") or []:
        lines.append(f"- {r}")
    return "\n".join(lines) + "\n"


def normalise_outline(outline: dict, blueprint: dict | None = None) -> dict:
    """Turn a model outline into a bounded, corpus-shaped rhetorical contract.

    Mathematical genres do not all use the same five literal headings. We retain the
    corpus/model wording and validate rhetorical coverage instead: context, foundations,
    established contribution, and a section that states scope or limitations.
    """

    bp = blueprint or {}
    allowed_roles = {
        "introduction", "setup", "methods", "results", "proof",
        "discussion", "conclusion", "other",
    }
    template = [
        row for row in (bp.get("section_template") or [])
        if isinstance(row, dict)
        and _heading_role(str(row.get("name") or "")) != "appendix"
    ]

    def entry(item: dict | str) -> dict | None:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            return None
        name = _clean_heading(str(item.get("name") or ""))[:100]
        if not name or _heading_role(name) == "appendix":
            return None
        declared_role = str(
            item.get("rhetorical_role") or item.get("role") or "").lower().strip()
        role = declared_role if declared_role in allowed_roles else _heading_role(name)
        intent = " ".join(str(item.get("intent") or "").split())[:700]
        return {
            "name": name,
            "rhetorical_role": role,
            "intent": intent or _role_intent(role),
        }

    raw = outline.get("sections") if isinstance(outline, dict) else []
    sections = []
    seen = set()
    for item in raw or []:
        parsed = entry(item)
        if not parsed:
            continue
        key = parsed["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        sections.append(parsed)
        if len(sections) >= 8:
            break

    if not sections:
        for item in template:
            parsed = entry(item)
            if parsed and parsed["name"].lower() not in seen:
                sections.append(parsed)
                seen.add(parsed["name"].lower())

    role_groups = {
        "introduction": {"introduction"},
        "foundations": {"setup", "methods"},
        "contribution": {"results", "proof"},
        "scope": {"discussion", "conclusion"},
    }
    fallbacks = {
        "introduction": ("Introduction", "introduction"),
        "foundations": ("Setup and conventions", "setup"),
        "contribution": ("Results", "results"),
        "scope": ("Discussion and limitations", "discussion"),
    }

    def has_group(group: str) -> bool:
        return any(
            section["rhetorical_role"] in role_groups[group]
            for section in sections
        )

    for group in ("introduction", "foundations", "contribution", "scope"):
        if has_group(group):
            continue
        candidate = next((
            entry(item) for item in template
            if _heading_role(str(item.get("name") or "")) in role_groups[group]
        ), None)
        if candidate is None:
            name, role = fallbacks[group]
            candidate = {
                "name": name,
                "rhetorical_role": role,
                "intent": _role_intent(role),
            }
        if candidate["name"].lower() not in seen:
            sections.append(candidate)
            seen.add(candidate["name"].lower())

    rank = {
        "introduction": 0, "setup": 1, "methods": 2, "other": 3,
        "results": 4, "proof": 5, "discussion": 6, "conclusion": 7,
    }
    sections = [
        section for _index, section in sorted(
            enumerate(sections),
            key=lambda row: (rank.get(row[1]["rhetorical_role"], 3), row[0]),
        )
    ][:8]
    return {
        "title": " ".join(str((outline or {}).get("title") or "").split())[:180],
        "sections": sections,
        "rhetorical_contract": {
            group: sorted(roles) for group, roles in role_groups.items()
        },
    }


def validate_model_blueprint(model_blueprint: dict, deterministic: dict) -> tuple[dict, list[str]]:
    """Validate citation/notation choices before they reach the prose generator."""

    model_blueprint = dict(model_blueprint or {})
    issues = []
    anchors = {str(c.get("id")): c for c in (deterministic.get("citation_anchors") or []) if c.get("id")}
    citations = []
    for item in model_blueprint.get("citation_plan") or []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("paper") or "").replace("arXiv:", "").split("v")[0]
        if pid not in anchors:
            issues.append(f"citation plan dropped unknown paper {pid or '?'}")
            continue
        citations.append({"paper": pid, "use": " ".join(str(item.get("use") or "").split())[:400]})
    if not citations:
        citations = [
            {"paper": pid, "use": str(anchor.get("use_for") or "")}
            for pid, anchor in list(anchors.items())[:10]
        ]
        if anchors:
            issues.append("citation plan replaced with deterministic corpus anchors")

    notation = []
    symbols = set()
    concepts = set()
    ledger_entries = {
        str(row.get("symbol") or ""): row
        for row in (
            (deterministic.get("notation_ledger") or {}).get("symbol_entries")
            or []
        )
        if isinstance(row, dict) and row.get("symbol")
    }
    for item in model_blueprint.get("notation_plan") or []:
        if isinstance(item, str):
            item = {"concept": item, "chosen_symbol": "", "definition": item}
        if not isinstance(item, dict):
            continue
        concept = " ".join(str(item.get("concept") or "").split())[:240]
        symbol = " ".join(str(item.get("chosen_symbol") or item.get("symbol") or "").split())[:80]
        definition = " ".join(str(item.get("definition") or item.get("convention") or "").split())[:500]
        if not concept:
            continue
        concept_key = concept.lower()
        if concept_key in concepts:
            issues.append(f"notation plan dropped duplicate concept {concept}")
            continue
        if symbol and symbol in symbols:
            issues.append(f"notation plan dropped duplicate symbol {symbol}")
            continue
        if symbol and not definition:
            issues.append(f"notation plan dropped undefined symbol {symbol}")
            continue
        concepts.add(concept_key)
        if symbol:
            symbols.add(symbol)
        observed = ledger_entries.get(symbol) or {}
        notation.append({
            "concept": concept,
            "chosen_symbol": symbol,
            "definition": definition,
            "grounding": "corpus-observed" if observed else "writer-decision",
            "source_papers": list(observed.get("papers") or []),
        })
        if symbol and not observed:
            issues.append(
                f"notation symbol {symbol} retained as an explicit writer decision")
    if not notation:
        examples = (deterministic.get("notation_ledger") or {}).get("convention_examples") or []
        notation = [
            {"concept": "corpus convention", "chosen_symbol": "",
             "definition": str(e.get("sentence") or "")}
            for e in examples[:8]
        ]

    known_equations = {}
    for row in deterministic.get("equation_map") or []:
        equation = " ".join(str(row.get("equation") or "").split())
        normalized = _normalise_tex_equation(equation)
        if normalized:
            known_equations[normalized] = {
                "equation": equation,
                "paper": str(row.get("paper") or ""),
            }
    equation_conventions = []
    seen_equations = set()
    for item in model_blueprint.get("equation_conventions") or []:
        if isinstance(item, dict):
            concept = " ".join(str(item.get("concept") or "").split())[:240]
            convention = " ".join(str(
                item.get("chosen_form") or item.get("convention")
                or item.get("equation") or "").split())[:600]
            source_equation = " ".join(str(
                item.get("source_equation") or item.get("equation") or "").split())[:600]
            claimed_paper = str(item.get("paper") or item.get("source_paper") or "")
            required = bool(item.get("required") or item.get("use_in_paper"))
        else:
            concept = ""
            convention = " ".join(str(item).split())[:600]
            source_equation = convention
            claimed_paper = ""
            # A raw corpus example is a style reference, not an instruction to copy it.
            required = False
        if not convention:
            continue
        normalized = _normalise_tex_equation(convention)
        if not normalized or normalized in seen_equations:
            issues.append("equation convention dropped as a duplicate")
            continue
        seen_equations.add(normalized)
        source = (
            known_equations.get(_normalise_tex_equation(source_equation))
            or known_equations.get(normalized)
            or {}
        )
        source_paper = str(source.get("paper") or "")
        if claimed_paper and source_paper and claimed_paper != source_paper:
            issues.append(
                f"equation convention source corrected from {claimed_paper} to {source_paper}")
        if not source_paper:
            issues.append(
                "equation convention retained as a writer decision, not a corpus-derived equation")
        equation_conventions.append({
            "concept": concept,
            "convention": convention,
            "source_equation": str(source.get("equation") or source_equation),
            "source_paper": source_paper,
            "grounding": "exact-corpus-equation" if source_paper else "writer-decision",
            "required": required,
        })

    known_vocabulary = {
        " ".join(str(value).lower().split()): str(value)
        for value in (deterministic.get("vocabulary") or [])
        if str(value).strip()
    }
    vocabulary = []
    for value in model_blueprint.get("vocabulary") or []:
        key = " ".join(str(value).lower().split())
        if not key or key in {item.lower() for item in vocabulary}:
            continue
        if key not in known_vocabulary:
            issues.append(f"vocabulary term dropped because it was not corpus-observed: {value}")
            continue
        vocabulary.append(known_vocabulary[key])
    if not vocabulary:
        vocabulary = list(known_vocabulary.values())[:18]
        if known_vocabulary:
            issues.append("vocabulary plan replaced with deterministic corpus terms")

    deterministic_sections = {
        str(row.get("name") or "").lower(): str(row.get("name") or "")
        for row in (deterministic.get("section_template") or [])
        if isinstance(row, dict) and row.get("name")
    }
    section_arc = []
    for value in model_blueprint.get("section_arc") or []:
        key = " ".join(str(value).lower().split())
        section = deterministic_sections.get(key)
        if section and section not in section_arc:
            section_arc.append(section)
        elif key:
            issues.append(f"section arc dropped non-corpus section {value}")
    if not section_arc:
        section_arc = list(deterministic_sections.values())[:8]

    out = {
        **model_blueprint,
        "section_arc": section_arc,
        "notation_plan": notation,
        "citation_plan": citations,
        "vocabulary": vocabulary[:32],
        "equation_conventions": equation_conventions[:16],
        "writing_contract_enforced": not bool(model_blueprint.get("_fallback")),
        "validation_checks": list(dict.fromkeys(
            [str(v) for v in (model_blueprint.get("validation_checks") or [])]
            + [str(v) for v in (deterministic.get("audit_requirements") or [])]
        )),
    }
    return out, issues


def notation_consistency_report(body_tex: str, blueprint: dict | None) -> dict:
    """Check the selected notation contract against the finished body.

    This deliberately checks only observable syntax: one concept per symbol, use in
    mathematical text, and an explicit definition before the Results section. It does
    not pretend to infer semantic equivalence between two different formulae.
    """

    body = strip_latex_wrappers(body_tex)
    plan = [
        row for row in ((blueprint or {}).get("selected_notation") or [])
        if isinstance(row, dict)
    ]
    section_contract = [
        row for row in ((blueprint or {}).get("selected_section_contract") or [])
        if isinstance(row, dict)
    ]
    result_names = {
        _clean_heading(str(row.get("name") or "")).lower()
        for row in section_contract
        if str(row.get("rhetorical_role") or "") in {"results", "proof"}
    }
    results_match = next((
        match for match in _SECTION_RE.finditer(body)
        if (
            _clean_heading(match.group(1)).lower() in result_names
            or _heading_role(match.group(1)) in {"results", "proof"}
        )
    ), None)
    pre_results = body[:results_match.start()] if results_match else body
    definition_cue = re.compile(
        r"\b(?:let|denote|write|define|defined|means|set|where|throughout|convention)\b",
        re.I,
    )

    def clean_symbol(value: str) -> str:
        return " ".join(str(value or "").strip().strip("$").split())

    math_pattern = re.compile(
        r"(?<!\\)\$(?!\$)(?:\\.|[^$\\])*?(?<!\\)\$"
        r"|\\\((?:\\.|[^\\])*?\\\)"
        r"|\\\[(?:\\.|[^\\])*?\\\]"
        r"|\\begin\{(?:equation|align|alignat|gather|multline|displaymath|"
        r"flalign|eqnarray)\*?\}.*?\\end\{(?:equation|align|alignat|gather|"
        r"multline|displaymath|flalign|eqnarray)\*?\}",
        re.S,
    )

    def math_occurrences(text: str, symbol: str) -> list[int]:
        """Return exact symbol offsets, but only inside observable TeX math."""

        if not symbol:
            return []
        if re.fullmatch(r"[A-Za-z]", symbol):
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])")
        else:
            pattern = re.compile(re.escape(symbol))
        positions = []
        for fragment in math_pattern.finditer(text):
            positions.extend(
                fragment.start() + match.start()
                for match in pattern.finditer(fragment.group(0))
            )
        return positions

    rows = []
    issues = []
    symbol_owner: dict[str, str] = {}
    for item in plan:
        concept = " ".join(str(item.get("concept") or "").split())
        symbol = clean_symbol(item.get("chosen_symbol") or item.get("symbol") or "")
        definition = " ".join(str(item.get("definition") or "").split())
        if not symbol:
            continue
        owner = symbol_owner.get(symbol)
        if owner and owner.lower() != concept.lower():
            issues.append(
                f"notation symbol {symbol} is assigned to both {owner} and {concept}")
            continue
        symbol_owner[symbol] = concept
        all_positions = math_occurrences(body, symbol)
        definition_positions = math_occurrences(pre_results, symbol)
        explicit = False
        for position in definition_positions:
            window = pre_results[max(0, position - 220):position + len(symbol) + 220]
            if definition_cue.search(window):
                explicit = True
                break
        used = bool(all_positions)
        if used and not explicit:
            issues.append(
                f"planned symbol {symbol} ({concept}) is used without an explicit "
                "pre-results definition")
        rows.append({
            "concept": concept,
            "symbol": symbol,
            "definition": definition,
            "occurrences": len(all_positions),
            "used": used,
            "defined_before_results": explicit,
            "first_offset": all_positions[0] if all_positions else None,
        })
    contract_enforced = bool(
        (blueprint or {}).get("writing_contract_enforced", True))
    compact_body = _normalise_tex_equation(body)
    equation_rows = []
    for item in (blueprint or {}).get("selected_equation_conventions") or []:
        if isinstance(item, dict):
            convention = str(
                item.get("convention") or item.get("equation") or "")
            grounding = str(item.get("grounding") or "")
            source_paper = str(item.get("source_paper") or "")
            required = bool(item.get("required"))
        else:
            convention = str(item)
            grounding = ""
            source_paper = ""
            required = False
        compact = _normalise_tex_equation(convention)
        used = bool(compact and compact in compact_body)
        if contract_enforced and required and compact and not used:
            issues.append(
                "selected equation convention is absent from the finished paper: "
                + " ".join(convention.split())[:180])
        equation_rows.append({
            "convention": " ".join(convention.split()),
            "grounding": grounding,
            "source_paper": source_paper,
            "required": required,
            "used": used,
        })

    vocabulary_rows = []
    body_lower = body.lower()
    for term in (blueprint or {}).get("selected_vocabulary") or []:
        value = " ".join(str(term).lower().split())
        if not value:
            continue
        used = bool(re.search(
            rf"(?<![a-z-]){re.escape(value)}(?![a-z-])", body_lower))
        vocabulary_rows.append({"term": value, "used": used})
    used_vocabulary = sum(row["used"] for row in vocabulary_rows)
    if contract_enforced and len(vocabulary_rows) >= 4:
        required = max(2, min(5, (len(vocabulary_rows) + 4) // 5))
        if used_vocabulary < required:
            issues.append(
                f"paper uses only {used_vocabulary}/{len(vocabulary_rows)} selected "
                f"corpus vocabulary terms; at least {required} are required")
    return {
        "schema_version": 1,
        "symbols": rows,
        "equation_conventions": equation_rows,
        "vocabulary": vocabulary_rows,
        "writing_contract_enforced": contract_enforced,
        "issues": issues,
        "ready": not issues,
        "scope": (
            "Syntactic notation-contract audit; semantic equivalence and mathematical "
            "meaning remain the responsibility of proof and referee gates."
        ),
    }


def corpus_style_profile(papers, *, max_papers: int = 12) -> dict:
    """Extract structural habits from the corpus without copying prose.

    This is deliberately about genre and scaffolding: section order, theorem-like
    environments, and common mathematical register. It avoids lifting sentences from
    source papers; the writer should sound like the field, not clone a paper.
    """
    section_counter: Counter = Counter()
    transition_counter: Counter = Counter()
    env_counter: Counter = Counter()
    patterns: Counter = Counter()
    examples: list[list[str]] = []
    for p in list(papers)[:max_papers]:
        text = p.text or ""
        if not text:
            continue
        headings = [_clean_heading(h) for h in _SECTION_RE.findall(text)]
        headings = [
            h for h in headings
            if h and not any(cue in h.lower() for cue in (
                "acknowledg", "references", "bibliography",
            ))
        ]
        if headings:
            examples.append(headings[:8])
            section_counter.update(h.lower() for h in headings[:10])
            roles = [_heading_role(heading) for heading in headings[:8]]
            transition_counter.update(zip(roles, roles[1:]))
        env_counter.update(_ENV_RE.findall(text))
        low = text.lower()
        for pat, name in (
            ("we prove", "states results as theorem/proof"),
            ("we show", "uses compact contribution claims"),
            ("let ", "introduces assumptions before statements"),
            ("it remains to show", "proofs decompose into remaining cases"),
            ("without loss of generality", "uses standard reductions"),
            ("throughout", "states conventions early"),
        ):
            if pat in low:
                patterns[name] += 1

    def lcs_ratio(left: list[str], right: list[str]) -> float:
        if not left or not right:
            return 0.0
        row = [0] * (len(right) + 1)
        for item in left:
            current = [0]
            for index, other in enumerate(right, 1):
                current.append(
                    row[index - 1] + 1
                    if item == other else max(row[index], current[-1])
                )
            row = current
        return row[-1] / max(len(left), len(right))

    representative = []
    if examples:
        scored = []
        for index, arc in enumerate(examples):
            roles = [_heading_role(heading) for heading in arc]
            score = sum(
                lcs_ratio(roles, [_heading_role(heading) for heading in other])
                for other in examples
            )
            # Prefer an actual corpus arc with enough argumentative structure.
            score += 0.15 * len(set(roles) & {
                "introduction", "setup", "methods", "results", "proof",
                "discussion", "conclusion",
            })
            scored.append((score, -index, arc))
        representative = list(max(scored)[2])
    preferred = representative or [
        "Introduction", "Setup", "Results", "Discussion and limitations",
    ]

    return {
        "section_examples": examples[:5],
        "section_counts": dict(section_counter.most_common(20)),
        "preferred_sections": preferred[:8],
        "representative_arc": representative[:8],
        "representative_roles": [
            _heading_role(heading) for heading in representative[:8]
        ],
        "role_transitions": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in transition_counter.most_common(20)
        ],
        "environments": dict(env_counter.most_common()),
        "register": [k for k, _ in patterns.most_common(8)],
    }


def corpus_style_guide(papers) -> str:
    prof = corpus_style_profile(papers)
    blueprint = corpus_writing_blueprint(papers)
    envs = prof.get("environments") or {}
    sections = prof.get("preferred_sections") or ["Introduction", "Setup", "Results", "Discussion", "Conclusion"]
    register = prof.get("register") or [
        "state conventions early",
        "introduce assumptions before statements",
        "separate theorem statements from proofs",
    ]
    env_hint = ", ".join(f"{k} x{v}" for k, v in envs.items()) or "no theorem environments detected"
    examples = "; ".join(" → ".join(seq) for seq in prof.get("section_examples", [])[:3]) or "no section examples detected"
    ledger = blueprint.get("notation_ledger") or {}
    vocab = ", ".join((blueprint.get("vocabulary") or [])[:18]) or "no stable vocabulary extracted"
    macros = ", ".join((ledger.get("macros") or [])[:14]) or "no corpus macros extracted"
    symbols = ", ".join((ledger.get("symbols") or [])[:18]) or "no dominant symbols extracted"
    section_template = "; ".join(
        f"{s.get('name')}: {s.get('role')}" for s in (blueprint.get("section_template") or [])[:7]
    )
    return (
        "Corpus-derived writing guide:\n"
        f"- Preferred section arc: {' → '.join(sections)}.\n"
        f"- Paper template: {section_template}.\n"
        f"- Observed theorem-like environments: {env_hint}.\n"
        f"- Observed section arcs: {examples}.\n"
        f"- Register cues: {'; '.join(register)}.\n"
        f"- Vocabulary/register terms to keep available: {vocab}.\n"
        f"- Notation ledger: macros {macros}; recurring symbols {symbols}.\n"
        "- Use the corpus only for structure, terminology, and mathematical register; do not copy sentences.\n"
        "- State conventions before results, formulate precise assumptions, and distinguish proved statements from conjectural discussion.\n"
        "- Maintain a single notation system; if corpus conventions conflict, explicitly choose one and say so.\n"
    )


def _prose_words(text: str) -> list[str]:
    text = re.sub(r"\\(?:cite|ref|label)\{[^{}]*\}", " ", text or "")
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[$\\{}_^=+*/<>0-9]", " ", text)
    return re.findall(r"[a-z]{3,}", text.lower())


def suspicious_phrase_overlap(body_tex: str, papers, *, words: int = 14) -> list[dict]:
    """Find long exact prose overlap; field vocabulary alone is intentionally ignored."""

    body_words = _prose_words(body_tex)
    if len(body_words) < words:
        return []
    body_ngrams = {tuple(body_words[i:i + words]) for i in range(len(body_words) - words + 1)}
    matches = []
    for paper in papers:
        source = _prose_words((paper.text or paper.abstract or "")[:120_000])
        for i in range(len(source) - words + 1):
            gram = tuple(source[i:i + words])
            if gram in body_ngrams:
                matches.append({"paper": paper.bare_id, "phrase": " ".join(gram)})
                break
        if len(matches) >= 5:
            break
    return matches


def remove_suspicious_overlap_sentences(body_tex: str, papers, *, words: int = 14) -> tuple[str, list[dict]]:
    """Delete prose sentences implicated in a confirmed long source overlap.

    Literal overlap is a mechanical property, so repeated model paraphrase requests are
    the wrong recovery mechanism. The strict ``words``-gram detector first establishes
    that a real overlap exists; only sentences containing a substantial contiguous piece
    of that exact phrase are removed. Section commands are preserved for later audits.
    """

    matches = suspicious_phrase_overlap(body_tex, papers, words=words)
    if not matches:
        return body_tex, []
    phrases = [match["phrase"].split() for match in matches]

    def longest_run(left: list[str], right: list[str]) -> int:
        previous = [0] * (len(right) + 1)
        best = 0
        for lword in left:
            current = [0]
            for index, rword in enumerate(right, 1):
                value = previous[index - 1] + 1 if lword == rword else 0
                current.append(value)
                best = max(best, value)
            previous = current
        return best

    # Keep whitespace as separate entries so deleting a sentence does not weld the
    # surrounding LaTeX together. This intentionally treats sentence-boundary copying
    # as copying: a 14-word match assembled from two copied source sentences removes both.
    parts = re.split(r"(?<=[.!?])([ \t\r\n]+)(?=(?:\\|[A-Z]))", body_tex)
    removed = []
    threshold = max(5, words // 2)
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        sentence_words = _prose_words(sentence)
        if not sentence_words:
            continue
        implicated = [
            phrase for phrase in phrases
            if longest_run(sentence_words, phrase) >= threshold
        ]
        if not implicated:
            continue
        headings = "".join(re.findall(r"\\section\*?\{[^{}]+\}\s*", sentence))
        parts[index] = headings
        removed.append({
            "sentence": " ".join(sentence.split())[:500],
            "overlap": " ".join(implicated[0]),
        })
    cleaned = "".join(parts)
    return cleaned, removed


def citation_contexts(body_tex: str) -> list[dict]:
    """Return every distinct citation use with a stable context identifier."""

    contexts = []
    seen = set()
    paragraphs = re.split(r"\n\s*\n", strip_latex_wrappers(body_tex))
    for paragraph in paragraphs:
        groups = re.findall(r"\\cite\{([^}]*)\}", paragraph)
        if not groups:
            continue
        context = " ".join(paragraph.split())[:1800]
        for group in groups:
            for raw in group.split(","):
                paper = raw.strip().replace("arXiv:", "").split("v")[0]
                key = (paper, context)
                if not paper or key in seen:
                    continue
                seen.add(key)
                digest = hashlib.sha256(f"{paper}\n{context}".encode("utf-8")).hexdigest()[:20]
                contexts.append({"context_id": digest, "paper": paper, "claim_context": context})
    return contexts


def citation_evidence_packet(body_tex: str, papers, *, notes_root: str | Path | None = None) -> dict:
    """Build exact source anchors for the citation-support referee.

    Model notes may suggest which anchor is useful, but every retained anchor is checked
    against the held paper text here before entering the packet.
    """

    contexts = citation_contexts(body_tex)
    wanted = {row["paper"] for row in contexts}
    paper_list = list(papers)
    if not wanted:
        # A repair that deleted every citation still needs authenticated candidates
        # in order to restore one useful background citation.
        wanted = {p.bare_id for p in paper_list[:12]}
    held = {p.bare_id: p for p in paper_list if p.bare_id in wanted}
    note_rows: dict[str, list[dict]] = {}
    if notes_root:
        root = Path(notes_root)
        paths = []
        for sub in (root / "papers", root / "deep"):
            if sub.is_dir():
                paths.extend(sub.glob("*.json"))
        for path in paths:
            try:
                note = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pid = str(note.get("arxiv_id") or "").replace("arXiv:", "").split("v")[0]
            if pid not in held:
                continue
            evidence = note.get("evidence") or note.get("grounded_evidence") or []
            note_rows.setdefault(pid, []).extend(e for e in evidence if isinstance(e, dict))

    sources = []
    for pid, paper in held.items():
        raw_source = "\n".join([paper.abstract or "", paper.text or ""])
        normalised_source = " ".join(raw_source.lower().split())
        candidates = list(note_rows.get(pid, []))
        fallback = (paper.abstract or paper.text or "").strip()[:1200]
        if fallback:
            candidates.append({"supports": "paper-supplied background summary", "anchor": fallback})
        anchors = []
        seen_anchors = set()
        for item in candidates:
            anchor = " ".join(str(item.get("anchor") or "").split())[:900]
            normalised = " ".join(anchor.lower().split())
            if len(normalised) < 12 or normalised not in normalised_source or normalised in seen_anchors:
                continue
            seen_anchors.add(normalised)
            anchors.append({
                "supports": " ".join(str(item.get("supports") or "").split())[:500],
                "source_anchor": anchor,
            })
            if len(anchors) >= 8:
                break
        sources.append({"paper": pid, "title": paper.title, "anchors": anchors})
    return {
        "contexts": contexts,
        "sources": sources,
        "audit_kind": "model entailment judgment with deterministic exact-anchor validation",
    }


def validate_citation_audit(packet: dict, audit: dict, papers) -> list[str]:
    """Validate complete citation coverage and exact, non-invented source anchors."""

    issues = []
    expected = {row.get("context_id"): row for row in (packet.get("contexts") or [])}
    source_packet = {row.get("paper"): row for row in (packet.get("sources") or [])}
    held = {p.bare_id: p for p in papers}
    records = {
        row.get("context_id"): row for row in (audit.get("citations") or [])
        if isinstance(row, dict) and row.get("context_id")
    }
    for context_id, context in expected.items():
        record = records.get(context_id)
        if not record:
            issues.append(f"citation context {context_id} was not audited")
            continue
        paper_id = context.get("paper")
        if str(record.get("paper") or "").replace("arXiv:", "").split("v")[0] != paper_id:
            issues.append(f"citation context {context_id} was assigned to the wrong paper")
            continue
        if record.get("supported") is not True:
            issues.append(f"citation context {context_id} is not supported by {paper_id}")
            continue
        anchor = " ".join(str(record.get("source_anchor") or "").split())
        normalised = " ".join(anchor.lower().split())
        supplied = {
            " ".join(str(item.get("source_anchor") or "").lower().split())
            for item in (source_packet.get(paper_id, {}).get("anchors") or [])
        }
        paper = held.get(paper_id)
        full_source = " ".join(
            "\n".join([getattr(paper, "abstract", "") or "",
                        getattr(paper, "text", "") or ""]).lower().split()) if paper else ""
        if len(normalised) < 12 or normalised not in supplied or normalised not in full_source:
            issues.append(f"citation context {context_id} has an invented or unverified source anchor")
    return issues


def claim_scope_packet(text: str, findings: list, citation_packet: dict | None = None,
                       protocol_evidence: list[dict] | None = None) -> dict:
    """Enumerate prose assertions and the only evidence they may legitimately use."""

    raw = strip_latex_wrappers(text)
    raw = re.sub(r"\\(?:sub)*section\*?\{([^{}]*)\}", r". \1. ", raw)
    raw = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", " ", raw)
    raw = re.sub(r"\\(?:cite|ref|eqref|label)\{[^{}]*\}", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    sentences = []
    seen = set()
    claim_cues = re.compile(
        r"\b(hold|holds|fail|fails|valid|invalid|verify|verifies|verified|prove|proves|proved|require|requires|"
        r"implies|ensures|vanish|vanishes|unique|classif\w*|integrab\w*|novel\w*|"
        r"new|first|published|locate|did not find|found no|every|all|only|"
        r"cannot|never|always|no assumption|no other|strictly|universal)\b",
        re.I,
    )
    escalation_cues = re.compile(
        r"\b(fail|fails|failed|failure|cannot|impossible|necessary|necessity|sufficient|"
        r"only\s+if|if\s+and\s+only\s+if|unique|uniqueness|complete|completeness|"
        r"classif\w*|no[- ]?go|novel|novelty|new\s+family|first\s+(?:proof|result|"
        r"classification)|unknown\s+in\s+the\s+literature|not\s+(?:known|published)|"
        r"outside\s+(?:the\s+)?assumptions|without\s+(?:the\s+)?assumption|"
        r"remove\w*\s+(?:the\s+)?assumption|non[- ]?commutative|universal|exhaustive|"
        r"every|all|any|always|never)\b",
        re.I,
    )
    performative_cues = re.compile(
        r"^(?:let\b|we\s+(?:define|denote|write|work|restrict|focus|consider|adopt|use)|"
        r"under\b.{0,120}\bwe\s+(?:state|write|use)\b|"
        r"(?:the\s+)?(?:notation|symbol|scalar multiplication|coefficient ring)\b.{0,100}"
        r"\b(?:is|are)\s+defined\b|"
        r"the\s+(?:verification|evidence|paper|present)\s+scope\s+(?:is|will be))",
        re.I,
    )
    consequence_cues = re.compile(
        r"\b(?:ensure\w*|guarantee\w*|imply\w*|therefore|hence|establish\w*|"
        r"prove\w*|preserv\w*|yield\w*|demonstrat\w*)\b",
        re.I,
    )
    for sentence in _SENTENCE_RE.split(raw):
        sentence = " ".join(sentence.strip(" .").split())[:1000]
        if len(sentence) < 35 or len(_prose_words(sentence)) < 6 or sentence in seen:
            continue
        seen.add(sentence)
        high_risk = bool(claim_cues.search(sentence) or "$" in sentence or "=" in sentence)
        if not high_risk:
            continue
        digest = hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:20]
        sentences.append({
            "claim_id": digest,
            "sentence": sentence,
            "high_risk": high_risk,
            "escalate": bool(escalation_cues.search(sentence)),
            "performative": bool(
                performative_cues.search(sentence)
                and not consequence_cues.search(sentence)),
        })

    verified = []
    for index, finding in enumerate(findings):
        claim = finding.get("claim") or {}
        fid = str(finding.get("claim_id") or f"finding-{index + 1}")
        compact_claim = {
            key: (str(value)[:1200] if isinstance(value, (str, bytes)) else value)
            for key, value in claim.items()
            if key not in {"files", "code", "manifest", "repos"}
        }
        verified.append({
            "evidence_id": f"finding:{fid}",
            "strength": finding.get("strength", "unverified"),
            "backend": finding.get("backend", "unknown"),
            "encoded_claim": compact_claim,
            "verdict_detail": finding.get("detail", ""),
            "scope_warning": (
                "This evidence establishes only the encoded statement in the backend's "
                "semantics. It does not establish necessity of assumptions, failure outside "
                "those assumptions, uniqueness, completeness, or novelty unless encoded."
            ),
        })
    citations = [
        {"evidence_id": f"citation:{row.get('context_id')}",
         "context_id": row.get("context_id"), "paper": row.get("paper"),
         "claim_context": str(row.get("claim_context") or "")[:700]}
        for row in ((citation_packet or {}).get("contexts") or [])
    ]
    protocols = [
        {
            "evidence_id": str(row.get("evidence_id") or ""),
            "kind": str(row.get("kind") or "documented protocol"),
            "scope": str(row.get("scope") or "")[:1200],
            "queries": list(row.get("queries") or [])[:8],
            "sources": list(row.get("sources") or [])[:8],
            "result_count": int(row.get("result_count") or 0),
        }
        for row in (protocol_evidence or [])
        if isinstance(row, dict) and str(row.get("evidence_id") or "").strip()
    ]
    return {
        "claims": sentences,
        "verified_evidence": verified,
        "citation_evidence": citations,
        "protocol_evidence": protocols,
        "rules": [
            "A verified statement under assumptions does not imply failure when an assumption is removed.",
            "A symbolic check establishes the encoded algebraic expression, not an unstated classification theorem.",
            "Novelty, completeness, uniqueness, no-go, and RG-stability claims require their own evidence.",
            "A retrieval protocol supports only a bounded statement about what its documented searches found, never absolute novelty or nonexistence.",
        ],
    }


def validate_claim_scope_audit(packet: dict, audit: dict) -> list[str]:
    """Require an explicit, valid evidence disposition for every enumerated sentence."""

    issues = []
    expected = {row.get("claim_id"): row for row in (packet.get("claims") or [])}
    finding_rows = {
        row.get("evidence_id"): row for row in (packet.get("verified_evidence") or [])
    }
    findings = set(finding_rows)
    citations = {row.get("evidence_id") for row in (packet.get("citation_evidence") or [])}
    protocols = {row.get("evidence_id") for row in (packet.get("protocol_evidence") or [])}
    rows = {
        row.get("claim_id"): row for row in (audit.get("claims") or [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    qualifiers = (
        "not established here", "not proved here", "remains open", "is unknown",
        "we do not claim", "outside the scope", "requires separate verification",
    )
    for claim_id, claim in expected.items():
        row = rows.get(claim_id)
        if not row:
            issues.append(f"claim-scope sentence {claim_id} was not audited")
            continue
        status = str(row.get("status") or "").lower().replace("-", "_")
        evidence_id = str(row.get("evidence_id") or "")
        if status in {"unsupported", "contradicted"}:
            issues.append(f"claim-scope sentence {claim_id} is {status}")
        elif status == "verified":
            if evidence_id not in findings:
                issues.append(f"claim-scope sentence {claim_id} cites invalid verified evidence")
            else:
                strength = str((finding_rows.get(evidence_id) or {}).get("strength") or "")
                sentence = claim["sentence"].lower()
                if strength == "empirical" and not any(
                        cue in sentence for cue in (
                            "numerical", "empirical", "observed", "experiment")):
                    issues.append(
                        f"claim-scope sentence {claim_id} presents empirical evidence as proof")
                if strength == "executable" and not any(
                        cue in sentence for cue in (
                            "executable evidence", "self-check", "exploratory",
                            "not independently verified")):
                    issues.append(
                        f"claim-scope sentence {claim_id} overstates executable evidence")
        elif status == "source_supported":
            if evidence_id not in citations:
                issues.append(f"claim-scope sentence {claim_id} cites invalid source evidence")
        elif status == "protocol_supported":
            bounded_phrases = (
                "our search", "we searched", "we did not locate", "we found no",
                "documented search", "search protocol", "searched databases",
                "within the search", "under the search", "literature review did not",
            )
            absolute_cues = (
                "is novel", "is new", "no published", "does not exist",
                "has never", "first classification", "first proof", "previously unknown",
            )
            sentence = claim["sentence"].lower()
            if evidence_id not in protocols:
                issues.append(f"claim-scope sentence {claim_id} cites invalid protocol evidence")
            elif not any(phrase in sentence for phrase in bounded_phrases):
                issues.append(f"claim-scope sentence {claim_id} states an unbounded protocol conclusion")
            elif any(phrase in sentence for phrase in absolute_cues):
                issues.append(f"claim-scope sentence {claim_id} overstates a retrieval protocol")
        elif status == "qualified":
            if not any(phrase in claim["sentence"].lower() for phrase in qualifiers):
                issues.append(f"claim-scope sentence {claim_id} is not explicitly qualified")
        elif status == "nonclaim":
            if claim.get("high_risk") and not claim.get("performative"):
                issues.append(f"high-risk claim-scope sentence {claim_id} was labelled nonclaim")
        else:
            issues.append(f"claim-scope sentence {claim_id} has invalid status {status or '?'}")
    return issues


def remove_unsupported_claim_sentences(body_tex: str, packet: dict,
                                       audit: dict) -> tuple[str, list[dict]]:
    """Delete prose sentences the scope referee explicitly rejected.

    An unsupported assertion has no legitimate rewrite target.  Removing it is safer
    and more stable than repeatedly asking a model to paraphrase it.  Only isolated
    prose chunks whose complete set of audited claims is rejected are eligible; chunks
    containing LaTeX environments or any retained claim are left untouched.  All normal
    paper, citation, and claim-scope audits still run after this cleanup.
    """

    rejected = {
        str(row.get("claim_id") or "")
        for row in (audit.get("claims") or [])
        if isinstance(row, dict)
        and str(row.get("status") or "").lower().replace("-", "_")
        in {"unsupported", "contradicted"}
    }
    rejected.discard("")
    if not rejected:
        return body_tex, []

    supplied = {
        str(row.get("claim_id") or ""): str(row.get("sentence") or "")
        for row in (packet.get("claims") or [])
        if isinstance(row, dict)
    }
    parts = re.split(r"(?<=[.!?])(\s+)", body_tex)
    removed = []
    for index in range(0, len(parts), 2):
        chunk = parts[index]
        if not chunk.strip() or re.search(r"\\(?:begin|end)\{", chunk):
            continue
        chunk_packet = claim_scope_packet(chunk, [])
        chunk_ids = {
            str(row.get("claim_id") or "")
            for row in (chunk_packet.get("claims") or [])
            if isinstance(row, dict) and row.get("claim_id")
        }
        matched = chunk_ids & rejected
        if not matched or not chunk_ids.issubset(rejected):
            continue
        headings = "".join(
            re.findall(r"\\(?:sub)*section\*?\{[^{}]+\}\s*", chunk)
        )
        parts[index] = headings
        for claim_id in sorted(matched):
            removed.append({
                "claim_id": claim_id,
                "sentence": supplied.get(claim_id, "")[:1000],
                "status": "deleted_after_unsupported_scope_verdict",
            })
    return "".join(parts), removed


def claims_requiring_escalation(packet: dict, audit: dict) -> list[dict]:
    """Send every substantive assertion to the independent semantic reviewer.

    A fluent fast reviewer can attach a real evidence id to a broader sentence without
    establishing entailment.  Accepted mathematical/result claims therefore still need
    model diversity; only genuinely performative setup text may bypass escalation.
    """

    rows = {
        row.get("claim_id"): row for row in (audit.get("claims") or [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    selected = []
    for claim in packet.get("claims") or []:
        row = rows.get(claim.get("claim_id")) or {}
        status = str(row.get("status") or "").lower().replace("-", "_")
        accepted_nonclaim = status == "nonclaim" and claim.get("performative")
        if not accepted_nonclaim:
            selected.append(claim)
    return selected


def merge_claim_scope_audits(fast: dict, strong: dict,
                             escalated_claim_ids: set[str]) -> dict:
    """Make the strong verdict authoritative for every escalated assertion.

    Missing strong verdicts intentionally remain missing; the deterministic validator
    then blocks the paper instead of silently falling back to the weaker judgment.
    """

    rows = [
        row for row in (fast.get("claims") or [])
        if isinstance(row, dict) and row.get("claim_id") not in escalated_claim_ids
    ]
    rows.extend(
        row for row in (strong.get("claims") or [])
        if isinstance(row, dict) and row.get("claim_id") in escalated_claim_ids
    )
    return {"claims": rows}


def repeated_section_pairs(body_tex: str, *, ngram: int = 5,
                           threshold: float = 0.45) -> list[dict]:
    """Detect sections that substantially repeat one another under different headings."""

    body = strip_latex_wrappers(body_tex)
    matches = list(_SECTION_RE.finditer(body))
    sections = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        words = _prose_words(body[match.end():end])
        if len(words) < 45:
            continue
        grams = {tuple(words[i:i + ngram]) for i in range(len(words) - ngram + 1)}
        sections.append((_clean_heading(match.group(1)), grams))
    repeated = []
    for index, (left_name, left) in enumerate(sections):
        for right_name, right in sections[index + 1:]:
            union = left | right
            similarity = len(left & right) / max(1, len(union))
            if similarity >= threshold:
                repeated.append({
                    "left": left_name, "right": right_name,
                    "five_gram_jaccard": round(similarity, 4),
                })
    return repeated


def audit_body(body_tex: str, papers, findings: list, blueprint: dict | None = None,
               *, min_words: int = 300) -> list[str]:
    """Deterministic paper sanity checks before the model referee pass."""
    body = strip_latex_wrappers(body_tex)
    issues: list[str] = []
    if "\\section" not in body:
        issues.append("missing section structure")
    heading_names = [_clean_heading(h) for h in _SECTION_RE.findall(body)]
    headings = [heading.lower() for heading in heading_names]
    duplicate_headings = sorted({heading for heading in headings if headings.count(heading) > 1})
    if duplicate_headings:
        issues.append("duplicate section headings: " + ", ".join(duplicate_headings[:8]))
    if any("appendix" in heading for heading in headings) or "\\appendix" in body:
        issues.append("draft body contains an appendix; machine certificates are appended separately")
    if findings and len(headings) < 4:
        issues.append("paper needs at least four substantive sections")
    bp = blueprint or {}
    section_contract = [
        row for row in (bp.get("selected_section_contract") or [])
        if isinstance(row, dict) and row.get("name")
    ]
    if findings and section_contract:
        expected = []
        unmatched = set(range(len(heading_names)))
        missing = []
        for row in section_contract:
            name = _clean_heading(str(row.get("name") or "")).lower()
            role = str(
                row.get("rhetorical_role")
                or _heading_role(str(row.get("name") or "")))
            exact = next((
                index for index in sorted(unmatched)
                if headings[index] == name
            ), None)
            equivalent = exact if exact is not None else next((
                index for index in sorted(unmatched)
                if _heading_role(heading_names[index]) == role
            ), None)
            if equivalent is None:
                missing.append(name)
                continue
            unmatched.discard(equivalent)
            expected.append(equivalent)
        if missing:
            issues.append(
                "missing corpus-shaped outline sections: " + ", ".join(missing[:8]))
        if expected != sorted(expected):
            issues.append("corpus-shaped outline sections are out of order")

        roles = {_heading_role(heading) for heading in heading_names}
        role_groups = {
            "introduction": {"introduction"},
            "foundations": {"setup", "methods"},
            "contribution": {"results", "proof"},
            "scope": {"discussion", "conclusion"},
        }
        for group, choices in role_groups.items():
            if not roles & choices:
                issues.append(f"outline lacks a {group} rhetorical section")
    elif findings:
        inferred_roles = {_heading_role(heading) for heading in heading_names}
        for group, choices in (
            ("introduction", {"introduction"}),
            ("foundations", {"setup", "methods"}),
            ("contribution", {"results", "proof"}),
            ("scope", {"discussion", "conclusion"}),
        ):
            if not inferred_roles & choices:
                issues.append(f"missing {group} rhetorical section")
    if findings and len(_prose_words(body)) < min_words:
        issues.append(
            f"paper body is too short ({len(_prose_words(body))} words; minimum {min_words})")
    if papers and "\\cite{" not in body:
        issues.append("no corpus citations in body")
    if re.search(r"(?<!\\)\[(?:\d+[ ,;-]*)+\]", body):
        issues.append("literal numeric citation markers bypass the held-source bibliography")
    if findings and not any(w in body.lower() for w in ("verified", "proved", "checked", "certificate")):
        issues.append("verified findings are not explicitly identified")
    if "\\begin{document}" in body or "\\documentclass" in body:
        issues.append("body contains a document wrapper")
    ledger = bp.get("notation_ledger") or {}
    body_lower = body.lower()
    explicit_setup = bool(
        re.search(r"\\section\*?\{[^{}]*(?:setup|notation|convention|preliminar)[^{}]*\}",
                  body, re.I)
        and re.search(r"\b(?:let|denote|defined as|means)\b", body_lower)
    )
    if (ledger.get("symbols") or ledger.get("convention_examples")) and not (
        explicit_setup or any(
            w in body_lower for w in (
                "throughout", "notation", "convention", "we write", "we denote",
                "normalization", "normalisation",
            )
    )):
        issues.append("paper uses a mathematical corpus but does not state notation/conventions")
    anchors = {c.get("id") for c in (bp.get("citation_anchors") or []) if c.get("id")}
    if anchors and "\\cite{" not in body:
        issues.append("citation anchors exist but no background citations are used")
    if re.search(r"\\cite\{\s*arXiv:[^}]*\}", body) and not papers:
        issues.append("body cites arXiv ids but corpus is empty")
    held = {p.bare_id for p in papers}
    unknown = []
    for group in re.findall(r"\\cite\{([^}]*)\}", body):
        for raw in group.split(","):
            cid = raw.strip().replace("arXiv:", "").split("v")[0]
            if cid and cid not in held:
                unknown.append(cid)
    if unknown:
        issues.append("unknown corpus citations: " + ", ".join(sorted(set(unknown))[:8]))
    labels = re.findall(r"\\label\{([^{}]+)\}", body)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        issues.append("duplicate LaTeX labels: " + ", ".join(duplicate_labels[:8]))
    missing_refs = sorted(set(re.findall(r"\\(?:eqref|ref)\{([^{}]+)\}", body)) - set(labels))
    if missing_refs:
        issues.append("unresolved LaTeX references: " + ", ".join(missing_refs[:8]))
    notation_report = notation_consistency_report(body, bp)
    issues.extend(notation_report.get("issues") or [])
    weak = [f for f in findings if f.get("ok") and f.get("strength") in {"empirical", "executable"}]
    if any(f.get("strength") == "empirical" for f in weak) and "numer" not in body.lower():
        issues.append("empirical findings are not explicitly labelled numerical")
    if any(f.get("strength") == "executable" for f in weak) and not any(
        phrase in body.lower() for phrase in ("executable evidence", "not independently verified", "exploratory")
    ):
        issues.append("executable self-checks are not distinguished from verified results")
    overlaps = suspicious_phrase_overlap(body, papers)
    if overlaps:
        issues.append(
            "possible copied source phrase: " + overlaps[0]["paper"] + " — " + overlaps[0]["phrase"])
    repeated = repeated_section_pairs(body)
    if repeated:
        pair = repeated[0]
        issues.append(
            f"substantial repetition between {pair['left']} and {pair['right']} sections "
            f"({pair['five_gram_jaccard']:.0%} shared five-grams)")
    return issues


def reconcile_referee_audit(result: dict, deterministic_issues: list[str], *,
                            expository: bool = False,
                            held_citation_ids: set[str] | None = None) -> dict:
    """Make static checks authoritative over contradictory model formatting advice.

    Mathematical, evidentiary, citation-purpose, and exposition objections remain
    untouched. Only complaints already decided by a green deterministic structure
    audit are demoted to recorded diagnostics.
    """

    result = dict(result or {})
    structural_audit_failed = any(
        cue in issue.lower()
        for issue in deterministic_issues
        for cue in ("section", "appendix", "document wrapper", "citation markers")
    )
    static_cues = (
        "section arc", "section count", "section heading", "heading name",
        "section order", "missing appendix", "citation format", "bibtex key",
        "missing section", "core section", "five-core-section", "five core section",
        "uses 'setup' instead", "uses 'preliminaries' instead",
        "appendix section", "required body contract",
    )
    semantic_cues = (
        "mathemat", "false", "incorrect", "inaccurate", "evidence", "claim",
        "citation", "unsupported", "overstat", "conject", "speculat", "assumption",
        "proof", "verified", "verification", "novel", "significance", "repetit",
    )
    ignored = []
    retained = []
    held_citation_ids = {
        str(identifier).lower().replace("arxiv:", "").split("v")[0]
        for identifier in (held_citation_ids or set())
    }
    for issue in result.get("issues") or []:
        issue = str(issue)
        low_issue = issue.lower()
        cited_ids = {
            match.lower().replace("arxiv:", "").split("v")[0]
            for match in re.findall(
                r"(?:arxiv:)?(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})v?\d*",
                low_issue, flags=re.I)
        }
        false_citation_existence_complaint = bool(
            cited_ids
            and cited_ids <= held_citation_ids
            and any(cue in low_issue for cue in (
                "placeholder", "non-existent", "nonexistent", "hallucinat",
                "verify if", "not a valid corpus reference",
            ))
        )
        heading_preference = bool(
            "heading" in low_issue
            and any(cue in low_issue for cue in (
                "should be called", "might be better", "would be better",
                "rename", "prefer"))
            and not any(cue in low_issue for cue in ("missing", "order", "duplicate"))
        )
        mode_mismatch = bool(
            expository
            and any(cue in low_issue for cue in (
                "trivial high-school", "trivial high school", "not novel",
                "novel research contribution", "inappropriate for an arxiv",
                "inappropriate for a journal", "lacks novelty",
            ))
            and not any(cue in low_issue for cue in (
                "false", "incorrect", "unsupported", "contradict", "overstat",
            ))
        )
        known_format_only = any(
            cue in low_issue for cue in (
                "citation format", "bibtex key", "arxiv identifier format"))
        purely_static = (
            len(issue) <= 600
            and any(cue in low_issue for cue in static_cues)
            and (known_format_only or not any(cue in low_issue for cue in semantic_cues))
        )
        if (false_citation_existence_complaint or heading_preference or mode_mismatch
                or (not structural_audit_failed and purely_static)):
            ignored.append(issue)
        else:
            retained.append(issue)
    if ignored:
        result["ignored_deterministic_issues"] = ignored
        result["issues"] = retained
        if result.get("verdict") == "revise" and not retained:
            result["verdict"] = "accept"
            result["instructions"] = (
                "deterministic structure gate is authoritative; no semantic issue remains")
    return result


def build_document(title: str, abstract: str, body_tex: str, papers, out_dir: str | Path,
                   *, author: str = "", association: str = "", appendix: str = "") -> Path:
    """Write ``paper.tex`` + ``refs.bib`` into ``out_dir``; return the .tex path.

    ``body_tex`` cites papers by **arXiv id** in ``\\cite{arXiv:ID}`` form; those are
    rewritten to the real BibTeX keys, and any citation to a paper not in the corpus is
    **dropped** rather than left dangling — a citation must point at a source we hold, the
    exact discipline that keeps fabricated references out. ``author``/``association`` set
    the byline; ``appendix`` (verbatim certificates) is inserted before the bibliography."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    unsafe = re.compile(
        r"\\(?:input|include|lstinputlisting|verbatiminput|openin|openout|read|write|"
        r"immediate|special|directlua|luaexec|catcode|csname|usepackage|documentclass|"
        r"bibliography|addbibresource)\b",
        re.I,
    )
    for label, fragment in (
        ("title", title), ("abstract", abstract), ("body", body_tex),
    ):
        match = unsafe.search(str(fragment or ""))
        if match:
            raise ValueError(
                f"unsafe model-authored TeX primitive in {label}: {match.group(0)}")
    for match in re.finditer(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", body_tex or ""):
        candidate = (out_dir / match.group(1)).resolve()
        try:
            candidate.relative_to(out_dir)
        except ValueError as exc:
            raise ValueError("figure path escapes the writeup directory") from exc
        if not candidate.is_file():
            raise ValueError(
                f"paper figure does not exist in the writeup directory: {match.group(1)}")
    bib, keymap = bibtex_from_corpus(papers)
    (out_dir / "refs.bib").write_text(bib, encoding="utf-8")

    def _resolve(m):
        ids = [i.strip().replace("arXiv:", "").split("v")[0] for i in m.group(1).split(",")]
        keys = [keymap[i] for i in ids if i in keymap]
        return f"\\cite{{{','.join(keys)}}}" if keys else ""
    body = re.sub(r"\\cite\{([^}]*)\}", _resolve, body_tex)

    byline = _tex_escape(author) if author else r"spiral\textsuperscript{research}"
    affil = f"\\\\ \\small {_tex_escape(association)}" if association else ""
    body = strip_latex_wrappers(body)
    tex = _TEMPLATE % {"title": _tex_escape_plain(title or "Untitled"),
                       "abstract": _tex_escape_plain(abstract.strip()),
                       "author": byline, "affil": affil,
                       "body": body.strip(), "appendix": appendix.strip()}
    tp = out_dir / "paper.tex"
    tp.write_text(tex, encoding="utf-8")
    return tp


def certificate_appendix(findings: list) -> str:
    """A verbatim reproducibility appendix from the machine-verified findings — the exact
    claims and the backend that certified each, so a referee can re-run them. This is the
    'release exact certificates' half: the paper's rigor is auditable, not asserted."""
    ok = [f for f in findings if f.get("ok")]
    if not ok:
        return ""
    lines = [r"\appendix", r"\section{Machine-verification certificates}",
             "The evidence grade is reported for every successful check: formal and exact "
             "checks establish the encoded statement; computational checks are independently "
             "cross-validated executable evidence; empirical and executable checks are not "
             "presented as proofs. The exact claim and verdict are reproduced verbatim.",
             r"\begin{lstlisting}"]
    for i, f in enumerate(ok, 1):
        c = f.get("claim", {})
        detail = " ".join(
            f"{k}={v}" for k, v in c.items()
            if k not in {
                "note", "files", "code", "manifest", "repos", "datasets",
                "analysis_plan", "alignment", "_result_summary",
            }
        )
        lines.append(f"[{i}] {c.get('note','claim')}")
        lines.append(f"    {detail}")
        lines.append(
            f"    -> [{f.get('strength','ungraded')}] {f.get('backend','?')}: {f.get('detail','')}")
        manifest_path = Path(str(c.get("manifest") or ""))
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                data = manifest.get("data_evidence") or {}
                if data and not data.get("not_applicable"):
                    lines.append(
                        f"    analysis-plan-sha256={data.get('plan_hash', '')}")
                    lines.append(
                        "    data-provenance="
                        f"{'complete' if data.get('provenance_complete') else 'incomplete'}; "
                        f"confirmatory-ready={bool(data.get('confirmatory_ready'))}")
                    for record in data.get("records") or []:
                        lines.append(
                            "    dataset "
                            f"{record.get('source')}:{record.get('dataset_id')} "
                            f"version={record.get('version') or 'unknown'} "
                            f"doi={record.get('doi') or 'none'} "
                            f"license={record.get('license') or 'unresolved'} "
                            f"files={record.get('file_count', len(record.get('files') or []))} "
                            f"bytes={record.get('bytes', 0)}")
                    aggregate = (
                        (data.get("result_summary") or {}).get("summary") or {})
                    if aggregate:
                        lines.append(
                            "    aggregate-result="
                            + json.dumps(aggregate, ensure_ascii=False, sort_keys=True))
            except Exception:
                pass
    lines.append(r"\end{lstlisting}")
    return "\n".join(lines)


def compile_pdf(tex_path: str | Path) -> tuple[Path | None, str]:
    """Compile to PDF. Returns ``(pdf_path_or_None, log)`` — the log carries LaTeX errors
    so the caller can feed them back for a repair pass (the compile step is a *gate*: a
    paper that does not compile is not done). ``(None, "")`` iff no TeX toolchain exists."""
    tex_path = Path(tex_path)
    d, stem = tex_path.parent, tex_path.stem
    generated = ["pdf", "aux", "bbl", "blg", "log", "out", "fdb_latexmk", "fls"]
    for suffix in generated:
        try:
            (d / f"{stem}.{suffix}").unlink(missing_ok=True)
        except OSError:
            pass
    if shutil.which("latexmk"):
        seq = [[
            "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
            "-pdflatex=pdflatex -no-shell-escape %O %S", tex_path.name,
        ]]
    elif shutil.which("tectonic"):
        seq = [[
            "tectonic", "--keep-logs", "--keep-intermediates",
            "--synctex", "0", tex_path.name,
        ]]
    elif shutil.which("pdflatex"):
        pdflatex = [
            "pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
            "-halt-on-error", tex_path.name,
        ]
        seq = [pdflatex]
        if "\\bibliography{" in tex_path.read_text(encoding="utf-8", errors="ignore"):
            if not shutil.which("bibtex"):
                return None, "BibTeX is required but not installed"
            seq.append(["bibtex", stem])
        seq += [pdflatex, pdflatex]
    else:
        return None, ""
    logs = []
    failed = False
    for cmd in seq:
        try:
            r = subprocess.run(
                cmd, cwd=d, capture_output=True, text=True, timeout=180,
                stdin=subprocess.DEVNULL,
            )
            logs.append(r.stdout + r.stderr)
            if r.returncode != 0:
                failed = True
                break
        except Exception as e:
            logs.append(str(e))
            failed = True
            break
    log = "\n".join(logs)
    pdf = d / f"{stem}.pdf"
    final_log_path = d / f"{stem}.log"
    final_log = final_log_path.read_text(encoding="utf-8", errors="ignore") \
        if final_log_path.is_file() else ""
    unresolved = []
    for pattern, message in (
        (r"Citation `[^']+' .* undefined", "undefined bibliography citation"),
        (r"Reference `[^']+' .* undefined", "undefined cross-reference"),
        (r"There were undefined references", "undefined references remain"),
        (r"No file .*\.bbl", "bibliography was not generated"),
    ):
        if re.search(pattern, final_log, re.I):
            unresolved.append(message)
    tex_source = tex_path.read_text(encoding="utf-8", errors="ignore")
    if ("\\cite{" in tex_source
            and "\\begin{thebibliography}" not in tex_source
            and not (d / f"{stem}.bbl").is_file()):
        # an inline thebibliography resolves \cite without any .bbl
        unresolved.append("cited document has no generated bibliography")
    errors = []
    if failed:
        errors.append("LaTeX command failed. Log tail:\n" + log[-1800:])
    errors.extend(dict.fromkeys(unresolved))
    if not pdf.is_file() and not errors:
        errors.append("LaTeX toolchain produced no fresh PDF. Log tail:\n" + log[-1600:])
    if errors:
        pdf.unlink(missing_ok=True)
        return None, "\n".join(errors)[:3000]
    return pdf, ""


def audit_pdf_layout(
    pdf_path: str | Path, *, render_dir: str | Path | None = None,
    max_pages: int = 16,
) -> dict:
    """Collect deterministic publication-layout evidence from the finished PDF.

    This checks page geometry, extractable content, severe overfull boxes, rasterized
    blank pages, and content touching a physical page edge. It is intentionally not a
    taste judge; a local vision referee may add that separate layer.
    """

    pdf = Path(pdf_path).resolve()
    issues: list[str] = []
    warnings: list[str] = []
    rendered: list[str] = []
    pages = 0
    page_rows = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf))
        pages = len(reader.pages)
        if pages < 1:
            issues.append("PDF contains no pages")
        for index, page in enumerate(reader.pages, 1):
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            text = (page.extract_text() or "").strip()
            if width < 300 or height < 300:
                issues.append(
                    f"page {index} has implausible dimensions {width:.0f}x{height:.0f} pt")
            if len(text) < 20:
                issues.append(f"page {index} has no substantive extractable content")
            page_rows.append({
                "page": index, "width_pt": round(width, 2),
                "height_pt": round(height, 2), "text_characters": len(text),
            })
    except ModuleNotFoundError:
        pdfinfo = shutil.which("pdfinfo")
        if pdfinfo:
            probe = subprocess.run(
                [pdfinfo, str(pdf)], capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=30)
            match = re.search(r"^Pages:\s+(\d+)", probe.stdout, re.M)
            pages = int(match.group(1)) if match else 0
            if probe.returncode or pages < 1:
                issues.append(
                    "PDF structure could not be decoded by pdfinfo: "
                    + (probe.stderr.strip() or "no pages reported")[:400])
            else:
                warnings.append(
                    "pypdf unavailable; page geometry/text extraction used pdfinfo/raster fallback")
        else:
            header = pdf.read_bytes()[:8] if pdf.is_file() else b""
            if not header.startswith(b"%PDF-"):
                issues.append("PDF container header is invalid")
            else:
                pages = 1
                warnings.append(
                    "pypdf/pdfinfo unavailable; only PDF header and raster evidence were checked")
    except Exception as exc:
        issues.append(f"PDF structure could not be decoded: {type(exc).__name__}: {exc}")

    log_path = pdf.with_suffix(".log")
    log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    overfull = [
        float(value) for value in re.findall(
            r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    ]
    severe_overfull = [value for value in overfull if value > 6.0]
    if severe_overfull:
        issues.append(
            f"{len(severe_overfull)} severe overfull line(s); widest "
            f"{max(severe_overfull):.1f} pt")
    elif overfull:
        warnings.append(
            f"{len(overfull)} minor overfull line(s); widest {max(overfull):.1f} pt")

    target = Path(render_dir).resolve() if render_dir else pdf.parent / "rendered-pages"
    rasterizer = shutil.which("pdftoppm")
    if rasterizer and pages:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        prefix = target / "page"
        result = subprocess.run(
            [
                rasterizer, "-png", "-r", "120", "-f", "1",
                "-l", str(min(max_pages, pages)), str(pdf), str(prefix),
            ],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=240,
        )
        images = sorted(target.glob("page-*.png"))
        if result.returncode or not images:
            warnings.append(
                "PDF rasterization failed: "
                + (result.stderr.strip() or "no page images produced")[:500])
        else:
            try:
                from PIL import Image

                for index, image_path in enumerate(images, 1):
                    with Image.open(image_path).convert("L") as image:
                        mask = image.point(lambda pixel: 255 if pixel < 245 else 0)
                        bbox = mask.getbbox()
                        dark = sum(mask.histogram()[1:])
                        ratio = dark / max(1, image.width * image.height)
                        if not bbox or ratio < 0.00035:
                            issues.append(f"rendered page {index} is effectively blank")
                        elif (
                            bbox[0] <= image.width * 0.004
                            or bbox[1] <= image.height * 0.004
                            or bbox[2] >= image.width * 0.996
                            or bbox[3] >= image.height * 0.996
                        ):
                            issues.append(
                                f"rendered page {index} has content touching a page edge")
                    rendered.append(str(image_path))
            except Exception as exc:
                warnings.append(
                    f"raster pages could not be inspected: {type(exc).__name__}: {exc}")
    else:
        warnings.append(
            "no PDF rasterizer available; visual page evidence was not generated")

    if pages > max_pages:
        warnings.append(
            f"raster inspection sampled the first {max_pages}/{pages} pages")
    return {
        "schema_version": 1,
        "pdf": str(pdf),
        "pages": pages,
        "page_geometry": page_rows,
        "overfull_boxes_pt": overfull,
        "rendered_pages": rendered,
        "issues": issues,
        "warnings": warnings,
        "ready": not issues,
        "scope": (
            "Deterministic PDF structure and raster-layout audit; mathematical content "
            "and aesthetic judgment are handled by separate evidence/referee gates."
        ),
    }
