"""When is a corpus big enough to stop reading and start mining?

A real run answered this badly five times running. Its supervisor chose "deepen corpus
before verification" in every one of five rounds — including the round after its own
citation graph reported saturation — and finished with 279 papers, 1.94M tokens and zero
findings. Roughly four of those five hours went into re-reading a literature it had
already exhausted, because nothing measured exhaustion and made the loop act on it.

The question is not a matter of taste, and it is not new. Sampling references out of a
literature is drawing from a population of unknown size, which is the species-sampling
problem, and it has estimators with theory behind them:

**Good–Turing coverage.** With ``f1`` items seen exactly once in ``N`` draws, the
probability that the *next* draw is something unseen is estimated by ``f1/N`` — so
coverage is ``C = 1 - f1/N``. This is the stopping rule stated directly: at C = 0.97,
97% of what the next paper cites is already in hand, and fetching it buys almost
nothing. Good (1953), following Turing's wartime work.

**Chao1 richness.** ``S = S_obs + f1(f1-1) / (2(f2+1))`` estimates how large the
reachable population is, from the singleton and doubleton counts. A lower bound, and the
bias-corrected form is used here because it stays defined when no item is seen twice.
Completeness is then ``S_obs / S``.

**Marginal gain.** The empirical derivative of the accumulation curve: distinct new
items contributed per document over the most recent window. Robust, assumption-free, and
the one that notices when a literature has genuinely closed.

Coverage says "the next document is redundant". Completeness says "the field is mostly
seen". Marginal gain says "the last few documents taught us nothing". Requiring all
three keeps a lucky plateau from being mistaken for closure.

Domain-agnostic on purpose: the input is documents-as-bags-of-items. The items are
references for one reading and concept terms for another, and the module never needs to
know which field it is looking at.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Sufficiency:
    """What the sample says about the population it was drawn from."""
    observed: int                  # distinct items seen
    total_draws: int               # item occurrences, counting repeats
    singletons: int                # f1 — seen exactly once
    doubletons: int                # f2 — seen exactly twice
    coverage: float                # Good–Turing: P(next draw is already held)
    estimated_total: float         # Chao1 richness
    completeness: float            # observed / estimated_total
    marginal_gain: float           # new distinct items per document, recent window
    enough: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "observed": self.observed, "total_draws": self.total_draws,
            "singletons": self.singletons, "doubletons": self.doubletons,
            "coverage": round(self.coverage, 4),
            "estimated_total": round(self.estimated_total, 1),
            "completeness": round(self.completeness, 4),
            "marginal_gain": round(self.marginal_gain, 4),
            "enough": self.enough, "reason": self.reason,
        }

    def sentence(self) -> str:
        return (f"coverage {self.coverage:.1%} · completeness {self.completeness:.1%} "
                f"· {self.marginal_gain:.2f} new/doc · {self.observed} of "
                f"~{self.estimated_total:.0f}")


def good_turing_coverage(counts: Counter) -> float:
    """P(the next draw is an item already seen), from the singleton rate.

    Turing's estimate of the unseen mass is f1/N, so coverage is its complement. With
    no draws at all, coverage is 0 — an empty sample has seen nothing, and returning 1
    would read as "done" to every caller."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    f1 = sum(1 for c in counts.values() if c == 1)
    return max(0.0, min(1.0, 1.0 - f1 / total))


def chao1(counts: Counter) -> float:
    """Estimated richness of the population, bias-corrected.

    S_obs + f1(f1-1) / (2(f2+1)). The +1 keeps it finite when nothing has been seen
    exactly twice, which is the common case early in a run — the uncorrected form
    divides by zero there."""
    observed = len(counts)
    if not observed:
        return 0.0
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    return observed + (f1 * (f1 - 1)) / (2 * (f2 + 1))


def marginal_gain(documents: list[list[str]], window: int = 10) -> float:
    """Distinct items per document that the last ``window`` documents ADDED.

    The empirical derivative of the accumulation curve. Assumption-free, unlike a Heaps
    fit, and it is the measurement that actually notices closure: when the last ten
    papers between them introduce two new concepts, the literature has stopped moving.
    """
    if not documents:
        return 0.0
    window = max(1, min(window, len(documents)))
    seen_before: set[str] = set()
    for doc in documents[:-window]:
        seen_before.update(doc)
    fresh: set[str] = set()
    for doc in documents[-window:]:
        fresh.update(item for item in doc if item not in seen_before)
    return len(fresh) / window


def assess(documents: list[list[str]], *, min_coverage: float = 0.95,
           min_completeness: float = 0.80, max_marginal: float = 1.0,
           min_documents: int = 20, window: int = 10) -> Sufficiency:
    """Is this sample enough, or should more be gathered?

    All three criteria must hold, plus a floor on the number of documents: with five
    papers every estimator is noise, and a run that stops there has not sampled a
    literature, it has sampled an accident.
    """
    counts: Counter = Counter()
    for doc in documents or []:
        counts.update(set(doc or []))          # presence per document, not raw frequency
    coverage = good_turing_coverage(counts)
    estimated = chao1(counts)
    observed = len(counts)
    completeness = (observed / estimated) if estimated > 0 else 0.0
    gain = marginal_gain(documents or [], window=window)

    n_docs = len(documents or [])
    checks = [
        (n_docs >= min_documents, f"only {n_docs} documents (need {min_documents})"),
        (coverage >= min_coverage,
         f"coverage {coverage:.1%} below {min_coverage:.0%} — the next document is "
         "still likely to bring something unseen"),
        (completeness >= min_completeness,
         f"completeness {completeness:.1%} below {min_completeness:.0%} — Chao1 puts "
         f"the reachable set near {estimated:.0f}, and {observed} is seen"),
        (gain <= max_marginal,
         f"the last {min(window, n_docs)} documents still add {gain:.2f} new items "
         f"each (limit {max_marginal})"),
    ]
    failed = [why for ok, why in checks if not ok]
    return Sufficiency(
        observed=observed, total_draws=sum(counts.values()),
        singletons=sum(1 for c in counts.values() if c == 1),
        doubletons=sum(1 for c in counts.values() if c == 2),
        coverage=coverage, estimated_total=estimated, completeness=completeness,
        marginal_gain=gain, enough=not failed,
        reason=("the sample is representative: " +
                f"coverage {coverage:.1%}, completeness {completeness:.1%}, "
                f"{gain:.2f} new items per recent document"
                if not failed else "; ".join(failed)),
    )


# ── turning a corpus into the two samples worth measuring ────────────────────
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has",
    "have", "had", "not", "but", "all", "can", "which", "these", "those", "than",
    "then", "there", "their", "them", "they", "its", "it's", "our", "one", "two",
    "also", "such", "been", "more", "most", "only", "other", "some", "into", "over",
    "under", "between", "where", "when", "while", "each", "both", "any", "may",
    "using", "used", "use", "given", "show", "shown", "shows", "we", "here", "thus",
    "however", "therefore", "paper", "results", "result", "study", "new", "based",
    "well", "case", "cases", "first", "second", "three", "section", "figure", "table",
    "appendix", "et", "al", "eq", "eqs", "ref", "refs", "see", "note", "notes",
}


# Markup, not language. Measured on a real 279-paper physics corpus, these and the
# citation keys below were most of the vocabulary: the ten commonest "concepts" included
# def, begin, end, newcommand, documentclass, label and cite, and 6551 of 11332 terms
# were singletons — overwhelmingly \cite keys (chapcft, damtp-r-94-7), author surnames
# and stray non-English. Good–Turing counts singletons, so that noise WAS the verdict:
# coverage read 85% and marginal gain 36 terms per document, and a corpus of copies of
# one paper would still have looked wide open.
_LATEX_STOP = {
    "def", "begin", "end", "newcommand", "renewcommand", "documentclass", "usepackage",
    "label", "cite", "citep", "citet", "ref", "eqref", "cal", "mathcal", "mathbb",
    "mathrm", "mathfrak", "textrm", "textbf", "textit", "emph", "frac", "partial",
    "left", "right", "big", "bigg", "quad", "qquad", "hspace", "vspace", "centering",
    "center", "item", "itemize", "enumerate", "section", "subsection", "subsubsection",
    "caption", "includegraphics", "bibliography", "bibitem", "bibliographystyle",
    "abstract", "document", "article", "preprint", "titlepage", "maketitle", "author",
    "title", "date", "address", "email", "footnote", "acknowledgments",
    "acknowledgements", "appendix", "array", "align", "aligned", "equation", "eqnarray",
    "matrix", "pmatrix", "cases", "text", "rm", "it", "bf", "tt", "sf", "phantom",
    "nonumber", "newline", "linebreak", "pagebreak", "clearpage", "newpage", "input",
    "include", "sqrt", "sum", "int", "prod", "lim", "log", "exp", "sin", "cos", "tan",
    "university", "department", "institute", "physics", "arxiv", "eprint", "doi",
}
# Academic prose. This list exists because a frequency band cannot replace it, which was
# measured rather than assumed: across 278 real papers, `coset` appeared in 13% of
# documents and `corresponding` in 18%, `wess-zumino-witten` in 6% and `introduction` in
# 7%. Signal and scaffolding occupy the same range, so any ceiling that removes the
# second removes the first. Without this the top "discoveries" were
# "introduction ←→ observer" and "coset models ←→ corresponding".
_PROSE_STOP = {
    "introduction", "conclusion", "conclusions", "discussion", "summary", "overview",
    "background", "outline", "remarks", "remark", "preliminaries", "acknowledgment",
    "references", "contents", "corresponding", "corresponds", "correspond", "following",
    "follows", "follow", "followed", "works", "work", "working", "consider",
    "considered", "considering", "obtain", "obtained", "obtaining", "provide",
    "provided", "provides", "present", "presented", "presents", "describe",
    "described", "describes", "discuss", "discussed", "introduce", "introduced",
    "define", "defined", "defines", "denote", "denoted", "denotes", "assume",
    "assumed", "assumes", "suppose", "recall", "note", "noted", "observe", "observed",
    "find", "found", "finds", "known", "similar", "similarly", "analogous", "various",
    "several", "certain", "particular", "specific", "general", "generally", "simple",
    "simply", "explicit", "explicitly", "implicit", "direct", "directly", "natural",
    "naturally", "possible", "possibly", "necessary", "sufficient", "important",
    "interesting", "useful", "relevant", "appropriate", "suitable", "standard",
    "usual", "common", "typical", "typically", "essentially", "effectively",
    "respectively", "namely", "moreover", "furthermore", "nevertheless", "hence",
    "consequently", "accordingly", "indeed", "clearly", "obviously", "actually",
    "really", "rather", "quite", "very", "much", "many", "few", "less", "least",
    "greater", "smaller", "larger", "small", "large", "long", "short", "high", "low",
    "good", "better", "best", "full", "whole", "part", "parts", "way", "ways",
    "form", "forms", "type", "types", "kind", "kinds", "sort", "class", "classes",
    "term", "terms", "expression", "expressions", "relation", "relations", "property",
    "properties", "feature", "features", "aspect", "aspects", "detail", "details",
    "context", "framework", "approach", "method", "methods", "technique", "techniques",
    "procedure", "analysis", "discussion", "argument", "arguments", "proof", "proofs",
    "theorem", "lemma", "corollary", "proposition", "definition", "example",
    "examples", "problem", "problems", "question", "questions", "answer", "issue",
    "point", "points", "line", "lines", "sections", "chapter", "value", "values",
    "number", "numbers", "amount", "level", "levels", "step", "steps", "process",
    "system", "systems", "structure", "structures", "condition", "conditions",
    "assumption", "constraint", "requirement", "consequence", "implication",
    "application", "applications", "purpose", "reason", "reasons", "sense",
    "respect", "regard", "view", "list", "set", "sets", "group",
}
_LATEX_CMD = re.compile(r"\\[A-Za-z@]+\s*")
_MATH = re.compile(r"\$[^$]{0,400}\$|\\\[[\s\S]{0,800}?\\\]|"
                   r"\\begin\{(?:equation|align|eqnarray|array|matrix)\*?\}"
                   r"[\s\S]{0,2000}?\\end\{(?:equation|align|eqnarray|array|matrix)\*?\}")
_BRACED = re.compile(r"\{[^{}]{0,200}\}")
_HAS_DIGIT = re.compile(r"\d")


def strip_markup(text: str) -> str:
    """Remove the parts of a TeX source that are notation rather than language.

    Math, commands and their braced arguments go; the prose between them stays. Crude
    on purpose — this feeds a word counter, not a compiler."""
    out = _MATH.sub(" ", text or "")
    out = _LATEX_CMD.sub(" ", out)
    out = _BRACED.sub(" ", out)
    return out


def concept_terms(text: str, *, min_len: int = 4, top: int = 0) -> list[str]:
    """Content terms in a document, lowercased and de-stopworded.

    Deliberately plain: a term is a word. Domain vocabulary is whatever survives the
    stoplist, so the same function serves a physics abstract and a molecular-biology
    one without being told which it has.

    Terms carrying a digit are dropped, which is what removes ``damtp-r-94-7`` and the
    rest of the citation-key vocabulary without needing to recognise citation keys.
    """
    words = [w.lower() for w in _WORD.findall(strip_markup(text))]
    terms = [w for w in words
             if len(w) >= min_len and w not in _STOP and w not in _LATEX_STOP
             and w not in _PROSE_STOP and not _HAS_DIGIT.search(w)]
    if top:
        return [w for w, _ in Counter(terms).most_common(top)]
    return terms


# Identifiers a bibliography actually contains. The first version required an explicit
# ``arXiv:`` or ``doi:`` prefix and so found reference lists in 24 of 279 real papers —
# hep-th/9802109 is how this literature cites, and a sample that thin is not a
# measurement of anything.
_REFERENCE = re.compile(
    r"\b(?:arXiv:\s*)?("
    r"(?:hep-th|hep-ph|hep-ex|hep-lat|gr-qc|math-ph|cond-mat|nucl-th|nucl-ex|"
    r"astro-ph|quant-ph|math|cs|q-bio|stat)(?:\.[A-Za-z]{2})?/\d{7}"     # old style
    r"|\d{4}\.\d{4,5}"                                                    # new style
    r")(?:v\d+)?\b", re.I)
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:\w]+\b")


def references_in(text: str) -> list[str]:
    """Work identifiers cited by a document, lowercased and de-versioned.

    References are the clean sample: unlike prose they are identifiers, so a singleton
    here is a genuinely once-cited work rather than a LaTeX artefact."""
    body = text or ""
    found = {m.lower() for m in _REFERENCE.findall(body)}
    found |= {m.lower().rstrip(".,;") for m in _DOI.findall(body)}
    return sorted(found)


def bigrams(terms: list[str]) -> list[str]:
    """Adjacent pairs, which is where most technical concepts actually live —
    "sigma model", "anomaly matching", "single cell" are each one idea, not two."""
    return [f"{a} {b}" for a, b in zip(terms, terms[1:])]


@dataclass
class CorpusReadiness:
    """Both readings of the same corpus, and the phase they imply."""
    references: Sufficiency
    concepts: Sufficiency
    phase: str = "gather"                 # gather | mine
    detail: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"phase": self.phase, "detail": self.detail,
                "references": self.references.as_dict(),
                "concepts": self.concepts.as_dict()}


def readiness(reference_lists: list[list[str]], concept_lists: list[list[str]],
              **kw) -> CorpusReadiness:
    """Measure both samples and decide the phase.

    Two readings, because they fail differently. A corpus can close its citation graph
    while its *vocabulary* is still opening — that is a field being read narrowly, and
    more of the same papers will not fix it. Mining starts only when both have settled;
    until then the honest instruction is to keep gathering, and to say which of the two
    is short.
    """
    refs = assess(reference_lists, **kw)
    concepts = assess(concept_lists, **kw)
    if refs.enough and concepts.enough:
        return CorpusReadiness(
            refs, concepts, phase="mine",
            detail=("the literature is sampled to closure — further fetching is "
                    f"redundant. references: {refs.sentence()} · concepts: "
                    f"{concepts.sentence()}"))
    short = []
    if not refs.enough:
        short.append(f"references: {refs.reason}")
    if not concepts.enough:
        short.append(f"concepts: {concepts.reason}")
    return CorpusReadiness(refs, concepts, phase="gather",
                           detail="; ".join(short), notes=short)
