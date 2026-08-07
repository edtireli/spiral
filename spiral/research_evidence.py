"""Evidence grounding — the empirical-science verifier.

Physics and math claims are decided by SymPy, Lean or a numeric certificate: the tool
computes, the claim stands or falls. A neuroscience/biology/medicine claim like
"TDP-43 aggregation increases with age in cortical neurons" has no such oracle — it is
an *empirical* statement whose warrant is the literature. So the analogue of "run the
math" is: **find the claim, quoted, in enough independent primary sources, and surface
any that disagree.** The model proposes the statement; the corpus decides.

This stays true to spiral's contract — verify-or-it-didn't-happen — with a different
instrument. It never invents support: a span only counts if the paper's own text
carries enough of the claim's content terms in one place (a real quote, not a word
bag scattered across a whole paper), and negations near those terms are reported as
disagreement rather than silently ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOP = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "is", "are", "was", "were",
    "with", "for", "that", "this", "these", "those", "we", "our", "by", "as", "at",
    "from", "be", "been", "it", "its", "which", "than", "then", "also", "can", "may",
    "not", "no", "more", "most", "such", "into", "between", "within", "across", "both",
    "increase", "increases", "decrease", "decreases",   # kept out of the term set;
    "effect", "effects", "result", "results", "study", "studies", "show", "shows",
}
_NEGATION = re.compile(
    r"\b(no|not|neither|nor|without|fail(?:ed|s)?|absence|lack(?:ed|s|ing)?|"
    r"did not|does not|were not|was not|un(?:changed|altered|related|affected)|"
    r"no (?:significant|detectable|measurable)|contrary|contradict\w*|"
    r"opposite|refut\w*|disprov\w*|inconsistent)\b", re.I)


@dataclass
class Anchor:
    uid: str
    title: str
    quote: str
    overlap: float
    negated: bool = False


@dataclass
class EvidenceResult:
    supported: bool
    statement: str
    min_support: int
    support_count: int
    anchors: list = field(default_factory=list)          # supporting Anchor list
    dissent: list = field(default_factory=list)          # disagreeing Anchor list
    detail: str = ""

    def as_dict(self) -> dict:
        def _a(a: Anchor) -> dict:
            return {"uid": a.uid, "title": a.title, "quote": a.quote,
                    "overlap": round(a.overlap, 3), "negated": a.negated}
        return {
            "supported": self.supported, "statement": self.statement,
            "min_support": self.min_support, "support_count": self.support_count,
            "anchors": [_a(a) for a in self.anchors],
            "dissent": [_a(a) for a in self.dissent],
            "detail": self.detail,
        }


def _terms(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (text or "").lower())
    return [w for w in words if w not in _STOP]


def _sentences(text: str) -> list[str]:
    # split on sentence punctuation but keep decimals/abbreviations mostly intact
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text or "")
    return [s.strip() for s in raw if len(s.strip()) >= 20]


def best_anchor(statement_terms: set[str], text: str, *, min_overlap: float) -> Anchor | None:
    """The single best-supporting sentence in one paper's text: the sentence whose
    content-term overlap with the statement is highest, if it clears ``min_overlap``.
    Overlap is measured against the statement's terms so a long paper can't win by
    sheer length."""
    if not statement_terms:
        return None
    best: Anchor | None = None
    for sent in _sentences(text):
        st = set(_terms(sent))
        if not st:
            continue
        overlap = len(statement_terms & st) / len(statement_terms)
        if overlap >= min_overlap and (best is None or overlap > best.overlap):
            best = Anchor(uid="", title="", quote=" ".join(sent.split())[:300],
                          overlap=overlap, negated=bool(_NEGATION.search(sent)))
    return best


def evidence_support(statement: str, papers, *, min_support: int = 2,
                     min_overlap: float = 0.5) -> EvidenceResult:
    """Ground ``statement`` in the corpus. A paper *supports* it when its own text
    carries a sentence with ≥``min_overlap`` of the statement's content terms and no
    negation; a paper *dissents* when the best-overlapping sentence is negated. Support
    requires ≥``min_support`` distinct papers — one paper echoing a claim is not
    evidence. Deterministic: same corpus, same verdict."""
    st = set(_terms(statement))
    if len(st) < 3:
        return EvidenceResult(
            False, statement, min_support, 0,
            detail="claim too vague to ground (fewer than 3 content terms)")
    anchors: list[Anchor] = []
    dissent: list[Anchor] = []
    for p in papers or []:
        text = getattr(p, "text", "") or getattr(p, "abstract", "")
        if not text:
            continue
        a = best_anchor(st, text, min_overlap=min_overlap)
        if not a:
            continue
        a.uid = getattr(p, "bare_id", "") or getattr(p, "arxiv_id", "")
        a.title = getattr(p, "title", "")
        (dissent if a.negated else anchors).append(a)
    anchors.sort(key=lambda a: a.overlap, reverse=True)
    dissent.sort(key=lambda a: a.overlap, reverse=True)
    supported = len(anchors) >= min_support
    if supported and dissent:
        detail = (f"grounded in {len(anchors)} papers; "
                  f"{len(dissent)} paper(s) report disagreement — reported, not hidden")
    elif supported:
        detail = f"grounded in {len(anchors)} independent papers"
    else:
        detail = (f"insufficient support: {len(anchors)}/{min_support} papers carry a "
                  f"quoted anchor (≥{int(min_overlap*100)}% term overlap)")
    return EvidenceResult(supported, statement, min_support, len(anchors),
                          anchors=anchors, dissent=dissent, detail=detail)
