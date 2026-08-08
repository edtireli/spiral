"""Which words are this corpus's subject, and which are just how papers are written.

Bag-of-words plus a stoplist cannot answer that, and it was measured rather than
assumed. Across 278 real physics papers, ``coset`` appeared in 13% of documents and
``corresponding`` in 18%; ``wess-zumino-witten`` in 6% and ``introduction`` in 7%.
Signal and scaffolding occupy the same frequency range, so no ceiling separates them,
and a hand-written stoplist is an unbounded game — after two hundred entries the top
"discoveries" were still ``make ←→ tensors`` and ``algebra ←→ sitter``.

The question is comparative, so it needs a comparison. Against a background sample of
ordinary writing from the same field, ``corresponding`` occurs at its usual rate and
``coset`` does not, and that difference is the whole signal. The estimator is the
log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008),
which is the standard tool for exactly this and is deliberately not a raw frequency
ratio: a word appearing twice in the target and never in the background has an infinite
ratio and means nothing, while the prior shrinks rare words toward zero and the variance
turns the score into a z-score that can be thresholded honestly.

The background is drawn from the same broad field on purpose. It subtracts generic
academic English *and* generic physics vocabulary together, leaving what is specific to
the question being asked — ``theory`` and ``field`` are as uninformative here as
``introduction``, and a general-English baseline would keep them.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


def _cache_dir() -> Path:
    root = Path.home() / ".cache" / "spiral" / "background"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Background:
    """Word frequencies in ordinary writing from a field.

    ``counts`` is over abstracts, which are prose: full TeX bodies would put the
    markup back in on the baseline side and cancel the very thing being measured.
    """
    counts: Counter = field(default_factory=Counter)
    documents: int = 0
    label: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "label": self.label, "documents": self.documents,
            "counts": dict(self.counts)}))

    @classmethod
    def load(cls, path: Path) -> "Background | None":
        try:
            raw = json.loads(path.read_text())
        except Exception:
            return None
        return cls(counts=Counter(raw.get("counts") or {}),
                   documents=int(raw.get("documents") or 0),
                   label=str(raw.get("label") or ""))

    @classmethod
    def build(cls, texts: list[str], *, label: str = "") -> "Background":
        from spiral.corpus_sufficiency import concept_terms

        counts: Counter = Counter()
        n = 0
        for text in texts or []:
            terms = concept_terms(text or "")
            if not terms:
                continue
            n += 1
            counts.update(terms)
        return cls(counts=counts, documents=n, label=label)


# Broad, ordinary queries. The point is a REPRESENTATIVE sample of how the field writes,
# not a relevant one — a background assembled from papers about the research question
# would subtract the question itself and leave nothing.
_FIELD_PROBES = {
    "physics-math": ["quantum field theory", "differential geometry",
                     "statistical mechanics", "algebraic topology",
                     "general relativity", "condensed matter"],
    "bio-med": ["gene expression", "clinical trial", "protein structure",
                "neural circuit", "cell signalling", "population genetics"],
}


def fetch_background(domain: str = "physics-math", *, per_query: int = 60,
                     max_age_days: float = 30.0, refresh: bool = False) -> Background:
    """A background sample for a domain, cached on disk.

    Network failure is not fatal: an empty Background makes ``specificity`` fall back to
    plain frequency, which is worse but still runs. A hard dependency on the network for
    something this peripheral would be the wrong trade.
    """
    probes = _FIELD_PROBES.get(domain) or _FIELD_PROBES["physics-math"]
    path = _cache_dir() / f"{domain.replace('/', '-')}.json"
    if not refresh and path.is_file():
        age = (time.time() - path.stat().st_mtime) / 86400.0
        cached = Background.load(path)
        if cached and cached.documents and age <= max_age_days:
            return cached

    from spiral.research import arxiv

    texts: list[str] = []
    for probe in probes:
        try:
            for hit in arxiv(probe, k=per_query) or []:
                body = f"{getattr(hit, 'title', '')} {getattr(hit, 'text', '')}"
                if body.strip():
                    texts.append(body)
        except Exception:
            continue
    got = Background.build(texts, label=domain)
    if got.documents:
        try:
            got.save(path)
        except OSError:
            pass
    return got


def specificity(counts: Counter, background: Background, *,
                prior_strength: float = 1.0) -> dict[str, float]:
    """z-scores for how over-represented each term is against the background.

    Monroe et al.'s log-odds ratio with an informative Dirichlet prior. The prior is
    the background's own distribution scaled by ``prior_strength``, so a term seen once
    in the target and never in the background is shrunk rather than declared infinitely
    distinctive — which is the failure mode of every raw ratio.

    With no background at all this degrades to log frequency: honest, and clearly
    labelled as the fallback it is.
    """
    n_t = sum(counts.values())
    n_b = background.total
    if not n_t:
        return {}
    if not n_b:
        return {w: math.log1p(c) for w, c in counts.items()}

    a0 = prior_strength * n_b
    out: dict[str, float] = {}
    for word, y_t in counts.items():
        y_b = background.counts.get(word, 0)
        a_w = prior_strength * (y_b + 1)          # +1 so unseen words still get mass
        # log-odds in each corpus, with the prior folded in
        num_t = y_t + a_w
        den_t = n_t + a0 - num_t
        num_b = y_b + a_w
        den_b = n_b + a0 - num_b
        if den_t <= 0 or den_b <= 0:
            continue
        delta = math.log(num_t / den_t) - math.log(num_b / den_b)
        variance = 1.0 / num_t + 1.0 / num_b
        out[word] = delta / math.sqrt(variance)
    return out


def distinctive_terms(documents: list[list[str]], background: Background, *,
                      min_z: float = 2.0, min_df: int = 2,
                      top: int = 0) -> set[str]:
    """The subset of a corpus's vocabulary that is actually about its subject.

    ``min_z`` is a z-score, so 2.0 is the usual two-sigma reading: the term is
    over-represented against the background by more than sampling noise explains.
    """
    df = Counter()
    freq = Counter()
    for doc in documents or []:
        df.update(set(doc or []))
        freq.update(doc or [])
    scores = specificity(freq, background)
    keep = [(w, z) for w, z in scores.items() if z >= min_z and df.get(w, 0) >= min_df]
    keep.sort(key=lambda x: -x[1])
    if top:
        keep = keep[:top]
    return {w for w, _ in keep}


def filtered(documents: list[list[str]], background: Background,
             **kw) -> list[list[str]]:
    """The same documents with only their subject-bearing terms kept.

    Filter each side of a comparison SEPARATELY. Scoring a pooled corpus measures the
    bigger half: pooling 50 seeded papers with 228 pulled in by the citation graph put
    ``coset`` at z=2.81 while the neighbourhood's own vocabulary — anti-de sitter,
    holography — sat far above it, so the seeded subject was filtered out of its own
    analysis and nothing was left to bridge from.
    """
    vocab = distinctive_terms(documents, background, **kw)
    return [[t for t in (doc or []) if t in vocab] for doc in (documents or [])]
