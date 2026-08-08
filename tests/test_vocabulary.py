"""A word is this corpus's subject, or it is just how papers are written.

Frequency cannot tell those apart, and that was measured on 278 real physics papers
rather than assumed: `coset` appeared in 13% of documents and `corresponding` in 18%,
`wess-zumino-witten` in 6% and `introduction` in 7%. A hand-written stoplist is an
unbounded game — after two hundred entries the top "discoveries" were still
`make ←→ tensors`. The comparison against a background is what actually separates them.
"""
from collections import Counter

import pytest

from spiral.vocabulary import (
    Background, distinctive_terms, filtered, specificity,
)


def _background(**counts) -> Background:
    return Background(counts=Counter(counts), documents=100, label="test")


# ------------------------------------------------------------------ the estimator
def test_a_word_at_its_usual_rate_scores_near_zero():
    """`corresponding` is as common here as anywhere; that is the whole point."""
    bg = _background(corresponding=100, coset=1, theory=200)
    z = specificity(Counter({"corresponding": 10, "coset": 10, "theory": 20}), bg)
    assert abs(z["corresponding"]) < 1.0
    assert z["coset"] > 2.0, "a term absent from ordinary writing must stand out"


def test_generic_field_vocabulary_is_subtracted_too():
    """The background is same-field on purpose: `theory` is as uninformative as
    `introduction` when every paper in the field says it."""
    bg = _background(theory=500, coset=2)
    z = specificity(Counter({"theory": 40, "coset": 8}), bg)
    assert z["theory"] < z["coset"]


def test_a_rare_word_is_shrunk_not_declared_infinite():
    """The failure mode of every raw frequency ratio: seen twice here, never in the
    background, ratio infinite, meaning nothing. The prior is what prevents it."""
    bg = _background(common=1000, other=1000)
    z = specificity(Counter({"fluke": 2, "common": 50}), bg)
    assert z["fluke"] < 6.0, f"a doubleton scored {z['fluke']}"


def test_more_evidence_earns_a_higher_score():
    bg = _background(word=10, filler=5000)
    weak = specificity(Counter({"word": 3, "filler": 100}), bg)["word"]
    strong = specificity(Counter({"word": 60, "filler": 100}), bg)["word"]
    assert strong > weak


def test_no_background_degrades_to_frequency_rather_than_crashing():
    """A network failure must not take the run with it."""
    z = specificity(Counter({"a": 10, "b": 1}), Background())
    assert z["a"] > z["b"]


def test_an_empty_target_yields_nothing():
    assert specificity(Counter(), _background(x=5)) == {}


# ------------------------------------------------------------------ selection
def test_distinctive_terms_keeps_the_subject_and_drops_the_scaffolding():
    # enough documents for the evidence to survive the prior: two occurrences in a
    # seven-token corpus SHOULD be shrunk, which is what the doubleton test asserts
    bg = _background(corresponding=200, introduction=200, coset=1, cohomology=1)
    docs = ([["coset", "corresponding", "introduction"]] * 5
            + [["coset", "cohomology", "corresponding"]] * 5
            + [["coset", "cohomology", "introduction"]] * 5)
    kept = distinctive_terms(docs, bg, min_z=1.5, min_df=2)
    assert "coset" in kept and "cohomology" in kept
    assert "corresponding" not in kept and "introduction" not in kept


def test_a_term_in_one_document_is_not_a_concept():
    bg = _background(filler=500)
    docs = [["once"], ["filler"], ["filler"]]
    assert "once" not in distinctive_terms(docs, bg, min_z=0.0, min_df=2)


def test_filtering_is_per_side_not_pooled():
    """Scoring a pooled corpus measures its bigger half. Pooling 50 seeded papers with
    228 pulled in by the citation graph put `coset` below the neighbourhood's own
    vocabulary, so the seeded subject was filtered out of its own analysis."""
    bg = _background(filler=5000, small=1, big=1)
    small_side = [["small", "filler"]] * 4
    big_side = [["big", "filler"]] * 60
    assert "small" in distinctive_terms(small_side, bg, min_z=1.0, min_df=2)
    pooled = distinctive_terms(small_side + big_side, bg, min_z=1.0, min_df=2)
    assert "big" in pooled


def test_filtered_documents_keep_only_the_kept_terms():
    bg = _background(noise=900, signal=1)
    docs = [["signal", "noise"], ["signal", "noise"], ["signal", "noise"]]
    out = filtered(docs, bg, min_z=1.0, min_df=2)
    assert all("noise" not in d for d in out)
    assert all("signal" in d for d in out)


def test_it_is_deterministic():
    bg = _background(a=100, b=1)
    docs = [["a", "b"], ["a", "b"], ["b"]]
    assert distinctive_terms(docs, bg) == distinctive_terms(docs, bg)


# ------------------------------------------------------------------ persistence
def test_a_background_round_trips(tmp_path):
    bg = _background(alpha=3, beta=4)
    path = tmp_path / "bg.json"
    bg.save(path)
    back = Background.load(path)
    assert back and back.counts == bg.counts and back.documents == bg.documents


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "bg.json"
    path.write_text("{not json")
    assert Background.load(path) is None


def test_building_a_background_strips_markup_like_the_corpus_does():
    bg = Background.build([r"\begin{abstract} We study the coset \end{abstract}"])
    assert "coset" in bg.counts
    for markup in ("begin", "end", "abstract"):
        assert markup not in bg.counts, markup
