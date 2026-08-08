"""Stopping is a measurement, not a mood.

A real run chose "deepen corpus before verification" in five consecutive rounds — the
round after its own citation graph reported saturation included — and ended with 279
papers, 1.94M tokens and zero findings. Nothing measured exhaustion, so nothing could
act on it. These pin the estimators against values worked out by hand, because an
estimator that is merely plausible is how you get a confident wrong number.
"""
from collections import Counter

import pytest

from spiral.corpus_sufficiency import (
    assess, chao1, concept_terms, good_turing_coverage, marginal_gain, readiness,
)


# ------------------------------------------------------------------ Good–Turing
def test_coverage_matches_the_hand_computation():
    """C = 1 - f1/N. Ten draws, three seen once, so the unseen mass is 0.3."""
    assert good_turing_coverage(
        Counter({"a": 4, "b": 3, "c": 1, "d": 1, "e": 1})) == pytest.approx(0.7)


def test_all_singletons_means_no_coverage():
    """Every item seen exactly once — the next draw is almost certainly new, which is
    the one case where continuing to gather is unambiguously right."""
    assert good_turing_coverage(Counter({k: 1 for k in "abcde"})) == 0.0


def test_no_singletons_means_full_coverage():
    assert good_turing_coverage(Counter({"a": 5, "b": 5})) == 1.0


def test_an_empty_sample_has_seen_nothing():
    """Returning 1.0 here would read as 'done' to every caller — an empty corpus is
    the least finished a corpus can be."""
    assert good_turing_coverage(Counter()) == 0.0


# ------------------------------------------------------------------ Chao1
def test_chao1_matches_the_bias_corrected_formula():
    """S_obs + f1(f1-1)/(2(f2+1)) = 5 + 3*2/(2*2) = 6.5"""
    assert chao1(Counter({"a": 4, "b": 2, "c": 1, "d": 1, "e": 1})) == pytest.approx(6.5)


def test_chao1_stays_finite_with_no_doubletons():
    """The uncorrected form divides by f2, and f2 is zero early in every run."""
    assert chao1(Counter({"a": 1, "b": 1, "c": 1})) == pytest.approx(6.0)


def test_chao1_of_a_closed_sample_is_what_was_seen():
    assert chao1(Counter({"a": 3, "b": 3})) == pytest.approx(2.0)
    assert chao1(Counter()) == 0.0


# ------------------------------------------------------------------ marginal gain
def test_marginal_gain_measures_only_what_the_recent_window_added():
    closed = [["a", "b"], ["a", "b"], ["a", "b"], ["c"]]
    assert marginal_gain(closed, window=2) == pytest.approx(0.5)
    opening = [["a"], ["b"], ["c"], ["d"]]
    assert marginal_gain(opening, window=2) == pytest.approx(1.0)


def test_marginal_gain_is_safe_on_degenerate_input():
    assert marginal_gain([]) == 0.0
    assert marginal_gain([["a"]], window=99) == 1.0


# ------------------------------------------------------------------ the verdict
def _closed(n=40):
    """A literature that has stopped moving: everything drawn from a small fixed set."""
    core = ["a", "b", "c", "d", "e", "f"]
    return [core[i % 3:(i % 3) + 3] for i in range(n)]


def _opening(n=40):
    """A literature still unfolding: every document brings terms nothing else uses.

    Written first as a chain — ``[t{i}, t{i+1}]`` — which the estimators correctly
    called closed, because a term shared by two adjacent documents is not a singleton
    and Good–Turing counts singletons. That fixture was the mistake, not the verdict.
    """
    return [[f"t{i}a", f"t{i}b", "shared"] for i in range(n)]


def test_a_closed_literature_is_enough():
    got = assess(_closed())
    assert got.enough, got.reason
    assert got.coverage == 1.0 and got.marginal_gain == 0.0


def test_an_opening_literature_is_not_enough_and_says_why():
    got = assess(_opening())
    assert not got.enough
    assert "coverage" in got.reason or "new items" in got.reason


def test_a_tiny_sample_is_never_enough_however_clean():
    """Five documents that all say the same thing score perfect coverage and mean
    nothing — a floor on sample size is what stops an accident reading as closure."""
    got = assess([["a", "b"]] * 5)
    assert got.coverage == 1.0
    assert not got.enough and "only 5 documents" in got.reason


def test_the_verdict_is_reported_with_its_numbers():
    got = assess(_closed())
    for key in ("coverage", "completeness", "marginal_gain", "estimated_total"):
        assert key in got.as_dict()
    assert "%" in got.sentence()


# ------------------------------------------------------------------ both readings
def test_mining_waits_for_concepts_as_well_as_references():
    """A citation graph can close while the vocabulary is still opening — that is a
    field being read narrowly, and more of the same papers will not fix it."""
    got = readiness(_closed(), _opening())
    assert got.phase == "gather"
    assert "concepts" in got.detail


def test_both_closed_means_mine():
    got = readiness(_closed(), _closed())
    assert got.phase == "mine"
    assert "redundant" in got.detail


def test_neither_closed_names_both_shortfalls():
    got = readiness(_opening(), _opening())
    assert got.phase == "gather"
    assert "references" in got.detail and "concepts" in got.detail


# ------------------------------------------------------------------ term extraction
def test_concept_terms_drop_scaffolding_not_content():
    terms = concept_terms(
        "We show that the results of this study are based on a new sigma model.")
    assert "sigma" in terms and "model" in terms
    for scaffold in ("we", "show", "that", "the", "results", "this", "study", "based"):
        assert scaffold not in terms, scaffold


def test_term_extraction_is_domain_agnostic():
    """The stoplist is generic English; nothing here knows what field it is reading."""
    physics = concept_terms("The Wess-Zumino-Witten term on the coset is quantised.")
    biology = concept_terms("Microglial activation precedes amyloid plaque formation.")
    assert "wess-zumino-witten" in physics and "coset" in physics
    assert "microglial" in biology and "amyloid" in biology
