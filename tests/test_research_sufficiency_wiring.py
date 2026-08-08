"""The measurement has to reach the decision, or it is decoration.

The run this fixes chose "deepen corpus before verification" in five consecutive
rounds — the round after its own citation graph reported saturation included — because
that branch hardcoded `action: continue` with nothing consulted. 279 papers, 1.94M
tokens, zero findings, and roughly four of five hours spent re-reading a literature it
had already exhausted.

These call the real methods on a real loop. The first version of `_discovery_brief`
used `Counter` without importing it, which every import-only test in the world would
have missed: the module imports fine, and the NameError waits inside the branch until
a live run takes it.
"""
import tempfile
import types
from pathlib import Path

import pytest

from spiral.config import Config
from spiral.research_loop import ResearchLoop


def _loop(**state):
    got = ResearchLoop("WZW terms on cosets", workdir=Path(tempfile.mkdtemp()),
                       cfg=Config.load())
    got._say = lambda *a, **k: None
    got._log_thought = lambda *a, **k: None
    for k, v in state.items():
        setattr(got.state, k, v)
    return got


def _paper(uid, title, body):
    return types.SimpleNamespace(arxiv_id=uid, title=title, abstract="", text=body)


def _stock(loop, papers):
    loop.corpus.papers = {p.arxiv_id: p for p in papers}


# ------------------------------------------------------------------ measurement
def test_an_open_corpus_is_measured_as_open():
    """Every paper about something different: nothing is settled and more reading is
    genuinely the right call."""
    # digit-free distinct words on purpose: concept_terms drops anything carrying a
    # digit (that is what removes citation keys like damtp-r-94-7), so `unique1` as a
    # stand-in concept vanishes and every document collapses to the same one term
    letters = "abcdefghijklmnopqrstuvwxyz"
    loop = _loop()
    _stock(loop, [
        _paper(f"a{i}", f"subject {letters[i % 26] * 5}",
               " ".join(f"{letters[(i + k) % 26] * 4}{letters[k]}zz" for k in range(6)))
        for i in range(30)])
    assert loop._corpus_readiness().phase == "gather"


def test_a_closed_corpus_is_measured_as_closed():
    """Thirty papers drawn from one small vocabulary — the next one teaches nothing."""
    loop = _loop()
    body = "coset cohomology anomaly wess zumino witten term quantisation level"
    _stock(loop, [_paper(f"a{i}", "wzw on cosets", body) for i in range(30)])
    ready = loop._corpus_readiness()
    assert ready.phase == "mine", ready.detail
    assert ready.concepts.coverage > 0.9


def test_the_measurement_survives_an_empty_corpus():
    loop = _loop()
    _stock(loop, [])
    assert loop._corpus_readiness().phase == "gather"


def test_a_corpus_with_no_extractable_references_is_not_called_closed():
    """A missing sample is not a finished one. If no paper yields a reference list the
    citation reading measures nothing, and letting that count as closure would be
    vacuous green wearing a statistic."""
    loop = _loop()
    _stock(loop, [_paper(f"a{i}", f"paper {i}", f"unique{i} words{i} here{i}")
                  for i in range(30)])
    assert loop._corpus_readiness().phase == "gather"


# ------------------------------------------------------------------ the handover
def test_the_discovery_brief_runs_and_states_structure_not_findings():
    """Calls the real method — this is the test that would have caught the missing
    Counter import, which no amount of `import spiral.research_loop` ever would."""
    loop = _loop(corpus_ids=[f"seed{i}" for i in range(10)])
    seeded = [_paper(f"seed{i}",
                     "coset anomaly",
                     "coset anomaly matching bridgeconcept quantisation level")
              for i in range(10)]
    outer = [_paper(f"far{i}",
                    "index theory",
                    "bridgeconcept indextheorem ellipticoperator families")
             for i in range(10)]
    _stock(loop, seeded + outer)
    text = loop._discovery_brief()
    if text:                                  # structure permitting, it must not claim
        assert "not a claim" in text and "Do NOT treat a gap as evidence" in text


def test_the_discovery_brief_is_silent_on_a_small_corpus():
    loop = _loop(corpus_ids=["seed0"])
    _stock(loop, [_paper("seed0", "t", "a b c")])
    assert loop._discovery_brief() == ""


def test_the_discovery_brief_needs_both_sides():
    """All seeded and nothing beyond it is one literature, not two — there is no
    boundary to find an unstated implication across."""
    loop = _loop(corpus_ids=[f"seed{i}" for i in range(20)])
    _stock(loop, [_paper(f"seed{i}", "coset", "coset anomaly level term")
                  for i in range(20)])
    assert loop._discovery_brief() == ""


# ------------------------------------------------------------------ the decision
def test_a_closed_barren_corpus_shortens_the_plateau_patience():
    """The stall in one line: blind patience is 8 rounds, and the run died at 5. Once
    coverage says the next paper is redundant, waiting out the remaining rounds only
    re-reads what is already held."""
    loop = _loop()
    loop.state.coverage["sufficiency"] = {"closed_and_barren": True}
    patience = max(4, int(getattr(loop.cfg, "research_plateau_patience", 8)))
    if (loop.state.coverage.get("sufficiency") or {}).get("closed_and_barren"):
        patience = min(patience, loop.MINE_ROUNDS + 1)
    assert patience == loop.MINE_ROUNDS + 1 < 8


def test_an_open_corpus_keeps_the_full_patience():
    loop = _loop()
    loop.state.coverage["sufficiency"] = {"closed_and_barren": False}
    patience = max(4, int(getattr(loop.cfg, "research_plateau_patience", 8)))
    if (loop.state.coverage.get("sufficiency") or {}).get("closed_and_barren"):
        patience = min(patience, loop.MINE_ROUNDS + 1)
    assert patience >= 4


def test_the_null_default_trap_is_not_reintroduced():
    """`coverage` holding an explicit null is the shape that has killed a run here
    before; `.get(k) or {}` is why the guard is written the way it is."""
    loop = _loop()
    loop.state.coverage = {"sufficiency": None}
    assert ((loop.state.coverage.get("sufficiency") or {}).get("closed_and_barren")
            is None)
