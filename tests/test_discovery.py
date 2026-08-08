"""Novelty as a structural gap, checked against the case that defined the method.

Swanson, 1986: one literature said fish oil lowers blood viscosity, another said Raynaud
patients suffer high viscosity, and no paper said both. The connection was already
public and unread. That example is the fixture here because it has a known right answer,
and both of this module's design bugs were invisible until it was run against one.
"""
import pytest

from spiral.discovery import Associations, Connection, brief, find_connections

# A-side: the problem's own literature. Note `raynaud` appears in 5 of 6 documents —
# the subject of a corpus always saturates that corpus, and that is what broke the
# first two versions of this module.
SOURCE = [
    ["raynaud", "viscosity", "circulation"],
    ["raynaud", "viscosity", "platelet"],
    ["raynaud", "viscosity", "vasospasm"],
    ["raynaud", "platelet", "aggregation"],
    ["raynaud", "circulation", "vasospasm"],
    ["viscosity", "platelet", "circulation"],
]
# C-side: the other literature, whose own subject `fishoil` saturates it in turn.
DISTANT = [
    ["fishoil", "viscosity", "lipid"],
    ["fishoil", "viscosity", "membrane"],
    ["fishoil", "platelet", "aggregation"],
    ["fishoil", "platelet", "lipid"],
    ["viscosity", "lipid", "membrane"],
    ["fishoil", "viscosity", "platelet"],
]


def _find(source=SOURCE, distant=DISTANT, **kw):
    kw.setdefault("seeds", ["raynaud"])
    kw.setdefault("min_bridges", 1)
    kw.setdefault("min_strength", 0.2)
    return find_connections(source, distant, **kw)


def test_the_canonical_discovery_is_found():
    assert any(c.distant == "fishoil" for c in _find()), (
        "the A-B-C structure the method was invented for was not detected")


def test_a_saturated_seed_still_bridges():
    """First bug. PMI was the association measure, and the A-term is ubiquitous in its
    own corpus BY CONSTRUCTION — the corpus was fetched about it. With raynaud in 5 of 6
    documents, PMI against it is negative for every term, so the correct answer scored
    below threshold and nothing was ever returned."""
    a = Associations(SOURCE)
    assert a.df["raynaud"] == 5 and a.total == 6
    assert a.confidence("raynaud", "platelet") > 0
    assert any(c.distant == "fishoil" for c in _find())


def test_a_saturated_distant_subject_is_still_reachable():
    """Second bug, the same blind spot mirrored. The document-frequency band that keeps
    generic words out of the BRIDGE set was also applied to endpoints, so `fishoil` —
    in 5 of 6 distant documents — was excluded from its own corpus's vocabulary and
    could never be discovered."""
    c = Associations(DISTANT)
    assert "fishoil" not in c.vocabulary, "the band should still exclude it as a bridge"
    assert "fishoil" in c.attested, "but it must remain reachable as an endpoint"
    assert any(x.distant == "fishoil" for x in _find())


@pytest.mark.parametrize("where", ["source", "distant"])
def test_a_link_someone_already_stated_is_not_a_discovery(where):
    """The whole test of the method. A pair that already appears together is a citation,
    not a discovery — and it must be suppressed whichever corpus happens to state it."""
    stated = ["raynaud", "fishoil", "viscosity"]
    source = SOURCE + [stated] if where == "source" else SOURCE
    distant = DISTANT + [stated] if where == "distant" else DISTANT
    assert not any(c.distant == "fishoil" for c in _find(source, distant)), (
        f"a link stated in the {where} corpus was reported as new")


def test_bridges_carry_their_evidence():
    for c in _find():
        assert c.bridges, "a connection with no bridge is an assertion"
        for b in c.bridges:
            assert b.term and 0 <= b.strength <= 1
        assert c.a_documents > 0 and c.c_documents > 0


def test_the_chain_is_only_as_strong_as_its_weaker_link():
    for c in _find():
        for b in c.bridges:
            assert b.strength == min(b.a_strength, b.c_strength)


def test_requiring_more_bridges_narrows_the_result():
    """One bridge is a coincidence of vocabulary; several independent ones are a
    pattern. The knob has to actually bite."""
    loose = _find(min_bridges=1)
    strict = _find(min_bridges=3)
    assert len(strict) <= len(loose)


def test_generic_vocabulary_cannot_act_as_a_bridge():
    """A word in every document distinguishes nothing, whatever its correlation — this
    is the job the frequency band exists to do."""
    noisy_source = [doc + ["theory"] for doc in SOURCE]
    noisy_distant = [doc + ["theory"] for doc in DISTANT]
    for c in _find(noisy_source, noisy_distant):
        assert "theory" not in [b.term for b in c.bridges]


def test_nothing_in_common_yields_nothing():
    assert find_connections([["a", "b"], ["a", "c"]], [["x", "y"], ["x", "z"]],
                            seeds=["a"]) == []


def test_it_is_deterministic():
    assert [c.sentence() for c in _find()] == [c.sentence() for c in _find()]


def test_degenerate_input_does_not_raise():
    for args in (([], []), ([[]], [[]]), (SOURCE, []), ([], DISTANT)):
        find_connections(*args, seeds=["raynaud"])


# ------------------------------------------------------------------ handover
def test_the_brief_offers_structure_and_refuses_to_claim():
    text = brief(_find(), source_field="the problem", distant_field="the other field")
    assert "not a claim" in text
    assert "Do NOT treat a gap as evidence" in text
    assert "together in none" in text
    # it must also invite rejection, or every coincidence becomes a finding
    assert "coincidences" in text and "discard" in text


def test_an_empty_result_says_nothing_at_all():
    assert brief([]) == ""
