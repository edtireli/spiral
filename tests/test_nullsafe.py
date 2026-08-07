"""The null-default trap, pinned.

`{"k": None}.get("k", "fallback")` returns None — the default applies to a *missing*
key, not a null one. Model-authored JSON and public APIs emit explicit nulls constantly,
so this has now caused three separate crashes in this project.
"""
from spiral.nullsafe import pick, pick_dict, pick_list, pick_str


def test_null_value_takes_the_default_unlike_dict_get():
    d = {"arxiv_id": None}
    assert d.get("arxiv_id", "paper") is None      # the trap, documented
    assert pick(d, "arxiv_id", "paper") == "paper"  # the fix


def test_missing_key_still_takes_the_default():
    assert pick({}, "k", "d") == "d"
    assert pick({"k": "v"}, "k", "d") == "v"


def test_falsey_but_real_values_survive():
    assert pick({"k": 0}, "k", 9) == 0
    assert pick({"k": ""}, "k", "d") == ""
    assert pick({"k": False}, "k", True) is False


def test_typed_helpers_are_always_safe_to_use():
    assert pick_str({"k": None}, "k") == ""
    assert pick_str({"k": 12}, "k") == "12"
    assert pick_list({"k": None}, "k") == []
    for _ in pick_list({"k": None}, "k"):
        raise AssertionError("iterating a null list must be a no-op")
    assert pick_dict({"k": None}, "k") == {}
    assert pick_dict({"k": None}, "k").get("anything") is None
    assert pick(None, "k", "d") == "d"             # not even a dict


def test_the_reading_note_crash_cannot_recur():
    """A model emitting {"arxiv_id": null} killed a 50-minute research run at
    research_loop._ensure_reading_notes."""
    note = {"arxiv_id": None, "summary": "x"}
    label = pick_str(note, "arxiv_id") or "paper"
    assert label[:24] == "paper"


def test_the_citation_graph_crash_cannot_recur():
    """Semantic Scholar returns {"data": null} for some papers."""
    from spiral.cite_graph import parse_edges

    assert parse_edges({"data": None}, "references") == []
