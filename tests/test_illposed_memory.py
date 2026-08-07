"""A deterministic refusal the model never sees is one it walks into again.

Watched live: the well-posedness check rejected H^5 of SU(3)/SU(2)xU(1) — that space
is CP^2, dimension 4, so the group is zero before any differential is written. It then
rejected the identical angle again the next round. The reason had been computed, printed,
and written onto the angle object, which was immediately discarded. Nothing carried it
back to the model, and nothing stopped the same question being adjudicated twice.

Same lesson as the build loop's identical-reply guard and the harness-error channel:
the reason has to reach whoever can act on it.
"""
import tempfile
from pathlib import Path

import pytest

from spiral.config import Config
from spiral.research_loop import ResearchLoop


@pytest.fixture
def loop():
    got = ResearchLoop("WZW terms on cosets", workdir=Path(tempfile.mkdtemp()),
                       cfg=Config.load())
    got._say = lambda *a, **k: None
    got._log_thought = lambda *a, **k: None
    return got


VACUOUS = {"question": "Which cosets admit a nonvanishing H^5(SU(3)/SU(2)xU(1))?",
           "target": "SU(3)/SU(2)xU(1)",
           "check_plan": "Compute H^5(su(3), su(2)+u(1); R)"}
GOOD = {"question": "Does the 4D WZW coefficient constrain HNL generations?",
        "target": "SU(3) in D=4",
        "check_plan": "Compute H^5(su(3);R) and match the anomaly coefficient"}


def test_the_live_failure_is_recorded_not_just_printed(loop):
    assert loop._precheck_angles([dict(VACUOUS)]) == []
    assert loop.state.ruled_out_angles, "the refusal was printed and then forgotten"
    row = next(iter(loop.state.ruled_out_angles.values()))
    assert "SU(3)/SU(2)xU(1)" in row["question"] or "H^5" in row["question"]
    assert "dim" in row["reason"], f"the arithmetic must be kept: {row['reason']}"


def test_the_same_angle_is_not_adjudicated_twice(loop):
    loop._precheck_angles([dict(VACUOUS)])
    before = dict(loop.state.ruled_out_angles)
    loop._precheck_angles([dict(VACUOUS)])
    assert loop.state.ruled_out_angles == before, "a second refusal was recorded"


def test_the_reason_reaches_the_next_prompt(loop):
    loop._precheck_angles([dict(VACUOUS)])
    brief = loop._illposed_brief()
    assert "ALREADY RULED OUT" in brief
    assert "SU(3)/SU(2)xU(1)" in brief or "H^5" in brief
    assert "dim" in brief, "the arithmetic reason must travel with the refusal"
    # actionable, not merely a prohibition
    assert "before proposing" in brief.lower() or "check dim" in brief.lower()


def test_an_empty_history_adds_nothing_to_the_prompt(loop):
    assert loop._illposed_brief() == ""


def test_a_well_posed_angle_is_never_ruled_out(loop):
    kept = loop._precheck_angles([dict(GOOD)])
    assert [a["question"] for a in kept] == [GOOD["question"]]
    assert loop.state.ruled_out_angles == {}


def test_the_memory_survives_a_resume(loop):
    """--resume rebuilds the loop from state.json; a refusal that does not persist
    means round 1 after a resume re-learns what round 1 before it already knew."""
    from dataclasses import asdict

    loop._precheck_angles([dict(VACUOUS)])
    snapshot = asdict(loop.state)
    assert snapshot.get("ruled_out_angles"), "not serialised into the run state"

    revived = ResearchLoop("WZW terms on cosets", workdir=Path(tempfile.mkdtemp()),
                           cfg=Config.load())
    revived._say = lambda *a, **k: None
    revived.state.ruled_out_angles = snapshot["ruled_out_angles"]
    assert "ALREADY RULED OUT" in revived._illposed_brief()


def test_the_brief_is_bounded(loop):
    """A run that refuses many angles must not grow the prompt without limit."""
    for i in range(40):
        loop._precheck_angles([{
            "question": f"Compute H^{9 + i}(SU(3)/SU(2)xU(1)) for case {i}",
            "check_plan": f"H^{9 + i}(su(3),su(2);R)"}])
    assert len(loop.state.ruled_out_angles) >= 20
    assert loop._illposed_brief().count("\n    why:") <= 12


def _quiet(work):
    got = ResearchLoop("WZW", workdir=work, cfg=Config.load())
    got._say = lambda *a, **k: None
    got._log_thought = lambda *a, **k: None
    return got


def test_the_memory_round_trips_through_the_real_save_path(tmp_path):
    """``asdict``/json is what actually writes state.json, and ``self.dir`` IS the
    workdir — pin the whole trip through the real functions, because a snapshot built
    by hand proves only that the snapshot was built by hand."""
    import json

    work = tmp_path / "ws2"
    work.mkdir()
    first = _quiet(work)
    first._precheck_angles([dict(VACUOUS)])
    first._save()

    written = json.loads(first._statefile().read_text())
    assert written.get("ruled_out_angles"), "not written to disk by _save"

    second = _quiet(work)
    assert second.state.ruled_out_angles, "not read back by _load"
    assert "ALREADY RULED OUT" in second._illposed_brief()


@pytest.mark.parametrize("stored", [
    {"topic": "WZW"},                                # written before the field existed
    {"topic": "WZW", "ruled_out_angles": None},      # present, but null
])
def test_a_state_file_predating_the_memory_still_resumes(tmp_path, stored):
    """``_load`` copies every key straight onto the state, so a null in the file lands
    as a null on the field — `dict.get(k, default)` does not protect against that, and
    the first refusal of the resumed run would raise instead of record. This is why
    _load carries a migration list, and why the new field has to be on it."""
    import json

    work = tmp_path / f"ws{abs(hash(str(stored)))}"
    work.mkdir()
    (work / "state.json").write_text(json.dumps(stored))

    revived = _quiet(work)
    assert isinstance(revived.state.ruled_out_angles, dict), stored
    revived._precheck_angles([dict(VACUOUS)])          # must not raise
    assert revived.state.ruled_out_angles
