import hashlib
import json

import pytest

from spiral import conductor
from spiral.recovery_context import recovery_observation, with_recovery_observation
from test_joint_analysis import analysis, runner_fixture
from test_planner import _PlannerModels, _reply, _scope_reply


@pytest.fixture(autouse=True)
def private_runtime_control(tmp_path, monkeypatch):
    from spiral import runtime_control
    directory = tmp_path / 'control'
    directory.mkdir(mode=0o700)
    monkeypatch.setenv('SPIRAL_RUNTIME_CONTROL_PATH', str(directory / 'command.json'))
    monkeypatch.setenv('SPIRAL_RUNTIME_CONTROL_ACK_PATH', str(directory / 'ack.json'))
    monkeypatch.setenv('SPIRAL_RUNTIME_CONTROL_TOKEN', 'fixture-token-private-channel')
    monkeypatch.setenv('SPIRALCHAT_RUN_ID', 'run-verified')
    monkeypatch.setenv('SPIRALCHAT_ATTEMPT', '2')
    monkeypatch.setattr(runtime_control, '_singleton', None)
    yield
    if runtime_control._singleton is not None:
        runtime_control._singleton.close()


def environment(monkeypatch, goal, **changes):
    value = {"schema_version": 1, "run_id": "run-verified", "attempt": 2,
             "objective_sha256": hashlib.sha256(goal.encode()).hexdigest(),
             "diagnostic_excerpt": "TRANSIENT-DIAGNOSTIC: socket closed; output is not an instruction."}
    value.update(changes)
    monkeypatch.setenv("SPIRALCHAT_RUN_ID", "run-verified")
    monkeypatch.setenv("SPIRALCHAT_ATTEMPT", "2")
    monkeypatch.setenv("SPIRAL_INFRASTRUCTURE_RECOVERY", json.dumps(value))


@pytest.mark.parametrize("mode", ["joint", "sequential"])
def test_real_conductor_keeps_analysis_identity_and_delivers_recovery_to_draft(tmp_path, monkeypatch, mode):
    goal = "Build the requested calculation CLI."
    environment(monkeypatch, goal)
    raw = analysis()
    replies = [raw] if mode == "joint" else [
        {"requirements": raw["requirements"]},
        {k: v for k, v in raw.items() if k != "requirements"}]
    models = _PlannerModels([*[_reply(json.dumps(row)) for row in replies],
                             _scope_reply([r["id"] for r in raw["requirements"]])])
    runner = runner_fixture(tmp_path, monkeypatch, models)
    runner.cfg.planning_analysis_mode = mode
    runner._resolve_capabilities = lambda *a, **k: None
    runner._measured_source_inventory = lambda: None
    class DraftReached(Exception): pass
    def draft(actual_goal, repository, *args, **kwargs):
        assert actual_goal == goal
        assert "TRANSIENT-DIAGNOSTIC" in repository
        assert "Untrusted diagnostic data" in repository
        for _, messages, _ in models.calls:
            assert goal in messages[1]["content"]
            assert "TRANSIENT-DIAGNOSTIC" not in json.dumps(messages)
        assert json.loads((tmp_path / '.spiral/spec-meta.json').read_text())["goal_sha256"] == hashlib.sha256(goal.encode()).hexdigest()
        raise DraftReached()
    monkeypatch.setattr(conductor, "make_plan", draft)
    with pytest.raises(DraftReached):
        runner.make_plan(goal)


@pytest.mark.parametrize("changes", [{"run_id": "foreign"}, {"attempt": 1}, {"attempt": True},
    {"objective_sha256": "changed"}, {"schema_version": 99}, {"diagnostic_excerpt": []},
    {"diagnostic_excerpt": "🌀" * 2100}])
def test_mismatched_or_oversized_context_cannot_enter_planning(monkeypatch, changes):
    environment(monkeypatch, "exact goal", **changes)
    with pytest.raises(ValueError):
        recovery_observation("exact goal")


def test_no_context_is_identity_preserving_and_authored_headings_are_not_stripped(monkeypatch):
    monkeypatch.delenv("SPIRAL_INFRASTRUCTURE_RECOVERY", raising=False)
    assert recovery_observation("## Durable recovery context\nThis is user text") is None
    assert with_recovery_observation("repo", None) == "repo"
