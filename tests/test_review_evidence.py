from types import SimpleNamespace
import json
import subprocess

import pytest

from spiral.review_evidence import ReviewEvidence
from test_validation_checkpoints import runner, seed, verdicts


def planned_check(root, command, *, goal="goal", requirements=None, duplicate=False):
    task = {"title": "Verify", "description": "Run the acceptance check",
            "verify": command, "requirements": requirements or ["R1"]}
    path = root / ".spiral/plan.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"goal": goal, "plan": {"milestones": [
        {"title": "Verify", "tasks": [task, task] if duplicate else [task]}]}}))


@pytest.mark.parametrize("exit_code", [0, 1])
def test_final_review_observes_real_planned_check_output_once(tmp_path, monkeypatch, exit_code):
    seed(tmp_path)
    command = f"printf 'observed CLI output: 2\\n'; exit {exit_code}"
    planned_check(tmp_path, command, duplicate=True)
    reply = {"verdicts": [{"id": "R1", "status": "unjudged",
                           "evidence": "Output alone does not establish every constraint"}]}
    obj = runner(tmp_path, monkeypatch, [reply], requirements=[
        {"id": "R1", "text": "The CLI prints 2 and preserves behavior"}])
    def execute(command, **options):
        obj.gate_calls.append(command)
        result = subprocess.run(command, shell=True, cwd=tmp_path,
                                capture_output=True, text=True, timeout=5)
        return SimpleNamespace(code=result.returncode, ok=result.returncode == 0,
                               out=result.stdout + result.stderr)
    obj._run_verified_command = execute
    result = obj.validate_only("goal")
    prompt = obj.ol.inputs[0][-1]["content"]
    assert obj.gate_calls == [command]
    assert f'"exit_code": {exit_code}' in prompt
    assert "observed CLI output: 2" in prompt
    assert '"requirements": ["R1"]' in prompt
    assert '"same_workspace_revision": true' in prompt
    # Neither a completed task nor a passing command overrides the actual review.
    assert result[0]["status"] == "unjudged" and not result[0]["fresh"]


@pytest.mark.parametrize("binding", ["other goal", "unknown requirement", "malformed plan"])
def test_final_review_does_not_run_unbound_plan_checks(tmp_path, monkeypatch, binding):
    seed(tmp_path)
    planned_check(tmp_path, "do-not-execute", goal="other" if binding == "other goal" else "goal",
                  requirements=["foreign"] if binding == "unknown requirement" else ["R1"])
    if binding == "malformed plan":
        (tmp_path / ".spiral/plan.json").write_text("invalid JSON")
    obj = runner(tmp_path, monkeypatch, [verdicts("R1")],
                 requirements=[{"id": "R1", "text": "Inspect"}])
    obj.validate_only("goal")
    assert obj.gate_calls == []


def test_plan_check_already_observed_for_requirement_is_not_repeated(tmp_path, monkeypatch):
    seed(tmp_path)
    planned_check(tmp_path, "verify-command")
    obj = runner(tmp_path, monkeypatch, [verdicts("R1")], requirements=[
        {"id": "R1", "text": "Inspect the observed result"},
        {"id": "R2", "text": "Execute", "check": "verify-command"}])
    obj.validate_only("goal")
    assert obj.gate_calls == ["verify-command"]


def test_planned_check_output_is_not_reused_after_source_changes(tmp_path, monkeypatch):
    seed(tmp_path)
    planned_check(tmp_path, "verify-command")
    obj = runner(tmp_path, monkeypatch, [verdicts("R1")],
                 requirements=[{"id": "R1", "text": "Inspect"}])
    obj.validate_only("goal")
    (tmp_path / "unselected.py").write_text("B = 2\n")
    resumed = runner(tmp_path, monkeypatch, [verdicts("R1")], resume=True,
                     requirements=[{"id": "R1", "text": "Inspect"}])
    result = resumed.validate_only("goal")
    assert resumed.gate_calls == ["verify-command"] and resumed.ol.calls == 1
    assert not result[0]["inference_reused"]


def test_an_unselected_source_change_invalidates_observed_check(tmp_path):
    seed(tmp_path)
    evidence = ReviewEvidence(tmp_path)
    evidence.record_check("task 1.1", "check", SimpleNamespace(code=0, out="ok"), [])
    assert evidence.current()
    (tmp_path / "unselected.py").write_text("B = 2\n")
    assert not evidence.current()
    assert '"same_workspace_revision": false' in evidence.render()


def hidden_check(root):
    path = root / ".spiral/probe/check.py"
    path.parent.mkdir(parents=True)
    path.write_text("print('real verification ran')\n")
    return path


def test_actual_validator_gets_executed_check_source_despite_filtered_tree(tmp_path, monkeypatch):
    seed(tmp_path)
    hidden_check(tmp_path)
    obj = runner(tmp_path, monkeypatch, [verdicts("R1")], requirements=[
        {"id": "R1", "text": "The supplied check works"},
        {"id": "R2", "text": "Run the check", "check": "python .spiral/probe/check.py"},
    ])
    result = obj.validate_only("Check the existing project")
    prompt = obj.ol.inputs[0][-1]["content"]
    assert ".spiral/probe/check.py; sha256 " in prompt
    assert "print('real verification ran')" in prompt
    assert '"exit_code": 0' in prompt and '"output_tail": "verified"' in prompt
    assert all(row["fresh"] for row in result)


def test_unknown_evidence_triggers_one_bounded_source_lookup_and_current_review(tmp_path, monkeypatch):
    seed(tmp_path)
    hidden_check(tmp_path)
    unknown = {"verdicts": [{"id": "R1", "status": "unjudged", "evidence": "need the supplied source",
                            "context_requests": [{"path": ".spiral/probe/check.py", "offset": 0}]}]}
    obj = runner(tmp_path, monkeypatch, [unknown, verdicts("R1")],
                 requirements=[{"id": "R1", "text": "Inspect the check"}])
    result = obj.validate_only("goal")
    assert obj.ol.calls == 2 and result[0]["status"] == "implemented"
    assert "print('real verification ran')" not in obj.ol.inputs[0][-1]["content"]
    assert "print('real verification ran')" in obj.ol.inputs[1][-1]["content"]
    resumed = runner(tmp_path, monkeypatch, [], resume=True,
                     requirements=[{"id": "R1", "text": "Inspect the check"}])
    result = resumed.validate_only("goal")
    assert resumed.ol.calls == 0 and result[0]["inference_reused"]


def test_missing_evidence_remains_unjudged_without_editing_or_unbounded_retries(tmp_path, monkeypatch):
    seed(tmp_path)
    hidden_check(tmp_path)
    unknown = {"verdicts": [{"id": "R1", "status": "unjudged", "context_requests": [
        {"path": ".spiral/probe/check.py"}]}]}
    obj = runner(tmp_path, monkeypatch, [unknown, unknown],
                 requirements=[{"id": "R1", "text": "Inspect"}])
    result = obj.validate_only("goal")
    assert obj.ol.calls == 2 and result[0]["status"] == "unjudged" and not result[0]["fresh"]
    assert obj._remediate("goal", None, result) is False


def test_hidden_check_change_invalidates_judgment_and_checkpoint(tmp_path, monkeypatch):
    seed(tmp_path)
    path = hidden_check(tmp_path)
    def changed():
        path.write_text("raise SystemExit(1)\n")
        return verdicts("R1")
    obj = runner(tmp_path, monkeypatch, [changed], requirements=[
        {"id": "R1", "text": "Check source"},
        {"id": "R2", "text": "Run", "check": "python .spiral/probe/check.py"}])
    result = obj.validate_only("goal")
    assert all(not row["fresh"] for row in result)
    assert not (tmp_path / ".spiral/planning-checkpoints/validation_1_0.json").exists()


def test_explicit_lookup_refuses_escape_symlink_and_missing_paths(tmp_path):
    workspace = tmp_path / "work"; workspace.mkdir()
    outside = tmp_path / "outside.py"; outside.write_text("outside data")
    (workspace / "link.py").symlink_to(outside)
    evidence = ReviewEvidence(workspace)
    assert not evidence.request([{"path": "../outside.py"}, {"path": "link.py"},
                                 {"path": "missing.py"}])
    assert "outside data" not in evidence.render()
    with pytest.raises(ValueError): evidence.request([{"path": "a", "offset": True}])
    with pytest.raises(ValueError): evidence.request([{"path": "a"}] * 9)
