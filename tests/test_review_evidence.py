from types import SimpleNamespace

import pytest

from spiral.review_evidence import ReviewEvidence
from test_validation_checkpoints import runner, seed, verdicts


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
