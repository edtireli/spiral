"""Completed inspection groups survive restart without replaying executable checks."""
import json
from types import SimpleNamespace

import pytest

from spiral import conductor
from spiral.conductor import Conductor, WorkerContext
from spiral.execution import BudgetExceeded
from spiral.llm import ChatResult
from test_planning_memory import Model, Config
from test_worker_memory import Quiet


def verdicts(*ids):
    return {"verdicts": [{"id": key, "status": "implemented", "evidence": "inspected source"}
                         for key in ids]}


def runner(root, monkeypatch, replies, *, resume=False, binding="same-gguf", requirements=None):
    obj = object.__new__(Conductor)
    obj.ws = root; obj.cfg = Config()
    obj.cfg.critic = SimpleNamespace(name=obj.cfg.planner.name, think=False)
    obj.c = Quiet(); obj.ledger = Quiet(); obj.gate = "check"
    obj._resume_validation = resume
    obj._build_worker_context = WorkerContext("Keep every authored constraint", "fresh runtime notes")
    obj._load_spec = lambda goal: requirements or [
        *[{"id": f"R{i}", "text": f"Requirement {i}", "kind": "feature"} for i in range(1, 6)],
        {"id": "R6", "text": "Run the behavioral check", "check": "verify-command"},
    ]
    obj._delivery_manifest = lambda goal: {}
    obj.gate_calls = []
    def gate(command, **options):
        obj.gate_calls.append(command)
        return SimpleNamespace(ok=True, code=0, out="verified")
    obj._run_verified_command = gate
    obj.ol = Model(binding=binding)
    obj.ol.replies = list(replies)
    obj.ol.inputs = []
    def chat(model, messages, **options):
        obj.ol.calls += 1; obj.ol.inputs.append(messages)
        value = obj.ol.replies.pop(0)
        if isinstance(value, BaseException): raise value
        if callable(value): value = value()
        return ChatResult(json.dumps(value), 10, 20, raw={"done_reason": "stop"})
    obj.ol.chat = chat
    monkeypatch.setattr(conductor, "build_relevant_repomap", lambda *a, **k: ("selected source", ["a.py"]))
    monkeypatch.setattr(conductor, "reveal", lambda *a, **k: None)
    return obj


def seed(root):
    (root / "a.py").write_text("A = 1\n")
    (root / "unselected.py").write_text("B = 1\n")


def test_actual_validator_restarts_completed_groups_and_reruns_executable_check(tmp_path, monkeypatch):
    seed(tmp_path)
    first = runner(tmp_path, monkeypatch, [])
    exhausted = BudgetExceeded("wall", first.ol.budget.snapshot())
    first.ol.replies = [verdicts("R1", "R2", "R3", "R4"), exhausted]
    with pytest.raises(BudgetExceeded): first.validate_only("exact goal")
    assert first.ol.calls == 2 and first.gate_calls == ["verify-command"]
    checkpoints = tmp_path / ".spiral/planning-checkpoints"
    assert (checkpoints / "validation_1_0.json").is_file()
    assert not (checkpoints / "validation_1_4.json").exists()
    resumed = runner(tmp_path, monkeypatch, [verdicts("R5")], resume=True)
    resumed._build_worker_context = WorkerContext("Keep every authored constraint", "changed runtime notes")
    result = resumed.validate_only("exact goal")
    assert resumed.ol.calls == 1 and resumed.gate_calls == ["verify-command"]
    assert {row["id"] for row in result} == {f"R{i}" for i in range(1, 7)}
    assert all(row["fresh"] for row in result)
    assert {row["id"] for row in result if row.get("inference_reused")} == {"R1", "R2", "R3", "R4"}


@pytest.mark.parametrize("change", ["unselected source", "requirements", "model binding"])
def test_changed_evidence_or_identity_refuses_old_review(tmp_path, monkeypatch, change):
    seed(tmp_path)
    first = runner(tmp_path, monkeypatch, [verdicts("R1", "R2", "R3", "R4"), verdicts("R5")])
    first.validate_only("exact goal")
    resumed = runner(tmp_path, monkeypatch, [verdicts("R1", "R2", "R3", "R4"), verdicts("R5")],
                     resume=True, binding="new-gguf" if change == "model binding" else "same-gguf")
    if change == "unselected source": (tmp_path / "unselected.py").write_text("B = 2\n")
    if change == "requirements": resumed._build_worker_context = WorkerContext("Changed authored constraint")
    result = resumed.validate_only("exact goal")
    assert resumed.ol.calls == 2 and not any(row.get("inference_reused") for row in result)


@pytest.mark.parametrize("failure", ["incomplete", "workspace mutation"])
def test_incomplete_or_stale_review_is_never_checkpointed(tmp_path, monkeypatch, failure):
    seed(tmp_path)
    def changed():
        (tmp_path / "unselected.py").write_text("changed during model call")
        return verdicts("R1")
    obj = runner(tmp_path, monkeypatch, [verdicts() if failure == "incomplete" else changed],
                 requirements=[{"id": "R1", "text": "Check source"}])
    result = obj.validate_only("exact goal")
    assert result[0]["status"] == "unjudged" and result[0]["fresh"] is False
    assert not (tmp_path / ".spiral/planning-checkpoints/validation_1_0.json").exists()


def test_structured_review_recovery_reports_attempts_without_simulated_deltas():
    from spiral.planner import validate_spec

    cfg = Config()
    cfg.critic = SimpleNamespace(name=cfg.planner.name, think=True)
    calls, attempts, deltas = [], [], []
    class Replies:
        providers = {}
        def chat(self, model, messages, **options):
            calls.append(options)
            if len(calls) == 1:
                return ChatResult("", 30, 2048, raw={"done_reason": "length"})
            options["on_delta"]("answer", "actual streamed delta")
            return ChatResult(json.dumps(verdicts("R1")), 30, 40, raw={"done_reason": "stop"})
    result, _ = validate_spec("goal", [{"id": "R1", "text": "inspect"}], "source",
        cfg=cfg, ol=Replies(), progress=deltas.append, on_attempt=attempts.append)
    assert result[0]["id"] == "R1"
    assert [call["think"] for call in calls] == [True, False]
    assert [a["outcome"] for a in attempts] == ["empty", "parsed"]
    assert deltas == ["answer"]
