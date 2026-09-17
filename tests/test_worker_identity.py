"""A changing environment must not erase work or relax changed requirements."""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.conductor import Conductor, _worker_task_spec
from spiral.planner import Task
from spiral.worker_memory import WorkerMemory


def test_typed_environment_changes_keep_exact_history_and_fresh_model_context(tmp_path):
    conductor = object.__new__(Conductor)
    conductor.ws = tmp_path
    conductor._capability_brief = "observed runtime A"
    conductor.toolsmith = SimpleNamespace(capability_brief=lambda: "recipe A")
    goal = "Preserve this requirement.\n\nEMPIRICAL LOCAL TOOL PROFILE\nUser-authored constraint."
    planned = Task("Implement", "Keep all requirements", files=["main.py"])
    initial = _worker_task_spec(planned, conductor._goal_context(goal), "check")
    first = WorkerMemory(tmp_path, initial, "selected:exact")
    first.record("verification", summary="prior failure", exit_code=1)
    conductor._capability_brief = "observed runtime B"
    conductor.toolsmith.capability_brief = lambda: "recipe B"
    resumed = _worker_task_spec(planned, conductor._goal_context(goal), "check")
    memory = WorkerMemory(tmp_path, resumed, "selected:exact")
    assert initial.context == resumed.context == goal
    assert initial.runtime_context != resumed.runtime_context
    assert memory.task_id == first.task_id
    history, _ = memory.recent()
    assert "prior failure" in history
    prompt = Atom(tmp_path, Config())._prompt(resumed, [], "fresh failure", "")
    assert goal in prompt and "recipe B" in prompt and "recipe A" not in prompt


@pytest.mark.parametrize("change", [
    {"goal": "different task"}, {"context": "different user constraint"},
    {"verify_cmd": "different verification"}, {"files": ["other.py"]},
    {"exports": ["different interface"]},
])
def test_actual_scope_changes_invalidate_history(tmp_path, change):
    task = TaskSpec("task", "check", ["main.py"], "requirements", ["interface"])
    initial = WorkerMemory(tmp_path, task, "selected:exact")
    changed = WorkerMemory(tmp_path, replace(task, **change), "selected:exact")
    assert changed.task_id != initial.task_id
    assert WorkerMemory(tmp_path, task, "selected:other").task_id != initial.task_id


def test_design_changes_invalidate_scope_and_legacy_identity_is_preserved(tmp_path):
    runner = object.__new__(Conductor); runner.ws = tmp_path
    runner.toolsmith = SimpleNamespace(capability_brief=lambda: "current tools")
    task = Task("Implement", "Design", files=["main.py"])
    (runner._dir() / "design.md").write_text("Use the agreed interface")
    before = _worker_task_spec(task, runner._goal_context("requirements"), "check")
    (runner._dir() / "design.md").write_text("Use the revised interface")
    after = _worker_task_spec(task, runner._goal_context("requirements"), "check")
    assert WorkerMemory(tmp_path, before, "exact").task_id != WorkerMemory(tmp_path, after, "exact").task_id
    legacy_task = TaskSpec("task", "check", ["main.py"], "all legacy context")
    old_fields = asdict(legacy_task); old_fields.pop("runtime_context")
    old_scope = {"schema": "spiral.worker-scope.v1", "task": old_fields, "model": "exact"}
    expected = hashlib.sha256(json.dumps(old_scope, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert WorkerMemory(tmp_path, legacy_task, "exact").task_id == expected
