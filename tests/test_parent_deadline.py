"""A managed child never receives a fresh outer deadline by pausing/retrying."""
import pytest
import copy
import json
import httpx

from spiral.execution import BudgetExceeded, BudgetLimits, RunBudget
from spiral.llm import Ollama


def clocks(monkeypatch, deadline="1030"):
    mono, wall, paused = [50.0], [1000.0], [0.0]
    monkeypatch.setenv("SPIRAL_PARENT_DEADLINE_UNIX", deadline)
    monkeypatch.setattr("spiral.execution.time.time", lambda: wall[0])
    budget = RunBudget(BudgetLimits(7200, 10000, 20), clock=lambda: mono[0], paused_clock=lambda: paused[0])
    return budget, mono, wall, paused


def test_parent_deadline_caps_larger_child_tier(monkeypatch):
    budget, mono, _, _ = clocks(monkeypatch)
    assert budget.snapshot()["remaining"]["wall_seconds"] == 30
    mono[0] += 31
    with pytest.raises(BudgetExceeded) as caught:
        budget.begin_call(8)
    assert caught.value.dimension == "wall"
    assert budget.calls == 0


def test_pause_and_clock_rewind_cannot_extend_captured_parent_deadline(monkeypatch):
    budget, mono, wall, paused = clocks(monkeypatch)
    mono[0] += 20
    paused[0] += 20
    wall[0] -= 100
    assert budget.elapsed_seconds == 0
    assert budget.snapshot()["remaining"]["wall_seconds"] == 10
    mono[0] += 11
    assert budget.exhausted_dimension() == "wall"


def test_reconstructing_child_preserves_absolute_parent_deadline(monkeypatch):
    budget, _, wall, _ = clocks(monkeypatch)
    wall[0] += 25
    restarted = RunBudget(BudgetLimits(7200, 10000, 20), clock=lambda: 900, paused_clock=lambda: 0)
    assert budget.snapshot()["remaining"]["wall_seconds"] == 30
    assert restarted.snapshot()["remaining"]["wall_seconds"] == 5


def test_request_timeout_and_admission_use_the_same_outer_limit(monkeypatch):
    budget, mono, _, _ = clocks(monkeypatch)
    client = Ollama(providers={})
    try:
        client.budget = budget
        assert client._remaining_request_timeout() == 30
        mono[0] += 31
        with pytest.raises(BudgetExceeded):
            client._remaining_request_timeout()
    finally:
        client.close()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "bad"])
def test_invalid_parent_deadline_never_falls_back_to_a_fresh_default(monkeypatch, value):
    with pytest.raises(ValueError, match="parent deadline"):
        clocks(monkeypatch, deadline=value)


def test_absent_parent_deadline_preserves_standalone_budget_and_pause(monkeypatch):
    monkeypatch.delenv("SPIRAL_PARENT_DEADLINE_UNIX", raising=False)
    mono, paused = [10], [0]
    budget = RunBudget(BudgetLimits(100, 10000, 20), clock=lambda: mono[0], paused_clock=lambda: paused[0])
    mono[0] += 30
    paused[0] += 20
    assert budget.snapshot()["remaining"]["wall_seconds"] == 90


def test_managed_model_receives_bounded_budget_tail_without_rewriting_input(monkeypatch):
    budget, _, _, _ = clocks(monkeypatch)
    client = Ollama(providers={})
    original = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "Build the requested project."}]
    before = copy.deepcopy(original)
    requests = []
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(lambda request: (
        requests.append(json.loads(request.content)) or httpx.Response(200, json={
            "done": True, "message": {"content": "fixture"}, "eval_count": 1}))))
    client.budget = budget
    try:
        client.chat("exact:test", original, num_predict=8)
    finally:
        client.close()
    messages = requests[0]["messages"]
    assert messages[0] == original[0]
    assert messages[-1]["content"].startswith(original[-1]["content"])
    assert '"remaining_wall_seconds":30.0' in messages[-1]["content"]
    assert "verification" in messages[-1]["content"]
    assert "not a new request" in messages[-1]["content"]
    assert len(messages[-1]["content"]) - len(original[-1]["content"]) < 1024
    assert original == before


def test_retry_does_not_rewrite_frozen_budget_message_or_relax_deadline(monkeypatch):
    budget, mono, _, _ = clocks(monkeypatch)
    client = Ollama(providers={})
    client._client.close()
    requests = []
    def reply(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            mono[0] += 3
            return httpx.Response(500, json={"error": "fixture"})
        return httpx.Response(200, json={"done": True, "message": {"content": "fixture"}, "eval_count": 1})
    client._client = httpx.Client(transport=httpx.MockTransport(reply))
    client.budget = budget
    try:
        client.chat("exact:test", [{"role": "user", "content": "TASK"}], num_predict=8)
    finally:
        client.close()
    assert len(requests) == 2
    assert requests[0]["messages"] == requests[1]["messages"]
    assert "remaining_wall_seconds" in requests[0]["messages"][-1]["content"]
    assert budget.remaining_wall_seconds == 27
