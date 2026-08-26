"""Offline tests for budgets, leases, evidence DAGs, and self-protection."""
from __future__ import annotations

import fcntl
import errno
import httpx
import json
import os
import threading
import time
import types
from pathlib import Path

import pytest

from spiral.edits import EditBlock, apply_edits
from spiral.command_broker import CommandBroker, _popen_with_headroom_retry
from spiral.config import Config
from spiral.execution import (
    BudgetExceeded, BudgetLimits, EvidenceLevel, EvidenceRecord, RunBudget,
    TaskEvidenceDAG, TaskState,
    OrchestrationPolicy,
)
from spiral.llm import (
    InferenceLease,
    LocalModelStreamStalled,
    OfflineModelAccess,
    Ollama,
    _prompt_token_reserve,
)
from spiral.planner import Milestone, Plan, Task
from spiral.safety_kernel import (
    SafetyBoundaryError,
    protected_boundaries,
    protected_paths,
    protected_relative_path,
)
from spiral.transactions import TaskTransaction


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_command_spawn_retries_transient_mac_headroom_without_model_involvement(
        monkeypatch):
    calls = {"spawn": 0, "checkpoint": 0}
    waits = []
    child = object()

    class Runtime:
        def checkpoint(self):
            calls["checkpoint"] += 1

    def popen(*_args, **_kwargs):
        calls["spawn"] += 1
        if calls["spawn"] < 3:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return child

    monkeypatch.setattr("spiral.command_broker.subprocess.Popen", popen)
    monkeypatch.setattr(
        "spiral.command_broker.time.sleep", lambda delay: waits.append(delay))

    result = _popen_with_headroom_retry(
        ["true"], runtime_control=Runtime(),
        on_wait=lambda attempt, delay, error: waits.append(
            (attempt, delay, error.errno)),
    )

    assert result is child
    assert calls == {"spawn": 3, "checkpoint": 3}
    assert waits == [
        (1, 0.25, errno.EAGAIN), 0.25,
        (2, 0.5, errno.EAGAIN), 0.5,
    ]


def test_joint_budget_is_finite_across_wall_tokens_and_calls():
    clock = Clock()
    budget = RunBudget(BudgetLimits(10, 100, 2), clock=clock)
    assert budget.begin_call(80) == 80
    budget.record(20, 70)
    assert budget.begin_call(80) == 10
    budget.record(5, 5)
    with pytest.raises(BudgetExceeded) as stopped:
        budget.begin_call(1)
    assert stopped.value.dimension == "token"

    wall = RunBudget(BudgetLimits(10, 100, 2), clock=clock)
    clock.now = 11
    with pytest.raises(BudgetExceeded) as stopped:
        wall.begin_call(1)
    assert stopped.value.dimension == "wall"


def test_complexity_tier_sets_all_three_finite_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPIRAL_COMPLEXITY", "quick")
    cfg = Config.load()
    assert (cfg.run_wall_budget_seconds, cfg.run_token_budget, cfg.run_call_budget) == (
        1800, 80_000, 24,
    )
    assert cfg.builder_token_budget > 0


def test_loaded_default_uses_one_local_model_for_every_role(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = Config.load()
    assert cfg.prefer_single_resident_model is True
    assert {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name,
        cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name,
    } == {cfg.worker.name}


def test_malformed_overlay_still_fails_safe_to_one_resident_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "research_search_results_per_query": "not-an-integer",
    }))
    cfg = Config.load()
    assert cfg.prefer_single_resident_model is True
    assert {
        cfg.worker.name, cfg.planner.name, cfg.escalation.name,
        cfg.critic.name, cfg.research_auditor.name, cfg.janitor.name,
    } == {cfg.worker.name}


def test_different_local_critic_requires_explicit_residency_opt_out(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / ".config" / "spiral"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "prefer_single_resident_model": False,
        "models": {"worker": "large:27b", "critic": "small:12b"},
    }))
    cfg = Config.load()
    assert cfg.prefer_single_resident_model is False
    assert cfg.worker.name == "large:27b"
    assert cfg.critic.name == "small:12b"


def test_independent_critic_is_risk_triggered_not_routine():
    policy = OrchestrationPolicy.from_values("standard")
    assert policy.critic_rounds(
        deterministic_defects=0, task_count=6, requested_rounds=3) == 0
    assert policy.critic_rounds(
        deterministic_defects=2, task_count=6, requested_rounds=3) == 1


def test_serial_typed_dag_carries_evidence_and_uncertainty(tmp_path):
    plan = Plan("x", [Milestone("m", [
        Task("compile", "", verify="python -m compileall src"),
        Task("test", "", verify="python -m pytest -q"),
    ])])
    dag = TaskEvidenceDAG.from_plan(plan)
    assert [node.id for node in dag.ready()] == ["1.1"]
    dag.start("1.1")
    dag.finish("1.1", TaskState.COMPLETE, [
        EvidenceRecord(EvidenceLevel.COMPILE, "sources compile"),
    ])
    assert [node.id for node in dag.ready()] == ["1.2"]
    dag.start("1.2")
    dag.finish("1.2", TaskState.COMPLETE, [
        EvidenceRecord(EvidenceLevel.TEST, "tests pass"),
    ])
    report = dag.evidence_report().to_dict()
    assert report["coverage"]["compile"] is True
    assert report["coverage"]["test"] is True
    assert report["coverage"]["human"] is False
    assert report["uncertainty"]["level"] == "medium"
    dag.save(tmp_path / "dag.json")
    assert json.loads((tmp_path / "dag.json").read_text())["execution"] == "serial"


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"content": "done"},
            "prompt_eval_count": 7,
            "eval_count": 3,
            "done_reason": "stop",
        }


def test_offline_mode_is_a_hard_model_transport_boundary(monkeypatch):
    monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")
    client = Ollama(providers={})
    try:
        assert client.models() == []
        with pytest.raises(OfflineModelAccess, match="forbids model-service access"):
            client.chat(
                "must-not-run", [{"role": "user", "content": "do not infer"}],
                num_predict=8,
            )
    finally:
        client.close()


def _can_lock(path: Path) -> bool:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_fake_llm_holds_lease_only_during_inference_and_accounts_call(tmp_path):
    lease_path = tmp_path / "spiral-compute.lease"
    client = Ollama(providers={})
    client.inference_lease = InferenceLease(
        lease_path, owner={"type": "test", "run_id": "r1"}, timeout=1,
    )
    client.configure_budget(wall_seconds=60, total_tokens=300, model_calls=3)
    observed = {}

    def post(_url, json=None, **_kwargs):
        observed["locked"] = not _can_lock(lease_path)
        observed["owner"] = __import__("json").loads(lease_path.read_text())
        observed["num_predict"] = json["options"]["num_predict"]
        return FakeResponse()

    client._client = types.SimpleNamespace(post=post)
    result = client.chat("fake:27b", [{"role": "user", "content": "x"}], num_predict=100)
    assert result.text == "done"
    assert observed["locked"] is True
    assert observed["owner"]["run_id"] == "r1"
    reserve = _prompt_token_reserve([{"role": "user", "content": "x"}])
    assert observed["num_predict"] == min(100, 300 - reserve)
    assert _can_lock(lease_path), "lease leaked beyond the inference response"
    assert client.budget.calls == 1 and client.budget.total_tokens == 10


def test_provider_minimum_is_rejected_before_request_when_ledger_cannot_fit(
        monkeypatch):
    messages = [{"role": "user", "content": "json"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={"kimi-k3": {
        "base_url": "https://api.moonshot.test/v1",
        "api_key_env": "FAKE_PROVIDER_KEY",
    }})
    client.configure_budget(
        wall_seconds=60,
        total_tokens=reserve + 32_767,
        model_calls=3,
    )
    calls = []
    client._client = types.SimpleNamespace(
        post=lambda *_args, **_kwargs: calls.append(_kwargs)
    )
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "offline")

    with pytest.raises(BudgetExceeded, match="requires at least 32768") as stopped:
        client.chat(
            "kimi-k3", messages, think=False, num_predict=8192, fmt="json"
        )
    assert stopped.value.dimension == "token"
    assert calls == []
    assert client.budget.calls == 0


def test_provider_floor_fits_exactly_without_exceeding_total_ledger(monkeypatch):
    messages = [{"role": "user", "content": "json"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={"kimi-k3": {
        "base_url": "https://api.moonshot.test/v1",
        "api_key_env": "FAKE_PROVIDER_KEY",
    }})
    client.configure_budget(
        wall_seconds=60,
        total_tokens=reserve + 32_768,
        model_calls=3,
    )
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [{
                    "message": {"content": "done"}, "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }

    def post(_url, **kwargs):
        captured.update(kwargs["json"])
        captured["_timeout"] = kwargs["timeout"]
        return Response()

    client._client = types.SimpleNamespace(post=post)
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "offline")
    result = client.chat(
        "kimi-k3", messages, think=False, num_predict=8192, fmt="json"
    )
    assert result.text == "done"
    assert captured["max_completion_tokens"] == 32_768
    assert 0 < captured["_timeout"] <= 60
    assert captured["max_completion_tokens"] + reserve <= (
        client.budget.limits.total_tokens
    )


def test_provider_retry_is_recapped_against_unrecorded_first_attempt(monkeypatch):
    messages = [{"role": "user", "content": "answer"}]
    recovery = (
        "The previous provider attempt emitted reasoning but no final answer. "
        "Return the final answer now without restarting the analysis. "
        "Be concise and directly answer the original request."
    )
    recovery_messages = [*messages, {"role": "user", "content": recovery}]
    second_reserve = _prompt_token_reserve(recovery_messages)
    total_limit = 70 + second_reserve + 50
    client = Ollama(providers={"remote": {
        "base_url": "https://provider.test/v1",
        "api_key_env": "FAKE_PROVIDER_KEY",
        "min_completion_tokens": 10,
        "retries": 2,
        "blank_retries": 1,
    }})
    client.configure_budget(
        wall_seconds=60, total_tokens=total_limit, model_calls=3,
    )
    bodies = []

    class Response:
        status_code = 200

        def __init__(self, final):
            self.final = final

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "done" if self.final else ""},
                    "finish_reason": "stop" if self.final else "length",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20 if self.final else 60,
                },
            }

    def post(_url, **kwargs):
        bodies.append(kwargs["json"].copy())
        return Response(final=len(bodies) == 2)

    client._client = types.SimpleNamespace(post=post)
    monkeypatch.setenv("FAKE_PROVIDER_KEY", "offline")
    result = client.chat("remote", messages, num_predict=100)
    assert result.text == "done"
    assert [body["max_tokens"] for body in bodies] == [100, 50]
    assert 70 + second_reserve + bodies[1]["max_tokens"] <= total_limit
    assert client.budget.calls == 2


def test_provider_server_error_retries_instead_of_returning_an_empty_reply(monkeypatch):
    messages = [{"role": "user", "content": "answer"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={"remote": {
        "base_url": "https://provider.test/v1",
        "api_key_env": "FAKE_PROVIDER_KEY",
        "min_completion_tokens": 1,
        "retries": 2,
    }})
    client.configure_budget(
        wall_seconds=60, total_tokens=(reserve + 20) * 3,
        model_calls=3,
    )
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                500, json={"error": {"type": "server_error"}})
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        })

    monkeypatch.setenv("FAKE_PROVIDER_KEY", "offline")
    monkeypatch.setattr("spiral.llm.time.sleep", lambda _delay: None)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = client.chat("remote", messages, num_predict=20)
    finally:
        client.close()

    assert result.text == "done"
    assert len(requests) == 2
    assert client.budget.calls == 2
    assert client.budget.total_tokens == reserve + 20 + 3 + 4


def test_inference_lease_releases_on_cancellation(tmp_path):
    lease_path = tmp_path / "spiral-compute.lease"
    client = Ollama(providers={})
    client.inference_lease = InferenceLease(lease_path, timeout=1)
    client._client = types.SimpleNamespace(
        post=lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        client.chat("fake", [{"role": "user", "content": "x"}], num_predict=5)
    assert _can_lock(lease_path)
    assert client.budget.total_tokens == (
        _prompt_token_reserve([{"role": "user", "content": "x"}]) + 5
    )


def test_blocking_request_timeout_is_bounded_and_failure_is_charged():
    clock = Clock()
    messages = [{"role": "user", "content": "x"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={})
    client.budget = RunBudget(BudgetLimits(10, reserve + 20, 2), clock=clock)
    clock.now = 7
    observed = {}

    def post(_url, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        raise httpx.ReadTimeout("offline timeout")

    client._client = types.SimpleNamespace(post=post)
    with pytest.raises(httpx.ReadTimeout, match="offline timeout"):
        client.chat("fake", messages, num_predict=20)
    assert 0 < observed["timeout"] <= 3
    assert client.budget.total_tokens == reserve + 20
    assert client.budget.total_tokens <= client.budget.limits.total_tokens


def test_local_transport_retry_recovers_and_charges_each_dispatch(monkeypatch):
    messages = [{"role": "user", "content": "finish"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={})
    client.local_model_retry_attempts = 2
    client.configure_budget(
        wall_seconds=60, total_tokens=(reserve + 10) * 3,
        model_calls=3,
    )
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "message": {"content": "done"},
                "prompt_eval_count": 3,
                "eval_count": 4,
                "done_reason": "stop",
            }

    def post(_url, **_kwargs):
        calls.append("post")
        if len(calls) == 1:
            raise httpx.ReadTimeout("local stream stalled")
        return Response()

    monkeypatch.setattr("spiral.llm.time.sleep", lambda _delay: None)
    client._client = types.SimpleNamespace(post=post)
    result = client.chat("local:27b", messages, num_predict=10)

    assert result.text == "done"
    assert result.raw["spiral_local_transport_attempts"] == 2
    assert calls == ["post", "post"]
    assert client.budget.calls == 2
    assert client.budget.total_tokens == reserve + 10 + 3 + 4


def test_local_stream_disconnect_is_retried_without_aborting_the_run(monkeypatch):
    messages = [{"role": "user", "content": "finish"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(providers={})
    client.local_model_retry_attempts = 2
    client.configure_budget(
        wall_seconds=60, total_tokens=(reserve + 10) * 3,
        model_calls=3,
    )
    calls = []

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"message":{"content":"partial"}}\n'
            raise httpx.ReadError("connection reset")

        def close(self):
            return None

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, stream=BrokenStream())
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"done"},"done":true,'
                b'"prompt_eval_count":3,"eval_count":4}\n'
            ),
        )

    monkeypatch.setattr("spiral.llm.time.sleep", lambda _delay: None)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    deltas = []
    try:
        result = client.chat(
            "local:27b", messages, num_predict=10,
            on_delta=lambda kind, piece: deltas.append((kind, piece)),
        )
    finally:
        client.close()

    assert result.text == "done"
    assert result.raw["spiral_local_transport_attempts"] == 2
    assert len(calls) == 2
    assert deltas == [
        ("text", "partial"), ("reset", ""), ("text", "done"),
    ]
    visible_attempt = ""
    for kind, piece in deltas:
        if kind == "reset":
            visible_attempt = ""
        elif kind == "text":
            visible_attempt += piece
    assert visible_attempt == "done", (
        "the failed partial stream must not concatenate with its replay")
    assert client.budget.calls == 2
    assert client.budget.total_tokens == reserve + 10 + 3 + 4


def test_local_stream_stall_after_partial_is_closed_reset_and_retried(
        tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "finish"}]
    reserve = _prompt_token_reserve(messages)
    lease_path = tmp_path / "spiral-compute.lease"
    client = Ollama(timeout=2, providers={})
    client.local_model_retry_attempts = 2
    client.local_stream_stall_seconds = 0.05
    client.inference_lease = InferenceLease(lease_path, timeout=1)
    client.configure_budget(
        wall_seconds=5, total_tokens=(reserve + 10) * 3, model_calls=3,
    )
    calls = []
    stalled_closed = threading.Event()

    class StalledStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"message":{"content":"partial"}}\n'
            stalled_closed.wait(5)

        def close(self):
            stalled_closed.set()

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, stream=StalledStream())
        return httpx.Response(
            200,
            content=(
                b'{"message":{"content":"done"},"done":true,'
                b'"prompt_eval_count":3,"eval_count":4}\n'
            ),
        )

    monkeypatch.setattr("spiral.llm.time.sleep", lambda _delay: None)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    deltas = []
    started = time.monotonic()
    try:
        result = client.chat(
            "local:27b", messages, num_predict=10,
            on_delta=lambda kind, piece: deltas.append((kind, piece)),
        )
    finally:
        client.close()

    assert time.monotonic() - started < 1
    assert stalled_closed.is_set(), "the timed-out response was not closed"
    assert _can_lock(lease_path), "the timed-out stream leaked its inference lease"
    assert result.text == "done"
    assert result.raw["spiral_local_transport_attempts"] == 2
    assert len(calls) == 2
    assert deltas == [
        ("text", "partial"), ("reset", ""), ("text", "done"),
    ]
    assert client.budget.calls == 2
    assert client.budget.total_tokens == reserve + 10 + 3 + 4


def test_local_stream_stall_limit_does_not_cut_off_slow_first_chunk():
    messages = [{"role": "user", "content": "finish"}]
    client = Ollama(timeout=1, providers={})
    client.local_model_retry_attempts = 1
    client.local_stream_stall_seconds = 0.01
    client.configure_budget(
        wall_seconds=3, total_tokens=1_000, model_calls=2,
    )
    calls = []

    class SlowFirstStream(httpx.SyncByteStream):
        def __iter__(self):
            threading.Event().wait(0.08)
            yield (
                b'{"message":{"content":"done"},"done":true,'
                b'"prompt_eval_count":3,"eval_count":4}\n'
            )

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=SlowFirstStream())

    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    deltas = []
    started = time.monotonic()
    try:
        result = client.chat(
            "local:27b", messages, num_predict=10,
            on_delta=lambda kind, piece: deltas.append((kind, piece)),
        )
    finally:
        client.close()

    assert time.monotonic() - started >= 0.06
    assert result.text == "done"
    assert len(calls) == 1
    assert deltas == [("text", "done")]


def test_blank_keepalives_neither_start_nor_extend_progress_deadline():
    messages = [{"role": "user", "content": "finish"}]
    reserve = _prompt_token_reserve(messages)
    client = Ollama(timeout=0.4, providers={})
    client.local_model_retry_attempts = 1
    client.local_stream_stall_seconds = 0.05
    client.configure_budget(
        wall_seconds=1, total_tokens=reserve + 10, model_calls=1,
    )
    closed = threading.Event()

    class BlankKeepaliveStream(httpx.SyncByteStream):
        def __iter__(self):
            # More than one stall interval of blank transport activity must not
            # start the post-progress clock before any model data exists.
            for _ in range(16):
                if closed.wait(0.005):
                    return
                yield b'\n'
            yield b'{"message":{"content":"partial"}}\n'
            # Once real model data arrives, the same keepalives must not refresh
            # its absolute no-progress deadline.
            while not closed.wait(0.005):
                yield b'\n'

        def close(self):
            closed.set()

    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, stream=BlankKeepaliveStream())
    ))
    deltas = []
    started = time.monotonic()
    try:
        with pytest.raises(LocalModelStreamStalled, match="no progress"):
            client.chat(
                "local:27b", messages, num_predict=10,
                on_delta=lambda kind, piece: deltas.append((kind, piece)),
            )
    finally:
        client.close()

    elapsed = time.monotonic() - started
    assert 0.1 <= elapsed < 0.3
    assert closed.is_set(), "blank keepalive stall did not close its response"
    assert deltas == [("text", "partial")]
    assert client.budget.calls == 1
    assert client.budget.total_tokens == reserve + 10


def test_local_stream_stall_exception_is_a_transient_transport_error():
    assert issubclass(LocalModelStreamStalled, httpx.TransportError)


def test_local_stream_stall_bound_is_configurable_but_always_finite(monkeypatch):
    monkeypatch.setenv("SPIRAL_LOCAL_STREAM_STALL_SECONDS", "0.25")
    configured = Ollama(providers={})
    try:
        assert configured.local_stream_stall_seconds == 0.25
    finally:
        configured.close()

    monkeypatch.setenv("SPIRAL_LOCAL_STREAM_STALL_SECONDS", "inf")
    invalid = Ollama(providers={})
    try:
        assert invalid.local_stream_stall_seconds == 240.0
    finally:
        invalid.close()


def test_background_inference_observes_interactive_priority_tickets(tmp_path):
    lease_path = tmp_path / "spiral-compute.lease"
    lease = InferenceLease(lease_path, timeout=1)
    priority = tmp_path / "spiral-compute.priority"
    priority.mkdir()
    ticket = priority / f"interactive-{os.getpid()}-test.json"
    ticket.write_text(json.dumps({"pid": os.getpid(), "created_at": 1}))
    assert lease._interactive_waiting() is True
    ticket.unlink()
    with lease.hold(model="fake"):
        assert not _can_lock(lease_path)
    assert _can_lock(lease_path)


def _spiral_tree(root: Path) -> None:
    (root / "spiral").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='spiral'\n")
    (root / "spiral" / "command_broker.py").write_text("SAFE = True\n")
    (root / "spiral" / "config.py").write_text("LIMIT = 1\n")


def test_managed_edits_need_separate_safety_kernel_capability(tmp_path, monkeypatch):
    _spiral_tree(tmp_path)
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    blocked = apply_edits(tmp_path, [
        EditBlock("spiral/config.py", "LIMIT = 1", "LIMIT = 0"),
    ])[0]
    assert not blocked.ok and "SPIRAL_ALLOW_SAFETY_KERNEL_EDIT" in blocked.reason
    assert "LIMIT = 1" in (tmp_path / "spiral" / "config.py").read_text()

    monkeypatch.setenv("SPIRAL_ALLOW_SAFETY_KERNEL_EDIT", "1")
    allowed = apply_edits(tmp_path, [
        EditBlock("spiral/config.py", "LIMIT = 1", "LIMIT = 2"),
    ])[0]
    assert allowed.ok


def test_managed_boundary_covers_all_runtime_and_package_entry_surfaces(
        tmp_path, monkeypatch):
    _spiral_tree(tmp_path)
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")

    for relative in (
        "spiral/conductor.py",
        "spiral/planner.py",
        "spiral/__init__.py",
        "spiral/future_permission_adapter.py",
        "pyproject.toml",
        "setup.py",
        "sitecustomize.py",
    ):
        assert protected_relative_path(tmp_path, tmp_path / relative) == relative

    boundaries = protected_paths(tmp_path, existing_only=True)
    assert tmp_path / "spiral" in boundaries
    assert tmp_path / "pyproject.toml" in boundaries
    assert protected_relative_path(tmp_path, tmp_path / "README.md") == ""


def test_controller_global_boundaries_survive_source_edit_capability(
        tmp_path, monkeypatch):
    project = tmp_path / "ordinary-project"
    source = tmp_path / "spiral-source"
    helper = tmp_path / "Application Support" / "spiral" / "host"
    plist = tmp_path / "Library" / "LaunchAgents" / "ed.spiral.host.plist"
    project.mkdir()
    source.mkdir()
    helper.mkdir(parents=True)
    plist.parent.mkdir(parents=True)
    plist.write_text("host launch agent")

    records = []
    for path, kind, editable in (
        (source, "tree", True),
        (helper, "tree", False),
        (plist, "file", False),
    ):
        info = path.stat()
        records.append({
            "path": str(path), "kind": kind,
            "dev": info.st_dev, "ino": info.st_ino,
            "editable_with_safety_capability": editable,
        })
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.setenv("SPIRALCHAT_PROTECTED_PATHS", json.dumps(records))
    monkeypatch.setattr("spiral.command_broker.sys.platform", "darwin")
    monkeypatch.setattr(
        "spiral.command_broker.shutil.which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    assert {boundary.path for boundary in protected_boundaries(project)} == {
        source, helper, plist,
    }
    before_argv, before_sandboxed = CommandBroker(project)._argv(
        "python build.py", project, allow_network=True,
        allow_host_read=True, full_access=True,
    )
    assert before_sandboxed is True
    assert f'(subpath "{source}")' in before_argv[2]
    assert f'(subpath "{helper}")' in before_argv[2]
    assert f'(literal "{plist}")' in before_argv[2]

    monkeypatch.setenv("SPIRAL_ALLOW_SAFETY_KERNEL_EDIT", "1")
    assert {boundary.path for boundary in protected_boundaries(project)} == {
        helper, plist,
    }
    assert protected_relative_path(project, source / "spiral" / "llm.py") == ""
    assert protected_relative_path(project, helper / "current" / "spiral-host")

    argv, sandboxed = CommandBroker(project)._argv(
        "python build.py", project, allow_network=True,
        allow_host_read=True, full_access=True,
    )
    assert sandboxed is True
    profile = argv[2]
    assert str(source) not in profile
    assert f'(subpath "{helper}")' in profile
    assert f'(literal "{plist}")' in profile


def test_controller_protected_identity_change_fails_closed(tmp_path, monkeypatch):
    project = tmp_path / "project"
    protected = tmp_path / "protected"
    project.mkdir()
    protected.mkdir()
    info = protected.stat()
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.setenv("SPIRALCHAT_PROTECTED_PATHS", json.dumps([{
        "path": str(protected), "kind": "tree",
        "dev": info.st_dev, "ino": info.st_ino + 1,
    }]))
    with pytest.raises(SafetyBoundaryError, match="identity changed"):
        protected_boundaries(project)


def test_managed_transaction_rejects_new_or_changed_runtime_files(
        tmp_path, monkeypatch):
    _spiral_tree(tmp_path)
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    tx = TaskTransaction.begin(tmp_path, "indirect policy change")
    (tmp_path / "spiral" / "conductor.py").write_text("PERMISSIONS = 'weakened'\n")
    with pytest.raises(RuntimeError, match="runtime boundary"):
        tx.commit("must fail")
    tx.rollback(reason="protected runtime creation")
    assert not (tmp_path / "spiral" / "conductor.py").exists()
    tx.close()


def test_managed_transaction_detects_shell_bypass_and_can_rollback(tmp_path, monkeypatch):
    _spiral_tree(tmp_path)
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    tx = TaskTransaction.begin(tmp_path, "self change")
    (tmp_path / "spiral" / "config.py").write_text("LIMIT = 0\n")
    with pytest.raises(RuntimeError, match="safety kernel"):
        tx.commit("must fail")
    tx.rollback(reason="protected mutation")
    assert (tmp_path / "spiral" / "config.py").read_text() == "LIMIT = 1\n"
    tx.close()


def test_full_access_shell_gets_narrow_kernel_sandbox_in_managed_self_edit(
        tmp_path, monkeypatch):
    _spiral_tree(tmp_path)
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.setattr("spiral.command_broker.sys.platform", "darwin")
    monkeypatch.setattr(
        "spiral.command_broker.shutil.which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )
    argv, sandboxed = CommandBroker(tmp_path)._argv(
        "python build.py", tmp_path, allow_network=True,
        allow_host_read=True, full_access=True,
    )
    profile = argv[2]
    assert sandboxed is True
    assert "(allow default)" in profile
    assert f'(subpath "{tmp_path / "spiral"}")' in profile
    assert f'(literal "{tmp_path / "pyproject.toml"}")' in profile
    assert "deny file-write" in profile
