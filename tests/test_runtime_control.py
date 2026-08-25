"""Offline safety tests for cooperative controller-managed pause/resume."""
from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from spiral.command_broker import CommandBroker
from spiral.execution import BudgetLimits, RunBudget
from spiral.llm import InferenceLease
from spiral.runtime_control import (
    CAPABILITY,
    SCHEMA,
    RuntimeControl,
    _message_mac,
    _replace_runtime_control_for_tests,
)


TOKEN = "offline-test-token-0123456789abcdef"


def _channel(tmp_path: Path) -> tuple[RuntimeControl, Path, Path]:
    private = tmp_path / "control"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    command = private / "command.json"
    ack = private / "ack.json"
    control = RuntimeControl(
        command, ack, TOKEN, "run-offline", 3, poll_seconds=0.01)
    control.start()
    return control, command, ack


def _atomic_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".command.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, json.dumps(payload).encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _command(generation: int, desired: str, **changes) -> dict:
    payload = {
        "schema": SCHEMA,
        "run_id": "run-offline",
        "attempt": 3,
        "generation": generation,
        "request_id": f"request-{generation}",
        "desired": desired,
    }
    payload.update(changes)
    payload["mac"] = _message_mac(TOKEN, payload)
    return payload


def _wait_ack(path: Path, state: str, *, generation: int, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        try:
            last = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.005)
            continue
        if last.get("state") == state and last.get("generation") == generation:
            return last
        time.sleep(0.005)
    raise AssertionError(f"never observed {state=} {generation=}; last ack: {last}")


def test_startup_hello_and_pause_only_at_application_checkpoint(tmp_path: Path) -> None:
    control, command, ack = _channel(tmp_path)
    try:
        hello = json.loads(ack.read_text(encoding="utf-8"))
        assert hello == {
            **hello,
            "schema": SCHEMA,
            "run_id": "run-offline",
            "attempt": 3,
            "generation": 0,
            "request_id": "",
            "desired": "running",
            "state": "running",
            "activity": {"count": 0, "kinds": []},
            "capability": CAPABILITY,
        }
        assert hello["mac"] == _message_mac(
            TOKEN, {key: value for key, value in hello.items() if key != "mac"}
        )
        assert TOKEN not in ack.read_text(encoding="utf-8")
        assert ack.stat().st_mode & 0o777 == 0o600

        _atomic_json(command, _command(1, "paused"))
        observed = _wait_ack(ack, "pausing", generation=1)
        assert observed["activity"] == {"count": 0, "kinds": []}
        # The monitor observes intent but cannot claim an idle-looking interval is
        # a safe point. Only the application thread below may publish paused.
        time.sleep(0.04)
        assert json.loads(ack.read_text())["state"] == "pausing"

        returned = threading.Event()
        waiter = threading.Thread(
            target=lambda: (control.checkpoint(), returned.set()), daemon=True)
        waiter.start()
        paused = _wait_ack(ack, "paused", generation=1)
        assert paused["activity"]["count"] == 0
        assert not returned.is_set()

        _atomic_json(command, _command(2, "running"))
        _wait_ack(ack, "running", generation=2)
        waiter.join(timeout=1)
        assert returned.is_set()
    finally:
        control.close()


def test_active_inference_drains_and_releases_lease_before_paused_ack(
    tmp_path: Path,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    control, command, ack = _channel(tmp_path)
    _replace_runtime_control_for_tests(control)
    lease_path = tmp_path / "model.lease"
    lease = InferenceLease(lease_path, timeout=1, poll=0.01)
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def infer() -> None:
        with lease.hold(model="offline-model"):
            entered.set()
            release.wait(timeout=2)
        finished.set()

    worker = threading.Thread(target=infer, daemon=True)
    worker.start()
    assert entered.wait(timeout=1)
    try:
        _atomic_json(command, _command(1, "paused"))
        pausing = _wait_ack(ack, "pausing", generation=1)
        assert pausing["activity"] == {"count": 1, "kinds": ["inference"]}
        assert not finished.is_set()

        release.set()
        paused = _wait_ack(ack, "paused", generation=1)
        assert paused["activity"] == {"count": 0, "kinds": []}
        assert not finished.is_set()

        # The inference context releases flock before its activity exit can block
        # at the pause checkpoint.
        fd = os.open(lease_path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

        _atomic_json(command, _command(2, "running"))
        _wait_ack(ack, "running", generation=2)
        worker.join(timeout=1)
        assert finished.is_set()
    finally:
        release.set()
        control.cancel()
        worker.join(timeout=1)
        _replace_runtime_control_for_tests(None)


def test_detached_command_finishes_before_pause_and_resume_returns_result(
    tmp_path: Path,
) -> None:
    control, command, ack = _channel(tmp_path)
    _replace_runtime_control_for_tests(control)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    started = threading.Event()
    completed_line = threading.Event()
    result = {}
    code = (
        "import time; print('started', flush=True); time.sleep(.15); "
        "print('completed', flush=True)"
    )
    shell = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    def run() -> None:
        result["value"] = CommandBroker(workspace).run(
            shell,
            timeout=5,
            on_line=lambda line: (
                started.set() if line == "started" else
                completed_line.set() if line == "completed" else None
            ),
            require_sandbox=False,
            full_access=True,
        )

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert started.wait(timeout=2)
    try:
        _atomic_json(command, _command(1, "paused"))
        pausing = _wait_ack(ack, "pausing", generation=1)
        assert pausing["activity"] == {"count": 1, "kinds": ["command"]}
        assert completed_line.wait(timeout=2)
        paused = _wait_ack(ack, "paused", generation=1)
        assert paused["activity"]["count"] == 0
        assert worker.is_alive(), "broker must remain at the safe checkpoint"

        _atomic_json(command, _command(2, "running"))
        _wait_ack(ack, "running", generation=2)
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert result["value"].result.ok
        assert "completed" in result["value"].result.out
    finally:
        control.cancel()
        worker.join(timeout=1)
        _replace_runtime_control_for_tests(None)


def test_wrong_identity_is_ignored_and_cancel_releases_pause_wait(tmp_path: Path) -> None:
    control, command, ack = _channel(tmp_path)
    try:
        forged = _command(1, "paused")
        forged["mac"] = "0" * 64
        _atomic_json(command, forged)
        time.sleep(0.05)
        assert json.loads(ack.read_text())["generation"] == 0

        _atomic_json(command, _command(1, "paused"))
        _wait_ack(ack, "pausing", generation=1)
        returned = threading.Event()
        worker = threading.Thread(
            target=lambda: (control.checkpoint(), returned.set()), daemon=True)
        worker.start()
        _wait_ack(ack, "paused", generation=1)
        control.cancel()
        worker.join(timeout=1)
        assert returned.is_set()

        # Cancellation/KeyboardInterrupt cleanup is not redirected into another
        # pause wait, even though the last authenticated command still says paused.
        with pytest.raises(KeyboardInterrupt):
            with control.activity("cleanup"):
                raise KeyboardInterrupt
    finally:
        control.close()


def test_run_budget_excludes_only_genuine_paused_time() -> None:
    clock = [0.0]
    paused = [0.0]
    budget = RunBudget(
        BudgetLimits(100, 100, 10),
        clock=lambda: clock[0],
        paused_clock=lambda: paused[0],
    )
    clock[0] = 10.0
    assert budget.elapsed_seconds == 10.0
    # Five seconds of the next ten were spent in acknowledged paused state.
    clock[0] = 20.0
    paused[0] = 5.0
    assert budget.elapsed_seconds == 15.0
    assert budget.snapshot()["remaining"]["wall_seconds"] == 85.0
