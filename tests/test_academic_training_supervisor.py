from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.academic_finetune.train_qlora import (
    _prior_training_status,
    _resume_command,
    _write_training_status,
)
from scripts.academic_finetune.training_support import (
    CheckpointLedger,
    HarnessError,
    TrainingRunLock,
    TrainingSupervisorJournal,
    plan_training_retry,
    run_training_process,
)


RUN_ID = "a" * 64


def test_signal_crash_restarts_fresh_then_resumes_a_checkpoint() -> None:
    fresh = plan_training_retry(
        -6, retries_used=0, max_retries=5,
        base_delay_seconds=5, maximum_delay_seconds=60,
        checkpoint=None,
    )
    assert fresh.retry is True
    assert fresh.signal_name == "SIGABRT"
    assert fresh.recovery_mode == "fresh_restart"
    assert fresh.delay_seconds == 5

    checkpoint = SimpleNamespace(step=8)
    resumed = plan_training_retry(
        -6, retries_used=3, max_retries=5,
        base_delay_seconds=5, maximum_delay_seconds=60,
        checkpoint=checkpoint,
    )
    assert resumed.retry is True
    assert resumed.recovery_mode == "checkpoint_resume"
    assert resumed.delay_seconds == 40

    capped = plan_training_retry(
        -6, retries_used=4, max_retries=5,
        base_delay_seconds=5, maximum_delay_seconds=60,
        checkpoint=checkpoint,
    )
    assert capped.delay_seconds == 60


def test_retry_policy_fails_closed_for_deterministic_exit_and_budget() -> None:
    deterministic = plan_training_retry(
        2, retries_used=0, max_retries=5,
        base_delay_seconds=5, maximum_delay_seconds=60,
        checkpoint=None,
    )
    assert deterministic.retry is False
    assert deterministic.reason == "non_signal_exit"

    exhausted = plan_training_retry(
        -6, retries_used=5, max_retries=5,
        base_delay_seconds=5, maximum_delay_seconds=60,
        checkpoint=None,
    )
    assert exhausted.retry is False
    assert exhausted.reason == "retry_budget_exhausted"

    with pytest.raises(HarnessError, match="must not exceed 60"):
        plan_training_retry(
            -6, retries_used=0, max_retries=1,
            base_delay_seconds=5, maximum_delay_seconds=61,
            checkpoint=None,
        )


def test_signal_exit_is_not_masked_by_partial_native_checkpoint(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    script = """
import os
import signal
from pathlib import Path

work = Path(os.environ["SPIRAL_TEST_WORK"])
(work / "0000008_adapters.safetensors").write_bytes(b"partial")
os.kill(os.getpid(), signal.SIGABRT)
"""
    # Pass the path as a literal assignment so the helper remains hermetic and
    # does not need to mutate the pytest process environment.
    command = [
        sys.executable, "-c",
        f"import os; os.environ['SPIRAL_TEST_WORK']={str(work)!r};" + script,
    ]
    result = run_training_process(
        command, work, CheckpointLedger(tmp_path / "checkpoints"),
        poll_seconds=0.01,
    )
    assert result == -6


def test_supervisor_journal_is_durable_sequential_and_identity_locked(
    tmp_path: Path,
) -> None:
    journal = TrainingSupervisorJournal(
        tmp_path, run_identity=RUN_ID, session_id="session-one",
        max_retries=5, base_delay_seconds=5, maximum_delay_seconds=60,
    )
    journal.append("session_started", recovery_mode="fresh_restart")
    journal.append(
        "attempt_failed", attempt_id="attempt-one", exit_status=-6,
        automatic_retry=True,
    )
    status = journal.write_status(
        state="retry_wait", retries_used=1, attempt_id="attempt-one")
    assert status["last_event_sequence"] == 2

    records = [
        json.loads(line)
        for line in (tmp_path / "training-supervisor-events.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [1, 2]
    assert all(record["run_identity"] == RUN_ID for record in records)

    resumed = TrainingSupervisorJournal(
        tmp_path, run_identity=RUN_ID, session_id="session-two",
        max_retries=5, base_delay_seconds=5, maximum_delay_seconds=60,
    )
    resumed.append("session_started", recovery_mode="fresh_restart")
    assert json.loads(
        (tmp_path / "training-supervisor-events.jsonl")
        .read_text(encoding="utf-8").splitlines()[-1]
    )["sequence"] == 3

    with pytest.raises(HarnessError, match="identity"):
        TrainingSupervisorJournal(
            tmp_path, run_identity="b" * 64, session_id="wrong-run",
            max_retries=5, base_delay_seconds=5, maximum_delay_seconds=60,
        )


def test_training_run_lock_prevents_duplicate_supervisors(tmp_path: Path) -> None:
    first = TrainingRunLock(tmp_path / "run.lock")
    second = TrainingRunLock(tmp_path / "run.lock")
    first.acquire(owner={"run_identity": RUN_ID})
    try:
        with pytest.raises(HarnessError, match="already supervised"):
            second.acquire(owner={"run_identity": RUN_ID})
    finally:
        first.release()
    second.acquire(owner={"run_identity": RUN_ID})
    second.release()


def test_status_reconciles_step_zero_as_restart_and_command_keeps_supervisor(
    tmp_path: Path,
) -> None:
    run_contract = {
        "run_identity": RUN_ID,
        "resume_semantics": {
            "state": "adapter_weights_only",
            "optimizer_moments_restored": False,
            "rng_state_restored": False,
            "bit_exact": False,
        },
    }
    status = _write_training_status(
        tmp_path, state="failed", run_contract=run_contract,
        schedule={"in_process_validation": False},
        resume_command="resume", total_steps=1200, checkpoint=None,
        attempt_id="failed-at-zero", error="status -6",
    )
    assert status["resume_available"] is False
    assert status["restart_available"] is True
    assert status["recovery_mode"] == "fresh_restart"
    assert _prior_training_status(tmp_path, RUN_ID)["attempt_id"] == "failed-at-zero"

    arguments = SimpleNamespace(
        python="/runtime/python3",
        config=tmp_path / "config.toml",
        data_dir=tmp_path / "data",
        model=tmp_path / "model",
        output=tmp_path / "run",
        model_view_cache=tmp_path / "cache",
        lease_path=tmp_path / "lease",
        ollama_url="http://127.0.0.1:11434",
        safe_schedule=True,
        initialize_from_manifest=None,
        supervise=True,
        max_retries=5,
        retry_base_seconds=5.0,
        retry_max_seconds=60.0,
    )
    command = _resume_command(arguments)
    assert "--supervise" in command
    assert "--max-retries 5" in command
    assert "--resume --execute" in command
