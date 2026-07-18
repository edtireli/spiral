"""Exercises the signature router — the ledger-fed decision that sends error
classes the worker has never beaten straight to the escalation lane.
Runs standalone (`python tests/test_route.py`) or under pytest.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.route import SigStat, decide, mine, norm_sig  # noqa: E402

WORKER, ESC = "qwen3.6:latest", "qwen3.6:27b"


def _ledger(d: Path, recs: list[dict]) -> Path:
    p = d / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\njunk not json\n")
    return p


def _attempt(model: str, sig: str, exit_code: int, task: str = "t") -> dict:
    return {"kind": "attempt", "model": model, "sig": sig, "verify_exit": exit_code, "task": task}


def test_norm_sig_is_stable_across_lines_and_addresses():
    a = norm_sig("e: MainActivity.kt:116:21 Unresolved reference 'messageText'")
    b = norm_sig("e: MainActivity.kt:98:5 Unresolved reference 'messageText'")
    assert a == b and ":116" not in a
    assert norm_sig("crash at 0xDEADBEEF") == norm_sig("crash at 0x1234")


def test_hard_signature_routes_to_escalation():
    sig = "e: X.kt Unresolved reference 'f'"
    with tempfile.TemporaryDirectory() as d:
        p = _ledger(Path(d), [
            _attempt(WORKER, sig, 1), _attempt(WORKER, sig, 1), _attempt(WORKER, sig, 1),
            _attempt(ESC, sig, 0),
        ])
        stats = mine(p, WORKER, ESC)
    assert decide(sig, stats)
    # and line-number variants of the same mistake hit the same verdict
    assert decide("e: X.kt:44:9 Unresolved reference 'f'", stats)


def test_one_worker_win_keeps_the_fast_lane():
    sig = "error: cannot find symbol"
    with tempfile.TemporaryDirectory() as d:
        p = _ledger(Path(d), [
            _attempt(WORKER, sig, 1), _attempt(WORKER, sig, 1), _attempt(WORKER, sig, 1),
            _attempt(WORKER, sig, 0), _attempt(ESC, sig, 0),
        ])
        assert not decide(sig, mine(p, WORKER, ESC))


def test_too_few_failures_stay_undecided():
    sig = "FAILURE: Build failed"
    with tempfile.TemporaryDirectory() as d:
        p = _ledger(Path(d), [_attempt(WORKER, sig, 1), _attempt(WORKER, sig, 1), _attempt(ESC, sig, 0)])
        assert not decide(sig, mine(p, WORKER, ESC))


def test_escalation_must_have_actually_cleared_it():
    sig = "error: resource linking failed"
    with tempfile.TemporaryDirectory() as d:
        p = _ledger(Path(d), [_attempt(WORKER, sig, 1)] * 5 + [_attempt(ESC, sig, 1)])
        assert not decide(sig, mine(p, WORKER, ESC))


def test_unknown_models_and_junk_are_skipped():
    sig = "error: whatever"
    with tempfile.TemporaryDirectory() as d:
        p = _ledger(Path(d), [
            _attempt("some-old-model", sig, 1),
            {"kind": "plan", "phase": "spec"},
            {"kind": "attempt", "model": WORKER, "verify_exit": 1},  # no sig
        ])
        assert mine(p, WORKER, ESC) == {}


def test_missing_ledger_is_empty():
    assert mine("/nonexistent/ledger.jsonl", WORKER, ESC) == {}
    assert not decide("anything", {})


def test_sigstat_total():
    assert SigStat(worker_fail=2, worker_green=1, esc_fail=1, esc_green=1).total == 5


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  \033[32mPASS\033[0m {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  \033[31mFAIL\033[0m {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
