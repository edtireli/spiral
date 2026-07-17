"""Deterministic planner checks — the zero-token ground truth that runs before
any model opinion. Standalone (`python tests/test_planner.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.planner import coverage_gaps, Plan, Milestone, Task  # noqa: E402


def _plan(*tasks: tuple[str, str]) -> Plan:
    return Plan("u", [Milestone("m", [Task(t, d) for t, d in tasks])])


def test_flags_a_forgotten_requirement():
    spec = [
        {"id": "R1", "text": "Messages are encrypted before sending"},
        {"id": "R2", "text": "Export the chat history to a PDF file"},
    ]
    plan = _plan(("Add encryption", "Encrypt each message body with AES before it is sent"))
    gaps = coverage_gaps(spec, plan)
    assert any("R2" in g for g in gaps), gaps          # export/PDF forgotten → flagged
    assert not any("R1" in g for g in gaps), gaps      # encryption covered → silent


def test_no_gaps_when_all_covered():
    spec = [{"id": "R1", "text": "A settings screen toggles dark mode"}]
    plan = _plan(("Settings", "Build a settings screen with a switch that toggles dark mode"))
    assert coverage_gaps(spec, plan) == []


def test_conservative_generic_requirement_not_flagged():
    # a requirement made only of stopwords/generic UI words has no distinctive
    # terms, so it must NOT be flagged (avoid false positives)
    spec = [{"id": "R1", "text": "The user can use the app"}]
    plan = _plan(("Home", "A basic landing area"))
    assert coverage_gaps(spec, plan) == []


def _run():
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
