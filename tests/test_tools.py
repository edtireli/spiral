"""Exercises the shell denylist and the run() primitive — the safety floor
under full-auto. A denylist that silently stops blocking is the worst possible
regression, so every entry is pinned here.
Runs standalone (`python tests/test_tools.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.tools import DENY, RunResult, is_dangerous, run  # noqa: E402


def test_every_deny_entry_triggers():
    for bad in DENY:
        assert is_dangerous(bad), f"entry no longer blocks: {bad!r}"
        assert is_dangerous(f"echo hi && {bad} x"), f"embedded form passes: {bad!r}"


def test_denylist_is_case_insensitive():
    assert is_dangerous("RM -RF /tmp/x")
    assert is_dangerous("Sudo Reboot")
    assert is_dangerous("GIT PUSH origin main")


def test_denylist_blocks_network_egress():
    assert is_dangerous("curl http://example.com")
    assert is_dangerous("wget http://example.com/payload.sh")


def test_ordinary_commands_pass():
    for ok in (
        "python -m pytest -q",
        "git commit -m 'msg'",
        "git pull",
        "rm build.log",          # plain rm is allowed; only recursive-force is not
        "echo curling",          # substring must not overmatch
        "ls -la",
    ):
        assert not is_dangerous(ok), f"safe command blocked: {ok!r}"


def test_blocked_command_never_executes():
    with tempfile.TemporaryDirectory() as d:
        # harmless even if the denylist regressed, but the touch proves execution
        r = run("rm -rf ./no-such-dir && touch pwned", d)
        assert r.blocked and r.code == 126 and not r.ok
        assert "denylist" in r.out
        assert not (Path(d) / "pwned").exists(), "blocked command actually ran"


def test_blocked_is_never_ok_even_with_exit_zero():
    assert not RunResult("x", 0, "", blocked=True).ok
    assert RunResult("x", 0, "").ok
    assert not RunResult("x", 1, "").ok


def test_run_captures_output_and_cwd():
    with tempfile.TemporaryDirectory() as d:
        assert run("echo hello", d).out == "hello"
        assert run("touch made.txt", d).ok
        assert (Path(d) / "made.txt").exists()


def test_run_merges_stderr():
    with tempfile.TemporaryDirectory() as d:
        r = run("echo out; echo err 1>&2", d)
        assert "out" in r.out and "err" in r.out


def test_run_timeout():
    with tempfile.TemporaryDirectory() as d:
        r = run("sleep 5", d, timeout=1)
        assert r.code == 124 and not r.ok
        assert "timed out" in r.out


def test_run_streaming_path():
    with tempfile.TemporaryDirectory() as d:
        seen: list[str] = []
        r = run("printf 'a\\nb\\n'", d, on_line=seen.append)
        assert r.ok and r.out == "a\nb"
        assert seen == ["a", "b"]


def test_streaming_path_still_blocked():
    with tempfile.TemporaryDirectory() as d:
        r = run("sudo rm x", d, on_line=lambda _: None)
        assert r.blocked and not r.ok


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
