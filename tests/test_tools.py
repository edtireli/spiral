"""Exercises the shell denylist and the run() primitive — the safety floor
under full-auto. A denylist that silently stops blocking is the worst possible
regression, so every entry is pinned here.
Runs standalone (`python tests/test_tools.py`) or under pytest.
"""
# Run with pytest. There was once a hand-rolled runner below this point, which
# collected globals() from where it sat mid-file, called each test with no
# arguments, and caught only AssertionError. So it silently skipped every test
# defined after it and every test taking a fixture, then printed "N/N passed".
# A runner that reports a pass count over a subset it chose is the vacuous green
# this suite exists to catch.
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






def test_external_symlink_guard_ignores_spiral_dependency_cache(tmp_path):
    """A dependency-cache venv symlinks bin/python outside the workspace; that must
    not refuse the task commit, while a task-created escaping symlink still does."""
    from spiral.transactions import _external_symlinks

    (tmp_path / ".spiral" / "dependency-cache" / "python" / "venv" / "bin").mkdir(
        parents=True)
    (tmp_path / ".spiral" / "dependency-cache" / "python" / "venv" / "bin"
     / "python").symlink_to("/usr/bin/python3")
    assert _external_symlinks(tmp_path) == []

    (tmp_path / "escape").symlink_to("/etc/passwd")
    assert _external_symlinks(tmp_path) == ["escape"]


def test_sandbox_allows_stat_but_not_content_outside_workspace(tmp_path):
    """Build tools resolve their root by stat-ing ancestors of the workspace; a
    blanket read-deny on $HOME makes that raise and pins the gate red forever."""
    import sys

    if sys.platform != "darwin":
        return
    from spiral.command_broker import CommandBroker
    from spiral.config import Config

    argv, sandboxed = CommandBroker(tmp_path, Config())._argv(
        "true", tmp_path, allow_network=False, allow_host_read=False)
    if not sandboxed:
        return
    profile = argv[argv.index("-p") + 1]
    assert "(allow file-read-metadata)" in profile
    deny = profile.index("(deny file-read* (subpath")
    assert deny < profile.index("(allow file-read-metadata)"), (
        "the metadata allowance must come after the deny to take effect")


def test_skills_are_gated_by_what_the_project_actually_is(tmp_path):
    """Generic words in spiral's own bootstrap prompt ("configuration, resources,
    manifests") scored android-kotlin at exactly min_overlap, so a Python project
    got Kotlin discipline on every task while dependency-medic was filtered out."""
    from spiral.skillpack import load_skills, match_skills, project_ecosystems

    cards = load_skills()
    bootstrap = (
        "The project build is broken. Repair whatever the build gate reports — "
        "configuration, resources, manifests, or source — until it passes."
    )
    python_pick = [c.name for c in match_skills(
        bootstrap, cards, ecosystems={"python", "web"})]
    assert "android-kotlin" not in python_pick
    assert "dependency-medic" in python_pick, (
        "a failing build gate is exactly what dependency-medic is for")
    assert "android-kotlin" in [c.name for c in match_skills(
        bootstrap, cards, ecosystems={"android"})]

    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x = 1\n")
    (tmp_path / "index.html").write_text("<html></html>")
    assert {"python", "web"} <= project_ecosystems(tmp_path)
    assert "android" not in project_ecosystems(tmp_path)


def test_a_stylesheet_task_gets_the_design_discipline():
    """File evidence beats prose: a task editing CSS needs design craft whatever
    its title says."""
    from spiral.skillpack import load_skills, match_skills

    cards = load_skills()
    picked = [c.name for c in match_skills(
        "Style keypad keys.", cards, files=["style.css"], ecosystems={"web"})]
    assert "design-principles" in picked
    assert [c.name for c in match_skills(
        "Implement the SQLite schema.", cards, files=["app/database.py"],
        ecosystems={"python"})] == [], "a data layer needs no design discipline"


def test_a_web_page_is_recognised_as_a_ui(tmp_path):
    """"static web page" and "HTML page" fell through to "other", which skipped the
    design stage, the deterministic foundation, and the visual review for the most
    common ways people phrase a web goal."""
    from spiral.conductor import Conductor

    (tmp_path / ".git").mkdir()
    conductor = Conductor(tmp_path)
    for goal in ("Build a calculator as a single static web page",
                 "Make an HTML page that shows a clock",
                 "A static site for a bakery",
                 "Build a web app dashboard"):
        assert conductor._heuristic_project_kind(goal) == "web", goal
    for goal, expected in (("A python CLI that renames files", "other"),
                           ("A FastAPI service with no interface", "other"),
                           ("An Android timer app", "android")):
        assert conductor._heuristic_project_kind(goal) == expected, goal
