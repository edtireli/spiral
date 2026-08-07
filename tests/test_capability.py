"""The gap has to be found, closed the safe way, and proven — not assumed.

The failure this exists to prevent: a worker facing a missing package spent six
attempts, three diversity candidates and ninety thousand tokens editing source,
because nothing had ever compared what the build needs against what is installed,
and it never reached for `ASK: install`.

Runs standalone (`python tests/test_capability.py`) or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.capability import (  # noqa: E402
    Need, declare_python, detect_needs, is_present, resolve, write_capabilities,
)


def _ids(needs) -> set[str]:
    return {n.id for n in needs}


def test_domain_words_imply_their_obvious_toolchain():
    assert "python:torch" in _ids(detect_needs("a diffusion image generator"))
    assert "python:praw" in _ids(detect_needs("a Reddit bot that replies"))
    assert "python:pandas" in _ids(detect_needs("analyse this csv analysis task"))
    assert "binary:ffmpeg" in _ids(detect_needs("stitch the frames into a video"))
    assert detect_needs("a static calculator web page") == [], (
        "an ordinary web page needs nothing special; guessing costs an install")


def test_a_fragment_of_a_word_is_not_evidence_of_a_dependency():
    """Bare `word in text` matching invented dependencies out of ordinary English.

    Every hit here was declared in requirements.txt, committed, and then really
    installed against the run's install budget — so "gamers" bought pygame and
    "ratios" bought an Xcode hunt.
    """
    assert detect_needs("a chat app for gamers") == []
    assert detect_needs("report the win ratios") == [], "'ios' inside 'ratios'"
    assert "python:matplotlib" not in _ids(detect_needs("render a flowchart in svg"))
    assert "python:scikit-learn" not in _ids(
        detect_needs("add regression tests for the parser")), (
        "in a coding agent's goals 'regression' means a test, not a model fit")


def test_the_true_positives_survive_the_word_boundary():
    assert "python:praw" in _ids(detect_needs("a Reddit bot that replies"))
    assert "python:pygame" in _ids(detect_needs("a snake game in python"))
    assert "python:matplotlib" in _ids(detect_needs("draw two charts"))
    assert "python:scikit-learn" in _ids(detect_needs("fit a linear regression"))
    assert "python:requests" in _ids(detect_needs("a scraper for job ads"))
    assert "binary:xcodebuild" in _ids(detect_needs("an iOS app"))
    assert "binary:docker" in _ids(detect_needs("add a Dockerfile"))


def test_analyst_tool_families_count_as_evidence_too():
    needs = detect_needs("build the thing", ["flask", "sqlalchemy"])
    assert {"python:flask", "python:sqlalchemy"} <= _ids(needs)


def test_declaring_is_idempotent_and_respects_existing_pins(tmp_path):
    (tmp_path / "requirements.txt").write_text("fastapi==0.110.0\npytest\n")
    added = declare_python(tmp_path, ("fastapi", "torch"))
    assert added == ["torch"], "an existing pin must not be duplicated or rewritten"
    assert "fastapi==0.110.0" in (tmp_path / "requirements.txt").read_text()
    assert declare_python(tmp_path, ("torch",)) == [], "declaring twice adds nothing"


def test_extras_syntax_is_recognised_as_already_declared(tmp_path):
    (tmp_path / "requirements.txt").write_text("uvicorn[standard]\n")
    assert declare_python(tmp_path, ("uvicorn",)) == []


def test_requirements_is_created_when_absent(tmp_path):
    assert declare_python(tmp_path, ("praw",)) == ["praw"]
    assert (tmp_path / "requirements.txt").read_text() == "praw\n"


def test_resolution_declares_python_and_only_reports_binaries(tmp_path):
    outcome = resolve(tmp_path, "a diffusion model, and encode it to video")
    declared = {p for need in outcome.declared for p in need.packages}
    assert {"torch", "diffusers"} <= declared
    assert "torch" in (tmp_path / "requirements.txt").read_text()
    for need in outcome.blocked:
        assert need.kind != "python", (
            "python is declarable, so it must never be reported as blocked")
        assert need.install_hint, "a blocked capability must say how to unblock it"


def test_a_missing_binary_is_reported_not_installed(tmp_path):
    need = Need(id="binary:definitely-not-here", kind="binary",
                binary="definitely-not-here-xyz",
                certificate="command -v definitely-not-here-xyz",
                why="testing", install_hint="brew install nothing")
    assert is_present(tmp_path, need) is False


def test_presence_is_proven_by_running_the_certificate(tmp_path):
    assert is_present(tmp_path, Need(
        id="python:sys", kind="python", packages=("sys",),
        certificate="import sys, json; json.dumps({})")) is True
    assert is_present(tmp_path, Need(
        id="python:nope", kind="python", packages=("nope",),
        certificate="import definitely_not_a_module_xyz")) is False, (
        "'pip said ok' is not evidence — the certificate has to actually run")


def test_the_brief_tells_the_planner_what_it_may_rely_on(tmp_path):
    outcome = resolve(tmp_path, "a Reddit bot")
    brief = outcome.brief()
    assert "praw" in brief
    assert "dependency manifest" in brief


def test_capabilities_are_recorded_for_the_run(tmp_path):
    outcome = resolve(tmp_path, "a Reddit bot")
    path = write_capabilities(tmp_path, outcome)
    assert path.is_file() and "praw" in path.read_text()


if __name__ == "__main__":
    import tempfile

    failures = 0
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    total, ran = len(cases), 0
    for name, fn in cases:
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            ran += 1
            print(f"  ok   {name}")
        except AssertionError as exc:
            ran += 1
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:                    # an error is a failure, not a skip
            ran += 1
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    # report the COUNT, not just the failures: "0 failure(s)" over a subset this
    # runner quietly declined to call reads exactly like a clean suite.
    print(f"\n{ran}/{total} ran · {failures} failure(s)")
    sys.exit(1 if failures or ran != total else 0)
