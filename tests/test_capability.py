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
    Need, declare_node, declare_python, detect_needs, inspect_workspace, is_present,
    manifest_tool_families, resolve, setup_capabilities, write_capabilities,
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


def test_generic_llm_client_does_not_invent_transformers_dependency(tmp_path):
    goals = (
        "Build a CLI around a local LLM exposed through an HTTP endpoint",
        "A language-model CLI that calls the model server's JSON API",
        "Talk to a local LLM through Ollama's HTTP API",
        "Call a hosted Hugging Face model through its HTTP inference API",
    )

    for goal in goals:
        assert "python:transformers" not in _ids(detect_needs(goal)), goal
    assert "python:transformers" not in _ids(
        detect_needs("build the CLI", ["local LLM", "language model API"])
    ), "legacy analyst prose is not a typed Python dependency declaration"
    outcome = resolve(tmp_path, goals[0], ["local LLM"])
    assert "python:transformers" not in _ids(outcome.declared)
    assert not (tmp_path / "requirements.txt").exists(), (
        "generic LLM/API language must not mutate the project manifest"
    )


def test_explicit_local_model_runtime_still_requires_transformers(tmp_path):
    explicit_goals = (
        "Load a Hugging Face checkpoint locally and perform inference",
        "Use the Transformers library to load the language model",
        "Build a Python CLI that locally loads pretrained language-model weights "
        "and performs inference",
    )

    for goal in explicit_goals:
        packages = {
            package for need in detect_needs(goal) for package in need.packages
        }
        assert "transformers" in packages, goal
    assert "python:transformers" in _ids(
        detect_needs("build it", ["python:transformers>=4.44"])
    ), "an explicit typed package requirement remains authoritative"
    outcome = resolve(tmp_path, explicit_goals[0])
    assert "python:transformers" in _ids(outcome.declared)
    assert "transformers" in (tmp_path / "requirements.txt").read_text()


def test_explicit_analyst_model_family_becomes_a_typed_ollama_need():
    needs = detect_needs("build the CLI", [
        "brew:shellcheck", "ollama:qwen3:8b", "python:httpx>=0.27",
        "node:eslint", "binary:xcodebuild",
    ])
    model = next(need for need in needs if need.kind == "model")

    assert model.id == "model:qwen3:8b"
    assert model.setup_request == "ollama qwen3:8b"
    assert model.access == "full-access"
    assert "binary:shellcheck" in _ids(needs)
    assert next(
        need for need in needs if need.id == "binary:shellcheck"
    ).setup_request == "brew shellcheck"
    assert "python:httpx" in _ids(needs)
    assert "node:eslint" in _ids(needs)
    assert "python:matplotlib" not in _ids(detect_needs(
        "build it", ["node:plot"])), (
        "typed package names must not leak into legacy keyword inference")
    assert next(
        need for need in needs if need.id == "binary:xcodebuild"
    ).setup_request == "", "binary: detects but never invents an install source"
    assert not any("third/party" in need.setup_request for need in detect_needs(
        "build it", ["brew:third/party"])), "third-party taps are not typed formulas"
    assert not any(need.kind == "model" for need in detect_needs(
        "build an AI assistant")), "large model pulls must require explicit evidence"


def test_manifest_tool_families_are_flattened_and_deduplicated():
    manifest = {"deliverables": [
        {"tool_families": ["ffmpeg", "ollama:qwen3:8b"]},
        {"tool_families": ["ffmpeg", "fastapi"]},
    ]}

    assert manifest_tool_families(manifest) == [
        "ffmpeg", "ollama:qwen3:8b", "fastapi",
    ]


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


def test_typed_node_dependency_is_durable_and_idempotent(tmp_path):
    assert declare_node(tmp_path, ("@scope/pkg@^2.1", "eslint@9")) == [
        "@scope/pkg@^2.1", "eslint@9",
    ]
    package = __import__("json").loads((tmp_path / "package.json").read_text())
    assert package["dependencies"] == {"@scope/pkg": "^2.1", "eslint": "9"}
    assert declare_node(tmp_path, ("eslint@9",)) == []


def test_typed_node_family_is_declared_before_planning(tmp_path, monkeypatch):
    from spiral import capability

    monkeypatch.setattr(capability, "is_present", lambda *_args: False)
    outcome = resolve(tmp_path, "build it", ["node:commander@^12"])

    assert "node:commander" in _ids(outcome.declared)
    package = __import__("json").loads((tmp_path / "package.json").read_text())
    assert package["dependencies"]["commander"] == "^12"


def test_resolution_declares_python_and_only_reports_binaries(tmp_path):
    outcome = resolve(tmp_path, "a diffusion model, and encode it to video")
    declared = {p for need in outcome.declared for p in need.packages}
    assert {"torch", "diffusers"} <= declared
    assert "torch" in (tmp_path / "requirements.txt").read_text()
    for need in outcome.blocked:
        assert need.kind != "python", (
            "python is declarable, so it must never be reported as blocked")
        assert need.install_hint, "a blocked capability must say how to unblock it"


def test_project_dependency_is_declared_even_when_spiral_host_can_import_it(
        tmp_path, monkeypatch):
    from spiral import capability

    monkeypatch.setattr(capability, "is_present", lambda *_args: True)
    outcome = resolve(tmp_path, "build it", ["python:httpx>=0.27"])

    assert "httpx>=0.27" in (tmp_path / "requirements.txt").read_text()
    assert "python:httpx" in _ids(outcome.declared)


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

    followup = resolve(tmp_path, "encode video")
    followup.setup_reports.append({
        "kind": "project-dependencies", "ok": True, "detail": "inspected",
    })
    write_capabilities(tmp_path, followup)
    recorded = __import__("json").loads(path.read_text())
    assert len(recorded["phases"]) == 2
    assert "praw" in str(recorded["phases"][0]), (
        "the analyst follow-up must not erase goal-only inspect/setup receipts")


def test_inspection_records_existing_design_and_dependency_surface(tmp_path):
    (tmp_path / "package.json").write_text('{"name":"demo"}')
    (tmp_path / "DESIGN.md").write_text("# Existing visual language\n")

    inspection = inspect_workspace(tmp_path)

    assert "package.json" in inspection["dependency_manifests"]
    assert "DESIGN.md" in inspection["design_inputs"]
    assert inspection["project_roots"] == ["."]


def test_host_model_setup_requires_full_access_and_is_recertified(
        tmp_path, monkeypatch):
    from spiral import capability
    from spiral.command_broker import ProvisionOutcome

    acquired = set()

    class Broker:
        calls = []

        def provision_typed(self, request, **kwargs):
            self.calls.append((request, kwargs))
            acquired.add(request)
            return ProvisionOutcome(True, "tool installed: " + request)

    broker = Broker()
    monkeypatch.setattr(
        capability, "is_present",
        lambda _root, need: need.setup_request in acquired,
    )

    denied = setup_capabilities(
        tmp_path, "build it", ["brew:shellcheck", "ollama:qwen3:8b"], broker=broker,
        synchronize_projects=False, full_access=False,
    )
    assert broker.calls == []
    assert _ids(denied.blocked) == {"binary:shellcheck", "model:qwen3:8b"}

    allowed = setup_capabilities(
        tmp_path, "build it", ["brew:shellcheck", "ollama:qwen3:8b"], broker=broker,
        synchronize_projects=False, full_access=True,
    )
    assert {call[0] for call in broker.calls} == {
        "brew shellcheck", "ollama qwen3:8b",
    }
    assert _ids(allowed.acquired) == {
        "binary:shellcheck", "model:qwen3:8b",
    }
    assert allowed.blocked == []


def test_analyst_followup_skips_duplicate_dependency_sync_without_new_declaration(
        tmp_path, monkeypatch):
    from spiral import builder_tools, capability

    calls = []
    monkeypatch.setattr(
        builder_tools, "ensure_builder_dependencies",
        lambda *_args, **_kwargs: calls.append(1) or {
            "applicable": False, "ok": True,
        },
    )

    setup_capabilities(
        tmp_path, "ordinary CLI", ["binary:xcodebuild"],
        synchronize_projects="if-declared", tool_auto=True,
    )
    assert calls == []

    monkeypatch.setattr(capability, "is_present", lambda *_args: False)
    setup_capabilities(
        tmp_path, "ordinary CLI", ["python:httpx>=0.27"],
        synchronize_projects="if-declared", tool_auto=True,
    )
    assert calls == [1]


def test_conductor_stops_before_planning_when_capability_setup_failed(
        tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from spiral import capability
    from spiral.conductor import Conductor
    from spiral.harness_check import HarnessFault

    failed = capability.Resolution()
    failed.inspection = capability.inspect_workspace(tmp_path)
    failed.setup_reports.append({
        "kind": "project-dependencies", "ok": False,
        "failure_kind": "setup", "detail": "No matching distribution pytest-c",
    })
    monkeypatch.setattr(capability, "setup_capabilities", lambda *_args, **_kwargs: failed)

    conductor = object.__new__(Conductor)
    conductor.ws = tmp_path
    conductor.command_broker = None
    conductor.cfg = SimpleNamespace(
        builder_tool_auto=True, builder_full_access=True,
        verify_timeout=30, builder_allow_install_scripts=False,
    )
    conductor.c = SimpleNamespace(print=lambda *_args, **_kwargs: None)

    with pytest.raises(HarnessFault, match="before planning"):
        conductor._resolve_capabilities("build a CLI", setup=True)

    receipt = (tmp_path / ".spiral/capabilities.json").read_text()
    assert "pytest-c" in receipt and '"ok": false' in receipt


def test_conductor_persists_and_stops_on_unexpected_setup_exception(
        tmp_path, monkeypatch):
    import json
    import pytest
    from types import SimpleNamespace
    from spiral import capability
    from spiral.conductor import Conductor
    from spiral.harness_check import HarnessFault

    monkeypatch.setattr(
        capability, "setup_capabilities",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("registry reset")),
    )
    conductor = object.__new__(Conductor)
    conductor.ws = tmp_path
    conductor.command_broker = None
    conductor.cfg = SimpleNamespace(
        builder_tool_auto=True, builder_full_access=True,
        verify_timeout=30, builder_allow_install_scripts=False,
    )
    conductor.c = SimpleNamespace(print=lambda *_args, **_kwargs: None)

    with pytest.raises(HarnessFault, match="failed before planning"):
        conductor._resolve_capabilities("build a CLI", setup=True)

    receipt = json.loads(
        (tmp_path / ".spiral/capabilities.json").read_text())
    setup = receipt["phases"][-1]["setup"][0]
    assert setup["kind"] == "capability-preflight"
    assert setup["ok"] is False
    assert "RuntimeError: registry reset" in setup["detail"]


def test_resume_without_saved_plan_reenables_analyst_capability_setup(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from spiral import conductor as conductor_module
    from spiral.conductor import Conductor

    class StopAfterObservation(Exception):
        pass

    runner = object.__new__(Conductor)
    runner.c = SimpleNamespace(print=lambda *_args, **_kwargs: None)
    runner.state = {"goal": "Build a CLI"}
    runner._preflight = lambda: None
    runner._snapshot = lambda **_kwargs: None
    runner.load_plan = lambda: None
    runner._raw_goal = lambda goal: goal
    observed = []

    def observe(_goal):
        observed.append(runner._capability_setup_enabled)
        raise StopAfterObservation

    runner.make_plan = observe
    monkeypatch.setattr(conductor_module, "runtime_checkpoint", lambda: None)

    with pytest.raises(StopAfterObservation):
        runner.build("Build a CLI", resume=True)
    assert observed == [True]


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
