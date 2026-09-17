"""Explicit declaration semantics; no real model, registry or installer calls."""
import copy
import json
import platform
import sys
from types import SimpleNamespace

import pytest

from spiral import capability, planner
from spiral.prerequisites import PrerequisiteError, parse_families, parse_family


@pytest.mark.parametrize("value", ["3.11", "1.7", "2026.9", "v9.2rc1"])
def test_legacy_version_literals_require_explicit_type(value, tmp_path):
    with pytest.raises(PrerequisiteError, match="ambiguous legacy"):
        capability.resolve(tmp_path, "any objective", ["python:pytest", f"python:{value}"])
    assert not (tmp_path / "requirements.txt").exists()
    # No name blacklist: an explicitly declared numeric distribution remains a
    # registry request, not silently reinterpreted as an interpreter constraint.
    assert parse_family(f"python-package:{value}").kind == "python"


@pytest.mark.parametrize("family", [
    "python-package:thing @ https://example.invalid/archive.whl",
    "python-package:thing\n--index-url=https://example.invalid",
    "python-runtime:3.11", "python-runtime:>=3.11; run()",
    "brew:third/party", "brew:--cask", "binary:$(id)", "node:git+https://x/y",
    "ollama:;echo", "python:foo" + "x" * 160, "", {"certificate": "run code"},
])
def test_invalid_later_declaration_cannot_mutate_or_acquire(family, tmp_path, monkeypatch):
    from spiral import builder_tools
    calls = []
    monkeypatch.setattr(builder_tools, "ensure_builder_dependencies", lambda *a, **k: calls.append(a))
    broker = SimpleNamespace(provision_typed=lambda *a, **k: calls.append(a))
    with pytest.raises(PrerequisiteError):
        capability.setup_capabilities(
            tmp_path, "goal", ["python-package:valid-name", family], broker=broker, full_access=True)
    assert calls == []
    assert not (tmp_path / "requirements.txt").exists()
    assert not (tmp_path / "package.json").exists()


def test_duplicate_conflicting_declarations_fail_before_mutation(tmp_path):
    with pytest.raises(PrerequisiteError, match="conflicting prerequisite"):
        capability.resolve(tmp_path, "goal", ["python:Example>=1", "python-package:example<1"])
    assert not (tmp_path / "requirements.txt").exists()


def test_unambiguous_legacy_declaration_compatibility():
    parsed = parse_families(["python:Example_Pkg>=1", "node:commander@^12", "binary:curl"])
    assert [(item.kind, item.name) for item in parsed] == [
        ("python", "example-pkg"), ("node", "commander"), ("binary", "curl"),
    ]


@pytest.mark.parametrize("specifier,satisfied", [(">=3.0", True), (">=9999.0", False)])
def test_runtime_check_is_exact_existing_interpreter_not_install(specifier, satisfied, tmp_path, monkeypatch):
    monkeypatch.setattr(capability.subprocess, "run", lambda *a, **k: pytest.fail("runtime must not spawn"))
    broker = SimpleNamespace(provision_typed=lambda *a, **k: pytest.fail("runtime must not install"))
    result = capability.setup_capabilities(
        tmp_path, "any objective", [f"python-runtime:{specifier}"],
        synchronize_projects=False, full_access=True, broker=broker)
    need = (result.present if satisfied else result.blocked)[0]
    assert need.kind == "runtime" and need.packages == () and need.setup_request == ""
    assert need.runtime_observation == {
        "executable": sys.executable, "version": platform.python_version(),
        "satisfied": satisfied, "scope": "selected_engine_interpreter_only",
    }
    assert "does not certify a different project venv" in result.brief()
    assert not (tmp_path / "requirements.txt").exists()
    assert result.acquired == []


def test_existing_manifests_still_synchronize_without_goal_inference(tmp_path, monkeypatch):
    from spiral import builder_tools
    original = "# User-authored content\n3.11\nexisting-package==2\n"
    (tmp_path / "requirements.txt").write_text(original)
    calls = []
    monkeypatch.setattr(builder_tools, "ensure_builder_dependencies",
                        lambda *a, **k: calls.append(a) or {"applicable": True, "ok": True})
    result = capability.setup_capabilities(tmp_path, "diffusion reddit video ollama model name:tag")
    assert len(calls) == 1 and result.declared == [] and result.acquired == []
    assert (tmp_path / "requirements.txt").read_text() == original


def test_existing_constraint_difference_blocks_all_new_declarations_without_overwrite(tmp_path):
    original = "Example_pkg==1.0\n"
    (tmp_path / "requirements.txt").write_text(original)
    with pytest.raises(PrerequisiteError, match="reconcile the manifest"):
        capability.resolve(tmp_path, "goal", ["node:new-node", "python-package:a-new-package",
                                                 "python-package:example-pkg>=2"])
    assert (tmp_path / "requirements.txt").read_text() == original
    assert not (tmp_path / "package.json").exists()
    assert capability.declare_python(tmp_path, ("example-pkg",)) == []
    assert capability.declare_python(tmp_path, ("Example.Pkg==1.0",)) == []


@pytest.mark.parametrize("installed,expected", [("1.0", False), ("2.5", True), ("not-a-version", False)])
def test_metadata_presence_checks_requested_version(installed, expected, tmp_path, monkeypatch):
    need = capability.detect_needs("goal", ["python-package:example>=2,<3"])[0]
    seen = []
    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout=installed)
    monkeypatch.setattr(capability.subprocess, "run", run)
    assert capability.is_present(tmp_path, need) is expected
    assert seen[0][1:3] == ["-I", "-c"]
    assert 'm.version("example")' in seen[0][3]


@pytest.mark.parametrize("requirement", ["example[feature]>=1", 'example; python_version >= "3.0"'])
def test_metadata_alone_does_not_certify_extras_or_markers(requirement, tmp_path, monkeypatch):
    need = capability.detect_needs("goal", [f"python-package:{requirement}"])[0]
    monkeypatch.setattr(capability.subprocess, "run", lambda *a, **k: pytest.fail("not a full resolver"))
    assert capability.is_present(tmp_path, need) is False


def test_legacy_artifact_metadata_is_revalidated_without_truncation():
    good = {"deliverables": [{"tool_families": ["python:pytest", "binary:curl"]}]}
    assert capability.manifest_tool_families(good) == ["python:pytest", "binary:curl"]
    for bad in ("python:3.11", "python-package:" + "a" * 161, "flask"):
        with pytest.raises(PrerequisiteError):
            capability.manifest_tool_families({"deliverables": [{"tool_families": [bad]}]})


def manifest(families):
    return {"primary_id": "program", "deliverables": [{
        "id": "program", "kind": "cli", "description": "Requested reusable program",
        "root_hint": ".", "output_globs": [], "visual": False, "interactive": False,
        "acceptance_evidence": ["run the saved program"], "tool_families": families,
    }]}


def test_analyst_corrects_runtime_package_ambiguity_before_acceptance():
    from test_planner import _PlannerConfig, _PlannerModels, _reply, _scope_reply
    bad = manifest(["python:3.11"])
    good = manifest(["python-runtime:>=3.11", "python-package:example>=1"])
    original = copy.deepcopy(bad)
    models = _PlannerModels([_reply(json.dumps(bad)), _reply(json.dumps(good)),
                             _scope_reply()])
    config = _PlannerConfig()
    config.planner = SimpleNamespace(name="planner", think=False)
    result, _ = planner.analyze_deliverables("Build a CLI", [], cfg=config, ol=models)
    assert len(models.calls) == 3
    assert "ambiguous legacy" in models.calls[1][1][0]["content"]
    assert result["deliverables"][0]["tool_families"] == good["deliverables"][0]["tool_families"]
    assert bad == original


def test_twice_invalid_analyst_fails_in_protocol_not_package_install():
    from test_planner import _PlannerConfig, _PlannerModels, _reply
    bad = manifest(["python:2026.9"])
    models = _PlannerModels([_reply(json.dumps(bad)), _reply(json.dumps(bad))])
    config = _PlannerConfig()
    config.planner = SimpleNamespace(name="planner", think=False)
    with pytest.raises(planner.DeliverableManifestError, match="ambiguous legacy"):
        planner.analyze_deliverables("Build a CLI", [], cfg=config, ol=models)
    assert len(models.calls) == 2


def test_conductor_revalidates_before_writing_artifacts_or_setting_up(tmp_path, monkeypatch):
    from spiral import conductor as module
    from test_planner import _reply
    runner = object.__new__(module.Conductor)
    runner.ws = tmp_path
    runner.cfg = SimpleNamespace(planner=SimpleNamespace(name="planner"))
    runner.ol = object()
    runner.gate_disp = "none"
    runner.c = SimpleNamespace(print=lambda *a, **k: None)
    runner.ledger = SimpleNamespace(log=lambda *a, **k: None, thinking=lambda *a, **k: None)
    runner._raw_goal = lambda goal: goal
    runner._capability_setup_enabled = True
    runner._resolve_capabilities = lambda *a, **k: pytest.fail("must not set up invalid declaration")
    monkeypatch.setattr(module, "build_repomap", lambda *a: "")
    monkeypatch.setattr(module, "list_files", lambda *a: [])
    monkeypatch.setattr(module, "extract_spec", lambda *a, **k: ([], _reply("{}")))
    monkeypatch.setattr(module, "analyze_deliverables",
                        lambda *a, **k: (manifest(["python:3.11"]), _reply("{}")))
    with pytest.raises(planner.DeliverableManifestError, match="invalid prerequisite"):
        runner.make_plan("Build a CLI")
    assert not (tmp_path / ".spiral/artifacts.json").exists()
    assert not (tmp_path / "requirements.txt").exists()
