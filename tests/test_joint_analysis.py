"""Offline checks for opt-in shared analysis; no model/backend/installer calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral import conductor, planner
from spiral.config import Config
from spiral.execution import BudgetExceeded, BudgetLimits, RunBudget
from spiral.planner import DeliverableManifestError
from test_planner import _PlannerConfig, _PlannerModels, _reply, _scope_reply


class AnalysisConfig(_PlannerConfig):
    planner = SimpleNamespace(name="qwen3.8:27b-heretic", think=False)
    planning_analysis_mode = "joint"


def analysis():
    return {
        "requirements": [{"id": "R1", "text": "Run the requested calculation.",
                          "kind": "feature", "check": "python3 verify.py"}],
        "primary_id": "program",
        "deliverables": [{
            "id": "program", "kind": "cli", "description": "Runnable calculation CLI",
            "root_hint": ".", "output_globs": [], "visual": False,
            "interactive": False, "acceptance_evidence": ["Run the requested workflow"],
            "tool_families": ["python-runtime:>=3.11"],
        }],
    }


def test_joint_analysis_and_scope_review_keep_same_model_context_and_budget():
    original = analysis()
    models = _PlannerModels([_reply(json.dumps(original)), _scope_reply(["R1"])])
    goal = "exact objective\n" + "λ" * 9000 + "\nkeep this final commitment"
    records = []
    spec, manifest, _ = planner.analyze_project(
        goal, "untrusted repository instructions", AnalysisConfig(), models,
        on_attempt=records.append)
    assert len(models.calls) == 2
    model, messages, options = models.calls[0]
    assert model == "qwen3.8:27b-heretic"
    assert "GOAL:\n" + goal + "\n\nCURRENT" in messages[1]["content"]
    assert planner.REPOSITORY_DATA_BOUNDARY in messages[0]["content"]
    assert options["num_predict"] == 4096 + 6144
    assert options["num_ctx"] == AnalysisConfig.spec_for(model).num_ctx
    assert options["keep_alive"] == AnalysisConfig.keep_alive
    assert options["think"] is False
    assert options["fmt"] == planner.PROJECT_ANALYSIS_SCHEMA
    assert spec == original["requirements"]
    assert manifest["deliverables"] == original["deliverables"]
    assert records[0]["outcome"] == "parsed"
    assert records[0]["analysis_attempt"] == 1
    assert records[-1]["stage"] == "scope_review"
    assert models.calls[-1][0] == model
    assert models.calls[-1][2]["num_ctx"] == options["num_ctx"]


def test_contract_emission_reserves_output_and_records_each_corrective_call():
    cfg = AnalysisConfig()
    cfg.planner = SimpleNamespace(name="qwen3.8:27b-heretic", think=True)
    good = _reply(json.dumps(analysis()))
    good.raw.update({"prompt_eval_cached_count": 7, "load_duration": 3,
                     "eval_duration": -1, "total_duration": True,
                     "unsafe_text": "must never enter numeric accounting"})
    models = _PlannerModels([_reply("", thinking="private", reason="length"), good,
                             _scope_reply(["R1"])])
    records = []
    planner.analyze_project("Build a CLI", cfg=cfg, ol=models, on_attempt=records.append)
    assert [call[2]["think"] for call in models.calls] == [False, False, False]
    assert [call[2]["num_predict"] for call in models.calls] == [10240, 10240, 2048]
    assert [record["outcome"] for record in records] == ["empty", "parsed", "parsed"]
    assert records[1]["backend_metrics"] == {"prompt_eval_cached_count": 7, "load_duration": 3}
    assert "private" not in json.dumps(records)
    assert "unsafe_text" not in json.dumps(records)
    assert all(record["elapsed_seconds"] >= 0 for record in records)


@pytest.mark.parametrize("mode", ["joint", "sequential"])
def test_complete_contracts_do_not_prepend_a_reasoning_probe_or_mutate_model_policy(mode):
    cfg = AnalysisConfig()
    cfg.planner = SimpleNamespace(name="qwen3.8:27b-heretic", think=True)
    raw = analysis()
    reply = raw if mode == "joint" else {key: value for key, value in raw.items() if key != "requirements"}
    models = _PlannerModels([_reply(json.dumps(reply)), _scope_reply(["R1"])])
    if mode == "joint":
        spec, manifest, _ = planner.analyze_project("Build a CLI", cfg=cfg, ol=models)
        assert spec == raw["requirements"]
    else:
        manifest, _ = planner.analyze_deliverables("Build a CLI", raw["requirements"], cfg=cfg, ol=models)
    assert manifest["deliverables"] == raw["deliverables"]
    assert len(models.calls) == 2
    assert models.calls[0][0] == cfg.planner.name
    assert models.calls[0][2]["think"] is False
    assert cfg.planner.think is True  # Reviews still receive the selected configuration.


@pytest.mark.parametrize("bad", [
    "not JSON", '{"requirements":[{"id":"R1","text":"partial"}],',
    json.dumps({**analysis(), "requirements": []}),
    json.dumps({**analysis(), "requirements": [{"id": "R1", "text": "x"}] * 2}),
    json.dumps({**analysis(), "primary_id": "missing"}),
    json.dumps({**analysis(), "extra": "not in combined schema"}),
])
def test_joint_invalid_output_never_salvages_a_partial_contract(bad):
    models = _PlannerModels([_reply(bad), _reply(bad)])
    with pytest.raises(DeliverableManifestError, match="after two attempts"):
        planner.analyze_project("Build a CLI", cfg=AnalysisConfig(), ol=models)
    assert len(models.calls) == 2


def test_closed_but_length_capped_json_is_not_a_complete_contract():
    models = _PlannerModels([_reply(json.dumps(analysis()), reason="length")] * 2)
    with pytest.raises(DeliverableManifestError):
        planner.analyze_project("Build a CLI", cfg=AnalysisConfig(), ol=models)


@pytest.mark.parametrize("field,value", [
    ("tool_families", ["python:3.11"]), ("visual", "false"),
    ("description", None), ("output_globs", "file.py"), ("kind", []),
])
def test_shared_manifest_validator_rejects_same_defects_in_both_modes(field, value):
    bad = analysis()
    bad["deliverables"][0][field] = value
    assert planner.deliverable_manifest_defects("Build a CLI", bad)
    models = _PlannerModels([_reply(json.dumps(bad)), _reply(json.dumps(bad))])
    with pytest.raises(DeliverableManifestError):
        planner.analyze_project("Build a CLI", cfg=AnalysisConfig(), ol=models)


def test_joint_and_sequential_materialization_share_normalization():
    raw = analysis()
    raw["deliverables"][0]["output_globs"] = ["./dist/result.zip", "src/*.py"]
    original = copy.deepcopy(raw)
    separate = {key: value for key, value in raw.items() if key != "requirements"}
    sequential, _ = planner.analyze_deliverables(
        "Build a CLI", raw["requirements"], cfg=AnalysisConfig(),
        ol=_PlannerModels([_reply(json.dumps(separate)), _scope_reply(["R1"])]))
    _, joint, _ = planner.analyze_project(
        "Build a CLI", cfg=AnalysisConfig(), ol=_PlannerModels([
            _reply(json.dumps(raw)), _scope_reply(["R1"])]))
    assert joint == sequential
    assert raw == original


@pytest.mark.parametrize("mode", ["joint", "sequential"])
def test_numeric_primary_id_cannot_be_coerced_into_a_valid_string(mode):
    raw = analysis()
    raw["primary_id"] = 1
    raw["deliverables"][0]["id"] = "1"
    models = _PlannerModels([_reply(json.dumps(raw)), _reply(json.dumps(raw))])
    with pytest.raises(DeliverableManifestError, match="primary_id"):
        if mode == "joint":
            planner.analyze_project("Build a CLI", cfg=AnalysisConfig(), ol=models)
        else:
            planner.analyze_deliverables("Build a CLI", raw["requirements"],
                                        cfg=AnalysisConfig(), ol=models)
    assert len(models.calls) == 2


def test_budget_exception_propagates_without_new_attempt_or_fallback():
    failure = BudgetExceeded("wall", RunBudget(BudgetLimits(60, 1000, 4)).snapshot())
    class BudgetModels(_PlannerModels):
        def chat(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise failure
    models = BudgetModels([])
    records = []
    with pytest.raises(BudgetExceeded) as error:
        planner.analyze_project("Build a CLI", cfg=AnalysisConfig(), ol=models,
                                on_attempt=records.append)
    assert error.value is failure and len(models.calls) == 1
    assert records[0]["outcome"] == "exception"
    assert records[0]["prompt_tokens"] is None


def runner_fixture(tmp_path, monkeypatch, models):
    runner = object.__new__(conductor.Conductor)
    runner.ws = tmp_path
    runner.cfg = AnalysisConfig()
    runner.ol = models
    runner.c = SimpleNamespace(print=lambda *a, **k: None)
    runner.gate_disp = "none"
    runner.gate = ""
    runner.ledger = SimpleNamespace(log=lambda *a, **k: None, thinking=lambda *a, **k: None)
    runner._raw_goal = lambda goal: goal
    runner._goal_with_design = lambda goal: goal
    runner._project_kind = lambda goal: "cli"
    runner._is_ui = lambda kind: False
    runner._capability_setup_enabled = True
    monkeypatch.setattr(conductor, "build_repomap", lambda *a: "repo")
    monkeypatch.setattr(conductor, "list_files", lambda *a: [])
    monkeypatch.setattr(conductor, "reveal", lambda *a, **k: None)
    return runner


def test_invalid_joint_never_writes_contracts_or_invokes_setup(tmp_path, monkeypatch):
    bad = analysis()
    bad["deliverables"][0]["tool_families"] = ["python:3.11"]
    runner = runner_fixture(tmp_path, monkeypatch,
                            _PlannerModels([_reply(json.dumps(bad))] * 2))
    runner._resolve_capabilities = lambda *a, **k: pytest.fail("setup before valid analysis")
    monkeypatch.setattr(conductor, "extract_spec", lambda *a, **k: pytest.fail("hidden fallback"))
    with pytest.raises(DeliverableManifestError):
        runner.make_plan("Build a CLI")
    for name in ("spec.json", "artifacts.json", "spec-meta.json"):
        assert not (tmp_path / ".spiral" / name).exists()
    assert not (tmp_path / "requirements.txt").exists()


def test_joint_keeps_canonical_tail_and_existing_draft_critic_and_coverage(tmp_path, monkeypatch):
    raw = analysis()
    raw["requirements"] = [{"id": f"R{i}", "text": f"Required behavior {i}", "kind": "feature"}
                           for i in range(1, 45)]
    runner = runner_fixture(tmp_path, monkeypatch, _PlannerModels([
        _reply(json.dumps(raw)), _scope_reply([r["id"] for r in raw["requirements"]])]))
    seen = []
    runner.cfg.critic = SimpleNamespace(name=runner.cfg.planner.name)
    runner.cfg.plan_rounds = 1
    runner.orchestration_policy = SimpleNamespace(critic_rounds=lambda **k: 1)
    runner._resolve_capabilities = lambda *a, **k: seen.append("setup")
    runner._save_plan = lambda *a: seen.append("save")
    def draft(goal, repomap, *args, spec, manifest, **kwargs):
        seen.append("draft")
        saved = json.loads((tmp_path / ".spiral/spec.json").read_text())
        assert spec == saved
        assert spec[:44] == raw["requirements"]
        assert not any(row.get("origin") == "inferred-product-baseline" for row in spec)
        assert manifest == json.loads((tmp_path / ".spiral/artifacts.json").read_text())
        return planner.Plan("Requested tool", [planner.Milestone("Core", [planner.Task(
            "Create program", "Create the full requested reusable program and all required behaviors.",
            requirements=[row["id"] for row in spec])])]), _reply("{}")
    def critic(*args, **kwargs):
        seen.append("critic")
        return "pass", [], _reply("{}")
    monkeypatch.setattr(conductor, "make_plan", draft)
    monkeypatch.setattr(conductor, "critique_plan", critic)
    monkeypatch.setattr(conductor, "lint_contracts", lambda *a: seen.append("contracts") or [])
    result = runner.make_plan("Build a CLI")
    assert result.task_count >= 1
    assert seen.index("setup") < seen.index("draft") < seen.index("critic") < seen.index("save")
    assert seen.count("contracts") == 2
    assert (tmp_path / ".spiral/plan_reviews.json").exists()


@pytest.mark.parametrize("mode", ["joint", "sequential"])
@pytest.mark.parametrize("kind", ["dataset", "web", "cli", "other"])
def test_product_kind_cannot_inject_unrequested_mandatory_requirements(tmp_path, monkeypatch, mode, kind):
    raw = analysis()
    original = copy.deepcopy(raw)
    runner = runner_fixture(tmp_path, monkeypatch, _PlannerModels([]))
    runner.cfg.planning_analysis_mode = mode
    runner._project_kind = lambda goal: kind
    runner._is_ui = lambda value: False  # This test stops before design/worker inference.
    runner._resolve_capabilities = lambda *a, **k: None
    monkeypatch.setattr(conductor, "analyze_project", lambda *a, **k:
                        (copy.deepcopy(raw["requirements"]), copy.deepcopy(raw), _reply("{}")))
    monkeypatch.setattr(conductor, "extract_spec", lambda *a, **k:
                        (copy.deepcopy(raw["requirements"]), _reply("{}")))
    monkeypatch.setattr(conductor, "analyze_deliverables", lambda *a, **k:
                        (copy.deepcopy(raw), _reply("{}")))
    class DraftReached(Exception):
        pass
    def inspect_draft(*args, spec, manifest, **kwargs):
        assert spec[0] == original["requirements"][0]
        assert spec[0]["check"] == "python3 verify.py"
        assert not any(row.get("origin") == "inferred-product-baseline" for row in spec)
        assert any(row.get("deliverable") == "program" for row in spec)
        assert spec == json.loads((tmp_path / ".spiral/spec.json").read_text())
        raise DraftReached()
    monkeypatch.setattr(conductor, "make_plan", inspect_draft)
    with pytest.raises(DraftReached):
        runner.make_plan("Deliver the requested calculation; keep the explicit check.")
    assert raw == original


def test_config_default_overlay_and_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("SPIRAL_PLANNING_ANALYSIS_MODE", raising=False)
    assert Config().planning_analysis_mode == Config.load().planning_analysis_mode == "sequential"
    target = tmp_path / ".config/spiral/config.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"planning_analysis_mode": "joint"}))
    assert Config.load().planning_analysis_mode == "joint"
    monkeypatch.setenv("SPIRAL_PLANNING_ANALYSIS_MODE", "sequential")
    assert Config.load().planning_analysis_mode == "sequential"
    monkeypatch.setenv("SPIRAL_PLANNING_ANALYSIS_MODE", "typo")
    with pytest.raises(ValueError, match="planning_analysis_mode"):
        Config.load()
