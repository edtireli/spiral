"""Generated prompt context must not replace the user's typed project identity."""
import json
from types import SimpleNamespace

import pytest

from spiral import conductor
from spiral.conductor import Conductor, RenderedGoal


@pytest.mark.parametrize("kind", ["dataset", "cli", "library", "other"])
def test_capability_appendix_does_not_invalidate_nonvisual_manifest(tmp_path, kind):
    runner = object.__new__(Conductor)
    runner.ws = tmp_path
    goal = "Compute a reusable summary of these measurements."
    directory = runner._dir()
    (directory / "artifacts.json").write_text(json.dumps({
        "goal_sha256": runner._goal_hash(goal), "primary_id": "result",
        "deliverables": [{"id": "result", "kind": kind, "visual": False}],
    }))
    runner._capability_brief = "An unrelated installed tool can create a desktop app."
    runner.toolsmith = SimpleNamespace(capability_brief=lambda: "")
    enriched = runner._goal_with_design(goal)
    assert "CAPABILITIES FOR THIS BUILD" in enriched
    assert enriched != goal
    assert runner._raw_goal(enriched) == goal
    assert runner._goal_hash(enriched) == runner._goal_hash(goal)
    assert runner._project_kind(enriched) == kind
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    # This real method must return before any transaction, gate, or icon write.
    runner._foundation(SimpleNamespace(print=lambda *args: None), enriched)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
    assert runner._goal_with_design(enriched) == enriched


@pytest.mark.parametrize("marker", [
    "DESIGN SPECIFICATION (implement these decisions literally):",
    "CAPABILITIES FOR THIS BUILD (already resolved by the harness — do not re-install):",
    "CANONICAL PALETTE — generated",
    "CANONICAL DESIGN TOKENS — generated",
    "EMPIRICAL LOCAL TOOL PROFILE (observed):",
])
def test_heading_text_is_not_authority_to_discard_authored_requirements(marker):
    goal = "Keep this exact user constraint."
    authored = goal + "\n\n" + marker + "\nKeep this second constraint too."
    assert Conductor._raw_goal(authored) == authored
    assert Conductor._goal_hash(authored) != Conductor._goal_hash(goal)
    generated = RenderedGoal(authored + "\nGenerated observations", authored)
    assert Conductor._raw_goal(generated) == authored
    assert Conductor._goal_hash(generated) == Conductor._goal_hash(authored)
    assert Conductor._goal_hash(goal) != Conductor._goal_hash(goal + " Changed request.")


def test_validation_load_preserves_saved_contract_without_category_expansion(tmp_path):
    runner = object.__new__(Conductor)
    runner.ws = tmp_path
    goal = "Build a CLI."
    spec = [{"id": "R7", "kind": "feature", "text": "Print a greeting.",
             "check": "python3 verify.py"}]
    directory = runner._dir()
    spec_path = directory / "spec.json"
    saved = json.dumps(spec)
    spec_path.write_text(saved)
    (directory / "spec-meta.json").write_text(json.dumps({
        "goal_sha256": runner._goal_hash(goal)}))
    assert runner._load_spec(goal) == spec
    assert spec_path.read_text() == saved


def test_new_validation_spec_uses_raw_request_without_generic_requirements(tmp_path, monkeypatch):
    runner = object.__new__(Conductor)
    runner.ws = tmp_path
    runner.cfg = object()
    runner.ol = object()
    goal = "Build a CLI."
    seen = []
    spec = [{"id": "R7", "kind": "feature", "text": "Print a greeting.",
             "check": "python3 verify.py"}]
    def extract(actual_goal, *args, **kwargs):
        seen.append(actual_goal)
        return spec, object()
    monkeypatch.setattr(conductor, "extract_spec", extract)
    result = runner._load_spec(RenderedGoal(
        goal + "\n\nCAPABILITIES FOR THIS BUILD (already resolved by the harness):\ninstalled tools", goal))
    assert seen == [goal]
    assert result == spec
    assert json.loads((runner._dir() / "spec.json").read_text()) == spec


def test_serialized_goal_roundtrip_preserves_heading_like_user_text(tmp_path):
    import copy
    from spiral.planner import Plan, Milestone, Task

    runner = object.__new__(Conductor); runner.ws = tmp_path
    runner.toolsmith = SimpleNamespace(capability_brief=lambda: "generated environment notes")
    authored = "First requirement\n\nEMPIRICAL LOCAL TOOL PROFILE\nSecond requirement"
    rendered = runner._goal_context(authored).render()
    assert Conductor._raw_goal(copy.deepcopy(rendered)) == authored
    runner._save_plan(rendered, Plan("work", [Milestone("work", [Task("edit", "preserve constraints")])]))
    saved = json.loads((runner._dir() / "plan.json").read_text())["goal"]
    assert saved == authored and Conductor._raw_goal(saved) == authored
    assert Conductor._goal_hash(saved) == Conductor._goal_hash(rendered)
    # Untyped legacy data cannot prove which prose was authored by the harness.
    legacy = json.loads(json.dumps(str(rendered)))
    assert Conductor._raw_goal(legacy) == legacy
