"""Requested changes, not nouns in background context, govern deliverables."""
import copy
import json
import pytest

from spiral import planner
from spiral.execution import BudgetExceeded, BudgetLimits, RunBudget
from test_planner import _PlannerConfig, _PlannerModels, _reply, _scope_reply


def manifest(kind="document"):
    return {"primary_id":"result", "deliverables":[{
        "id":"result", "kind":kind, "description":"Document the actual CLI tests and setup",
        "root_hint":".", "output_globs":["README.md"], "visual":False,
        "interactive":False, "acceptance_evidence":["Compare examples with actual command output"],
        "tool_families":[]}]}


def test_document_change_in_cli_project_passes_scope_review_without_kind_rewrite():
    goal="Correct only README.md for this existing command-line interface; preserve all code."
    spec=[{"id":"R1", "text":goal}]
    proposed=manifest(); before=copy.deepcopy(proposed)
    models=_PlannerModels([_reply(json.dumps(proposed)), _scope_reply(["R1"])])
    result, accounting=planner.analyze_deliverables(goal,spec,cfg=_PlannerConfig(),ol=models)
    assert result["deliverables"] == proposed["deliverables"] == before["deliverables"]
    assert accounting.total_tokens == 60  # Both real adapter responses counted.
    assert len(models.calls)==2
    assert all(c[0]==_PlannerConfig.planner.name for c in models.calls)
    assert accounting.raw["scope_review"]["verdict"]=="accept"


@pytest.mark.parametrize("changes",[
    {"reviewed_requirement_ids":[]}, {"reviewed_requirement_ids":["R1","R1"]},
    {"reviewed_requirement_ids":["invented"]}, {"goal_source_ids":["repo:G1"]},
    {"goal_source_ids":[]}, {"goal_source_ids":["G1","G1"]}, {"goal_source_ids":[{}]},
    {"goal_source_ids":["G2"]}, {"goal_quotes":["requested change"]},
    {"verdict":"unjudged"}, {"verdict":"accept", "missing_requirement_ids":["R1"]},
    {"verdict":"accept", "out_of_scope_deliverable_ids":["result"]},
    {"missing_requirement_ids":[{}]}, {"reason":""},
])
def test_incomplete_or_unsupported_review_cannot_accept_a_manifest(changes):
    raw=json.loads(_scope_reply(["R1"]).text);raw.update(changes)
    defects,_,_=planner.review_deliverable_scope("requested change",[{"id":"R1"}],manifest(),
        _PlannerConfig(),_PlannerModels([_reply(json.dumps(raw))]))
    assert defects


def test_rejected_scope_repair_keeps_exact_goal_and_all_requirements():
    goal="Write a report about the existing library; preserve its source."
    spec=[{"id":"R1","text":"The report"},{"id":"R2","text":"Preserve source"}]
    wrong=manifest("library"); correct=manifest()
    models=_PlannerModels([_reply(json.dumps(wrong)),
        _scope_reply(["R1","R2"],verdict="revise",outside=["result"],
            reason="Requested report, not a new library."),
        _reply(json.dumps(correct)), _scope_reply(["R1","R2"])])
    result,_=planner.analyze_deliverables(goal,spec,cfg=_PlannerConfig(),ol=models)
    assert result["deliverables"][0]["kind"]=="document"
    assert len(models.calls)==4
    for _,messages,_ in models.calls:
        body=messages[-1]["content"]
        assert goal in body and "R1" in body and "R2" in body
    assert "Requested report, not a new library" in models.calls[2][1][0]["content"]


@pytest.mark.parametrize("failure",[RuntimeError("offline"),
    BudgetExceeded("wall",RunBudget(BudgetLimits(60,1000,4)).snapshot())])
def test_unavailable_review_never_falls_back_and_budget_exception_is_preserved(failure):
    class Models(_PlannerModels):
        def chat(self,*a,**kw):
            if self.calls: raise failure
            return super().chat(*a,**kw)
    models=Models([_reply(json.dumps(manifest()))])
    expected=BudgetExceeded if isinstance(failure,BudgetExceeded) else planner.DeliverableManifestError
    with pytest.raises(expected) as caught:
        planner.analyze_deliverables("Write a report",[],cfg=_PlannerConfig(),ol=models)
    if isinstance(failure,BudgetExceeded): assert caught.value is failure


def test_goal_provenance_excludes_runtime_notes_from_review():
    from spiral.conductor import RenderedGoal
    # Runtime text is not a new requested product or quotable user instruction.
    goal=RenderedGoal("Write a report\nRuntime mentions a server", "Write a report")
    models=_PlannerModels([_scope_reply(sources=["G2"])])
    defects,_,_=planner.review_deliverable_scope(goal,[],manifest(),_PlannerConfig(),models)
    assert defects
    supplied=json.loads(models.calls[0][1][-1]["content"])["authored_goal"]
    assert "".join(row["text"] for row in supplied["sources"])=="Write a report"
    assert models.calls[0][2]["fmt"]["properties"]["goal_source_ids"]["items"]["enum"]==["G1"]


@pytest.mark.parametrize("goal", ["A short answer", "🌀 e\u0301\n"*900,
    "{\"delegated_task\":\"first\\nnext\\nlast\"}"*90, "x"*3000], ids=["short", "unicode", "escaped", "unbroken"])
def test_source_pages_are_lossless_and_resolved_from_original_text(goal):
    import hashlib
    sources=planner._scope_goal_sources(goal)
    assert "".join(row["text"] for row in sources)==goal
    assert all(row["text"]==goal[row["start"]:row["end"]] and len(row["text"])<=800
               for row in sources)
    chosen=[sources[-1]["id"]]
    models=_PlannerModels([_scope_reply(sources=chosen)])
    defects,_,review=planner.review_deliverable_scope(goal,[],manifest(),_PlannerConfig(),models)
    assert not defects
    assert review["resolved_goal_evidence"]=={
        "goal_sha256":hashlib.sha256(goal.encode()).hexdigest(),
        "offset_unit":"unicode_characters", "sources":[sources[-1]]}
    supplied=json.loads(models.calls[0][1][-1]["content"])["authored_goal"]
    assert supplied["sources"]==sources
    assert models.calls[0][2]["fmt"]["properties"]["goal_source_ids"]["items"]["enum"]==[
        row["id"] for row in sources]


def test_empty_goal_cannot_be_reviewed_or_dispatch_inference():
    models=_PlannerModels([])
    with pytest.raises(planner.DeliverableManifestError, match="non-empty"):
        planner.review_deliverable_scope("  ",[],manifest(),_PlannerConfig(),models)
    assert not models.calls
