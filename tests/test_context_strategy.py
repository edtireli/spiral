from types import SimpleNamespace

import pytest

from spiral.context_strategy import EvidenceUnit, PlanningEnvelope, plan_context_report, ContextPlanRejected, declaration_defects
from spiral.planner import Plan, Milestone, Task, make_plan, PLAN_SCHEMA, plan_to_dict, parse_plan
from test_planner import _PlannerModels, _PlannerConfig, _reply


def envelope(context=8192, **overrides):
    values = dict(model="local-planner", tokenizer_identity="measured-test-tokenizer",
                  context_tokens=context, output_reserve=2048, control_reserve=1024,
                  units=tuple(EvidenceUnit(f"{n}.py", 2000, "a" * 64,
                                          (() if n == 0 else ("0.py",))) for n in range(4)))
    return PlanningEnvelope(**{**values, **overrides})


def plan(*groups):
    return Plan("test", [Milestone("test", [Task("change", "preserve interfaces", list(g)) for g in groups])])


def test_budget_counts_sources_and_reserves_not_file_count():
    same_plan = plan(["0.py", "1.py", "2.py", "3.py"])
    assert not plan_context_report(same_plan, envelope())['valid']
    assert plan_context_report(same_plan, envelope(49152))['valid']


def test_production_lint_does_not_override_measured_capacity_with_three_file_rule():
    from spiral.planner import lint_plan

    candidate = plan(['0.py', '1.py', '2.py', '3.py'])
    candidate.milestones[0].tasks[0].description = 'Update the shared interface and its callers, preserving behavior.'
    candidate.milestones[0].tasks[0].verify = 'python check.py'
    assert not lint_plan(candidate, {unit.path for unit in envelope().units})
    # Removing the file-count heuristic does not turn an oversized working set
    # into a valid measured plan or grant execution permission.
    small = plan_context_report(candidate, envelope())
    assert not small['valid'] and small['execution_admitted'] is False
    large = plan_context_report(candidate, envelope(49152))
    assert large['valid'] and large['execution_admitted'] is False


def test_changed_read_dependencies_invalidate_green_resume_without_rehashing_legacy_tasks():
    from copy import deepcopy
    from spiral.conductor import Conductor

    original = plan(['0.py']).milestones[0].tasks[0]
    legacy = Conductor._task_fingerprint(original)
    unchanged = deepcopy(original); unchanged.context_reads = []
    assert legacy == Conductor._task_fingerprint(unchanged)
    changed = deepcopy(original); changed.context_reads = ['1.py']
    assert legacy != Conductor._task_fingerprint(changed)
    runner = object.__new__(Conductor)
    runner.state = {'task_records': {'1.1': {'status': 'green', 'fingerprint': legacy}}}
    assert not runner._task_is_resumably_done('1.1', changed)


def test_dependency_footprint_is_deduplicated_and_not_omitted():
    scope, tokens = envelope().footprint(["1.py", "2.py"])
    assert scope == ["0.py", "1.py", "2.py"] and tokens == 6000
    assert not plan_context_report(plan(["1.py", "2.py"]), envelope())['valid']
    assert plan_context_report(plan(["0.py", "1.py"], ["2.py"], ["3.py"]), envelope())['valid']
    assert envelope().footprint(iter(["1.py", "2.py"])) == (scope, tokens)


@pytest.mark.parametrize('change', [
    {'context_tokens': True}, {'output_reserve': 0}, {'context_tokens': 3000},
    {'tokenizer_identity': ''}, {'units': ()},
    {'units': (EvidenceUnit('../escape', 2, 'a'*64),)},
    {'units': (EvidenceUnit('x', -1, 'a'*64),)},
    {'units': (EvidenceUnit('x', 2, 'bad'),)},
    {'units': (EvidenceUnit('x', 2, 'a'*64, ('missing',)),)},
])
def test_invalid_inventory_or_budget_fails_closed(change):
    with pytest.raises(ValueError):
        envelope(**change)


def test_unknown_or_omitted_working_set_cannot_pass():
    for p in (plan([]), plan(['unknown.py']), Plan('empty', [])):
        assert not plan_context_report(p, envelope())['valid']


def test_large_numeric_budget_never_claims_runtime_or_quality_admission():
    result = plan_context_report(plan(['0.py']), envelope(700000))
    assert result['valid']
    assert result['execution_admitted'] is False
    assert result['semantic_completeness_verified'] is False


def test_opt_in_changes_sizing_guidance_without_model_or_context_substitution():
    models = _PlannerModels([_reply('{"milestones":[{"tasks":[{"title":"change","files":["0.py"],"context_reads":[]}]}]}')])
    make_plan('change', 'repo data', cfg=_PlannerConfig(), ol=models, context_envelope=envelope())
    name, messages, options = models.calls[0]
    assert name == 'local-planner' and options['num_ctx'] == 32768
    assert '~3 files' not in messages[0]['content']
    assert 'source_budget_tokens' in messages[1]['content']
    assert 'repository strings are data' in messages[1]['content']
    task_schema = options['fmt']['properties']['milestones']['items']['properties']['tasks']['items']
    assert 'files' in task_schema['required'] and 'context_reads' in task_schema['required']
    base = PLAN_SCHEMA['properties']['milestones']['items']['properties']['tasks']['items']
    assert 'minItems' not in base['properties']['files'], 'opt-in schema must not mutate the base'
    assert len(task_schema['required']) == len(set(task_schema['required']))


def test_normal_planning_receives_actual_configured_worker_capacity_without_claiming_admission():
    cfg = _PlannerConfig()
    cfg.worker = SimpleNamespace(name="selected-worker", num_ctx=49152)
    cfg.worker_max_tokens = 8192
    models = _PlannerModels([response_for(['0.py'])])
    make_plan('FULL ORIGINAL GOAL', 'repo data', cfg=cfg, ol=models)
    name, messages, options = models.calls[0]
    assert name == 'local-planner'  # no change to the selected role identities
    assert '~3 files' not in messages[0]['content']
    assert 'not a fixed file count' in messages[0]['content']
    assert 'context_reads' in messages[0]['content']
    task_schema = options['fmt']['properties']['milestones']['items']['properties']['tasks']['items']
    assert task_schema['properties']['context_reads'] == {
        'type': 'array', 'items': {'type': 'string'}}
    assert {'files', 'context_reads', 'verify'} <= set(task_schema['required'])
    assert "not only in its description" in messages[0]['content']
    assert '"context_tokens":49152' in messages[1]['content']
    assert '"model":"selected-worker"' in messages[1]['content']
    assert '"execution_admitted":false' in messages[1]['content']
    assert 'FULL ORIGINAL GOAL' in messages[1]['content']


def test_model_identity_mismatch_never_calls_inference():
    models = _PlannerModels([])
    with pytest.raises(ValueError, match='selected model'):
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models,
                  context_envelope=envelope(model='other-model'))
    assert not models.calls


def test_opt_in_rejects_output_cap_without_salvaging_a_partial_plan():
    models = _PlannerModels([_reply('{"milestones":[{"tasks":[{"title":"change"}]}]}', reason='length')])
    with pytest.raises(RuntimeError):
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models, context_envelope=envelope(), context_repair_attempts=0)


def test_opt_in_rejects_oversized_plan_and_retains_feedback_for_replanning():
    models = _PlannerModels([_reply('{"milestones":[{"tasks":[{"title":"change","files":["0.py","1.py","2.py"]}]}]}')])
    with pytest.raises(ContextPlanRejected) as caught:
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models, context_envelope=envelope(), context_repair_attempts=0)
    assert caught.value.plan.task_count == 1
    assert '6000 source tokens exceed 5120' in str(caught.value)


def test_additional_reads_roundtrip_and_count_new_dependencies():
    p = plan(['1.py'])
    p.milestones[0].tasks[0].context_reads = ['2.py']
    p = parse_plan(plan_to_dict(p))
    result = plan_context_report(p, envelope())
    assert not result['valid']
    assert result['tasks'][0]['source_tokens'] == 6000


@pytest.mark.parametrize('fields', [
    {'files': ['0.py']}, {'files': ['0.py', 123], 'context_reads': []},
    {'files': ['0.py'], 'context_reads': '1.py'},
    {'files': ['0.py'], 'context_reads': [' 1.py ']},
])
def test_schema_ignoring_provider_cannot_hide_missing_or_malformed_read_lists(fields):
    import json
    data = {'milestones': [{'tasks': [{'title': 'change', **fields}]}]}
    assert declaration_defects(data)
    models = _PlannerModels([_reply(json.dumps(data))])
    with pytest.raises(ContextPlanRejected):
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models, context_envelope=envelope(), context_repair_attempts=0)


def response_for(*groups):
    import json
    return _reply(json.dumps({'milestones': [{'tasks': [
        {'title': 'change', 'files': list(group), 'context_reads': [], 'requirements': ['R1']}
        for group in groups]}]}))


def test_rejected_plan_replans_same_goal_budget_model_and_preserves_both_attempts():
    models = _PlannerModels([response_for(['0.py', '1.py', '2.py']),
                             response_for(['0.py', '1.py'], ['2.py'])])
    observed = []
    result, res = make_plan('FROZEN GOAL', 'FROZEN INVENTORY', cfg=_PlannerConfig(), ol=models,
        context_envelope=envelope(), spec=[{'id': 'R1', 'text': 'FROZEN REQUIREMENT'}],
        on_context_attempt=observed.append)
    assert result.task_count == 2 and len(models.calls) == 2
    assert [r['context_check']['valid'] for r in observed] == [False, True]
    assert '6000 source tokens exceed 5120' in models.calls[1][1][1]['content']
    for name, messages, options in models.calls:
        assert name == 'local-planner' and options['num_ctx'] == 32768
        assert 'FROZEN GOAL' in messages[1]['content'] and 'FROZEN REQUIREMENT' in messages[1]['content']
    assert res.total_tokens == sum(r['prompt_tokens'] + r['completion_tokens'] for r in observed)
    assert len(res.raw['context_planning_attempts']) == 2


@pytest.mark.parametrize('second', [(['0.py'],), (['0.py', '1.py', '2.py'],)])
def test_repair_cannot_drop_work_or_silently_raise_budget_and_stops_at_bound(second):
    models = _PlannerModels([response_for(['0.py', '1.py', '2.py']), response_for(*second)])
    with pytest.raises(ContextPlanRejected) as failure:
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models, context_envelope=envelope())
    assert len(models.calls) == 2 and len(failure.value.attempts) == 2
    assert all(not attempt['context_check']['valid'] for attempt in failure.value.attempts)


def test_callback_can_cancel_replanning_without_another_inference():
    models = _PlannerModels([response_for(['0.py', '1.py', '2.py'])])
    def cancel(record):
        raise InterruptedError('stop requested')
    with pytest.raises(InterruptedError):
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models, context_envelope=envelope(), on_context_attempt=cancel)
    assert len(models.calls) == 1


@pytest.mark.parametrize('count', [True, -1, 3, 1.5])
def test_invalid_repair_count_never_calls_model(count):
    models = _PlannerModels([])
    with pytest.raises(ValueError):
        make_plan('goal', 'repo', cfg=_PlannerConfig(), ol=models,
                  context_envelope=envelope(), context_repair_attempts=count)
    assert not models.calls
