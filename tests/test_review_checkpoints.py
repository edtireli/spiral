"""Conductor resumes reviews using current parsers and unchanged task budgets."""
import copy
import json
from types import SimpleNamespace

import pytest

from spiral import conductor, planner
from spiral.execution import BudgetExceeded
from spiral.llm import ChatResult
from test_joint_analysis import analysis, runner_fixture
from test_planning_memory import Model


def configured(tmp_path, monkeypatch, replies, *, resume=False):
    raw = analysis()
    model = Model()
    model.replies = list(replies)
    def chat(*args, **kwargs):
        model.calls += 1
        reply = model.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(json.dumps(reply), 10, 20, raw={'done_reason': 'stop'})
    model.chat = chat
    runner = runner_fixture(tmp_path, monkeypatch, model)
    runner.cfg.critic = SimpleNamespace(name=runner.cfg.planner.name, think=False)
    runner.cfg.plan_rounds = 1
    runner.orchestration_policy = SimpleNamespace(critic_rounds=lambda **k: 1)
    runner._resume_planning = resume
    runner._resolve_capabilities = lambda *a, **k: None
    runner._measured_source_inventory = lambda: None
    runner.saved = []
    runner._save_plan = lambda goal, plan: runner.saved.append(plan)
    draft = planner.Plan('Keep the exact goal', [planner.Milestone('Core', [planner.Task(
        'Implement behavior', 'Implement and verify every requirement while preserving existing assertions.',
        files=['app.py'], requirements=['R1'], exports=[])])])
    monkeypatch.setattr(conductor, 'analyze_project', lambda *a, **k:
        (copy.deepcopy(raw['requirements']), copy.deepcopy(raw), ChatResult('{}', 0, 0)))
    monkeypatch.setattr(conductor, 'make_plan', lambda *a, **k: (copy.deepcopy(draft), ChatResult('{}', 0, 0)))
    monkeypatch.setattr(conductor, 'lint_contracts', lambda *a: [])
    return runner, planner.plan_to_dict(draft)


def test_actual_review_and_repair_parsers_replay_through_conductor(tmp_path, monkeypatch):
    first, repaired = configured(tmp_path, monkeypatch, [])
    repaired['understanding'] = 'Full repaired scope'
    first.ol.replies = [{'verdict': 'revise', 'defects': [{'issue': 'Missing edge case', 'fix_hint': 'Keep it'}]}, repaired]
    original = first.make_plan('Build a CLI')
    assert first.ol.calls == 2
    for stage in ('critic_round_1', 'repair_round_1'):
        assert (tmp_path / '.spiral/planning-checkpoints' / (stage + '.json')).is_file()
    resumed, _ = configured(tmp_path, monkeypatch, [], resume=True)
    restored = resumed.make_plan('Build a CLI')
    assert resumed.ol.calls == 0
    assert planner.plan_to_dict(restored) == planner.plan_to_dict(original)
    # Current validation still owns acceptance of a restored model response.
    monkeypatch.setattr(planner, 'parse_plan', lambda *a: (_ for _ in ()).throw(ValueError('new repair validator')))
    checked, _ = configured(tmp_path, monkeypatch, [], resume=True)
    retained = checked.make_plan('Build a CLI')
    assert checked.ol.calls == 0
    assert retained.understanding == 'Keep the exact goal'


@pytest.mark.parametrize('phase', ['critic', 'repair'])
def test_budget_exhaustion_cannot_become_optional_review_failure(tmp_path, monkeypatch, phase):
    runner, _ = configured(tmp_path, monkeypatch, [])
    failure = BudgetExceeded('wall', runner.ol.budget.snapshot())
    runner.ol.replies = ([{'verdict': 'revise', 'defects': [{'issue': 'Missing edge case'}]}]
                        if phase == 'repair' else []) + [failure]
    with pytest.raises(BudgetExceeded) as caught:
        runner.make_plan('Build a CLI')
    assert caught.value is failure
    assert not runner.saved
    stage = 'critic_round_1' if phase == 'critic' else 'repair_round_1'
    assert not (tmp_path / '.spiral/planning-checkpoints' / (stage + '.json')).exists()


def test_invalid_critic_is_not_published_as_completed_checkpoint(tmp_path, monkeypatch):
    runner, _ = configured(tmp_path, monkeypatch, [{'verdict': 'pass', 'defects': [{'no_issue': True}]}])
    runner.make_plan('Build a CLI')
    assert runner.ol.calls == 1
    assert not (tmp_path / '.spiral/planning-checkpoints/critic_round_1.json').exists()
