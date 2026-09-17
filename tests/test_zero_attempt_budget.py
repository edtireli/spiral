"""An observed live zero-budget override must never turn into a default retry."""
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.conductor import Conductor


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **kw: None


@pytest.mark.parametrize('explicit', [True, False])
def test_zero_escalation_stops_actual_worker_after_one_call(tmp_path, monkeypatch, explicit):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path/'module.py').write_text('value=1\n')
    cfg=Config()
    cfg.worker.name=cfg.escalation.name='selected:exact'
    cfg.diversity_samples=0
    if not explicit:
        cfg.escalation_attempts=0
    calls=[]
    atom=Atom(tmp_path,cfg)
    atom.ol=SimpleNamespace(chat=lambda model,messages,**options:
        calls.append((model,options)) or SimpleNamespace(text='',prompt_tokens=1,completion_tokens=1,total_tokens=2))
    atom._run_gate=lambda *a,**kw: SimpleNamespace(ok=False,code=1,out='AssertionError: wrong result')
    runner=object.__new__(Conductor); runner.cfg=cfg
    assert runner._run_task(atom,TaskSpec('repair','python3 check.py',files=['module.py']),Quiet(),
                            attempts=1,esc_attempts=0 if explicit else None)=='blocked'
    assert len(calls)==1 and calls[0][0]=='selected:exact'
    assert atom.run_stats['esc_lanes']==0


@pytest.mark.parametrize('explicit', [True, False])
def test_zero_worker_budget_has_no_model_gate_or_transaction_effects(tmp_path, explicit):
    cfg=Config()
    if not explicit:
        cfg.task_attempt_budget=0
    atom=Atom(tmp_path,cfg)
    def forbidden(*a,**kw):
        pytest.fail('zero budget must not start work')
    atom.ol=SimpleNamespace(chat=forbidden)
    atom._run_gate=forbidden
    before = {str(p.relative_to(tmp_path)): p.read_bytes()
              for p in tmp_path.rglob('*') if p.is_file()}
    before_paths = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')}
    assert not atom.run(TaskSpec('repair','python3 check.py'),attempts=0 if explicit else None,ui=Quiet())
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')} == before_paths
    assert {str(p.relative_to(tmp_path)): p.read_bytes()
            for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('budget', [-1, True, 0.5, '0'])
def test_invalid_lane_limits_do_not_dispatch(tmp_path,budget):
    atom=Atom(tmp_path,Config())
    with pytest.raises(ValueError,match='attempt budget'):
        atom.run(TaskSpec('repair',''),attempts=budget,ui=Quiet())
    runner=object.__new__(Conductor);runner.cfg=Config()
    with pytest.raises(ValueError,match='attempt budget'):
        runner._run_task(atom,TaskSpec('repair',''),Quiet(),esc_attempts=budget)
