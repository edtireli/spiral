"""A model's completion claim cannot overrule executable evidence."""
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from test_worker_memory import Quiet, reply


@pytest.mark.parametrize('green,missing,answer,expected', [
    (False,False,'ALREADY_DONE',False),
    (True,True,'ALREADY_DONE',False),
    (True,False,'I cannot say ALREADY_DONE because it is incomplete.',False),
    (True,False,'ALREADY_DONE',True),
])
def test_actual_worker_completion_requires_current_gate_and_artifacts(
        tmp_path,monkeypatch,green,missing,answer,expected):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL','1')
    (tmp_path/'module.py').write_text('VALUE = 1\n')
    cfg=Config();cfg.worker.name='exact:model';cfg.diversity_samples=0
    task=TaskSpec('Complete the required behavior','check',
                  ['missing.py' if missing else 'module.py'])
    worker=Atom(tmp_path,cfg)
    worker.ol=SimpleNamespace(chat=lambda *a,**k: reply(answer))
    calls=[]
    def gate(*a,**k):
        calls.append(1)
        return SimpleNamespace(ok=green,code=0 if green else 1,out='' if green else 'AssertionError: requirement unmet')
    worker._run_gate=gate
    assert worker.run(task,attempts=1,diversity=False,ui=Quiet()) is expected
    assert calls and (tmp_path/'module.py').read_text()=='VALUE = 1\n'
