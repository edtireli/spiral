"""Declared checks follow requirement dependencies, never task-title keywords."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from spiral.conductor import Conductor
from spiral.planner import Plan, Milestone, Task, bind_requirement_checks, plan_to_dict


def plan():
    return Plan('scope',[Milestone('work',[
        Task('alpha','First contribution',requirements=['R1']),
        Task('beta','Another independent contribution',verify='true',requirements=['R2']),
        Task('gamma','Last contribution',verify='true',requirements=['R1']),
    ])])


def test_only_last_contributor_executes_requirement_check_and_keeps_original_plan(tmp_path):
    original=plan();before=plan_to_dict(original)
    (tmp_path/'accept.py').write_text('raise SystemExit(7)\n')
    import sys,shlex
    command=shlex.quote(sys.executable)+' accept.py'
    bound,bindings=bind_requirement_checks(original,[{'id':'R1','check':command}])
    tasks=bound.milestones[0].tasks
    assert tasks[0].verify=='' and tasks[1].verify=='true'
    assert bindings==[{'requirement':'R1','task':'1.3','command':command}]
    assert plan_to_dict(original)==before
    assert subprocess.run(tasks[2].verify,shell=True,cwd=tmp_path).returncode==7
    (tmp_path/'accept.py').write_text('raise SystemExit(0)\n')
    assert subprocess.run(tasks[2].verify,shell=True,cwd=tmp_path).returncode==0


def test_duplicate_checks_run_once_and_existing_task_check_still_vetoes():
    original=plan();original.milestones[0].tasks[2].requirements.append('R3')
    bound,_=bind_requirement_checks(original,[{'id':'R1','check':'false'},{'id':'R3','check':'false'}])
    assert bound.milestones[0].tasks[2].verify=='(true) && (false)'
    original.milestones[0].tasks[2].verify='false'
    bound,_=bind_requirement_checks(original,[{'id':'R1','check':'false'}])
    assert bound.milestones[0].tasks[2].verify=='false'


@pytest.mark.parametrize('row',[
    {'id':'unknown','check':'python check.py'},
    {'id':'R1','check':'python check.py || true'},
    {'id':'R1','check':True},
])
def test_unmapped_vacuous_or_malformed_checks_cannot_create_green_tasks(row):
    with pytest.raises(ValueError):bind_requirement_checks(plan(),[row])


def test_conductor_binds_only_exact_goal_spec_and_invalidates_old_green_fingerprint(tmp_path):
    runner=object.__new__(Conductor);runner.ws=tmp_path
    runner.ledger=SimpleNamespace(log=lambda *a,**k:None)
    saved=runner._dir();original=plan();before=plan_to_dict(original)
    (saved/'spec.json').write_text(json.dumps([{'id':'R1','check':'python accept.py'}]))
    meta=saved/'spec-meta.json';meta.write_text(json.dumps({'goal_sha256':runner._goal_hash('other goal')}))
    assert runner._bind_saved_requirement_checks(original,'exact goal') is original
    meta.write_text(json.dumps({'goal_sha256':runner._goal_hash('exact goal')}))
    bound=runner._bind_saved_requirement_checks(original,'exact goal')
    first,last=original.milestones[0].tasks[0],original.milestones[0].tasks[2]
    assert runner._task_fingerprint(first)==runner._task_fingerprint(bound.milestones[0].tasks[0])
    assert runner._task_fingerprint(last)!=runner._task_fingerprint(bound.milestones[0].tasks[2])
    runner.state={'task_records':{'1.3':{'status':'green','fingerprint':runner._task_fingerprint(last),'head':'old'}}}
    assert not runner._task_is_resumably_done('1.3',bound.milestones[0].tasks[2])
    assert plan_to_dict(original)==before
