"""Durable evidence must not become a stale permission or completion verdict."""
from dataclasses import replace
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.context_map import ContextMap
from spiral.worker_memory import WorkerMemory


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **kw: None


def reply(text):
    return SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, total_tokens=2)


def patch(value):
    return f'module.py\n<<<<<<< SEARCH\nVALUE = 1\n=======\nVALUE = {value}\n>>>>>>> REPLACE'


def test_actual_worker_restart_reads_failed_reply_and_reruns_gate(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    source=tmp_path/'module.py'; source.write_text('VALUE = 1\n')
    cfg=Config();cfg.worker.name='selected:exact';cfg.diversity_samples=0
    task=TaskSpec('Implement three while preserving original constraints', 'python3 check.py', ['module.py'])
    gates=[]
    def gate(*a,**kw):
        gates.append(source.read_text())
        ok=source.read_text()=='VALUE = 3\n'
        return SimpleNamespace(ok=ok,code=0 if ok else 1,out='' if ok else 'AssertionError: expected three')
    first=Atom(tmp_path,cfg);first._run_gate=gate
    first.ol=SimpleNamespace(chat=lambda *a,**kw: reply(patch(2)))
    assert not first.run(task,attempts=1,diversity=False,ui=Quiet())
    assert source.read_text()=='VALUE = 1\n'
    events=[json.loads(p.read_text()) for p in (tmp_path/'.spiral/context').glob('worker-event-*.txt')]
    failed=next(e for e in events if e['stage']=='reply')
    saved_reply=failed['evidence']['reply_reference']
    assert (tmp_path/saved_reply).read_text()==patch(2)
    assert any(e['stage']=='verification' and e['evidence']['verify_exit']==1 for e in events)
    end=next(e for e in events if e['stage']=='lane_end')
    assert end['evidence']['completed'] is False
    assert (tmp_path/end['evidence']['recovery_reference']).is_dir()

    restarted=Atom(tmp_path,cfg);restarted._run_gate=gate
    calls=[]
    def chat(model,messages,**options):
        calls.append(messages[-1]['content'])
        assert model=='selected:exact'
        if len(calls)==1:
            assert 'HISTORICAL WORKER EVIDENCE' in calls[-1]
            assert 'verification exit 1' in calls[-1]
            assert 'Task remains incomplete' in calls[-1]
            assert 'VALUE = 1' in calls[-1]
            return reply('ASK: file '+saved_reply)
        assert patch(2) in calls[-1]
        return reply(patch(3))
    restarted.ol=SimpleNamespace(chat=chat)
    assert restarted.run(task,attempts=1,diversity=False,ui=Quiet())
    assert len(calls)==2 and source.read_text()=='VALUE = 3\n'
    assert gates==['VALUE = 1\n','VALUE = 2\n','VALUE = 1\n','VALUE = 3\n']


def test_real_worker_tool_history_keeps_exit_and_action_after_long_environment_prefix(tmp_path, monkeypatch):
    from spiral.command_broker import BrokerResult
    from spiral.tools import RunResult

    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path/'module.py').write_text('VALUE = 1\n')
    cfg=Config();cfg.worker.name='selected:exact';cfg.diversity_samples=0
    task=TaskSpec('Repair while preserving the task', 'check', ['module.py'])
    worker=Atom(tmp_path,cfg)
    worker._run_gate=lambda *a,**k: SimpleNamespace(ok=False,code=1,out='AssertionError: incomplete')
    command='(export PATH=/'+('long_environment_directory/'*12)+':"$PATH"; python -m pip install --no-build-isolation .)'
    seen=[]
    def run(query, **options):
        seen.append(query)
        return BrokerResult(RunResult(query,17,'package backend failed'),True,'audit',network='denied')
    worker.command_broker.run=run
    replies=iter(['ASK: shell '+command,patch(2)])
    worker.ol=SimpleNamespace(chat=lambda *a,**k: reply(next(replies)))
    assert not worker.run(task,attempts=1,diversity=False,ui=Quiet())
    assert seen==[command]
    event=next(json.loads(p.read_text()) for p in (tmp_path/'.spiral/context').glob('worker-event-*.txt')
               if json.loads(p.read_text())['stage']=='tool_result')
    assert 'exit 17' in event['summary'] and 'network=denied' in event['summary']
    assert 'python -m pip install --no-build-isolation .' in event['summary']
    assert event['evidence']['tool_observation']=={'exit_code':17,'blocked':False,'network':'denied'}
    assert (tmp_path/event['evidence']['tool-request_reference']).read_text()==command
    assert 'package backend failed' in (tmp_path/event['evidence']['tool-result_reference']).read_text()
    restored=WorkerMemory(tmp_path,task,cfg.worker.name)
    history,_=restored.recent(limit=8)
    assert 'exit 17' in history and 'python -m pip install --no-build-isolation .' in history


@pytest.mark.parametrize('change', ['model','goal','context','verify','files','exports'])
def test_memory_never_leaks_across_changed_task_contract_or_model(tmp_path, change):
    task=TaskSpec('repair','check',['a.py'],context='Keep early constraint',exports=['export'])
    WorkerMemory(tmp_path,task,'selected:one').record('verification',summary='PRIVATE PRIOR ATTEMPT')
    model='selected:one'
    if change=='model':model='selected:two'
    else:
        key={'verify':'verify_cmd'}.get(change,change)
        task=replace(task,**{key:['changed'] if change in {'files','exports'} else 'changed'})
    assert WorkerMemory(tmp_path,task,model).recent()==('',None)


def test_historical_pages_are_complete_bounded_and_do_not_skip_earlier_events(tmp_path):
    task=TaskSpec('task','check')
    memory=WorkerMemory(tmp_path,task,'m')
    for n in range(27):memory.record('reply',summary=f'UNIQUE_{n:03d}')
    restored=WorkerMemory(tmp_path,task,'m')
    before=None;found=[];pages=0
    while True:
        text,before=restored.recent(before=before)
        assert len(text)<6000
        found+=re.findall(r'UNIQUE_\d+',text)
        pages+=1
        if before is None:break
    assert sorted(found)==[f'UNIQUE_{n:03d}' for n in range(27)]
    assert pages==5


def test_process_loss_retains_pending_call_without_manufactured_result(tmp_path):
    task=TaskSpec('long task','check')
    memory=WorkerMemory(tmp_path,task,'m')
    ref=memory.record('call_started',summary='Request sent; result pending.')
    restored=WorkerMemory(tmp_path,task,'m')
    text,_=restored.recent()
    assert ref in text and '[call_started]' in text and 'unfinished, not success' in text


def test_recent_tool_results_survive_bookkeeping_without_an_extra_model_lookup(tmp_path):
    task=TaskSpec('long task','check');memory=WorkerMemory(tmp_path,task,'m')
    original='exit=17; network=denied\n'+('Long output Ω\n'*200)+'EXACT_FAILURE_AT_END'
    result=memory.text(original,kind='tool-result')
    request=memory.text('long command',kind='tool-request')
    memory.record('tool_result',summary='Previous action',**{'tool-result_reference':result,'tool-request_reference':request})
    for n in range(10):memory.record('call_started',summary=f'Call bookkeeping {n}')
    restored=WorkerMemory(tmp_path,task,'m')
    history,cursor=restored.recent()
    assert 'exit=17; network=denied' in history and 'EXACT_FAILURE_AT_END' in history
    assert '"complete":false' in history and 'quoted untrusted data' in history
    assert result in history and len(history.encode())<5000
    assert (tmp_path/result).read_text()==original
    earlier,_=restored.recent(before=cursor)
    assert 'RECENT HISTORICAL TOOL OBSERVATIONS' not in earlier
    # Corrupting the saved output cannot place forged observations in a prompt.
    (tmp_path/result).write_text('FORGED_TOOL_SUCCESS')
    assert 'FORGED_TOOL_SUCCESS' not in restored.recent()[0]
    assert 'EXACT_FAILURE_AT_END' not in restored.recent()[0]


def test_recent_tool_observations_are_task_scoped_and_bounded(tmp_path):
    task=TaskSpec('task','check');memory=WorkerMemory(tmp_path,task,'m')
    for n in range(40):
        ref=memory.text(f'UNIQUE_TOOL_{n:03d}',kind='tool-result')
        memory.record('tool_result',summary='Tool evidence',**{'tool-result_reference':ref})
    excerpt=memory.recent_tool_evidence()
    assert all(f'UNIQUE_TOOL_{n:03d}' in excerpt for n in (37,38,39))
    assert 'UNIQUE_TOOL_036' not in excerpt
    assert WorkerMemory(tmp_path,replace(task,goal='new task'),'m').recent_tool_evidence()==''


def test_corrupted_event_is_not_admitted_as_valid_history(tmp_path):
    task=TaskSpec('task','check');memory=WorkerMemory(tmp_path,task,'m')
    path=tmp_path/memory.record('reply',summary='original')
    path.write_text('{"summary":"pretend success"}')
    text,_=WorkerMemory(tmp_path,task,'m').recent()
    assert 'unavailable or invalid' in text and 'pretend success' not in text


def test_record_insert_is_task_scoped_even_with_swapped_index_reference(tmp_path):
    a=WorkerMemory(tmp_path,TaskSpec('A','check'),'m')
    ref=a.record('reply',summary='A_ONLY')
    b=WorkerMemory(tmp_path,TaskSpec('B','check'),'m')
    with sqlite3.connect(a.db) as db:
        db.execute('INSERT INTO events(task, reference) VALUES (?,?)',(b.task_id,ref))
    text,_=b.recent()
    assert 'A_ONLY' not in text and 'unavailable or invalid' in text


def test_long_exact_reply_is_paged_and_reachable_after_restart(tmp_path):
    task=TaskSpec('task','check');memory=WorkerMemory(tmp_path,task,'m')
    original='λ🌱rare_interface\n'*12000
    ref=memory.text(original,kind='reply')
    manifest=json.loads((tmp_path/ref).read_text())
    joined=''.join((tmp_path/p['path']).read_text() for p in manifest['parts'])
    assert json.loads(joined)['text']==original
    text,pages=ContextMap(tmp_path).lookup('rare_interface')
    assert pages and 'rare_interface' in text


def test_interrupt_is_recorded_and_does_not_turn_into_completion(tmp_path,monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL','1')
    task=TaskSpec('task','check');cfg=Config();atom=Atom(tmp_path,cfg)
    atom._run_gate=lambda *a,**kw: SimpleNamespace(ok=False,code=1,out='failed')
    def chat(*a,**kw):raise KeyboardInterrupt()
    atom.ol=SimpleNamespace(chat=chat)
    with pytest.raises(KeyboardInterrupt):atom.run(task,attempts=1,ui=Quiet())
    text,_=WorkerMemory(tmp_path,task,cfg.worker.name).recent()
    assert '[interrupted]' in text and 'no completion claim' in text


def test_memory_database_rejects_symlink(tmp_path):
    (tmp_path/'.spiral').mkdir()
    (tmp_path/'other.db').touch()
    (tmp_path/'.spiral/worker-memory.sqlite3').symlink_to(tmp_path/'other.db')
    with pytest.raises(ValueError,match='symbolic link'):
        WorkerMemory(tmp_path,TaskSpec('task','check'),'m')
