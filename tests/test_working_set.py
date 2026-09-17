"""Measured source packing is not backend context or quality admission."""
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.conductor import Conductor
from spiral.working_set import WorkingSetTooSmall, size_working_set


class ByteCounter:
    model='selected:exact'
    identity='fixture-byte-counter'
    def count(self, texts):
        return [len(t.encode()) for t in texts]


class Quiet:
    def __getattr__(self,name):return lambda *a,**kw:None


def test_small_context_keeps_control_and_unicode_and_only_measures_fitting_candidate():
    source='λ🌱'*15000
    control='ORIGINAL REQUIREMENT: preserve format\nVERIFY: original checks\n'
    def render(body):return control+source[:body]+f'\nNEXT_PAGE={body}', (body,)
    small=size_working_set(render,ByteCounter(),'SYSTEM',context_tokens=8192,
                           output_reserved=1024,max_characters=len(source))
    large=size_working_set(render,ByteCounter(),'SYSTEM',context_tokens=65536,
                           output_reserved=1024,max_characters=len(source))
    assert control in small.prompt and control in large.prompt
    assert small.pages==(small.body_characters,) and large.pages==(large.body_characters,)
    assert small.body_characters<large.body_characters
    for result in (small,large):
        assert result.raw_text_tokens==sum(ByteCounter().count(['SYSTEM',result.prompt]))
        assert result.raw_text_tokens+result.output_reserved+result.framing_reserved<=result.context_tokens
        assert result.trials<=10


def test_control_cannot_be_silently_cut_to_fit():
    with pytest.raises(WorkingSetTooSmall,match='control evidence alone'):
        size_working_set(lambda body:('essential'*5000,()),ByteCounter(),'SYSTEM',
                         context_tokens=8192,output_reserved=1024,max_characters=100)


def test_stop_propagates_without_another_count():
    def stop():raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        size_working_set(lambda body:('source',()),ByteCounter(),'sys',context_tokens=8192,
                         output_reserved=1024,max_characters=10,checkpoint=stop)


def test_actual_worker_can_read_and_edit_beyond_old_prefix_without_extra_model_call(tmp_path,monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL','1')
    source=tmp_path/'long.py';original='# filler\n'*3000+'TAIL_CONTRACT = 1\n';source.write_text(original)
    cfg=Config();cfg.worker.name='selected:exact';cfg.worker.num_ctx=65536;cfg.worker_max_tokens=4096
    atom=Atom(tmp_path,cfg);calls=[];meters=[]
    def meter(model):meters.append(model);return ByteCounter()
    def chat(model,messages,**options):
        calls.append(messages[-1]['content'])
        assert model=='selected:exact' and options['num_ctx']==65536
        assert 'TAIL_CONTRACT = 1' in calls[-1]
        return SimpleNamespace(text='long.py\n<<<<<<< SEARCH\nTAIL_CONTRACT = 1\n=======\nTAIL_CONTRACT = 2\n>>>>>>> REPLACE',
                               prompt_tokens=1,completion_tokens=1,total_tokens=2)
    atom.ol=SimpleNamespace(chat=chat,source_tokenizer=meter)
    atom._run_gate=lambda *a,**kw: SimpleNamespace(ok='TAIL_CONTRACT = 2' in source.read_text(),
        code=0 if 'TAIL_CONTRACT = 2' in source.read_text() else 1,out='AssertionError: tail contract')
    assert atom.run(TaskSpec('Repair tail contract','check',['long.py']),attempts=1,diversity=False,ui=Quiet())
    assert len(calls)==1 and meters==['selected:exact']
    assert source.read_text()==original.replace('TAIL_CONTRACT = 1','TAIL_CONTRACT = 2')


def test_context_refusal_blocks_before_inference_and_does_not_enlarge_settings(tmp_path,monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL','1')
    cfg=Config();cfg.worker.name='selected:exact';cfg.worker.num_ctx=2048;cfg.worker_max_tokens=4096
    atom=Atom(tmp_path,cfg)
    def forbidden(*a,**kw):pytest.fail('context cannot admit inference')
    atom.ol=SimpleNamespace(chat=forbidden,source_tokenizer=lambda model:ByteCounter())
    atom._run_gate=lambda *a,**kw: SimpleNamespace(ok=False,code=1,out='failed')
    runner=object.__new__(Conductor);runner.cfg=cfg
    assert runner._run_task(atom,TaskSpec('Keep every constraint','check'),Quiet(),attempts=1,esc_attempts=1)=='blocked'
    assert atom.budget_exhausted and atom._budget_stop.dimension=='context'
    assert cfg.worker.num_ctx==2048 and atom.run_stats['attempts']==0 and atom.run_stats['esc_lanes']==0


def test_counter_unavailable_preserves_existing_bounded_path(tmp_path,monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL','1')
    from spiral.context_measurement import ContextMeasurementUnavailable
    cfg=Config();cfg.worker.name='selected:exact';cfg.worker.num_ctx=65536
    source=tmp_path/'long.py';source.write_text('# filler\n'*3000+'TAIL_CONTRACT = 1\n')
    def unavailable(model):raise ContextMeasurementUnavailable('no host helper')
    calls=[]
    atom=Atom(tmp_path,cfg)
    atom.ol=SimpleNamespace(source_tokenizer=unavailable,chat=lambda model,messages,**kw:
        calls.append((model,messages[-1]['content'])) or SimpleNamespace(text='',prompt_tokens=1,completion_tokens=1,total_tokens=2))
    atom._run_gate=lambda *a,**kw: SimpleNamespace(ok=False,code=1,out='failed')
    assert not atom.run(TaskSpec('Repair','check',['long.py']),attempts=1,diversity=False,ui=Quiet())
    assert calls[0][0]=='selected:exact'
    assert 'TAIL_CONTRACT = 1' not in calls[0][1]
    assert 'offset 16000' in calls[0][1]


def test_large_prompt_chunks_stay_within_counter_item_bounds():
    seen=[]
    class Meter(ByteCounter):
        def count(self,texts):
            seen.extend(len(t.encode()) for t in texts)
            return [len(t)//10 for t in texts]
    result=size_working_set(lambda body:('🌱'*min(body,300000),()),Meter(),'sys',context_tokens=65536,
                            output_reserved=4096,max_characters=500000)
    assert result.trials==1 and max(seen)<=500000
