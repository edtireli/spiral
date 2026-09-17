import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from spiral.context_map import read_source
from spiral.context_measurement import ContextMeasurementUnavailable, SourceTokenizer, planning_inventory_view
from spiral.conductor import Conductor
from spiral.config import Config
from spiral.llm import Ollama
from spiral.planner import make_plan
from test_planner import _PlannerConfig, _PlannerModels, _reply


def helper(tmp_path, body=None):
    script = tmp_path / 'helper'
    script.write_text('#!' + sys.executable + '\n' + (body or '''import json,sys
index=0
while line:=sys.stdin.buffer.readline():
 data=sys.stdin.buffer.read(int(line))
 print(json.dumps({'index':index,'tokens':len(data)}))
 index+=1
print(json.dumps({'schema':'spiral.tokenizer.v1','scope':'raw_text_no_bos_no_special','count':index}))
'''))
    script.chmod(0o700)
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'GGUF-test-vocabulary-only')
    return script, model


def test_protocol_counts_unicode_and_caches_only_exact_content(tmp_path):
    binary, model = helper(tmp_path)
    meter = SourceTokenizer(binary, model, 'selected:variant')
    assert meter.count(['λ\n', '', 'λ\n']) == [3, 0, 3]
    assert len(meter._cache) == 2
    assert meter.count(['λ\n', 'new']) == [3, 3]


def test_delayed_reader_receives_entire_batch_larger_than_a_pipe(tmp_path):
    binary, model = helper(tmp_path)
    body=binary.read_text()
    binary.write_text(body.replace('import json,sys', 'import json,sys,time\ntime.sleep(0.35)'))
    text='λ🌱' * 100000
    meter=SourceTokenizer(binary,model,'selected:variant',timeout=3)
    assert meter.count([text,'tail'])==[len(text.encode()),4]


@pytest.mark.parametrize('output', [
    '', '{}',
    '{"index":0,"tokens":true}\n{"schema":"spiral.tokenizer.v1","scope":"raw_text_no_bos_no_special","count":1}',
    '{"index":0,"tokens":1,"tokens":2}\n{"schema":"spiral.tokenizer.v1","scope":"raw_text_no_bos_no_special","count":1}',
    '{"index":0,"tokens":1}\n{"schema":"spiral.tokenizer.v1","scope":"raw_text_no_bos_no_special","count":true}',
    '{"index":1,"tokens":1}\n{"schema":"spiral.tokenizer.v1","scope":"raw_text_no_bos_no_special","count":1}',
])
def test_malformed_or_partial_counts_never_become_measurements(tmp_path, output):
    binary, model = helper(tmp_path, 'print(' + repr(output) + ')\n')
    meter = SourceTokenizer(binary, model, 'selected:variant')
    with pytest.raises(ContextMeasurementUnavailable):
        meter.count(['abc'])
    assert not meter._cache


def test_artifact_change_invalidates_cached_counts(tmp_path):
    binary, model = helper(tmp_path)
    meter = SourceTokenizer(binary, model, 'selected:variant')
    meter.count(['abc'])
    model.write_bytes(b'GGUF-changed-vocabulary')
    with pytest.raises(ContextMeasurementUnavailable, match='changed'):
        meter.count(['abc'])


@pytest.fixture
def spawned_tokenizers(monkeypatch):
    import spiral.context_measurement as module
    launched = []
    launch = module.subprocess.Popen
    def capture(*args, **kwargs):
        process = launch(*args, **kwargs)
        launched.append(process)
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', capture)
    return launched


def test_timeout_reaps_only_the_owned_helper(tmp_path, spawned_tokenizers):
    binary, model = helper(tmp_path, "import time\ntime.sleep(30)\n")
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    spawned_tokenizers.clear()
    meter = SourceTokenizer(binary, model, 'selected:variant', timeout=0.3)
    try:
        with pytest.raises(ContextMeasurementUnavailable, match='time bound'):
            meter.count(['abc'])
        assert len(spawned_tokenizers) == 1
        process = spawned_tokenizers[0]
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_cooperative_stop_is_not_swallowed_or_cached(tmp_path):
    binary, model = helper(tmp_path, 'import time\ntime.sleep(30)\n')
    calls = []
    def stop():
        calls.append(1)
        if len(calls) >= 3:
            raise RuntimeError('owner stopped work')
    meter = SourceTokenizer(binary, model, 'selected:variant', checkpoint=stop)
    with pytest.raises(RuntimeError, match='owner stopped work'):
        meter.count(['abc'])
    assert not meter._cache


def test_large_inventory_uses_explicit_bounded_navigation_view():
    inventory = {'model':'selected:variant','execution_admitted':False,'files':[
        {'path':f'package_{i}/source.py','tokens':i,'sha256':'a'*64,'dependencies':[]} for i in range(500)]}
    view = planning_inventory_view(inventory, '.spiral/context/full.json', max_bytes=3000)
    assert len(json.dumps(view, ensure_ascii=False, separators=(',',':')).encode()) <= 3000
    assert view['total_source_tokens'] == sum(range(500))
    assert view['omitted_view_files'] > 0 and view['omitted_view_directories'] > 0
    assert view['execution_admitted'] is False and len(inventory['files']) == 500


def test_runtime_factory_uses_exact_selected_model_and_does_not_call_generation(tmp_path, monkeypatch):
    binary, model = helper(tmp_path)
    monkeypatch.setenv('SPIRAL_CONTEXT_TOKENIZER', str(binary))
    client = Ollama(providers={})
    requests = []
    client._client.close()
    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200,json={'modelfile':f'FROM {model}\n'})
    client._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        meter = client.source_tokenizer('selected:exact-variant')
        assert meter.model == 'selected:exact-variant'
        assert meter.count(['abc']) == [3]
        assert requests == [('/api/show', {'model':'selected:exact-variant'})]
        assert client.budget.calls == 0
    finally:
        client.close()


def test_missing_helper_and_remote_provider_do_not_create_local_measurement(tmp_path, monkeypatch):
    monkeypatch.delenv('SPIRAL_CONTEXT_TOKENIZER', raising=False)
    with Ollama(providers={}) as client:
        assert client.source_tokenizer('selected') is None
    binary, _ = helper(tmp_path)
    monkeypatch.setenv('SPIRAL_CONTEXT_TOKENIZER', str(binary))
    with Ollama(providers={'remote': {}}) as client:
        assert client.source_tokenizer('remote') is None


def test_conductor_passes_real_measured_sources_and_saves_exact_inventory(tmp_path):
    binary, model = helper(tmp_path)
    (tmp_path / 'core.py').write_text('def action(): return 1\n')
    runner = Conductor(tmp_path, Config())
    meter = SourceTokenizer(binary, model, runner.cfg.worker.name)
    runner.ol.close()
    runner.ol = SimpleNamespace(source_tokenizer=lambda model: meter)
    inventory = runner._measured_source_inventory()
    assert inventory['model'] == runner.cfg.worker.name and inventory['execution_admitted'] is False
    core = next(row for row in inventory['files'] if row['path'] == 'core.py')
    assert core['tokens'] == len((tmp_path/'core.py').read_bytes())
    assert json.loads((tmp_path/inventory['full_inventory']).read_text())['files']


def test_normal_planner_receives_measurements_without_forbidding_future_outputs():
    cfg = _PlannerConfig()
    cfg.worker = SimpleNamespace(name='selected:variant',num_ctx=32768)
    cfg.worker_max_tokens=4096
    inventory={'model':'selected:variant','execution_admitted':False,'files':[{'path':'existing.py','tokens':123}]}
    models=_PlannerModels([_reply('{"milestones":[{"tasks":[{"title":"create","files":["new.py"],"context_reads":["existing.py"]}]}]}')])
    plan,_=make_plan('Keep the complete goal','repo',cfg=cfg,ol=models,source_inventory=inventory)
    prompt=models.calls[0][1][1]['content']
    assert 'MEASURED EXISTING SOURCE COSTS' in prompt and '"tokens":123' in prompt
    assert plan.milestones[0].tasks[0].files == ['new.py']
    assert 'Future outputs and omitted files' in prompt and 'provider admission' in prompt


def test_mismatched_inventory_cannot_reach_planner():
    cfg=_PlannerConfig(); cfg.worker=SimpleNamespace(name='selected')
    models=_PlannerModels([])
    with pytest.raises(ValueError,match='worker model'):
        make_plan('goal','repo',cfg=cfg,ol=models,source_inventory={'model':'other','execution_admitted':False})
    assert not models.calls


def test_owned_helper_is_drained_before_runtime_activity_ends(tmp_path, monkeypatch, spawned_tokenizers):
    from contextlib import contextmanager
    import spiral.context_measurement as module
    binary, model = helper(tmp_path, "import time\ntime.sleep(30)\n")
    events=[]
    @contextmanager
    def activity(kind):
        events.append(('enter',kind))
        try:
            yield
        finally:
            # Popen owns the identity even if timeout precedes interpreter startup.
            assert len(spawned_tokenizers) == 1
            process = spawned_tokenizers[0]
            assert process.returncode is not None
            with pytest.raises(ProcessLookupError):
                os.kill(process.pid,0)
            events.append(('drained',kind))
    monkeypatch.setattr(module,'activity',activity)
    meter=SourceTokenizer(binary,model,'selected',timeout=0.3)
    with pytest.raises(ContextMeasurementUnavailable):
        meter.count(['input'])
    assert events == [('enter','source-tokenization'),('drained','source-tokenization')]


def test_oversized_helper_output_never_enters_the_count_cache(tmp_path):
    binary,model=helper(tmp_path,"print('x'*600000)\n")
    meter=SourceTokenizer(binary,model,'selected')
    with pytest.raises(ContextMeasurementUnavailable,match='output bound'):
        meter.count(['input'])
    assert not meter._cache
