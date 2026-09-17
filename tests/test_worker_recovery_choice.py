"""Repeated failure should expose evidence, not synthesize a filename search."""
from types import SimpleNamespace

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from test_worker_memory import Quiet, reply


def test_real_worker_reconsiders_actual_failure_before_choosing_web(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    source = tmp_path / 'module.py'
    source.write_text('VALUE = 1\n')
    cfg = Config(); cfg.worker.name = 'selected:exact'; cfg.diversity_samples = 0
    cfg.web_research = True
    worker = Atom(tmp_path, cfg)
    worker._hunt_symbols = lambda *a: ''
    worker._run_gate = lambda *a, **kw: SimpleNamespace(
        ok=False, code=1, out='FAIL: opaque_case\nObserved value violates the invariant.')
    calls, queries = [], []
    worker._web_research = lambda query, **kw: queries.append(query) or 'Untrusted documentation evidence.'
    def chat(model, messages, **options):
        calls.append(messages[-1]['content'])
        assert model == 'selected:exact'
        if len(calls) == 3:
            assert queries == []
            assert 'same verification failure remains after two edit attempts' in calls[-1]
            assert 'Observed value violates the invariant.' in calls[-1]
            return reply('ASK: web model-selected question about the actual invariant')
        if len(calls) == 4:
            assert queries == ['model-selected question about the actual invariant']
            assert 'Untrusted documentation evidence.' in calls[-1]
        before = source.read_text().strip()
        after = 'VALUE = ' + str(len(calls) + 1)
        return reply(f'module.py\n<<<<<<< SEARCH\n{before}\n=======\n{after}\n>>>>>>> REPLACE')
    worker.ol = SimpleNamespace(chat=chat)
    assert not worker.run(TaskSpec('Repair the invariant', 'check', ['module.py']),
                          attempts=3, diversity=False, ui=Quiet())
    assert len(calls) == 4
    assert queries == ['model-selected question about the actual invariant']
