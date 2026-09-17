"""Explicit continuation restores pending work, never old authority or green state."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.context_store import ContextStore
from spiral.transactions import TaskTransaction
from spiral.working_checkpoint import CheckpointUnavailable, seal, restore
from test_worker_memory import Quiet, reply, patch


def archived(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path/'module.py').write_text('VALUE = 1\n')
    tx = TaskTransaction.begin(tmp_path, 'unfinished work')
    (tmp_path/'module.py').write_text('VALUE = 2\n')
    (tmp_path/'new.py').write_text('NEW = True\n')
    archive = tx.rollback(reason='budget exhausted')
    reference = seal(tx, archive)
    tx.close()
    return archive, reference


def test_exact_baseline_restores_unverified_files_and_remains_rollbackable(tmp_path, monkeypatch):
    _, reference = archived(tmp_path, monkeypatch)
    tx = TaskTransaction.begin(tmp_path, 'explicit resume')
    result = restore(tx, reference)
    assert result == {'changed': 2, 'deleted': 0, 'verified': False}
    assert (tmp_path/'module.py').read_text() == 'VALUE = 2\n'
    assert (tmp_path/'new.py').exists()
    tx.rollback(reason='still failing')
    assert (tmp_path/'module.py').read_text() == 'VALUE = 1\n'
    assert not (tmp_path/'new.py').exists()
    tx.close()


@pytest.mark.parametrize('mutation', ['baseline', 'candidate', 'manifest', 'symlink', 'unrelated-file'])
def test_changed_or_escaping_checkpoint_performs_no_writes(tmp_path, monkeypatch, mutation):
    archive, reference = archived(tmp_path, monkeypatch)
    if mutation == 'baseline': (tmp_path/'module.py').write_text('USER EDIT\n')
    if mutation == 'candidate': (archive/'changed/module.py').write_text('CORRUPT\n')
    if mutation == 'manifest': (tmp_path/reference).write_text('{}')
    if mutation == 'unrelated-file': (tmp_path/'user.txt').write_text('new user data')
    if mutation == 'symlink':
        original = archive/'changed'
        original.rename(archive/'original-changed')
        original.symlink_to(archive/'original-changed', target_is_directory=True)
    before = (tmp_path/'module.py').read_bytes()
    tx = TaskTransaction.begin(tmp_path, 'resume')
    with pytest.raises(CheckpointUnavailable): restore(tx, reference)
    assert (tmp_path/'module.py').read_bytes() == before
    assert not (tmp_path/'new.py').exists()
    tx.close()


def test_protected_path_in_content_addressed_manifest_is_not_execution_authority(tmp_path, monkeypatch):
    _, reference = archived(tmp_path, monkeypatch)
    value = json.loads((tmp_path/reference).read_text())
    value['changed'] = {'.spiral/control.py': 'a'*64}
    forged = ContextStore(tmp_path).save(json.dumps(value), kind='working-checkpoint')
    tx = TaskTransaction.begin(tmp_path, 'resume')
    with pytest.raises(CheckpointUnavailable): restore(tx, forged)
    assert not (tmp_path/'.spiral/control.py').exists()
    tx.close()


def test_actual_worker_explicit_resume_rechecks_candidate_before_model_and_completion(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    source = tmp_path/'module.py'; source.write_text('VALUE = 1\n')
    cfg = Config(); cfg.worker.name = 'exact:model'; cfg.diversity_samples = 0
    task = TaskSpec('Finish three without losing pending work', 'check', ['module.py'],
                    context='Preserve authored constraints', runtime_context='old environment')
    gates = []
    def gate(*a, **k):
        current = source.read_text(); gates.append(current)
        ok = current == 'VALUE = 3\n'
        return SimpleNamespace(ok=ok, code=0 if ok else 1, out='' if ok else 'AssertionError: expected three')
    first = Atom(tmp_path, cfg); first._run_gate = gate
    first.ol = SimpleNamespace(chat=lambda *a, **k: reply(patch(2)))
    assert not first.run(task, attempts=1, diversity=False, ui=Quiet())
    assert source.read_text() == 'VALUE = 1\n'
    resumed = Atom(tmp_path, cfg); resumed.resume_candidates = True; resumed._run_gate = gate
    def chat(model, messages, **options):
        assert model == 'exact:model'
        assert source.read_text() == 'VALUE = 2\n'
        assert gates[-1] == 'VALUE = 2\n'
        assert 'AssertionError: expected three' in messages[-1]['content']
        assert 'new environment' in messages[-1]['content']
        return reply('module.py\n<<<<<<< SEARCH\nVALUE = 2\n=======\nVALUE = 3\n>>>>>>> REPLACE')
    resumed.ol = SimpleNamespace(chat=chat)
    assert resumed.run(replace(task, runtime_context='new environment'),
                       attempts=1, diversity=False, ui=Quiet())
    assert source.read_text() == 'VALUE = 3\n'
    assert gates == ['VALUE = 1\n', 'VALUE = 2\n', 'VALUE = 2\n', 'VALUE = 3\n']


def test_changed_task_or_model_does_not_restore_other_work(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    source = tmp_path/'module.py'; source.write_text('VALUE = 1\n')
    cfg = Config(); cfg.worker.name = 'exact:one'; cfg.diversity_samples = 0
    first = Atom(tmp_path, cfg)
    first._run_gate = lambda *a, **k: SimpleNamespace(ok=False,code=1,out='AssertionError: incomplete')
    first.ol = SimpleNamespace(chat=lambda *a, **k: reply(patch(2)))
    task = TaskSpec('Original task', 'check', ['module.py'])
    assert not first.run(task, attempts=1, diversity=False, ui=Quiet())
    cfg.worker.name = 'exact:two'
    second = Atom(tmp_path, cfg); second.resume_candidates = True; second._run_gate = first._run_gate
    def chat(*a, **k):
        assert source.read_text() == 'VALUE = 1\n'
        return reply('ALREADY_DONE')
    second.ol = SimpleNamespace(chat=chat)
    assert not second.run(task, attempts=1, diversity=False, ui=Quiet())


def test_partial_restoration_error_rolls_back_before_any_model_call(tmp_path, monkeypatch):
    from spiral.worker_memory import WorkerMemory
    import spiral.working_checkpoint as checkpoints

    _, reference = archived(tmp_path, monkeypatch)
    cfg = Config(); cfg.worker.name = 'exact:model'; cfg.diversity_samples = 0
    task = TaskSpec('Finish pending work', 'check', ['module.py'])
    WorkerMemory(tmp_path, task, cfg.worker.name).record('working_checkpoint',
        summary='Unverified candidate', working_checkpoint_reference=reference)
    original_replace = checkpoints.os.replace
    def fail_second(source, destination):
        if destination.name == 'new.py':
            raise OSError('injected publication failure')
        return original_replace(source, destination)
    monkeypatch.setattr(checkpoints.os, 'replace', fail_second)
    worker = Atom(tmp_path, cfg); worker.resume_candidates = True
    worker.ol = SimpleNamespace(chat=lambda *a, **k: pytest.fail('model cannot run after partial restore'))
    with pytest.raises(OSError, match='publication failure'):
        worker.run(task, attempts=1, diversity=False, ui=Quiet())
    assert (tmp_path/'module.py').read_text() == 'VALUE = 1\n'
    assert not (tmp_path/'new.py').exists()


def test_deletions_resume_only_from_exact_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path/'old.py').write_text('retired = True\n')
    tx = TaskTransaction.begin(tmp_path, 'remove obsolete file')
    (tmp_path/'old.py').unlink()
    archive = tx.rollback(reason='unfinished')
    reference = seal(tx, archive); tx.close()
    tx = TaskTransaction.begin(tmp_path, 'resume removal')
    assert restore(tx, reference) == {'changed': 0, 'deleted': 1, 'verified': False}
    assert not (tmp_path/'old.py').exists()
    tx.rollback(reason='still unfinished'); tx.close()
    assert (tmp_path/'old.py').read_text() == 'retired = True\n'
