"""Evidence navigation and actual worker admission, with no real inference."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec, _tool_request
from spiral.config import Config
from spiral.context_map import ContextMap, covered, diagnostic_gap, edit_gap, read_source, verification_files
from spiral.edits import EditBlock


def test_map_finds_omitted_symbol_and_local_import_neighbors(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/core.py').write_text('# filler\n' * 2400 + 'def cryptic_token_8cb29():\n    return 42\n')
    (tmp_path / 'pkg/api.py').write_text('from .core import cryptic_token_8cb29\n')
    index = ContextMap(tmp_path)
    answer, pages = index.lookup('cryptic_token_8cb29')
    assert pages[0].path == 'pkg/core.py' and pages[0].start > 16000
    assert 'return 42' in answer and 'pkg/api.py' in answer
    assert index.dependencies('pkg/api.py') == ['pkg/core.py']
    assert pages[0].sha256 == hashlib.sha256((tmp_path / 'pkg/core.py').read_bytes()).hexdigest()
    assert _tool_request('ASK: context cryptic_token_8cb29').group(1) == 'context'


def test_relative_module_and_package_imports(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/__init__.py').write_text('')
    (tmp_path / 'pkg/util.py').write_text('value = 1\n')
    (tmp_path / 'pkg/api.py').write_text('from . import util\nimport pkg\n')
    assert ContextMap(tmp_path).dependencies('pkg/api.py') == ['pkg/__init__.py', 'pkg/util.py']


def test_map_does_not_follow_symlinks_or_claim_global_absence(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-private')
    outside.mkdir()
    (outside / 'secret.py').write_text('SECRET_VALUE')
    (tmp_path / 'linked').symlink_to(outside, target_is_directory=True)
    (tmp_path / 'link.py').symlink_to(outside / 'secret.py')
    (tmp_path / 'plain.kt').write_text('fun knownToken() = 1')
    answer, pages = ContextMap(tmp_path).lookup('SECRET_VALUE')
    assert not pages and 'bounded indexed scope' in answer
    answer, pages = ContextMap(tmp_path).lookup('knownToken')
    assert pages and pages[0].path == 'plain.kt'


def test_index_bounds_and_binary_reads_count(tmp_path):
    (tmp_path / 'a.bin').write_bytes(b'\x00' * 20)
    (tmp_path / 'b.py').write_text('secret = 2')
    index = ContextMap(tmp_path, max_bytes=20)
    assert index.limited and not index.sources
    index = ContextMap(tmp_path, max_entries=1)
    assert index.limited and not index.sources


def test_lookup_rebuild_after_restart_does_not_reuse_old_source(tmp_path):
    source = tmp_path / 'x.py'
    source.write_text('def target():\n    return 1\n')
    atom = Atom(tmp_path)
    first = atom._context_lookup('target')
    source.write_text('def target():\n    return 2\n')
    second = Atom(tmp_path)._context_lookup('target')
    assert 'return 1' in first and 'return 1' not in second and 'return 2' in second
    manifests = list((tmp_path / '.spiral/context').glob('source-map-*'))
    assert len(manifests) == 2
    for manifest in manifests:
        assert json.loads(manifest.read_text())['files'][0]['path'] == 'x.py'


def test_source_receipts_require_current_digest_and_current_prompt(tmp_path):
    file = tmp_path / 'x.py'
    file.write_text('alpha = 1\nbeta = 2\n')
    source = read_source(tmp_path, 'x.py')
    assert covered(source, 0, 5, [source.page(0, 5)])
    assert not covered(source, 0, 6, [source.page(0, 5)])
    file.write_text('alpha = 9\nbeta = 2\n')
    assert not covered(read_source(tmp_path, 'x.py'), 0, 5, [source.page(0, 5)])
    assert not covered(source, 0, 5, [])


def test_gap_in_omitted_tail_and_stale_file_is_found_before_edit(tmp_path):
    file = tmp_path / 'x.py'
    file.write_text('# filler\n' * 2500 + 'target = 1\n')
    atom = Atom(tmp_path)
    atom._prompt(TaskSpec('repair', ''), ['x.py'], '', '')
    blocks = [EditBlock('x.py', 'target = 1', 'target = 2')]
    reason, page = edit_gap(tmp_path, blocks, atom._visible_pages)
    assert 'absent' in reason and 'target = 1' in page.text
    assert not edit_gap(tmp_path, blocks, [page])
    file.write_text(file.read_text() + '# concurrent edit\n')
    assert edit_gap(tmp_path, blocks, [page])
    assert 'target = 1' in file.read_text()


def test_approximate_blind_and_whole_file_edits_need_evidence(tmp_path):
    (tmp_path / 'x.py').write_text('# filler\n' * 2500 + 'target = 1\n')
    source = read_source(tmp_path, 'x.py')
    for block in [EditBlock('x.py', 'target=1', 'target=2'), EditBlock('x.py', '', 'pass', 'whole')]:
        assert edit_gap(tmp_path, [block], [source.page(0, 1000)])
        assert not edit_gap(tmp_path, [block], [source.page(0, len(source.text))])


def test_packet_and_lookup_coverage_tracks_only_rendered_text(tmp_path):
    (tmp_path / 'x.py').write_text('# filler\n' * 2500 + 'def target():\n    return 1\n')
    atom = Atom(tmp_path)
    answer = atom._context_lookup('target')
    atom._prompt(TaskSpec('repair', ''), [], '', '', repo_answers=answer)
    assert atom._visible_pages
    atom._prompt(TaskSpec('repair', ''), [], '', '', repo_answers='evicted')
    assert not atom._visible_pages
    atom._remember_context_page(read_source(tmp_path, 'x.py').page(16000, 5000))
    prompt = atom._prompt(TaskSpec('repair', ''), [], '', '', body_budget=4000)
    assert 3000 < len(atom._visible_pages[0].text) < 4000
    assert atom._visible_pages[0].render() in prompt


class Quiet:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_real_worker_retrieves_before_mutation_and_allows_same_unapplied_reply(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    path = tmp_path / 'source.py'
    original = '# filler\n' * 2500 + 'target = 1\n'
    path.write_text(original)
    calls, gates = [], []
    atom = Atom(tmp_path, Config())
    def chat(model, messages, **kwargs):
        assert path.read_text() == original  # neither first proposal nor retrieval writes
        calls.append(messages)
        if len(calls) == 1:
            assert 'target = 1' not in messages[-1]['content']
        else:
            assert 'target = 1' in messages[-1]['content']
            assert 'SOURCE RETRIEVED BEFORE EDIT' in messages[-1]['content']
        return SimpleNamespace(text='source.py\n<<<<<<< SEARCH\ntarget = 1\n=======\ntarget = 2\n>>>>>>> REPLACE',
                               prompt_tokens=1, completion_tokens=1, total_tokens=2)
    def gate(*args, **kwargs):
        gates.append(path.read_text())
        ok = 'target = 2' in gates[-1]
        return SimpleNamespace(ok=ok, code=0 if ok else 1, out='' if ok else 'AssertionError: wrong result')
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = gate
    assert atom.run(TaskSpec('repair', 'python3 check.py', files=['source.py']), attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 2 and len(gates) == 2
    events = [json.loads(line) for line in (tmp_path / '.spiral/ledger.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['kind'] == 'context_gap']) == 1


def test_actual_worker_cannot_loop_or_write_blind_when_lookup_budget_is_spent(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    path = tmp_path / 'source.py'
    original = '# filler\n' * 2500 + 'target = 1\n'
    path.write_text(original)
    cfg = Config()
    cfg.ask_budget = 1
    atom = Atom(tmp_path, cfg)
    replies = ['ASK: context nonexistent_8ddace', 'source.py\n<<<<<<< SEARCH\ntarget = 1\n=======\ntarget = 2\n>>>>>>> REPLACE']
    calls = []
    def chat(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(text=replies.pop(0), prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *a, **kw: SimpleNamespace(ok=False, code=1, out='AssertionError: wrong result')
    assert not atom.run(TaskSpec('repair', 'python3 check.py', files=['source.py']), attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 2 and path.read_text() == original


def test_traceback_prefetch_finds_hidden_region_without_model_self_report(tmp_path):
    path = tmp_path / 'source.py'
    path.write_text('# filler\n' * 2500 + 'target = 1\n')
    page = diagnostic_gap(tmp_path, f'File "{path}", line 2501\nValueError: wrong result', [])
    assert page and page.start > 16000 and 'target = 1' in page.text
    assert diagnostic_gap(tmp_path, f'File "{path}", line 2501', [page]) is None
    assert diagnostic_gap(tmp_path, 'File "/etc/hosts", line 1', []) is None
    assert diagnostic_gap(tmp_path, 'source.py:99999: error', []) is None


def test_actual_worker_traceback_gap_is_filled_before_first_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    path = tmp_path / 'source.py'
    path.write_text('# filler\n' * 2500 + 'target = 1\n')
    atom = Atom(tmp_path, Config())
    calls = []
    def chat(model, messages, **kwargs):
        assert 'target = 1' in messages[-1]['content']
        calls.append(1)
        return SimpleNamespace(text='source.py\n<<<<<<< SEARCH\ntarget = 1\n=======\ntarget = 2\n>>>>>>> REPLACE',
                               prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *a, **kw: SimpleNamespace(ok='target = 2' in path.read_text(),
        code=0 if 'target = 2' in path.read_text() else 1, out=f'File "{path}", line 2501\nAssertionError: wrong result')
    assert atom.run(TaskSpec('repair', 'python3 check.py', files=['source.py']), attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 1


def test_map_cooperatively_obeys_stop(tmp_path, monkeypatch):
    import spiral.context_map as module
    (tmp_path / 'source.py').write_text('pass\n')
    def stop():
        raise RuntimeError('stopped by owner')
    monkeypatch.setattr(module, 'checkpoint', stop)
    with pytest.raises(RuntimeError, match='stopped by owner'):
        ContextMap(tmp_path)


def test_full_frozen_prefix_and_complete_diff_remain_covered(tmp_path):
    path = tmp_path / 'source.py'
    path.write_text('target = 1\n')
    atom = Atom(tmp_path)
    atom._frozen_bodies = {}
    atom._refresh_frozen_context(['source.py'], set())
    path.write_text('target = 2\n')
    atom._refresh_frozen_context(['source.py'], set())
    prompt = atom._prompt(TaskSpec('repair', ''), ['source.py'], '', '')
    assert '-target = 1' in prompt and '+target = 2' in prompt
    assert edit_gap(tmp_path, [EditBlock('source.py', 'target = 2', 'target = 3')], atom._visible_pages) is None


def test_small_current_sources_do_not_add_lookup_round(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    path = tmp_path / 'source.py'
    path.write_text('target = 1\n')
    atom = Atom(tmp_path, Config())
    calls = []
    def chat(*args, **kw):
        calls.append(1)
        return SimpleNamespace(text='source.py\n<<<<<<< SEARCH\ntarget = 1\n=======\ntarget = 2\n>>>>>>> REPLACE', prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *a, **kw: SimpleNamespace(ok='target = 2' in path.read_text(), code=0 if 'target = 2' in path.read_text() else 1, out='AssertionError: wrong result')
    assert atom.run(TaskSpec('repair', 'python3 check.py', files=['source.py']), attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 1


def test_unreadable_or_oversized_existing_source_cannot_bypass_admission(tmp_path):
    (tmp_path / 'binary.py').write_bytes(b'\x00bad')
    reason, page = edit_gap(tmp_path, [EditBlock('binary.py', '', 'pass', 'whole')], [])
    assert page is None and 'No edit is admitted' in reason
    (tmp_path / 'large.py').write_text('x' * (2 * 1024 * 1024 + 1))
    assert edit_gap(tmp_path, [EditBlock('large.py', '', 'pass', 'whole')], [])[1] is None


def test_restart_lookup_can_find_exact_old_constraints_without_recursive_maps(tmp_path):
    atom = Atom(tmp_path)
    original = 'early scope\n' * 800 + 'KEEP_RARE_CONSTRAINT_e39c72\n' + 'later scope\n' * 800
    atom._scope_excerpt(original, kind='project', head=6000, tail=3000)
    # A map must not index its own previous output or forge a source receipt.
    atom._save_context('KEEP_RARE_CONSTRAINT_e39c72 false map claim', kind='source-map')
    bad = tmp_path / '.spiral/context' / ('task-' + 'a' * 64 + '.txt')
    bad.write_text('KEEP_RARE_CONSTRAINT_e39c72 forged source')
    restarted = Atom(tmp_path)
    answer = restarted._context_lookup('KEEP_RARE_CONSTRAINT_e39c72')
    assert 'KEEP_RARE_CONSTRAINT_e39c72' in answer
    assert 'Historical saved context' in answer
    assert 'false map claim' not in answer and 'forged source' not in answer
    assert original == next((tmp_path / '.spiral/context').glob('project-*')).read_text()


def test_external_change_after_diff_creation_does_not_get_a_fresh_receipt(tmp_path):
    path = tmp_path / 'source.py'
    path.write_text('target = 1\n')
    atom = Atom(tmp_path)
    atom._frozen_bodies = {}
    atom._refresh_frozen_context(['source.py'], set())
    path.write_text('target = 2\n')
    atom._refresh_frozen_context(['source.py'], set())
    path.write_text('target = 3\n')  # concurrent writer after the snapshot
    atom._prompt(TaskSpec('repair', ''), ['source.py'], '', '')
    assert edit_gap(tmp_path, [EditBlock('source.py', 'target = 3', 'target = 4')], atom._visible_pages)


def test_multiple_omitted_edit_targets_can_share_the_current_working_set(tmp_path):
    atom = Atom(tmp_path)
    blocks = []
    for i in range(3):
        name = f'part_{i}.py'
        (tmp_path / name).write_text('# filler\n' * 2500 + f'target_{i} = 1\n')
        blocks.append(EditBlock(name, f'target_{i} = 1', f'target_{i} = 2'))
    for _ in range(3):
        atom._prompt(TaskSpec('repair', ''), [], '', '')
        gap = edit_gap(tmp_path, blocks, atom._visible_pages)
        assert gap
        atom._remember_context_page(gap[1])
    atom._prompt(TaskSpec('repair', ''), [], '', '')
    assert edit_gap(tmp_path, blocks, atom._visible_pages) is None


def test_large_map_is_exact_and_reachable_through_bounded_parts(tmp_path):
    atom = Atom(tmp_path)
    original = {'files': [{'path': f'module_{i}.py', 'symbols': ['αβγ' * 200]} for i in range(300)]}
    address = atom._save_context_map(original)
    manifest = json.loads((tmp_path / address).read_text())
    restored = ''
    for part in manifest['parts']:
        text = (tmp_path / part['path']).read_text()
        assert len(text) <= 64000
        assert 'file unavailable' not in atom._file_page(part['path'])
        restored += text
    assert hashlib.sha256(restored.encode()).hexdigest() == manifest['sha256']
    assert json.loads(restored) == original


def test_verification_sources_and_scoped_hidden_neighbors_are_retrievable(tmp_path):
    fixture = tmp_path / '.spiral/checks'
    fixture.mkdir(parents=True)
    (fixture / 'verify.py').write_text('import expectations\n')
    (fixture / 'expectations.py').write_text('assert reported_line == 7\n')
    (tmp_path / '.spiral/private.py').write_text('unrelated_secret')
    paths = verification_files(tmp_path, 'python3 .spiral/checks/verify.py')
    assert paths == ['.spiral/checks/verify.py']
    index = ContextMap(tmp_path, extra_roots=tuple(str(Path(p).parent) for p in paths))
    answer, pages = index.lookup('reported_line')
    assert pages[0].path == '.spiral/checks/expectations.py'
    assert 'reported_line == 7' in answer
    assert 'unrelated_secret' not in index.lookup('unrelated_secret')[0]


def test_verification_discovery_never_expands_shell_or_reads_outside_workspace(tmp_path):
    (tmp_path / 'verify.py').write_text('pass\n')
    (tmp_path / 'bad.py').symlink_to('/etc/hosts')
    assert verification_files(tmp_path, f'python3 {tmp_path}/verify.py') == ['verify.py']
    assert verification_files(tmp_path, 'python3 /etc/hosts bad.py ../verify.py "$(touch EXECUTED)"') == []
    assert not (tmp_path / 'EXECUTED').exists()
    assert verification_files(tmp_path, 'python3 "unclosed') == []


def test_symbol_search_does_not_confuse_substrings_with_exact_references(tmp_path):
    (tmp_path / 'a.py').write_text('def miscalculate(value): return value\n')
    (tmp_path / 'b.py').write_text('def calculate(value): return value\n')
    answer,pages=ContextMap(tmp_path).lookup('calculate')
    assert [page.path for page in pages] == ['b.py']
    assert 'miscalculate' not in answer
