"""Long coding rounds must retain complete applied changes and model identity."""
from types import SimpleNamespace

from spiral.agent import Atom, TaskSpec
from spiral.config import Config


def test_delta_budget_rebases_whole_files_instead_of_truncating_changes(tmp_path):
    atom = Atom(tmp_path)
    files = [f"module_{i}.py" for i in range(5)]
    original = "".join(f"value_{n:04d} = {n}\n" for n in range(700))
    for name in files:
        (tmp_path / name).write_text(original)
    atom._frozen_bodies = {}
    atom._refresh_frozen_context(files, set())
    changed = original.replace(" = ", " = -", 100)
    for name in files:
        (tmp_path / name).write_text(changed)
    atom._refresh_frozen_context(files, set(), delta_budget=4500)
    delta = atom._frozen_deltas
    assert len(delta) <= 4500
    assert delta.endswith("\n")
    # Every changed file is either rebased in full or has its COMPLETE diff.
    import difflib
    for name in files:
        if atom._frozen_bodies[name] == changed:
            assert f"--- {name} (as shown)" not in delta
        else:
            assert atom._frozen_bodies[name] == original
            expected = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                changed.splitlines(keepends=True), fromfile=f"{name} (as shown)",
                tofile=f"{name} (current)"))
            assert expected in delta
    # A later unchanged round must retain the exact same coherent view.
    before = dict(atom._frozen_bodies), delta
    atom._refresh_frozen_context(files, set(), delta_budget=4500)
    assert before == (atom._frozen_bodies, atom._frozen_deltas)


def test_failed_match_rebases_file_and_does_not_replay_its_delta(tmp_path):
    atom = Atom(tmp_path)
    atom._frozen_bodies = {"a.py": "OLD\n" * 1000}
    (tmp_path / "a.py").write_text("NEW\n" + "OLD\n" * 999)
    atom._refresh_frozen_context(["a.py"], {"a.py"})
    assert atom._frozen_bodies["a.py"].startswith("NEW\n")
    assert atom._frozen_deltas == ""


def test_attempt_compaction_uses_exact_active_worker_in_single_model_mode(tmp_path):
    cfg = Config()
    cfg.prefer_single_resident_model = True
    cfg.worker.name = "selected:variant"
    cfg.worker.num_ctx = 32768
    cfg.janitor.name = "unselected:small"
    calls = []
    atom = Atom(tmp_path, cfg)
    atom.ol = SimpleNamespace(chat=lambda model, messages, **kwargs:
        (calls.append((model, messages, kwargs)) or SimpleNamespace(text="Do not repeat the failed edit.")))
    result = atom._compact_tried([f"attempt {n}" for n in range(10)], model_name="selected:variant")
    assert calls[0][0] == "selected:variant"
    assert calls[0][2]["num_ctx"] == 32768
    assert result[-4:] == [f"attempt {n}" for n in range(6, 10)]


def test_explicit_multi_model_mode_retains_configured_compactor(tmp_path):
    cfg = Config()
    cfg.prefer_single_resident_model = False
    cfg.janitor.name = "configured:compactor"
    calls = []
    atom = Atom(tmp_path, cfg)
    atom.ol = SimpleNamespace(chat=lambda model, messages, **kwargs:
        (calls.append(model) or SimpleNamespace(text="Retained lesson.")))
    atom._compact_tried([str(n) for n in range(10)], model_name=cfg.worker.name)
    assert calls == ["configured:compactor"]


def test_long_scope_keeps_exact_retrievable_middle_across_new_worker(tmp_path):
    import re
    scope = "initial requirements\n" * 400 + "MIDDLE REQUIREMENT — preserve café\n" + "final requirements\n" * 400
    atom = Atom(tmp_path)
    prompt = atom._prompt(TaskSpec(scope, "", context=scope), [], "", "")
    references = re.findall(r"ASK: file (\.spiral/context/[^ ]+) :: offset (\d+)", prompt)
    assert len(references) == 2
    assert "Incomplete task" in prompt and "Incomplete project" in prompt
    restarted = Atom(tmp_path)
    for name, offset in references:
        assert (tmp_path / name).read_text() == scope
        page = restarted._resolve_file_query(f"{name} :: offset {offset}", [], cap=6000)
        assert "MIDDLE REQUIREMENT — preserve café" in page
        assert "full-file sha256" in page


def test_long_file_tail_is_reachable_and_unicode_offsets_do_not_drop_characters(tmp_path):
    text = "λ🌱" * 10000 + "TAIL_INTERFACE = True\n"
    (tmp_path / "large.py").write_text(text)
    atom = Atom(tmp_path)
    prompt = atom._prompt(TaskSpec("repair", ""), ["large.py"], "", "", body_budget=4000)
    assert "ASK: file large.py :: offset 4000" in prompt
    page = atom._resolve_file_query("large.py :: offset 19000", [], cap=6000)
    assert text[19000:] in page
    assert f"characters 19000:{len(text)} of {len(text)}" in page


def test_file_pages_reject_escape_and_out_of_range(tmp_path):
    atom = Atom(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-private.txt")
    outside.write_text("outside content")
    (tmp_path / "link.txt").symlink_to(outside)
    assert "outside content" not in atom._resolve_file_query("link.txt :: offset 0", [])
    (tmp_path / "small.txt").write_text("abc")
    assert "outside 0..3" in atom._resolve_file_query("small.txt :: offset 9", [])


def test_changed_scope_has_a_new_identity_and_keeps_old_scope(tmp_path):
    atom = Atom(tmp_path)
    original = "A" * 10000
    atom._scope_excerpt(original, kind="project", head=6000, tail=3000)
    old = next((tmp_path / ".spiral/context").glob("*.txt"))
    atom._scope_excerpt(original + "new constraint", kind="project", head=6000, tail=3000)
    assert old.read_text() == original
    assert len(list((tmp_path / ".spiral/context").glob("*.txt"))) == 2


def test_declared_read_dependencies_reach_actual_worker_prompt(tmp_path):
    from spiral.conductor import _worker_task_spec
    from spiral.planner import Task
    task = Task("implement", "preserve the shared interface", ["implementation.py"])
    task.context_reads = ["interface.py", "implementation.py"]
    (tmp_path / "implementation.py").write_text("# implementation\n")
    (tmp_path / "interface.py").write_text("def required_contract(value): ...\n")
    spec = _worker_task_spec(task, "ORIGINAL PROJECT REQUIREMENTS", "python -m unittest")
    assert spec.files == ["implementation.py", "interface.py"]
    assert task.files == ["implementation.py"]
    prompt = Atom(tmp_path)._prompt(spec, spec.files, "", "")
    assert "def required_contract(value)" in prompt
    assert "ORIGINAL PROJECT REQUIREMENTS" in prompt
    assert "VERIFY (must exit 0): python -m unittest" in prompt


def test_failed_summary_retains_exact_attempt_history_across_restart(tmp_path):
    atom = Atom(tmp_path, Config())
    def unavailable(*args, **kwargs):
        raise RuntimeError("compactor unavailable")
    atom.ol = SimpleNamespace(chat=unavailable)
    tried = [f"attempt {i}: exact observed error {i}" for i in range(14)]
    compacted = atom._compact_tried(tried, model_name=atom.cfg.worker.name)
    assert len(compacted) == 8 and compacted[-7:] == tried[-7:]
    import re
    path = re.search(r"ASK: file ([^;]+);", compacted[0])[1]
    page = Atom(tmp_path)._resolve_file_query(path, [])
    assert all(entry in page for entry in tried)
    assert "not verification evidence" in compacted[0]
    prompt = atom._prompt(TaskSpec('continue', ''), [], '', '', tried=compacted)
    assert path in prompt  # the real prompt must not discard the retrieval address


def test_omitted_verification_output_remains_retrievable(tmp_path):
    atom = Atom(tmp_path)
    output = 'EARLIEST_ROOT_FAILURE\n' + 'diagnostic\n' * 1000 + 'FINAL_FAILURE\n'
    prompt = atom._prompt(TaskSpec('repair', ''), [], output, '')
    import re
    address = re.search(r'ASK: file (\.spiral/context/verification-output-[^ ]+) :: offset 0', prompt)
    assert address and 'FINAL_FAILURE' in prompt
    assert (tmp_path / address[1]).read_text() == output
    assert 'EARLIEST_ROOT_FAILURE' in Atom(tmp_path)._resolve_file_query(address[1], [])


def test_tool_result_and_old_tool_history_keep_exact_sources_after_two_compactions(tmp_path):
    atom = Atom(tmp_path)
    first = 'EARLY_INTERFACE\n' + 'x' * 10000 + '\nNEXT_PAGE'
    history = atom._append_repo_answer('', 'ASK file large.py', first)
    for i in range(12):
        history = atom._append_repo_answer(history, f'ASK file file{i}.py', f'result {i}\n' + 'y' * 3000)
    prompt = atom._prompt(TaskSpec('continue', ''), [], '', '', repo_answers=history)
    import re
    references = re.findall(r'ASK: file (\.spiral/context/[^ ]+) :: offset \d+', prompt)
    assert references and len(history) < 24500 and len(prompt) < 15000
    # Every omitted original remains on disk; traversal survives a new worker.
    originals = list((tmp_path / '.spiral/context').glob('tool-output-*.txt'))
    original = next(path for path in originals if path.read_text() == first)
    assert 'EARLY_INTERFACE' in Atom(tmp_path)._resolve_file_query(str(original.relative_to(tmp_path)), [])
    assert all((tmp_path / reference).is_file() for reference in references)


def test_head_only_excerpt_does_not_append_the_whole_original(tmp_path):
    atom = Atom(tmp_path)
    excerpt = atom._scope_excerpt('q' * 10000, kind='tool-output', head=2000, tail=0)
    assert len(excerpt) < 2600


def test_unit_test_summary_retains_test_stage_when_packaging_failure_is_fixed():
    before = ('FAIL: behavior_check\nresult still incorrect\n'
              'FAIL: installed_command\npackage installation error\nFAILED (failures=2)')
    after = 'FAIL: behavior_check\nresult still incorrect\nFAILED (failures=1)'
    old, new = Atom._failure_state(before), Atom._failure_state(after)
    assert old.stage == new.stage == 4
    assert Atom._is_ratchet_progress(old, new)
    assert not Atom._is_ratchet_progress(new, old)
    assert not Atom._is_ratchet_progress(new, new)


def test_package_error_without_test_summary_remains_package_stage():
    assert Atom._failure_state('package installation error\nFAILED').stage == 3


def test_next_lane_receives_patch_failure_and_rollback_state_without_leaking_to_new_task(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path / 'module.py').write_text('ACTUAL_INTERFACE = 1\n')
    cfg = Config()
    cfg.worker.name = cfg.escalation.name = 'selected:exact'
    atom = Atom(tmp_path, cfg)
    calls = []
    def chat(name, messages, **kw):
        calls.append(messages)
        text = ('module.py\n<<<<<<< SEARCH\nNONEXISTENT_INTERFACE = 999\n=======\nVALUE = 2\n>>>>>>> REPLACE'
                if len(calls) == 1 else '')
        return SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *a, **kw: SimpleNamespace(ok=False, code=1, out='FAIL: behavior\nFAILED (failures=1)')
    class Quiet:
        def __getattr__(self, name): return lambda *a, **kw: None
    task = TaskSpec('repair', 'python3 checks.py', files=['module.py'])
    atom.run(task, attempts=1, diversity=False, ui=Quiet())
    atom.run(task, model=cfg.escalation.name, lane='escalation', attempts=1, diversity=False, ui=Quiet())
    feedback = calls[1][-1]['content']
    assert 'search block not found in file' in feedback
    assert 'retained checkpoint' in feedback and 'historical' in feedback
    assert (tmp_path / 'module.py').read_text() == 'ACTUAL_INTERFACE = 1\n'
    atom.run(TaskSpec('new task', '', files=['module.py']), attempts=1, diversity=False, ui=Quiet())
    assert 'search block not found in file' not in calls[2][-1]['content']


def test_same_model_escalation_honors_its_reasoning_and_context_policy(tmp_path, monkeypatch):
    from spiral.conductor import Conductor
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path / 'module.py').write_text('VALUE = 1\n')
    cfg = Config()
    cfg.worker.name = cfg.escalation.name = 'selected:exact'
    cfg.worker.num_ctx, cfg.worker.think = 32768, False
    cfg.escalation.num_ctx, cfg.escalation.think = 16384, True
    calls = []
    atom = Atom(tmp_path, cfg)
    atom.ol = SimpleNamespace(chat=lambda name, messages, **options:
        (calls.append((name, options)) or SimpleNamespace(text='', prompt_tokens=1,
            completion_tokens=1, total_tokens=2)))
    atom._run_gate = lambda *args, **kwargs: SimpleNamespace(ok=False, code=1, out='gate failed')
    class Quiet:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    runner = object.__new__(Conductor)
    runner.cfg = cfg
    assert runner._run_task(atom, TaskSpec('repair', 'python3 check.py', files=['module.py']),
                            Quiet(), attempts=1, esc_attempts=1) == 'blocked'
    # No valid edits means the diversity helper performs no extra model call.
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == 'selected:exact'
    assert (calls[0][1]['think'], calls[0][1]['num_ctx']) == (False, 32768)
    assert (calls[1][1]['think'], calls[1][1]['num_ctx']) == (True, 16384)
