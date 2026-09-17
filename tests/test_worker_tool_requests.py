"""Real observed commentary+ASK failure, without inferring tools from prose."""
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, TaskSpec, _tool_request
from spiral.config import Config


OBSERVED = ('The CLI tests fail with returncode 2, which means argparse is erroring out. '
    'The issue is that `main()` is called with no arguments when run as a script, '
    'but the test passes a filename. Let me check the actual error by running the CLI directly.\n\n'
    'ASK: shell python3 csvcli.py tests/test_data.csv 2>&1 || echo "exit: $?"')


def test_observed_explicit_request_survives_commentary_without_rewriting_arguments():
    match = _tool_request(OBSERVED)
    assert match.group(1) == 'shell'
    assert match.group(2) == 'python3 csvcli.py tests/test_data.csv 2>&1 || echo "exit: $?"'


@pytest.mark.parametrize('text', [
    'I will run python3 program.py.',
    '> ASK: shell python3 program.py',
    '```\nASK: shell python3 program.py\n```',
    'ASK: file a.py\nASK: shell python3 program.py',
    'ASK: file a.py\nThis is only an example.',
    'x.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\nASK: file a.py',
    'HARNESS_ERROR: unavailable\nASK: shell python3 program.py',
    'ASK: shell python3 program.py ' + 'x' * 2000,
    'ASK: unknown something',
])
def test_ambiguous_examples_mixed_actions_and_oversized_requests_do_not_dispatch(text):
    assert _tool_request(text) is None


def test_original_single_line_request_remains_supported():
    match = _tool_request('  ASK: file source.py :: offset 16000\n')
    assert match.groups() == ('file', 'source.py :: offset 16000')


def test_actual_worker_loop_dispatches_one_read_and_receives_its_page(tmp_path, monkeypatch):
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path / 'source.py').write_text('# filler\n' * 2200 + 'TAIL_INTERFACE = True\n')
    cfg = Config()
    atom = Atom(tmp_path, cfg)
    calls = []
    def chat(model, messages, **options):
        calls.append(messages)
        text = ('I need the omitted source region.\n\nASK: file source.py :: offset 19000'
                if len(calls) == 1 else '')
        return SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *args, **kwargs: SimpleNamespace(ok=False, code=1, out='verification failed')
    class Quiet:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    assert not atom.run(TaskSpec('repair', 'python3 checks.py', files=['source.py']),
                        attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 2  # a read is not an edit attempt
    assert 'TAIL_INTERFACE = True' in calls[1][-1]['content']
    assert 'full-file sha256' in calls[1][-1]['content']


def test_actual_worker_can_retrieve_the_middle_of_an_omitted_tool_result(tmp_path, monkeypatch):
    import re
    monkeypatch.setenv('SPIRALCHAT_EXTERNAL_GIT_APPROVAL', '1')
    (tmp_path / 'source.py').write_text('#' * 19000 + 'a' * 3100 + '\nREQUIRED_TAIL_INTERFACE\n' + 'b' * 2700)
    atom = Atom(tmp_path, Config())
    calls = []
    def chat(model, messages, **options):
        calls.append(messages)
        if len(calls) == 1:
            text = 'ASK: file source.py :: offset 19000'
        elif len(calls) == 2:
            prompt = messages[-1]['content']
            assert 'REQUIRED_TAIL_INTERFACE' not in prompt
            address = re.search(r'ASK: file (\.spiral/context/tool-output-[^ ]+) :: offset (\d+)', prompt)
            assert address, 'the actual tool reply must retain a reachable original'
            text = f'ASK: file {address[1]} :: offset {address[2]}'
        else:
            assert 'REQUIRED_TAIL_INTERFACE' in messages[-1]['content']
            text = ''
        return SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, total_tokens=2)
    atom.ol = SimpleNamespace(chat=chat)
    atom._run_gate = lambda *args, **kwargs: SimpleNamespace(ok=False, code=1, out='verification failed')
    class Quiet:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    assert not atom.run(TaskSpec('repair', 'python3 checks.py', files=['source.py']),
                        attempts=1, diversity=False, ui=Quiet())
    assert len(calls) == 3
