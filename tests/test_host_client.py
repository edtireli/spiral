import argparse
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

from spiral import cli, session_cli
from spiral.host_client import HostClient, HostClientError, READ_TOOLS, start_body


MANIFEST = {"protocol": "spiral-v2", "protocol_version": 2, "minimum_client_protocol": 2,
            "capabilities": ["durable_assistant_turns", "host_assistant_runtime",
                             "assistant_dialogue", "assistant_tool_inspection_v1"]}


def client(handler, host="http://127.0.0.1:8124", **kwargs):
    def route(request):
        if request.url.path.endswith("/v2/manifest"):
            return httpx.Response(200, json=MANIFEST)
        return handler(request)
    return HostClient(host, transport=httpx.MockTransport(route), **kwargs)


def test_gateway_prefix_and_auth_are_preserved_for_reads_and_mutations():
    requests = []
    with client(lambda r: requests.append(r) or httpx.Response(200, json={}),
                host="https://spiral.example/spiral", token="private-token") as c:
        c.get("aw_a+b=")
        c.message("aw_a+b=", "How is it going?", request_id="question-1")
    assert [r.url.path for r in requests] == [
        "/spiral/v2/assistant/turns/aw_a+b=",
        "/spiral/v2/assistant/turns/aw_a+b=/messages"]
    assert requests[0].url.raw_path == b"/spiral/v2/assistant/turns/aw_a+b="
    assert all(r.headers["authorization"] == "Bearer private-token" for r in requests)
    assert json.loads(requests[1].content) == {"request_id": "question-1", "message": "How is it going?"}


@pytest.mark.parametrize("work_id", [".", "..", "../tasks", "work?admin=1"])
def test_work_identity_cannot_change_the_request_path(work_id):
    with client(lambda r: pytest.fail("invalid identity reached an endpoint")) as c:
        with pytest.raises(HostClientError):
            c.get(work_id)


@pytest.mark.parametrize("host", ["http://192.168.1.3:8124", "http://spiral.example",
    "https://user:secret@spiral.example", "https://spiral.example?token=secret",
    "https://spiral.example/#fragment", "file:///tmp/host"])
def test_invalid_or_unencrypted_remote_host_is_rejected_before_network(host):
    with pytest.raises(HostClientError):
        HostClient(host)


@pytest.mark.parametrize("field,value", [("protocol_version", 3), ("protocol_version", True),
    ("minimum_client_protocol", 3), ("protocol", "unrelated"), ("capabilities", [])])
def test_incompatible_host_cannot_receive_a_mutation(field, value):
    requests = []
    def route(request):
        requests.append(request)
        return httpx.Response(200, json={**MANIFEST, field: value})
    with HostClient("http://localhost:8124", transport=httpx.MockTransport(route)) as c:
        with pytest.raises(HostClientError):
            c.start({})
    assert len(requests) == 1
    assert requests[0].method == "GET"


def test_journal_only_predecessor_is_readable_but_cannot_start_work():
    manifest = {**MANIFEST, "capabilities": ["durable_assistant_turns"]}
    requests = []
    def route(request):
        requests.append(request)
        return httpx.Response(200, json=manifest if request.url.path.endswith("manifest") else {"turns": []})
    with HostClient("http://localhost:8124", transport=httpx.MockTransport(route)) as c:
        assert c.list() == {"turns": []}
        with pytest.raises(HostClientError, match="unavailable"):
            c.start({})
    assert all(r.method == "GET" for r in requests)


def test_start_identity_is_recoverable_across_processes_and_preserves_host_workspace():
    kwargs = dict(request_id="start-1", workspace="/remote/workspace", conversation_id="phone-conversation")
    first = start_body("Fix the tests", **kwargs)
    assert first == start_body("Fix the tests", **kwargs)
    assert first["conversation_id"] == "phone-conversation"
    assert first["execution"]["run_policy"]["project_path"] == "/remote/workspace"
    assert first["execution"]["run_policy"]["create_project"] is False
    assert "model" not in first["execution"]
    assert "workspace.command" in first["execution"]["allowed_tools"]
    limited = start_body("Explain the code", read_only=True, **kwargs)
    assert limited["execution"]["allowed_tools"] == list(READ_TOOLS)
    assert not {"spiral.run", "workspace.write", "workspace.command", "network.download"} & set(
        limited["execution"]["allowed_tools"])


def test_dropped_post_is_not_repeated_or_reported_as_task_failure():
    requests = []
    def dropped(request):
        requests.append(request)
        raise httpx.ReadError("sensitive arbitrary exception", request=request)
    with client(dropped) as c:
        with pytest.raises(HostClientError, match="outcome is unknown") as error:
            c.message("aw_1", "status?", request_id="same-request")
    assert len(requests) == 1
    assert "sensitive arbitrary" not in str(error.value)


def test_redirect_never_forwards_credentials_or_repeats_post():
    requests = []
    def route(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://other.example"}, json={"error": "redirect"})
    with client(route, token="private-token") as c:
        with pytest.raises(HostClientError, match="HTTP 307"):
            c.direct("aw_1", "steer", "Keep the blue theme", request_id="steer-1")
    assert len(requests) == 1


def test_error_redacts_token_and_response_size_is_bounded(monkeypatch):
    with client(lambda r: httpx.Response(401, json={"error": "bad private-token"}), token="private-token") as c:
        with pytest.raises(HostClientError, match="redacted") as error:
            c.get("aw_1")
        assert "private-token" not in str(error.value)
    monkeypatch.setattr("spiral.host_client.MAX_RESPONSE_BYTES", 512)
    with client(lambda r: httpx.Response(200, content=b" " * 513)) as c:
        with pytest.raises(HostClientError, match="limit"):
            c.get("aw_1")


@pytest.mark.parametrize("events,cursor,more", [
    ([{"seq": 2}, {"seq": 2}], 2, False), ([{"seq": 1}], 1, False),
    ([{"seq": True}], 2, False), ([], 10, False), ([], 1, True),
    ([{"seq": 2}], 2, "false"),
])
def test_invalid_event_pages_do_not_advance_a_reconnect_cursor(events, cursor, more):
    with client(lambda r: httpx.Response(200, json={"events": events, "next_seq": cursor,
                                                  "has_more": more})) as c:
        with pytest.raises(HostClientError):
            c.events("aw_1", after=1)


def test_global_event_sequence_gaps_are_allowed_and_opaque_list_cursor_is_encoded():
    requests = []
    def route(request):
        requests.append(request)
        return httpx.Response(200, json={"events": [{"seq": 5}, {"seq": 9}], "next_seq": 9, "has_more": False})
    with client(route) as c:
        assert c.events("aw_1", after=2)["next_seq"] == 9
        c.list(conversation_id="phone", cursor="a+/=&b", delivery_state="pending")
    assert requests[1].url.params["cursor"] == "a+/=&b"
    assert requests[1].url.params["delivery_state"] == "pending"


def test_list_cursor_requires_the_host_delivery_pagination_scope():
    with client(lambda r: pytest.fail("invalid pagination reached host")) as c:
        with pytest.raises(HostClientError, match="delivery-state"):
            c.list(cursor="opaque")


def test_human_watch_uses_real_host_progress_final_message_and_clarification_wait():
    stream = io.StringIO()
    args = SimpleNamespace(interval=1, after=0, work_id="aw_1", once=False, json=False)
    pages = iter([
        {"events": [{"seq": 1, "type": "assistant_turn.action_update", "payload": {
            "progress": [{"message": "Checking the actual tests"}]}}], "next_seq": 1,
         "has_more": False, "turn": {"id": "aw_1", "status": "waiting_effect"},
         "runtime": {"clarification_question": "Which format?"}},
        {"events": [], "next_seq": 1, "has_more": False,
         "turn": {"id": "aw_1", "status": "completed", "outcome": {"message": "All tests pass."}}},
    ])
    class Client:
        def events(self, *_a, **_kw): return next(pages)
    console = Console(file=stream, color_system=None)
    for _ in range(2):
        assert session_cli.watch(Client(), args, console,
                                 sleep=lambda _: pytest.fail("wait ignored")) == 0
    assert "Checking the actual tests" in stream.getvalue()
    assert "Which format?" in stream.getvalue()
    assert "All tests pass." in stream.getvalue()


def test_watch_drains_terminal_backlog_before_finishing_without_acknowledging_delivery(capsys):
    pages = iter([
        {"events": [{"seq": 2, "type": "first"}], "next_seq": 2, "has_more": True,
         "turn": {"id": "aw_1", "status": "completed"}},
        {"events": [{"seq": 5, "type": "terminal"}], "next_seq": 5, "has_more": False,
         "turn": {"id": "aw_1", "status": "completed"}},
    ])
    cursors = []
    class Client:
        def events(self, work_id, *, after):
            cursors.append(after)
            return next(pages)
    args = SimpleNamespace(interval=1, after=0, work_id="aw_1", once=False, json=True)
    assert session_cli.watch(Client(), args, None, sleep=lambda _: pytest.fail("unnecessary wait")) == 0
    assert cursors == [0, 2]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [x["event"]["seq"] for x in lines if x["type"] == "event"] == [2, 5]
    assert lines[-1]["cursor"] == 5


def test_session_flags_work_before_and_after_action_and_never_touch_local_models(monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(cli, "print_banner", lambda *_a, **_k: pytest.fail("session banner"))
    monkeypatch.setattr(cli, "_admit_model_command", lambda *_a: pytest.fail("local model admission"))
    monkeypatch.setattr("spiral.llm.Ollama", lambda *_a, **_k: pytest.fail("local model client"))
    def run(args, console):
        observed.append(args)
        print('{"turns": []}')
        return 0
    monkeypatch.setattr(session_cli, "run", run)
    for flags in (["--json", "--host", "http://localhost:8124", "list"],
                  ["list", "--json", "--host", "http://localhost:8124"]):
        monkeypatch.setattr("sys.argv", ["spiral", "session", *flags])
        with pytest.raises(SystemExit) as exit:
            cli.entry()
        assert exit.value.code == 0
    assert all(args.json and args.host == "http://localhost:8124" for args in observed)
    assert [json.loads(x) for x in capsys.readouterr().out.splitlines()] == [{"turns": []}] * 2


def test_interrupted_watch_detaches_without_cancelling_host_work(capsys):
    class Client:
        def __init__(self, *_a, **_kw): pass
        def __enter__(self): return self
        def __exit__(self, *_a): pass
        def events(self, *_a, **_kw): raise KeyboardInterrupt()
        def direct(self, *_a, **_kw): pytest.fail("detach cancelled the host")
    args = SimpleNamespace(session_action="watch", interval=1, after=0, work_id="aw_1", once=False)
    assert session_cli.run(args, None, client_factory=Client) == 130
    assert "Host work continues" in capsys.readouterr().err


def test_display_does_not_interpret_terminal_or_rich_markup():
    stream = io.StringIO()
    session_cli._print(Console(file=stream, color_system=None), "[red]literal[/red]\x1b]8;;https://bad\x07")
    value = stream.getvalue()
    assert "[red]literal[/red]" in value
    assert "\x1b" not in value and "\x07" not in value
