"""A successful HTTP exchange is not evidence that local inference completed."""
import json

import httpx
import pytest

from spiral.llm import Ollama, _prompt_token_reserve


MESSAGES = [{"role": "user", "content": "finish the task"}]
COMPLETE = {
    "message": {"content": "complete"}, "done": True,
    "prompt_eval_count": 3, "eval_count": 4,
}


def _stream(*frames):
    return "".join(json.dumps(frame) + "\n" for frame in frames).encode()


def _client(handler, monkeypatch, *, attempts=2):
    client = Ollama(providers={})
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.local_model_retry_attempts = attempts
    client.configure_budget(wall_seconds=10, total_tokens=10_000, model_calls=8)
    monkeypatch.setattr("spiral.llm.time.sleep", lambda _delay: None)
    return client


@pytest.mark.parametrize("failed_frames", [
    [],
    [{"message": {"content": "unfinished"}}],
    [{"message": {"content": "unfinished"}}, {"error": "runner crashed"}],
    [{"error": "runner crashed", "done": True}],
    [{"done": "true", "message": {"content": "unfinished"}}],
    [None],
    [{"message": []}],
    [{"message": {"content": ["bad type"]}}],
])
def test_incomplete_or_failed_inference_is_replayed_and_not_delivered(
        failed_frames, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        frames = failed_frames if len(calls) == 1 else [COMPLETE]
        return httpx.Response(200, content=_stream(*frames))

    deltas = []
    with _client(handler, monkeypatch) as client:
        result = client.chat(
            "local:27b", MESSAGES, num_predict=10,
            on_delta=lambda kind, piece: deltas.append((kind, piece)),
        )
        assert client.budget.calls == 2
        assert client.budget.total_tokens == _prompt_token_reserve(MESSAGES) + 10 + 7

    assert len(calls) == 2
    assert result.text == "complete"
    assert result.raw["spiral_local_transport_attempts"] == 2
    assert deltas.count(("reset", "")) == 1
    visible = ""
    for kind, text in deltas:
        if kind == "reset":
            visible = ""
        elif kind == "text":
            visible += text
    assert visible == "complete"


def test_repeated_clean_truncation_exhausts_retries_instead_of_returning_success(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=_stream({"message": {"content": "partial"}}))

    with _client(handler, monkeypatch) as client:
        with pytest.raises(httpx.RemoteProtocolError, match="completion"):
            client.chat("local:27b", MESSAGES, num_predict=10, on_delta=lambda *_: None)
        assert client.budget.calls == 2
        assert client.budget.total_tokens == 2 * (_prompt_token_reserve(MESSAGES) + 10)
    assert len(calls) == 2


@pytest.mark.parametrize("frames, error", [
    ([{"message": {"content": "partial"}}], "completion"),
    ([{"message": {"content": "partial"}}, {"error": "runner crashed"}], "runner crashed"),
])
def test_generator_reports_failure_without_replaying_already_yielded_text(frames, error, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=_stream(*frames))

    with _client(handler, monkeypatch) as client:
        stream = client.chat_stream("local:27b", MESSAGES, num_predict=10)
        assert next(stream) == "partial"
        with pytest.raises(httpx.RemoteProtocolError, match=error):
            next(stream)
        assert client.budget.total_tokens == _prompt_token_reserve(MESSAGES) + 10
    assert len(calls) == 1


def test_successful_empty_completion_is_not_confused_with_missing_completion(monkeypatch):
    completed = {**COMPLETE, "message": {"content": ""}}
    with _client(lambda _: httpx.Response(200, content=_stream(completed)), monkeypatch) as client:
        result = client.chat("local:27b", MESSAGES, num_predict=10, on_delta=lambda *_: None)
        assert result.text == ""
        assert result.raw["done"] is True
        assert client.budget.calls == 1


def test_blocking_in_band_error_is_recovered(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"error": "runner crashed"} if len(calls) == 1 else COMPLETE)

    with _client(handler, monkeypatch) as client:
        result = client.chat("local:27b", MESSAGES, num_predict=10)
    assert result.text == "complete"
    assert len(calls) == 2


def test_blocking_model_remembers_unsupported_thinking_toggle(monkeypatch):
    payloads = []

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.32.14"})
        payload = json.loads(request.content)
        payloads.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": '"plain:12b" does not support thinking'})
        return httpx.Response(200, json=COMPLETE)

    with _client(handler, monkeypatch) as client:
        for _ in range(2):
            assert client.chat("plain:12b", MESSAGES, num_predict=10).text == "complete"
    assert len(payloads) == 3, "later requests should skip the already-rejected toggle"
    assert "think" in payloads[0]
    assert all("think" not in payload for payload in payloads[1:])
