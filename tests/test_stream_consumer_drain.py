"""Consumer failure is not an inference failure or a license to replay it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import httpx
import pytest

from spiral.llm import InferenceLease, InferenceLeaseTimeout, Ollama


@pytest.fixture
def client(tmp_path):
    instance = Ollama(providers={})
    instance.inference_lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    instance.configure_budget(wall_seconds=10, total_tokens=10000, model_calls=10)
    yield instance
    instance.close()


def marker(tmp_path):
    return json.loads((tmp_path / "spiral-compute.inference.json").read_text())


def wire(*frames):
    return "".join(json.dumps(frame) + "\n" for frame in frames).encode()


PARTIAL = {"message": {"content": "first"}}
TAIL = {"message": {"thinking": "private", "content": "not published"}}
DONE = {"done": True, "prompt_eval_count": 3, "eval_count": 4}


def install(client, handler):
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("error", [RuntimeError("consumer"), httpx.ReadTimeout("consumer"),
                                  KeyboardInterrupt(), SystemExit(7)])
def test_consumer_stop_drains_same_request_without_publication_or_retry(client, tmp_path, error):
    requests, visible = [], []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=wire(PARTIAL, TAIL, DONE))
    install(client, handler)
    def publish(kind, piece):
        visible.append((kind, piece))
        assert marker(tmp_path)["state"] == "dispatching"
        raise error
    with pytest.raises(type(error)) as caught:
        client.chat("exact:test", [], num_predict=8, on_delta=publish)
    assert caught.value is error
    assert visible == [("text", "first")]
    assert len(requests) == 1
    assert client.budget.calls == 1
    assert client.budget.total_tokens >= 8  # No free interrupted inference.
    assert marker(tmp_path)["state"] == "drained"
    with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="next:test"):
        pass


@pytest.mark.parametrize("tail", [(), ({"error": "backend failed"},), ({"done": "yes"},)])
def test_incomplete_drain_preserves_consumer_error_and_quarantine(client, tmp_path, tail):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=wire(PARTIAL, *tail))
    install(client, handler)
    failure = httpx.ReadTimeout("publication failed, not the backend")
    def publish(*_):
        raise failure
    with pytest.raises(httpx.ReadTimeout) as caught:
        client.chat("exact:test", [], num_predict=8, on_delta=publish)
    assert caught.value is failure
    assert len(requests) == 1
    assert marker(tmp_path)["state"] == "uncertain"


def test_closed_generator_discards_tail_and_accounts_actual_terminal(client, tmp_path):
    install(client, lambda _: httpx.Response(200, content=wire(PARTIAL, TAIL, DONE)))
    response = client.chat_stream("exact:test", [], num_predict=8)
    assert next(response) == "first"
    response.close()
    assert marker(tmp_path)["state"] == "drained"
    assert client.budget.total_tokens == 7


def test_stop_holds_real_machine_lease_until_terminal_and_reader_close(client, tmp_path):
    stopped = threading.Event()
    checks = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield wire(PARTIAL)
            assert stopped.wait(2)
            checks.append(marker(tmp_path)["state"])
            with pytest.raises(InferenceLeaseTimeout):
                with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="foreign:test"):
                    pytest.fail("lease escaped while draining")
            yield wire(DONE)
        def close(self):
            checks.append("closed")
    install(client, lambda _: httpx.Response(200, stream=Stream()))
    def publish(*_):
        stopped.set()
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        client.chat("exact:test", [], num_predict=8, on_delta=publish)
    assert checks[0] == "dispatching"
    assert "closed" in checks
    assert marker(tmp_path)["state"] == "drained"


def test_cooperative_drain_is_bounded_and_never_claims_timeout_as_release(client, tmp_path, monkeypatch):
    monkeypatch.setattr("spiral.llm._LOCAL_CONSUMER_DRAIN_SECONDS", 0.03, raising=False)
    closed = threading.Event()
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield wire(PARTIAL)
            closed.wait(2)
        def close(self):
            closed.set()
    install(client, lambda _: httpx.Response(200, stream=Stream()))
    started = time.monotonic()
    def publish(*_):
        raise RuntimeError("stop")
    with pytest.raises(RuntimeError, match="stop"):
        client.chat("exact:test", [], num_predict=8, on_delta=publish)
    elapsed = time.monotonic() - started
    assert 0.02 <= elapsed < 0.8
    assert closed.is_set()
    assert marker(tmp_path)["state"] == "uncertain"


def test_drain_never_extends_original_request_wall_allowance(client, tmp_path, monkeypatch):
    monkeypatch.setattr(client, "_remaining_request_timeout", lambda: 0.03)
    closed = threading.Event()
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield wire(PARTIAL)
            closed.wait(2)
        def close(self):
            closed.set()
    install(client, lambda _: httpx.Response(200, stream=Stream()))
    def publish(*_):
        raise RuntimeError("stop")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="stop"):
        client.chat("exact:test", [], num_predict=8, on_delta=publish)
    assert 0.02 <= time.monotonic() - started < 0.8
    assert marker(tmp_path)["state"] == "uncertain"


def test_repeat_interrupt_does_not_replenish_drain_window(monkeypatch):
    from spiral.llm import _LocalStreamStop
    now = [100.0]
    monkeypatch.setattr("spiral.llm.time.monotonic", lambda: now[0])
    stopped = _LocalStreamStop()
    original = KeyboardInterrupt()
    stopped.capture(original)
    deadline = stopped.deadline
    now[0] += 3
    stopped.capture(KeyboardInterrupt())
    assert stopped.deadline == deadline
    assert stopped.error is original


@pytest.mark.parametrize("generator", [False, True])
@pytest.mark.skipif(os.name != "posix", reason="real SIGINT receipt uses POSIX signals")
def test_real_sigint_at_reader_wait_drains_without_publishing(tmp_path, generator):
    # Signal only our subprocess, at a deterministic reader-wait seam. No real
    # model, Ollama daemon, or shared production compute marker is touched.
    program = r'''
import json, os, queue, signal, sys, threading
from pathlib import Path
import httpx
from spiral.llm import Ollama, InferenceLease
root = Path(sys.argv[1])
client = Ollama(providers={})
client.configure_budget(wall_seconds=10, total_tokens=10000, model_calls=10)
client.inference_lease = InferenceLease(root / "spiral-compute.lease", timeout=0)
client._client.close()
client._client = httpx.Client(transport=httpx.MockTransport(lambda request:
    httpx.Response(200, content='{"message":{"content":"discard"}}\n{"done":true,"eval_count":2}\n')))
original_get = queue.Queue.get
signalled = []
def interrupted_get(self, *args, **kwargs):
    if threading.current_thread() is threading.main_thread() and not signalled:
        signalled.append(True)
        os.kill(os.getpid(), signal.SIGINT)
    return original_get(self, *args, **kwargs)
queue.Queue.get = interrupted_get
visible = []
try:
    if sys.argv[2] == "True":
        visible.extend(client.chat_stream("exact:test", [], num_predict=8))
    else:
        client.chat("exact:test", [], num_predict=8, on_delta=lambda *p: visible.append(p))
    raise AssertionError("SIGINT was swallowed")
except KeyboardInterrupt:
    assert not visible
    assert signalled
    assert client.budget.calls == 1
    assert json.loads((root / "spiral-compute.inference.json").read_text())["state"] == "drained"
finally:
    client.close()
'''
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path), str(generator)],
                            cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "SPIRAL_OFFLINE_TESTS": "1"},
                            capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
