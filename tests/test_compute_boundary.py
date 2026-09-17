"""Host/engine compatibility for the persistent compute-admission wire fence."""
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx

import pytest

from spiral.llm import InferenceLease, InferenceLeaseTimeout, Ollama


@pytest.fixture
def fenced_client(tmp_path):
    client = Ollama(providers={})
    client.inference_lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    client.configure_budget(wall_seconds=30, total_tokens=10000, model_calls=10)
    yield client
    client.close()


def transport(client, handler):
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))


def record(root):
    return json.loads((root / "spiral-compute.inference.json").read_text())


@pytest.mark.parametrize("stream", [False, True])
def test_success_is_registered_before_post_and_settled_after_terminal_close(fenced_client, tmp_path, stream):
    client = fenced_client
    attempts = []
    def handler(request):
        marker = record(tmp_path)
        assert marker["state"] == "dispatching"
        assert marker["model"] == "exact:test"
        assert "PRIVATE-PROMPT" not in json.dumps(marker)
        attempts.append(marker["attempt_id"])
        done = {"done": True, "message": {"content": "result"}, "eval_count": 1, "prompt_eval_count": 1}
        return httpx.Response(200, content=json.dumps(done) + ("\n" if stream else ""))
    transport(client, handler)
    for _ in range(2):
        result = client.chat("exact:test", [{"role": "user", "content": "PRIVATE-PROMPT"}],
                             num_predict=8, on_delta=(lambda *_: None) if stream else None)
        assert result.text == "result"
        assert record(tmp_path)["state"] == "drained"
    assert attempts[0] != attempts[1]


@pytest.mark.parametrize("failure", ["timeout", "http500", "eof", "invalid", "error"])
@pytest.mark.parametrize("stream", [False, True])
def test_unknown_dispatch_cannot_retry_or_disappear_on_engine_restart(fenced_client, tmp_path, failure, stream):
    calls = []
    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("lost acknowledgment")
        if failure == "http500":
            return httpx.Response(500, json={"error": "runner failed"})
        content = {"eof": '{"message":{"content":"unfinished"}}',
                   "invalid": '{"done":', "error": '{"error":"runner failed"}'}[failure]
        return httpx.Response(200, content=content + ("\n" if stream else ""))
    transport(fenced_client, handler)
    with pytest.raises(InferenceLeaseTimeout, match="reconciliation required"):
        fenced_client.chat("exact:test", [], num_predict=8, on_delta=(lambda *_: None) if stream else None)
    assert len(calls) == 1
    original = record(tmp_path)
    assert original["state"] == "uncertain"
    with pytest.raises(InferenceLeaseTimeout):
        with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="another:test"):
            pytest.fail("unknown request was admitted after restart")
    assert record(tmp_path) == original
    assert fenced_client.budget.calls == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("version,exact,allowed", [("0.32.14", True, True), ("other", True, False),
                                                  ("0.32.14", False, False)])
def test_only_versioned_exact_thinking_rejection_replays(fenced_client, tmp_path, stream, version, exact, allowed):
    posts = []
    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        body = json.loads(request.content)
        posts.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"exact:test" does not support thinking' if exact else "bad request"})
        done = {"done": True, "message": {"content": "result"}}
        return httpx.Response(200, content=json.dumps(done) + ("\n" if stream else ""))
    transport(fenced_client, handler)
    if allowed:
        assert fenced_client.chat("exact:test", [], num_predict=8,
            on_delta=(lambda *_: None) if stream else None).text == "result"
        assert len(posts) == 2
        assert record(tmp_path)["state"] == "drained"
    else:
        with pytest.raises(httpx.HTTPStatusError):
            fenced_client.chat("exact:test", [], num_predict=8, on_delta=(lambda *_: None) if stream else None)
        assert len(posts) == 1
        assert record(tmp_path)["state"] == "uncertain"


def test_generator_abandonment_has_no_backend_stop_claim(fenced_client, tmp_path):
    transport(fenced_client, lambda _: httpx.Response(200, content='{"message":{"content":"partial"}}\n'))
    generator = fenced_client.chat_stream("exact:test", [], num_predict=8)
    assert next(generator) == "partial"
    generator.close()
    assert record(tmp_path)["state"] == "uncertain"


def test_request_outside_current_machine_lease_is_rejected(tmp_path):
    lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    with pytest.raises(RuntimeError, match="current machine lease"):
        lease.request(model="exact:test", operation="engine_chat")
    assert not (tmp_path / "spiral-compute.inference.json").exists()


def test_failed_settlement_write_keeps_dispatch_marker_and_cleans_temporary(tmp_path, monkeypatch):
    lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    with lease.hold(model="exact:test"):
        with pytest.raises(OSError, match="disk fault"):
            with lease.request(model="exact:test", operation="engine_chat") as boundary:
                boundary.terminal = True
                boundary.reader_closed = True
                def fail_replace(*_):
                    raise OSError("disk fault")
                monkeypatch.setattr("spiral.inference_boundary.os.replace", fail_replace)
    assert record(tmp_path)["state"] == "dispatching"
    assert not list(tmp_path.glob(".inference-*"))


def test_stale_settlement_cannot_clear_different_nonce(tmp_path):
    lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    with lease.hold(model="exact:test"):
        with pytest.raises(RuntimeError, match="stale"):
            with lease.request(model="exact:test", operation="engine_chat") as boundary:
                other = {**record(tmp_path), "attempt_id": "other-owner"}
                (tmp_path / "spiral-compute.inference.json").write_text(json.dumps(other))
                boundary.terminal = True
                boundary.reader_closed = True
    assert record(tmp_path)["attempt_id"] == "other-owner"
    assert record(tmp_path)["state"] == "dispatching"


def test_real_process_death_keeps_dispatch_fenced_after_kernel_unlock(tmp_path):
    code = """
import os, sys
from spiral.llm import InferenceLease
lease = InferenceLease(sys.argv[1], timeout=0)
with lease.hold(model='exact:test'):
    with lease.request(model='exact:test', operation='engine_chat'):
        os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "spiral-compute.lease")],
                            cwd=Path(__file__).resolve().parents[1], env={**os.environ, "SPIRAL_OFFLINE_TESTS": "1"},
                            timeout=10, capture_output=True)
    assert result.returncode == 17, result.stderr.decode()
    assert record(tmp_path)["state"] == "dispatching"
    with pytest.raises(InferenceLeaseTimeout):
        with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="another:test"):
            pytest.fail("process death is not inference drainage")


@pytest.mark.parametrize("state", ["dispatching", "uncertain", "future", None])
def test_engine_does_not_enter_or_overwrite_unknown_host_attempt(tmp_path, state):
    record = {"schema": "spiral.compute.inference.v1", "attempt_id": "host-one", "state": state}
    marker = tmp_path / "spiral-compute.inference.json"
    marker.write_text(json.dumps(record))
    lease = InferenceLease(tmp_path / "spiral-compute.lease", timeout=0)
    with pytest.raises(InferenceLeaseTimeout, match="reconciliation required"):
        with lease.hold(model="exact:test"):
            pytest.fail("must not start inference")
    assert json.loads(marker.read_text()) == record


@pytest.mark.parametrize("state", ["drained", "not_dispatched", "rejected"])
def test_engine_accepts_only_settled_current_protocol_records(tmp_path, state):
    marker = tmp_path / "spiral-compute.inference.json"
    marker.write_text(json.dumps({"schema": "spiral.compute.inference.v1",
                                 "attempt_id": "host-one", "state": state}))
    with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="exact:test"):
        pass


@pytest.mark.parametrize("raw", [b"", b"x" * 9000, b"\xff", b"{}",
    b'{"schema":"future","state":"drained","attempt_id":"one"}'])
def test_engine_fails_closed_for_corrupt_or_future_boundary(tmp_path, raw):
    (tmp_path / "spiral-compute.inference.json").write_bytes(raw)
    with pytest.raises(InferenceLeaseTimeout):
        with InferenceLease(tmp_path / "spiral-compute.lease", timeout=0).hold(model="exact:test"):
            pytest.fail("must not start inference")
