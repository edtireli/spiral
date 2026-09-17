"""Academic native-model admissions remain read-only and finite."""
import io
import json

import pytest

from scripts.academic_finetune.training_support import HarnessError, verify_ollama_empty


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def test_academic_waits_only_for_natural_expiry():
    clock, requests = Clock(), []

    def opener(request, timeout):
        requests.append((request.get_method(), request.data, timeout))
        return response({"models": [] if clock() >= 2 else [{"name": "action:2s"}]})

    result = verify_ollama_empty(opener=opener, clock=clock, wait=clock.wait)
    assert result["verified_empty"] is True
    assert clock() <= 2.1
    assert all(method == "GET" and body is None for method, body, _ in requests)
    assert requests[-1][2] < requests[0][2]


def test_academic_refreshed_foreign_resident_cannot_extend_deadline():
    clock, requests = Clock(), []

    def opener(request, timeout):
        requests.append(request.get_method())
        return response({"models": [{"name": "private:foreign", "expires_at": clock() + 999}]})

    with pytest.raises(HarnessError, match="will not evict"):
        verify_ollama_empty(opener=opener, clock=clock, wait=clock.wait)
    assert clock() == 4.0
    assert set(requests) == {"GET"}


def test_academic_drain_is_cancellable_before_another_probe():
    clock, requests = Clock(), []

    def opener(request, timeout):
        requests.append(request)
        return response({"models": [{"name": "m"}]})

    with pytest.raises(HarnessError, match="cancelled"):
        verify_ollama_empty(
            opener=opener, clock=clock, wait=clock.wait, cancelled=lambda: clock() >= .1,
        )
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [{}, {"models": [None]}, {"models": [{"name": ""}]}, {"models": "unknown"}])
def test_academic_malformed_residency_never_admits(payload):
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return response(payload)

    with pytest.raises(HarnessError, match="invalid"):
        verify_ollama_empty(opener=opener)
    assert len(requests) == 1


def test_academic_unreachable_residency_is_not_retried_as_empty():
    requests = []

    def opener(request, timeout):
        requests.append(request)
        raise ConnectionRefusedError("offline")

    with pytest.raises(HarnessError, match="cannot verify"):
        verify_ollama_empty(opener=opener)
    assert len(requests) == 1


def test_academic_oversize_residency_is_rejected_before_parsing():
    with pytest.raises(HarnessError, match="exceeded 1 MiB"):
        verify_ollama_empty(opener=lambda request, timeout: io.BytesIO(b"x" * ((1 << 20) + 1)))


def test_text_and_vlm_use_the_packaged_guard_inside_compute_admission():
    from scripts.academic_finetune import serve_adapter, serve_vlm_adapter
    assert serve_adapter.verify_ollama_empty is verify_ollama_empty
    assert serve_vlm_adapter.verify_ollama_empty is verify_ollama_empty
