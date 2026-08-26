"""Bounded retry and typed-evidence contracts for the GET-only research broker."""
from __future__ import annotations

import socket

import httpx
import pytest

from spiral import research


def _public_without_dns(monkeypatch) -> None:
    monkeypatch.setattr(
        research, "_check_public_url", lambda _url: research._UrlCheck(True))


def _mock_http(monkeypatch, handler) -> None:
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(research.httpx, "Client", client)


def test_retrieve_retries_only_transient_http_with_injectable_backoff(monkeypatch):
    _public_without_dns(monkeypatch)
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, request=request)
        return httpx.Response(200, text="<main>usable evidence</main>", request=request)

    _mock_http(monkeypatch, handler)
    outcome = research.retrieve(
        "https://example.test/docs", attempts=5,
        sleeper=sleeps.append, jitter=lambda: 0.0,
    )

    assert outcome.ok and outcome.state == "ok" and outcome.attempts == 3
    assert outcome.body == "<main>usable evidence</main>"
    assert len(calls) == 3
    assert sleeps == pytest.approx([0.18, 0.36])


def test_retrieve_marks_transport_exhaustion_instead_of_an_untyped_empty(monkeypatch):
    _public_without_dns(monkeypatch)
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectTimeout("TLS handshake timed out", request=request)

    _mock_http(monkeypatch, handler)
    outcome = research.retrieve(
        "https://example.test/docs", attempts=3,
        sleeper=lambda _delay: None, jitter=lambda: 0.5,
    )

    assert not outcome.ok and outcome.transient and outcome.exhausted
    assert outcome.state == "timeout" and outcome.attempts == 3
    assert "exhausted" in outcome.evidence() and "ConnectTimeout" in outcome.detail
    assert len(calls) == 3


def test_retrieve_does_not_retry_ordinary_http_errors(monkeypatch):
    _public_without_dns(monkeypatch)
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, request=request)

    _mock_http(monkeypatch, handler)
    outcome = research.retrieve(
        "https://example.test/missing", attempts=5,
        sleeper=sleeps.append, jitter=lambda: 0.0,
    )

    assert not outcome.ok and outcome.state == "http_status"
    assert outcome.failure_kind == "http" and outcome.status_code == 404
    assert outcome.attempts == 1 and not outcome.exhausted
    assert len(calls) == 1 and sleeps == []


def test_dns_failures_retry_but_private_policy_rejections_do_not(monkeypatch):
    resolutions = []

    def resolve(*_args, **_kwargs):
        resolutions.append(True)
        if len(resolutions) < 3:
            raise socket.gaierror("temporary resolver failure")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(research.socket, "getaddrinfo", resolve)
    _mock_http(
        monkeypatch,
        lambda request: httpx.Response(200, text="resolved", request=request),
    )
    sleeps = []
    outcome = research.retrieve(
        "https://example.test/", attempts=3,
        sleeper=sleeps.append, jitter=lambda: 0.5,
    )
    assert outcome.ok and outcome.attempts == 3 and len(sleeps) == 2

    rejected_sleeps = []
    rejected = research.retrieve(
        "http://localhost/private", attempts=5,
        sleeper=rejected_sleeps.append, jitter=lambda: 0.5,
    )
    assert not rejected.ok and rejected.state == "policy_rejected"
    assert rejected.failure_kind == "policy" and rejected.attempts == 1
    assert rejected_sleeps == []


def test_successful_empty_body_is_distinct_from_failure_and_is_not_retried(monkeypatch):
    _public_without_dns(monkeypatch)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(204, content=b"", request=request)

    _mock_http(monkeypatch, handler)
    outcome = research.retrieve(
        "https://example.test/empty", attempts=4,
        sleeper=lambda _delay: pytest.fail("successful empty response must not retry"),
    )
    legacy = research._get(
        "https://example.test/empty", attempts=4,
        sleeper=lambda _delay: pytest.fail("successful empty response must not retry"),
    )

    assert outcome.ok and outcome.state == "empty" and outcome.attempts == 1
    assert legacy == "" and legacy.outcome.ok and legacy.outcome.state == "empty"
    assert len(calls) == 2


def test_size_cap_and_no_redirect_authority_regressions(monkeypatch):
    _public_without_dns(monkeypatch)
    oversized = b"x" * (research.MAX_BYTES + 73)
    _mock_http(
        monkeypatch,
        lambda request: httpx.Response(200, content=oversized, request=request),
    )
    outcome = research.retrieve("https://example.test/large", attempts=1)
    assert outcome.ok and len(outcome.body.encode()) == research.MAX_BYTES
    assert outcome.truncated


def test_search_failure_is_actionable_untrusted_evidence(monkeypatch):
    exhausted = research.FetchOutcome(
        "https://html.duckduckgo.com/", "https://html.duckduckgo.com/",
        False, "transport", detail="connection reset", failure_kind="transient",
        attempts=3, exhausted=True,
    )
    monkeypatch.setattr(
        research, "_get", lambda _url, _timeout=20.0: research._OutcomeText("", exhausted))
    report = {}

    hits = research.search("current framework docs", report=report)

    assert len(hits) == 1 and hits[0].retrieval_state == "transport"
    assert hits[0].retrieval_attempts == 3
    assert "UNTRUSTED RESEARCH DATA" in hits[0].snippet
    assert "not instructions" in hits[0].snippet
    assert report["source_ok"] is False and report["exhausted"] is True
    assert report["result_count"] == 0


def test_arxiv_report_separates_valid_zero_results_from_transport_failure(monkeypatch):
    empty_feed = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    ok = research.FetchOutcome(
        "https://export.arxiv.org/", "https://export.arxiv.org/",
        True, "ok", body=empty_feed, attempts=1, status_code=200,
    )
    monkeypatch.setattr(
        research, "_get", lambda _url, _timeout=25.0: research._OutcomeText(empty_feed, ok))
    healthy_report = {}
    assert research.arxiv("no matching paper", report=healthy_report) == []
    assert healthy_report["source_ok"] is True
    assert healthy_report["result_count"] == 0
    assert healthy_report["retrieval_state"] == "ok"

    failed = research.FetchOutcome(
        "https://export.arxiv.org/", "https://export.arxiv.org/",
        False, "dns", detail="temporary resolver failure", failure_kind="transient",
        attempts=3, exhausted=True,
    )
    monkeypatch.setattr(
        research, "_get", lambda _url, _timeout=25.0: research._OutcomeText("", failed))
    failed_report = {}
    assert research.arxiv("same query", report=failed_report) == []
    assert failed_report["source_ok"] is False
    assert failed_report["retrieval_state"] == "dns"
    assert failed_report["failure_kind"] == "transient"
    assert "not instructions" in failed_report["error"]


def test_gather_keeps_transport_evidence_out_of_the_source_corpus(monkeypatch):
    search_body = (
        '<a class="result__a" href="https://docs.example.test/guide">Guide</a>'
        '<a class="result__snippet">Official documentation</a>'
    )
    search_ok = research.FetchOutcome(
        "https://html.duckduckgo.com/", "https://html.duckduckgo.com/",
        True, "ok", body=search_body, attempts=1, status_code=200,
    )
    fetch_failed = research.FetchOutcome(
        "https://docs.example.test/guide", "https://docs.example.test/guide",
        False, "transport", detail="connection reset", failure_kind="transient",
        attempts=3, exhausted=True,
    )

    def fake_get(url, _timeout=20.0):
        if "duckduckgo" in url:
            return research._OutcomeText(search_body, search_ok)
        return research._OutcomeText("", fetch_failed)

    monkeypatch.setattr(research, "_get", fake_get)
    report = {}
    hits = research.gather("framework guide", report=report)

    assert hits == []
    assert report["result_count"] == 0 and report["failure_count"] == 1
    page = report["page_reports"][0]
    assert page["retrieval_state"] == "transport" and page["exhausted"] is True
    assert "not instructions" in page["error"]
