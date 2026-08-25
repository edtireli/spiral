"""Model-free receipts for exact-owned Ollama terminal cleanup."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import types

import pytest

from spiral import cli
from spiral import llm
from spiral.llm import ChatResult, Ollama


class _Response:
    status_code = 200

    @staticmethod
    def raise_for_status():
        return None

    @staticmethod
    def json():
        return {
            "message": {"content": "done"},
            "prompt_eval_count": 1,
            "eval_count": 1,
            "done_reason": "stop",
        }


class _CleanupClient:
    def __init__(self, base_url, timeout, providers, events):
        events.append(("open", base_url, timeout, providers))
        self.base_url = base_url
        self.events = events

    def evict(self, model, *, timeout=None):
        self.events.append(("evict", self.base_url, model, timeout))
        return True

    def close(self):
        self.events.append(("close", self.base_url))


def _release_with_fake(monkeypatch):
    events = []
    # Every transport is a fake in this test; clearing the hard-offline switch is
    # necessary only so the cleanup factory itself can be observed.
    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS", raising=False)
    released = llm.release_owned_local_models(
        client_factory=lambda base_url, timeout, providers: _CleanupClient(
            base_url, timeout, providers, events,
        )
    )
    return released, events


def test_terminal_cleanup_evicts_only_exact_local_models_used_by_run(monkeypatch):
    llm.begin_owned_local_model_run()
    local = Ollama("http://ollama.test", providers={})
    local._client = types.SimpleNamespace(post=lambda *_args, **_kwargs: _Response())
    assert local.chat(
        "owned:27b", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "done"

    remote = Ollama(providers={"remote": {"base_url": "https://provider.test/v1"}})
    remote._openai_chat = lambda *_args, **_kwargs: ChatResult(
        text="remote", prompt_tokens=1, completion_tokens=1,
    )
    assert remote.chat(
        "remote", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "remote"

    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "owned:27b")]
    assert any(
        event[:3] == ("evict", "http://ollama.test", "owned:27b")
        for event in events
    )
    assert not any("remote" in str(event) for event in events)


def test_deliberate_role_switch_evicts_only_prior_run_owned_model(monkeypatch):
    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(post=lambda *_args, **_kwargs: _Response())
    assert client.chat(
        "prior:27b", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "done"

    evicted = []

    def evict(model):
        evicted.append(model)
        return True

    monkeypatch.setattr(client, "evict", evict)
    monkeypatch.setattr(
        client, "resident",
        lambda: (_ for _ in ()).throw(
            AssertionError("owned role switch enumerated unrelated residents")
        ),
    )
    assert client.evict_owned_local_models_except(
        {"next:12b"}, strict=True) == ["prior:27b"]
    assert evicted == ["prior:27b"]

    assert client.chat(
        "next:12b", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "done"
    assert client.evict_owned_local_models_except({"next:12b"}, strict=True) == []
    assert evicted == ["prior:27b"]
    assert not hasattr(client, "free_foreign")

    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "next:12b")]
    terminal_evicts = [event for event in events if event[0] == "evict"]
    assert len(terminal_evicts) == 1
    assert terminal_evicts[0][:3] == ("evict", "http://ollama.test", "next:12b")


def test_failed_owned_role_switch_restores_receipt_for_terminal_cleanup(monkeypatch):
    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(post=lambda *_args, **_kwargs: _Response())
    assert client.chat(
        "prior:27b", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "done"
    monkeypatch.setattr(client, "evict", lambda _model: False)

    assert client.evict_owned_local_models_except({"next:12b"}) == []
    with pytest.raises(llm.OwnedLocalModelEvictionError, match="prior:27b"):
        client.evict_owned_local_models_except({"next:12b"}, strict=True)
    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "prior:27b")]
    assert any(
        event[:3] == ("evict", "http://ollama.test", "prior:27b")
        for event in events
    )


def test_strict_owned_role_switch_wraps_unload_exception_and_restores_receipt(
        monkeypatch):
    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(post=lambda *_args, **_kwargs: _Response())
    assert client.chat(
        "prior:27b", [{"role": "user", "content": "work"}], num_predict=4,
    ).text == "done"

    def broken_evict(_model):
        raise RuntimeError("transport failed")

    monkeypatch.setattr(client, "evict", broken_evict)
    with pytest.raises(llm.OwnedLocalModelEvictionError, match="prior:27b") as failure:
        client.evict_owned_local_models_except(set(), strict=True)
    assert isinstance(failure.value.__cause__, RuntimeError)

    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "prior:27b")]
    assert any(
        event[:3] == ("evict", "http://ollama.test", "prior:27b")
        for event in events
    )


def test_request_blocked_before_dispatch_creates_no_ownership_receipt(monkeypatch):
    class BlockedScope:
        def __enter__(self):
            raise RuntimeError("lane unavailable")

        def __exit__(self, *_args):
            return False

    class BlockedLease:
        @staticmethod
        def hold(**_kwargs):
            return BlockedScope()

    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client.inference_lease = BlockedLease()
    client._client = types.SimpleNamespace(
        post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport must not run without the lease")
        ),
    )
    with pytest.raises(RuntimeError, match="lane unavailable"):
        client.chat(
            "not-dispatched:27b", [{"role": "user", "content": "work"}],
            num_predict=4,
        )

    released, events = _release_with_fake(monkeypatch)
    assert released == []
    assert not any(event[0] == "evict" for event in events)


def test_residency_preflight_fails_closed_without_mutating_foreign_model(monkeypatch):
    from spiral.config import Config

    cfg = Config()
    cfg.base_url = "http://ollama.test"
    events = []

    class Probe:
        def __init__(self, base_url, timeout=1200.0, providers=None):
            events.append(("open", base_url, timeout, providers))
            self.base_url = base_url

        def resident_strict(self):
            events.append(("resident", self.base_url))
            return ["phone-owned:27b"] if self.base_url == cfg.base_url else []

        def close(self):
            events.append(("close", self.base_url))

        def __getattr__(self, name):
            if name in {"evict", "free_foreign"}:
                raise AssertionError(f"preflight attempted mutating operation {name}")
            raise AttributeError(name)

    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS", raising=False)
    monkeypatch.setattr(cli, "Ollama", Probe)
    messages = []
    for _ in range(2):
        with pytest.raises(SystemExit) as stopped:
            cli._free_foreign_models(None, cfg)
        messages.append(str(stopped.value))

    assert messages[0] == messages[1]
    assert "phone-owned:27b at http://ollama.test" in messages[0]
    assert "will not claim or evict it" in messages[0]
    assert {event[0] for event in events} == {"open", "resident", "close"}


def test_residency_preflight_rejects_preexisting_same_configured_name(monkeypatch):
    from spiral.config import Config

    cfg = Config()
    cfg.base_url = "http://ollama.test"
    configured_name = cfg.worker.name
    mutations = []

    class Probe:
        def __init__(self, base_url, timeout=1200.0, providers=None):
            self.base_url = base_url

        def resident_strict(self):
            return [configured_name] if self.base_url == cfg.base_url else []

        @staticmethod
        def close():
            return None

        def evict(self, model):
            mutations.append(model)
            raise AssertionError("preflight tried to claim a same-name model")

    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS", raising=False)
    monkeypatch.setattr(cli, "Ollama", Probe)
    with pytest.raises(SystemExit) as stopped:
        cli._free_foreign_models(None, cfg)

    assert configured_name in str(stopped.value)
    assert "resident before this run began" in str(stopped.value)
    assert mutations == []


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timed out"), RuntimeError("HTTP 500"), ValueError("bad JSON")],
)
def test_residency_preflight_fails_closed_when_strict_probe_is_unknown(
        monkeypatch, failure):
    from spiral.config import Config

    cfg = Config()
    cfg.base_url = "http://ollama.test"
    events = []

    class Probe:
        def __init__(self, base_url, timeout=1200.0, providers=None):
            self.base_url = base_url

        def resident_strict(self):
            events.append(("strict", self.base_url))
            if self.base_url == cfg.base_url:
                raise failure
            return []

        def resident(self):
            raise AssertionError("admission fell back to best-effort residency")

        def close(self):
            events.append(("close", self.base_url))

        def evict(self, model):
            raise AssertionError(f"admission tried to evict {model}")

    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS", raising=False)
    monkeypatch.setattr(cli, "Ollama", Probe)
    with pytest.raises(SystemExit) as stopped:
        cli._free_foreign_models(None, cfg)

    assert "could not prove Ollama residency is safe" in str(stopped.value)
    assert "No model was unloaded" in str(stopped.value)
    assert ("strict", cfg.base_url) in events


def test_strict_residency_rejects_unknown_shape_while_display_stays_best_effort():
    class MalformedResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"unexpected": []}

    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(get=lambda *_args, **_kwargs: MalformedResponse())
    with pytest.raises(ValueError, match="unknown response shape"):
        client.resident_strict()
    assert client.resident() == []


@pytest.mark.parametrize("argv", [
    ["spiral", "research", "find evidence", "--solve"],
    ["spiral", "research", "answer this"],
    ["spiral", "do", "fix it", "--verify", "true"],
    ["spiral", "chat", "hello"],
    ["spiral", "prose", "paper.md", "--rewrite"],
    ["spiral", "build", "make it"],
])
def test_shared_cli_admission_precedes_every_model_bearing_branch(
        monkeypatch, argv):
    from spiral.config import Config

    cfg = Config()
    admitted = []
    monkeypatch.setattr(cli, "print_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(sys, "argv", argv)

    def stop_at_admission(_console, observed_cfg):
        admitted.append(observed_cfg)
        raise SystemExit("admission boundary")

    monkeypatch.setattr(cli, "_free_foreign_models", stop_at_admission)
    with pytest.raises(SystemExit, match="admission boundary"):
        cli.main()
    assert admitted == [cfg]


@pytest.mark.parametrize("args", [
    types.SimpleNamespace(cmd=None),
    types.SimpleNamespace(cmd="doctor"),
    types.SimpleNamespace(cmd="research", history=True),
    types.SimpleNamespace(cmd="research", graph=True),
    types.SimpleNamespace(cmd="research", audit=True),
    types.SimpleNamespace(cmd="research", taste_like="good angle"),
    types.SimpleNamespace(cmd="prose", rewrite=False, deep=False, beef_up=False,
                          restructure=False, audit=False),
])
def test_non_model_and_status_branches_skip_residency_admission(monkeypatch, args):
    monkeypatch.setattr(
        cli.Config, "load",
        classmethod(lambda cls: (_ for _ in ()).throw(
            AssertionError("non-model branch loaded admission config")
        )),
    )
    monkeypatch.setattr(
        cli, "_free_foreign_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-model branch probed residency")
        ),
    )
    assert cli._admit_model_command(args, None) is None


def test_research_role_switch_uses_exact_owned_seam():
    from spiral.research_loop import ResearchLoop

    events = []

    class FakeOllama:
        providers = {}

        @staticmethod
        def evict_owned_local_models_except(keep, log=None):
            events.append(("switch", keep))
            return []

        @staticmethod
        def chat(model, _messages, **_kwargs):
            events.append(("chat", model))
            return ChatResult("answer", 1, 1)

    loop = ResearchLoop.__new__(ResearchLoop)
    loop.ol = FakeOllama()
    loop.cfg = types.SimpleNamespace(
        planner=types.SimpleNamespace(name="next:12b", think=False, num_ctx=4096),
        planner_max_tokens=64,
        keep_alive="45m",
    )
    loop.state = types.SimpleNamespace(tokens=0, api_tokens=0, local_tokens=0)
    loop._compact_user_prompt = lambda user, max_chars: user
    loop._audit_model_call = lambda **_kwargs: None
    loop._say = lambda _message: None

    assert loop._think("system", "user") == ("answer", 1)
    assert events == [("switch", {"next:12b"}), ("chat", "next:12b")]


def test_conductor_cached_path_never_evicts_unreceipted_configured_name(monkeypatch):
    from spiral.conductor import Conductor

    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(post=lambda *_args, **_kwargs: _Response())
    evicted = []

    def evict(model):
        evicted.append(model)
        return True

    monkeypatch.setattr(client, "evict", evict)
    conductor = Conductor.__new__(Conductor)
    conductor.ol = client

    # A cached validate/design path may know this configured name without this
    # process ever dispatching it. The receipt seam must be a no-op.
    assert conductor._prepare_owned_local_model("critic:b") == []
    assert evicted == []

    assert client.chat(
        "planner:a", [{"role": "user", "content": "plan"}], num_predict=4,
    ).text == "done"
    assert conductor._prepare_owned_local_model("critic:b") == ["planner:a"]
    assert conductor._prepare_owned_local_model("critic:b") == []
    assert evicted == ["planner:a"]

    assert client.chat(
        "critic:b", [{"role": "user", "content": "review"}], num_predict=4,
    ).text == "done"
    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "critic:b")]
    assert [event[:3] for event in events if event[0] == "evict"] == [
        ("evict", "http://ollama.test", "critic:b"),
    ]


def test_planner_fallback_consumes_prior_receipt_exactly_once(monkeypatch):
    from spiral.config import Config
    from spiral.planner import design_brief

    class ModelResponse:
        status_code = 200

        def __init__(self, model):
            self.model = model

        @staticmethod
        def raise_for_status():
            return None

        def json(self):
            return {
                "message": {
                    "content": "x" * 500 if self.model == "critic:b" else "short",
                },
                "prompt_eval_count": 1,
                "eval_count": 1,
                "done_reason": "stop",
            }

    cfg = Config()
    cfg.planner.name = "planner:a"
    cfg.critic.name = "critic:b"
    cfg.prefer_single_resident_model = True
    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})
    client._client = types.SimpleNamespace(
        post=lambda _url, json, **_kwargs: ModelResponse(json["model"]),
    )
    evicted = []

    def evict(model):
        evicted.append(model)
        return True

    monkeypatch.setattr(client, "evict", evict)
    text, result = design_brief("make a UI", [], cfg, client)

    assert len(text) == 500
    assert result.raw["spiral_role_model"] == "critic:b"
    assert evicted == ["planner:a"]
    assert client.evict_owned_local_models_except({"critic:b"}) == []
    assert evicted == ["planner:a"]

    released, events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "critic:b")]
    assert [event[:3] for event in events if event[0] == "evict"] == [
        ("evict", "http://ollama.test", "critic:b"),
    ]


def test_research_notes_and_vision_prepare_each_distinct_local_role(
        tmp_path, monkeypatch):
    from spiral.research_loop import ResearchLoop
    from spiral import visual_review

    events = []

    class FakeOllama:
        providers = {}

        @staticmethod
        def evict_owned_local_models_except(keep, log=None):
            events.append(("switch", keep))
            return []

        @staticmethod
        def chat(model, _messages, **_kwargs):
            events.append(("chat", model))
            return ChatResult('{"ok":true}', 1, 1)

    loop = ResearchLoop.__new__(ResearchLoop)
    loop.ol = FakeOllama()
    loop.cfg = types.SimpleNamespace(
        keep_alive="45m",
        spec_for=lambda _model: types.SimpleNamespace(num_ctx=4096),
    )
    loop.state = types.SimpleNamespace(tokens=0, api_tokens=0, local_tokens=0)
    loop._audit_model_call = lambda **_kwargs: None
    loop._say = lambda _message: None

    assert loop._think_json_model(
        "notes:small", 4096, "system", "paper",
    ) == {"ok": True}
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(
        visual_review, "choose_vision_model", lambda _cfg, _ol: "vision:local",
    )
    vision = loop._local_vision_json("system", "inspect", [image])

    assert vision["ok"] is True
    assert events == [
        ("switch", {"notes:small"}), ("chat", "notes:small"),
        ("switch", {"vision:local"}), ("chat", "vision:local"),
    ]


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_local_request_keeps_exact_cleanup_receipt(
        monkeypatch, failure):
    llm.begin_owned_local_model_run()
    client = Ollama("http://ollama.test", providers={})

    def fail(*_args, **_kwargs):
        raise failure

    client._client = types.SimpleNamespace(post=fail)
    with pytest.raises(type(failure)):
        client.chat(
            "attempted:27b", [{"role": "user", "content": "work"}], num_predict=4,
        )

    released, _events = _release_with_fake(monkeypatch)
    assert released == [("http://ollama.test", "attempted:27b")]


def test_hard_offline_cleanup_opens_no_client_and_clears_receipt(monkeypatch):
    llm.begin_owned_local_model_run()
    llm._remember_owned_local_model("http://ollama.test", "never-touch")
    monkeypatch.setenv("SPIRAL_OFFLINE_TESTS", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline cleanup constructed a transport")

    assert llm.release_owned_local_models(client_factory=forbidden) == []
    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS")
    assert llm.release_owned_local_models(client_factory=forbidden) == []


def test_sigterm_cleanup_is_bounded_behind_interactive_model_lane(
        tmp_path, monkeypatch):
    lease_path = tmp_path / "spiral-compute.lease"
    priority = tmp_path / "spiral-compute.priority"
    priority.mkdir()
    (priority / f"interactive-{os.getpid()}-phone.json").write_text(json.dumps({
        "pid": os.getpid(), "created_at": time.time(),
    }))
    monkeypatch.setenv("SPIRAL_MODEL_LEASE_PATH", str(lease_path))
    monkeypatch.delenv("SPIRAL_OFFLINE_TESTS", raising=False)
    posts = []

    class ForbiddenHTTP:
        def post(self, *_args, **_kwargs):
            posts.append("post")
            raise AssertionError("contended terminal cleanup reached transport")

        @staticmethod
        def close():
            return None

    def factory(base_url, timeout, providers):
        client = Ollama(base_url, timeout=timeout, providers=providers)
        client._client = ForbiddenHTTP()
        return client

    def begin():
        llm.begin_owned_local_model_run()
        llm._remember_owned_local_model("http://ollama.test", "owned:27b")

    monkeypatch.setattr(cli, "begin_owned_local_model_run", begin)
    monkeypatch.setattr(
        cli, "release_owned_local_models",
        lambda: llm.release_owned_local_models(
            client_factory=factory, timeout_seconds=0.05,
        ),
    )
    monkeypatch.setattr(
        cli, "main",
        lambda: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None),
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    started = time.monotonic()
    with pytest.raises(SystemExit) as stopped:
        cli.entry()

    assert stopped.value.code == 128 + signal.SIGTERM
    assert time.monotonic() - started < 0.5
    assert posts == []
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm


@pytest.mark.parametrize("outcome", ["success", "error", "interrupt", "sigterm"])
def test_cli_terminal_boundary_always_runs_cleanup(monkeypatch, outcome):
    events = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(cli, "begin_owned_local_model_run", lambda: events.append("begin"))
    monkeypatch.setattr(cli, "release_owned_local_models", lambda: events.append("release"))
    monkeypatch.setattr(
        cli, "make_console", lambda: types.SimpleNamespace(print=lambda *_a, **_k: None),
    )
    installed = {}
    real_signal = signal.signal

    def record_signal(kind, handler):
        if kind == signal.SIGTERM:
            installed["sigterm"] = handler
        return real_signal(kind, handler)

    monkeypatch.setattr(cli.signal, "signal", record_signal)

    def main():
        events.append("main")
        if outcome == "error":
            raise RuntimeError("failed")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        if outcome == "sigterm":
            installed["sigterm"](signal.SIGTERM, None)

    monkeypatch.setattr(cli, "main", main)
    if outcome == "error":
        with pytest.raises(RuntimeError, match="failed"):
            cli.entry()
    elif outcome == "interrupt":
        with pytest.raises(SystemExit) as stopped:
            cli.entry()
        assert stopped.value.code == 130
    elif outcome == "sigterm":
        with pytest.raises(SystemExit) as stopped:
            cli.entry()
        assert stopped.value.code == 128 + signal.SIGTERM
    else:
        cli.entry()

    assert events == ["begin", "main", "release"]
    assert signal.getsignal(signal.SIGTERM) == previous_sigterm
