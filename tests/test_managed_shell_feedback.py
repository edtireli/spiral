"""Capability feedback must reflect the broker, not a caller's access hint."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spiral.agent import Atom, SYSTEM, TaskSpec, _shell_feedback
from spiral.command_broker import BrokerResult, CommandBroker, shell_network_policy
from spiral.config import Config
from spiral.tools import RunResult


@pytest.mark.parametrize("command,expected", [
    ("(printf 'producer failed\\n'; exit 7) | cat", 7),
    ("(printf 'producer passed\\n'; exit 0) | cat", 0),
    ("printf 'input\\n' | (cat; exit 9)", 9),
])
def test_filtered_output_preserves_real_pipeline_failure(tmp_path, monkeypatch, command, expected):
    import spiral.safety_kernel as kernel

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "0")
    monkeypatch.setattr(kernel, "protected_boundaries", lambda _: [])
    action = CommandBroker(tmp_path).run(command, full_access=True)
    assert action.result.code == expected and not action.result.blocked
    row = json.loads((tmp_path / ".spiral/actions.jsonl").read_text().splitlines()[-1])
    assert row["command"] == command and row["exit"] == expected
    assert row["pipeline_status"] == "pipefail"
    assert row["ok"] == (expected == 0)


@pytest.mark.parametrize("managed,full,expected", [
    (True, True, "denied"), (True, False, "denied"),
    (False, True, "allowed"), (False, False, "denied"),
])
def test_executed_network_metadata_matches_final_policy(
    tmp_path, monkeypatch, managed, full, expected,
):
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1" if managed else "0")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(tmp_path)
    monkeypatch.setattr(
        broker, "_argv", lambda *args, **kwargs: (["/bin/sh", "-c", "true"], True),
    )
    action = broker.run("true", full_access=full, allow_network=True)
    assert action.result.ok
    assert action.network == expected
    assert shell_network_policy(full_access=full) == expected
    rows = [json.loads(line) for line in (tmp_path / ".spiral/actions.jsonl").read_text().splitlines()]
    assert rows[-1]["network"] == action.network


@pytest.mark.parametrize("reason", ["policy", "boundary", "protected", "managed", "workspace"])
def test_every_prelaunch_refusal_reports_not_executed(tmp_path, monkeypatch, reason):
    import spiral.safety_kernel as kernel

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1" if reason == "managed" else "0")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(tmp_path)
    monkeypatch.setattr(kernel, "protection_active", lambda _: reason == "protected")

    def argv(*args, **kwargs):
        if reason == "boundary":
            raise kernel.SafetyBoundaryError("changed inode")
        return ["must-not-execute"], False

    monkeypatch.setattr(broker, "_argv", argv)
    action = broker.run(
        "git push" if reason == "policy" else "true",
        full_access=reason != "workspace", require_sandbox=True,
    )
    assert action.result.blocked and action.result.code == 126
    assert action.network == "not-executed"


def test_spawn_failure_does_not_claim_executed_network_authority(tmp_path, monkeypatch):
    import spiral.command_broker as module

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "0")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(tmp_path)
    monkeypatch.setattr(broker, "_argv", lambda *a, **k: (["missing"], True))

    def fail(*args, **kwargs):
        raise OSError("spawn refused")

    monkeypatch.setattr(module, "_popen_with_headroom_retry", fail)
    assert broker.run("true", full_access=True).network == "not-executed"


def test_optional_standalone_unsandboxed_fallback_does_not_claim_network_isolation(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "0")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(tmp_path)
    monkeypatch.setattr(
        broker, "_argv", lambda *a, **k: (["/bin/sh", "-c", "true"], False),
    )
    action = broker.run("true", full_access=False, require_sandbox=False)
    assert action.result.ok and not action.sandboxed
    assert action.network == "not-isolated"
    assert "does not grant permission for network access" in _shell_feedback(action)
    assert "Shell network is disabled" not in _shell_feedback(action)


def test_standalone_linux_full_access_fallback_reports_generated_network_namespace(
    tmp_path, monkeypatch,
):
    import subprocess
    import spiral.command_broker as module
    import spiral.safety_kernel as kernel

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "0")
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    monkeypatch.setattr(kernel, "protected_boundaries", lambda _: [
        kernel.ProtectedBoundary(tmp_path / "not-created", "file"),
    ])
    monkeypatch.setattr(kernel, "protection_active", lambda _: True)
    launches = []

    def harmless_launch(argv, *, runtime_control, on_wait, **kwargs):
        launches.append(argv)
        return subprocess.Popen(["/bin/sh", "-c", "true"], **kwargs)

    monkeypatch.setattr(module, "_popen_with_headroom_retry", harmless_launch)
    action = CommandBroker(tmp_path).run("true", full_access=True, allow_network=False)
    assert "--unshare-net" in launches[0][1:-3]
    assert action.result.ok and action.sandboxed
    assert action.network == "denied"


def test_untrusted_shell_body_cannot_forge_bwrap_network_metadata(tmp_path, monkeypatch):
    import subprocess
    import spiral.command_broker as module
    import spiral.safety_kernel as kernel

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "0")
    monkeypatch.setattr(kernel, "protection_active", lambda _: False)
    broker = CommandBroker(tmp_path)
    monkeypatch.setattr(broker, "_argv", lambda *a, **k: (
        ["/usr/bin/bwrap", "--bind", "/", "/", "/bin/sh", "-lc", "--unshare-net"], True,
    ))
    monkeypatch.setattr(module, "_popen_with_headroom_retry", lambda argv, runtime_control, on_wait, **kwargs:
                        subprocess.Popen(["/bin/sh", "-c", "true"], **kwargs))
    assert broker.run("--unshare-net", full_access=True).network == "allowed"


def test_legacy_result_does_not_invent_network_permission():
    action = BrokerResult(RunResult("true", 0, ""), False, "unused")
    assert action.network == "unknown"
    assert "network=unknown" in _shell_feedback(action)


def test_policy_metadata_precedes_truncated_untrusted_output():
    action = BrokerResult(RunResult("pip install package", 1, "DNS error\n" * 1000), True, "unused", "denied")
    bounded = _shell_feedback(action)[:3000]
    assert bounded.startswith("exit=1; sandboxed=True; network=denied\n")
    assert "not evidence that a remote service is down" in bounded
    assert "approved typed web, browser, download, or install" in bounded
    assert "DNS error" in bounded


@pytest.mark.parametrize("managed", [True, False])
def test_full_access_prompt_and_banner_match_managed_authority(tmp_path, monkeypatch, managed):
    from spiral.cli import _full_access_notice

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1" if managed else "0")
    cfg = Config()
    cfg.builder_full_access = True
    prompt = Atom(tmp_path, cfg)._worker_system()
    notice = _full_access_notice()
    assert "FULL ACCESS" in prompt
    if managed:
        assert "mandatory OS isolation and network disabled" in prompt
        assert "has network access" not in prompt
        assert "unsandboxed" not in prompt
        assert "shell network is off" in notice
        assert "network on" not in notice
    else:
        assert "has network access" in prompt
        assert "normally unsandboxed" in prompt
        assert "network on" in notice


def test_python_installer_reports_its_exact_existing_target(tmp_path, monkeypatch):
    import spiral.command_broker as module

    broker = CommandBroker(tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    outcome = broker.provision_typed("python example-package==1.2.3")
    target = tmp_path.resolve() / ".spiral/tooling/python"
    assert outcome.ok
    assert f"installation target={target}" in outcome.message
    assert f"interpreter={target / 'bin/python'}" in outcome.message
    assert "does not install into any other requested environment" in outcome.message
    assert calls[-1][0] == str(target / "bin/python")
    assert "--only-binary=:all:" in calls[-1]
    assert ".spiral/tooling/python" in SYSTEM
    assert ".spiral/dependency-cache/python/venv" in SYSTEM
    assert "Neither location satisfies" in SYSTEM
    denied = broker.provision_typed("python example-package==1.2.3 custom-env")
    assert not denied.ok and denied.failure_kind == "policy"
    assert len(calls) == 2, "an extra target must not become a new execution capability"


def test_worker_next_model_request_contains_actual_broker_network_metadata(tmp_path, monkeypatch):
    import spiral.agent as module

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    cfg = Config()
    cfg.builder_full_access = True
    atom = Atom(tmp_path, cfg)
    monkeypatch.setattr(atom, "_ensure_git", lambda: None)
    monkeypatch.setattr(atom, "_run_gate", lambda *a: RunResult("true", 0, ""))
    monkeypatch.setattr(module.tools, "run", lambda *a, **k: RunResult("git status", 0, ""))
    monkeypatch.setattr(atom.command_broker, "run", lambda *a, **k: BrokerResult(
        RunResult("python install.py", 1, "DNS unavailable\n" * 1000), True, "unused", "denied",
    ))
    messages = []

    class Done(Exception):
        pass

    def chat(model, msgs, **kwargs):
        messages.append(msgs)
        if len(messages) == 2:
            raise Done()
        return SimpleNamespace(
            text="ASK: shell python install.py", completion_tokens=10, prompt_tokens=20,
            total_tokens=30, total_duration=0, eval_count=10, load_duration=0, eval_duration=0,
        )

    atom.ol = SimpleNamespace(chat=chat)

    class Quiet:
        def __getattr__(self, _):
            return lambda *a, **k: None

    with pytest.raises(Done):
        atom._run(TaskSpec("prepare a local environment", "true", ["result.py"]),
                  None, 2, False, False, True, False, None, Quiet())
    assert "network=denied" in messages[1][1]["content"]
    assert "not evidence that a remote service is down" in messages[1][1]["content"]
    assert "DNS unavailable" in messages[1][1]["content"]
