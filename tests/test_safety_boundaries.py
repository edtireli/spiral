"""Security boundaries shared by edits, context tools, and broker processes."""
from __future__ import annotations

import os
import json
import shlex
import signal
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.command_broker import CommandBroker, scrubbed_environment
from spiral.config import Config
from spiral.edits import EditBlock, apply_edits
from spiral.tools import list_dir, read_file


def test_full_access_uses_an_available_shell_on_linux(tmp_path, monkeypatch):
    import spiral.command_broker as broker_module

    monkeypatch.setattr(broker_module.sys, "platform", "linux")
    monkeypatch.setattr(
        broker_module.shutil, "which",
        lambda name: "/usr/bin/bash" if name == "bash" else None,
    )

    argv, sandboxed = CommandBroker(tmp_path)._argv(
        "true", tmp_path, allow_network=True, allow_host_read=True,
        full_access=True,
    )

    assert argv == ["/usr/bin/bash", "-lc", "true"]
    assert sandboxed is False


def test_linux_workspace_profile_uses_bwrap_with_network_and_reference_isolation(
    tmp_path, monkeypatch,
):
    import spiral.command_broker as broker_module

    target = tmp_path / "target"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    cfg = Config()
    cfg.builder_reference_roots = [str(reference)]
    broker = CommandBroker(target, cfg)
    monkeypatch.setattr(broker_module.sys, "platform", "linux")
    monkeypatch.setattr(
        broker_module.shutil, "which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
    )

    argv, sandboxed = broker._argv(
        "true", target, allow_network=False, allow_host_read=False,
    )

    assert sandboxed is True
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert ["--bind", str(target), str(target)] == argv[
        argv.index("--bind"):argv.index("--bind") + 3
    ]
    ref_at = next(
        index for index in range(len(argv) - 2)
        if argv[index:index + 3] == ["--ro-bind", str(reference), str(reference)]
    )
    assert ref_at > 0


def test_managed_linux_full_access_unshares_network_and_masks_git_metadata(
    tmp_path, monkeypatch,
):
    import spiral.command_broker as broker_module

    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    monkeypatch.setattr(broker_module.sys, "platform", "linux")
    monkeypatch.setattr(
        broker_module.shutil, "which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
    )

    argv, sandboxed = CommandBroker(tmp_path)._argv(
        "python build.py", tmp_path, allow_network=True,
        allow_host_read=True, full_access=True,
    )

    assert sandboxed is True
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv
    assert ["--ro-bind", str(gitdir), str(gitdir)] == argv[
        argv.index("--ro-bind"):argv.index("--ro-bind") + 3
    ]


def test_workspace_mode_fails_closed_when_linux_sandbox_is_unavailable(
    tmp_path, monkeypatch,
):
    import spiral.command_broker as broker_module

    monkeypatch.setattr(broker_module.sys, "platform", "linux")
    monkeypatch.setattr(broker_module.shutil, "which", lambda _name: None)

    result = CommandBroker(tmp_path).run("true", require_sandbox=True).result

    assert result.blocked and result.code == 126
    assert "no network/filesystem sandbox" in result.out


def test_workspace_broker_cannot_be_tricked_into_host_reads(tmp_path, monkeypatch):
    """Even a stale/hostile local-model caller cannot widen project-only access."""
    import spiral.command_broker as broker_module

    monkeypatch.setattr(broker_module.sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sandbox-exec")

    argv, sandboxed = CommandBroker(tmp_path)._argv(
        "cat ~/.ssh/id_ed25519", tmp_path,
        allow_network=False,
        allow_host_read=True,  # the historical unsafe local-model value
        full_access=False,
    )

    assert sandboxed is True
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"]
    assert f'(deny file-read* (subpath "{Path.home()}"))' in argv[2]
    assert f'(allow file-read* (subpath "{tmp_path}"))' in argv[2]


def test_explicit_references_are_canonical_read_only_sandbox_grants(
    tmp_path, monkeypatch,
):
    import spiral.command_broker as broker_module
    from spiral.command_broker import canonical_reference_roots

    target = tmp_path / "target"
    target.mkdir()
    source = tmp_path / "source-project"
    source.mkdir()
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-reference")
    cfg = Config()
    cfg.builder_reference_roots = canonical_reference_roots(
        [str(source), str(paper)], workspace=target,
    )
    monkeypatch.setattr(broker_module.sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/sandbox-exec")

    broker = CommandBroker(target, cfg)
    argv, sandboxed = broker._argv(
        "true", target, allow_network=False, allow_host_read=False,
    )

    assert sandboxed
    profile = argv[2]
    assert f'(allow file-read* (subpath "{source}"))' in profile
    assert f'(deny file-write* (subpath "{source}"))' in profile
    assert f'(allow file-read* (literal "{paper}"))' in profile
    assert f'(deny file-write* (literal "{paper}"))' in profile


def test_reference_validation_rejects_target_overlap_and_identity_replacement(tmp_path):
    from spiral.command_broker import canonical_reference_roots

    target = tmp_path / "target"
    target.mkdir()
    nested = target / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="writable target"):
        canonical_reference_roots([nested], workspace=target)
    with pytest.raises(ValueError, match="contains the writable target"):
        canonical_reference_roots([tmp_path], workspace=target)

    reference = tmp_path / "reference.txt"
    reference.write_text("first")
    cfg = Config()
    cfg.builder_reference_roots = canonical_reference_roots(
        [reference], workspace=target,
    )
    broker = CommandBroker(target, cfg)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("replacement")
    os.replace(replacement, reference)

    result = broker.run("true", require_sandbox=False).result
    assert result.blocked
    assert "reference identity changed" in result.out


def test_controller_reference_identity_closes_prelaunch_path_swap(tmp_path):
    import json
    from spiral.command_broker import (
        canonical_reference_roots, require_reference_identities,
    )

    target = tmp_path / "target"
    target.mkdir()
    reference = tmp_path / "paper.pdf"
    reference.write_bytes(b"first")
    paths = canonical_reference_roots([reference], workspace=target)
    info = reference.stat()
    encoded = json.dumps([{
        "path": paths[0], "dev": info.st_dev, "ino": info.st_ino,
        "size": info.st_size, "mtime_ns": info.st_mtime_ns,
    }])
    require_reference_identities(paths, encoded)

    replacement = tmp_path / "replacement.pdf"
    replacement.write_bytes(b"other")
    os.replace(replacement, reference)
    with pytest.raises(ValueError, match="identity changed"):
        require_reference_identities(paths, encoded)


def test_broker_detects_in_place_reference_file_change(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    reference = tmp_path / "paper.pdf"
    reference.write_bytes(b"original")
    cfg = Config()
    cfg.builder_reference_roots = [str(reference)]
    broker = CommandBroker(target, cfg)

    reference.write_bytes(b"changed-content")
    result = broker.run("true", require_sandbox=False).result

    assert result.blocked
    assert "reference file content changed" in result.out


@pytest.mark.parametrize("command", [
    "git init", "git clone https://example.test/x.git", "git add -A",
    "git commit -m x", "git pull", "git fetch origin", "git push origin main",
    "/usr/bin/git -C /tmp reset --hard HEAD", "git remote add origin x",
])
def test_model_git_mutations_always_require_separate_approval(tmp_path, command):
    for full_access in (False, True):
        result = CommandBroker(tmp_path).run(
            command, full_access=full_access, require_sandbox=False,
        ).result
        assert result.blocked
        assert "separate explicit user approval" in result.out


def test_model_can_still_inspect_git(tmp_path):
    for command in (
        "git status", "git diff", "git log -1", "git diff -- config",
        "git grep commit", "git -C /tmp status",
    ):
        assert CommandBroker.policy_error(command, full_access=True) == ""


def test_managed_task_transactions_checkpoint_without_mutating_git(
    tmp_path, monkeypatch,
):
    import subprocess
    from spiral.transactions import TaskTransaction

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")

    tx = TaskTransaction.begin(tmp_path, "managed task")
    tracked.write_text("green\n")
    (tmp_path / "new.txt").write_text("new\n")
    revision, moved = tx.commit("must not become a Git commit")
    assert moved and revision.startswith("worktree-")
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True,
    ).strip() == head

    tracked.write_text("failed\n")
    (tmp_path / "failed.txt").write_text("failed\n")
    tx.rollback(reason="test")
    assert tracked.read_text() == "green\n"
    assert (tmp_path / "new.txt").read_text() == "new\n"
    assert not (tmp_path / "failed.txt").exists()
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True,
    ).strip() == head
    tx.close()


def test_workspace_run_normalizes_network_and_host_read_before_audit(
    tmp_path, monkeypatch,
):
    broker = CommandBroker(tmp_path)
    captured = {}

    def fake_argv(command, cwd, allow_network, allow_host_read, full_access=False):
        captured.update(network=allow_network, host_read=allow_host_read)
        return ["/bin/sh", "-lc", "true"], True

    monkeypatch.setattr(broker, "_argv", fake_argv)
    result = broker.run(
        "true", allow_network=True, allow_host_read=True,
        full_access=False, require_sandbox=True,
    )

    assert result.result.code == 0
    assert captured == {"network": False, "host_read": False}
    manifest = Path(result.manifest).read_text()
    assert '"network": "denied"' in manifest
    assert '"host_read": "workspace-only"' in manifest


def test_homebrew_provisioning_requires_explicit_full_access(tmp_path):
    broker = CommandBroker(tmp_path)

    denied = broker.provision("brew tectonic", full_access=False)

    assert denied == "tool request rejected: Homebrew provisioning requires full access"


def test_ollama_model_pull_requires_full_access_and_records_certificate(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    from spiral import command_broker as broker_module

    broker = CommandBroker(tmp_path)
    calls = []
    show_calls = 0

    def fake_run(argv, **_kwargs):
        nonlocal show_calls
        calls.append(list(argv))
        if argv[1] == "show":
            show_calls += 1
            if show_calls == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="missing")
            return SimpleNamespace(
                returncode=0, stdout="architecture qwen\nparameters 8B\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="pulled", stderr="")

    monkeypatch.setattr(
        broker_module.shutil, "which",
        lambda name: "/usr/local/bin/ollama" if name == "ollama" else None,
    )
    monkeypatch.setattr(broker_module.subprocess, "run", fake_run)

    denied = broker.provision_typed("ollama qwen3:8b", full_access=False)
    allowed = broker.provision_typed("ollama qwen3:8b", full_access=True)

    assert denied.failure_kind == "policy"
    assert allowed.ok and not allowed.failure_kind
    assert ["/usr/local/bin/ollama", "pull", "qwen3:8b"] in calls
    actions = (tmp_path / ".spiral/actions.jsonl").read_text()
    assert '"certificate"' in actions and '"ollama", "show", "qwen3:8b"' in actions


def test_workspace_download_is_https_bounded_hashed_and_receipted(
        tmp_path, monkeypatch):
    import httpx
    from spiral import command_broker as broker_module

    real_client = httpx.Client
    body = b"audited artifact\n"

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, content=body, request=request)

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(
        broker_module.socket, "getaddrinfo",
        lambda *_args, **_kwargs: [
            (broker_module.socket.AF_INET, broker_module.socket.SOCK_STREAM,
             6, "", ("93.184.216.34", 443)),
        ],
    )
    monkeypatch.setattr(broker_module.httpx, "Client", client_factory)
    broker = CommandBroker(tmp_path)
    expected = __import__("hashlib").sha256(body).hexdigest()

    outcome = broker.download_workspace(
        f"https://example.com/tool.bin -> assets/tool.bin sha256={expected}",
        network_allowed=True, max_bytes=1024,
    )

    assert outcome.ok
    assert (tmp_path / "assets/tool.bin").read_bytes() == body
    receipt = json.loads(Path(outcome.receipt).read_text())
    assert receipt["sha256"] == expected and receipt["integrity"] == "pinned"


def test_workspace_download_rejects_network_denial_credentials_and_overwrite(
        tmp_path):
    broker = CommandBroker(tmp_path)
    (tmp_path / "keep.bin").write_bytes(b"keep")

    denied = broker.download_workspace(
        "https://example.com/a -> new.bin", network_allowed=False)
    credential = broker.download_workspace(
        "https://user:pass@example.com/a -> new.bin", network_allowed=True)
    overwrite = broker.download_workspace(
        "https://example.com/a -> keep.bin", network_allowed=True)

    assert {denied.failure_kind, credential.failure_kind, overwrite.failure_kind} == {"policy"}
    assert (tmp_path / "keep.bin").read_bytes() == b"keep"


def test_transient_workspace_download_retries_inside_setup_lane(
        tmp_path, monkeypatch):
    import httpx
    from spiral import command_broker as broker_module

    real_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadTimeout("temporary CDN stall", request=request)
        return httpx.Response(200, content=b"ready", request=request)

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(
        broker_module.socket, "getaddrinfo",
        lambda *_args, **_kwargs: [
            (broker_module.socket.AF_INET, broker_module.socket.SOCK_STREAM,
             6, "", ("93.184.216.34", 443)),
        ],
    )
    monkeypatch.setattr(broker_module.httpx, "Client", client_factory)
    monkeypatch.setattr(broker_module.time, "sleep", lambda _seconds: None)

    outcome = CommandBroker(tmp_path).download_workspace(
        "https://example.com/tool -> tool.bin", network_allowed=True)

    assert outcome.ok and len(calls) == 3
    assert (tmp_path / "tool.bin").read_bytes() == b"ready"


def test_transient_workspace_download_retries_http_statuses(
        tmp_path, monkeypatch):
    import httpx
    from spiral import command_broker as broker_module

    real_client = httpx.Client
    statuses = [429, 503, 200]

    def handler(request):
        status = statuses.pop(0)
        return httpx.Response(status, content=b"ready", request=request)

    def client_factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(
        broker_module.socket, "getaddrinfo",
        lambda *_args, **_kwargs: [
            (broker_module.socket.AF_INET, broker_module.socket.SOCK_STREAM,
             6, "", ("93.184.216.34", 443)),
        ],
    )
    monkeypatch.setattr(broker_module.httpx, "Client", client_factory)
    monkeypatch.setattr(broker_module.time, "sleep", lambda _seconds: None)

    outcome = CommandBroker(tmp_path).download_workspace(
        "https://example.com/tool -> tool.bin", network_allowed=True)

    assert outcome.ok and statuses == []
    assert (tmp_path / "tool.bin").read_bytes() == b"ready"


def test_edit_paths_reject_absolute_and_parent_traversal(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("keep\n")

    absolute = apply_edits(
        tmp_path, [EditBlock(str(outside), "keep", "changed")])[0]
    traversal = apply_edits(
        tmp_path, [EditBlock(f"../{outside.name}", "keep", "changed")])[0]

    assert not absolute.ok and "absolute paths" in absolute.reason
    assert not traversal.ok and "parent traversal" in traversal.reason
    assert outside.read_text() == "keep\n"


def test_edit_paths_reject_symlink_escapes_and_non_files(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("keep\n")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    (tmp_path / "directory.txt").mkdir()

    existing = apply_edits(
        tmp_path, [EditBlock("escape/secret.txt", "keep", "changed")])[0]
    create = apply_edits(
        tmp_path, [EditBlock("escape/new.txt", "", "created")])[0]
    non_file = apply_edits(
        tmp_path, [EditBlock("directory.txt", "anything", "changed")])[0]

    assert not existing.ok and "outside the workspace" in existing.reason
    assert not create.ok and "outside the workspace" in create.reason
    assert not non_file.ok and "not a regular file" in non_file.reason
    assert secret.read_text() == "keep\n"
    assert not (outside / "new.txt").exists()


def test_context_helpers_share_the_workspace_boundary(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-context"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not disclose\n")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    for result in (
        read_file(tmp_path, str(secret)),
        read_file(tmp_path, f"../{outside.name}/secret.txt"),
        read_file(tmp_path, "escape/secret.txt"),
        list_dir(tmp_path, str(outside)),
        list_dir(tmp_path, f"../{outside.name}"),
        list_dir(tmp_path, "escape"),
    ):
        assert result.startswith("(invalid path:"), result
        assert "do not disclose" not in result


def test_full_access_uses_real_home_but_still_scrubs_secret_environment(
    tmp_path, monkeypatch,
):
    user_home = tmp_path.parent / f"{tmp_path.name}-user-home"
    user_home.mkdir()
    (user_home / "Documents").mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("SPIRAL_TEST_API_KEY", "must-not-leak")
    code = (
        "import os,pathlib;"
        "print(pathlib.Path.home());"
        "print(pathlib.Path('~/Documents').expanduser().is_dir());"
        "print(os.environ.get('SPIRAL_TEST_API_KEY','SCRUBBED'))"
    )

    result = CommandBroker(tmp_path).run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        cwd=tmp_path,
        timeout=5,
        require_sandbox=False,
        full_access=True,
    ).result

    assert result.code == 0, result.out
    assert result.out.splitlines() == [str(user_home), "True", "SCRUBBED"]


def test_managed_full_access_uses_isolated_home_and_fixed_vcs_environment(
    tmp_path, monkeypatch,
):
    host_home = tmp_path / "host-home"
    host_home.mkdir()
    monkeypatch.setenv("HOME", str(host_home))
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("GITHUB_TOKEN", "host-token")

    env = scrubbed_environment(tmp_path, {
        "HOME": str(host_home),
        "XDG_CONFIG_HOME": str(host_home / ".config"),
        "GIT_CONFIG_GLOBAL": str(host_home / ".gitconfig"),
        "GIT_SSH_COMMAND": "ssh -F secret-config",
        "SSH_AUTH_SOCK": "/tmp/injected-agent.sock",
        "GH_TOKEN": "injected-token",
        "PATH": "/safe/tools:/usr/bin",
    }, full_access=True)

    runtime_home = tmp_path / ".spiral" / "runtime-home"
    assert env["HOME"] == str(runtime_home)
    assert env["XDG_CONFIG_HOME"] == str(runtime_home / ".config")
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_SYSTEM"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_ASKPASS"] == "/usr/bin/false"
    assert env["PATH"] == "/safe/tools:/usr/bin"
    assert "GIT_SSH_COMMAND" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert "GH_TOKEN" not in env
    assert "GITHUB_TOKEN" not in env


def test_managed_full_access_profile_denies_network_credentials_and_real_gitdirs(
    tmp_path, monkeypatch,
):
    import spiral.command_broker as broker_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gitdir = tmp_path / "metadata" / "worktrees" / "one"
    common = tmp_path / "metadata"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    (workspace / ".git").write_text(f"gitdir: {gitdir}\n")
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    monkeypatch.setattr(broker_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        broker_module.shutil, "which",
        lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None,
    )

    argv, sandboxed = CommandBroker(workspace)._argv(
        "python build.py", workspace, allow_network=True,
        allow_host_read=True, full_access=True,
    )

    assert sandboxed is True
    profile = argv[2]
    assert "(deny network*)" in profile
    assert "(deny appleevent-send)" in profile
    assert r'(deny file-write* (regex #".*/[.]git(/.*)?$"))' in profile
    assert f'(literal "{workspace / ".git"}")' in profile
    assert f'(subpath "{gitdir}")' in profile
    assert f'(subpath "{common}")' in profile
    assert f'(deny file-read* (subpath "{Path.home() / ".ssh"}"))' in profile
    assert '(deny process-exec (literal "/usr/bin/ssh"))' in profile
    assert '(deny process-exec (literal "/usr/bin/open"))' in profile


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="adversarial enforcement check uses the macOS kernel sandbox",
)
def test_managed_full_access_os_boundary_blocks_direct_and_subprocess_git_writes(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    gitdir = workspace / ".git"
    gitdir.mkdir(parents=True)
    head = gitdir / "HEAD"
    head.write_text("ref: refs/heads/main\n")
    ordinary = workspace / "ordinary.txt"
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(workspace)

    ordinary_code = (
        "from pathlib import Path;"
        f"Path({str(ordinary)!r}).write_text('ordinary write allowed')"
    )
    ordinary_result = broker.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(ordinary_code)}",
        timeout=10, require_sandbox=False, full_access=True,
    ).result
    assert ordinary_result.code == 0, ordinary_result.out
    assert ordinary.read_text() == "ordinary write allowed"

    direct_code = (
        "import os;"
        f"fd=os.open({str(head)!r},os.O_WRONLY|os.O_TRUNC);"
        "os.write(fd,b'compromised');os.close(fd)"
    )
    direct_result = broker.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(direct_code)}",
        timeout=10, require_sandbox=False, full_access=True,
    ).result
    assert direct_result.code != 0
    assert head.read_text() == "ref: refs/heads/main\n"

    child_code = (
        "from pathlib import Path;"
        f"Path({str(head)!r}).write_text('subprocess compromised')"
    )
    parent_code = (
        "import subprocess,sys;"
        f"p=subprocess.run([sys.executable,'-c',{child_code!r}]);"
        "print(p.returncode);"
        "raise SystemExit(0 if p.returncode != 0 else 9)"
    )
    subprocess_result = broker.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}",
        timeout=10, require_sandbox=False, full_access=True,
    ).result
    assert subprocess_result.code == 0, subprocess_result.out
    assert head.read_text() == "ref: refs/heads/main\n"

    network_code = (
        "import errno,socket,sys;"
        "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);"
        "code=s.connect_ex(('127.0.0.1',9));"
        "print(code);"
        "raise SystemExit(0 if code in (errno.EACCES,errno.EPERM) else 7)"
    )
    network_result = broker.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(network_code)}",
        timeout=10, require_sandbox=False, full_access=True,
    ).result
    assert network_result.code == 0, network_result.out


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file(),
    reason="Git mutation enforcement check uses the macOS kernel sandbox",
)
def test_managed_shell_keeps_git_inspection_but_os_denies_hidden_git_add(
    tmp_path, monkeypatch,
):
    import subprocess

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=workspace, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=workspace, check=True,
    )
    tracked = workspace / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
    index = workspace / ".git" / "index"
    index_before = index.read_bytes()
    tracked.write_text("changed\n")
    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    broker = CommandBroker(workspace)

    inspection = broker.run(
        "git status --short", timeout=10,
        require_sandbox=False, full_access=True,
    ).result
    assert inspection.code == 0, inspection.out
    assert "tracked.txt" in inspection.out

    child_code = (
        "import subprocess,sys;"
        f"p=subprocess.run(['/usr/bin/'+'g'+'it','-C',{str(workspace)!r},"
        "'a'+'dd','tracked.txt']);"
        "print(p.returncode);"
        "raise SystemExit(0 if p.returncode != 0 else 8)"
    )
    mutation = broker.run(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(child_code)}",
        timeout=10, require_sandbox=False, full_access=True,
    ).result

    assert mutation.code == 0, mutation.out
    assert index.read_bytes() == index_before


def test_managed_full_access_fails_closed_without_os_sandbox(tmp_path, monkeypatch):
    import spiral.command_broker as broker_module

    monkeypatch.setenv("SPIRALCHAT_EXTERNAL_GIT_APPROVAL", "1")
    monkeypatch.delenv("SPIRALCHAT_PROTECTED_PATHS", raising=False)
    monkeypatch.setattr(broker_module.sys, "platform", "linux")
    monkeypatch.setattr(broker_module.shutil, "which", lambda _name: None)

    result = CommandBroker(tmp_path).run(
        "true", full_access=True, require_sandbox=False,
    ).result

    assert result.blocked and result.code == 126
    assert "mandatory Git/network isolation" in result.out


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Linux containers may leave an orphan as a zombie briefly; a zombie cannot
    # execute and is therefore no longer a leaked broker process.
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().split()[2] != "Z"
        except (OSError, IndexError):
            pass
    return True


def _wait_not_running(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.025)
    return not _pid_is_running(pid)


def _tree_command(pid_file: Path, ignored_signals: tuple[int, ...]) -> str:
    handlers = ";".join(
        f"signal.signal({number}, signal.SIG_IGN)" for number in ignored_signals)
    child = f"import signal,time;{handlers};time.sleep(30)"
    parent = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(30)"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(parent)}"


@pytest.mark.skipif(os.name != "posix", reason="process-group ownership is POSIX-specific")
def test_broker_timeout_terminates_resistant_descendants(tmp_path):
    pid_file = tmp_path / "child.pid"
    result = CommandBroker(tmp_path).run(
        _tree_command(pid_file, (signal.SIGTERM,)),
        cwd=tmp_path,
        timeout=1,
        require_sandbox=False,
        full_access=True,
    ).result

    assert result.code == 124 and "timed out" in result.out
    assert pid_file.is_file(), result.out
    child_pid = int(pid_file.read_text())
    assert _wait_not_running(child_pid), f"descendant {child_pid} survived broker timeout"


@pytest.mark.skipif(os.name != "posix", reason="process-group ownership is POSIX-specific")
def test_broker_keyboard_interrupt_terminates_descendants(tmp_path, monkeypatch):
    import spiral.command_broker as broker_module

    pid_file = tmp_path / "interrupt-child.pid"

    def interrupt_when_started(*_args, **_kwargs):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not pid_file.exists():
            time.sleep(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(broker_module.select, "select", interrupt_when_started)
    with pytest.raises(KeyboardInterrupt):
        CommandBroker(tmp_path).run(
            _tree_command(pid_file, (signal.SIGINT, signal.SIGTERM)),
            cwd=tmp_path,
            timeout=30,
            require_sandbox=False,
            full_access=True,
        )

    assert pid_file.is_file()
    child_pid = int(pid_file.read_text())
    assert _wait_not_running(child_pid), (
        f"descendant {child_pid} survived broker KeyboardInterrupt")
