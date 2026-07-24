"""Audited execution broker for autonomous Builder actions.

Model-requested commands run without credentials, with network egress denied and
writes confined to the workspace and temporary directories when the host provides
an OS sandbox. Package acquisition is a separate, typed operation so ordinary
shell commands cannot quietly become download or messaging channels.
"""
from __future__ import annotations

import json
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from spiral.tools import RunResult


_SECRET = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|COOKIE|CREDENTIAL|AUTH|SESSION|"
    r"API_KEY|PRIVATE_KEY|SSH_|AWS_|AZURE_|GOOGLE_|GITHUB_|GITLAB_|"
    r"OPENAI_|ANTHROPIC_|MOONSHOT_|SLACK_|DISCORD_)",
    re.I,
)
_COMMUNICATION = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:mailx?|sendmail|ssh|scp|sftp|nc|ncat|socat|"
    r"osascript|imessage|telegram|discord|slack)(?:\s|$)",
    re.I,
)
_FORBIDDEN = (
    "rm -rf", "rm -fr", "mkfs", "diskutil erase", "dd if=", "shutdown",
    "reboot", "launchctl", "sudo ", "git push", "gh pr ", "gh issue ",
    "curl ", "wget ", "history -c", "chmod -r", "chown -r",
)
_GIT_MUTATION = re.compile(
    r"(?:^|[;&|]\s*|\s)git\s+(?:add|commit|reset|clean|restore|checkout|switch|"
    r"branch|merge|rebase|cherry-pick|revert|tag|stash|push|fetch|pull|clone|"
    r"remote|config|submodule|worktree)\b",
    re.I,
)
_NODE_PACKAGE = re.compile(
    r"^(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+"
    r"(?:@[A-Za-z0-9*^~<>=_.+-]+)?$"
)
_BREW_FORMULA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,100}$")


def scrubbed_environment(workspace: str | Path, extra: dict | None = None) -> dict[str, str]:
    root = Path(workspace).resolve()
    home = root / ".spiral" / "runtime-home"
    cache = root / ".spiral" / "runtime-cache"
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    keep = {
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "SHELL",
        "CC", "CXX", "JAVA_HOME", "SDKROOT", "DEVELOPER_DIR",
    }
    env = {
        key: value for key, value in os.environ.items()
        if key in keep and not _SECRET.search(key)
    }
    env.update({
        "HOME": str(home),
        "XDG_CACHE_HOME": str(cache),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
        "NO_COLOR": "1",
        "CI": "1",
    })
    for key, value in (extra or {}).items():
        if not _SECRET.search(str(key)):
            env[str(key)] = str(value)
    return env


def _sandbox_string(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class BrokerResult:
    result: RunResult
    sandboxed: bool
    manifest: str


class CommandBroker:
    def __init__(self, workspace: str | Path, cfg=None):
        self.root = Path(workspace).resolve()
        self.cfg = cfg
        self.environment: dict[str, str] = {}
        self.audit = self.root / ".spiral" / "actions.jsonl"
        self.audit.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, payload: dict) -> str:
        payload = {
            "schema_version": 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **payload,
        }
        with self.audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return str(self.audit)

    @staticmethod
    def policy_error(command: str) -> str:
        lowered = f" {command.lower()} "
        if len(command) > 2000 or "\x00" in command:
            return "command is empty, binary, or too long"
        if _COMMUNICATION.search(command):
            return "communication, messaging, remote login, and social actions are forbidden"
        if _GIT_MUTATION.search(command):
            return "model shell may inspect git but transaction/history mutations belong to the harness"
        if any(token in lowered for token in _FORBIDDEN):
            return "command requests a destructive, remote, or acquisition action"
        if re.search(r"(?:^|\s)(?:open|xdg-open)\s+https?://", command, re.I):
            return "opening remote URLs belongs to the research/browser broker"
        if ".spiral/tools/" in command.replace("\\", "/"):
            return "inspection-only repository caches may not be executed"
        return ""

    def _argv(
        self, command: str, cwd: Path, allow_network: bool,
        allow_host_read: bool,
    ) -> tuple[list[str], bool]:
        sandbox = shutil.which("sandbox-exec")
        if sandbox and sys.platform == "darwin":
            temp = os.environ.get("TMPDIR") or "/tmp"
            profile = [
                "(version 1)",
                "(allow default)",
            ]
            if not allow_network:
                profile.append("(deny network*)")
            if not allow_host_read:
                tool_roots = [
                    Path.home() / "Library/Android/sdk",
                    Path.home() / ".cargo",
                    Path.home() / ".rustup",
                    Path.home() / ".elan",
                    Path.home() / ".local/bin",
                    Path.home() / ".local/pipx",
                    Path.home() / ".local/share/uv",
                    Path.home() / ".cache/uv",
                    Path.home() / ".cache/pip",
                ]
                profile += [
                    f'(deny file-read* (subpath "{_sandbox_string(Path.home())}"))',
                    f'(allow file-read* (subpath "{_sandbox_string(self.root)}"))',
                    f'(allow file-read* (subpath "{_sandbox_string(temp)}"))',
                    *[
                        f'(allow file-read* (subpath "{_sandbox_string(path)}"))'
                        for path in tool_roots if path.exists()
                    ],
                ]
            profile += [
                "(deny signal (target others))",
                "(deny file-write*)",
                f'(allow file-write* (subpath "{_sandbox_string(self.root)}"))',
                f'(allow file-write* (subpath "{_sandbox_string(temp)}"))',
                '(allow file-write* (subpath "/tmp"))',
                '(allow file-write* (subpath "/private/tmp"))',
                '(allow file-write* (literal "/dev/null"))',
                f'(deny file-write* (subpath "{_sandbox_string(self.root / ".git")}"))',
                f'(deny file-write* (subpath "{_sandbox_string(self.root / ".spiral" / "tools")}"))',
            ]
            return [
                sandbox, "-p", " ".join(profile), "/bin/zsh", "-lc", command,
            ], True
        bwrap = shutil.which("bwrap")
        if bwrap and not allow_network:
            argv = [
                bwrap, "--unshare-net", "--die-with-parent", "--ro-bind", "/", "/",
                "--bind", str(self.root), str(self.root), "--dev", "/dev",
            ]
            for protected in (
                self.root / ".git", self.root / ".spiral" / "tools",
            ):
                if protected.exists():
                    argv.extend(["--ro-bind", str(protected), str(protected)])
            argv.extend([
                "--proc", "/proc", "--chdir", str(cwd),
                "/bin/sh", "-lc", command,
            ])
            return argv, True
        return ["/bin/sh", "-lc", command], False

    def run(
        self, command: str, *, cwd: str | Path | None = None, timeout: int = 300,
        on_line=None, purpose: str = "model-shell", allow_network: bool = False,
        require_sandbox: bool = True, allow_host_read: bool = True,
    ) -> BrokerResult:
        command = str(command or "").strip()
        error = self.policy_error(command)
        work = Path(cwd).resolve() if cwd else self.root
        try:
            work.relative_to(self.root)
        except ValueError:
            error = "working directory escapes the workspace"
        if error:
            result = RunResult(command, 126, f"broker rejected command: {error}", True)
            manifest = self._record({
                "kind": purpose, "command": command, "cwd": str(work),
                "ok": False, "blocked": True, "reason": error,
            })
            return BrokerResult(result, False, manifest)

        argv, sandboxed = self._argv(
            command, work, allow_network, allow_host_read)
        if require_sandbox and not sandboxed:
            result = RunResult(
                command, 126,
                "broker refused host execution because no network/filesystem sandbox is available",
                True,
            )
            manifest = self._record({
                "kind": purpose, "command": command, "cwd": str(work),
                "ok": False, "blocked": True, "reason": "sandbox unavailable",
            })
            return BrokerResult(result, False, manifest)

        env = scrubbed_environment(self.root, self.environment)
        started = time.monotonic()
        lines: list[str] = []
        try:
            process = subprocess.Popen(
                argv, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, stdin=subprocess.DEVNULL, env=env,
            )
            assert process.stdout is not None
            deadline = time.monotonic() + max(1, timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    lines.append(f"\n(timed out after {timeout}s)")
                    break
                ready, _, _ = select.select(
                    [process.stdout], [], [], min(0.25, remaining))
                if ready:
                    line = process.stdout.readline()
                    if line:
                        lines.append(line)
                        if on_line:
                            try:
                                on_line(line.rstrip())
                            except Exception:
                                pass
                if process.poll() is not None:
                    tail = process.stdout.read()
                    if tail:
                        lines.append(tail)
                    break
            code = process.wait(timeout=15)
        except Exception as exc:
            code = 124
            lines.append(f"\n(command broker error: {type(exc).__name__}: {exc})")
        output = "".join(lines).strip()
        result = RunResult(command, code, output)
        manifest = self._record({
            "kind": purpose,
            "command": command,
            "cwd": str(work.relative_to(self.root) or "."),
            "sandboxed": sandboxed,
            "network": "allowed" if allow_network else "denied",
            "host_read": "allowed" if allow_host_read else "workspace-only",
            "environment_keys": sorted(env),
            "exit": code,
            "seconds": round(time.monotonic() - started, 2),
            "output_tail": output[-2000:],
            "ok": code == 0,
        })
        return BrokerResult(result, sandboxed, manifest)

    def provision(self, request: str, *, timeout: int = 900) -> str:
        """Install one typed tool and expose its local bin directory to later actions."""

        try:
            parts = shlex.split(request)
        except ValueError as exc:
            return f"tool request rejected: {exc}"
        if len(parts) != 2:
            return "tool request rejected: use `python PACKAGE`, `node PACKAGE`, or `brew FORMULA`"
        ecosystem, package = parts[0].lower(), parts[1]
        if package.startswith(("-", ".", "/")) or re.search(
                r"(?:https?://|git(?:\+|hub:|lab:)?|ssh:|file:)", package, re.I):
            return "tool request rejected: package/formula name is not registry-safe"
        if ecosystem == "python":
            try:
                from packaging.requirements import Requirement
            except Exception:
                from pip._vendor.packaging.requirements import Requirement  # type: ignore
            try:
                parsed = Requirement(package)
            except Exception:
                return "tool request rejected: invalid Python registry requirement"
            if parsed.url:
                return "tool request rejected: Python tool must come from the configured registry"
        elif ecosystem == "node":
            if not _NODE_PACKAGE.fullmatch(package):
                return "tool request rejected: invalid Node registry package"
        elif ecosystem == "brew":
            if not _BREW_FORMULA.fullmatch(package):
                return "tool request rejected: invalid Homebrew core formula"
        else:
            return "tool request rejected: unsupported ecosystem"
        tooling = self.root / ".spiral" / "tooling"
        tooling.mkdir(parents=True, exist_ok=True)
        command: list[str]
        bin_dir: Path | None = None
        already = False
        cleanup_command: list[str] | None = None
        remove_on_failure: Path | None = None
        if ecosystem == "python":
            venv = tooling / "python"
            python = venv / "bin" / "python"
            created = not venv.exists()
            if not python.is_file():
                made = subprocess.run(
                    [sys.executable, "-m", "venv", str(venv)],
                    cwd=self.root, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, timeout=180,
                    env=scrubbed_environment(self.root),
                )
                if made.returncode != 0:
                    if created:
                        shutil.rmtree(venv, ignore_errors=True)
                    return f"tool install failed: {made.stderr or made.stdout}"
            command = [
                str(python), "-m", "pip", "install", "--no-input",
                "--disable-pip-version-check", "--only-binary=:all:", package,
            ]
            bin_dir = venv / "bin"
            cleanup_command = [
                str(python), "-m", "pip", "uninstall", "-y",
                re.split(r"[@<>=!~\\[]", package, maxsplit=1)[0],
            ]
            remove_on_failure = venv if created else None
        elif ecosystem == "node":
            npm = shutil.which("npm")
            if not npm:
                return "tool install failed: npm is unavailable"
            prefix = tooling / "node"
            created = not prefix.exists()
            command = [
                npm, "install", "--prefix", str(prefix), "--ignore-scripts",
                "--no-audit", "--no-fund", package,
            ]
            bin_dir = prefix / "node_modules" / ".bin"
            cleanup_command = [
                npm, "uninstall", "--prefix", str(prefix), "--ignore-scripts",
                "--no-audit", "--no-fund", package,
            ]
            remove_on_failure = prefix if created else None
        elif ecosystem == "brew":
            brew = shutil.which("brew")
            if not brew:
                return "tool install failed: Homebrew is unavailable"
            if "/" in package:
                return "tool request rejected: third-party Homebrew taps are not automatic"
            info = subprocess.run(
                [brew, "info", "--json=v2", "--formula", package],
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
                env=scrubbed_environment(self.root),
            )
            try:
                formulae = json.loads(info.stdout).get("formulae") or []
                tap = str((formulae[0] if formulae else {}).get("tap") or "")
            except Exception:
                tap = ""
            if info.returncode != 0 or tap not in {"homebrew/core", ""}:
                return "tool request rejected: formula is not resolvable from Homebrew core"
            check = subprocess.run(
                [brew, "list", "--formula", package], capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=60,
            )
            already = check.returncode == 0
            command = [brew, "install", package]
        else:
            return "tool request rejected: unsupported ecosystem"

        env = scrubbed_environment(self.root)
        env.update({
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_NO_INPUT": "1",
            "GIT_TERMINAL_PROMPT": "0",
        })
        started = time.monotonic()
        try:
            result = subprocess.run(
                command, cwd=self.root, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=timeout, env=env,
            )
            ok = result.returncode == 0
            detail = (result.stdout + result.stderr)[-3000:]
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        cleanup = ""
        if not ok:
            if remove_on_failure is not None:
                shutil.rmtree(remove_on_failure, ignore_errors=True)
                cleanup = "partial tool environment removed"
            elif cleanup_command is not None:
                removed = subprocess.run(
                    cleanup_command, capture_output=True, text=True,
                    stdin=subprocess.DEVNULL, timeout=300, env=env,
                )
                cleanup = (
                    "partial package removed" if removed.returncode == 0
                    else "package cleanup failed"
                )
            elif ecosystem == "brew" and not already:
                brew = command[0]
                removed = subprocess.run(
                    [brew, "uninstall", "--force", package],
                    capture_output=True, text=True, stdin=subprocess.DEVNULL,
                    timeout=300, env=env,
                )
                cleanup = "failed formula removed" if removed.returncode == 0 else "cleanup failed"
        if ok and bin_dir:
            current = self.environment.get("PATH") or os.environ.get("PATH", "")
            self.environment["PATH"] = str(bin_dir) + os.pathsep + current
        self._record({
            "kind": "tool-install", "ecosystem": ecosystem, "package": package,
            "command": [Path(command[0]).name, *command[1:]],
            "ok": ok, "seconds": round(time.monotonic() - started, 2),
            "credential_environment": "scrubbed",
            "cleanup": cleanup, "detail_tail": detail[-1200:],
        })
        return (
            f"tool installed: {ecosystem} {package}"
            + (f"; PATH includes {bin_dir}" if bin_dir else "")
            if ok else f"tool install failed: {detail[-1200:]}{'; ' + cleanup if cleanup else ''}"
        )
