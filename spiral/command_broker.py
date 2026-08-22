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
import signal
import stat
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

# The backstop that survives even --full-access. Everything else opens up in that
# mode — arbitrary paths, network, self-modification — but these are the actions
# an autonomous loop can take once, by hallucination, and never undo: wiping a
# disk or the home root, halting the machine, or a fork bomb. `sudo` stays out
# because a detached run cannot answer its password prompt and would simply hang.
# Git mutations are rejected separately by policy_error in every access mode.
_CATASTROPHIC_TOKENS = (
    "mkfs", "diskutil erase", "dd of=/dev/", "shutdown", "reboot", ":(){",
    "sudo ", "> /dev/disk", "> /dev/rdisk",
)
_CATASTROPHIC_RM = re.compile(
    r"\brm\s+-[a-z]*[rf][a-z]*\s+(?:-\S+\s+)*(?:/|~|\$home|/\*|~/\*)(?:\s|/|$)", re.I)
_NO_PUSH = re.compile(r"(?:^|[;&|]\s*|\s)(?:git\s+push|gh\s+(?:pr|issue))\b", re.I)


def catastrophic_error(command: str) -> str:
    """The only policy that holds in full-access mode. '' when the command is safe
    enough to run with the whole machine in reach."""
    low = f" {command.lower()} "
    for token in _CATASTROPHIC_TOKENS:
        if token in low:
            return (f"blocked even in full-access: '{token.strip()}' can destroy the "
                    "machine or hang a detached run")
    if _CATASTROPHIC_RM.search(command):
        return "blocked even in full-access: recursive delete of a filesystem or home root"
    if _NO_PUSH.search(command):
        return "publishing is yours, not the run's — spiral never pushes or opens PRs"
    return ""
_GIT_COMMAND = re.compile(
    r"(?:^|[^A-Za-z0-9_.-])(?:/[^\s;&|]*/)?git\b(?P<tail>[^;&|\n]*)", re.I)
_GIT_READ_ONLY = {
    "status", "diff", "log", "show", "grep", "blame", "shortlog",
    "describe", "rev-parse", "ls-files", "ls-tree", "cat-file", "name-rev",
    "merge-base", "check-ignore", "check-attr", "version", "help", "whatchanged",
}


def _git_mutation_requested(command: str) -> bool:
    """Conservatively admit only a small set of inspection-only Git verbs."""

    takes_value = {
        "-C", "-c", "--git-dir", "--work-tree", "--namespace",
        "--super-prefix", "--config-env", "--exec-path",
    }
    for match in _GIT_COMMAND.finditer(command):
        try:
            tokens = shlex.split("git" + match.group("tail"))
        except ValueError:
            return True
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in takes_value:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            return token.lower() not in _GIT_READ_ONLY
        # Bare `git` / `git --version` only displays help/version information.
    return False
_NODE_PACKAGE = re.compile(
    r"^(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+"
    r"(?:@[A-Za-z0-9*^~<>=_.+-]+)?$"
)
_BREW_FORMULA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,100}$")


def shell_executable() -> str:
    """Return a real POSIX shell on this host.

    macOS always ships zsh, while the Linux runner (and many minimal Linux
    installations) does not.  Commands are deliberately limited to portable
    shell syntax, so bash/sh are valid fallbacks instead of a hidden platform
    dependency.
    """
    if sys.platform == "darwin" and Path("/bin/zsh").is_file():
        return "/bin/zsh"
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def scrubbed_environment(
    workspace: str | Path,
    extra: dict | None = None,
    *,
    full_access: bool = False,
) -> dict[str, str]:
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
        # Workspace runs must not discover dotfiles, credentials, or unrelated
        # personal data through ordinary `~` expansion. An explicitly approved
        # full-access run has different semantics: point `~` at the user's actual
        # home so "find the project in Documents" works like it does in an
        # interactive coding agent. Secret-valued process variables remain omitted.
        "HOME": str(Path.home()) if full_access else str(home),
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


def canonical_reference_roots(
    paths, *, workspace: str | Path,
) -> list[str]:
    """Resolve an explicit reference grant to stable, non-overlapping objects.

    A reference may be one regular file (for example a paper) or one directory
    (for example an existing project).  Ancestors/descendants of the writable
    target are rejected: otherwise one path would simultaneously be advertised
    as read-only data and writable project state.  Device/inode identity is
    captured again by :class:`CommandBroker` and checked before every command.
    """

    root = Path(workspace).expanduser().resolve(strict=False)
    rows: list[str] = []
    seen: set[tuple[int, int]] = set()
    for raw in paths or []:
        value = os.fspath(raw)
        if not value.strip() or "\x00" in value:
            raise ValueError("a reference path is empty or contains NUL")
        try:
            resolved = Path(value).expanduser().resolve(strict=True)
            info = resolved.stat()
        except OSError as exc:
            raise ValueError(f"{value!r} does not resolve to an existing path: {exc}") from exc
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"{resolved} is not a regular file or directory")
        try:
            resolved.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{resolved} is inside the writable target {root}")
        try:
            root.relative_to(resolved)
        except ValueError:
            pass
        else:
            raise ValueError(f"{resolved} contains the writable target {root}")
        identity = (int(info.st_dev), int(info.st_ino))
        if identity not in seen:
            seen.add(identity)
            rows.append(str(resolved))
    return rows


def require_reference_identities(paths: list[str], encoded: str) -> None:
    """Bind controller-approved paths to the exact filesystem objects it saw."""

    try:
        records = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("SPIRALCHAT_REFERENCE_IDENTITIES is not valid JSON") from exc
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("SPIRALCHAT_REFERENCE_IDENTITIES must be a JSON list of objects")
    indexed: dict[str, dict] = {}
    for row in records:
        path = row.get("path")
        if not isinstance(path, str) or not path or path in indexed:
            raise ValueError("reference identity records need unique canonical paths")
        indexed[path] = row
    if set(indexed) != set(paths):
        raise ValueError("controller reference identities do not exactly match --reference")
    aliases = {
        "dev": ("dev", "device", "st_dev"),
        "ino": ("ino", "inode", "st_ino"),
        "size": ("size", "st_size"), "mtime_ns": ("mtime_ns", "st_mtime_ns"),
    }
    for value in paths:
        info = Path(value).stat()
        actual = {
            "dev": int(info.st_dev), "ino": int(info.st_ino),
            "size": int(info.st_size), "mtime_ns": int(info.st_mtime_ns),
        }
        row = indexed[value]
        for field in ("dev", "ino"):
            names = aliases[field]
            supplied = next((row[name] for name in names if name in row), None)
            if supplied is None or int(supplied) != actual[field]:
                raise ValueError(f"controller identity changed for reference: {value}")
        for field in ("size", "mtime_ns"):
            names = aliases[field]
            supplied = next((row[name] for name in names if name in row), None)
            if supplied is not None and int(supplied) != actual[field]:
                raise ValueError(f"controller identity metadata changed for reference: {value}")


def _reference_read_rule(path: Path) -> str:
    kind = "subpath" if path.is_dir() else "literal"
    return f'(allow file-read* ({kind} "{_sandbox_string(path)}"))'


def _reference_write_deny_rule(path: Path) -> str:
    kind = "subpath" if path.is_dir() else "literal"
    return f'(deny file-write* ({kind} "{_sandbox_string(path)}"))'


@dataclass
class BrokerResult:
    result: RunResult
    sandboxed: bool
    manifest: str


def _process_group_alive(process: subprocess.Popen) -> bool:
    """Whether anything remains in the session-owned process group."""
    process.poll()  # reap the leader before probing for surviving descendants
    if os.name != "posix":
        return process.returncode is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_tree(
    process: subprocess.Popen, *, interrupted: bool = False,
) -> None:
    """Stop a broker-owned process group, escalating when descendants resist.

    ``start_new_session`` makes the Popen pid the process-group id on POSIX.
    Signals therefore cover the shell, its command, and ordinary descendants,
    even when the shell exits before one of its children does.
    """
    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        return

    sequence = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL] if interrupted else [
        signal.SIGTERM, signal.SIGKILL,
    ]
    for signum in sequence:
        if not _process_group_alive(process):
            break
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + (0.75 if signum != signal.SIGKILL else 0.25)
        while time.monotonic() < deadline:
            if not _process_group_alive(process):
                break
            time.sleep(0.025)
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass


class CommandBroker:
    def __init__(self, workspace: str | Path, cfg=None):
        self.root = Path(workspace).resolve()
        self.cfg = cfg
        granted = getattr(cfg, "builder_reference_roots", []) if cfg else []
        canonical = canonical_reference_roots(granted, workspace=self.root)
        self.reference_roots = tuple(Path(path) for path in canonical)
        self._reference_identities = {}
        for path in self.reference_roots:
            info = path.stat()
            self._reference_identities[path] = (
                int(info.st_dev), int(info.st_ino), int(info.st_size),
                int(info.st_mtime_ns), stat.S_ISREG(info.st_mode),
            )
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
    def policy_error(command: str, full_access: bool = False) -> str:
        lowered = f" {command.lower()} "
        if len(command) > 2000 or "\x00" in command:
            return "command is empty, binary, or too long"
        # Git history, worktree and remote mutations always go through the
        # controller's separately approved typed endpoint.  A permission mode can
        # widen filesystem access, but it cannot grant the model an approval that
        # belongs to the user for this exact Git action.
        if _git_mutation_requested(command):
            return (
                "git history/worktree/remote mutations require a separate explicit "
                "user approval"
            )
        # Full-access keeps only the catastrophic backstop; the rest of the fence
        # (network tools, communication, arbitrary git inspection, workspace-only
        # writes) is exactly what this mode exists to remove.
        if full_access:
            return catastrophic_error(command)
        if _COMMUNICATION.search(command):
            return "communication, messaging, remote login, and social actions are forbidden"
        if any(token in lowered for token in _FORBIDDEN):
            return "command requests a destructive, remote, or acquisition action"
        if re.search(r"(?:^|\s)(?:open|xdg-open)\s+https?://", command, re.I):
            return "opening remote URLs belongs to the research/browser broker"
        if ".spiral/tools/" in command.replace("\\", "/"):
            return "inspection-only repository caches may not be executed"
        return ""

    def _reference_error(self) -> str:
        for path, expected in self._reference_identities.items():
            try:
                info = path.stat()
            except OSError as exc:
                return f"approved reference disappeared: {path}: {exc}"
            if (int(info.st_dev), int(info.st_ino)) != expected[:2]:
                return f"approved reference identity changed: {path}"
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                return f"approved reference is no longer a regular file or directory: {path}"
            if expected[4] and (
                int(info.st_size), int(info.st_mtime_ns)
            ) != expected[2:4]:
                return f"approved reference file content changed: {path}"
        return ""

    def _argv(
        self, command: str, cwd: Path, allow_network: bool,
        allow_host_read: bool, full_access: bool = False,
    ) -> tuple[list[str], bool]:
        # Full access is the whole machine, on purpose: writes anywhere, reads
        # anywhere, network on — so spiral can find a folder across the disk,
        # write into it, browse, and modify its own installed source. There is no
        # confinement profile to build; the catastrophic backstop in policy_error
        # is the only thing standing. Reported as sandboxed=False so the run log
        # is honest about what it was.
        if full_access:
            return [shell_executable(), "-lc", command], False
        # Workspace confinement is an enforcement mode, not a hint from a caller.
        # Older local-model call sites passed allow_host_read=True because prompts
        # stayed on-device; that still exposed SSH keys and every other home file to
        # the model. Only explicit full access may widen filesystem reads.
        allow_host_read = False
        sandbox = shutil.which("sandbox-exec")
        if sandbox and sys.platform == "darwin":
            # macOS TMPDIR carries a trailing slash, and a sandbox `subpath`
            # with one matches nothing — every write to $TMPDIR (which is where
            # python's tempfile points) was silently denied inside the gate.
            # /var is also a symlink to /private/var and the kernel checks the
            # REAL path, so both spellings must be allowed.
            temp = (os.environ.get("TMPDIR") or "/tmp").rstrip("/") or "/tmp"
            temp_real = str(Path(temp).resolve())
            profile = [
                "(version 1)",
                "(allow default)",
            ]
            if not allow_network:
                profile.append("(deny network*)")
            if not allow_host_read:
                tool_roots = [
                    # the harness runs its own gates (artifact_gate, footguns) with
                    # sys.executable — an editable/clone install lives outside the
                    # packaged tool roots below, and denying it exits the gate 127.
                    # NOT resolve()d: that follows a venv's symlink out to the base
                    # interpreter and allows the wrong directory.
                    Path(sys.executable).parent.parent,
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    # spiral's own package tree, for the same reason: an editable
                    # install imports it from the clone, not from site-packages
                    Path(__file__).resolve().parent.parent,
                    Path.home() / "Library/Android/sdk",
                    Path.home() / ".cargo",
                    Path.home() / ".rustup",
                    Path.home() / ".elan",
                    Path.home() / ".local/bin",
                    # pyenv shims are how `python` resolves on machines that use
                    # it; denying them made every ladder rung fail with
                    # "Operation not permitted", which the rung label then
                    # misreported as a syntax error
                    Path.home() / ".pyenv",
                    Path.home() / ".local/pipx",
                    Path.home() / ".local/share/uv",
                    Path.home() / ".cache/uv",
                    Path.home() / ".cache/pip",
                ]
                profile += [
                    f'(deny file-read* (subpath "{_sandbox_string(Path.home())}"))',
                    # ...but let anything be STAT-ed. Build tools resolve their root
                    # by walking ancestors of the workspace (pytest looks for an
                    # inifile, npm for package.json, cargo for a workspace manifest);
                    # denying metadata makes that walk raise PermissionError and the
                    # gate is then permanently red for any project under $HOME.
                    # Contents stay denied — only existence/size/mode leak.
                    "(allow file-read-metadata)",
                    f'(allow file-read* (subpath "{_sandbox_string(self.root)}"))',
                    f'(allow file-read* (subpath "{_sandbox_string(temp)}"))',
                    f'(allow file-read* (subpath "{_sandbox_string(temp_real)}"))',
                    *[
                        f'(allow file-read* (subpath "{_sandbox_string(path)}"))'
                        for path in tool_roots if path.exists()
                    ],
                    *[_reference_read_rule(path) for path in self.reference_roots],
                ]
            profile += [
                "(deny signal (target others))",
                "(deny file-write*)",
                f'(allow file-write* (subpath "{_sandbox_string(self.root)}"))',
                f'(allow file-write* (subpath "{_sandbox_string(temp)}"))',
                f'(allow file-write* (subpath "{_sandbox_string(temp_real)}"))',
                '(allow file-write* (subpath "/tmp"))',
                '(allow file-write* (subpath "/private/tmp"))',
                '(allow file-write* (literal "/dev/null"))',
                f'(deny file-write* (subpath "{_sandbox_string(self.root / ".git")}"))',
                f'(deny file-write* (subpath "{_sandbox_string(self.root / ".spiral" / "tools")}"))',
                *[_reference_write_deny_rule(path) for path in self.reference_roots],
            ]
            return [
                sandbox, "-p", " ".join(profile), shell_executable(), "-lc", command,
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
            # These binds are intentionally explicit even though the base tree is
            # read-only: they preserve the grant when the Linux profile is later
            # narrowed without accidentally turning references writable.
            for reference in self.reference_roots:
                argv.extend(["--ro-bind", str(reference), str(reference)])
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
        full_access: bool = False,
    ) -> BrokerResult:
        command = str(command or "").strip()
        # The broker is the final capability boundary. A stale or hostile caller
        # cannot turn a workspace run into a host-reading or networked run by passing
        # permissive booleans; only full_access authorizes either capability.
        if not full_access:
            allow_network = False
            allow_host_read = False
        error = self.policy_error(command, full_access=full_access)
        reference_error = self._reference_error()
        if reference_error:
            error = reference_error
        # Full access lets the working directory be anywhere on disk — that is the
        # "find this folder across my machine and work in it" case. Otherwise the
        # cwd must stay inside the workspace.
        work = Path(cwd).resolve() if cwd else self.root
        if not full_access:
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
            command, work, allow_network, allow_host_read, full_access=full_access)
        # Full access is deliberately unsandboxed; require_sandbox does not apply.
        if require_sandbox and not sandboxed and not full_access:
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

        env = scrubbed_environment(
            self.root, self.environment, full_access=full_access)
        started = time.monotonic()
        lines: list[str] = []
        process: subprocess.Popen | None = None
        timed_out = False
        try:
            process = subprocess.Popen(
                argv, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, stdin=subprocess.DEVNULL, env=env,
                start_new_session=(os.name == "posix"),
            )
            assert process.stdout is not None
            deadline = time.monotonic() + max(1, timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _terminate_process_tree(process)
                    timed_out = True
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
            code = 124 if timed_out else process.wait(timeout=15)
        except KeyboardInterrupt:
            if process is not None:
                _terminate_process_tree(process, interrupted=True)
            raise
        except Exception as exc:
            if process is not None:
                _terminate_process_tree(process)
            code = 124
            lines.append(f"\n(command broker error: {type(exc).__name__}: {exc})")
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
        output = "".join(lines).strip()
        result = RunResult(command, code, output)
        try:
            audit_cwd = str(work.relative_to(self.root) or ".")
        except ValueError:
            audit_cwd = str(work)
        manifest = self._record({
            "kind": purpose,
            "command": command,
            "cwd": audit_cwd,
            "sandboxed": sandboxed,
            "network": "allowed" if allow_network or full_access else "denied",
            "host_read": (
                "allowed" if allow_host_read or full_access else
                "workspace+references" if self.reference_roots else "workspace-only"
            ),
            "reference_roots": [str(path) for path in self.reference_roots],
            "environment_keys": sorted(env),
            "exit": code,
            "seconds": round(time.monotonic() - started, 2),
            "output_tail": output[-2000:],
            "ok": code == 0,
        })
        return BrokerResult(result, sandboxed, manifest)

    def provision(
        self, request: str, *, timeout: int = 900, full_access: bool = False,
    ) -> str:
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
            if not full_access:
                return "tool request rejected: Homebrew provisioning requires full access"
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
