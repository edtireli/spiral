"""Worker-facing tools: read context, and run commands. Edits go through edits.py.

`run()` is both the verify primitive and the model's shell. A denylist blocks
genuinely destructive ops even in full-auto — unattended autonomy is only safe if
it physically cannot wipe the repo or reach the network to push.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Blocked even when the user runs full-auto. Substrings, matched case-insensitively.
DENY = (
    "rm -rf", "rm -fr", "rm -r /", ":(){", "mkfs", "dd if=", "> /dev/",
    "shutdown", "reboot", "sudo ", "git push", "git reset --hard",
    "curl ", "wget ", "chmod -r", "chown -r", "> ~", "history -c",
)

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}


@dataclass
class RunResult:
    cmd: str
    code: int
    out: str
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.blocked


def is_dangerous(cmd: str) -> bool:
    c = " " + cmd.strip().lower() + " "
    return any(bad in c for bad in DENY)


def run(cmd: str, cwd: str | Path, timeout: int = 120, on_line=None) -> RunResult:
    """Run a shell command. With on_line, output is streamed line-by-line to the
    callback as it happens (build liveness) while still captured in full."""
    if is_dangerous(cmd):
        return RunResult(cmd, 126, f"blocked by denylist: {cmd!r}", blocked=True)
    if on_line is None:
        try:
            p = subprocess.run(
                cmd, shell=True, cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout,
            )
            return RunResult(cmd, p.returncode, (p.stdout + p.stderr).strip())
        except subprocess.TimeoutExpired:
            return RunResult(cmd, 124, f"timed out after {timeout}s")

    import time as _time
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(cwd), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    lines: list[str] = []
    deadline = _time.monotonic() + timeout
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            try:
                on_line(line.rstrip())
            except Exception:
                pass
            if _time.monotonic() > deadline:
                proc.kill()
                lines.append(f"\n(timed out after {timeout}s)")
                break
        code = proc.wait(timeout=30)
    except Exception as e:
        proc.kill()
        return RunResult(cmd, 124, "".join(lines) + f"\n(stream error: {e})")
    return RunResult(cmd, code, "".join(lines).strip())


def read_file(root: str | Path, path: str, start: int | None = None, end: int | None = None) -> str:
    fp = Path(root) / path
    if not fp.is_file():
        return f"(no such file: {path})"
    lines = fp.read_text(errors="replace").splitlines()
    s = (start - 1) if start else 0
    e = end if end else len(lines)
    return "\n".join(f"{s + 1 + i}\t{ln}" for i, ln in enumerate(lines[s:e]))


def list_dir(root: str | Path, path: str = ".") -> str:
    base = Path(root) / path
    if not base.is_dir():
        return f"(no such dir: {path})"
    out = []
    for p in sorted(base.iterdir()):
        if p.name in _SKIP_DIRS or p.name.startswith("."):
            continue
        out.append(p.name + ("/" if p.is_dir() else ""))
    return "\n".join(out) or "(empty)"


def grep(root: str | Path, pattern: str, path: str = ".", max_hits: int = 80) -> str:
    root = Path(root)
    base = root / path
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"(bad pattern: {e})"
    files = [base] if base.is_file() else base.rglob("*")
    hits: list[str] = []
    for f in files:
        if not f.is_file() or any(part in _SKIP_DIRS for part in f.parts):
            continue
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:200]}")
                if len(hits) >= max_hits:
                    return "\n".join(hits) + "\n(truncated)"
    return "\n".join(hits) if hits else "(no matches)"
