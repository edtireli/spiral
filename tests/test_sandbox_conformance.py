"""The sandbox must starve the model, never the harness.

One test per capability the sandbox was observed denying the gate itself: exec of
the interpreter that runs the rungs, stat on workspace ancestors, pyenv shims,
writes to TMPDIR. Each of these produced hours of misattributed failure before it
was found; the suite makes the next regression cost one test run instead.

Runs under pytest; skips cleanly off macOS.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.command_broker import CommandBroker  # noqa: E402
from spiral.config import Config  # noqa: E402


def _run(root: Path, command: str):
    broker = CommandBroker(root, Config())
    return broker.run(
        command, timeout=60, purpose="verification-gate", allow_network=False,
        allow_host_read=False, require_sandbox=True).result


def _darwin() -> bool:
    import shutil

    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def test_the_gates_own_interpreter_is_executable(tmp_path):
    if not _darwin():
        return
    result = _run(tmp_path, f"{sys.executable} -c 'print(\"alive\")'")
    assert result.code == 0, result.out[-200:]
    assert "alive" in result.out


def test_ancestors_can_be_stat_ed_but_not_read(tmp_path):
    """pytest walks ancestors for its inifile; blanket read-denial pinned the
    gate red for every project under $HOME."""
    if not _darwin():
        return
    home = Path.home()
    result = _run(
        tmp_path,
        f'{sys.executable} -c "import os; os.stat(\'{home}\'); print(\'stat ok\')"')
    assert result.code == 0, result.out[-200:]
    denied = _run(tmp_path, f"ls {home} > /dev/null 2>&1 && echo LISTED || echo DENIED")
    assert "DENIED" in denied.out, "content enumeration under $HOME must stay denied"


def test_pyenv_shims_are_executable_when_present(tmp_path):
    """`python` resolves through pyenv on machines that use it; denying the shim
    made every rung exit 126, reported as a syntax error for six attempts."""
    if not _darwin():
        return
    shim = Path.home() / ".pyenv" / "shims" / "python"
    if not shim.exists():
        return
    result = _run(tmp_path, f"{shim} -c 'print(\"shim ok\")'")
    assert result.code == 0, result.out[-200:]


def test_workspace_and_tmp_are_writable_home_is_not(tmp_path):
    if not _darwin():
        return
    ok = _run(tmp_path, "echo x > probe.txt && cat probe.txt")
    assert ok.code == 0 and "x" in ok.out
    tmp = _run(tmp_path, "echo y > \"$TMPDIR/spiral-conformance-probe\" && echo TMP_OK")
    assert "TMP_OK" in tmp.out, tmp.out[-200:]
    home = _run(tmp_path, f"touch {Path.home()}/spiral-conformance-should-fail 2>/dev/null && echo WROTE || echo REFUSED")
    assert "REFUSED" in home.out
