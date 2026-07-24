"""Numerical experiment runner — the loop's second verifier.

Some claims are not closed-form: a proposed spectrum, a Monte-Carlo cross-section, a
lattice sum, a stability scan. Those are checked by *running code*, not by algebra. This
executes a Python snippet the reasoning model writes, in a separate process with a
timeout, in a scratch dir, screened by spiral's command denylist — and reports what it
printed. Like every other verifier here, it exists so a numerical claim is a measurement,
not an assertion.

The snippet is model-written code and is therefore treated as such: it runs in its own
process (crashes/hangs are contained), is denylist-screened for obvious destructive/
network calls, and gets no arguments from untrusted fetched text. numpy/scipy/sympy are
available if installed; their absence just narrows what an experiment can do.
"""

from __future__ import annotations

import tempfile
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunOutput:
    ok: bool
    stdout: str
    error: str = ""
    timed_out: bool = False


_BANNED = (
    "shutil.rmtree", "os.remove", "os.unlink", "os.system", "subprocess",
    "socket.", "urllib.request", "httpx", "requests.", "open('/'", 'open("/',
    "__import__('os')", "eval(", "exec(", "os.rmdir", "sys.exit",
)


def _screen(code: str) -> str | None:
    low = code.replace(" ", "")
    for bad in _BANNED:
        if bad.replace(" ", "") in low:
            return bad
    return None


def _numeric_evidence_issue(code: str) -> str | None:
    if re.search(r"print\s*\(\s*(?:True|['\"]True['\"])\s*\)\s*$", code.strip(), re.I):
        return "constant success output"
    evidence = ("assert", "abs(", "allclose(", "isclose(", "norm(", "residual",
                "error", "<", ">", "==")
    if not any(signal in code.lower() for signal in evidence):
        return "no falsifiable numeric check"
    return None


def run_python(code: str, *, timeout: float = 20.0) -> RunOutput:
    """Run a Python snippet in an isolated process; return its stdout.

    The snippet should ``print`` whatever the loop needs to read back (a computed value,
    ``True``/``False``, a residual). A non-zero exit or a banned call is a failed run —
    an experiment that can't run is simply unverified, never a crash in the loop."""
    banned = _screen(code)
    if banned:
        return RunOutput(False, "", error=f"blocked call in snippet: {banned}")
    from spiral.research_workbench import run_workbench_claim

    with tempfile.TemporaryDirectory() as td:
        result = run_workbench_claim({
            "kind": "workbench",
            "files": {"experiment.py": code},
            "cmd": "python experiment.py",
            "expect": "",
            "note": "isolated numerical experiment",
            "evidence_mode": "exploratory",
        }, Path(td), timeout=timeout)
        return RunOutput(
            result.ok,
            result.stdout,
            error=result.stderr or ("" if result.ok else result.detail),
            timed_out=result.timed_out,
        )


def check_numeric_claim(code: str, *, expect: str = "True", timeout: float = 20.0) -> RunOutput:
    """Run ``code`` and require its final printed line to equal ``expect`` (default the
    string ``True``) — the convention for a snippet that prints its own pass/fail."""
    issue = _numeric_evidence_issue(code)
    if issue:
        return RunOutput(False, "", error=f"inconclusive numeric certificate: {issue}")
    r = run_python(code, timeout=timeout)
    if not r.ok:
        return r
    last = r.stdout.splitlines()[-1].strip() if r.stdout else ""
    return RunOutput(last == expect, r.stdout,
                     error="" if last == expect else f"expected {expect!r}, got {last!r}")
