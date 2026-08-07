"""When the instrument is broken, a correct diagnosis must not be scored as failure.

Two failures motivate this. A composed gate once read as shell arithmetic — ``((…))`` —
so every task aborted at bootstrap while the model was told the *project* was red. And a
gate ending in ``|| true`` cannot report a failure at all, yet its green was accepted as
evidence. In both cases the model can see what is wrong and say so, and in both cases
saying so cost it an attempt, because the only reply the loop accepts is an edit.

So there are two halves here, and they are the same half twice:

- **the gate self-check** — a gate that cannot fail is not measuring anything, and a
  failure whose signature is ``command not found`` indicts the harness, not the code;
- **the harness-error channel** — the model may reply ``HARNESS_ERROR: <reason>``, and
  that claim is *adjudicated against the evidence*, never taken on its word.

Which is the project's contract applied to the project's own instruments: the model
proposes the diagnosis, deterministic code decides whether it holds. A confirmed claim
costs no attempt and surfaces the fault; a refuted one comes back with the specific
evidence that refutes it, so the next attempt is better informed rather than merely
scolded.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass

# ── faults that belong to the harness, not to the code under test ────────────
# Kept deliberately narrow. "No such file or directory" is far more often a real bug --
# a test opening a fixture that was never written -- than a broken harness, so it counts
# only when the thing not found is the command being executed. A detector that cries
# harness on ordinary red is worse than none: it hands the model an excuse.


class HarnessFault(RuntimeError):
    """The instrument is broken, so no verdict on the code is available.

    Raised only after a model's HARNESS_ERROR claim is corroborated by the gate's own
    output. It aborts the run rather than failing the task: a gate that cannot measure
    this task cannot measure the next one either, escalating to a second model only
    walks it into the same wall, and every "failure" reported afterwards would be the
    harness's, not the code's."""


@dataclass(frozen=True)
class Fault:
    """One reason the failure is the instrument's, not the code's."""
    kind: str          # missing-tool | permission | sandbox | resource | network |
                       # shell-parse | readonly
    detail: str        # what to do about it, in one line
    evidence: str      # the exact output line that proves it

    def sentence(self) -> str:
        return f"[{self.kind}] {self.detail} — evidence: {self.evidence.strip()[:160]}"


# Each rule owns one signature. The capture group, when present, names the missing tool.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("missing-tool", re.compile(
        r"(?:command not found|:\s*not found\s*$|"
        r"env:\s*[\"']?([\w.\-/]+)[\"']?:\s*No such file or directory)", re.I | re.M),
     "the gate invokes a program that is not installed on this machine"),
    ("missing-tool", re.compile(
        r"No module named ['\"]?(pytest|pip|setuptools|build|virtualenv|tox|nox|"
        r"mypy|ruff|flake8|black)['\"]?", re.I),
     "a tool the gate depends on is not importable in this interpreter"),
    ("permission", re.compile(r"[Pp]ermission denied.*(?:exec|/bin/|\.sh\b)|"
                              r"cannot execute:\s*[Pp]ermission denied"),
     "the gate cannot execute its own command (file mode, not code)"),
    # macOS ships this when a sandbox blocks a shared library. It reads like a broken
    # install and is not one; nothing the model edits can clear it.
    ("sandbox", re.compile(r"library load disallowed by system policy|"
                           r"denied by system policy|Operation not permitted.*dylib", re.I),
     "the sandbox is blocking a shared library — an environment restriction"),
    ("resource", re.compile(r"^Killed\s*$|Out of memory|Cannot allocate memory|"
                            r"No space left on device", re.I | re.M),
     "the run was killed for resources, so the result is not a verdict on the code"),
    ("network", re.compile(r"Temporary failure in name resolution|Could not resolve host|"
                           r"Network is unreachable|Connection refused.*(?:proxy|registry)|"
                           r"getaddrinfo (?:ENOTFOUND|EAI_AGAIN)", re.I),
     "the gate needs the network and the network is not reachable"),
    ("shell-parse", re.compile(r"syntax error near unexpected token|parse error near|"
                               r"bad math expression|unmatched ['\"]", re.I),
     "the gate does not parse as shell — the composed command is malformed"),
    ("readonly", re.compile(r"Read-only file system", re.I),
     "the workspace is not writable, so no edit could take effect"),
]

# Exit codes the shell reserves for "could not run it": 126 not executable,
# 127 not found, 124 timed out. 128+n is death by signal n.
_UNRUNNABLE = {124: "the gate timed out", 126: "the gate command is not executable",
               127: "the gate command was not found"}


def instrument_faults(out: str, code: int | None = None) -> list[Fault]:
    """Faults in the output that indict the instrument rather than the code.

    Empty when the failure is an ordinary red — which is the common case, and must
    stay the common case."""
    text = out or ""
    faults: list[Fault] = []
    seen: set[tuple[str, str]] = set()
    for kind, rx, detail in _RULES:
        m = rx.search(text)
        if not m:
            continue
        tool = next((g for g in m.groups() if g), None)
        line = next((ln for ln in text.splitlines() if m.group(0).strip()[:40] in ln),
                    m.group(0))
        why = f"{detail}{f' (missing: {tool})' if tool else ''}"
        key = (kind, why)
        if key in seen:
            continue
        seen.add(key)
        faults.append(Fault(kind, why, line))
    if code in _UNRUNNABLE and not any(f.kind == "missing-tool" for f in faults):
        faults.append(Fault("missing-tool", _UNRUNNABLE[code],
                            f"exit status {code}"))
    return faults


_BUILTINS = {"cd", "echo", "exit", "set", "test", "true", "false", "if", "then",
             "else", "elif", "fi", "for", "do", "done", "while", "until", "case",
             "esac", "return", "export", "source", "eval", "read", "shift", "trap",
             "unset", "local", "printf", "command", "wait", "exec", "time", ":",
             "[", "[[", "{", "}", "(", ")"}
# after these, the next word is a command name; anywhere else it is an argument
_CMD_POSITION = {"&&", "||", "|", ";", "&", "(", "{", "\n", "!"}


def missing_tools(command: str) -> list[str]:
    """Programs the gate names that PATH cannot resolve.

    Static, so it runs before the gate does — but it must never invent one. An earlier
    version split on `;` with a regex and so walked *into* quoted strings: a rung's own
    ``echo 'spiral rung failed: parse — …; fix the syntax error above'`` was read as two
    commands named ``this`` and ``fix``. So tokenize with shlex, which respects quoting,
    and bail out entirely on constructs it cannot analyse. Returning nothing is the
    right answer whenever the shape is uncertain: exit 127 catches at run time what this
    misses, but a fabricated 'missing tool' is a false accusation with no backstop."""
    import shlex

    text = command or ""
    if not text.strip():
        return []
    if "$(" in text or "`" in text:
        return []          # command substitution — the nesting is past this analysis
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []          # unbalanced quotes; the parse check owns that failure
    missing: list[str] = []
    expect_command = True
    for tok in tokens:
        if any(ch in tok for ch in "&|;"):     # an operator token
            expect_command = True
            continue
        if not expect_command:
            continue
        expect_command = False
        word = tok.strip("({")
        if (not word or word in _BUILTINS or "/" in word or "=" in word
                or not re.fullmatch(r"[\w.\-]+", word)):
            continue
        if shutil.which(word) is None and word not in missing:
            missing.append(word)
    return missing


# ── a gate that cannot fail ──────────────────────────────────────────────────
# Statically decidable cases only. `|| true` is the one that actually happens: it is
# added to silence a flaky step and it silences every other step with it.
_ALWAYS_GREEN_TAIL = re.compile(r"(?:\|\||;)\s*(?:true|:|exit\s+0)\s*\)?\s*$")
_NO_OP = re.compile(r"^\s*\(?\s*(?:true|:|exit\s+0)\s*\)?\s*$")


def vacuous_gate(command: str) -> str | None:
    """Why this gate can never report a failure, or None if it can.

    One-directional on purpose: a `None` here means "not provably vacuous", not
    "this gate is good". Proving a gate meaningful requires breaking the code and
    watching it go red, which is `regress`'s job, not a static check's."""
    cmd = (command or "").strip()
    if not cmd:
        return "the gate is empty, so every task is green by default"
    body = "\n".join(ln for ln in cmd.splitlines()
                     if ln.strip() and not ln.strip().startswith("#")).strip()
    if not body:
        return "the gate is only comments, so it runs nothing"
    if _NO_OP.match(body):
        return f"the gate is `{body}`, which always exits 0 without testing anything"
    if _ALWAYS_GREEN_TAIL.search(body):
        return ("the gate ends in `|| true` (or `; true`), which discards the exit "
                "status of everything before it — no failure can reach the caller")
    if re.search(r"\bset\s+\+e\b", body) and not re.search(r"\bset\s+-e\b", body):
        return ("the gate runs `set +e` and never restores it, so a failing step "
                "does not stop the script or change its exit status")
    return None


# ── the channel: a claim, adjudicated ────────────────────────────────────────
_CLAIM = re.compile(r"^\s*HARNESS[_ ]ERROR\s*:?\s*(?P<why>.*)", re.I | re.M)


@dataclass
class Verdict:
    """The result of checking a model's harness-fault claim against the evidence."""
    claimed: bool                       # did the reply make the claim at all
    confirmed: bool = False             # does the evidence bear it out
    faults: list[Fault] = None          # what the evidence actually shows
    reason: str = ""                    # returned to the model when refuted
    claim_text: str = ""

    def __post_init__(self):
        if self.faults is None:
            self.faults = []


def claim_in(reply: str) -> str | None:
    """The model's stated reason, if the reply opens the channel. None otherwise.

    Required near the top of the reply so that merely *discussing* a harness problem
    mid-explanation does not trip it — the channel has to be chosen, not stumbled into."""
    m = _CLAIM.search((reply or "")[:400])
    return (m.group("why").strip() or "(no reason given)") if m else None


def adjudicate(reply: str, gate_out: str, code: int | None,
               command: str = "") -> Verdict:
    """Check a HARNESS_ERROR claim against the gate's own output.

    The model gets no credit for asserting it. Either the output carries a fault
    signature, or the gate is statically unable to fail, or the claim is refused with
    the reason — and the attempt is scored as any other."""
    why = claim_in(reply)
    if why is None:
        return Verdict(claimed=False)

    command = command or ""
    faults = instrument_faults(gate_out or "", code)
    vacuity = vacuous_gate(command)
    if vacuity:
        faults = faults + [Fault("vacuous-gate", vacuity, command[:160] or "(no gate)")]
    absent = [t for t in missing_tools(command) if t]
    if absent and not any(f.kind == "missing-tool" for f in faults):
        faults = faults + [Fault(
            "missing-tool",
            f"the gate names {', '.join(absent)}, which PATH cannot resolve",
            command[:160])]

    if faults:
        return Verdict(True, True, faults, claim_text=why)
    return Verdict(
        True, False, [],
        reason=(
            "You reported a harness error, but the evidence does not support it: the "
            "gate parses, every program it names resolves on PATH, it is able to "
            "report a failure, and its output carries no environment-fault signature "
            f"(exit status {code}). The failure is in the code under test. Fix it with "
            "edit blocks, or if you still believe the instrument is at fault, say "
            "which exact line of gate output shows it."),
        claim_text=why)


def report(faults: list[Fault]) -> str:
    """Operator-facing summary of a confirmed harness fault."""
    if not faults:
        return ""
    lines = ["The build gate is not a usable instrument here — this is not a verdict "
             "on the code:"]
    lines += [f"  - {f.sentence()}" for f in faults]
    return "\n".join(lines)
