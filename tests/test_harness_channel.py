"""The channel, driven through the real Atom loop and the real gate composer.

Unit tests prove the adjudicator is sound. These prove it is actually wired: that a
confirmed fault stops the run instead of blaming the code, that a refused claim costs
the model its edit but not the whole lane, and that a gate which cannot fail is refused
at composition rather than believed.
"""
import subprocess
import types

import pytest

from spiral.agent import Atom, TaskSpec
from spiral.config import Config
from spiral.harness_check import HarnessFault


class Quiet:
    """Dash stand-in — swallows the cockpit calls the loop makes."""
    def print(self, *a, **k): pass
    def __getattr__(self, _): return lambda *a, **k: None


def _repo(tmp_path):
    subprocess.run("git init -q .", shell=True, cwd=tmp_path, check=True)
    subprocess.run("git config user.email t@t && git config user.name t",
                   shell=True, cwd=tmp_path, check=True)
    (tmp_path / "thing.py").write_text("X = 1\n")
    subprocess.run("git add -A && git commit -qm init", shell=True, cwd=tmp_path,
                   check=True)
    return tmp_path


def _atom_replying(tmp_path, reply: str):
    """An Atom whose model always says `reply`, with the token accounting stubbed."""
    atom = Atom(_repo(tmp_path), Config())
    calls = {"n": 0}

    def chat(model, msgs, **kw):
        calls["n"] += 1
        return types.SimpleNamespace(
            text=reply, completion_tokens=20, prompt_tokens=100, total_tokens=120,
            total_duration=0, eval_count=20, load_duration=0, eval_duration=0)

    atom.ol = types.SimpleNamespace(
        chat=chat, health=lambda: "stub", evict=lambda *a, **k: None)
    return atom, calls


# ------------------------------------------------------------------ confirmed
def test_a_confirmed_fault_stops_the_run_instead_of_blaming_the_code(tmp_path):
    """The gate names a program that does not exist. Nothing the model edits can fix
    that, so the honest outcome is to stop and say so."""
    atom, calls = _atom_replying(
        tmp_path, "HARNESS_ERROR: the gate calls a tool that is not installed")
    # Hide the inner command from static top-level preflight so this test continues
    # to exercise the model-authored HARNESS_ERROR channel itself.  A directly named
    # missing binary is now caught even earlier, without spending a model call.
    spec = TaskSpec("add a feature",
                    "sh -c 'definitely-not-a-real-binary-xyz --run'")

    with pytest.raises(HarnessFault) as caught:
        atom.run(spec, ui=Quiet(), attempts=3)

    assert "not a verdict on the code" in str(caught.value)
    assert calls["n"] == 1, "it kept asking the model after the fault was proven"


def test_the_original_arithmetic_gate_bug_is_diagnosable(tmp_path):
    """`((…))` composed into a gate read as shell arithmetic and reported every healthy
    project red. The model can now say so and be believed on the evidence."""
    from spiral.harness_check import adjudicate

    v = adjudicate("HARNESS_ERROR: the gate is not valid shell",
                   "zsh: bad math expression: operand expected", 1,
                   "((pytest -q && ruff check .))")
    assert v.confirmed and any(f.kind == "shell-parse" for f in v.faults)


# ------------------------------------------------------------------ refused
def test_a_refused_claim_gets_one_correction_then_is_charged(tmp_path):
    """An ordinary red must not become unfalsifiable by asserting harness trouble.
    The model is corrected once, then the claim is charged like any empty reply."""
    atom, calls = _atom_replying(tmp_path, "HARNESS_ERROR: the test runner is broken")
    spec = TaskSpec("fix the failure", "python3 -c \"raise SystemExit(1)\"")

    # diversity=False isolates the attempt loop from the best-of-N sampler that runs
    # at lane exit, so the count below is the loop's own budget and nothing else.
    assert atom.run(spec, ui=Quiet(), attempts=3, diversity=False) is False
    # 3 attempts + exactly one uncharged correction — bounded, not an escape hatch
    assert calls["n"] == 4, f"expected 3 attempts + 1 free correction, got {calls['n']}"


def test_a_repeated_correct_diagnosis_is_not_punished_as_a_repeat(tmp_path):
    """A model that is right about the gate says the same true thing every time. The
    identical-reply detector would answer that with 'take a different approach', so the
    adjudication has to run ahead of it — otherwise the channel is unreachable from
    attempt two onward, which is precisely the bug this closes."""
    atom, calls = _atom_replying(tmp_path, "HARNESS_ERROR: the gate tool is missing")
    spec = TaskSpec(
        "add a feature", "sh -c 'definitely-not-a-real-binary-xyz --run'")

    with pytest.raises(HarnessFault):
        atom.run(spec, ui=Quiet(), attempts=3, diversity=False)
    assert calls["n"] == 1, "the fault was proven on the first reply; it kept asking"


def test_the_claim_must_lead_the_reply(tmp_path):
    """Mentioning a harness problem inside a long explanation is not opening the
    channel — otherwise ordinary reasoning about the environment would trip it."""
    from spiral.harness_check import claim_in

    assert claim_in("The gate looks fine.\n" + "detail " * 90 +
                    "\nHARNESS_ERROR: buried") is None


# ------------------------------------------------------------------ gate self-check
@pytest.mark.parametrize("bad", ["pytest -q || true", "ruff check . ; true",
                                 "npm test || :", "set +e\npytest -q"])
def test_an_extra_gate_that_cannot_fail_is_refused(tmp_path, bad):
    """A vacuous gate is worse than one that will not start: it reports green and is
    believed. `|| true` gets appended to quiet one flaky step and silences the rest.

    It must be judged ON ITS OWN, before composition — `base && (B || true)` can still
    fail through `base`, so the composed gate looks healthy while B's veto is worthless.
    """
    from spiral.conductor import Conductor

    cfg = Config()
    cfg.extra_gate = bad
    with pytest.raises(RuntimeError, match="can never fail"):
        Conductor(workspace=str(_repo(tmp_path)), cfg=cfg)


def test_a_real_extra_gate_still_composes(tmp_path):
    """The check earns its keep only if ordinary gates sail through it."""
    from spiral.conductor import Conductor

    cfg = Config()
    cfg.extra_gate = "python3 -c 'pass'"
    cond = Conductor(workspace=str(_repo(tmp_path)), cfg=cfg)
    assert cond.gate and cond.gate.rstrip().endswith("(python3 -c 'pass')")


def test_the_composed_gate_yields_no_phantom_missing_tools(tmp_path):
    """The rungs echo their own failure messages, and one contains
    `…does not parse; fix the syntax error above`. Splitting on `;` with a regex read
    that as commands named `this` and `fix`, and reported both as missing programs."""
    from spiral.conductor import Conductor
    from spiral.harness_check import missing_tools

    cond = Conductor(workspace=str(_repo(tmp_path)), cfg=Config())
    assert missing_tools(cond.gate) == []


def test_quoted_text_is_never_read_as_a_command():
    from spiral.harness_check import missing_tools

    assert missing_tools("echo 'boom; fix the thing above' && pytest -q") == []
    assert missing_tools('echo "a; b; c"') == []
    # but a genuinely absent program, outside quotes, is still found
    assert missing_tools("echo 'safe; text' && no-such-binary-abc123") == [
        "no-such-binary-abc123"]
