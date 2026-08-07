"""A broken instrument must indict itself, not the code under test.

Both halves are grounded in real failures: a composed gate that read as shell arithmetic
(``((…))``) reported every healthy project as red, and a gate ending in ``|| true``
reported every project as green. In each case the model could see the problem, and in
each case saying so cost it an attempt.
"""
from spiral.harness_check import (
    Fault, adjudicate, claim_in, instrument_faults, missing_tools, report,
    vacuous_gate,
)


# ------------------------------------------------------------------ fault signatures
def test_missing_command_indicts_the_harness():
    faults = instrument_faults("zsh: command not found: pytest", code=127)
    assert faults and faults[0].kind == "missing-tool"
    assert "not installed" in faults[0].detail


def test_env_style_missing_interpreter_is_caught():
    faults = instrument_faults("env: 'node': No such file or directory", code=127)
    assert [f.kind for f in faults] == ["missing-tool"]
    assert "node" in faults[0].detail


def test_the_macos_sandbox_dylib_block_is_an_environment_fault():
    """Reads like a broken install; no edit the model makes can clear it."""
    out = "ImportError: library load disallowed by system policy: libopenblas.dylib"
    assert [f.kind for f in instrument_faults(out)] == ["sandbox"]


def test_the_arithmetic_gate_bug_is_recognised_as_a_parse_fault():
    faults = instrument_faults("zsh: bad math expression: operand expected", code=1)
    assert [f.kind for f in faults] == ["shell-parse"]


def test_resource_and_network_faults():
    assert instrument_faults("Killed\n")[0].kind == "resource"
    assert instrument_faults("No space left on device")[0].kind == "resource"
    assert instrument_faults("Could not resolve host: pypi.org")[0].kind == "network"


def test_unrunnable_exit_codes_count_even_without_a_message():
    for code in (124, 126, 127):
        assert instrument_faults("", code=code), f"exit {code} left unexplained"
    assert instrument_faults("", code=1) == []


def test_ordinary_red_is_not_a_harness_fault():
    """The detector earns its keep by staying quiet. Anything else hands out excuses."""
    for out in (
        "FAILED tests/test_math.py::test_add - assert 3 == 4",
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        "FileNotFoundError: [Errno 2] No such file or directory: 'fixtures/data.csv'",
        "ModuleNotFoundError: No module named 'myapp.utils'",
        "error[E0308]: mismatched types",
        "1 failing, 12 passing",
    ):
        assert instrument_faults(out, code=1) == [], f"false harness claim on: {out}"


def test_a_missing_project_module_is_the_project_not_the_harness():
    """No module named 'myapp' is a bug; No module named 'pytest' is a setup problem."""
    assert instrument_faults("ModuleNotFoundError: No module named 'myapp'") == []
    assert instrument_faults("ModuleNotFoundError: No module named 'pytest'")


def test_faults_carry_their_evidence():
    for f in instrument_faults("running checks\nzsh: command not found: cargo\n", 127):
        assert f.evidence.strip(), "a fault with no evidence is an accusation"


# ------------------------------------------------------------------ static PATH check
def test_missing_tools_finds_unresolvable_programs():
    out = missing_tools("npm run build && definitely-not-a-real-binary-xyz test")
    assert "definitely-not-a-real-binary-xyz" in out


def test_missing_tools_ignores_builtins_paths_and_assignments():
    assert missing_tools("cd src && ./scripts/build.sh && echo done") == []
    assert missing_tools("FOO=1 true; exit 0") == []
    assert missing_tools("python3 -c 'import sys'") == []


# ------------------------------------------------------------------ vacuous gates
def test_a_gate_that_swallows_its_exit_status_is_vacuous():
    why = vacuous_gate("pytest -q || true")
    assert why and "|| true" in why


def test_no_op_and_empty_gates_are_vacuous():
    assert vacuous_gate("") and "empty" in vacuous_gate("")
    assert vacuous_gate("true")
    assert vacuous_gate("exit 0")
    assert vacuous_gate("# nothing to do here\n") is not None


def test_unrestored_set_plus_e_is_vacuous():
    assert vacuous_gate("set +e\npytest -q")
    assert vacuous_gate("set +e\nfoo\nset -e\npytest -q") is None


def test_a_real_gate_is_not_flagged():
    for good in ("pytest -q", "npm test && npm run lint",
                 "cargo test --all && cargo clippy -- -D warnings",
                 "python3 -m pytest tests/ -q && python3 -m spiral.footguns ."):
        assert vacuous_gate(good) is None, f"false vacuity on: {good}"


def test_vacuity_check_is_one_directional():
    """`None` means 'not provably vacuous', never 'this gate is good' — proving a gate
    meaningful takes breaking the code, which is a different tool's job."""
    assert vacuous_gate("pytest tests/test_nothing_at_all.py") is None


# ------------------------------------------------------------------ the channel
def test_claim_must_be_opened_deliberately():
    assert claim_in("HARNESS_ERROR: pytest is not installed") == "pytest is not installed"
    assert claim_in("harness_error: the gate does not parse")
    # merely discussing one, deep in prose, does not open the channel
    body = "x" * 500 + "\nHARNESS_ERROR: sneaky"
    assert claim_in(body) is None
    assert claim_in("Here is the fix:\n<<<<<<< SEARCH\nfoo\n") is None


def test_a_true_claim_is_confirmed_and_costs_nothing():
    v = adjudicate("HARNESS_ERROR: pytest is missing",
                   "zsh: command not found: pytest", 127, "pytest -q")
    assert v.claimed and v.confirmed
    assert any(f.kind == "missing-tool" for f in v.faults)
    assert "not a verdict on the code" in report(v.faults)


def test_a_vacuous_gate_claim_is_confirmed_from_the_command_alone():
    v = adjudicate("HARNESS_ERROR: this gate cannot fail", "", 0, "pytest -q || true")
    assert v.confirmed and any(f.kind == "vacuous-gate" for f in v.faults)


def test_a_false_claim_is_refused_with_the_evidence():
    v = adjudicate("HARNESS_ERROR: the test runner is broken",
                   "FAILED test_add - assert 3 == 4", 1, "pytest -q")
    assert v.claimed and not v.confirmed
    assert "does not support it" in v.reason
    assert "in the code under test" in v.reason
    assert v.claim_text                       # what it said is kept, for the record


def test_a_reply_without_the_claim_is_left_alone():
    v = adjudicate("<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE", "boom", 1, "x")
    assert not v.claimed and not v.confirmed


def test_report_is_empty_with_nothing_to_report():
    assert report([]) == ""


def test_adjudication_never_raises_on_junk():
    for args in (("", "", None, ""), ("HARNESS_ERROR:", None, 0, None),
                 ("HARNESS_ERROR: x", "\x00\x1b[0m", -9, "((")):
        adjudicate(*args)
