"""The maths verifier is the research loop's build gate: it must certify true claims,
refute false ones (especially plausible-looking LLM algebra slips), and never crash on
a malformed claim. Runs standalone (`python tests/test_verify_math.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.verify_math import detect_backends, verify, verify_identity  # noqa: E402


def test_true_identities_certified():
    for lhs, rhs in [
        ("sin(x)**2 + cos(x)**2", "1"),
        ("(x+1)**2", "x**2 + 2*x + 1"),
        ("(x+1)^2", "x^2 + 2x + 1"),
        ("exp(I*x)", "cos(x) + I*sin(x)"),
        ("sin(2*x)", "2*sin(x)*cos(x)"),
        ("diff(sin(x), x)", "cos(x)"),
        ("integrate(x**2, x)", "x**3/3"),
    ]:
        assert verify_identity(lhs, rhs).ok, f"should certify {lhs} == {rhs}"


def test_false_identities_refuted_with_counterexample():
    v = verify_identity("(x+1)**2", "x**2 + 3*x + 1")     # a classic expansion slip
    assert not v.ok and v.refuted_at is not None           # refuted, with a witness point


def test_conditional_identity_needs_its_assumption():
    # log(xy)=log x+log y holds only for positive reals — the verifier must NOT
    # certify it generally (complex branch cut) but MUST under the right assumption.
    assert not verify_identity("log(x*y)", "log(x)+log(y)").ok
    assert verify_identity("log(x*y)", "log(x)+log(y)",
                           assume={"x": {"positive": True}, "y": {"positive": True}}).ok


def test_solution_checking():
    assert verify({"kind": "solution", "equation": "x**2 - 5*x + 6 = 0",
                   "var": "x", "value": "3"}).ok
    assert not verify({"kind": "solution", "equation": "x**2 - 5*x + 6 = 0",
                       "var": "x", "value": "4"}).ok


def test_numeric_constant_equality():
    assert verify({"kind": "numeric_equal", "lhs": "pi**2/6",
                   "rhs": "Sum(1/n**2, (n, 1, oo)).doit()"}).ok     # Basel problem


def test_groebner_certificate_and_ideal_membership():
    ideal = {"generators": ["x^2 - y", "x*y - 1"], "variables": ["x", "y"], "order": "lex"}
    g = verify({"kind": "groebner", **ideal, "basis": ["x - y^2", "y^3 - 1"]})
    assert g.ok and g.extra["computed"] == ["x - y**2", "(y - 1)*(y**2 + y + 1)"]
    assert not verify({"kind": "groebner", **ideal, "basis": ["x - y", "y - 1"]}).ok

    assert verify({"kind": "ideal_membership", **ideal, "expr": "x^3 - 1"}).ok
    missing = verify({"kind": "ideal_membership", **ideal, "expr": "x - 1"})
    assert not missing.ok and missing.extra["remainder"] == "(y - 1)*(y + 1)"


def test_malformed_claim_never_raises():
    assert not verify({"kind": "woozle", "expr": "x"}).ok           # unknown kind
    assert not verify({"kind": "identity", "lhs": "x"}).ok          # missing 'rhs'
    assert not verify({"kind": "identity", "lhs": "))(", "rhs": "1"}).ok  # unparseable


def test_backends_always_include_sympy_and_numeric():
    b = detect_backends()
    assert b[-2:] == ["sympy", "numeric"]                  # always-present tail


def test_lean_backend_when_available():
    # Lean is a pluggable backend: exercised when present, skipped (not failed) when not.
    import os
    import shutil
    from pathlib import Path

    from spiral.verify_math import lean_available, prove_lean, verify
    if not (shutil.which("lean") or Path(os.path.expanduser("~/.elan/bin/lean")).is_file()):
        return
    if not lean_available():
        return
    assert prove_lean(": (2:Nat)+2 = 4", "by decide").ok           # kernel-proven
    assert not prove_lean(": (2:Nat)+2 = 5", "by decide").ok        # kernel-rejected
    v = verify({"kind": "theorem", "statement": ": (1:Nat)+1=2", "proof": "by rfl"})
    assert v.ok and v.backend == "lean"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all verify_math tests passed  ·  backends:", detect_backends())
