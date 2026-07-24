"""Deterministic mathematics verification — the load-bearing wall of `spiral research`.

The research loop's LLM *proposes* derivations, identities, solutions and conjectures.
Nothing it proposes is believed until a deterministic tool checks it — this module is
that tool. It is the research analogue of spiral's build gate: verify, or it didn't
happen. An LLM being fluent about gauge anomalies is not evidence the algebra is right;
`simplify(lhs - rhs) == 0` is.

Backends, strongest first, auto-detected (never required):

* **lean**       — formal proof; a machine-checked ``theorem`` is the gold standard.
* **wolfram / sage / maxima** — full computer-algebra systems, if installed.
* **sympy**      — always available (a hard dep); covers identities, solutions,
                   simplification, limits, series, matrices/commutators.
* **numeric**    — a Monte-Carlo cross-check that rides *every* symbolic claim: sample
                   the free symbols at many random points and compare both sides. This
                   catches a false "identity" symbolic simplification fails to close,
                   and never certifies a false one — its job is to *refute* cheaply.

Only ``sympy``/``numeric`` run on a machine with nothing else installed; the CAS/Lean
backends light up automatically when their binary appears, exactly like spiral's
``detect_gate``. Absence degrades the strength of a check, never blocks it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Verdict:
    """The result of a verification. ``ok`` is the machine's answer, not the model's."""
    ok: bool
    backend: str
    kind: str
    detail: str = ""
    refuted_at: dict[str, float] | None = None      # a counterexample, when found
    extra: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


# ── backend detection (auto, never required) ─────────────────────────────────
_CAS_BINARIES = {
    "lean": "lean",
    "wolfram": "wolframscript",
    "sage": "sage",
    "maxima": "maxima",
    "pari": "gp",
}
_LEAN_HEALTHY: bool | None = None


def detect_backends() -> list[str]:
    """Verification backends available on this machine, strongest first. ``sympy`` and
    ``numeric`` are always present (pure Python); the rest appear iff their binary does."""
    have = []
    if lean_available():                              # PATH/elan and a responsive toolchain
        have.append("lean")
    have += [b for b in ("wolfram", "sage", "maxima", "pari") if shutil.which(_CAS_BINARIES[b])]
    return have + ["sympy", "numeric"]


# ── sympy helpers ────────────────────────────────────────────────────────────
def _sympify(expr: str, local_dict: dict | None = None):
    """Parse a claim expression. Python/`sympy` syntax; LaTeX is accepted when the
    optional parser (antlr) is installed, else the caller gets a clear error."""
    import sympy as sp
    s = str(expr).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if "\\" in s or "^{" in s:                       # looks like LaTeX
        try:
            from sympy.parsing.latex import parse_latex
            return parse_latex(s)
        except Exception:
            pass                                     # fall through to sympify
    try:
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        transformations = standard_transformations + (convert_xor, implicit_multiplication_application)
        return parse_expr(s, local_dict=local_dict, transformations=transformations)
    except Exception:
        return sp.sympify(s, locals=local_dict)


def _poly_context(variables: list[str] | tuple[str, ...]):
    """Create an ordered exact polynomial context for certificate checks."""
    import sympy as sp
    names = [str(v).strip() for v in variables or [] if str(v).strip()]
    if not names:
        raise ValueError("polynomial certificate needs explicit variables")
    syms = tuple(sp.symbols(" ".join(names), seq=True))
    local = {n: s for n, s in zip(names, syms)}
    local.update({"I": sp.I, "pi": sp.pi})
    return syms, local


def _expr_list(items: list, local: dict | None = None) -> list:
    return [_sympify(str(x), local) for x in (items or [])]


def _is_zero_expr(expr) -> bool:
    import sympy as sp
    return sp.simplify(expr) == 0


def _normal_basis(gens: list, variables: list[str], order: str = "lex") -> list:
    """Reduced Groebner basis expressions, used as a canonical certificate form."""
    import sympy as sp
    syms, local = _poly_context(variables)
    exprs = [sp.expand(e) for e in _expr_list(gens, local) if not _is_zero_expr(e)]
    if not exprs:
        return []
    G = sp.groebner(exprs, *syms, order=order)
    out = []
    for p in G.polys:
        poly = sp.Poly(p.as_expr(), *syms)
        try:
            expr = poly.monic().as_expr()
        except Exception:
            lc = poly.LC()
            expr = sp.cancel(poly.as_expr() / lc) if lc else poly.as_expr()
        out.append(sp.factor(expr))
    return out


def _basis_equal(a: list, b: list) -> bool:
    return len(a) == len(b) and all(_is_zero_expr(x - y) for x, y in zip(a, b))


def _reduce_mod(gens: list, expr, variables: list[str], order: str = "lex"):
    import sympy as sp
    syms, local = _poly_context(variables)
    ideal = [sp.expand(g) for g in _expr_list(gens, local) if not _is_zero_expr(g)]
    e = sp.expand(expr if hasattr(expr, "free_symbols") else _sympify(str(expr), local))
    if not ideal:
        return sp.simplify(e)
    G = sp.groebner(ideal, *syms, order=order)
    return sp.factor(G.reduce(e)[1])


def _matrix(data: list, local: dict | None = None):
    import sympy as sp
    return sp.Matrix([[_sympify(str(x), local) for x in row] for row in data])


def _free_symbols(*exprs):
    out: set = set()
    for e in exprs:
        out |= set(getattr(e, "free_symbols", set()))
    return sorted(out, key=lambda s: s.name)


def numeric_identity(lhs, rhs, *, trials: int = 40, tol: float = 1e-9,
                     lo: float = -3.0, hi: float = 3.0) -> Verdict:
    """Refute (or fail to refute) ``lhs == rhs`` by sampling the free symbols.

    A true identity holds at every point, so any sampled point where the two sides
    differ by more than ``tol`` is a hard counterexample — a refutation, not an
    opinion. Surviving many random points is strong (not conclusive) support. Complex
    offsets are used so branch cuts and singularities don't masquerade as differences.
    Deterministic: the sampler is seeded, so a verdict reproduces exactly.
    """
    import random as _rnd

    import sympy as sp
    diff = sp.simplify(lhs - rhs)
    syms = _free_symbols(diff)
    rng = _rnd.Random(0xC0FFEE)

    def _sample(s):
        # honour the symbol's own assumptions so a "for positive reals" identity
        # (log(xy)=log x+log y) isn't spuriously refuted by a complex branch cut,
        # while a fully general symbol is still probed off the real axis.
        if s.is_positive:
            return sp.Float(rng.uniform(max(0.05, 0.05), hi))
        if s.is_real:
            return sp.Float(rng.uniform(lo, hi))
        return sp.Float(rng.uniform(lo, hi)) + sp.Float(rng.uniform(lo, hi)) * sp.I

    checked = 0
    for _ in range(trials):
        subs = {s: _sample(s) for s in syms}
        try:
            val = complex(diff.subs(subs).evalf())
        except Exception:
            continue                                 # undefined here (pole/branch) — skip
        checked += 1
        if abs(val) > tol:
            return Verdict(False, "numeric", "identity",
                           detail=f"sides differ by {abs(val):.2e} at a sampled point",
                           refuted_at={s.name: float(sp.re(v)) for s, v in subs.items()})
    if checked == 0:
        return Verdict(False, "numeric", "identity",
                       detail="could not evaluate at any sampled point")
    return Verdict(True, "numeric", "identity",
                   detail=f"held at {checked} random points (tol {tol:g})")


# ── the public check kinds ───────────────────────────────────────────────────
def verify_identity(lhs: str, rhs: str, *, assume: dict | None = None) -> Verdict:
    """Is ``lhs == rhs`` an identity? Tries symbolic simplification first (a proof when
    it closes), then always cross-checks numerically (a refutation when it fails)."""
    import sympy as sp
    L, R = _sympify(lhs), _sympify(rhs)
    if assume:
        subs = {sp.Symbol(k): sp.Symbol(k, **v) for k, v in assume.items()}
        L, R = L.subs(subs), R.subs(subs)
    num = numeric_identity(L, R)
    if not num.ok:                                   # a numeric counterexample is decisive
        return num
    d = sp.simplify(L - R)
    if d == 0:
        return Verdict(True, "sympy", "identity", detail="simplify(lhs - rhs) = 0")
    if sp.simplify(sp.nsimplify(d)) == 0:
        return Verdict(True, "sympy", "identity", detail="simplifies to 0 after nsimplify")
    # Sampling can refute an identity, but surviving samples is never a proof.  Report an
    # explicit inconclusive failure so the research loop cannot count it as established.
    return Verdict(False, "numeric", "identity",
                   detail=f"not refuted, but unproven symbolically; {num.detail}",
                   extra={"symbolic_residual": str(d), "inconclusive": True})


def verify_zero(expr: str) -> Verdict:
    """Is ``expr`` identically zero?"""
    return verify_identity(expr, "0")


def verify_solution(equation: str, var: str, value: str) -> Verdict:
    """Does ``var = value`` satisfy ``equation`` (``lhs = rhs`` or ``expr`` meaning
    ``expr = 0``)? Substitutes and checks — no trust in how the value was obtained."""
    import sympy as sp
    eqs = equation.split("=")
    expr = _sympify(eqs[0]) - _sympify(eqs[1]) if len(eqs) == 2 else _sympify(equation)
    res = expr.subs(sp.Symbol(var), _sympify(value))
    if sp.simplify(res) == 0:
        return Verdict(True, "sympy", "solution", detail=f"{var}={value} satisfies the equation")
    val = complex(res.evalf()) if not res.free_symbols else None
    return Verdict(False, "sympy", "solution",
                   detail=f"substitution gives {sp.simplify(res)}"
                          + (f" ≈ {val:.3g}" if val is not None else ""))


def verify_equal_numeric(lhs: str, rhs: str, *, tol: float = 1e-9) -> Verdict:
    """Constant expressions equal to numerical tolerance (e.g. ζ(2) == π²/6)."""
    import sympy as sp
    d = complex((_sympify(lhs) - _sympify(rhs)).evalf())
    ok = abs(d) <= tol
    return Verdict(ok, "sympy", "numeric_equal",
                   detail=f"|lhs - rhs| = {abs(d):.2e} {'≤' if ok else '>'} {tol:g}")


def verify_groebner(generators: list, variables: list[str], basis: list | None = None,
                    *, order: str = "lex") -> Verdict:
    """Check a reduced Gröbner-basis certificate for a polynomial ideal."""
    import sympy as sp

    got = _normal_basis(generators, variables, order=order)
    if basis is None:
        return Verdict(True, "sympy", "groebner",
                       detail=f"computed reduced Groebner basis with {len(got)} generators",
                       extra={"basis": [str(g) for g in got], "order": order})
    syms, local = _poly_context(variables)
    want = [sp.factor(e) for e in _expr_list(basis, local)]
    ok = _basis_equal(got, want)
    return Verdict(ok, "sympy", "groebner",
                   detail=f"reduced Groebner basis {'matches' if ok else 'does not match'} certificate",
                   extra={"computed": [str(g) for g in got], "expected": [str(g) for g in want],
                          "order": order})


def verify_ideal_membership(expr: str, generators: list, variables: list[str],
                            *, order: str = "lex") -> Verdict:
    """Check whether ``expr`` reduces to zero modulo the ideal generated by ``generators``."""
    _, local = _poly_context(variables)
    rem = _reduce_mod(generators, _sympify(expr, local), variables, order=order)
    ok = _is_zero_expr(rem)
    return Verdict(ok, "sympy", "ideal_membership",
                   detail=f"normal-form remainder is {rem}",
                   extra={"remainder": str(rem), "order": order})


# ── Lean formal-proof backend (gold standard, when a claim is a theorem) ──────
def _lean_exe() -> str | None:
    """Locate a responsive ``lean`` binary.

    Prefer installed elan toolchain binaries over the elan shim. The shim can hang
    while resolving ``stable`` or touching the network, even when an installed
    toolchain works perfectly.
    """
    import os
    root = Path(os.path.expanduser("~/.elan/toolchains"))
    if root.is_dir():
        for cand in sorted(root.glob("*/bin/lean"), reverse=True):
            if cand.is_file():
                return str(cand)
    exe = shutil.which("lean")
    if exe:
        return exe
    cand = Path(os.path.expanduser("~/.elan/bin/lean"))
    return str(cand) if cand.is_file() else None


def lean_available(*, timeout: float = 5.0) -> bool:
    """True only when Lean is installed *and* responds quickly.

    An elan shim can exist while the toolchain is downloading, wedged, or blocked on
    setup. In that state Lean must not be advertised as a verifier, otherwise every
    theorem claim can burn the full proof timeout before falling back.
    """
    global _LEAN_HEALTHY
    if _LEAN_HEALTHY is not None:
        return _LEAN_HEALTHY
    exe = _lean_exe()
    if exe is None:
        _LEAN_HEALTHY = False
        return False
    import subprocess
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=timeout)
        _LEAN_HEALTHY = p.returncode == 0 and bool((p.stdout + p.stderr).strip())
    except Exception:
        _LEAN_HEALTHY = False
    return _LEAN_HEALTHY


def prove_lean(statement: str, proof: str = "", *, imports: str = "",
               timeout: float = 30.0) -> Verdict:
    """Machine-check a Lean theorem. ``statement`` is the theorem signature (e.g.
    ``(n : Nat) : n + 0 = n``); ``proof`` is the tactic/term body (e.g. ``by simp``).

    A theorem the Lean kernel accepts is *proven* — the strongest verdict this module
    can return. Absence of Lean (or a ``sorry`` in the proof) is an honest "not proven",
    never a crash; the loop then falls back to sympy/numeric. Bare Lean handles
    ``rfl``/``decide``/``simp`` goals; installing mathlib (``import Mathlib``) unlocks the
    rest with no code change here."""
    exe = _lean_exe()
    if exe is None:
        return Verdict(False, "lean", "theorem", detail="lean not installed")
    if not lean_available():
        return Verdict(False, "lean", "theorem", detail="lean installed but failed its health check")
    body = (proof or "by rfl").strip()
    if "sorry" in body:
        return Verdict(False, "lean", "theorem", detail="proof contains `sorry` (not a proof)")
    src = (f"{imports}\n" if imports else "") + f"theorem _spiral_claim {statement} := {body}\n"
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "Claim.lean"
        f.write_text(src)
        try:
            p = subprocess.run([exe, str(f)], cwd=td, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Verdict(False, "lean", "theorem", detail=f"lean timed out after {timeout:g}s")
        err = (p.stdout + p.stderr).strip()
        if p.returncode == 0 and "error" not in err.lower():
            return Verdict(True, "lean", "theorem", detail="Lean kernel accepted the proof")
        return Verdict(False, "lean", "theorem", detail=(err[:300] or "Lean rejected the proof"))


_KINDS = {
    "identity": lambda c: verify_identity(c["lhs"], c["rhs"], assume=c.get("assume")),
    "zero": lambda c: verify_zero(c["expr"]),
    "solution": lambda c: verify_solution(c["equation"], c["var"], c["value"]),
    "numeric_equal": lambda c: verify_equal_numeric(c["lhs"], c["rhs"], tol=c.get("tol", 1e-9)),
    "groebner": lambda c: verify_groebner(c["generators"], c["variables"],
                                          c.get("basis"), order=c.get("order", "lex")),
    "ideal_membership": lambda c: verify_ideal_membership(c["expr"], c["generators"],
                                                          c["variables"], order=c.get("order", "lex")),
    "theorem": lambda c: prove_lean(c["statement"], c.get("proof", ""),
                                    imports=c.get("imports", "")),
}


def verify(claim: dict) -> Verdict:
    """Route a claim to the right check. ``claim`` is a small dict the research loop
    emits, e.g. ``{"kind": "identity", "lhs": "sin(x)**2+cos(x)**2", "rhs": "1"}``.
    A bad claim shape returns a failed Verdict rather than raising — a malformed claim
    is an unverified claim, never a crash in the loop."""
    kind = str(claim.get("kind", "")).lower()
    fn = _KINDS.get(kind)
    if fn is None:
        return Verdict(False, "none", kind or "unknown",
                       detail=f"unknown claim kind {kind!r}; known: {sorted(_KINDS)}")
    try:
        return fn(claim)
    except KeyError as e:
        return Verdict(False, "none", kind, detail=f"claim missing field {e}")
    except Exception as e:                            # a broken claim must not kill the loop
        return Verdict(False, "sympy", kind, detail=f"{type(e).__name__}: {e}")
