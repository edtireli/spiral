"""Is the proposed computation even well-posed? Checked before the loop commits.

A real failure, and the reason this module exists: an autonomous run proposed computing
``H^7(so(6), so(5); R)``. SO(6)/SO(5) is the 5-sphere. Degree 7 on a 5-manifold is zero
before a single differential is written. Fourteen of the fifteen cells in its proposed
table vanished by dimension alone, and two of its five "distinct test cases" were the
same space (Spin(6) ≅ SU(4) and Spin(5) ≅ Sp(4), so SO(6)/SO(5) = SU(4)/Sp(4) = S^5).

Strikingly, the loop had already *written down* the risk — "risks trivial results if
H^{D+1}(g,h;R) vanishes" — and proceeded anyway, because nothing forced it to act on its
own warning. That is the gap: not a missing insight, a missing five-line arithmetic check.

The general principle, which is the project's thesis applied one level earlier: **a
proposed check must be falsifiable and non-vacuous before it is worth running.**
``numeric_lab`` already refuses a certificate that prints a constant ``True``; this is
the same refusal at the level of the research question.

The framework is domain-agnostic — issues are produced by pluggable rules. One rule set
ships, covering the Lie-group/cohomology claims where this failed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Issue:
    """One reason a proposed computation is not worth running as stated."""
    kind: str              # vacuous | degenerate | duplicate | unparseable
    subject: str           # the claim or object it concerns
    detail: str            # why, concretely and checkably
    fatal: bool = True     # fatal issues must block; others are warnings

    def sentence(self) -> str:
        return f"[{self.kind}] {self.subject}: {self.detail}"


# ── dimensions of the classical compact Lie groups ───────────────────────────
def lie_dim(name: str) -> int | None:
    """dim of a classical compact Lie group from its name. None if unrecognised.

    Conventions: ``Sp(2n)`` and ``Sp(n)`` both appear in the literature; the symplectic
    group of rank n has dimension n(2n+1). ``Sp(4)`` in physics usually means rank 2
    (dimension 10), which is the reading used here."""
    s = re.sub(r"[\s{}\\$]", "", (name or "")).lower()
    s = s.replace("mathfrak", "").replace("mathrm", "")
    m = re.match(r"^(su|so|sp|u|spin|sl)\(?(\d+)\)?$", s)
    if not m:
        return None
    fam, n = m.group(1), int(m.group(2))
    if fam == "su":
        return n * n - 1
    if fam == "u":
        return n * n
    if fam in ("so", "spin"):
        return n * (n - 1) // 2
    if fam == "sp":
        # Sp(4) → rank 2 → dim 10 (the physics reading); odd n → treat as rank n
        rank = n // 2 if n % 2 == 0 else n
        return rank * (2 * rank + 1)
    if fam == "sl":
        return n * n - 1
    return None


_PRODUCT = re.compile(r"\s*(?:x|×|\\times)\s*", re.I)


def _sum_dims(spec: str) -> int | None:
    total = 0
    for part in _PRODUCT.split(spec or ""):
        part = part.strip()
        if not part:
            continue
        d = lie_dim(part)
        if d is None:
            return None
        total += d
    return total


def coset_dim(spec: str) -> int | None:
    """dim(G/H) for a coset written ``G/H`` (H may be a product). None if unrecognised."""
    text = re.sub(r"[\s{}$]", "", spec or "")
    if "/" not in text:
        return lie_dim(text)
    g, h = text.split("/", 1)
    dg, dh = _sum_dims(g), _sum_dims(h)
    if dg is None or dh is None:
        return None
    return dg - dh


# Low-dimensional exceptional isomorphisms — the reason two "different" cosets in a
# comparison set can silently be the same manifold.
_KNOWN_SPACES: dict[str, str] = {
    "so(6)/so(5)": "S^5", "spin(6)/spin(5)": "S^5", "su(4)/sp(4)": "S^5",
    "su(2)": "S^3", "so(4)/so(3)": "S^3", "so(3)/so(2)": "S^2",
    "so(5)/so(4)": "S^4", "sp(4)/sp(2)xsp(2)": "S^4",
    "su(3)/su(2)xu(1)": "CP^2", "su(2)/u(1)": "CP^1=S^2",
    "su(3)/su(2)": "S^5",
}


def canonical_space(spec: str) -> str:
    """A canonical name for a homogeneous space, resolving exceptional isomorphisms.
    Two specs sharing a canonical name are the same manifold."""
    key = re.sub(r"[\s{}$\\]", "", (spec or "")).lower().replace("×", "x")
    key = key.replace("mathrm", "").replace("mathfrak", "")
    if key in _KNOWN_SPACES:
        return _KNOWN_SPACES[key]
    d = coset_dim(key)
    return key if d is None else f"{key}(dim{d})"


# ── the shipped rule set: cohomology-degree claims ───────────────────────────
# H^{k}(g, h) / H^{k}(G/H) in any of the notations the loop actually emits
_COHO = re.compile(
    r"H\^?\{?\s*(?P<deg>\d+|D\s*\+\s*1)\s*\}?\s*"
    # the argument list contains its own parens — su(5), so(5) — so a naive [^)]
    # capture stops at the first inner ')' and the whole rule silently never fires
    r"\(\s*(?P<space>(?:[^;()]|\([^()]*\))+?)\s*(?:;[^)]*)?\)",
    re.I)


def _space_from_args(args: str) -> str:
    """``su(5),so(5)`` (relative Lie algebra pair) → ``su(5)/so(5)``."""
    text = re.sub(r"[\s{}$\\]", "", args or "").replace("mathfrak", "").replace("mathbb", "")
    parts = [p for p in text.split(",") if p]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else ""


def check_cohomology_claims(text: str, *, dims: list[int] | None = None) -> list[Issue]:
    """Flag cohomology computations that are zero (or degenerate) by dimension alone.

    ``dims`` supplies concrete values for a symbolic degree written ``D+1``."""
    issues: list[Issue] = []
    seen: set[tuple[str, int]] = set()
    for m in _COHO.finditer(text or ""):
        raw_deg = re.sub(r"\s", "", m.group("deg"))
        space = _space_from_args(m.group("space"))
        dim = coset_dim(space)
        if dim is None:
            continue
        degrees = ([d + 1 for d in (dims or [])] if raw_deg.lower() == "d+1"
                   else [int(raw_deg)])
        for k in degrees:
            if (space, k) in seen:
                continue
            seen.add((space, k))
            subject = f"H^{k}({space})"
            if k > dim:
                issues.append(Issue(
                    "vacuous", subject,
                    f"dim({space}) = {dim}; every cohomology group above degree {dim} "
                    f"is zero, so this computation returns 0 without being performed",
                    fatal=True))
            elif k == dim:
                issues.append(Issue(
                    "degenerate", subject,
                    f"degree {k} equals dim({space}); this is the top-degree volume "
                    "class, forced by Poincare duality rather than by structure",
                    fatal=False))
    return issues


def check_distinct_objects(objects: list[str]) -> list[Issue]:
    """Flag a comparison set whose members are not actually distinct."""
    issues: list[Issue] = []
    groups: dict[str, list[str]] = {}
    for obj in objects or []:
        if not str(obj).strip():
            continue
        groups.setdefault(canonical_space(str(obj)), []).append(str(obj).strip())
    for canon, members in groups.items():
        if len(members) > 1:
            issues.append(Issue(
                "duplicate", ", ".join(members),
                f"these are the same space ({canon}); a comparison set that repeats an "
                "object measures nothing by the repetition",
                fatal=True))
    return issues


_LIST_SPLIT = re.compile(r"[,;]| and ")


def extract_spaces(text: str) -> list[str]:
    """Coset-shaped tokens (``G/H``) mentioned in a proposal."""
    found: list[str] = []
    for tok in re.findall(r"[A-Za-z]+\(\d+\)(?:\s*[x×]\s*[A-Za-z]+\(\d+\))*"
                          r"\s*/\s*[A-Za-z]+\(\d+\)(?:\s*[x×]\s*[A-Za-z]+\(\d+\))*", text or ""):
        tok = re.sub(r"\s+", "", tok)
        if tok not in found:
            found.append(tok)
    return found


# Compact symmetric spaces G/H satisfy [m,m] ⊆ h, so the relative Chevalley-Eilenberg
# differential vanishes identically (É. Cartan) and the "computation" degenerates to
# counting H-invariants. Proposing CE as the *method* there signals the wrong tool.
_SYMMETRIC = {
    "su(n)/so(n)", "su(5)/so(5)", "su(3)/so(3)", "su(4)/sp(4)", "su(2n)/sp(2n)",
    "so(6)/so(5)", "so(5)/so(4)", "so(4)/so(3)", "so(3)/so(2)", "so(n+1)/so(n)",
    "su(3)/su(2)xu(1)", "su(n+1)/su(n)xu(1)", "sp(4)/u(2)", "so(6)/so(4)",
    "sp(4)/sp(2)xsp(2)", "so(10)/u(5)", "e6/f4",
}


def check_method_fit(text: str, spaces: list[str] | None = None) -> list[Issue]:
    """Flag a proposed method that is degenerate on the proposed objects."""
    body = (text or "").lower().replace("–", "-")
    if not re.search(r"chevalley\s*-?\s*eilenberg", body):
        return []
    hits = []
    for s in (spaces if spaces is not None else extract_spaces(text)):
        key = re.sub(r"[\s{}$\\]", "", s).lower().replace("×", "x")
        if key in _SYMMETRIC:
            hits.append(s)
    if not hits:
        return []
    return [Issue(
        "degenerate", ", ".join(hits),
        "these are compact symmetric spaces, so [m,m] is contained in h and the relative "
        "Chevalley-Eilenberg differential vanishes identically (Cartan); the computation "
        "reduces to counting H-invariants and the method adds nothing",
        fatal=False)]


def precheck(text: str, *, dims: list[int] | None = None,
             objects: list[str] | None = None) -> list[Issue]:
    """Run every shipped rule over a proposed research angle.

    Returns the issues; the caller decides whether a fatal issue blocks. Deterministic:
    same proposal in, same issues out, no model involved."""
    spaces = objects if objects is not None else extract_spaces(text)
    issues = list(check_cohomology_claims(text, dims=dims))
    issues += check_distinct_objects(spaces)
    issues += check_method_fit(text, spaces)
    return issues


def report(issues: list[Issue]) -> str:
    """Feedback text handed back to the model so the next proposal is better posed."""
    if not issues:
        return ""
    fatal = [i for i in issues if i.fatal]
    lines = ["The proposed checks are not well posed. Fix these before proposing again:"]
    for i in fatal + [i for i in issues if not i.fatal]:
        lines.append(f"  - {i.sentence()}")
    if fatal:
        lines.append("A computation whose answer is forced to zero by dimension proves "
                     "nothing; choose degrees k <= dim of the space, or choose spaces "
                     "whose dimension admits the degree you need.")
    return "\n".join(lines)
