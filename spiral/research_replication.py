"""Blind-replication helpers for independently regenerated certificates."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path


PRIVATE_SOLUTION_KEYS = {
    "basis", "code", "cmd", "command", "files", "manifest", "proof", "repos",
    "rhs", "steps", "value",
}


def method_family(claim: dict, backend: str = "") -> str:
    kind = str(claim.get("kind") or "").lower()
    if kind == "theorem" or backend == "lean":
        return "formal lean proof"
    if kind in {"identity", "solution", "groebner", "ideal_membership"}:
        return "exact symbolic algebra"
    if kind == "numeric":
        return "numerical experiment"
    if kind in {"workbench", "certificate", "code_certificate"}:
        files = claim.get("files") or {}
        suffixes = {Path(str(name)).suffix.lower() for name in files}
        content = "\n".join(str(value).lower() for value in files.values())
        if ".lean" in suffixes:
            return "formal lean proof"
        if ".singular" in suffixes:
            return "singular computer algebra"
        if suffixes & {".cpp", ".cc", ".cxx", ".c"}:
            return "compiled numerical or symbolic program"
        if "groebner" in content or "resultant(" in content or "std(" in content:
            return "exact groebner computation"
        if "mpmath" in content or "solve_ivp" in content or "allclose" in content:
            return "independent numerical computation"
        if "sympy" in content or "simplify(" in content:
            return "executable symbolic algebra"
        return "executable python certificate"
    explicit = str(claim.get("method_family") or "").strip().lower()
    if explicit:
        return re.sub(r"\s+", " ", explicit)[:120]
    return backend or kind or "unspecified method"


def blind_brief(claim: dict, *, question: str, conventions: str = "") -> dict:
    """Expose the claim and falsification contract, never its original solution."""

    statement = str(claim.get("statement") or claim.get("note") or "").strip()
    public = {
        "question": question,
        "statement": statement,
        "assumptions": [str(x) for x in (claim.get("assumptions") or [])],
        "conventions": claim.get("conventions") or conventions,
        "falsifier": claim.get("falsifier") or "produce a nonzero residual or counterexample",
        "acceptance_criteria": claim.get("acceptance_criteria") or (
            (claim.get("validation") or {}).get("acceptance_criteria") or []),
        "original_method_family": method_family(claim),
    }
    if claim.get("datasets"):
        # Dataset identity and the preregistered estimand are part of the proposition,
        # not the hidden solution. A replica needs the same frozen data contract while
        # remaining blind to original code, estimates, diagnostics, and output.
        public["datasets"] = claim.get("datasets")
        public["analysis_plan"] = claim.get("analysis_plan") or {}
        public["alignment"] = claim.get("alignment") or {}
    # Exact identities/equations need their statement fields to be meaningful.  The
    # right-hand side is hidden only when a natural-language statement already supplies
    # the target; otherwise expose the proposition, not the prior derivation.
    kind = str(claim.get("kind") or "").lower()
    if kind in {"identity", "solution", "groebner", "ideal_membership", "theorem"}:
        proposition = {}
        for key, value in claim.items():
            if key.startswith("_") or key in {
                    "code", "cmd", "command", "files", "manifest", "proof", "repos", "steps"}:
                continue
            if key in {"kind", "lhs", "rhs", "equation", "expr", "generators",
                       "variables", "basis", "order", "statement", "var", "value"}:
                proposition[key] = value
        public["proposition"] = proposition
    public["brief_sha256"] = hashlib.sha256(
        json.dumps(public, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return public


def independent_enough(original_claim: dict, original_backend: str,
                       replica_claim: dict, replica_backend: str,
                       planner_model: str, replica_model: str) -> dict:
    original_method = method_family(original_claim, original_backend)
    replica_method = method_family(replica_claim, replica_backend)
    different_method = original_method != replica_method
    different_backend = bool(original_backend and replica_backend and original_backend != replica_backend)
    different_model = bool(planner_model and replica_model and planner_model != replica_model)
    solution_material = {
        key: replica_claim.get(key) for key in (
            "files", "proof", "code", "basis", "steps", "cmd", "command")
        if replica_claim.get(key)
    }
    replica_code_hash = hashlib.sha256(
        json.dumps(solution_material, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest() if solution_material else ""
    return {
        "original_method_family": original_method,
        "replica_method_family": replica_method,
        "different_method_family": different_method,
        "different_verifier_backend": different_backend,
        "different_generation_model": different_model,
        "replica_code_sha256": replica_code_hash,
        "independent": bool(
            different_model and (different_method or different_backend) and replica_code_hash),
    }


def inspect_replica_methods(claim: dict, manifest_path: str = "") -> dict:
    """Authenticate that a workbench replica uses distinct observable techniques."""

    files = {str(name): str(content) for name, content in (claim.get("files") or {}).items()}
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    step_rows = manifest.get("steps_run") or []
    validation = manifest.get("validation_evidence") or {}
    methods = [row for row in (validation.get("methods") or []) if row.get("valid")]

    def signatures(text: str, argv: list[str]) -> set[str]:
        blob = f"{' '.join(argv)}\n{text}".lower()
        found = set()
        rules = {
            "formal_proof": (" lean", "by ring", "by simp", "theorem ", "lemma "),
            "singular_groebner": ("singular", "groebner", "std("),
            "exact_symbolic": ("sympy", "simplify(", "factor(", "expand(", "resultant("),
            "high_precision_numeric": ("mpmath", "mp.dps", "arb", "decimal("),
            "floating_numeric": ("numpy", "scipy", "allclose(", "isclose(", "solve_ivp"),
            "compiled_direct": ("clang", "g++", "c++", "rustc", "go run", "swiftc", "javac"),
            "case_enumeration": ("for ", "while ", "itertools", "cases :=", "foreach"),
            "interval_or_bound": ("interval", "error bound", "upper bound", "lower bound"),
        }
        for label, cues in rules.items():
            if any(cue in blob for cue in cues):
                found.add(label)
        return found or {"unclassified"}

    observed = []
    for method in methods:
        step = int(method.get("step", -1))
        row = step_rows[step] if 0 <= step < len(step_rows) else {}
        argv = [str(part) for part in (row.get("argv") or [])]
        referenced = "\n".join(
            content for name, content in files.items()
            if name in " ".join(argv) or not argv
        )
        observed.append({
            "name": method.get("name", ""), "step": step, "argv": argv,
            "signatures": sorted(signatures(referenced, argv)),
        })
    signature_sets = {tuple(row["signatures"]) for row in observed}
    all_source = "\n".join(files.values()).lower()
    adversarial = (
        any(cue in all_source for cue in (
            "counterexample", "boundary", "edge case", "random", "sample",
            "residual", "falsif", "parameter sweep"))
        and any(cue in all_source for cue in ("assert", "raise", "panic", "stopifnot"))
    )
    return {
        "observed_methods": observed,
        "distinct_signature_count": len(signature_sets),
        "method_diversity": len(observed) >= 2 and len(signature_sets) >= 2,
        "adversarial_falsifier_check": adversarial,
    }


def write_report(root: str | Path, claim_id: str, report: dict) -> Path:
    directory = Path(root) / "replications"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{claim_id}.json"
    value = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **report,
    }
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    lines = [
        "# Blind replication", "",
        f"Claim: `{claim_id}`", "",
        f"Status: **{value.get('status', 'unknown')}**", "",
        f"Passed: **{str(bool(value.get('passed'))).lower()}**", "",
        f"Original method: {value.get('independence', {}).get('original_method_family', '')}",
        f"Replica method: {value.get('independence', {}).get('replica_method_family', '')}",
        "",
        "The replication model received the proposition, assumptions, conventions, "
        "falsifier, and acceptance criteria, but not the original proof, code, command, "
        "output, or certificate.", "",
    ]
    (directory / f"{claim_id}.md").write_text("\n".join(lines), encoding="utf-8")
    return path
