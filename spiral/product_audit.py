"""Deterministic finish checks that prevent Builder from shipping a scaffold."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from spiral.planner import _is_product_build, product_profile


_SOURCE_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".html", ".css",
    ".java", ".kt", ".swift", ".go", ".rs", ".c", ".cpp", ".cs", ".lean",
    ".v", ".agda", ".thy", ".tf", ".hcl", ".yaml", ".yml", ".sh", ".sql",
    ".nim", ".hs", ".lhs", ".ml", ".mli", ".ex", ".exs", ".erl", ".hrl",
    ".clj", ".cljs", ".fs", ".fsx", ".vb", ".pl", ".pm", ".cr", ".groovy",
}
_PLACEHOLDER = re.compile(
    r"\b(?:TODO|FIXME|NotImplementedError|coming soon|lorem ipsum|mock data|dummy data)\b"
    r"|\b(?:todo!|unimplemented!)\s*\(|\bTODO\s*\(|"
    r"(?:throw\s+new\s+Error|fatalError)\s*\([^\n]{0,80}not implemented",
    re.I,
)
_DELIVERABLE_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".tiff"},
    "video": {".mp4", ".mov", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"},
    "document": {".pdf", ".docx", ".odt", ".tex", ".md", ".rst", ".html", ".rtf"},
    "presentation": {".pptx", ".odp", ".pdf"},
    "dataset": {
        ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".arrow",
        ".sqlite", ".sqlite3", ".db", ".xlsx", ".ods",
    },
    "notebook": {".ipynb"},
    "3d": {".glb", ".gltf", ".obj", ".fbx", ".blend", ".stl", ".ply"},
    "plot": {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf", ".html"},
    "formal-proof": {".lean", ".v", ".agda", ".thy"},
}
_CODE_KINDS = {
    "web", "android", "ios", "desktop", "cli", "service", "library",
    "simulation", "plot", "notebook", "game", "firmware", "infrastructure",
    "formal-proof", "other",
}


def _production_files(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_EXT:
            continue
        rel = path.relative_to(root)
        low_parts = {part.lower() for part in rel.parts}
        if any(part.startswith(".") for part in rel.parts):
            continue
        if low_parts & {"node_modules", "dist", "build", "vendor", "tests", "test",
                        "fixtures", "examples", "generated", "coverage"}:
            continue
        if re.search(r"(?:^|[._-])(?:test|spec)(?:[._-]|$)", path.name, re.I):
            continue
        out.append(path)
    return out


def _test_files(root: Path) -> list[Path]:
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        rel = str(path.relative_to(root)).lower()
        if ("/tests/" in f"/{rel}" or "/test/" in f"/{rel}"
                or re.search(r"(?:^|[._-])(?:test|spec)(?:[._-])", path.name, re.I)):
            rows.append(path)
    return rows


def _issue(identifier: str, severity: str, evidence: str, fix: str,
           files: list[str] | None = None) -> dict:
    return {
        "id": identifier,
        "severity": severity,
        "evidence": evidence,
        "fix": fix,
        "files": files or [],
    }


def _artifact_candidates(root: Path, kind: str) -> list[Path]:
    extensions = _DELIVERABLE_EXTENSIONS.get(kind, set())
    if not extensions:
        return []
    rows = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(root)
        if any(part.startswith(".") or part.lower() in {
                "node_modules", "target", "coverage",
                "fixtures", "examples", "test", "tests",
        } for part in rel.parts):
            continue
        if kind == "document" and path.name.lower().startswith(("readme", "changelog")):
            continue
        rows.append(path)
    return rows


def audit_product(workspace: str | Path, goal: str, project_kind: str = "other") -> dict:
    """Return fail-only evidence; a clean result is not a proof of product quality."""

    workspace_root = Path(workspace).resolve()
    from spiral.builder_tools import runnable_project_roots

    roots = runnable_project_roots(workspace_root)
    root = workspace_root
    profile = product_profile(goal, project_kind)
    if not _is_product_build(goal, project_kind):
        return {"applicable": False, "profile": profile, "issues": []}

    manifest_path = workspace_root / ".spiral" / "artifacts.json"
    try:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        raw_goal = str(goal).split(
            "\n\nDESIGN SPECIFICATION (implement these decisions literally):", 1
        )[0].split("\n\nEMPIRICAL LOCAL TOOL PROFILE", 1)[0].strip()
        if manifest.get("goal_sha256") != hashlib.sha256(
                raw_goal.encode("utf-8")).hexdigest():
            manifest = {}
    except Exception:
        manifest = {}
    deliverables = [
        row for row in (manifest.get("deliverables") or [])
        if isinstance(row, dict)
    ] or [{"id": "D1", "kind": project_kind, "description": goal}]

    source_files = _production_files(root)
    issues: list[dict] = []
    code_required = any(
        str(row.get("kind") or "other") in _CODE_KINDS for row in deliverables)
    if code_required and not source_files:
        issues.append(_issue(
            "product-implementation", "major",
            "No production source files were found for the requested product.",
            "Implement the actual runnable product, not only metadata, prose, or generated output.",
        ))
    artifact_counts: dict[str, int] = {}
    for row in deliverables:
        kind = str(row.get("kind") or "other")
        if kind not in _DELIVERABLE_EXTENSIONS:
            continue
        patterns = [
            str(item) for item in (row.get("output_globs") or [])
            if str(item).strip()
        ]
        if not patterns:
            patterns = ["output/*"]
        rejected = []
        from spiral.delivery import _declared_artifact_files, _output_evidence

        candidates, rejected = _declared_artifact_files(
            workspace_root, kind, patterns)
        identifier = str(row.get("id") or kind)
        artifact_counts[identifier] = len(candidates)
        if rejected:
            issues.append(_issue(
                f"deliverable-{identifier}-path", "major",
                "Unsafe or non-exact output declarations were rejected: "
                + ", ".join(rejected),
                "Use relative workspace-contained globs that select final exported "
                "artifacts, never parent paths or workspace-wide catchalls.",
            ))
        if not candidates:
            issues.append(_issue(
                f"deliverable-{identifier}", "major",
                (
                    f"No exact declared {kind} output was found for deliverable {identifier}: "
                    if patterns else
                    f"No inspectable {kind} artifact was found for deliverable {identifier}: "
                )
                + f"{row.get('description', '')}",
                f"Create, render, and retain the final {kind} artifact in the workspace; "
                "validate that it opens/decodes and is the requested output rather than an input asset.",
            ))
        else:
            decode_errors = []
            decode_cache = {}
            for candidate in candidates[:256]:
                evidence, error = _output_evidence(
                    candidate, workspace_root, decode_cache)
                if not evidence:
                    decode_errors.append(error)
            if len(candidates) > 256:
                decode_errors.append(
                    f"{len(candidates)} outputs are too broad for exact validation")
            if decode_errors:
                issues.append(_issue(
                    f"deliverable-{identifier}-integrity", "major",
                    "Exact output parser/decoder failures: "
                    + "; ".join(decode_errors[:8]),
                    "Repair or regenerate every declared final output, then open it "
                    "with the format's real parser/decoder before visual review.",
                    [str(path.relative_to(workspace_root))
                     for path in candidates[:8]],
                ))
    from spiral.artifact_gate import verify_workspace

    integrity = verify_workspace(workspace_root)
    integrity_errors = [
        error for error in integrity.errors
        if error != "no structurally verifiable artifacts were found"
    ]
    if integrity_errors:
        issues.append(_issue(
            "artifact-integrity", "major",
            "Structural artifact validation failed: "
            + "; ".join(integrity_errors[:8]),
            "Repair every malformed, corrupt, empty or undecodable delivered artifact and "
            "rerun the corresponding parser/decoder before sign-off.",
        ))
    placeholders = []
    source_text = []
    for path in source_files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        source_text.append(text)
        for line_no, line in enumerate(text.splitlines(), 1):
            if "placeholder=" in line.lower():
                continue
            if _PLACEHOLDER.search(line):
                placeholders.append(
                    f"{path.relative_to(root)}:{line_no}: {' '.join(line.split())[:120]}")
                if len(placeholders) >= 20:
                    break
    if placeholders:
        files = sorted({row.split(":", 1)[0] for row in placeholders})
        issues.append(_issue(
            "product-placeholders", "major",
            "Unfinished production markers: " + "; ".join(placeholders[:8]),
            "Replace every unfinished marker with the real behavior or remove the dead path; "
            "do not rename or hide markers.", files,
        ))

    tests = _test_files(root)
    if code_required and not tests:
        issues.append(_issue(
            "product-behavior-tests", "major",
            "No behavioral test files were found for a full product build.",
            "Add focused automated tests for the primary success path and meaningful boundary "
            "and failure paths, then wire them into the normal test/build command.",
        ))
    elif code_required:
        test_text = "\n".join(
            path.read_text(errors="replace") for path in tests if path.stat().st_size < 500_000)
        if not re.search(
                r"\b(?:assert|expect|should|test|it|describe)\s*(?:\(|\b)|@Test\b|#\[test\]",
                test_text, re.I):
            issues.append(_issue(
                "product-test-substance", "major",
                "Test files exist, but no executable assertions or test cases were detected.",
                "Replace empty or ceremonial tests with behavioral assertions for primary, "
                "boundary, and failure paths.",
                [str(path.relative_to(root)) for path in tests[:8]],
            ))

    from spiral.conductor import detect_gate

    if code_required and not detect_gate(workspace_root):
        issues.append(_issue(
            "product-verification-gate", "major",
            "No supported automated build/test gate is discoverable for this product.",
            "Wire the behavioral suite and production build into the ecosystem's normal "
            "project metadata so a fresh checkout has one deterministic non-interactive gate.",
        ))

    readme_locations = [workspace_root, *[
        candidate for candidate in roots if candidate != workspace_root
    ]]
    readme = next((
        location / name
        for location in readme_locations
        for name in ("README.md", "README.rst", "README.txt")
        if (location / name).is_file()
    ), None)
    readme_text = readme.read_text(errors="replace") if readme else ""
    if code_required and (len(readme_text.strip()) < 120 or not re.search(
            r"\b(?:install|setup|run|usage|quickstart|getting started)\b",
            readme_text, re.I)):
        issues.append(_issue(
            "product-runnable-delivery", "major",
            "No substantive setup/run instructions were found in the project README.",
            "Document exact prerequisites, install, run, test, and build/package commands plus "
            "safe example configuration for a fresh checkout.",
            [str(readme.relative_to(workspace_root))] if readme else ["README.md"],
        ))

    combined = "\n".join(source_text).lower()
    if profile == "plot" and any(token in combined for token in (
            "matplotlib", "plotly", "recharts", "chart.js", "d3.", "plt.plot", "go.figure")):
        label_signals = sum(token in combined for token in (
            "xlabel", "ylabel", "axis.title", "xaxis", "yaxis", "legend", "aria-label"))
        if label_signals < 2:
            issues.append(_issue(
                "plot-semantic-labels", "major",
                "Plotting code was found without enough axis/unit/legend accessibility signals.",
                "Add meaningful labels and units, direct labels or a legend, and non-color-only "
                "series distinction; cover the rendered semantics with a test.",
            ))
        if not any(token in combined for token in (
                "savefig", "downloadimage", "to_image", "write_image", "export", "download")):
            issues.append(_issue(
                "plot-export", "major",
                "Plotting code was found without a figure/data export path.",
                "Implement reproducible figure export and underlying data export with explicit "
                "format, filename, and error handling.",
            ))
    if profile == "simulation" and re.search(r"\b(?:random|randn|randint|monte carlo)\b", combined):
        if not re.search(r"\b(?:seed|random_state|rng)\b", combined):
            issues.append(_issue(
                "simulation-seed", "major",
                "Stochastic simulation code was found without an explicit reproducibility seed.",
                "Accept, validate, record, and test an explicit random seed; include it in exports.",
            ))
    if profile == "simulation" and tests:
        test_blob = "\n".join(
            path.read_text(errors="replace") for path in tests if path.stat().st_size < 500_000).lower()
        if not any(token in test_blob for token in (
                "allclose", "isclose", "approx", "reference", "invariant", "conservation",
                "residual", "benchmark")):
            issues.append(_issue(
                "simulation-reference-check", "major",
                "Simulation tests contain no recognizable numerical reference, invariant, or residual check.",
                "Test at least one analytic/reference case and one numerical invariant or residual "
                "with explicit tolerances.",
            ))

    report = {
        "schema_version": 1,
        "applicable": True,
        "profile": profile,
        "project_root": ".",
        "project_roots": [
            str(candidate.relative_to(workspace_root) or Path(".")) for candidate in roots
        ],
        "deliverables": [
            {
                "id": row.get("id"),
                "kind": row.get("kind"),
                "artifact_count": artifact_counts.get(
                    str(row.get("id") or row.get("kind") or "other"),
                    len(_artifact_candidates(
                        root, str(row.get("kind") or "other"))),
                ),
            }
            for row in deliverables
        ],
        "production_file_count": len(source_files),
        "test_file_count": len(tests),
        "artifact_integrity": {
            "ok": integrity.ok,
            "verified": integrity.verified,
            "skipped": integrity.skipped,
            "errors": integrity.errors,
        },
        "issues": issues,
        "scope": (
            "Fail-only deterministic checks. A clean report does not prove usability, semantic "
            "correctness, accessibility, or visual quality; runtime, spec, and visual gates remain required."
        ),
    }
    return report


def write_product_audit(report: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
