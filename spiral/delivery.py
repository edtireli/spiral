"""Exact delivery manifest for Builder finish gates."""
from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
import time
from pathlib import Path

from spiral.artifact_gate import verify_workspace
from spiral.builder_tools import discover_project_roots


KIND_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".tiff"},
    "video": {".mp4", ".mov", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"},
    "document": {".pdf", ".docx", ".odt", ".tex", ".md", ".rst", ".html", ".rtf"},
    "presentation": {".pptx", ".odp", ".pdf"},
    "dataset": {
        ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".arrow", ".sqlite",
        ".sqlite3", ".db", ".xlsx", ".ods",
    },
    "notebook": {".ipynb"},
    "3d": {".glb", ".gltf", ".obj", ".fbx", ".blend", ".stl", ".ply"},
    "plot": {".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf", ".html"},
    "formal-proof": {".lean", ".v", ".agda", ".thy"},
}
PROJECT_MARKERS = {
    "web": {"package.json", "index.html"},
    "android": {"AndroidManifest.xml", "build.gradle", "build.gradle.kts"},
    "ios": {"Package.swift"},
    "desktop": {
        "package.json", "pyproject.toml", "Cargo.toml", "Package.swift", "CMakeLists.txt",
    },
    "cli": {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Package.swift"},
    "service": {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml"},
    "library": {
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "pom.xml", "Package.swift",
    },
    "simulation": {
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "CMakeLists.txt",
    },
    "game": {"package.json", "project.godot", "Cargo.toml", "CMakeLists.txt"},
    "firmware": {"platformio.ini", "CMakeLists.txt", "Makefile", "Cargo.toml"},
    "infrastructure": {"main.tf", "Pulumi.yaml", "docker-compose.yml", "Dockerfile"},
    "other": {
        "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile",
    },
}
VISUAL_KINDS = {
    "web", "android", "ios", "desktop", "simulation", "plot", "image",
    "video", "document", "presentation", "notebook", "3d", "game",
}
PROJECT_OUTPUT_SUFFIXES = {
    ".apk", ".aab", ".ipa", ".app", ".xcarchive", ".dmg", ".pkg",
    ".exe", ".msi", ".appimage", ".deb", ".rpm", ".jar", ".war",
    ".whl", ".crate", ".nupkg", ".vsix", ".zip", ".tgz", ".tar",
    ".wasm",
}
OUTPUT_DIRECTORY_NAMES = {
    "dist", "build", "out", "output", "outputs", "release", "releases",
    "artifacts", "package", "packages", "bundle", "bundles", "bin",
}


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if path.is_symlink():
        try:
            path.resolve(strict=False).relative_to(root)
        except (OSError, ValueError):
            return True
    return any(
        part.startswith(".") or part.lower() in {
            "node_modules", "coverage", "__pycache__", ".pytest_cache",
            "tests", "test", "fixtures", "examples",
        }
        for part in rel.parts
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(path: Path, root: Path) -> tuple[list[Path], int]:
    files = []
    total = 0
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            try:
                member.resolve(strict=False).relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"bundle symlink escapes workspace: {member.relative_to(root)}"
                ) from exc
        if not member.is_file():
            continue
        files.append(member)
        total += member.stat().st_size
    return files, total


def _directory_sha256(path: Path, members: list[Path]) -> str:
    digest = hashlib.sha256()
    for member in members:
        rel = str(member.relative_to(path)).encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(member.stat().st_size.to_bytes(8, "big"))
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict:
    if path.is_dir():
        members, size = _directory_inventory(path, root)
        return {
            "path": str(path.relative_to(root)),
            "type": "directory",
            "members": len(members),
            "bytes": size,
            "sha256": (
                _directory_sha256(path, members)
                if size <= 250 * 1024 * 1024 else ""
            ),
            "modified": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
        }
    size = path.stat().st_size
    return {
        "path": str(path.relative_to(root)),
        "type": "file",
        "bytes": size,
        "sha256": _sha256(path) if size <= 250 * 1024 * 1024 else "",
        "modified": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
    }


def _artifact_files(root: Path, kind: str) -> list[Path]:
    extensions = KIND_EXTENSIONS.get(kind, set())
    rows = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
        and not _skip(path, root)
        and not (
            kind == "dataset"
            and path.name.lower() in {
                "package.json", "package-lock.json", "tsconfig.json",
                "composer.json", "artifacts.json", "delivery.json",
            }
        )
    ]

    def score(path: Path) -> tuple[int, int, float]:
        name = path.stem.lower()
        intent = sum(
            10 for token in (
                "final", "output", "result", "render", "figure", "plot",
                "report", "paper", "poster", "presentation",
            ) if token in name
        )
        return intent, path.stat().st_size, path.stat().st_mtime

    return sorted(rows, key=score, reverse=True)


def _declared_artifact_files(
    root: Path, kind: str, patterns: list[str],
) -> tuple[list[Path], list[str]]:
    """Resolve analyst-declared outputs without allowing an escaping glob."""

    resolved: list[Path] = []
    rejected: list[str] = []
    allowed = KIND_EXTENSIONS.get(kind, set())
    for raw in patterns[:12]:
        pattern = str(raw).strip().removeprefix("./")
        if (
            not pattern or pattern.startswith(("/", "~"))
            or ".." in Path(pattern).parts
            or pattern in {".", "*", "**", "**/*"}
        ):
            rejected.append(str(raw))
            continue
        for path in root.glob(pattern):
            try:
                resolved_path = path.resolve()
                resolved_path.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved_path == root:
                continue
            source_directory = (
                path.is_dir()
                and path.name.lower() in {
                    "src", "source", "app", "lib", "tests", "test",
                    "fixtures", "examples",
                }
            )
            project_output = (
                bool(allowed)
                or path.suffix.lower() in PROJECT_OUTPUT_SUFFIXES
                or path.name.lower().endswith(".tar.gz")
                or any(
                    part.lower() in OUTPUT_DIRECTORY_NAMES
                    for part in path.relative_to(root).parts[:-1]
                )
                or (
                    path.is_dir()
                    and path.name.lower() in OUTPUT_DIRECTORY_NAMES
                )
                or (
                    kind == "web" and path.is_file()
                    and path.name.lower() == "index.html"
                )
            )
            if (
                (path.is_file() or (path.is_dir() and not allowed))
                and project_output and not source_directory and not _skip(path, root)
                and (not allowed or (
                    path.is_file() and path.suffix.lower() in allowed
                ))
                and path not in resolved
            ):
                if path.is_dir() and not any(
                        member.is_file() for member in path.rglob("*")):
                    continue
                resolved.append(path)
    return sorted(resolved), rejected


def _apple_bundle_evidence(path: Path, root: Path) -> tuple[str, str]:
    """Validate the inspectable structure of an Apple app/archive bundle."""

    try:
        if path.suffix.lower() == ".xcarchive":
            info = path / "Info.plist"
            if not info.is_file():
                raise ValueError("XCArchive has no Info.plist")
            plistlib.loads(info.read_bytes())
            apps = sorted((path / "Products" / "Applications").glob("*.app"))
            if not apps:
                raise ValueError("XCArchive contains no application bundle")
            nested, error = _apple_bundle_evidence(apps[0], root)
            if error:
                raise ValueError(error)
            return (
                f"{path.relative_to(root)}: XCArchive metadata and nested "
                f"application validated; {nested}", ""
            )

        candidates = [path / "Contents" / "Info.plist", path / "Info.plist"]
        info = next((candidate for candidate in candidates if candidate.is_file()), None)
        if info is None:
            raise ValueError("application bundle has no Info.plist")
        metadata = plistlib.loads(info.read_bytes())
        executable = str(metadata.get("CFBundleExecutable") or "").strip()
        identifier = str(metadata.get("CFBundleIdentifier") or "").strip()
        if not executable or not identifier:
            raise ValueError(
                "application Info.plist lacks CFBundleExecutable or CFBundleIdentifier")
        executable_paths = [
            path / "Contents" / "MacOS" / executable,
            path / executable,
        ]
        binary = next((
            candidate for candidate in executable_paths
            if candidate.is_file() and candidate.stat().st_size > 0
        ), None)
        if binary is None:
            raise ValueError(
                f"application bundle lacks executable {executable}")
        members, _size = _directory_inventory(path, root)
        if len(members) < 2:
            raise ValueError("application bundle is effectively empty")
        return (
            f"{path.relative_to(root)}: application bundle {identifier} decoded "
            f"with {len(members)} file(s)", ""
        )
    except Exception as exc:
        return "", f"{path.relative_to(root)}: {type(exc).__name__}: {exc}"


def _output_evidence(
    path: Path, root: Path, cache: dict[Path, object] | None = None,
) -> tuple[str, str]:
    """Obtain parser/decoder evidence for one exact declared output."""

    if path.is_dir():
        if path.suffix.lower() in {".app", ".xcarchive"}:
            return _apple_bundle_evidence(path, root)
        report = cache.get(path) if cache is not None else None
        if report is None:
            report = verify_workspace(path)
            if cache is not None:
                cache[path] = report
        if report.ok:
            return (
                f"{path.relative_to(root)}: output directory decoded "
                f"({report.verified} verified item(s))", ""
            )
        detail = "; ".join(report.errors[:6]) or "no decodable members"
        return "", f"{path.relative_to(root)}: {detail}"

    report = cache.get(path.parent) if cache is not None else None
    if report is None:
        report = verify_workspace(path.parent)
        if cache is not None:
            cache[path.parent] = report
    local = str(path.relative_to(path.parent))
    evidence = next((
        row for row in report.evidence
        if row.split(":", 1)[0] == local
    ), "")
    error = next((
        row for row in report.errors
        if row.split(":", 1)[0] == local
    ), "")
    if evidence:
        suffix = evidence.split(":", 1)[1].strip()
        return f"{path.relative_to(root)}: {suffix}", ""
    return "", (
        f"{path.relative_to(root)}: "
        + (error.split(":", 1)[1].strip() if error else
           "no parser/decoder recognized this output")
    )


def _project_evidence(
    root: Path, project_roots: list[Path], kind: str,
) -> list[dict]:
    markers = PROJECT_MARKERS.get(kind, PROJECT_MARKERS["other"])
    records = []
    for project_root in project_roots:
        files = [
            project_root / marker for marker in sorted(markers)
            if (project_root / marker).is_file()
        ]
        if kind == "android":
            manifest = next(project_root.rglob("AndroidManifest.xml"), None)
            if manifest and manifest not in files:
                files.append(manifest)
        records.append({
            "root": str(project_root.relative_to(root) or Path(".")),
            "markers": [_file_record(path, root) for path in files[:8]],
        })
    return records


def _project_roots(root: Path, kind: str, hint: str) -> list[Path]:
    hinted = (root / hint).resolve() if hint else root
    candidates = [
        path for path in discover_project_roots(root)
        if path == hinted or hinted in path.parents or path in hinted.parents
    ] or discover_project_roots(root)
    markers = PROJECT_MARKERS.get(kind, PROJECT_MARKERS["other"])
    rows = []
    for candidate in candidates:
        if any((candidate / marker).exists() for marker in markers):
            rows.append(candidate)
            continue
        if kind == "android" and next(candidate.rglob("AndroidManifest.xml"), None):
            rows.append(candidate)
        elif kind == "ios" and (
                next(candidate.glob("*.xcodeproj"), None)
                or next(candidate.glob("*.xcworkspace"), None)):
            rows.append(candidate)
    return rows


def build_delivery_manifest(
    workspace: str | Path, declaration: dict, *,
    visual_status: str | dict = "", gate: str = "",
) -> dict:
    root = Path(workspace).resolve()
    integrity = verify_workspace(root)
    evidence_by_path: dict[str, str] = {}
    for evidence in integrity.evidence:
        evidence_by_path[evidence.split(":", 1)[0]] = evidence
    deliverables = []
    for row in declaration.get("deliverables") or []:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "other")
        hint = str(row.get("root_hint") or ".")
        patterns = [
            str(item) for item in (row.get("output_globs") or [])
            if str(item).strip()
        ]
        if kind in KIND_EXTENSIONS and not patterns:
            patterns = ["output/*"]
        if patterns:
            files, rejected_patterns = _declared_artifact_files(
                root, kind, patterns)
            resolution = "declared-output-glob"
        else:
            files = _artifact_files(root, kind)
            rejected_patterns = []
            resolution = "inferred-candidate"
        roots = _project_roots(root, kind, hint) if kind not in KIND_EXTENSIONS else []
        records = [_file_record(path, root) for path in files[:24]]
        structural = []
        output_errors = []
        output_evidence_cache: dict[Path, object] = {}
        for path in files[:256]:
            evidence = (
                evidence_by_path.get(str(path.relative_to(root)), "")
                if path.is_file() else ""
            )
            error = ""
            if not evidence:
                evidence, error = _output_evidence(
                    path, root, output_evidence_cache)
            if evidence:
                structural.append(evidence)
            else:
                output_errors.append(error or (
                    f"{path.relative_to(root)}: no structural evidence"))
        if len(files) > 256:
            output_errors.append(
                f"declared output globs resolve {len(files)} items; package them or "
                "narrow the declaration to at most 256 exact outputs")
        file_deliverable = kind in KIND_EXTENSIONS
        project_ok = bool(roots)
        files_required = file_deliverable or bool(patterns)
        files_ok = bool(records) if files_required else True
        files_structural = (
            bool(files) and len(structural) == len(files) and not output_errors
            if files_required else True
        )
        has_output = (
            files_ok if file_deliverable
            else project_ok and files_ok
        )
        structure_ok = (
            not rejected_patterns
            and files_structural
            and (True if file_deliverable else project_ok)
        )
        visual_required = bool(row.get("visual")) or kind in VISUAL_KINDS
        row_visual_status = (
            str(
                visual_status.get(str(row.get("id") or ""))
                or visual_status.get(kind)
                or visual_status.get("_overall")
                or ""
            )
            if isinstance(visual_status, dict) else str(visual_status or "")
        )
        visual_ok = (
            not visual_required
            or row_visual_status in {"green", "disabled-by-user"}
        )
        deliverables.append({
            "id": str(row.get("id") or ""),
            "kind": kind,
            "description": str(row.get("description") or ""),
            "declared_root_hint": hint,
            "project_roots": [str(path.relative_to(root) or Path(".")) for path in roots],
            "project_evidence": _project_evidence(root, roots, kind),
            "declared_output_globs": patterns,
            "rejected_output_globs": rejected_patterns,
            "resolution": resolution,
            "files": records,
            "resolved_file_count": len(files),
            "structural_evidence": structural,
            "acceptance_criteria": row.get("acceptance_evidence") or [],
            "visual_required": visual_required,
            "visual_status": (
                row_visual_status if visual_required else "not-applicable"),
            "output_present": has_output,
            "structure_ok": structure_ok,
            "ready": has_output and structure_ok and visual_ok,
            "issues": [
                *([] if has_output else [
                    (
                        "no exact declared output file resolved"
                        if patterns else
                        "no exact output file or runnable project root resolved"
                    )
                ]),
                *([] if structure_ok else [
                    "output lacks parser/decoder evidence",
                    *output_errors[:8],
                ]),
                *([] if visual_ok else [f"visual evidence is {visual_status or 'missing'}"]),
                *(
                    [f"unsafe output globs rejected: {', '.join(rejected_patterns)}"]
                    if rejected_patterns else []
                ),
            ],
        })
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, stdin=subprocess.DEVNULL,
    )
    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "revision": git.stdout.strip() if git.returncode == 0 else "",
        "gate": gate,
        "goal_sha256": declaration.get("goal_sha256") or "",
        "integrity": {
            "ok": integrity.ok,
            "verified": integrity.verified,
            "skipped": integrity.skipped,
            "errors": integrity.errors,
        },
        "deliverables": deliverables,
        "ready": bool(deliverables) and all(
            row.get("ready") for row in deliverables),
    }


def write_delivery_manifest(
    workspace: str | Path, declaration: dict, *,
    visual_status: str | dict = "", gate: str = "",
) -> Path:
    root = Path(workspace).resolve()
    target = root / ".spiral" / "delivery.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_delivery_manifest(
        root, declaration, visual_status=visual_status, gate=gate,
    ), indent=2), encoding="utf-8")
    return target
