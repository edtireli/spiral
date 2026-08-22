"""Managed self-modification boundary for Spiral's permission kernel."""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


MANAGED_ENV = "SPIRALCHAT_EXTERNAL_GIT_APPROVAL"
EDIT_CAPABILITY_ENV = "SPIRAL_ALLOW_SAFETY_KERNEL_EDIT"
PROTECTED_PATHS_ENV = "SPIRALCHAT_PROTECTED_PATHS"


class SafetyBoundaryError(RuntimeError):
    """The controller-supplied protected-path contract is unsafe or stale."""


@dataclass(frozen=True)
class ProtectedBoundary:
    path: Path
    kind: str                         # tree | file
    editable_with_safety_capability: bool = False
    device: int | None = None
    inode: int | None = None

# Protect the runtime as a boundary, not as a hand-maintained list of today's obvious
# security modules.  A change to conductor.py, planner.py, __init__.py, or a newly added
# import hook can weaken the exact same policy indirectly.  The separate capability is
# therefore required for every file below ``spiral/``.
PROTECTED_RUNTIME_ROOTS = ("spiral",)

# These files can redirect what code the ``spiral`` command imports before any in-package
# guard runs.  Include conventional alternatives even when this checkout does not have
# them yet, so creating a new packaging/entry surface is also a protected change.
PROTECTED_ENTRY_PATHS = frozenset({
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "spiral.py",
    "__main__.py",
    "sitecustomize.py",
    "usercustomize.py",
})

# Keep the boundary tests themselves guarded.  This is defence in depth rather than the
# enforcement mechanism: tests are evidence, while the runtime-tree rule above is the
# actual boundary.
PROTECTED_TEST_PATHS = frozenset({
    "tests/test_safety_boundaries.py",
    "tests/test_sandbox_conformance.py",
    "tests/test_execution_policy.py",
})

# Backwards-compatible export for callers that enumerate exact sentinels. Runtime files
# are intentionally represented by ``PROTECTED_RUNTIME_ROOTS`` instead of a stale list.
PROTECTED_RELATIVE_PATHS = PROTECTED_ENTRY_PATHS | PROTECTED_TEST_PATHS


def is_spiral_source_tree(root: str | Path) -> bool:
    base = Path(root).resolve()
    return (base / "spiral" / "command_broker.py").is_file() and (base / "pyproject.toml").is_file()


def _normal_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SafetyBoundaryError("protected paths must be canonical absolute paths")
    return Path(os.path.abspath(os.fspath(path)))


def _configured_boundaries() -> tuple[ProtectedBoundary, ...]:
    raw = os.environ.get(PROTECTED_PATHS_ENV, "").strip()
    if not raw:
        return ()
    if len(raw) > 64 * 1024:
        raise SafetyBoundaryError(f"{PROTECTED_PATHS_ENV} is too large")
    try:
        records = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise SafetyBoundaryError(f"{PROTECTED_PATHS_ENV} is not valid JSON") from exc
    if not isinstance(records, list) or len(records) > 64:
        raise SafetyBoundaryError(
            f"{PROTECTED_PATHS_ENV} must be a JSON array of at most 64 records")
    boundaries: list[ProtectedBoundary] = []
    for record in records:
        if not isinstance(record, dict):
            raise SafetyBoundaryError("each protected-path record must be an object")
        raw_path = record.get("path")
        kind = str(record.get("kind") or "").strip().lower()
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise SafetyBoundaryError("protected-path records need a non-empty path")
        if kind not in {"tree", "file"}:
            raise SafetyBoundaryError("protected-path kind must be 'tree' or 'file'")
        path = _normal_path(raw_path)
        supplied_dev = record.get("dev")
        supplied_ino = record.get("ino")
        if (supplied_dev is None) != (supplied_ino is None):
            raise SafetyBoundaryError("protected-path identity requires both dev and ino")
        try:
            device = int(supplied_dev) if supplied_dev is not None else None
            inode = int(supplied_ino) if supplied_ino is not None else None
        except (TypeError, ValueError) as exc:
            raise SafetyBoundaryError(
                "protected-path dev and ino must be integers") from exc
        if device is not None:
            try:
                info = path.stat()
            except OSError as exc:
                raise SafetyBoundaryError(
                    f"protected-path identity is unavailable for {path}: {exc}") from exc
            if (int(info.st_dev), int(info.st_ino)) != (device, inode):
                raise SafetyBoundaryError(
                    f"protected-path identity changed for {path}")
        boundaries.append(ProtectedBoundary(
            path=path,
            kind=kind,
            editable_with_safety_capability=(
                record.get("editable_with_safety_capability") is True
            ),
            device=device,
            inode=inode,
        ))
    return tuple(boundaries)


def _local_spiral_boundaries(root: str | Path) -> tuple[ProtectedBoundary, ...]:
    base = Path(root).resolve()
    if not is_spiral_source_tree(base):
        return ()
    return tuple([
        *(ProtectedBoundary(
            base / rel, "tree", editable_with_safety_capability=True,
        ) for rel in PROTECTED_RUNTIME_ROOTS),
        *(ProtectedBoundary(
            base / rel, "file", editable_with_safety_capability=True,
        ) for rel in sorted(PROTECTED_RELATIVE_PATHS)),
    ])


def protected_boundaries(root: str | Path) -> tuple[ProtectedBoundary, ...]:
    """Active global and local boundaries for this controller-managed process.

    The safety capability is deliberately narrow: it removes only records the
    controller marked editable. Host helpers and LaunchAgents default to immutable and
    remain protected even during an explicitly approved Spiral source change.
    """
    if os.environ.get(MANAGED_ENV) != "1":
        return ()
    boundaries = (*_configured_boundaries(), *_local_spiral_boundaries(root))
    allow_source_edit = os.environ.get(EDIT_CAPABILITY_ENV) == "1"
    active = [
        boundary for boundary in boundaries
        if not (allow_source_edit and boundary.editable_with_safety_capability)
    ]
    # Duplicate records only make the boundary stricter. Keep a stable order so audit
    # output and sandbox profiles are deterministic.
    unique: dict[tuple[str, str, bool], ProtectedBoundary] = {}
    for boundary in active:
        key = (
            str(boundary.path), boundary.kind,
            boundary.editable_with_safety_capability,
        )
        unique.setdefault(key, boundary)
    return tuple(unique[key] for key in sorted(unique))


def protection_active(root: str | Path) -> bool:
    return bool(protected_boundaries(root))


def protected_paths(root: str | Path, *, existing_only: bool = False) -> tuple[Path, ...]:
    paths = tuple(boundary.path for boundary in protected_boundaries(root))
    return tuple(path for path in paths if path.exists()) if existing_only else paths


def protected_relative_path(root: str | Path, target: str | Path) -> str:
    base = Path(root).resolve()
    candidate = Path(target).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    # Deliberately do not resolve the target: following a symlink first could turn a
    # lexical ``spiral/...`` write into an outside path and evade the tree boundary.
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    for boundary in protected_boundaries(base):
        if boundary.kind == "file" and candidate != boundary.path:
            continue
        if boundary.kind == "tree":
            try:
                candidate.relative_to(boundary.path)
            except ValueError:
                continue
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            return str(candidate)
    return ""


def rejection_reason(relative: str) -> str:
    return (
        f"managed self-modification cannot alter safety kernel/runtime boundary {relative}; "
        f"the controller must grant the separate {EDIT_CAPABILITY_ENV}=1 capability"
    )
