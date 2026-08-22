"""Task transactions for Builder.

Every autonomous task starts from a clean commit. Failed or interrupted work is
archived under ``.spiral/recovery`` and then restored with argv-based git calls;
model-facing command policy is deliberately not involved in this trusted harness
operation. SpiralChat-managed runs instead use private file snapshots and leave
successful edits uncommitted for the controller's explicit Git-approval flow.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


def external_git_approval() -> bool:
    """Whether Git changes are reserved for SpiralChat's typed approval API."""

    return os.environ.get("SPIRALCHAT_EXTERNAL_GIT_APPROVAL") == "1"


_SNAPSHOT_SKIP = {
    ".git", ".spiral", "node_modules", ".venv", ".gradle", "build", "target",
    "__pycache__", ".pytest_cache", ".mypy_cache",
}


def _managed_paths(root: Path) -> list[str]:
    """Files that constitute editable project state, without caches or Git data."""

    if (root / ".git").exists():
        listed = _git(root, "ls-files", "-co", "--exclude-standard", "-z")
        if listed.returncode == 0:
            rows = []
            for value in listed.stdout.split("\0"):
                if not value:
                    continue
                rel = Path(value)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] == (".spiral",):
                    continue
                path = root / rel
                if path.is_file() or path.is_symlink():
                    rows.append(rel.as_posix())
            return sorted(set(rows))
    rows = []
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        directories[:] = [
            name for name in directories
            if name not in _SNAPSHOT_SKIP and not (base / name).is_symlink()
        ]
        for name in files:
            path = base / name
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if not any(part in _SNAPSHOT_SKIP for part in rel.parts):
                rows.append(rel.as_posix())
        # os.walk lists directory symlinks only in ``directories``. Record them
        # as objects instead of following them.
        for name in list(os.listdir(base)):
            path = base / name
            if path.is_symlink():
                rel = path.relative_to(root)
                if not any(part in _SNAPSHOT_SKIP for part in rel.parts):
                    rows.append(rel.as_posix())
    return sorted(set(rows))


def _file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(b"link\0")
        digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        return digest.hexdigest()
    digest.update(b"file\0")
    try:
        digest.update(str(path.stat().st_mode & 0o777).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        digest.update(f"error:{type(exc).__name__}:{exc}".encode())
    return digest.hexdigest()


def workspace_fingerprint(workspace: str | Path) -> str:
    root = Path(workspace).resolve()
    digest = hashlib.sha256()
    for rel in _managed_paths(root):
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(_file_signature(root / rel).encode())
        digest.update(b"\0")
    return "worktree-" + digest.hexdigest()[:16]


def _copy_snapshot_object(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    if source.is_symlink():
        destination.symlink_to(os.readlink(source))
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _capture_snapshot(root: Path) -> tuple[Path, set[str], str]:
    parent = root / ".spiral" / "managed-snapshots"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="task-", dir=parent))
    paths = set(_managed_paths(root))
    for rel in sorted(paths):
        _copy_snapshot_object(root / rel, directory / "tree" / rel)
    fingerprint = workspace_fingerprint(root)
    (directory / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "paths": sorted(paths),
        "fingerprint": fingerprint,
    }, indent=2), encoding="utf-8")
    return directory, paths, fingerprint


def _restore_snapshot(root: Path, directory: Path, baseline: set[str]) -> None:
    current = set(_managed_paths(root))
    for rel in sorted(current - baseline, key=lambda value: value.count("/"), reverse=True):
        path = root / rel
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for rel in sorted(baseline):
        source = directory / "tree" / rel
        destination = root / rel
        if destination.exists() and destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        _copy_snapshot_object(source, destination)


def _git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "git", "-c", "user.name=Spiral", "-c",
            "user.email=spiral@localhost", *args,
        ], cwd=root, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, check=check,
    )


def head(root: str | Path) -> str:
    result = _git(Path(root).resolve(), "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else ""


def is_ancestor(root: str | Path, ancestor: str, descendant: str = "HEAD") -> bool:
    if not ancestor:
        return False
    return _git(
        Path(root).resolve(), "merge-base", "--is-ancestor", ancestor, descendant,
    ).returncode == 0


def _untracked(root: Path) -> set[str]:
    result = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    return {
        item for item in result.stdout.split("\0")
        if item and not item.startswith(".spiral/")
    }


def _dirty(root: Path) -> bool:
    result = _git(root, "status", "--porcelain", "--untracked-files=normal")
    for line in result.stdout.splitlines():
        if not line:
            continue
        payload = line[3:] if len(line) > 3 else line
        destination = payload.split(" -> ", 1)[-1].strip('"')
        if destination == ".spiral" or destination.startswith(".spiral/"):
            continue
        return True
    return False


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "task"


def _external_symlinks(root: Path) -> list[str]:
    """Escaping symlinks the TASK created. ``.spiral/`` is harness-owned and
    gitignored — its dependency-cache venvs legitimately symlink bin/python at an
    interpreter outside the workspace, so scanning it refuses every commit on any
    project that declares dependencies. Excluded for the same reason as in
    ``_dirty``/``_untracked``."""
    rows = []
    for path in root.rglob("*"):
        if not path.is_symlink():
            continue
        if path.relative_to(root).parts[:1] == (".spiral",):
            continue
        try:
            path.resolve(strict=False).relative_to(root)
        except (OSError, ValueError):
            rows.append(str(path.relative_to(root)))
    return rows


@dataclass
class TaskTransaction:
    root: Path
    label: str
    start_head: str
    baseline_untracked: set[str]
    managed: bool = False
    snapshot_dir: Path | None = None
    snapshot_paths: set[str] | None = None
    snapshot_fingerprint: str = ""

    @classmethod
    def begin(cls, workspace: str | Path, label: str) -> "TaskTransaction":
        root = Path(workspace).resolve()
        if external_git_approval():
            directory, paths, fingerprint = _capture_snapshot(root)
            return cls(
                root, label, fingerprint, set(), True,
                directory, paths, fingerprint,
            )
        if _dirty(root):
            raise RuntimeError(
                "workspace changed outside the active task; refusing to mix "
                "unrelated edits into an autonomous commit"
            )
        return cls(root, label, head(root), _untracked(root))

    def archive(self, reason: str) -> Path | None:
        if self.managed:
            if not self.has_changes():
                return None
            stamp = time.strftime("%Y%m%d-%H%M%S")
            out = self.root / ".spiral" / "recovery" / f"{stamp}-{_slug(self.label)}"
            suffix = 1
            while out.exists():
                suffix += 1
                out = out.with_name(f"{out.name}-{suffix}")
            out.mkdir(parents=True)
            current = set(_managed_paths(self.root))
            baseline = set(self.snapshot_paths or set())
            changed = [
                rel for rel in sorted(current)
                if rel not in baseline
                or _file_signature(self.root / rel) != _file_signature(
                    (self.snapshot_dir or Path()) / "tree" / rel)
            ]
            for rel in changed:
                _copy_snapshot_object(self.root / rel, out / "changed" / rel)
            (out / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "label": self.label,
                "reason": reason,
                "baseline": self.snapshot_fingerprint,
                "changed": changed,
                "deleted": sorted(baseline - current),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2), encoding="utf-8")
            return out
        diff = _git(self.root, "diff", "--binary", self.start_head or "HEAD", "--", ".")
        new_untracked = sorted(_untracked(self.root) - self.baseline_untracked)
        if not diff.stdout and not new_untracked:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = self.root / ".spiral" / "recovery" / f"{stamp}-{_slug(self.label)}"
        suffix = 1
        while out.exists():
            suffix += 1
            out = out.with_name(f"{out.name}-{suffix}")
        out.mkdir(parents=True)
        (out / "changes.patch").write_text(diff.stdout, encoding="utf-8")
        copied: list[str] = []
        skipped: list[str] = []
        for rel in new_untracked:
            source = self.root / rel
            destination = out / "untracked" / rel
            try:
                if source.is_symlink():
                    skipped.append(f"{rel} (symbolic link)")
                elif source.is_file() and source.stat().st_size <= 50 * 1024 * 1024:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    copied.append(rel)
                elif source.is_dir():
                    skipped.append(f"{rel} (directory)")
                else:
                    skipped.append(f"{rel} (larger than 50 MiB)")
            except OSError as exc:
                skipped.append(f"{rel} ({exc})")
        (out / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "label": self.label,
            "reason": reason,
            "start_head": self.start_head,
            "copied_untracked": copied,
            "skipped_untracked": skipped,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")
        return out

    def rollback(self, *, target: str | None = None, reason: str = "failed") -> Path | None:
        recovery = self.archive(reason)
        if self.managed:
            if self.snapshot_dir is None:
                raise RuntimeError("managed task snapshot is unavailable")
            _restore_snapshot(
                self.root, self.snapshot_dir, set(self.snapshot_paths or set()))
            return recovery
        revision = target or self.start_head or "HEAD"
        restored = _git(
            self.root, "restore", "--source", revision,
            "--staged", "--worktree", "--", ".",
        )
        if restored.returncode != 0:
            raise RuntimeError(
                restored.stderr.strip() or "transaction rollback failed")
        for rel in sorted(_untracked(self.root) - self.baseline_untracked, reverse=True):
            path = self.root / rel
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
        return recovery

    def commit(self, message: str) -> tuple[str, bool]:
        escaping = _external_symlinks(self.root)
        if escaping:
            raise RuntimeError(
                "task created symbolic links outside the workspace: "
                + ", ".join(escaping[:8])
            )
        if self.managed:
            before = self.snapshot_fingerprint
            after = workspace_fingerprint(self.root)
            if after == before:
                return before, False
            old = self.snapshot_dir
            directory, paths, fingerprint = _capture_snapshot(self.root)
            self.snapshot_dir = directory
            self.snapshot_paths = paths
            self.snapshot_fingerprint = fingerprint
            self.start_head = fingerprint
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
            return fingerprint, True
        before = head(self.root)
        _git(self.root, "add", "-A", "--", ".")
        staged = _git(self.root, "diff", "--cached", "--quiet")
        if staged.returncode == 0:
            return before[:7], False
        result = _git(self.root, "commit", "-q", "-m", message)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git commit failed")
        after = head(self.root)
        self.start_head = after
        self.baseline_untracked = _untracked(self.root)
        return after[:7], after != before

    def has_changes(self) -> bool:
        if self.managed:
            return workspace_fingerprint(self.root) != self.snapshot_fingerprint
        return _dirty(self.root)

    def restore_worktree_from_index(self) -> None:
        """Discard one uncommitted candidate while retaining the staged baseline."""

        if self.managed:
            if self.snapshot_dir is None:
                raise RuntimeError("managed task snapshot is unavailable")
            _restore_snapshot(
                self.root, self.snapshot_dir, set(self.snapshot_paths or set()))
            return

        _git(self.root, "restore", "--worktree", "--", ".")
        for rel in sorted(_untracked(self.root) - self.baseline_untracked, reverse=True):
            path = self.root / rel
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        if self.snapshot_dir is not None:
            shutil.rmtree(self.snapshot_dir, ignore_errors=True)
            self.snapshot_dir = None


def recover_interrupted_workspace(
    workspace: str | Path, *, last_green_head: str = "",
) -> dict:
    """Preserve interrupted edits and restore the last known committed state."""

    root = Path(workspace).resolve()
    result = {"changed": False, "recovery": "", "head": head(root)}
    if _dirty(root):
        tx = TaskTransaction(root, "interrupted-run", head(root), set())
        recovery = tx.rollback(target=head(root), reason="resume after interruption")
        result.update(changed=True, recovery=str(recovery or ""))
    if last_green_head and is_ancestor(root, last_green_head):
        current = head(root)
        if current != last_green_head:
            stem = f"spiral/recovery-{time.strftime('%Y%m%d-%H%M%S')}"
            branch = stem
            for index in range(1, 100):
                created = _git(root, "branch", branch, current)
                if created.returncode == 0:
                    break
                branch = f"{stem}-{index + 1}"
            else:
                raise RuntimeError(
                    "could not preserve interrupted commits on a recovery branch")
            reset = _git(root, "reset", "--hard", last_green_head)
            if reset.returncode != 0:
                raise RuntimeError(
                    reset.stderr.strip() or "could not restore last green commit")
            result.update(changed=True, recovery_branch=branch)
    result["head"] = head(root)
    return result
