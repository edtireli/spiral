"""Git-backed task transactions for Builder.

Every autonomous task starts from a clean commit. Failed or interrupted work is
archived under ``.spiral/recovery`` and then restored with argv-based git calls;
model-facing command policy is deliberately not involved in this trusted harness
operation.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def _git(root: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True,
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
    rows = []
    for path in root.rglob("*"):
        if not path.is_symlink():
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

    @classmethod
    def begin(cls, workspace: str | Path, label: str) -> "TaskTransaction":
        root = Path(workspace).resolve()
        if _dirty(root):
            raise RuntimeError(
                "workspace changed outside the active task; refusing to mix "
                "unrelated edits into an autonomous commit"
            )
        return cls(root, label, head(root), _untracked(root))

    def archive(self, reason: str) -> Path | None:
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
        return _dirty(self.root)

    def restore_worktree_from_index(self) -> None:
        """Discard one uncommitted candidate while retaining the staged baseline."""

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
