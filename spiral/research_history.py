"""Private, content-addressed version history for a Spiral research run.

This uses a dedicated Git object database at ``.research-git`` and never touches
the enclosing repository's branch, index, remotes, or configuration.  Raw corpus
downloads, cloned repositories, dependency trees, and caches are excluded; their
hashes and manifests are versioned instead.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class ResearchGit:
    ALLOWED_SUFFIXES = {
        ".bib", ".cfg", ".cpp", ".csv", ".html", ".json", ".jsonl", ".lean",
        ".ipynb", ".jl", ".log", ".m", ".md", ".pdf", ".png", ".py", ".r",
        ".sage", ".singular", ".svg", ".tex", ".toml", ".txt", ".yaml", ".yml",
    }
    EXCLUDED_PARTS = {
        ".research-git", "_acquisition_home", "_cache", "_deps", "_home",
        "_repos", "_tmp", "__pycache__", "node_modules", ".venv", ".git",
    }

    def __init__(self, research_root: str | Path, *, enabled: bool = True,
                 max_file_mb: int = 20):
        self.root = Path(research_root).resolve()
        self.git_dir = self.root / ".research-git"
        self.index = self.git_dir / "spiral-index"
        self.enabled = bool(enabled and shutil.which("git"))
        self.max_bytes = max(1, int(max_file_mb)) * 1024 * 1024
        self.timeline = self.root / "history" / "checkpoints.jsonl"

    def _env(self) -> dict[str, str]:
        return {
            **os.environ,
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.root),
            "GIT_INDEX_FILE": str(self.index),
            "GIT_AUTHOR_NAME": "Spiral Research",
            "GIT_AUTHOR_EMAIL": "research@localhost",
            "GIT_COMMITTER_NAME": "Spiral Research",
            "GIT_COMMITTER_EMAIL": "research@localhost",
        }

    def _run(self, args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [shutil.which("git") or "git", *args], cwd=self.root,
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=self._env(),
        )

    def _init(self) -> bool:
        if not self.enabled:
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.git_dir.is_dir():
            result = subprocess.run(
                [shutil.which("git") or "git", "init", "--bare", "-q", str(self.git_dir)],
                cwd=self.root, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=60,
            )
            if result.returncode != 0:
                return False
            self._run(["symbolic-ref", "HEAD", "refs/heads/main"])
        return True

    def _eligible_files(self) -> list[str]:
        files = []
        corpus_root = self.root / "corpus"
        for path in self.root.rglob("*"):
            try:
                if not path.is_file() or path == self.index:
                    continue
                rel = path.relative_to(self.root)
                if any(part in self.EXCLUDED_PARTS for part in rel.parts):
                    continue
                if corpus_root in path.parents:
                    # Keep corpus metadata/manifests, not downloaded source bodies.
                    if path.name not in {"manifest.json", "download-report.json"}:
                        continue
                if path.stat().st_size > self.max_bytes:
                    continue
                if path.suffix.lower() not in self.ALLOWED_SUFFIXES:
                    continue
                files.append(str(rel))
            except OSError:
                continue
        return sorted(files)

    def head(self) -> str:
        if not self.git_dir.is_dir():
            return ""
        result = self._run(["rev-parse", "--verify", "HEAD"], timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""

    def checkpoint(self, label: str, *, phase: str = "", metadata=None) -> dict:
        """Commit all eligible research artifacts into the private object database."""

        if not self._init():
            return {"ok": False, "commit": "", "error": "research git unavailable"}
        previous = self.head()
        if previous:
            if self._run(["read-tree", previous]).returncode != 0:
                return {"ok": False, "commit": "", "error": "could not read prior checkpoint"}
        else:
            self._run(["read-tree", "--empty"])
        # Stage tracked deletions in the private index, then current eligible files.
        self._run(["add", "-u", "--", "."])
        files = self._eligible_files()
        for start in range(0, len(files), 200):
            result = self._run(["add", "--", *files[start:start + 200]], timeout=180)
            if result.returncode != 0:
                return {
                    "ok": False, "commit": "",
                    "error": (result.stderr or result.stdout).strip()[-500:],
                }
        changed = self._run(["diff", "--cached", "--quiet"]).returncode != 0
        if not changed:
            return {"ok": True, "commit": previous, "changed": False, "files": len(files)}
        message = f"{phase + ': ' if phase else ''}{' '.join(str(label).split())[:180]}"
        commit = self._run(["commit", "-q", "--no-gpg-sign", "-m", message], timeout=180)
        if commit.returncode != 0:
            return {
                "ok": False, "commit": "",
                "error": (commit.stderr or commit.stdout).strip()[-500:],
            }
        head = self.head()
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "phase": phase,
            "label": label,
            "commit": head,
            "parent": previous,
            "eligible_files": len(files),
            "metadata": dict(metadata or {}),
        }
        try:
            self.timeline.parent.mkdir(parents=True, exist_ok=True)
            with self.timeline.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
        return {"ok": True, "commit": head, "changed": True, "files": len(files)}

    def log(self, limit: int = 30) -> list[dict]:
        if not self.head():
            return []
        result = self._run([
            "log", f"-{max(1, int(limit))}",
            "--format=%H%x09%P%x09%aI%x09%s",
        ])
        rows = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split("\t", 3)
                if len(parts) == 4:
                    rows.append({
                        "commit": parts[0], "parents": parts[1].split(),
                        "date": parts[2], "subject": parts[3],
                    })
        return rows

    def verify(self) -> dict:
        """Verify the private object database and checkpoint ancestry."""

        head = self.head()
        if not self.enabled:
            return {"valid": True, "enabled": False, "head": "", "issues": []}
        if not self.git_dir.is_dir() or not head:
            return {
                "valid": False, "enabled": True, "head": head,
                "issues": ["private research history has no valid HEAD"],
            }
        issues = []
        fsck = self._run(["fsck", "--full", "--no-dangling"], timeout=180)
        if fsck.returncode != 0:
            issues.append((fsck.stderr or fsck.stdout or "git fsck failed").strip()[-1000:])
        ancestry = self._run(["rev-list", "--topo-order", "HEAD"], timeout=60)
        commits = [line.strip() for line in ancestry.stdout.splitlines() if line.strip()]
        if ancestry.returncode != 0 or not commits:
            issues.append("checkpoint ancestry could not be read")
        timeline_rows = 0
        if self.timeline.is_file():
            try:
                for line_no, line in enumerate(
                        self.timeline.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    timeline_rows += 1
                    commit = str(row.get("commit") or "")
                    if commit and commit not in commits:
                        issues.append(
                            f"timeline line {line_no} names an unreachable commit {commit[:12]}")
            except Exception as exc:
                issues.append(f"invalid checkpoint timeline: {type(exc).__name__}: {exc}")
        return {
            "valid": not issues,
            "enabled": True,
            "head": head,
            "commit_count": len(commits),
            "timeline_rows": timeline_rows,
            "issues": issues,
        }
