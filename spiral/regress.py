"""Repair a runtime regression by construction: find the guilty commit and revert it.

When the finish probe says the page is broken, the current move is to ask a model
to repair it — and on a syntax-class fault the model flip-flops, because it cannot
see the imbalance the parser sees. But the harness holds two things a model does
not: the commit history, and a deterministic predicate for "broken". Between them,
which commit broke the product is not a question of judgment. It is a binary
search.

So: walk back to the last commit where the predicate passed, bisect the range to
the first bad commit, and ``git revert`` exactly that one. If the revert applies
cleanly and the predicate passes afterwards, the product is healed for zero model
tokens, and the requirement the reverted commit was chasing simply re-opens at the
next validation round — returning to the best verified state and moving forward
beats patching a broken one in place.

Everything here is deliberately cowardly. Probing happens in a **temporary
worktree**, never by checking out in the working tree. The revert happens only on
a clean tree, only for a cleanly applying revert, and is undone (``revert --abort``)
on any conflict. When any step cannot be done safely the answer is "not healed",
and the ordinary remediation path proceeds as before — this is a fast path, not a
replacement.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

Predicate = Callable[[Path], bool]      # True = this tree is healthy


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL)


@dataclass
class Healing:
    healed: bool
    guilty: str = ""
    reverted_as: str = ""
    detail: str = ""
    probed: int = 0
    log: list[str] = field(default_factory=list)


def _commits(root: Path, limit: int) -> list[str]:
    """HEAD-first commit shas on the current branch."""
    result = _git(root, "rev-list", "--first-parent", f"--max-count={limit}", "HEAD")
    return result.stdout.split()


def _dirty(root: Path) -> bool:
    out = _git(root, "status", "--porcelain").stdout
    return any(line and ".spiral" not in line for line in out.splitlines())


def heal(workspace: str | Path, predicate: Predicate, *,
         max_back: int = 32) -> Healing:
    """Locate the first commit that made ``predicate`` fail and revert it.

    ``predicate`` runs against a checkout in a temporary worktree; it must be
    deterministic and side-effect free. Returns ``healed=False`` with a reason
    whenever safety would be compromised — a dirty tree, no good commit within
    ``max_back``, a conflicting revert, or a revert that does not actually fix it.
    """
    root = Path(workspace).resolve()
    outcome = Healing(healed=False)
    if _dirty(root):
        outcome.detail = "working tree has uncommitted changes; not touching it"
        return outcome
    commits = _commits(root, max_back)
    if not commits:
        outcome.detail = "no history at all"
        return outcome

    with tempfile.TemporaryDirectory(prefix="spiral-bisect-") as tmp:
        probe_dir = Path(tmp) / "wt"
        added = _git(root, "worktree", "add", "--detach", str(probe_dir),
                     commits[0])
        if added.returncode != 0:
            outcome.detail = f"could not create a probe worktree: {added.stderr[:120]}"
            return outcome
        try:
            def healthy(sha: str) -> bool:
                outcome.probed += 1
                _git(probe_dir, "checkout", "--detach", "--force", sha)
                _git(probe_dir, "clean", "-fdq", "-e", ".spiral")
                verdict = bool(predicate(probe_dir))
                outcome.log.append(f"{sha[:10]} {'good' if verdict else 'bad'}")
                return verdict

            if healthy(commits[0]):
                outcome.detail = "HEAD already passes the probe; nothing to heal"
                return outcome
            if len(commits) < 2:
                outcome.detail = "not enough history to search"
                return outcome

            # walk back for the most recent good commit — doubling steps, CLAMPED
            # to the oldest commit. Unclamped doubling (1,2,4,8,16,32…) jumps
            # straight over a good commit sitting at index 21 of 22 and concludes
            # the fault predates the run when it does not.
            good_index = None
            step, index = 1, 1
            while True:
                index = min(index, len(commits) - 1)
                if healthy(commits[index]):
                    good_index = index
                    break
                if index == len(commits) - 1:
                    break
                index += step
                step *= 2
            if good_index is None:
                outcome.detail = (
                    f"no passing commit within the last {len(commits)} — the fault "
                    "predates this run, so reverting cannot fix it")
                return outcome

            # binary search the first bad commit in (good_index, 0]
            lo, hi = 0, good_index          # commits[lo] bad, commits[hi] good
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if healthy(commits[mid]):
                    hi = mid
                else:
                    lo = mid
            guilty = commits[lo]
            outcome.guilty = guilty
        finally:
            _git(root, "worktree", "remove", "--force", str(probe_dir))

    reverted = _git(root, "revert", "--no-edit", guilty)
    if reverted.returncode != 0:
        _git(root, "revert", "--abort")
        outcome.detail = (f"revert of {guilty[:10]} does not apply cleanly; "
                          "leaving it to remediation")
        return outcome
    if not predicate(root):
        # the revert landed but the product is still broken — undo it entirely
        undo = _git(root, "reset", "--hard", "HEAD~1")
        outcome.detail = (
            "reverting the guilty commit did not restore health"
            + ("" if undo.returncode == 0 else "; and undoing the revert failed"))
        return outcome
    outcome.healed = True
    outcome.reverted_as = _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    outcome.detail = (f"reverted {guilty[:10]} after {outcome.probed} probe(s); "
                      "the requirement it was chasing re-opens at next validation")
    return outcome


def runtime_predicate(workspace: str | Path) -> Predicate:
    """The finish probe as a predicate: healthy means no runtime issues.

    A probe that cannot run reads as HEALTHY on purpose: if playwright is missing,
    every commit would look broken, the walk-back would find no good commit, and
    heal() would refuse — correct, but noisy. Treating 'unknowable' as 'no
    evidence of breakage' keeps the fast path quiet exactly when it has no
    information to act on.
    """
    def check(tree: Path) -> bool:
        from spiral.uicheck import probe

        issues, _note = probe(tree)
        return not [i for i in issues if i["severity"] == "major"]
    return check


__all__ = ["heal", "Healing", "runtime_predicate"]
