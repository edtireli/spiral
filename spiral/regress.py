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
    unusable_probe: bool = False        # the sweep measured nothing; see ``detail``
    log: list[str] = field(default_factory=list)


class _ProbeFailed(RuntimeError):
    """The probe tree could not be moved to a commit, so no verdict names that sha.

    A failed ``git checkout`` still leaves *a* tree behind — sometimes the previous
    commit's, sometimes a half-applied one — and the predicate will judge it without
    complaint. Scoring that as "this commit is bad" attributes a verdict to content
    that was never materialised, so the search stops instead.
    """


def _commits(root: Path, limit: int) -> list[str]:
    """HEAD-first commit shas on the current branch."""
    result = _git(root, "rev-list", "--first-parent", f"--max-count={limit}", "HEAD")
    return result.stdout.split()


def _dirty(root: Path) -> bool:
    out = _git(root, "status", "--porcelain").stdout
    return any(line and ".spiral" not in line for line in out.splitlines())


# Dependency roots a build needs and git never tracks. Narrow on purpose, for the
# reason harness_check._RULES is narrow: a detector that cries "broken instrument"
# over an ignored .DS_Store would bury the honest "the fault predates this run"
# answer under a false excuse. Only directories a gate actually needs to run.
_DEP_ROOTS = {"node_modules", ".venv", "venv", "vendor", "target", ".yarn",
              "bower_components", "Pods", ".gradle", ".m2", ".bundle",
              ".pnpm-store", ".tox", ".nox", ".dart_tool", "deps", "_build"}


def _unmeasurable(root: Path, probe_dir: Path) -> str:
    """Why an all-bad sweep from this probe tree is no evidence at all, or "".

    The probe is a ``git worktree add`` checkout, so it holds exactly the tracked
    files and nothing else. When the workspace's gate needs installed dependencies
    — and gitignored is precisely how dependencies live — every commit shown to it
    reads BAD whatever its code says, including commits known to be good. Observed
    in both ecosystems: the instrument was blind and the report was confident.
    """
    listing = _git(root, "status", "--porcelain", "--ignored").stdout
    ignored = [line[3:].strip().rstrip("/")
               for line in listing.splitlines() if line.startswith("!! ")]
    blind = [path for path in ignored
             if Path(path).name in _DEP_ROOTS
             and (root / path).is_dir() and not (probe_dir / path).exists()]
    if not blind:
        return ""
    return ("the probe worktree has no " + ", ".join(sorted(blind)[:3])
            + " — nothing gitignored survives a worktree checkout, so the gate "
              "reads broken on every commit there whatever its code says")


def heal(workspace: str | Path, predicate: Predicate, *,
         max_back: int = 32) -> Healing:
    """Locate the first commit that made ``predicate`` fail and revert it.

    ``predicate`` runs against a checkout in a temporary worktree; it must be
    deterministic and side-effect free. Returns ``healed=False`` with a reason
    whenever safety would be compromised — a dirty tree, no good commit within
    ``max_back``, a conflicting revert, or a revert that does not actually fix it.

    ``unusable_probe`` separates the two ways a search comes back empty-handed:
    "every commit really is bad" from "nothing here could be measured". Only the
    first is a fact about the history.
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
                moved = _git(probe_dir, "checkout", "--detach", "--force", sha)
                if moved.returncode != 0:
                    outcome.log.append(f"{sha[:10]} unmeasurable")
                    first = (moved.stderr.strip().splitlines() or [""])[0]
                    raise _ProbeFailed(
                        f"the probe worktree could not check out {sha[:10]}, so no "
                        f"verdict belongs to it: {first[:120]}")
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
                # every probe read bad, which has two very different causes: the
                # history really is bad throughout, or the instrument cannot see.
                # Only one of them is a fact about the commits, so check the
                # instrument before stating one.
                blind = _unmeasurable(root, probe_dir)
                if blind:
                    outcome.unusable_probe = True
                    outcome.detail = (
                        f"could not measure any of the last {len(commits)} commits: "
                        f"{blind}. This is not a verdict on the history")
                    return outcome
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
        except _ProbeFailed as unmeasured:
            outcome.unusable_probe = True
            outcome.detail = str(unmeasured)
            return outcome
        finally:
            _git(root, "worktree", "remove", "--force", str(probe_dir))

    reverted = _git(root, "revert", "--no-edit", guilty)
    if reverted.returncode != 0:
        _git(root, "revert", "--abort")
        outcome.detail = (f"revert of {guilty[:10]} does not apply cleanly; "
                          "leaving it to remediation")
        return outcome
    try:
        restored = bool(predicate(root))
    except Exception as exc:
        # the verification RAISED, and the revert is already committed. Letting this
        # propagate left the caller printing "healer unavailable" over a workspace
        # whose HEAD had silently moved to an unverified Revert — worse than any
        # verdict, because nobody knows the tree changed. Roll it back and decline.
        undo = _git(root, "reset", "--hard", "HEAD~1")
        outcome.detail = (
            f"the verification probe raised after the revert landed ({exc}); "
            "the revert was rolled back"
            + ("" if undo.returncode == 0 else "; and rolling it back failed"))
        return outcome
    if not restored:
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
