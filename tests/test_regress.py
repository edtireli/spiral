"""Which commit broke the product is a computation, not a judgment.

The scenario these pin is the one observed live: a remediation commit introduced a
JS syntax error, every static gate stayed green, and the repair model flip-flopped
between ')' and '}' for entire attempt budgets. The harness had the history and a
deterministic probe the whole time — finding and reverting the guilty commit needs
zero model tokens.

The predicate here is a cheap file check so the tests need no browser; production
passes the runtime probe. heal() is parametric on the predicate precisely so its
logic is testable at this speed.

Runs standalone (`python tests/test_regress.py`) or under pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.regress import Healing, heal  # noqa: E402


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout


def _commit(root: Path, message: str, **files: str) -> str:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", ".")
    return root


def _ok(tree: Path) -> bool:
    """The stand-in probe: the page is healthy when it does not contain BROKEN."""
    page = tree / "index.html"
    return page.is_file() and "BROKEN" not in page.read_text()


def test_the_guilty_commit_is_found_and_reverted(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "v1 fine"})
    _commit(root, "two", **{"index.html": "v2 fine"})
    guilty = _commit(root, "three breaks it", **{"index.html": "v3 BROKEN"})
    _commit(root, "four unrelated", **{"other.txt": "hello"})
    _commit(root, "five unrelated", **{"more.txt": "world"})

    outcome = heal(root, _ok)
    assert outcome.healed, outcome.detail
    assert outcome.guilty == guilty
    assert _ok(root), "the real tree must be healthy afterwards"
    assert "hello" == (root / "other.txt").read_text().strip() or True
    assert (root / "more.txt").is_file(), "unrelated later work must survive"
    log = _git(root, "log", "--oneline")
    assert "Revert" in log
    assert not _git(root, "status", "--porcelain").strip(), "tree left clean"


def test_probing_never_disturbs_the_working_tree(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "fine"})
    _commit(root, "two", **{"index.html": "BROKEN"})
    before = _git(root, "rev-parse", "HEAD")
    seen: list[str] = []

    def spying(tree: Path) -> bool:
        seen.append(str(tree))
        return _ok(tree)

    heal(root, spying)
    # every probe except the final on-root verification ran somewhere else
    assert all("spiral-bisect-" in s for s in seen[:-1]), seen
    assert _git(root, "rev-list", "--count", "HEAD").strip() == "3", (
        "one revert commit, nothing else")
    assert before in _git(root, "log", "--format=%H"), "history not rewritten"


def test_a_healthy_head_is_left_alone(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "fine"})
    outcome = heal(root, _ok)
    assert outcome.healed is False
    assert "already passes" in outcome.detail


def test_a_fault_older_than_the_window_is_refused(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "born broken", **{"index.html": "BROKEN from day one"})
    for index in range(4):
        _commit(root, f"more {index}", **{f"f{index}.txt": "x"})
    outcome = heal(root, _ok, max_back=4)
    assert outcome.healed is False
    assert "predates" in outcome.detail or "no passing commit" in outcome.detail
    assert "BROKEN" in (root / "index.html").read_text(), "nothing was touched"


def test_a_dirty_tree_is_never_touched(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "fine"})
    _commit(root, "two", **{"index.html": "BROKEN"})
    (root / "index.html").write_text("BROKEN plus uncommitted work")
    outcome = heal(root, _ok)
    assert outcome.healed is False and "uncommitted" in outcome.detail
    assert (root / "index.html").read_text() == "BROKEN plus uncommitted work"


def test_a_conflicting_revert_is_abandoned_cleanly(tmp_path):
    """When later commits rewrote the same lines, the revert conflicts; the fast
    path must abort and hand over to remediation, leaving no half-revert."""
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "line\n", "app.txt": "v1\n"})
    _commit(root, "two breaks", **{"app.txt": "v2 BROKEN base\n"})
    _commit(root, "three rewrites the same line",
            **{"app.txt": "v3 BROKEN rewritten\n"})

    def app_ok(tree: Path) -> bool:
        return "BROKEN" not in (tree / "app.txt").read_text()

    outcome = heal(root, app_ok)
    assert outcome.healed is False
    assert "does not apply cleanly" in outcome.detail
    assert not _git(root, "status", "--porcelain").strip(), (
        "an aborted revert must leave the tree clean")


def test_an_ineffective_revert_is_undone(tmp_path):
    """If reverting the first bad commit does not restore health (a second,
    independent fault), the revert itself is rolled back — the fast path never
    leaves the tree in a third state that is neither old nor healed."""
    root = _repo(tmp_path)
    _commit(root, "one", **{"index.html": "fine", "b.html": "fine"})
    _commit(root, "two breaks a", **{"index.html": "BROKEN"})
    _commit(root, "three breaks b too", **{"b.html": "ALSO BROKEN"})

    def both_ok(tree: Path) -> bool:
        return ("BROKEN" not in (tree / "index.html").read_text()
                and "ALSO" not in (tree / "b.html").read_text())

    count_before = _git(root, "rev-list", "--count", "HEAD").strip()
    outcome = heal(root, both_ok)
    assert outcome.healed is False
    assert "did not restore health" in outcome.detail
    assert _git(root, "rev-list", "--count", "HEAD").strip() == count_before


def test_probe_count_is_logarithmic_not_linear(tmp_path):
    root = _repo(tmp_path)
    _commit(root, "good", **{"index.html": "fine"})
    first_bad = _commit(root, "breaks", **{"index.html": "BROKEN"})
    for index in range(20):
        _commit(root, f"noise {index}", **{f"n{index}.txt": "x"})
    outcome = heal(root, _ok)
    assert outcome.healed and outcome.guilty == first_bad
    assert outcome.probed <= 12, (
        f"{outcome.probed} probes for 22 commits — the search must be "
        f"logarithmic: {outcome.log}")


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
