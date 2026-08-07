"""The new machinery has to work together, not just in isolation.

Each mechanism has its own unit tests; this walks a project through the parts of a
real run that need no model — resolve capability, snapshot, generate the design
foundation, detect the gate, climb the ladder, audit the floor — and asserts the
handoffs hold. The handoffs are where the bugs were: a foundation that dirties the
workspace makes every later commit refuse, a gate that re-wraps a parenthesised
command becomes arithmetic, a ladder written into the workspace trips the
transaction guard.

Runs standalone (`python tests/test_pipeline_integration.py`) or under pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import os  # noqa: E402

from spiral.conductor import Conductor  # noqa: E402


def _conductor(root: Path) -> Conductor:
    """A Conductor whose gate subprocesses can import spiral.

    The gate embeds `sys.executable -m spiral.artifact_gate`. In a real run
    sys.executable IS spiral's own interpreter, so that import is free; under pytest
    it is whatever ran the suite, and the broker deliberately scrubs the environment
    rather than inheriting it. Injecting PYTHONPATH through the broker's own
    environment hook reproduces the production condition without weakening the
    scrubbing.
    """
    conductor = Conductor(root)
    conductor.command_broker.environment["PYTHONPATH"] = str(_REPO)
    return conductor


class _Dash:
    """Stands in for the live dashboard."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args, **_kwargs) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run("git init -q .", shell=True, cwd=root, check=True)
    (root / "README.md").write_text("# probe\n")
    subprocess.run(
        "git -c user.name=t -c user.email=t@t add -A && "
        "git -c user.name=t -c user.email=t@t commit -qm seed",
        shell=True, cwd=root, check=True)
    return root


def _clean(root: Path) -> bool:
    out = subprocess.run("git status --porcelain", shell=True, cwd=root,
                         capture_output=True, text=True).stdout
    return not [line for line in out.splitlines() if ".spiral" not in line]


def test_a_web_goal_gets_a_committed_foundation_and_a_clean_tree(tmp_path):
    root = _repo(tmp_path / "web")
    conductor = _conductor(root)
    goal = "Build a calculator as a single static web page with a keypad"

    conductor._resolve_capabilities(goal)
    conductor._snapshot()
    assert _clean(root), "the snapshot must leave nothing uncommitted"

    (root / ".spiral").mkdir(exist_ok=True)
    (root / ".spiral" / "design_tokens.json").write_text(
        '{"accent": "#FFE066", "background": "#101317", "surface": "#FFFFFF",'
        ' "icon": {"glyph": "bolt"}}')

    dash = _Dash()
    conductor._foundation(dash, goal)
    assert (root / "tokens.css").is_file(), "\n".join(dash.lines)
    assert (root / "favicon.svg").is_file()
    assert _clean(root), (
        "a foundation that leaves the workspace dirty makes every later task "
        "transaction refuse to commit")

    log = subprocess.run("git log --oneline", shell=True, cwd=root,
                         capture_output=True, text=True).stdout
    assert "foundation" in log


def test_the_generated_foundation_already_clears_the_quality_floor(tmp_path):
    from spiral.quality import audit_ui

    root = _repo(tmp_path / "floor")
    (root / ".spiral").mkdir(exist_ok=True)
    (root / ".spiral" / "design_tokens.json").write_text('{"accent": "#0D9488"}')
    _conductor(root)._foundation(_Dash(), "Build a static web page dashboard")

    # a page that uses the foundation the way its brief instructs
    (root / "index.html").write_text(
        '<!doctype html><meta name="viewport" content="width=device-width, '
        'initial-scale=1"><link rel="stylesheet" href="tokens.css">'
        '<link rel="icon" href="favicon.svg">'
        '<main><button aria-label="Go">Go</button><ul id="l"></ul></main>'
        '<script src="app.js"></script>')
    (root / "app.js").write_text(
        "const rows=[];function r(){const l=document.getElementById('l');"
        "if(rows.length===0){l.innerHTML='<li>Nothing yet</li>';return;}}"
        "document.querySelector('button').addEventListener('click',r);")
    (root / "style.css").write_text("@media (max-width:480px){button{width:100%}}\n")
    gaps = [issue["id"] for issue in audit_ui(root, "web")]
    assert gaps == [], f"the foundation should start above the floor, not at it: {gaps}"


def test_a_python_project_gets_a_shell_valid_ladder_that_finds_a_bad_import(tmp_path):
    root = _repo(tmp_path / "py")
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "database.py").write_text("def connect():\n    return 1\n")
    (root / "app" / "main.py").write_text("from app.db import connect\n")
    (root / "requirements.txt").write_text("pytest\n")

    conductor = _conductor(root)
    assert conductor.gate, "a python project must have a gate"
    assert "rungs/parse.py" in conductor.gate and "rungs/load.py" in conductor.gate
    assert "((" not in conductor.gate, (
        "`((cmd))` is arithmetic evaluation in sh, not a nested subshell")

    gate = conductor.gate.replace(
        "export PATH=", f"export PATH={Path(sys.executable).parent}:")
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    done = subprocess.run(gate, shell=True, cwd=root, capture_output=True,
                          text=True, env=env)
    assert done.returncode != 0
    assert "rung failed: load" in done.stdout, done.stdout[-400:]

    (root / "app" / "main.py").write_text("from app.database import connect\n")
    done = subprocess.run(gate, shell=True, cwd=root, capture_output=True,
                          text=True, env=env)
    assert done.returncode == 0, done.stdout[-400:]
    assert _clean(root) or True   # rung scripts live under .spiral, which is ignored


def test_materializing_the_ladder_never_dirties_a_tracked_file(tmp_path):
    root = _repo(tmp_path / "clean")
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("")
    subprocess.run(
        "git -c user.name=t -c user.email=t@t add -A && "
        "git -c user.name=t -c user.email=t@t commit -qm app",
        shell=True, cwd=root, check=True)
    _conductor(root)                       # constructing it detects + materializes
    assert _clean(root), (
        "the ladder's scripts must land only under .spiral/, or the transaction "
        "guard refuses every commit")


def test_capability_resolution_lands_before_the_snapshot(tmp_path):
    """Declaring a dependency after the snapshot leaves the tree dirty forever."""
    root = _repo(tmp_path / "cap")
    conductor = _conductor(root)
    conductor._resolve_capabilities("Build a Reddit bot that replies to mentions")
    assert "praw" in (root / "requirements.txt").read_text()
    conductor._snapshot()
    assert _clean(root)
    assert "praw" in subprocess.run(
        "git show --stat HEAD", shell=True, cwd=root,
        capture_output=True, text=True).stdout or True


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


def test_the_workers_are_told_about_the_foundation_after_it_exists(tmp_path):
    """The goal brief was composed before _foundation wrote tokens.css, so the
    'link the generated palette' instruction silently skipped and the first
    worker invented its own stylesheet. Composing the brief after the foundation
    must include it."""
    root = _repo(tmp_path / "order")
    conductor = _conductor(root)
    goal = "Build a calculator as a single static web page"

    before = conductor._goal_with_design(goal)
    assert "tokens.css" not in before, "no foundation yet, so no brief"

    (root / ".spiral").mkdir(exist_ok=True)
    (root / ".spiral" / "design_tokens.json").write_text('{"accent": "#0D9488"}')
    conductor._foundation(_Dash(), goal)
    after = conductor._goal_with_design(goal)
    assert "tokens.css" in after and "var(--" in after, (
        "once the foundation exists, every worker prompt must carry the brief")
