"""The ladder must be able to fail, and must say which rung failed.

The regressions these pin are the ones that cost real commits: a project whose
entry point cannot be imported reading green because there were no tests yet, and
a `VAR=x cmd` prefix reaching only the first command of a chain.

Runs standalone (`python tests/test_ladder.py`) or under pytest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.ladder import (  # noqa: E402
    Rung, compose, has_python_tests, is_python_gate, materialize,
    python_app_targets, python_entry_modules, python_ladder, venv_prefix,
)


def _project(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


def _run(root: Path) -> subprocess.CompletedProcess:
    command = python_ladder(root).replace("python ", f"{sys.executable} ")
    return subprocess.run(
        command, shell=True, cwd=root, capture_output=True, text=True)


def test_compose_labels_each_rung_and_short_circuits():
    command = compose([
        Rung("parse", "true", "does not parse"),
        Rung("load", "false", "cannot import"),
    ])
    assert "spiral rung failed: parse" in command
    assert "spiral rung failed: load" in command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == 1
    assert "load" in result.stdout and "parse" not in result.stdout, (
        "a passing rung must not report itself; only the failing one speaks")


def test_entry_modules_and_app_targets_are_discovered(tmp_path):
    root = _project(tmp_path, {
        "app/__init__.py": "",
        "app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "app/helpers.py": "X = 1\n",
        "tests/test_thing.py": "def test_x():\n    pass\n",
    })
    modules = python_entry_modules(root)
    assert "app.main" in modules and "app" in modules
    assert not any(m.startswith("tests") for m in modules), (
        "test files are not entry points")
    assert python_app_targets(root) == [
        {"module": "app.main", "attr": "app", "kind": "asgi"}]


def test_src_layout_modules_drop_the_src_prefix(tmp_path):
    root = _project(tmp_path, {
        "src/pkg/__init__.py": "",
        "src/pkg/cli.py": "def main():\n    pass\n",
    })
    assert sorted(python_entry_modules(root)) == ["pkg", "pkg.cli"]


def test_no_tests_is_green_but_a_test_file_removes_the_tolerance(tmp_path):
    root = _project(tmp_path, {"app/__init__.py": "", "app/main.py": "X = 1\n"})
    assert has_python_tests(root) is False
    assert "[ $? -eq 5 ]" in python_ladder(root), (
        "a greenfield project with no tests must not be permanently red")

    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "conftest.py").write_text("")
    assert has_python_tests(root) is False, (
        "__init__ and conftest are plumbing, not tests")

    (root / "tests" / "test_real.py").write_text("def test_x():\n    assert True\n")
    assert has_python_tests(root) is True
    assert "[ $? -eq 5 ]" not in python_ladder(root), (
        "once tests exist, 'no tests collected' is a lie, not a pass")


def test_parse_rung_fails_on_a_syntax_error(tmp_path):
    root = _project(tmp_path, {"app/__init__.py": "", "app/main.py": "def broken(\n"})
    result = _run(root)
    assert result.returncode == 1
    assert "spiral rung failed: parse" in result.stdout
    assert "app/main.py" in result.stdout


def test_load_rung_fails_on_an_import_that_does_not_exist(tmp_path):
    """The exact defect that earned two green commits in the trial build."""
    root = _project(tmp_path, {
        "app/__init__.py": "",
        "app/database.py": "def get_connection():\n    return None\n",
        "app/main.py": "from app.db import create_group\n",
    })
    result = _run(root)
    assert result.returncode == 1
    assert "spiral rung failed: load" in result.stdout
    assert "app.db" in result.stdout + result.stderr

    (root / "app" / "main.py").write_text(
        "from app.database import get_connection\n")
    assert _run(root).returncode == 0, "a correct import must climb past load"


def test_smoke_rung_drives_a_bare_asgi_app_without_an_http_client(tmp_path):
    """No httpx, no server — the rung must work before the project picks a client."""
    root = _project(tmp_path, {
        "app/__init__.py": "",
        "app/main.py": (
            "class App:\n"
            "    async def __call__(self, scope, receive, send):\n"
            "        if scope['type'] == 'lifespan':\n"
            "            while True:\n"
            "                message = await receive()\n"
            "                if message['type'] == 'lifespan.startup':\n"
            "                    await send({'type': 'lifespan.startup.complete'})\n"
            "                else:\n"
            "                    await send({'type': 'lifespan.shutdown.complete'})\n"
            "                    return\n"
            "        await send({'type': 'http.response.start', 'status': 200,\n"
            "                    'headers': []})\n"
            "        await send({'type': 'http.response.body', 'body': b'ok'})\n"
            "\n\n"
            "app = FastAPI()\n"
        ),
    })
    # the name is what the detector keys on; swap the constructor for the stub so
    # the test needs no web framework installed
    main = root / "app" / "main.py"
    main.write_text(main.read_text().replace("app = FastAPI()", "app = App()"))
    assert python_app_targets(root) == [], "no FastAPI() call, so no app target"

    main.write_text(main.read_text().replace("app = App()", "FastAPI = App\napp = FastAPI()"))
    assert python_app_targets(root) == [
        {"module": "app.main", "attr": "app", "kind": "asgi"}]
    result = _run(root)
    assert result.returncode == 0, f"smoke rung should pass: {result.stdout}"
    assert "HTTP 200" in result.stdout


def test_smoke_rung_fails_when_the_app_errors(tmp_path):
    root = _project(tmp_path, {
        "app/__init__.py": "",
        "app/main.py": (
            "class App:\n"
            "    async def __call__(self, scope, receive, send):\n"
            "        if scope['type'] == 'lifespan':\n"
            "            await receive()\n"
            "            await send({'type': 'lifespan.startup.complete'})\n"
            "            await receive()\n"
            "            await send({'type': 'lifespan.shutdown.complete'})\n"
            "            return\n"
            "        raise RuntimeError('boom')\n"
            "\n\n"
            "FastAPI = App\n"
            "app = FastAPI()\n"
        ),
    })
    result = _run(root)
    assert result.returncode == 1
    assert "spiral rung failed: run" in result.stdout, (
        "a 5xx or an exception on one path must not be laundered by a 404 on "
        f"another: {result.stdout}"
    )


def test_rung_scripts_land_only_under_dot_spiral(tmp_path):
    root = _project(tmp_path, {"app/__init__.py": "", "app/main.py": "X = 1\n"})
    before = {p for p in root.rglob("*") if ".spiral" not in p.parts}
    materialize(root)
    after = {p for p in root.rglob("*") if ".spiral" not in p.parts}
    assert before == after, (
        "materializing a ladder must never dirty the workspace — the "
        "transaction guard would refuse every commit")
    assert (root / ".spiral" / "rungs" / "smoke.py").is_file()


def test_venv_prefix_exports_so_every_rung_sees_it():
    prefix = venv_prefix(Path("/tmp/v/bin"))
    assert prefix.startswith("export PATH=")
    assert prefix.rstrip().endswith(";"), (
        "a `VAR=x cmd` prefix reaches only the first command of a && chain")
    assert is_python_gate("( python .spiral/rungs/parse.py ) && ( true )")
    assert is_python_gate("python -m pytest -q")
    assert not is_python_gate("./gradlew test")


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


def _node() -> bool:
    import shutil

    return shutil.which("node") is not None


def test_a_broken_inline_script_reds_the_gate(tmp_path):
    """The artifact gate validates HTML structure and never looks inside a
    <script>, so a page whose JavaScript does not parse reported "9 verified, 0
    errors" and earned a commit. Only the browser caught it, far too late."""
    from spiral.ladder import compose, web_rungs

    if not _node():
        return
    root = _project(tmp_path, {
        "index.html": "<button id=b>Go</button><script>\n"
                      "document.getElementById('b').addEventListener('click', () => {\n"
                      "  console.log('hi');\n"
                      "});)\n</script>",
    })
    rungs = web_rungs(root)
    assert [r.name for r in rungs] == ["script-parse"]
    done = subprocess.run(compose(rungs), shell=True, cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 1
    assert "index.html <script>" in done.stdout
    assert "rung failed: script-parse" in done.stdout

    (root / "index.html").write_text(
        "<button id=b>Go</button><script>\n"
        "document.getElementById('b').addEventListener('click', () => {\n"
        "  console.log('hi');\n"
        "});\n</script>")
    assert subprocess.run(compose(web_rungs(root)), shell=True, cwd=root,
                          capture_output=True, text=True).returncode == 0


def test_external_and_non_script_blocks_are_left_alone(tmp_path):
    """A <script src> is checked as a file; JSON and templates are not JavaScript."""
    from spiral.ladder import compose, web_rungs

    if not _node():
        return
    root = _project(tmp_path, {
        "index.html": '<script src="app.js"></script>'
                      '<script type="application/json">{"a": 1}</script>'
                      '<script type="text/template"><div>{{x}}</div></script>',
        "app.js": "const x = 1;\n",
    })
    assert subprocess.run(compose(web_rungs(root)), shell=True, cwd=root,
                          capture_output=True, text=True).returncode == 0

    (root / "app.js").write_text("const x = ;\n")
    done = subprocess.run(compose(web_rungs(root)), shell=True, cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 1 and "app.js" in done.stdout


def test_no_web_sources_means_no_web_rung(tmp_path):
    from spiral.ladder import has_web_sources, web_rungs

    root = _project(tmp_path, {"app/__init__.py": "", "app/main.py": "X = 1\n"})
    assert has_web_sources(root) is False
    assert web_rungs(root) == [], "a rung that has nothing to check must not exist"


def test_the_web_rung_disappears_without_node(tmp_path, monkeypatch):
    """A check that cannot run must not exist — one that always passes is worse
    than none at all."""
    import spiral.ladder as ladder

    root = _project(tmp_path, {"index.html": "<script>const x = ;</script>"})
    monkeypatch.setattr(ladder.shutil, "which", lambda _name: None)
    assert ladder.web_rungs(root) == []


def test_behave_rungs_are_per_ecosystem_not_web_specific(tmp_path):
    """One rung shape — drive the artifact, fail on what a user would call
    broken — instantiated per ecosystem. A CLI gets a --help rung with no HTML
    anywhere in sight; a page gets the runtime probe; a bare library gets none."""
    from spiral.ladder import behave_rungs

    cli = _project(tmp_path / "cli", {
        "tally/__init__.py": "",
        "tally/__main__.py": "print('hi')\n",
    })
    names = [r.name for r in behave_rungs(cli)]
    assert "behave-cli:tally" in names
    assert not any(n.startswith("behave-web") for n in names)

    lib = _project(tmp_path / "lib", {"pkg/__init__.py": "", "pkg/mod.py": "X=1\n"})
    assert behave_rungs(lib) == [], (
        "a library with no entry point has no behaviour to drive — a rung that "
        "cannot fail must not exist")

    page = _project(tmp_path / "page", {"index.html": "<button>Go</button>"})
    web_names = [r.name for r in behave_rungs(page)]
    try:
        import playwright  # noqa: F401

        assert "behave-web" in web_names
    except ImportError:
        assert web_names == []


def test_the_cli_behave_rung_fails_on_a_tracebacking_help(tmp_path):
    import subprocess
    import sys as _sys

    from spiral.ladder import behave_rungs, compose

    root = _project(tmp_path, {
        "tally/__init__.py": "",
        "tally/__main__.py": "raise RuntimeError('boom')\n",
    })
    rungs = [r for r in behave_rungs(root) if r.name.startswith("behave-cli")]
    command = compose(rungs).replace("python ", f"{_sys.executable} ")
    done = subprocess.run(command, shell=True, cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 1
    assert "behave-cli" in done.stdout

    (root / "tally" / "__main__.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser(prog='tally')\n"
        "parser.parse_args()\n")
    done = subprocess.run(command, shell=True, cwd=root,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-200:]
