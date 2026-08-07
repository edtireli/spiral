"""A promise a task cannot keep must not pass as kept.

Pins the failure the trial build shipped: routes written against
``app.db.create_group`` — a module and four functions no task ever produced — with
every gate green, because only declared FILES were checked, never the interface.

Runs standalone (`python tests/test_contracts.py`) or under pytest.
"""
# Run with pytest. There was once a hand-rolled runner below this point, which
# collected globals() from where it sat mid-file, called each test with no
# arguments, and caught only AssertionError. So it silently skipped every test
# defined after it and every test taking a fixture, then printed "N/N passed".
# A runner that reports a pass count over a subset it chose is the vacuous green
# this suite exists to catch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.contracts import (  # noqa: E402
    declared_routes, lint_contracts, missing_exports, module_symbols, parse_ref,
    resolve,
)
from spiral.planner import Milestone, Plan, Task  # noqa: E402


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


def test_ref_kinds():
    assert parse_ref("app.database:create_group").kind == "symbol"
    assert parse_ref("GET /groups/{id}").kind == "route"
    assert parse_ref("get /x").method == "GET"
    assert parse_ref("app/static/index.html").kind == "file"
    assert parse_ref("make it good").kind == "unknown", (
        "prose is not a checkable promise")


def test_symbols_include_definitions_and_re_exports(tmp_path):
    root = _write(tmp_path, {"pkg/__init__.py": "", "pkg/mod.py": (
        "from os import getenv\n"
        "CONST = 1\n"
        "typed: int = 2\n"
        "def fn():\n    pass\n"
        "async def afn():\n    pass\n"
        "class Cls:\n    pass\n"
    )})
    names = module_symbols(root / "pkg" / "mod.py")
    assert {"getenv", "CONST", "typed", "fn", "afn", "Cls"} <= names


def test_the_trial_failure_is_caught(tmp_path):
    """`app.db:create_group` promised, `app/database.py` delivered."""
    root = _write(tmp_path, {
        "app/__init__.py": "",
        "app/database.py": "def get_connection():\n    return None\n",
    })
    assert missing_exports(root, ["app.db:create_group"]) == ["app.db:create_group"]
    assert missing_exports(root, ["app.database:get_connection"]) == []
    assert missing_exports(root, ["app.database:create_group"]) == [
        "app.database:create_group"], "right module, absent symbol, still missing"


def test_routes_are_found_and_compared_without_parameter_names(tmp_path):
    root = _write(tmp_path, {"app/__init__.py": "", "app/main.py": (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/groups/{group_id}')\n"
        "def read(group_id: int):\n    return {}\n"
        "@app.post('/groups')\n"
        "def make():\n    return {}\n"
    )})
    assert declared_routes(root) == {
        ("GET", "/groups/{group_id}"), ("POST", "/groups")}
    assert missing_exports(root, ["GET /groups/{id}"]) == [], (
        "a parameter's name is not part of the endpoint's identity")
    assert missing_exports(root, ["DELETE /groups/{id}"]) == ["DELETE /groups/{id}"]


def test_flask_routes_and_their_methods(tmp_path):
    root = _write(tmp_path, {"app.py": (
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.route('/items', methods=['POST', 'GET'])\n"
        "def items():\n    return ''\n"
        "@app.route('/only')\n"
        "def only():\n    return ''\n"
    )})
    routes = declared_routes(root)
    assert ("POST", "/items") in routes and ("GET", "/items") in routes
    assert ("GET", "/only") in routes, "a bare route defaults to GET"


def test_unverifiable_refs_are_neither_satisfied_nor_missing(tmp_path):
    satisfied, missing = resolve(tmp_path, ["something vague"])
    assert not satisfied and not missing, (
        "a promise that cannot be checked must not become a gate that cannot "
        "be satisfied")


def _plan(*tasks: Task) -> Plan:
    return Plan("", [Milestone("M", list(tasks))])


def test_lint_flags_relying_on_a_later_task(tmp_path):
    root = _write(tmp_path, {"app/__init__.py": ""})
    plan = _plan(
        Task("Routes", "", ["app/main.py"], "", ["R1"],
             exports=["GET /groups"], imports=["app.database:create_group"]),
        Task("Storage", "", ["app/database.py"], "", ["R2"],
             exports=["app.database:create_group"], imports=[]),
    )
    defects = lint_contracts(plan, root)
    assert len(defects) == 1
    assert "only exported later" in defects[0] and "Storage" in defects[0]

    reordered = _plan(*reversed(plan.milestones[0].tasks))
    assert lint_contracts(reordered, root) == [], (
        "in the right order the seam is satisfied")


def test_lint_flags_an_interface_nobody_promises(tmp_path):
    root = _write(tmp_path, {"app/__init__.py": ""})
    plan = _plan(Task("Routes", "", [], "", ["R1"], exports=[],
                      imports=["app.db:create_group"]))
    defects = lint_contracts(plan, root)
    assert len(defects) == 1 and "no task exports" in defects[0]


def test_lint_accepts_what_the_repo_already_has(tmp_path):
    root = _write(tmp_path, {
        "app/__init__.py": "",
        "app/database.py": "def create_group():\n    return 1\n",
    })
    plan = _plan(Task("Routes", "", [], "", ["R1"], exports=[],
                      imports=["app.database:create_group"]))
    assert lint_contracts(plan, root) == []


def test_lint_rejects_an_export_it_could_never_verify(tmp_path):
    plan = _plan(Task("Vibes", "", [], "", ["R1"], exports=["make it beautiful"]))
    defects = lint_contracts(plan, tmp_path)
    assert len(defects) == 1 and "cannot be verified" in defects[0]



def test_a_bare_filename_is_a_verifiable_file_ref(tmp_path):
    """`style.css` at the repo root is exactly the export a web task should
    declare; requiring a '/' rejected it as unverifiable and flooded the critic
    with false defects on every web plan."""
    for raw in ("style.css", "index.html", "README.md", "script.js"):
        assert parse_ref(raw).kind == "file", raw
    assert parse_ref("v1.2beta thing").kind == "unknown", "spaces are not paths"
    assert parse_ref("../../etc/passwd").kind == "unknown", "no escaping refs"

    (tmp_path / "style.css").write_text("body{}")
    assert missing_exports(tmp_path, ["style.css"]) == []
    assert missing_exports(tmp_path, ["missing.css"]) == ["missing.css"]




def test_js_symbols_are_resolved_in_the_file_not_as_python_modules(tmp_path):
    """`app.js:Calculator` is a class in a FILE. Resolving it as the Python module
    `app.js` made the export unsatisfiable, and a live task burned its full budget
    plus the escalation lane against a contract nothing could ever meet."""
    assert parse_ref("app.js:Calculator").kind == "jssymbol"

    _write(tmp_path, {"app.js": "class Calculator {\n  constructor() {}\n}\n"})
    assert missing_exports(tmp_path, ["app.js:Calculator"]) == []
    assert missing_exports(tmp_path, ["app.js:Missing"]) == ["app.js:Missing"]

    _write(tmp_path, {"src/util.js": "export function handleKey(e) {}\n"
                                     "const fmt = (x) => x;\n"})
    assert missing_exports(tmp_path, ["util.js:handleKey"]) == [], (
        "a basename ref must find the file wherever it lives")
    assert missing_exports(tmp_path, ["src/util.js:fmt"]) == []
