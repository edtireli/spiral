"""Interface contracts between tasks: what a task promises, and who may rely on it.

A plan made of files and prose lets layers drift. The trial build was the clean
example — milestone 3 wrote FastAPI routes against ``app.db.create_group``, a
module and four functions that milestones 1 and 2 never produced. Every gate was
green, because nothing in the pipeline knew the routes were promised an interface
that did not exist.

So a task declares what it will export and what it relies on, and both halves are
checked without asking a model anything:

* at plan time — every ``import`` must be exported by an EARLIER task or already
  exist in the repo, so an ordering mistake is a plan defect, not a runtime
  surprise;
* at task time — the task's declared ``exports`` must actually be there before it
  counts as done, exactly as its declared files must.

A reference is one of:

    app.database:create_group     a module-level symbol
    GET /groups/{id}              an HTTP route
    app/static/index.html         a file

Anything unparseable is reported at plan time and never silently enforced: a
promise we cannot check must not become a gate that cannot be satisfied.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from spiral.ladder import SKIP_DIRS

_ROUTE = re.compile(
    r"^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/\S*)$", re.I)
_SYMBOL = re.compile(r"^([A-Za-z_][\w.]*):([A-Za-z_]\w*)$")
_DECORATED_ROUTE = re.compile(
    r"@\s*[\w.]+\.(get|post|put|patch|delete|head|options)\s*\(\s*[\"']([^\"']+)",
    re.I)
_FLASK_ROUTE = re.compile(
    r"@\s*[\w.]+\.route\s*\(\s*[\"']([^\"']+)[\"']([^)]*)\)", re.S)
_METHODS = re.compile(r"[\"'](GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)[\"']", re.I)
_JS_EXT = re.compile(r"\.(js|mjs|cjs|ts|tsx|jsx)$", re.I)
_PY_EXT = re.compile(r"\.(py|pyi)$", re.I)
_PATH_SYMBOL = re.compile(
    r"^([^\s:]+\.[A-Za-z][A-Za-z0-9]{0,5}):([A-Za-z_$][\w$]*)$")
_DOTTED = re.compile(r"^([A-Za-z_][\w.]*)\.([A-Za-z_]\w*)$")


def _js_declares(text: str, name: str) -> bool:
    """Whether a JS/TS source declares ``name`` at any reasonable granularity."""
    pattern = (
        r"(?:^|\b)(?:export\s+(?:default\s+)?)?"
        r"(?:class|function|async\s+function)\s+" + re.escape(name) + r"\b"
        r"|(?:^|\b)(?:const|let|var)\s+" + re.escape(name) + r"\s*="
        r"|\bexports\.(?:" + re.escape(name) + r")\s*="
    )
    return re.search(pattern, text, re.M) is not None


@dataclass(frozen=True)
class Ref:
    kind: str            # "symbol" | "dottedsymbol" | "route" | "file"
                         # | "jssymbol" | "filesymbol" | "unknown"
    raw: str
    module: str = ""
    name: str = ""
    method: str = ""
    path: str = ""

    def __str__(self) -> str:
        return self.raw


def parse_ref(text: str) -> Ref:
    raw = str(text).strip()
    if not raw:
        return Ref("unknown", raw)
    route = _ROUTE.match(raw)
    if route:
        return Ref("route", raw, method=route.group(1).upper(),
                   path=route.group(2))
    symbol = _SYMBOL.match(raw)
    if symbol:
        module = symbol.group(1)
        # `app.js:Calculator` is a JS class in a FILE, not a Python module path —
        # resolving it as Python made the export unsatisfiable, and the task
        # burned its entire budget against a contract nothing could ever meet
        if _JS_EXT.search(module):
            return Ref("jssymbol", raw, path=module, name=symbol.group(2))
        # `main.py:run` is that same mistake in Python and was left in place: the
        # module probe turned it into main/py.py, src/main/py.py, main/py/__init__.py,
        # so a single-file project could never satisfy its own export gate
        if _PY_EXT.search(module):
            return Ref("filesymbol", raw, path=module, name=symbol.group(2))
        return Ref("symbol", raw, module=module, name=symbol.group(2))
    # `app/main.py:create_app` — a PATH, so no dotted-module reading applies and
    # any extension is unambiguous here. It parsed as nothing, and an unparsed
    # export is neither satisfied nor missing, so the gate passed on an interface
    # nobody had written
    path_symbol = _PATH_SYMBOL.match(raw)
    if path_symbol and ".." not in Path(path_symbol.group(1)).parts:
        kind = "jssymbol" if _JS_EXT.search(path_symbol.group(1)) else "filesymbol"
        return Ref(kind, raw, path=path_symbol.group(1), name=path_symbol.group(2))
    # a file ref needs a real extension, not a slash — `style.css` at the repo
    # root is exactly the export a web task should declare, and requiring a "/"
    # rejected it as unverifiable, flooding the critic with false defects
    if (" " not in raw and re.match(r"[^\s:]+\.[A-Za-z][A-Za-z0-9]{0,5}$", raw)
            and ".." not in Path(raw).parts):
        return Ref("file", raw, path=raw)
    # `app.database.create_group` is the colon dropped, the other near-miss that
    # sailed through the gate unchecked. It reads as a symbol OR as a module of
    # that name, so resolve() accepts either — guessing only one reading would
    # trade a silent pass for a gate the task cannot satisfy
    dotted = _DOTTED.match(raw)
    if dotted:
        return Ref("dottedsymbol", raw, module=dotted.group(1),
                   name=dotted.group(2))
    return Ref("unknown", raw)


def _sources(root: Path) -> list[Path]:
    out = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            out.append(path)
    return sorted(out)


def _module_file(root: Path, module: str) -> Path | None:
    rel = Path(*module.split("."))
    for candidate in (
        root / rel.with_suffix(".py"),
        root / rel / "__init__.py",
        root / "src" / rel.with_suffix(".py"),
        root / "src" / rel / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def module_symbols(path: Path) -> set[str]:
    """Names a module makes available at top level, re-exports included."""
    try:
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _dotted_found(root: Path, raw: str) -> bool:
    """Whether a dotted ref is delivered under EITHER of its two readings — the
    symbol ``create_group`` in ``app.database``, or a module named
    ``app.database.create_group``. A dotted ref is genuinely ambiguous (a short
    last segment even looks like a file extension, so ``app.services.mailer``
    parses as a filename), and picking one reading turns the other into a gate the
    task can never satisfy."""
    dotted = _DOTTED.match(raw)
    if not dotted:
        return False
    if _module_file(root, raw) is not None:
        return True
    module = _module_file(root, dotted.group(1))
    return module is not None and dotted.group(2) in module_symbols(module)


def _declares(path: Path, name: str) -> bool:
    """Whether a source FILE declares ``name``, by the strongest check its
    language allows: Python its AST, JS/TS the declaration grammar above, and any
    other extension a whole-word match — a ref checked weakly still beats a ref
    that silently passes the gate."""
    suffix = path.suffix.lower()
    if suffix in (".py", ".pyi"):
        return name in module_symbols(path)
    text = path.read_text(errors="replace")
    if suffix in (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"):
        return _js_declares(text, name)
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def declared_routes(root: Path) -> set[tuple[str, str]]:
    """(METHOD, path) pairs the code actually registers."""
    routes: set[tuple[str, str]] = set()
    for path in _sources(root):
        text = path.read_text(errors="replace")
        for method, route in _DECORATED_ROUTE.findall(text):
            routes.add((method.upper(), route))
        for route, tail in _FLASK_ROUTE.findall(text):
            methods = [m.upper() for m in _METHODS.findall(tail)] or ["GET"]
            for method in methods:
                routes.add((method, route))
    return routes


def _route_matches(want: str, have: str) -> bool:
    """Compare routes with their parameter names normalised away, so
    ``/groups/{id}`` and ``/groups/{group_id}`` are the same endpoint."""
    def norm(value: str) -> str:
        value = re.sub(r"\{[^}]*\}", "{}", value)
        value = re.sub(r"<[^>]*>", "{}", value)
        value = re.sub(r":\w+", "{}", value)
        return value.rstrip("/") or "/"
    return norm(want) == norm(have)


def resolve(root: Path, refs: list[str]) -> tuple[set[str], set[str]]:
    """Split refs into (satisfied, missing). Unparseable refs count as neither —
    a promise that cannot be checked must not become an unsatisfiable gate."""
    root = Path(root).resolve()
    satisfied: set[str] = set()
    missing: set[str] = set()
    routes = None
    for raw in refs:
        ref = parse_ref(raw)
        if ref.kind == "symbol":
            module = _module_file(root, ref.module)
            if module is not None and ref.name in module_symbols(module):
                satisfied.add(ref.raw)
            else:
                missing.add(ref.raw)
        elif ref.kind == "route":
            if routes is None:
                routes = declared_routes(root)
            if any(method == ref.method and _route_matches(ref.path, path)
                   for method, path in routes):
                satisfied.add(ref.raw)
            else:
                missing.add(ref.raw)
        elif ref.kind == "dottedsymbol":
            (satisfied if _dotted_found(root, ref.raw) else missing).add(ref.raw)
        elif ref.kind in ("jssymbol", "filesymbol"):
            candidates = [root / ref.path, *sorted(root.rglob(Path(ref.path).name))]
            found = False
            for candidate in candidates[:8]:
                if not candidate.is_file():
                    continue
                if any(part in SKIP_DIRS for part in
                       candidate.relative_to(root).parts):
                    continue
                if _declares(candidate, ref.name):
                    found = True
                    break
            (satisfied if found else missing).add(ref.raw)
        elif ref.kind == "file":
            # the file rule claims a dotted symbol ref whose last segment is short
            # enough to pass for an extension (`app.services.mailer`), so the module
            # reading is tried before calling it missing — otherwise the task is
            # asked for a literal file of that name, which it will never create
            found = ((root / ref.path).exists() or _dotted_found(root, ref.raw))
            (satisfied if found else missing).add(ref.raw)
        else:
            # "unknown" only, and deliberately neither: prose cannot be checked,
            # and calling it missing is the unsatisfiable gate this module refuses
            # to build. lint_contracts rejects it at plan time, where it belongs —
            # the forms above exist so that a real ref never lands here
            pass
    return satisfied, missing


def missing_exports(root: Path, refs: list[str]) -> list[str]:
    """Declared exports that are not there yet — a task is not done until they are."""
    return sorted(resolve(root, refs)[1])


def satisfy_hint(raw: str) -> str:
    """How to satisfy a missing export, in the form the resolver will accept.

    'artifacts missing: app.js:bindUI' told the worker WHAT was absent and it
    emitted the same non-applying edit six times; saying HOW closes the loop —
    the resolver's acceptance grammar is precise, so state it.
    """
    ref = parse_ref(raw)
    if ref.kind == "jssymbol":
        return (f"declare `function {ref.name}(...)`, `class {ref.name}`, or "
                f"`const {ref.name} = ...` at top level in {ref.path}")
    if ref.kind == "filesymbol":
        return (f"define `{ref.name}` (def/class/assignment) at top level in "
                f"{ref.path}")
    if ref.kind == "symbol":
        return (f"define `{ref.name}` (def/class/assignment) at module level in "
                f"{ref.module.replace('.', '/')}.py")
    if ref.kind == "dottedsymbol":
        return (f"define `{ref.name}` (def/class/assignment) at module level in "
                f"{ref.module.replace('.', '/')}.py, or create the module "
                f"{ref.raw.replace('.', '/')}.py")
    if ref.kind == "route":
        return f"register a route handler for {ref.method} {ref.path}"
    if ref.kind == "file":
        return f"create the file {ref.path}"
    return ""


def lint_contracts(plan, root: Path) -> list[str]:
    """Plan defects: an import nothing earlier provides, or an unparseable ref.

    Ordering is the whole point. A task may rely on what the repo already has or
    on what an EARLIER task promised; relying on a later task is the drift that
    produced routes calling a module that was never written.
    """
    root = Path(root).resolve()
    defects: list[str] = []
    tasks = [(m, t) for m in plan.milestones for t in m.tasks]

    available: set[str] = set()
    for _milestone, task in tasks:
        for raw in getattr(task, "exports", []) or []:
            if parse_ref(raw).kind == "unknown":
                defects.append(
                    f"task {task.title[:48]!r} declares an export "
                    f"{raw!r} that is not a module:symbol, a 'GET /path', or a "
                    "file path — it cannot be verified, so it cannot be promised")

    already, _ = resolve(root, [
        raw for _m, t in tasks for raw in (getattr(t, "imports", []) or [])])
    available |= already

    for _milestone, task in tasks:
        for raw in getattr(task, "imports", []) or []:
            ref = parse_ref(raw)
            if ref.kind == "unknown" or raw in available:
                continue
            provider = next(
                (t.title for _m, t in tasks
                 if raw in (getattr(t, "exports", []) or [])), "")
            if provider:
                defects.append(
                    f"task {task.title[:48]!r} relies on {raw!r}, which is only "
                    f"exported later by {provider[:48]!r} — reorder them or move "
                    "the export earlier")
            else:
                defects.append(
                    f"task {task.title[:48]!r} relies on {raw!r}, which no task "
                    "exports and the repo does not contain — some task has to "
                    "promise it")
        available |= set(getattr(task, "exports", []) or [])
    return defects


__all__ = [
    "Ref", "parse_ref", "resolve", "missing_exports", "satisfy_hint",
    "lint_contracts",
    "module_symbols", "declared_routes",
]
