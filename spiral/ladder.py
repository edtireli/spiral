"""Verification rungs: a project has to climb, not just compile.

One gate command answers one question, and ``pytest -q || [ $? -eq 5 ]`` answers
"does this work?" with "there are no tests, so yes". That is how a project that
cannot even be imported earns a green commit. A ladder asks the questions in the
order a person would:

    parse  — does every source file parse?
    load   — does the entry point import without exploding?
    run    — does the thing actually start and answer?
    test   — does the suite pass? (and "no tests" stops being an answer once
             test files exist)

Two rules make it worth having. A rung must be able to fail — a check that is
vacuous on an empty project teaches the loop nothing. And a failing rung must say
which rung failed and what that means, because the worker only sees gate output
and "exit 1" is not a diagnosis.

The scripts a rung runs live in ``.spiral/rungs/`` — harness-owned, gitignored,
and therefore invisible to the transaction guard. They are written from constants
here so the project's own interpreter can run them with no dependency on spiral
being installed alongside it.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git", ".spiral", ".venv", "venv", "env", "build", "dist", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "site-packages",
    ".eggs", "migrations",
}


@dataclass(frozen=True)
class Rung:
    """One question, one command, and what a failure means in English."""

    name: str
    command: str
    why: str


def compose(rungs: list[Rung]) -> str:
    """Chain rungs so the first failure names itself and stops the climb.

    ``(cmd) || { echo …; exit 1; } && (next) || …`` — ``&&`` and ``||`` are
    left-associative with equal precedence in sh, so a passing rung skips its
    failure block and falls through to the next one.
    """
    parts = []
    for rung in rungs:
        label = (
            f"spiral rung failed: {rung.name} — {rung.why}"
        ).replace("'", "")
        # exit 126/127 is the RUNTIME failing to start, not the code failing the
        # check — reporting it as the rung's own failure sent a worker chasing a
        # phantom syntax error for six attempts
        broken = (f"spiral rung BROKEN: {rung.name} — its interpreter or tool "
                  "could not run (exit $s); this is an environment fault, not a "
                  "code fault").replace("'", "")
        parts.append(
            f"( {rung.command} ) || {{ s=$?; if [ $s -eq 126 ] || [ $s -eq 127 ]; "
            f"then echo '{broken}'; else echo '{label}'; fi; exit 1; }}")
    return " && ".join(parts)


def _sources(root: Path, suffix: str) -> list[Path]:
    out = []
    for path in root.rglob(f"*{suffix}"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            out.append(path)
    return sorted(out)


# ---- python ------------------------------------------------------------------

# the trailing \s*\( is load-bearing: without it the alternation matched a
# framework name as a PREFIX, so `router = FastAPIRouter(...)`, the bare class
# alias `AppClass = FastAPI` and even `defaults = StarletteConfigDefaults`
# became smoke targets — and the smoke script calling a class as an ASGI app
# reds the run rung with nothing for the worker to fix
_ASGI = re.compile(
    r"^\s*(\w+)\s*=\s*(?:FastAPI|Starlette|Quart|Sanic)\s*\(", re.M)
_WSGI = re.compile(r"^\s*(\w+)\s*=\s*(?:Flask|Bottle)\s*\(", re.M)
_ENTRY_NAMES = ("main.py", "app.py", "server.py", "api.py", "cli.py", "__main__.py")


def _module_path(root: Path, path: Path) -> str | None:
    """Dotted module name for a file, or None if it is not importable as one."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[0] == "src":                 # src-layout: src/pkg/mod.py -> pkg.mod
        parts = parts[1:]
    if not parts:
        return None
    parts[-1] = parts[-1].removesuffix(".py")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def python_entry_modules(root: Path) -> list[str]:
    """Modules worth importing to prove the project loads.

    Entry-point filenames, plus every package ``__init__`` — between them they
    pull in the modules a real run would touch, without importing test files or
    scripts that do work at import time.
    """
    found: list[str] = []
    for path in _sources(root, ".py"):
        name = path.name
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", "test"} or name.startswith("test_"):
            continue
        if name in _ENTRY_NAMES or name == "__init__.py":
            module = _module_path(root, path)
            if module and module not in found:
                found.append(module)
    return found


def python_app_targets(root: Path) -> list[dict]:
    """ASGI/WSGI application objects, as ``{module, attr, kind}`` records."""
    targets: list[dict] = []
    for path in _sources(root, ".py"):
        rel = path.relative_to(root)
        # same exclusion as python_entry_modules: a FastAPI() built as a test
        # fixture is not the project's app, and the smoke script importing a
        # test module outside pytest runs its import-time side effects
        if rel.parts[0] in {"tests", "test"} or path.name.startswith("test_"):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        module = _module_path(root, path)
        if not module:
            continue
        for kind, pattern in (("asgi", _ASGI), ("wsgi", _WSGI)):
            for match in pattern.finditer(text):
                targets.append(
                    {"module": module, "attr": match.group(1), "kind": kind})
    return targets


def has_python_tests(root: Path) -> bool:
    """True once a real test file exists — after which 'no tests' is not a pass."""
    for path in _sources(root, ".py"):
        rel = path.relative_to(root)
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            return True
        if rel.parts[0] in {"tests", "test"} and path.name != "__init__.py":
            if path.name != "conftest.py":
                return True
    return False


PARSE_SCRIPT = '''\
"""Parse every source file. Written by spiral; do not edit."""
import ast, json, pathlib, sys

root = pathlib.Path(".").resolve()
skip = set(json.loads(pathlib.Path(".spiral/rungs/targets.json").read_text())["skip"])
bad = []
for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root)
    if any(part in skip for part in rel.parts):
        continue
    try:
        ast.parse(path.read_text(errors="replace"), filename=str(rel))
    except SyntaxError as exc:
        bad.append(f"{rel}:{exc.lineno}: {type(exc).__name__}: {exc.msg}")
for line in bad:
    print(line)
sys.exit(1 if bad else 0)
'''

LOAD_SCRIPT = '''\
"""Import the entry points. Written by spiral; do not edit."""
import importlib, json, pathlib, sys, traceback

sys.path.insert(0, ".")
sys.path.insert(0, "src")
cfg = json.loads(pathlib.Path(".spiral/rungs/targets.json").read_text())
failed = 0
for module in cfg["modules"]:
    try:
        importlib.import_module(module)
    except BaseException as exc:
        failed += 1
        print(f"import {module} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
sys.exit(1 if failed else 0)
'''

SMOKE_SCRIPT = '''\
"""Start the app and ask it for a page. Written by spiral; do not edit.

Drives the ASGI/WSGI callable in-process — no http client, no server, no extra
dependency — so the rung works before the project has picked a test client.
"""
import asyncio, importlib, json, pathlib, sys, traceback

sys.path.insert(0, ".")
sys.path.insert(0, "src")
cfg = json.loads(pathlib.Path(".spiral/rungs/targets.json").read_text())
paths = cfg.get("probe_paths") or ["/"]


async def drive_asgi(app, path):
    """Run lifespan startup, one GET, then shutdown. Returns (status, note)."""
    sent, queue, task = [], asyncio.Queue(), None
    try:
        async def ls_receive():
            return await queue.get()

        async def ls_send(message):
            sent.append(message)

        task = asyncio.ensure_future(
            app({"type": "lifespan", "asgi": {"version": "3.0"}}, ls_receive, ls_send))
        await queue.put({"type": "lifespan.startup"})
        for _ in range(300):
            if any(m["type"].startswith("lifespan.startup.") for m in sent):
                break
            await asyncio.sleep(0.01)
        if any(m["type"] == "lifespan.startup.failed" for m in sent):
            detail = next((m.get("message", "") for m in sent
                           if m["type"] == "lifespan.startup.failed"), "")
            return None, f"startup failed: {detail[:300]}"
    except Exception:
        task = None                      # app does not speak lifespan; that is fine

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 12345), "server": ("testserver", 80),
    }
    try:
        await app(scope, receive, send)
    finally:
        if task is not None:
            await queue.put({"type": "lifespan.shutdown"})
            try:
                await asyncio.wait_for(asyncio.shield(task), 5)
            except Exception:
                task.cancel()
    status = next((m.get("status") for m in messages
                   if m.get("type") == "http.response.start"), None)
    return status, ""


def drive_wsgi(app, path):
    import io
    environ = {
        "REQUEST_METHOD": "GET", "PATH_INFO": path, "QUERY_STRING": "",
        "SERVER_NAME": "testserver", "SERVER_PORT": "80",
        "SERVER_PROTOCOL": "HTTP/1.1", "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(b""), "wsgi.errors": sys.stderr,
        "wsgi.multithread": False, "wsgi.multiprocess": False,
        "wsgi.run_once": False, "wsgi.version": (1, 0),
    }
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = int(str(status).split()[0])
        return lambda data: None

    body = app(environ, start_response)
    if hasattr(body, "close"):
        body.close()
    return captured.get("status"), ""


failed = 0
for target in cfg.get("apps", []):
    label = f"{target['module']}:{target['attr']}"
    try:
        app = getattr(importlib.import_module(target["module"]), target["attr"])
    except BaseException as exc:
        failed += 1
        print(f"{label}: could not be loaded: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        continue
    # A 404 is an answer — an API with only /api/* routes is not broken because
    # / is unrouted. A 5xx or an exception is NOT an answer, and a later path
    # answering cleanly must never launder it, so errors are collected and fail
    # the rung even if a subsequent probe succeeds.
    errors, answered = [], False
    for path in paths:
        try:
            if target["kind"] == "asgi":
                status, note = asyncio.run(drive_asgi(app, path))
            else:
                status, note = drive_wsgi(app, path)
        except BaseException as exc:
            errors.append(f"GET {path} raised {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=6)
            continue
        if note:
            errors.append(f"GET {path}: {note}")
            continue
        if status is None:
            errors.append(f"GET {path}: the app returned no response at all")
            continue
        if status >= 500:
            errors.append(f"GET {path}: HTTP {status} — up but erroring")
            continue
        print(f"{label} GET {path}: HTTP {status}")
        answered = True
        break
    if errors or not answered:
        failed += 1
        for line in errors:
            print(f"{label} {line}")
        if not answered and not errors:
            print(f"{label}: never answered a request on any of {paths}")
sys.exit(1 if failed else 0)
'''


# ---- web --------------------------------------------------------------------

JS_PARSE_SCRIPT = r"""// Parse every script the page will run. Written by spiral; do not edit.
const child_process = require("child_process");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const skip = new Set(JSON.parse(fs.readFileSync(".spiral/rungs/targets.json", "utf8")).skip);
const bad = [];

function walk(dir) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return []; }
  const out = [];
  for (const entry of entries) {
    if (skip.has(entry.name) || entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(full));
    else out.push(full);
  }
  return out;
}

// A statement-level `export` or `import` is the only thing that cannot appear
// in a classic script. `import(` and `import.meta` are legal in one, so neither
// counts as evidence.
const MODULE_SYNTAX = /^[ \t]*(?:export\b|import\s+[^(.]|import\s*[{*"'])/m;

function classicError(source, label) {
  try {
    new vm.Script(source, { filename: label });
  } catch (err) {
    if (err instanceof SyntaxError) return err.message;
  }
  return null;
}

// vm.Script always compiles a CLASSIC script, so it reports every module as
// "Unexpected token 'export'" — a syntax error that does not exist, which a
// worker will then spend its whole budget failing to fix. vm.SourceTextModule
// needs --experimental-vm-modules; `--input-type=module --check` is stable, so
// the module grammar comes from a real node parse in a child process.
function moduleError(source) {
  const done = child_process.spawnSync(
    process.execPath, ["--input-type=module", "--check"],
    { input: source, encoding: "utf8" });
  if (done.error || done.status === null) {
    // Never silently pass: an unrunnable check is worse than no check, so the
    // file is reported rather than let through unexamined.
    return `module syntax could not be checked (${done.error || "node produced no exit status"})`;
  }
  if (done.status === 0) return null;
  const line = (done.stderr || "").split("\n").find((l) => /^\w*Error: /.test(l));
  return line ? line.trim() : `module syntax error (node --check exited ${done.status})`;
}

function check(source, label, isModule) {
  if (!isModule) {
    const err = classicError(source, label);
    if (err === null) return;
    // Only a file that could BE a module gets the second chance: module code is
    // not a superset of script code (top-level await is valid in one and broken
    // in the other), so retrying everything would launder real breaks.
    if (!MODULE_SYNTAX.test(source)) {
      bad.push(`${label}: ${err}`);
      return;
    }
  }
  const err = moduleError(source);
  if (err !== null) bad.push(`${label}: ${err}`);
}

for (const file of walk(".")) {
  const lower = file.toLowerCase();
  if (lower.endsWith(".js") || lower.endsWith(".mjs") || lower.endsWith(".cjs")) {
    // .mjs is a module by definition and .cjs is a script by definition; a
    // plain .js is whatever it turns out to parse as
    check(fs.readFileSync(file, "utf8"), file, lower.endsWith(".mjs"));
  } else if (lower.endsWith(".html") || lower.endsWith(".htm")) {
    const html = fs.readFileSync(file, "utf8");
    const pattern = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
    let match;
    let index = 0;
    while ((match = pattern.exec(html)) !== null) {
      index += 1;
      const attrs = match[1] || "";
      if (/\bsrc\s*=/i.test(attrs)) continue;                 // external, checked above
      if (/type\s*=\s*["']?(application|text)\/(json|template)/i.test(attrs)) continue;
      const isModule = /type\s*=\s*["']?module\b/i.test(attrs);
      check(match[2], `${file} <script> #${index}`, isModule);
    }
  }
}

for (const line of bad) console.log(line);
process.exit(bad.length ? 1 : 0);
"""


def has_web_sources(root: Path) -> bool:
    return bool(_sources(root, ".html") or _sources(root, ".js"))


def web_rungs(root: Path) -> list[Rung]:
    """A parse rung for the scripts a page actually runs.

    The artifact gate validates HTML structure and never looks inside a
    ``<script>``, so a page whose JavaScript does not parse reports "9 verified, 0
    errors" and earns a commit. Only opening it in a browser found that — which is
    far too late and far too slow to be the first line of defence.

    Returns nothing when node is unavailable: a rung that cannot run must not exist,
    because a check that always passes is worse than no check.
    """
    root = Path(root).resolve()
    node = shutil.which("node")
    if not node or not has_web_sources(root):
        return []
    out = root / ".spiral" / "rungs"
    out.mkdir(parents=True, exist_ok=True)
    targets = out / "targets.json"
    if not targets.is_file():
        targets.write_text(json.dumps({"skip": sorted(SKIP_DIRS)}, indent=2))
    script = out / "jsparse.cjs"
    if not script.is_file() or script.read_text() != JS_PARSE_SCRIPT:
        script.write_text(JS_PARSE_SCRIPT)
    return [Rung(
        "script-parse", f"{shlex.quote(node)} .spiral/rungs/jsparse.cjs",
        "a script the page runs does not parse, so it never executes at all",
    )]


# ---- behaviour, per ecosystem ---------------------------------------------------

# `--help` is supposed to print and exit; a __main__ that ignores argv and starts
# serving instead used to run until the broker's verify_timeout (900s), so every
# task on such a project cost fifteen minutes and reported "timeout" rather than
# a rung name.
CLI_HELP_TIMEOUT_S = 20.0

CLI_HELP_SCRIPT = '''\
"""Ask a CLI package for its own help. Written by spiral; do not edit."""
import subprocess, sys

package = sys.argv[1]
limit = float(sys.argv[2])
try:
    done = subprocess.run(
        [sys.executable, "-m", package, "--help"],
        capture_output=True, text=True, timeout=limit)
except subprocess.TimeoutExpired:
    print(f"`python -m {package} --help` did not exit within {limit:g}s — it "
          "ignores --help and keeps running (a server or a loop); --help must "
          "print and return")
    sys.exit(1)
if done.returncode:
    # the whole point of this rung: the worker only sees gate output, so the
    # traceback has to BE gate output. Discarding it left a generic label and
    # nothing to act on.
    print(f"`python -m {package} --help` exited {done.returncode}")
    sys.stdout.write(done.stdout)
    sys.stdout.write(done.stderr)
    sys.exit(1)
'''


def behave_rungs(root: Path) -> list[Rung]:
    """Rungs that USE the artifact the way a person would, per ecosystem.

    This is deliberately not a web feature. The principle is one rung shape —
    "drive the thing and fail on what a user would call broken" — instantiated
    per ecosystem, exactly like the parse/load rungs:

      web page      press the buttons; fail on rendered null/undefined/NaN or a
                    throw (spiral.uicheck — hazard sequences included)
      python CLI    every package with a __main__ must answer `--help` with
                    exit 0 and no traceback
      python app    already covered by the ladder's run rung (in-process
                    ASGI/WSGI drive)

    Simulations, notebooks, and training runs get theirs from capability packs
    (a loss that must decrease over N steps is a behave rung too); nothing here
    assumes HTML. Rungs appear only when their runtime is present — a rung that
    cannot run must not exist.
    """
    root = Path(root).resolve()
    rungs: list[Rung] = []

    # web: the runtime probe as a gate check, under spiral's own interpreter
    # (playwright lives there, not in the project venv)
    try:
        from spiral.uicheck import find_page

        has_page = find_page(root) is not None
    except Exception:
        has_page = False
    if has_page:
        try:
            import playwright  # noqa: F401

            rungs.append(Rung(
                "behave-web",
                f"{shlex.quote(sys.executable)} -m spiral.uicheck .",
                "driving the page like a user surfaced a broken value or a "
                "throw; fix the code path it names",
            ))
        except ImportError:
            pass

    # python CLI: a --help that tracebacks is a program that does not start
    mains = [path for path in _sources(root, ".py") if path.name == "__main__.py"]
    packages = []
    for main in mains[:3]:
        module = _module_path(root, main)
        if not module:
            continue
        # pkg/__main__.py resolves to "pkg.__main__"; `python -m pkg` is the
        # form a person types, and the one the packaging actually promises
        packages.append(module.removesuffix(".__main__"))
    if packages:
        out = root / ".spiral" / "rungs"
        out.mkdir(parents=True, exist_ok=True)
        script = out / "clihelp.py"
        if not script.is_file() or script.read_text() != CLI_HELP_SCRIPT:
            script.write_text(CLI_HELP_SCRIPT)
    for package in packages:
        rungs.append(Rung(
            f"behave-cli:{package}",
            f"python .spiral/rungs/clihelp.py {shlex.quote(package)} "
            f"{CLI_HELP_TIMEOUT_S:g}",
            f"`python -m {package} --help` must exit 0; a CLI that cannot "
            "print its own help does not run",
        ))
    return rungs


def materialize(root: Path, probe_paths: list[str] | None = None) -> list[Rung]:
    """Write the rung scripts and return the ladder for a Python project.

    Side-effecting on purpose: the scripts have to live next to the project so
    the project's own interpreter can run them. Everything lands under
    ``.spiral/rungs/``, which is gitignored and skipped by the transaction
    guard, so materializing a ladder can never dirty the workspace.
    """
    root = Path(root).resolve()
    modules = python_entry_modules(root)
    apps = python_app_targets(root)
    out = root / ".spiral" / "rungs"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "skip": sorted(SKIP_DIRS),
        "modules": modules,
        "apps": apps,
        "probe_paths": probe_paths or ["/", "/health", "/docs"],
    }
    for name, body in (
        ("targets.json", json.dumps(payload, indent=2)),
        ("parse.py", PARSE_SCRIPT),
        ("load.py", LOAD_SCRIPT),
        ("smoke.py", SMOKE_SCRIPT),
    ):
        target = out / name
        if not target.is_file() or target.read_text() != body:
            target.write_text(body)

    rungs = [Rung(
        "parse", "python .spiral/rungs/parse.py",
        "a source file does not parse; fix the syntax error above",
    )]
    if modules:
        rungs.append(Rung(
            "load", "python .spiral/rungs/load.py",
            "the entry point cannot be imported; the names it imports must "
            "actually exist in the modules it imports them from",
        ))
    if apps:
        rungs.append(Rung(
            "run", "python .spiral/rungs/smoke.py",
            "the app imports but does not serve a request; it must start and "
            "answer without a server error",
        ))
    # "no tests collected" is a fair pass on a greenfield project and a lie once
    # test files exist — so the tolerance disappears the moment they do.
    tests = ("python -m pytest -q" if has_python_tests(root)
             else "python -m pytest -q || [ $? -eq 5 ]")
    rungs.append(Rung("test", tests, "the test suite does not pass"))
    return rungs


def python_ladder(root: Path, probe_paths: list[str] | None = None) -> str:
    """Composed shell command for the Python ladder."""
    return compose(materialize(root, probe_paths))


def is_python_gate(gate: str) -> bool:
    """Whether a composed gate needs the project's Python venv on PATH."""
    return ".spiral/rungs/" in gate or gate.startswith("python ")


def venv_prefix(venv_bin: Path) -> str:
    """``export`` rather than a one-shot assignment: a ``VAR=x cmd`` prefix only
    reaches the first command of a ``&&`` chain, which would leave every rung
    after the first running against the wrong interpreter.

    Spiral's own interpreter directory rides behind the project venv: before the
    venv exists (a fresh repo's first tasks), a bare ``python`` fell through to
    pyenv shims the sandbox denied, every rung exited 126, and the failure was
    misread as a syntax error for six model attempts. The rung scripts are
    stdlib-only, so ANY working python is correct for them."""
    own_bin = Path(sys.executable).parent
    return (f'export PATH={shlex.quote(str(venv_bin))}:'
            f'{shlex.quote(str(own_bin))}:"$PATH"; ')


__all__ = [
    "Rung", "compose", "materialize", "python_ladder", "python_entry_modules",
    "python_app_targets", "has_python_tests", "is_python_gate", "venv_prefix",
    "web_rungs", "has_web_sources", "behave_rungs",
    "SKIP_DIRS",
]
