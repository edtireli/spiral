"""The v1.5 conductor — orchestrate a whole project from a raw goal, autonomously.

    detect gate → snapshot → plan → reflect → bootstrap to green
      → grind tasks green-to-green (escalate when stuck) → report

Principles:
- GREEN-TO-GREEN: the detected build gate is injected into every task; a task only
  commits if the build passes. Integration debt cannot accumulate silently.
- REFLECTION: the planner critiques its own plan (bounded rounds) before execution.
- ESCALATION: a stuck task retries on the stronger dense model; if still stuck the
  tree reverts to the last green commit and the task is recorded as blocked —
  one wedge never deadlocks the whole run.
- Fully resumable state in .spiral/ (plan.json, state.json).
"""
from __future__ import annotations

import json
import hashlib
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from spiral.theme import CLAY as _CLAY, make_console, reveal
from rich.panel import Panel

from spiral import tools
from spiral.agent import Atom, TaskSpec
from spiral.appicon import TOKEN_COLORS, write_android_icon, write_android_tokens
from spiral.banner import Spinner
from spiral.config import Config
from spiral.llm import Ollama
from spiral.planner import (
    Milestone, Plan, Task, analyze_deliverables, coverage_gaps, critique_plan,
    default_output_globs, design_brief, design_tokens, enrich_deliverable_spec,
    enrich_product_spec,
    ensure_plan_coverage,
    extract_spec, lint_plan, make_plan, normalize_plan_requirements, parse_plan,
    plan_to_dict, repair_plan, sanitize_checks, validate_spec,
)
from spiral.ledger import Ledger
from spiral.repomap import build_relevant_repomap, build_repomap, list_files

CLAY = "rgb(217,119,87)"


@dataclass(frozen=True)
class GateSpec:
    root: Path
    command: str
    ecosystem: str


def _detect_gate_here(ws: Path) -> str:
    """Deterministic build-gate detection. The gate is ground truth; prefer the
    strongest cheap-to-run signal the project offers."""
    if (ws / "gradlew").is_file():
        android = any(
            path.is_file() and not any(
                part.startswith(".") or part in {"build", "node_modules"}
                for part in path.relative_to(ws).parts)
            for path in ws.rglob("AndroidManifest.xml")
        )
        return ("./gradlew testDebugUnitTest assembleDebug" if android
                else "./gradlew test build")
    if ((ws / "build.gradle").is_file()
            or (ws / "build.gradle.kts").is_file()):
        return "gradle test build"
    if (ws / "package.json").is_file():
        try:
            scripts = json.loads((ws / "package.json").read_text()).get("scripts", {})
            checks = [f"CI=1 npm run {key} --silent" for key in (
                "typecheck", "lint", "test", "build") if key in scripts]
            if checks:
                return " && ".join(checks)
        except Exception:
            pass
    if (ws / "mvnw").is_file():
        return "./mvnw test"
    if (ws / "pom.xml").is_file():
        return "mvn test"
    if (ws / "Cargo.toml").is_file():
        return "cargo test --quiet && cargo build --quiet"
    if (ws / "go.mod").is_file():
        return "go test ./... && go build ./..."
    xcode_workspace = next(ws.glob("*.xcworkspace"), None)
    xcode_project = next(ws.glob("*.xcodeproj"), None)
    if xcode_workspace or xcode_project:
        container = xcode_workspace or xcode_project
        selector = (
            f"-workspace {shlex.quote(container.name)}"
            if xcode_workspace else
            f"-project {shlex.quote(container.name)}"
        )
        shared_scheme = next(
            container.rglob("xcshareddata/xcschemes/*.xcscheme"), None)
        scheme = shared_scheme.stem if shared_scheme else container.stem
        quoted_scheme = shlex.quote(scheme)
        return (
            "DEST=$(xcrun simctl list devices available | "
            "awk -F '[()]' '/iPhone/{print $2; exit}'); "
            "if [ -n \"$DEST\" ]; then "
            f"xcodebuild {selector} -scheme {quoted_scheme} "
            "-destination \"platform=iOS Simulator,id=$DEST\" "
            "-derivedDataPath .spiral/xcode-derived-data "
            "CODE_SIGNING_ALLOWED=NO test; "
            "else "
            f"xcodebuild {selector} -scheme {quoted_scheme} "
            "-destination 'generic/platform=iOS Simulator' "
            "-derivedDataPath .spiral/xcode-derived-data "
            "CODE_SIGNING_ALLOWED=NO build; fi"
        )
    if (ws / "Package.swift").is_file():
        return "swift test"
    if (ws / "mix.exs").is_file():
        return "mix test"
    if (ws / "pubspec.yaml").is_file():
        return "dart analyze && dart test"
    if (ws / "Gemfile").is_file():
        if (ws / "Rakefile").is_file():
            return "bundle exec rake test"
        return (
            "find . -name '*.rb' -not -path './vendor/*' -print0 "
            "| xargs -0 -n1 bundle exec ruby -c"
        )
    if (ws / "composer.json").is_file():
        try:
            scripts = json.loads((ws / "composer.json").read_text()).get("scripts", {})
        except Exception:
            scripts = {}
        return "composer test" if "test" in scripts else "composer validate --strict"
    if (ws / "build.zig").is_file():
        return "zig build test"
    if (ws / "meson.build").is_file():
        return (
            "if [ -d .spiral/meson-build ]; then "
            "meson setup .spiral/meson-build --reconfigure; else "
            "meson setup .spiral/meson-build; fi && "
            "meson compile -C .spiral/meson-build && "
            "meson test -C .spiral/meson-build --print-errorlogs"
        )
    if (ws / "MODULE.bazel").is_file() or (ws / "WORKSPACE").is_file():
        return "bazel test //..."
    if (ws / "Project.toml").is_file():
        return "julia --project=. -e 'using Pkg; Pkg.test()'"
    if (ws / "DESCRIPTION").is_file():
        return "R CMD build . && R CMD check --no-manual --no-build-vignettes ."
    if (ws / "stack.yaml").is_file():
        return "stack test"
    if (ws / "cabal.project").is_file() or next(ws.glob("*.cabal"), None):
        return "cabal test all"
    if (ws / "dune-project").is_file():
        return "dune runtest && dune build"
    if (ws / "build.sbt").is_file():
        return "sbt test"
    if (ws / "shard.yml").is_file():
        return "crystal spec"
    if next(ws.glob("*.nimble"), None):
        return "nimble test"
    if next(ws.glob("*.tf"), None):
        return "terraform fmt -check -recursive && terraform validate"
    if (ws / "lakefile.lean").is_file() or (ws / "lakefile.toml").is_file():
        return "lake build"
    if (ws / "CMakeLists.txt").is_file():
        return ("cmake -S . -B .spiral/cmake-build && "
                "cmake --build .spiral/cmake-build && "
                "ctest --test-dir .spiral/cmake-build --output-on-failure")
    if (ws / "Makefile").is_file():
        try:
            has_test = bool(re.search(
                r"(?m)^test\s*:", (ws / "Makefile").read_text(errors="replace")))
        except Exception:
            has_test = False
        return "make test && make" if has_test else "make"
    if next(ws.glob("*.sln"), None) or next(ws.glob("*.csproj"), None):
        return "dotnet test"
    if (ws / "pyproject.toml").is_file() or (ws / "pytest.ini").is_file() or (ws / "tests").is_dir():
        # exit 5 = "no tests collected" — a project with a pyproject but no tests yet
        # is not failing, it just has nothing to assert. Treat 5 as green so an early
        # greenfield project isn't a permanently-red gate; real failures (1) stay red.
        return "python -m pytest -q || [ $? -eq 5 ]"
    return ""


def detect_gates(ws: Path) -> list[GateSpec]:
    """Discover every independently runnable component gate in the workspace."""

    ws = Path(ws).resolve()
    from spiral.builder_tools import discover_project_roots

    rows: list[GateSpec] = []
    seen: set[tuple[str, str]] = set()
    for project_root in discover_project_roots(ws):
        gate = _detect_gate_here(project_root)
        if not gate:
            continue
        if gate.startswith("python "):
            venv_bin = (
                project_root / ".spiral" / "dependency-cache" / "python"
                / "venv" / "bin"
            )
            gate = f"PATH={shlex.quote(str(venv_bin))}:\"$PATH\" {gate}"
        marker = next((
            name for name in (
                "gradlew", "package.json", "mvnw", "pom.xml", "Cargo.toml",
                "go.mod", "Package.swift", "lakefile.lean", "lakefile.toml",
                "CMakeLists.txt", "Makefile", "pyproject.toml", "pytest.ini",
                "mix.exs", "pubspec.yaml", "Gemfile", "composer.json",
                "build.zig", "meson.build", "MODULE.bazel", "WORKSPACE",
                "Project.toml", "DESCRIPTION", "main.tf",
                "build.gradle", "build.gradle.kts", "settings.gradle",
                "stack.yaml", "cabal.project", "dune-project", "build.sbt",
                "deps.edn", "shard.yml",
            ) if (project_root / name).exists()
        ), "project")
        key = (str(project_root), gate)
        if key not in seen:
            rows.append(GateSpec(project_root, gate, marker))
            seen.add(key)

    if not rows:
        from spiral.artifact_gate import verify_workspace

        artifact = verify_workspace(ws)
        meaningful_errors = [
            error for error in artifact.errors
            if error != "no structurally verifiable artifacts were found"
        ]
        if artifact.verified or meaningful_errors:
            rows.append(GateSpec(
                ws,
                f"{shlex.quote(sys.executable)} -m spiral.artifact_gate .",
                "artifact-integrity",
            ))
    return rows[:12]


def _compose_gates(ws: Path, gates: list[GateSpec]) -> str:
    commands = []
    for gate in gates:
        rel = gate.root.relative_to(ws)
        command = gate.command
        if rel != Path("."):
            command = f"cd {shlex.quote(str(rel))} && ({command})"
        commands.append(f"({command})")
    return " && ".join(commands)


def detect_gate(ws: Path) -> str:
    """Compatibility string for the composite workspace verification graph."""

    ws = Path(ws).resolve()
    return _compose_gates(ws, detect_gates(ws))


class Conductor:
    def __init__(self, workspace: str | Path = ".", cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.ws = Path(workspace).resolve()
        self.ol = Ollama(self.cfg.base_url)
        self.c = make_console()
        self._base_gate = ""
        self.gates: list[GateSpec] = []
        self.gate = ""
        self.gate_disp = "none detected"
        self._refresh_gate()
        state_path = self.ws / ".spiral" / "state.json"
        try:
            self.state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        except Exception:
            self.state = {}
        self.ledger = Ledger(self.ws)
        from spiral.toolsmith import Toolsmith

        self.toolsmith = Toolsmith(self.ws)
        from spiral.command_broker import CommandBroker

        self.command_broker = CommandBroker(self.ws, self.cfg)

    def _refresh_gate(self) -> bool:
        """(Re)detect the build gate against the *current* workspace and rebuild the
        composed gate command. Spiral often starts on an empty repo and creates the
        project as it goes (a pyproject.toml / tests dir only appears mid-run), so the
        gate has to be re-detected as files materialise — detecting once at construction
        leaves every task unverified. Returns True when the detected gate changed."""
        gates = detect_gates(self.ws)
        base = _compose_gates(self.ws, gates)
        if base == self._base_gate and (self.gate or not base):
            return False
        self._base_gate = base
        self.gates = gates
        gate = base
        disp = (
            " + ".join(
                f"{g.root.relative_to(self.ws) or Path('.')}:{g.ecosystem}"
                for g in gates
            )
            if gates else "none detected"
        )
        if gate:
            # runtime-footgun linter rides the gate: compiles-fine-crashes-at-runtime
            # patterns get fixed by the same loop as compile errors
            gate = f"({gate}) && ({sys.executable} -m spiral.footguns .)"
            disp += " +footguns"
        if self.cfg.extra_gate:
            # user-defined blocking gate (their linter/tests) — veto power on every task
            gate = f"({gate}) && ({self.cfg.extra_gate})" if gate else self.cfg.extra_gate
            disp += " +extra_gate"
        self.gate, self.gate_disp = gate, disp
        return True

    def _run_verified_command(self, command: str, on_line=None) -> tools.RunResult:
        """Provision declarative dependencies before any authoritative command."""

        from spiral.builder_tools import ensure_builder_dependencies

        deps = ensure_builder_dependencies(
            self.ws,
            timeout=self.cfg.verify_timeout,
            allow_scripts=bool(getattr(self.cfg, "builder_allow_install_scripts", False)),
        )
        if deps.get("applicable"):
            self.ledger.log(
                "dependencies", ok=bool(deps.get("ok")),
                changed=bool(deps.get("changed")), detail=str(deps.get("detail", ""))[:500],
            )
            if deps.get("changed"):
                self.c.print(f"  [dim]↓ dependencies synchronized · {deps.get('detail', '')[:140]}[/]")
            if not deps.get("ok"):
                return tools.RunResult(
                    "spiral dependency synchronization", 1,
                    str(deps.get("detail") or "dependency synchronization failed"),
                )
        started = time.monotonic()
        self.command_broker.environment.update(deps.get("environment") or {})
        result = self.command_broker.run(
            command, timeout=self.cfg.verify_timeout, on_line=on_line,
            purpose="verification-gate", allow_network=False,
            allow_host_read=not bool(self.cfg.providers),
            require_sandbox=bool(getattr(
                self.cfg, "builder_require_sandbox", True)),
        ).result
        try:
            self.toolsmith.record(
                context="builder_gate", command=command, ok=result.ok,
                duration=max(0.0, time.monotonic() - started), detail=result.out,
                recipe={
                    "summary": "authoritative project gate",
                    "command_shape": command,
                    "method_family": "build and acceptance gate",
                    "tags": [self._project_kind(self.state.get("goal", ""))],
                } if result.ok else None,
            )
        except Exception:
            pass
        return result

    # -- hooks: user shell commands fired on lifecycle events -------------------
    # ~/.config/spiral/config.json →  "hooks": {"task_green": "...", "blocked": "...",
    # "run_complete": "...", "spec_green": "..."}  · event details in $SPIRAL_EVENT/$SPIRAL_INFO
    def _hook(self, event: str, info: str = "") -> None:
        try:
            import os
            import subprocess

            f = Path.home() / ".config" / "spiral" / "config.json"
            cmd = (json.loads(f.read_text()).get("hooks", {}) if f.is_file() else {}).get(event)
            if cmd:
                subprocess.Popen(cmd, shell=True, cwd=self.ws,
                                 env={**os.environ, "SPIRAL_EVENT": event, "SPIRAL_INFO": info[:400]},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass  # hooks must never break the run

    # -- state ----------------------------------------------------------------
    def _dir(self) -> Path:
        d = self.ws / ".spiral"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_state(self, **kw) -> None:
        self.state.update(kw, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        target = self._dir() / "state.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state, indent=2))
        temporary.replace(target)

    def _save_plan(self, goal: str, plan: Plan) -> None:
        (self._dir() / "plan.json").write_text(json.dumps({
            "goal": self._raw_goal(goal),
            "plan": plan_to_dict(plan),
        }, indent=2))

    def load_plan(self) -> Plan | None:
        f = self.ws / ".spiral" / "plan.json"
        if not f.is_file():
            return None
        return parse_plan(json.loads(f.read_text())["plan"])

    # -- snapshot ---------------------------------------------------------------
    def _snapshot(self, *, resume: bool = False) -> None:
        """Commit the current tree so green-to-green reverts have a floor and
        untracked pre-existing files can never be swept by a revert. Work happens
        on a spiral/run-* BRANCH — never on the user's branch; they merge when
        they're happy."""
        if not (self.ws / ".git").is_dir():
            if resume:
                raise RuntimeError(
                    "cannot resume: the workspace has no git transaction history")
            initialized = tools.run("git init -q", self.ws)
            if not initialized.ok:
                raise RuntimeError(
                    initialized.out or "could not initialize workspace transaction history")
        if resume:
            from spiral.transactions import recover_interrupted_workspace

            recovered = recover_interrupted_workspace(
                self.ws, last_green_head=str(self.state.get("last_green_head") or ""))
            if recovered.get("changed"):
                detail = recovered.get("recovery_branch") or recovered.get("recovery")
                self.c.print(
                    f"  [yellow]⟲ recovered interrupted workspace[/]"
                    + (f" · [dim]{detail}[/]" if detail else "")
                )
            return
        cur = tools.run("git rev-parse --abbrev-ref HEAD", self.ws).out.strip()
        if not cur.startswith("spiral/"):
            stem = f"spiral/run-{time.strftime('%Y%m%d-%H%M')}"
            branch = stem
            switched = None
            for index in range(1, 100):
                switched = tools.run(
                    f"git checkout -q -b {shlex.quote(branch)}", self.ws)
                if switched.ok:
                    break
                branch = f"{stem}-{index + 1}"
            if switched is None or not switched.ok:
                raise RuntimeError(
                    switched.out if switched else "could not create an isolated run branch")
            self.c.print(f"  [dim]working on branch [bold]{branch}[/bold] — your branch is untouched[/]")
        gi = self.ws / ".gitignore"
        lines = gi.read_text().splitlines() if gi.is_file() else []
        for want in (
                ".spiral/", "node_modules/", ".venv/", ".pytest_cache/",
                ".mypy_cache/", ".gradle/", "build/", "app/build/", "target/",
                "local.properties"):
            if want not in lines:
                lines.append(want)
        gi.write_text("\n".join(lines) + "\n")
        snap = tools.run(
            "git add -A && git commit -q -m 'spiral: pre-run snapshot' --allow-empty",
            self.ws,
        )
        if not snap.ok:
            raise RuntimeError(snap.out or "could not create the pre-run snapshot")

    @staticmethod
    def _task_fingerprint(task: Task) -> str:
        payload = json.dumps({
            "title": task.title,
            "description": task.description,
            "files": task.files,
            "verify": task.verify,
            "requirements": task.requirements,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _task_is_resumably_done(self, key: str, task: Task) -> bool:
        from spiral.transactions import is_ancestor

        row = (self.state.get("task_records") or {}).get(key) or {}
        if row.get("status") not in {"green", "escalated", "skipped"}:
            return False
        if row.get("fingerprint") != self._task_fingerprint(task):
            return False
        commit = str(row.get("head") or "")
        return row.get("status") == "skipped" or is_ancestor(self.ws, commit)

    @staticmethod
    def _task_counts(plan: Plan, records: dict) -> tuple[int, int]:
        """Return (processed, green) for tasks in the current plan only."""

        processed = green = 0
        for mi, milestone in enumerate(plan.milestones, 1):
            for ti, _task in enumerate(milestone.tasks, 1):
                status = str((records.get(f"{mi}.{ti}") or {}).get("status") or "")
                if status in {"green", "escalated", "skipped"}:
                    processed += 1
                if status in {"green", "escalated"}:
                    green += 1
        return processed, green

    @staticmethod
    def _raw_goal(goal: str) -> str:
        """Remove generated prompt appendices from a persisted/resumed goal."""

        value = str(goal or "")
        markers = (
            "\n\nDESIGN SPECIFICATION (implement these decisions literally):",
            "\n\nEMPIRICAL LOCAL TOOL PROFILE",
        )
        cut = len(value)
        for marker in markers:
            index = value.find(marker)
            if index >= 0:
                cut = min(cut, index)
        return value[:cut].strip()

    @classmethod
    def _goal_hash(cls, goal: str) -> str:
        return hashlib.sha256(cls._raw_goal(goal).encode("utf-8")).hexdigest()

    def _heuristic_project_kind(self, goal: str) -> str:
        """Conservative fallback used only when deliverable analysis is unavailable."""

        ws = self.ws
        g = self._raw_goal(goal).lower()

        def has(pattern: str) -> bool:
            try:
                return next((
                    p for p in ws.rglob(pattern)
                    if "build" not in p.parts and ".spiral" not in p.parts
                ), None) is not None
            except Exception:
                return False

        if has("AndroidManifest.xml") or ("android" in g and ("app" in g or "kotlin" in g)):
            return "android"
        if has("*.xcodeproj") or has("*.xcworkspace") or "swiftui" in g or ("ios" in g and "app" in g):
            return "ios"
        web_dep = False
        for pkg in ws.rglob("package.json"):
            if any(part in {"node_modules", ".spiral", "build", "dist"} for part in pkg.parts):
                continue
            try:
                txt = pkg.read_text().lower()
                web_dep = any(k in txt for k in (
                    "react", "vue", "svelte", "next", "vite", "angular", "solid-js",
                ))
            except Exception:
                pass
            if web_dep:
                break
        if web_dep or has("index.html") or any(
            k in g for k in (
                "website", "web app", "web-app", "frontend", "landing page", "single-page",
            )
        ):
            return "web"
        if any(k in g for k in (
                "gui", "desktop app", "tkinter", "pyqt", "qt app", "electron",
                "gtk", "javafx", "swing", "kivy")):
            return "desktop"
        if any(k in g for k in (
                "plot", "chart", "visualization", "visualisation", "data viz",
                "interactive graph")):
            return "plot"
        if any(k in g for k in (
                "advertisement", "advert", "ad campaign", "commercial", "promo",
                "poster", "brochure", "infographic", "illustration", "image")):
            return "image"
        if any(k in g for k in ("paper", "report", "whitepaper", "document")):
            return "document"
        if any(k in g for k in ("slide deck", "slides", "presentation")):
            return "presentation"
        if "app" in g and any(k in g for k in (
                "screen", "button", "dashboard", "interface", "view", "page")):
            return "desktop"
        return "other"

    def _project_kind(self, goal: str) -> str:
        """Classify the product so the visual designer runs only when it applies —
        'invoked if needed'. Repo signals are ground truth; goal keywords are the
        fallback. The manifest is trusted only for the exact raw goal that produced it."""
        manifest_path = self.ws / ".spiral" / "artifacts.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("goal_sha256") != self._goal_hash(goal):
                    raise ValueError("stale deliverable manifest")
                primary = str(manifest.get("primary_id") or "")
                rows = manifest.get("deliverables") or []
                row = next(
                    (item for item in rows if str(item.get("id")) == primary),
                    rows[0] if rows else {},
                )
                kind = str(row.get("kind") or "")
                if kind:
                    return kind
            except Exception:
                pass
        return self._heuristic_project_kind(goal)

    @staticmethod
    def _is_ui(kind: str) -> bool:
        return kind in {
            "android", "ios", "web", "gui", "desktop", "visualization", "plot",
            "image", "video", "document", "presentation", "notebook", "3d", "game",
        }

    def _goal_with_design(self, goal: str) -> str:
        """Append the design spec so planner and workers implement decisions,
        not vibes. Sits in the stable prompt prefix → KV-cache friendly."""
        goal = self._raw_goal(goal)
        f = self._dir() / "design.md"
        if not f.is_file():
            out = goal
        else:
            # ~1.6k tokens riding every prompt, but it IS the product's taste — and it
            # sits in the stable prefix, so the KV cache pays for it once
            out = goal + "\n\nDESIGN SPECIFICATION (implement these decisions literally):\n" + f.read_text()[:6000]
            # if the palette was materialized, point every screen at the shared tokens
            if (self._dir() / "design_tokens.json").is_file() and self._project_kind(goal) == "android":
                names = ", ".join(f"@color/{n}" for n in TOKEN_COLORS)
                out += ("\n\nCANONICAL PALETTE — the app's colors are defined once in "
                        f"res/values/spiral_tokens.xml as {names}. Reference these for accent, "
                        "background, surface, and primary text; do not invent new color values.")
        try:
            capabilities = self.toolsmith.capability_brief()
            if capabilities:
                out += (
                    "\n\nEMPIRICAL LOCAL TOOL PROFILE (observed on this machine; use it to "
                    "choose realistic implementation and verification routes):\n"
                    + capabilities[:3000]
                )
        except Exception:
            pass
        return out

    # -- plan -------------------------------------------------------------------
    # pipeline: spec → draft → [lint → critic (different brain) → repair] × rounds
    def make_plan(self, goal: str) -> Plan:
        c = self.c
        goal = self._raw_goal(goal)
        repomap = build_repomap(self.ws)
        existing = set(list_files(self.ws))
        c.print(f"  [dim]gate: {self.gate_disp} · repo map: {len(repomap)} chars · planner {self.cfg.planner.name}[/]")

        with Spinner("extracting spec") as sp:
            spec, res = extract_spec(goal, self.cfg, self.ol, progress=lambda k: sp.tick())
            sp.update(tokens=res.total_tokens)
        self.ledger.log("plan", phase="spec", model=self.cfg.planner.name, ptok=res.prompt_tokens, ctok=res.completion_tokens)
        self.ledger.thinking("spec", res.thinking)
        check_notes = sanitize_checks(spec)

        try:
            with Spinner("mapping deliverables") as sp:
                manifest, ares = analyze_deliverables(
                    goal, spec, repomap, self.cfg, self.ol,
                    progress=lambda k: sp.tick(),
                )
        except Exception as exc:
            kind = self._heuristic_project_kind(goal)
            manifest = {
                "schema_version": 1,
                "primary_id": "D1",
                "deliverables": [{
                    "id": "D1",
                    "kind": kind,
                    "description": goal[:500],
                    "root_hint": ".",
                    "output_globs": default_output_globs(kind),
                    "visual": self._is_ui(kind),
                    "interactive": kind in {
                        "web", "android", "ios", "desktop", "game", "notebook",
                    },
                    "acceptance_evidence": [],
                    "tool_families": [],
                }],
                "analysis": f"deterministic fallback: {type(exc).__name__}: {exc}",
            }
            ares = None
            c.print(
                f"  [yellow]○ deliverable analyst unavailable[/] · "
                f"[dim]using conservative {kind} fallback[/]"
            )
        manifest["goal_sha256"] = self._goal_hash(goal)
        (self._dir() / "artifacts.json").write_text(json.dumps(manifest, indent=2))
        if ares is not None:
            self.ledger.log(
                "plan", phase="deliverables", model=self.cfg.planner.name,
                ptok=ares.prompt_tokens, ctok=ares.completion_tokens,
                count=len(manifest.get("deliverables") or []),
            )
            self.ledger.thinking("deliverables", ares.thinking)
        c.print(
            "  [green]●[/] deliverables · "
            + ", ".join(
                f"{row.get('id')}:{row.get('kind')}"
                for row in manifest.get("deliverables") or []
            )
        )

        kind = self._project_kind(goal)
        spec = enrich_deliverable_spec(spec, manifest)
        spec = enrich_product_spec(goal, spec, kind)
        check_notes.extend(sanitize_checks(spec))
        checked = sum(1 for r in spec if r.get("check"))
        reveal(c,
               *(f"     [yellow]check lint:[/] [dim]{note}[/]" for note in check_notes),
               f"  [green]●[/] spec: {len(spec)} requirements"
               + (f" · {checked} with executable checks" if checked else "")
               + f" · [dim]{res.total_tokens} tok[/]",
               *(f"     [dim]{r['id']} ({r.get('kind', 'feature') + (', check' if r.get('check') else '')}):[/] {r['text'][:90]}"
                 for r in spec),
               delay=0.06)
        (self._dir() / "spec.json").write_text(json.dumps(spec, indent=2))
        (self._dir() / "spec-meta.json").write_text(json.dumps({
            "schema_version": 1,
            "goal_sha256": self._goal_hash(goal),
        }, indent=2))

        design_f = self._dir() / "design.md"
        tokens_f = self._dir() / "design_tokens.json"
        design_meta_f = self._dir() / "design-meta.json"
        try:
            design_meta = (
                json.loads(design_meta_f.read_text())
                if design_meta_f.is_file() else {}
            )
        except Exception:
            design_meta = {}
        if design_meta.get("goal_sha256") != self._goal_hash(goal):
            design_f.unlink(missing_ok=True)
            tokens_f.unlink(missing_ok=True)
        if not self._is_ui(kind):
            c.print(f"  [dim]○ no visual design stage — {kind} project, not a UI[/]")
        else:
            design = design_f.read_text() if design_f.is_file() else ""
            if not design:
                self.ol.evict(self.cfg.planner.name)  # designer runs on the critic
                with Spinner("designing") as sp:
                    design, dres = design_brief(goal, spec, self.cfg, self.ol,
                                                progress=lambda k: sp.tick())
                if design:
                    design_f.write_text(design)
                    self.ledger.log("plan", phase="design", model=self.cfg.critic.name,
                                    ptok=dres.prompt_tokens, ctok=dres.completion_tokens)
                    self.ledger.thinking("design", dres.thinking)
                    c.print(f"  [green]●[/] design brief · {len(design)} chars → .spiral/design.md · [dim]{dres.total_tokens} tok[/]")
                self.ol.evict(self.cfg.critic.name)  # planner takes the lane back
            # distill the brief into concrete tokens the harness can materialize
            if not tokens_f.is_file():
                with Spinner("design tokens") as sp:
                    tokens, tres = design_tokens(goal, spec, design, self.cfg, self.ol,
                                                 progress=lambda k: sp.tick())
                if tokens:
                    tokens_f.write_text(json.dumps(tokens, indent=2))
                    self.ledger.log("plan", phase="tokens", model=self.cfg.planner.name,
                                    ptok=tres.prompt_tokens, ctok=tres.completion_tokens)
                    ic = tokens.get("icon", {}) if isinstance(tokens, dict) else {}
                    c.print(f"  [green]●[/] tokens · accent [bold]{tokens.get('accent', '?')}[/] · "
                            f"icon [bold]{ic.get('glyph', '?')}[/] → .spiral/design_tokens.json")
            design_meta_f.write_text(json.dumps({
                "schema_version": 1,
                "goal_sha256": self._goal_hash(goal),
                "kind": kind,
            }, indent=2))
        goal = self._goal_with_design(goal)

        with Spinner("planning") as sp:
            plan, res = make_plan(goal, repomap, self.gate, self.cfg, self.ol, progress=lambda k: sp.tick())
            sp.update(tokens=res.total_tokens)
        self.ledger.log("plan", phase="draft", model=self.cfg.planner.name, ptok=res.prompt_tokens, ctok=res.completion_tokens)
        self.ledger.thinking("draft", res.thinking)
        c.print(f"  [green]●[/] draft plan · {plan.task_count} tasks · [dim]{res.total_tokens} tok[/]")

        reviews = []
        for rnd in range(1, self.cfg.plan_rounds + 1):
            normalize_plan_requirements(spec, plan)
            lint = lint_plan(plan, existing) + coverage_gaps(spec, plan)
            for d in lint:
                c.print(f"     [yellow]lint:[/] {d}")
            self.ol.evict(self.cfg.planner.name)  # make room for the critic
            with Spinner(f"critic round {rnd}") as sp:
                try:
                    verdict, defects, res = critique_plan(
                        goal, spec, repomap, plan, lint, self.gate, self.cfg, self.ol,
                        progress=lambda k: (sp.tick(), sp.update(detail="thinking…" if k == "think" else "writing defects")),
                    )
                    sp.update(tokens=res.total_tokens)
                except Exception as e:
                    c.print(f"  [yellow]○ critic unavailable ({e}) — keeping current plan[/]")
                    break
            if lint:
                existing_issues = {str(row.get("issue") or "") for row in defects}
                defects.extend({
                    "where": "deterministic plan check",
                    "issue": issue,
                    "fix_hint": "Correct the plan so this mechanical check is clean.",
                } for issue in lint if issue not in existing_issues)
                verdict = "revise"
            self.ledger.log("plan", phase=f"critic{rnd}", model=self.cfg.critic.name, ptok=res.prompt_tokens, ctok=res.completion_tokens, verdict=verdict, defects=len(defects))
            self.ledger.thinking(f"critic{rnd}", res.thinking)
            reviews.append({"round": rnd, "verdict": verdict, "defects": defects})
            reveal(c,
                   f"  [green]●[/] critic {rnd} ({self.cfg.critic.name}): [bold]{verdict}[/] · {len(defects)} defects · [dim]{res.total_tokens} tok[/]",
                   *(f"     [red]✗[/] [{d.get('where', '?')}] {d['issue'][:110]}" for d in defects[:8]),
                   delay=0.06)
            if verdict == "pass" or not defects:
                break
            self.ol.evict(self.cfg.critic.name)  # planner returns
            with Spinner("repairing plan") as sp:
                try:
                    plan, res = repair_plan(goal, plan, defects, self.gate, self.cfg, self.ol, progress=lambda k: sp.tick())
                    sp.update(tokens=res.total_tokens)
                except Exception as e:
                    c.print(f"  [yellow]○ repair failed ({e}) — keeping current plan[/]")
                    break
            c.print(f"  [green]●[/] repaired → {plan.task_count} tasks · [dim]{res.total_tokens} tok[/]")

        normalized = normalize_plan_requirements(spec, plan)
        added = ensure_plan_coverage(spec, plan)
        if normalized or added:
            c.print(
                f"  [green]●[/] deterministic coverage · {normalized} mapping(s) normalized"
                + (f" · {added} omitted requirement task(s) added" if added else "")
            )
        (self._dir() / "plan_reviews.json").write_text(json.dumps(reviews, indent=2))
        self._save_plan(goal, plan)
        return plan

    # -- display ------------------------------------------------------------------
    def show_plan(self, plan: Plan) -> None:
        c = self.c
        reveal(c, Panel(plan.understanding.strip() or "(no summary)", title="[bold]spiral understands the goal as[/]",
                        border_style=CLAY, padding=(0, 1)), delay=0.15)
        for mi, m in enumerate(plan.milestones, 1):
            reveal(c,
                   f"\n  [bold {CLAY}]◆ M{mi}[/] [bold]{m.title}[/]",
                   *(f"     [dim]{mi}.{ti}[/] {t.title}" + (f" + [green]{t.verify}[/]" if t.verify else "")
                     for ti, t in enumerate(m.tasks, 1)),
                   delay=0.06)
        gate = self.gate_disp if self.gate else "[yellow]none — unverified run[/]"
        reveal(c, f"\n  [dim]{len(plan.milestones)} milestones · {plan.task_count} tasks · gate on every task:[/] {gate}\n")

    # -- distillation: the strong model teaches the fast one, persistently --------
    def _distill(self, goal: str) -> None:
        """After an escalation win: capture what the fast lane couldn't solve and
        how the strong lane solved it, as a workspace skill. Next run, the fast
        lane sees the recipe — the expensive model teaches the cheap one."""
        try:
            fail = self.ws / ".spiral" / "scratch" / "last_fail.txt"
            errs = ""
            if fail.is_file():
                lines = [ln for ln in fail.read_text().splitlines()
                         if "error" in ln.lower() or "e: " in ln][:5]
                errs = "\n".join(f"  {ln.strip()[:140]}" for ln in lines)
            diff = tools.run("git show --stat HEAD | head -12", self.ws).out
            from spiral.route import ensure_learned_fixes

            f = ensure_learned_fixes(self.ws)
            with f.open("a") as fh:
                fh.write(f"\n## {goal[:80]}\n")
                if errs:
                    fh.write(f"fast lane was stuck on:\n{errs}\n")
                fh.write(f"winning repair:\n```\n{diff[:500]}\n```\n")
            self.ledger.log("distill", task=goal[:80])
            self.c.print("  [dim]⚗ distilled escalation win → .spiral/skills/learned-fixes.md[/]")
        except Exception:
            pass  # distillation must never break the run

    # -- the victory lap: one card that tells the whole run ------------------------
    def _summary_card(self, atom: Atom, t0: float, green: int, blocked: list, total: int) -> None:
        st = atom.run_stats
        mins = (time.time() - t0) / 60
        # what this run would have cost on a typical cloud API (Sonnet-class rates)
        cloud = st["ptok"] * 3 / 1e6 + st["ctok"] * 15 / 1e6
        lines = []
        spec_green = self.state.get("spec_green")
        verdict = ("[bold green]SPEC-GREEN[/]" if spec_green
                   else f"[yellow]{len(self.state.get('gaps', []))} spec gap(s) remain[/]" if spec_green is False
                   else "[dim]spec not validated[/]")
        lines.append(f"[bold]{green}/{total}[/] tasks green · {len(blocked)} blocked · {verdict}")
        lines.append(f"Σ [bold]{atom.tokens:,}[/] tok ({st['ptok'] // 1000}k in / {st['ctok'] // 1000}k out) "
                     f"· {st['attempts']} attempts · {st['esc_lanes']} escalation(s) · {mins:.0f}m wall")
        for m, tps in sorted(st["tps"].items()):
            med = sorted(tps)[len(tps) // 2]
            lines.append(f"[dim]{m}[/] · {len(tps)} gen · median {med:.0f} t/s")
        lines.append(f"≈ [bold]${cloud:.2f}[/] of cloud API · spent [bold green]$0.00[/] · your hardware, your tokens")
        self.c.print(Panel("\n".join(lines), title=f"[{CLAY}]⠷ run summary[/]",
                           border_style=CLAY, padding=(0, 1)))

    # -- consult: a big-context API model reviews the WHOLE project at once -------
    def consult(self, question: str = "") -> None:
        """Send the entire project to a large-context API model and get its
        highest-value observations. Local models are scope-limited (per-task
        files); this spends the big model's context reading everything, then asks
        for a lot of insight in few output tokens — the cheap, high-leverage call."""
        if not self.cfg.providers:
            self.c.print("  [yellow]no API provider configured.[/] Add one to "
                         "~/.config/spiral/config.json and export its api_key_env. See README.")
            return
        model = next(iter(self.cfg.providers))
        # full dump — big model, big context: whole files, generous budget
        repo = build_repomap(self.ws, max_file_bytes=24_000, max_total=350_000)
        pf = self._dir() / "plan.json"
        goal = json.loads(pf.read_text()).get("goal", "") if pf.is_file() else ""
        val = self._dir() / "validation.json"
        gaps = ""
        if val.is_file():
            try:
                vs = json.loads(val.read_text())
                gaps = "\n".join(f"  {v['id']} [{v['status']}]: {v.get('evidence','')[:120]}"
                                 for v in vs if v.get("status") != "implemented")
            except Exception:
                pass

        system = (
            "You are a staff engineer reviewing an ENTIRE project in one pass. Give only "
            "high-value, specific, actionable observations — reference exact files. Cover: "
            "(1) correctness bugs and half-wired features, (2) architecture/structure risks, "
            "(3) anything the goal asks for that is missing or weak, (4) concrete improvement "
            "ideas the team likely hasn't considered. Terse. No preamble, no summary of what "
            "the project is."
        )
        user = (
            f"GOAL:\n{goal or '(none recorded)'}\n\n"
            + (f"KNOWN VALIDATION GAPS:\n{gaps}\n\n" if gaps else "")
            + f"YOUR FOCUS: {question or 'the most important issues and the best ideas to improve this project'}\n\n"
            f"PROJECT:\n{repo}"
        )
        self.c.print(f"  [dim]consulting {model} · {len(repo):,} chars of project (~{len(repo)//4:,} tokens in)[/]")
        with Spinner(f"consulting {model}") as sp:
            res = self.ol.chat(
                model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                num_predict=6000, temperature=1,
                on_delta=lambda kind, piece: sp.tick(),
            )
        if not res.text.strip():
            self.c.print(f"  [red]no response[/] [dim]{res.raw.get('error','')}[/]")
            return
        self.c.print(Panel(res.text.strip(), title=f"[{CLAY}]⠷ {model} · project consult[/]",
                           border_style=CLAY, padding=(0, 1)))
        (self._dir() / "consult.md").write_text(res.text)
        self.c.print(f"  [dim]{res.prompt_tokens:,} in / {res.completion_tokens:,} out · saved to .spiral/consult.md[/]\n")

    # -- validation: judge the CODE against the SPEC, then close the gaps ---------
    def _load_spec(self, goal: str) -> list[dict]:
        f = self._dir() / "spec.json"
        meta_f = self._dir() / "spec-meta.json"
        try:
            meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
        except Exception:
            meta = {}
        if f.is_file() and meta.get("goal_sha256") == self._goal_hash(goal):
            spec = enrich_product_spec(
                goal, json.loads(f.read_text()), self._project_kind(goal))
            f.write_text(json.dumps(spec, indent=2))
            return spec
        with Spinner("extracting spec") as sp:
            spec, _ = extract_spec(goal, self.cfg, self.ol, progress=lambda k: sp.tick())
        spec = enrich_product_spec(goal, spec, self._project_kind(goal))
        sanitize_checks(spec)
        f.write_text(json.dumps(spec, indent=2))
        meta_f.write_text(json.dumps({
            "schema_version": 1,
            "goal_sha256": self._goal_hash(goal),
        }, indent=2))
        return spec

    VALIDATE_CHUNK = 4  # small evidence batches survive local and API context limits

    def _delivery_manifest(self, goal: str) -> dict:
        declaration_path = self._dir() / "artifacts.json"
        try:
            declaration = json.loads(declaration_path.read_text())
            if declaration.get("goal_sha256") != self._goal_hash(goal):
                raise ValueError("stale deliverable declaration")
        except Exception:
            declaration = {
                "goal_sha256": self._goal_hash(goal),
                "primary_id": "D1",
                "deliverables": [{
                    "id": "D1", "kind": self._project_kind(goal),
                    "description": self._raw_goal(goal),
                    "root_hint": ".", "visual": self._is_ui(
                        self._project_kind(goal)),
                    "interactive": False,
                    "output_globs": default_output_globs(
                        self._project_kind(goal)),
                }],
            }
        from spiral.delivery import build_delivery_manifest

        delivery = build_delivery_manifest(
            self.ws, declaration,
            visual_status=(
                self.state.get("visual_reviews")
                or str(self.state.get("visual_review") or "")
            ),
            gate=self.gate_disp,
        )
        (self._dir() / "delivery.json").write_text(
            json.dumps(delivery, indent=2), encoding="utf-8")
        return delivery

    def validate_only(self, goal: str, rnd: int = 1) -> list[dict]:
        """One inspection pass: per-requirement verdicts from code, printed as a
        scoreboard. Requirements are judged in CHUNKS so no reply can truncate,
        and any requirement without a verdict is surfaced as 'unjudged' — silence
        must never read as coverage."""
        c = self.c
        spec = self._load_spec(goal)
        validation_path = self._dir() / "validation.json"
        try:
            previous_rows = json.loads(validation_path.read_text()) if validation_path.is_file() else []
        except Exception:
            previous_rows = []
        previous = {
            str(row.get("id")): row for row in previous_rows
            if isinstance(row, dict) and row.get("id")
        }
        delivery = self._delivery_manifest(goal)
        delivered = {
            str(row.get("id")): row
            for row in (delivery.get("deliverables") or [])
        }
        artifact_specs = [
            row for row in spec if row.get("origin") == "deliverable-manifest"
        ]
        det = [
            r for r in spec
            if r.get("check") and r not in artifact_specs
        ]
        opined = [
            r for r in spec
            if not r.get("check") and r not in artifact_specs
        ]
        judged_by = (f"{len(det)} by execution · {self.cfg.critic.name} judges the rest"
                     if det else f"{self.cfg.critic.name} judges code")
        c.print(f"[bold {CLAY}]━━ validation {rnd} · {len(spec)} requirements · {judged_by} ━━[/]")

        verdicts: list[dict] = []
        tok_total = 0
        for requirement in artifact_specs:
            identifier = str(requirement.get("deliverable") or "")
            row = delivered.get(identifier) or {}
            files = [
                str(item.get("path"))
                for item in (row.get("files") or [])
                if item.get("path")
            ]
            roots = [str(item) for item in (row.get("project_roots") or [])]
            paths = [*files, *roots]
            if row.get("ready"):
                evidence = (
                    f"delivery manifest resolves {identifier} to "
                    f"{', '.join(paths[:8]) or 'a runnable project root'}; "
                    f"decoder/parser evidence {len(row.get('structural_evidence') or [])}; "
                    f"visual {row.get('visual_status')}"
                )
                verdicts.append({
                    "id": requirement["id"], "status": "implemented",
                    "evidence": evidence, "fresh": True,
                    "judge": "delivery-manifest",
                })
            elif (row.get("output_present") and row.get("structure_ok")
                    and row.get("visual_required")
                    and row.get("visual_status") in {"", "skipped"}):
                verdicts.append({
                    "id": requirement["id"], "status": "unjudged",
                    "evidence": (
                        f"{identifier} is structurally present, but independent visual "
                        f"evidence is {row.get('visual_status') or 'not run'}"
                    ),
                    "fresh": False, "judge": "delivery-manifest",
                })
            else:
                issues = "; ".join(row.get("issues") or [
                    "declared deliverable was not resolved"])
                verdicts.append({
                    "id": requirement["id"], "status": "missing",
                    "evidence": f"{identifier}: {issues}",
                    "fresh": True, "judge": "delivery-manifest",
                    "fix": {
                        "title": f"finish deliverable {identifier}",
                        "description": (
                            f"Produce and independently validate {requirement.get('text')}. "
                            f"Current delivery issues: {issues}"
                        ),
                        "files": files,
                    },
                })
        # ---- executable acceptance checks first: exit codes, not opinions -------
        for r in det:
            with Spinner(f"check {r['id']}") as sp:
                v = self._run_verified_command(
                    r["check"], on_line=lambda ln: sp.update(detail=ln))
            self.ledger.log("check", id=r["id"], cmd=r["check"][:120], exit=v.code)
            if v.ok:
                verdicts.append({"id": r["id"], "status": "implemented", "check": r["check"],
                                 "evidence": f"acceptance check passed: {r['check'][:70]}",
                                 "fresh": True})
            elif v.code in (124, 126, 127):
                # the CHECK is broken (timeout / denylist / command not found) —
                # that must indict the check, not the requirement
                c.print(f"  [yellow]○ {r['id']} check unusable (exit {v.code}) — falling back to the validator[/]")
                opined.append(r)
            else:
                tail = " ".join(" ".join(v.out.splitlines()[-3:]).split())[:160]
                verdicts.append({
                    "id": r["id"], "status": "missing", "check": r["check"],
                    "evidence": f"acceptance check failed (exit {v.code}): {tail}",
                    "fresh": True,
                    "fix": {"title": f"make the acceptance check for {r['id']} pass",
                            "description": (f"Requirement: {r['text']}. Its executable acceptance check "
                                            f"`{r['check']}` exits {v.code}. Check output tail: {tail}"),
                            "files": []},
                })

        self.ol.evict(self.cfg.planner.name)
        for i in range(0, len(opined), self.VALIDATE_CHUNK):
            batch = opined[i:i + self.VALIDATE_CHUNK]
            label = f"validating {batch[0]['id']}–{batch[-1]['id']}"
            context_tokens = max(
                8192, int(self.cfg.spec_for(self.cfg.critic.name).num_ctx))
            # Reserve room for system/goal/schema, reasoning, and a complete JSON
            # verdict. Source code averages below four chars/token, so 2.7 is a
            # deliberately conservative conversion.
            source_tokens = max(
                5000, context_tokens - min(6144, context_tokens // 3) - 5000)
            context_chars = min(80_000, source_tokens * 27 // 10)
            repomap, selected = build_relevant_repomap(
                self.ws, batch,
                max_file_bytes=min(18_000, max(6_000, context_chars // 3)),
                max_total=context_chars,
            )
            with (self._dir() / "validation-retrieval.jsonl").open(
                    "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "round": rnd,
                    "requirements": [str(row.get("id")) for row in batch],
                    "selected_files": selected,
                    "context_chars": len(repomap),
                }) + "\n")
            try:
                with Spinner(label) as sp:
                    vs, res = validate_spec(
                        goal, batch, repomap, self.gate, self.cfg, self.ol,
                        progress=lambda k: (sp.tick(), sp.update(detail="reading code…" if k == "think" else "writing verdicts")),
                    )
                expected = {str(row["id"]) for row in batch}
                for row in vs:
                    if (not isinstance(row, dict)
                            or str(row.get("id")) not in expected
                            or row.get("status") not in {"implemented", "partial", "missing"}):
                        continue
                    verdicts.append({**row, "fresh": True})
                tok_total += res.total_tokens
                self.ledger.thinking(f"validate{rnd}-{batch[0]['id']}", res.thinking)
            except Exception as e:
                c.print(f"  [yellow]○ batch {batch[0]['id']}–{batch[-1]['id']} failed:[/] [dim]{e}[/]")

        judged = {v.get("id") for v in verdicts}
        for r in spec:
            if r["id"] not in judged:
                old = previous.get(str(r["id"]))
                if old and old.get("status") in {"implemented", "partial", "missing"}:
                    verdicts.append({
                        **old,
                        "fresh": False,
                        "evidence": (
                            f"validator unavailable this round; retained prior "
                            f"{old.get('status')} verdict: {old.get('evidence', '')}"
                        )[:500],
                    })
                else:
                    verdicts.append({
                        "id": r["id"], "status": "unjudged", "fresh": False,
                        "evidence": "validator returned no verdict; no remediation inferred",
                    })

        marks = {"implemented": ("✓", "green"), "partial": ("◐", "yellow"),
                 "missing": ("✗", "red"), "unjudged": ("?", "yellow")}
        counts: dict[str, int] = {}
        retained = 0
        order = {"implemented": 0, "partial": 1, "missing": 2, "unjudged": 3}
        board: list[str] = []
        for v in sorted(verdicts, key=lambda v: order.get(v.get("status"), 4)):
            m, style = marks.get(v.get("status"), ("?", "dim"))
            counts[v.get("status", "unjudged")] = counts.get(v.get("status", "unjudged"), 0) + 1
            stale = v.get("fresh") is False
            retained += int(stale)
            suffix = " [yellow](retained)[/]" if stale else ""
            board.append(
                f"  [{style}]{m} {v['id']}[/]{suffix} "
                f"[dim]{v.get('evidence', '')[:90]}[/]"
            )
        reveal(c, *board,
               f"  [bold]spec: {counts.get('implemented', 0)}/{len(spec)} implemented[/] · "
               f"[yellow]{counts.get('partial', 0)} partial[/] · [red]{counts.get('missing', 0)} missing[/] · "
               f"[yellow]{counts.get('unjudged', 0)} unjudged[/]"
               + (f" · [yellow]{retained} retained[/]" if retained else "")
               + f" · [dim]{tok_total} tok[/]\n",
               delay=0.06)
        validation_path.write_text(json.dumps(verdicts, indent=2))
        self.ledger.log("validate", round=rnd, model=self.cfg.critic.name, tok=tok_total,
                        retained=retained, **{k: counts.get(k, 0) for k in marks})
        return verdicts

    def _remediate(self, goal: str, atom: Atom, verdicts: list[dict]) -> bool:
        """Turn partial/missing verdicts into a remediation milestone and grind it
        through the same gated loop as any other work. Return whether HEAD moved."""
        from spiral.dash import Dash

        before = tools.run("git rev-parse HEAD", self.ws).out.strip()
        try:
            spec_by_id = {
                str(row.get("id")): str(row.get("text") or "")
                for row in self._load_spec(goal)
            }
        except Exception:
            spec_by_id = {}
        tasks = []
        for v in verdicts:
            if (v.get("status") not in {"partial", "missing"}
                    or v.get("fresh") is False):
                continue
            fix = v.get("fix") or {}
            requirement = spec_by_id.get(str(v.get("id")), "")
            # carry the validator's evidence AND its fix so the worker knows what
            # is wrong, not just which requirement to "implement"
            desc = (
                f"Requirement {v['id']} is NOT met. "
                + (f"Exact requirement: {requirement}. " if requirement else "")
                +
                f"Validator evidence: {v.get('evidence', '(none)')}. "
                f"Required fix: {fix.get('description', 'implement the requirement fully')}"
            )
            tasks.append(Task(
                title=fix.get(
                    "title",
                    f"implement {v['id']}: {requirement[:60]}" if requirement
                    else f"implement {v['id']}",
                ),
                description=desc,
                files=fix.get("files", []) or [],
                # a failed acceptance check becomes the task's own gate: the loop
                # drives the actual criterion to green, not a proxy for it
                verify=v.get("check", "") or "",
            ))
        batch_size = max(1, int(getattr(self.cfg, "builder_remediation_batch", 6)))
        deferred = max(0, len(tasks) - batch_size)
        tasks = tasks[:batch_size]
        if not tasks:
            return False
        if deferred:
            self.c.print(
                f"  [dim]remediation batch: {len(tasks)} now · {deferred} deferred "
                "until evidence is refreshed[/]"
            )
        self.ol.evict(self.cfg.critic.name)  # workers take the lane back
        plan = Plan("close validation gaps", [Milestone("validation gaps", tasks)])
        with Dash(console=self.c, plan=plan, gate=self.gate,
                  thought_log=self._dir() / "thoughts.jsonl") as dash:
            for ti, t in enumerate(tasks, 1):
                if atom.budget_exhausted:
                    dash.print("[red]■ token budget reached before next remediation task[/]")
                    break
                dash.task(1, ti, "run")
                dash.print(f"[bold]▶ V.{ti} {t.title}[/]")
                if self._refresh_gate():
                    dash.print(f"  [green]● verify gate now active:[/] [dim]{self.gate_disp}[/]")
                    dash.gate = self.gate
                verify = t.verify.strip()
                if self.gate:
                    verify = f"({verify}) && ({self.gate})" if verify else self.gate
                spec_task = TaskSpec(
                    goal=f"{t.title}\n{t.description}".strip(),
                    verify_cmd=verify, files=t.files or None, context=goal,
                )
                status = self._run_task(
                    atom, spec_task, dash, allow_done=False,
                    attempts=max(1, int(getattr(
                        self.cfg, "builder_remediation_attempts", 3))),
                    esc_attempts=max(1, int(getattr(
                        self.cfg, "builder_remediation_escalation_attempts", 2))),
                )
                if status != "blocked":
                    self._verify_new_gate(dash, atom, goal)
                dash.task(1, ti, "blocked" if status == "blocked" else "done")
                if atom.budget_exhausted:
                    dash.print("[red]■ token budget reached during remediation[/]")
                    break
        changed = bool(before and tools.run("git rev-parse HEAD", self.ws).out.strip() != before)
        if changed:
            updates = {"product_audit": "stale-after-remediation"}
            updates["last_green_head"] = tools.run(
                "git rev-parse HEAD", self.ws).out.strip()
            if self.state.get("hygiene_gate"):
                updates["hygiene_clean"] = False
            if (self._is_ui(self._project_kind(goal))
                    and self.state.get("visual_review") != "disabled-by-user"):
                updates["visual_review"] = "stale-after-remediation"
            self._write_state(**updates)
        return changed

    def _validate_loop(self, goal: str, atom: Atom) -> bool:
        """Validate → remediate, repeating while the gap count keeps dropping.
        Stop on SPEC-GREEN, on a plateau (a round that closes nothing net), or at
        the hard round cap. Remediation is whack-a-mole — fixing one gap can
        expose another — so 'no fixed count' is not the stop signal; 'no net
        progress' is."""
        prev_signature = None
        for rnd in range(1, self.cfg.validate_rounds + 1):
            authoritative_gate = self.gate
            hygiene_gate = str(self.state.get("hygiene_gate") or "")
            if hygiene_gate and self.state.get("hygiene_clean") is False:
                authoritative_gate = (
                    f"({hygiene_gate}) && ({self.gate})"
                    if self.gate else hygiene_gate
                )
            if authoritative_gate:
                with Spinner("final build gate") as sp:
                    gate_result = self._run_verified_command(
                        authoritative_gate,
                        on_line=lambda ln: sp.update(detail=ln))
                if not gate_result.ok:
                    tail = " ".join(gate_result.out.splitlines()[-4:])[:220]
                    self.c.print(f"[red]■ final build gate red[/] [dim]{tail}[/]")
                    gate_id = (
                        "clean-build-gate"
                        if hygiene_gate and self.state.get(
                            "hygiene_clean") is False
                        else "build-gate"
                    )
                    changed = self._remediate(goal, atom, [{
                        "id": gate_id,
                        "status": "missing",
                        "evidence": f"authoritative build/test command exits {gate_result.code}: {tail}",
                        "fix": {
                            "title": "restore the final build and test gate",
                            "description": (
                                "Repair the reported build, test, dependency, or runtime failure. "
                                "Do not weaken, remove, or bypass the gate."),
                            "files": [],
                        },
                    }])
                    if not changed:
                        self._write_state(
                            spec_green=False, validation_status="build-gate-failed",
                            gaps=[gate_id])
                        return False
                    continue
                if hygiene_gate:
                    self._write_state(hygiene_clean=True)
            verdicts = self.validate_only(goal, rnd)
            unavailable = [
                v for v in verdicts
                if v.get("fresh") is False or v.get("status") == "unjudged"
            ]
            gaps = [
                v for v in verdicts
                if v.get("status") != "implemented" or v.get("fresh") is False
            ]
            if not gaps:
                quality_gaps = []
                if self.state.get("product_audit") not in {"green", "not-applicable"}:
                    quality_gaps.append("product audit is not green")
                if self.state.get("visual_review") not in {
                        "green", "not-applicable", "disabled-by-user"}:
                    quality_gaps.append("visual review is not green")
                if not self.state.get("delivery_ready"):
                    quality_gaps.append(
                        "delivery manifest has unresolved or unverified outputs")
                if self.state.get("hygiene_clean") is False:
                    quality_gaps.append("clean rebuild hygiene gate is red")
                if quality_gaps:
                    self.c.print("[yellow]■ feature spec is green, but finish gates remain: "
                                 + "; ".join(quality_gaps) + "[/]")
                    self._write_state(
                        spec_green=False, validation_status="quality-pending",
                        gaps=quality_gaps)
                    return False
                self.c.print("[bold green]■ SPEC-GREEN — every requirement implemented per validator[/]")
                self._write_state(spec_green=True, validation_status="green", gaps=[])
                self._hook("spec_green", goal[:120])
                return True
            actionable = [
                v for v in gaps
                if v.get("status") in {"partial", "missing"}
                and v.get("fresh") is not False
            ]
            if unavailable and not actionable:
                self.c.print(
                    f"[yellow]■ validator unavailable — retained {len(unavailable)} prior/"
                    "unjudged verdict(s); no speculative remediation was started[/]"
                )
                self._write_state(
                    spec_green=False, validation_status="validator-unavailable",
                    gaps=[str(v.get("id")) for v in unavailable])
                return False
            signature = tuple(sorted(
                (str(v.get("id")), str(v.get("status")), str(v.get("evidence"))[:240])
                for v in gaps
            ))
            if prev_signature is not None and signature == prev_signature:
                self.c.print(f"[yellow]■ validation plateau — the same {len(gaps)} gap(s) remain "
                             "(see .spiral/validation.json)[/]")
                self._write_state(
                    spec_green=False, validation_status="plateau",
                    gaps=[v["id"] for v in gaps])
                return False
            if rnd >= self.cfg.validate_rounds:
                self.c.print(f"[yellow]■ validation round cap reached — {len(gaps)} gap(s) remain "
                             "(see .spiral/validation.json)[/]")
                self._write_state(
                    spec_green=False, validation_status="round-cap",
                    gaps=[v["id"] for v in gaps])
                return False
            prev_signature = signature
            changed = self._remediate(goal, atom, actionable)
            if actionable and not changed:
                self.c.print(
                    f"[yellow]■ remediation made no committed progress on "
                    f"{len(actionable)} evidence-backed gap(s)[/]"
                )
                self._write_state(
                    spec_green=False, validation_status="remediation-stalled",
                    gaps=[str(v.get("id")) for v in actionable])
                return False
        return False

    def _visual_review_loop(self, goal: str, atom: Atom, dash) -> None:
        """Review every declared visual deliverable, not only the primary medium."""

        declaration = {}
        try:
            declaration = json.loads(
                (self._dir() / "artifacts.json").read_text())
            if declaration.get("goal_sha256") != self._goal_hash(goal):
                declaration = {}
        except Exception:
            declaration = {}
        targets: dict[str, list[str]] = {}
        for row in declaration.get("deliverables") or []:
            if not isinstance(row, dict):
                continue
            kind = str(row.get("kind") or "")
            if not kind or not (bool(row.get("visual")) or self._is_ui(kind)):
                continue
            targets.setdefault(kind, []).append(str(row.get("id") or kind))
        primary_kind = self._project_kind(goal)
        if not targets and self._is_ui(primary_kind):
            targets = {primary_kind: ["D1"]}
        if not targets:
            self._write_state(
                visual_review="not-applicable", visual_reviews={})
            return
        if not getattr(self.cfg, "visual_review", True):
            dash.print("  [dim]○ visual review disabled[/]")
            self._write_state(
                visual_review="disabled-by-user",
                visual_reviews={
                    identifier: "disabled-by-user"
                    for identifiers in targets.values()
                    for identifier in identifiers
                },
            )
            return
        from spiral.visual_review import issues_to_verdicts, review_project_visuals

        rounds = max(1, int(getattr(self.cfg, "visual_review_rounds", 2)))
        latest_statuses: dict[str, str] = {}
        latest_reports: dict[str, dict] = {}
        for rnd in range(1, rounds + 1):
            dash.phase("visual review", model=getattr(self.cfg, "vision_model", "") or self.cfg.planner.name)
            dash.print(
                f"[bold {CLAY}]━━ visual review {rnd} · "
                f"{len(targets)} medium(s) ━━[/]")
            verdicts = []
            latest_statuses = {}
            latest_reports = {}
            for kind, identifiers in targets.items():
                result = review_project_visuals(
                    self.ws, self.cfg, self.ol, goal, kind,
                    round_no=rnd,
                    on=lambda msg, medium=kind: dash.detail(
                        f"{medium}: {msg[:80]}"),
                    on_thought=lambda piece, medium=kind: dash.thought(
                        piece, label=f"{medium} vision reviewer"),
                )
                status = (
                    "green" if result.status == "pass"
                    else "gaps" if result.status == "revise"
                    else "skipped"
                )
                for identifier in identifiers:
                    latest_statuses[identifier] = status
                    latest_reports[identifier] = {
                        "kind": kind,
                        "status": status,
                        "detail": result.detail,
                        "report": result.report,
                        "manifest": result.manifest,
                        "screenshots": result.screenshots,
                    }
                color = (
                    "green" if status == "green"
                    else "red" if status == "gaps" else "yellow"
                )
                dash.print(
                    f"  [{color}]●[/] {kind} visual {result.status} · "
                    f"{len(result.issues)} issue(s)"
                    + (f" · [dim]{result.report}[/]" if result.report else "")
                )
                if result.status == "revise" and result.issues:
                    for verdict in issues_to_verdicts(result):
                        verdict["id"] = (
                            f"visual-{identifiers[0]}-{verdict.get('id', 'gap')}")
                        fix = verdict.setdefault("fix", {})
                        fix["title"] = (
                            f"polish {','.join(identifiers)} {kind}: "
                            + str(fix.get("title") or "visual defect")
                        )
                        verdicts.append(verdict)

            if latest_statuses and all(
                    status == "green" for status in latest_statuses.values()):
                self._write_state(
                    visual_review="green",
                    visual_reviews=latest_statuses,
                    visual_review_reports=latest_reports,
                )
                return
            if not verdicts:
                self._write_state(
                    visual_review="skipped",
                    visual_reviews=latest_statuses,
                    visual_review_reports=latest_reports,
                    visual_review_detail=(
                        "one or more declared visual deliverables had no inspectable target"),
                )
                return
            if rnd >= rounds:
                self._write_state(
                    visual_review="gaps",
                    visual_reviews=latest_statuses,
                    visual_review_reports=latest_reports,
                    visual_gaps=[v["id"] for v in verdicts],
                )
                return
            if not self._remediate(goal, atom, verdicts):
                self._write_state(
                    visual_review="gaps",
                    visual_reviews=latest_statuses,
                    visual_review_reports=latest_reports,
                    visual_gaps=[v["id"] for v in verdicts],
                    visual_review_detail="visual remediation made no committed progress",
                )
                return

    def _product_audit_loop(self, goal: str, atom: Atom, dash) -> None:
        """Remediate objective scaffold markers before visual/semantic review."""
        from spiral.product_audit import audit_product, write_product_audit

        rounds = max(1, int(getattr(self.cfg, "product_audit_rounds", 3)))
        prior_signature = None
        for rnd in range(1, rounds + 1):
            report = audit_product(self.ws, goal, self._project_kind(goal))
            if not report.get("applicable"):
                self._write_state(product_audit="not-applicable")
                return
            path = write_product_audit(report, self._dir() / "product-audit.json")
            issues = report.get("issues") or []
            if not issues:
                dash.print(f"  [green]● product audit green[/] · [dim]{path}[/]")
                self._write_state(product_audit="green", product_audit_report=str(path))
                return
            signature = tuple((row.get("id"), row.get("evidence")) for row in issues)
            dash.print(f"[bold {CLAY}]━━ product audit {rnd} · {len(issues)} gap(s) ━━[/]")
            for issue in issues:
                dash.print(f"  [red]✗ {issue.get('id')}[/] [dim]{issue.get('evidence','')[:130]}[/]")
            if rnd >= rounds or signature == prior_signature:
                self._write_state(
                    product_audit="gaps", product_audit_report=str(path),
                    product_gaps=[row.get("id") for row in issues])
                return
            prior_signature = signature
            verdicts = [{
                "id": row.get("id"),
                "status": "missing" if row.get("severity") == "major" else "partial",
                "evidence": row.get("evidence", ""),
                "fix": {
                    "title": f"close {row.get('id')} finish gap",
                    "description": row.get("fix", "complete the product behavior"),
                    "files": row.get("files") or [],
                },
            } for row in issues]
            self._remediate(goal, atom, verdicts)

    # -- step-mode gatekeeper: shift-tab flips auto↔step live ---------------------
    def _gatekeep(self, dash, watcher, label: str) -> str:
        """Returns 'run' | 'skip' | 'quit'. Only prompts in step mode."""
        if watcher is None or not watcher.enabled:
            return "run"
        dash.mode = watcher.mode
        if watcher.mode != "step":
            return "run"
        watcher.drain()
        with dash.pause():
            self.c.print(f"  [bold yellow]⏸ step[/] next: [bold]{label}[/]  [dim](enter run · s skip · a auto · q quit)[/]")
            k = watcher.ask()
        if k in ("a", "A"):
            watcher.mode = "auto"
            dash.mode = "auto"
            return "run"
        if k in ("s", "S"):
            return "skip"
        if k in ("q", "Q"):
            return "quit"
        return "run"

    # -- run ---------------------------------------------------------------------
    def _gate_green(self, ui) -> bool:
        ui.phase("checking gate", model="gate")
        r = self._run_verified_command(
            self.gate, on_line=lambda ln: ui.detail(ln))
        return r.ok

    def _verify_new_gate(self, dash, atom, goal: str) -> None:
        """After a task's edits land, a build gate may have come into existence for
        the first time (the task that creates ``pyproject.toml`` / ``tests/`` is
        otherwise the one task never held to it). If so, run it now, and repair once
        if it's red — so the *creating* task meets the same bar as every later one.
        A no-op when no gate newly appeared."""
        if not self._refresh_gate() or not self.gate:
            return
        dash.gate = self.gate
        dash.print(f"  [green]● verify gate now active:[/] [dim]{self.gate_disp}[/]")
        if self._gate_green(dash):
            return
        dash.print("  [yellow]⚠ new gate is red — repairing the task that introduced it[/]")
        self._run_task(atom, TaskSpec(
            goal=("A build gate just became active and is failing. Repair whatever it "
                  "reports until it passes, with the smallest changes that keep the "
                  "project's intent and style."),
            verify_cmd=self.gate, files=None, context=goal), dash)

    def _run_task(
        self, atom: Atom, spec: TaskSpec, ui,
        attempts: int | None = None, esc_attempts: int | None = None,
        ratchet: bool = False, allow_done: bool = True,
    ) -> str:
        """Run with escalation. Returns 'green' | 'escalated' | 'blocked'.
        With ratchet (bootstrap), partial progress banks as checkpoints and
        compounds across both model lanes. allow_done=False forbids ALREADY_DONE
        (remediation of validator-proven gaps)."""
        strict = not ratchet
        if atom.budget_exhausted:
            ui.print("  [red]■ run token budget reached; task was not started[/]")
            return "blocked"
        if atom.run(spec, attempts=attempts, strict_green=strict, ratchet=ratchet,
                    allow_done=allow_done, ui=ui, route=getattr(self, "_route", None)):
            return "green"
        if atom.budget_exhausted:
            ui.print("  [red]■ run token budget reached; escalation suppressed[/]")
            return "blocked"
        ui.print(f"  [rgb(217,119,87)]⇑ escalating to {self.cfg.escalation.name}[/]")
        atom.run_stats["esc_lanes"] += 1
        if atom.run(
            spec, model=self.cfg.escalation.name,
            attempts=esc_attempts or self.cfg.escalation_attempts,
            strict_green=strict, ratchet=ratchet, allow_done=allow_done, ui=ui,
            diversity=False,  # the dense lane is the last resort — no second sampler
        ):
            self._distill(spec.goal)
            return "escalated"
        return "blocked"

    def _preflight(self) -> None:
        """One advisory line if the machine is untuned — never blocks autonomy."""
        try:
            from spiral.tune import CONFIG_PATH, kv_type

            if not (CONFIG_PATH.is_file() and kv_type()):
                self.c.print(
                    "  [yellow]⚠ untuned[/] [dim]— context windows are guesses and models may "
                    "page. Run [bold]spiral tune[/bold] once (+ ollama restart) between runs.[/]\n"
                )
        except Exception:
            pass

    def _revert(self, paths: list[str]) -> None:
        """Undo harness-written files precisely: restore tracked ones from HEAD,
        delete newly-created untracked ones. Never touches unrelated files."""
        import shlex
        for rel in paths:
            q = shlex.quote(rel)
            if tools.run(f"git ls-files --error-unmatch -- {q}", self.ws).ok:
                tools.run(f"git restore --worktree -- {q}", self.ws)
            else:
                (self.ws / rel).unlink(missing_ok=True)

    def _foundation(self, dash, goal: str) -> None:
        """Deterministic design ground truth before feature work. For an Android
        app, draw the launcher icon from the design tokens and wire the manifest —
        the fiddly, always-the-same plumbing a small model reliably gets wrong, so
        the app never ships the stock robot. Committed only if the gate stays green."""
        if self._project_kind(goal) != "android":
            return
        from spiral.transactions import TaskTransaction

        try:
            transaction = TaskTransaction.begin(
                self.ws, "android design foundation")
        except RuntimeError as exc:
            dash.print(
                f"  [yellow]○ foundation deferred:[/] [dim]{exc}[/]")
            return
        tf = self._dir() / "design_tokens.json"
        try:
            tokens = json.loads(tf.read_text()) if tf.is_file() else {}
        except Exception:
            tokens = {}
        if not isinstance(tokens, dict):
            tokens = {}
        icon = tokens.get("icon", {}) if isinstance(tokens.get("icon"), dict) else {}
        accent = icon.get("foreground") or tokens.get("accent") or "#D97757"
        bg = icon.get("background") or tokens.get("background") or "#0A0A0A"
        glyph = icon.get("glyph") or "spiral"
        try:
            written = write_android_icon(self.ws, accent, bg, glyph)
            written += write_android_tokens(
                self.ws, tokens)  # canonical palette resource
        except Exception:
            transaction.rollback(reason="foundation generation failed")
            raise
        if not written:
            return  # already wired — nothing to do
        if self.gate and not self._gate_green(dash):
            transaction.rollback(reason="foundation made the gate red")
            dash.print("  [yellow]○ foundation reverted — gate went red[/]")
            return
        try:
            _short_head, moved = transaction.commit(
                "spiral: foundation - launcher icon and palette")
        except Exception:
            transaction.rollback(reason="foundation commit failed")
            raise
        if not moved:
            return
        self._write_state(
            last_green_head=tools.run(
                "git rev-parse HEAD", self.ws).out.strip())
        dash.print(f"  [green]■ foundation:[/] launcher icon [bold]{glyph}[/] + palette · {len(written)} files")

    def build(self, goal: str, resume: bool = False, approve: bool = False) -> None:
        from spiral.dash import Dash

        c = self.c
        t0 = time.time()
        self._preflight()
        self._snapshot(resume=resume)
        if not resume:
            self.state = {
                "last_green_head": tools.run(
                    "git rev-parse HEAD", self.ws).out.strip(),
            }

        plan = self.load_plan() if resume else None
        if resume and not goal.strip():
            try:
                goal = str(json.loads(
                    (self._dir() / "plan.json").read_text()).get("goal") or "")
            except Exception:
                goal = str(self.state.get("goal") or "")
        goal = self._raw_goal(goal)
        if plan is None:
            plan = self.make_plan(goal)
        raw_goal = goal
        goal = self._goal_with_design(raw_goal)
        self.show_plan(plan)
        if approve:
            import sys as _sys
            if _sys.stdin.isatty():
                ans = input("  execute this plan? [y/N] ").strip().lower()
                if ans != "y":
                    c.print("  [dim]aborted — plan is saved; rerun with --resume to use it[/]")
                    return

        atom = Atom(self.ws, self.cfg, console=c)

        # the router: fold prior runs' ledger into per-signature verdicts, so
        # error classes the worker has never beaten skip its lane entirely
        from spiral import route as _route

        sig_stats = _route.mine(self.ws / ".spiral" / "ledger.jsonl",
                                self.cfg.worker.name, self.cfg.escalation.name)
        hard = sum(1 for s in sig_stats if _route.decide(s, sig_stats))
        if hard:
            c.print(f"  [dim]⇒ router: {hard} known hard signature(s) will skip the worker lane[/]")
        self._route = (lambda sig: _route.decide(sig, sig_stats)) if hard else None

        prior_records = dict(self.state.get("task_records") or {}) if resume else {}
        current_tasks = {
            f"{mi}.{ti}": task
            for mi, milestone in enumerate(plan.milestones, 1)
            for ti, task in enumerate(milestone.tasks, 1)
        }
        blocked: list[str] = [
            f"{key} {current_tasks[key].title}"
            + (" (skipped)" if row.get("status") == "skipped" else "")
            for key, row in prior_records.items()
            if key in current_tasks and row.get("status") in {"blocked", "skipped"}
        ]
        total = plan.task_count
        resumed_done = sum(
            self._task_is_resumably_done(f"{mi}.{ti}", task)
            for mi, milestone in enumerate(plan.milestones, 1)
            for ti, task in enumerate(milestone.tasks, 1)
        )
        self._write_state(
            goal=raw_goal, gate=self.gate, tasks_total=total,
            tasks_done=resumed_done,
            blocked=blocked, task_records=prior_records,
            run_status="active",
        )

        from spiral.keys import Watcher

        watcher = Watcher().start()
        # the cockpit: pinned plan panel + live status line for the whole grind
        with Dash(console=c, plan=plan, gate=self.gate,
                  thought_log=self._dir() / "thoughts.jsonl") as dash:
            dash.mode = watcher.mode if watcher.enabled else ""
            watcher.on_key("t", dash.toggle_thoughts)
            watcher.on_key("T", dash.toggle_thoughts)
            dash.set_tokens(0)

            # ---- milestone 0: the gate must be green before feature work -------
            if self.gate:
                gate_ok = self._gate_green(dash)
                if not gate_ok:
                    dash.task(0, 0, "run")
                    dash.print(f"[bold {CLAY}]━━ M0: bootstrap — make the build gate pass ━━[/]")
                    spec = TaskSpec(
                        goal=(
                            "The project build is broken. Repair whatever the build gate reports — "
                            "configuration, resources, manifests, or source — until it passes. Make the "
                            "smallest changes that preserve the project's existing intent and style."
                        ),
                        verify_cmd=self.gate,
                        files=None,
                        context=goal,
                    )
                    status = self._run_task(
                        atom, spec, dash,
                        attempts=self.cfg.bootstrap_attempts,
                        esc_attempts=self.cfg.bootstrap_attempts,
                        ratchet=True,
                    )
                    if status == "blocked":
                        dash.task(0, 0, "blocked")
                        dash.print("[red]■ bootstrap could not reach green — aborting run (nothing can be verified).[/]")
                        self._write_state(blocked=["M0 bootstrap"], tokens=atom.tokens, outcome="bootstrap_failed")
                        return
                    dash.task(0, 0, "done")
                    self._write_state(
                        last_green_head=tools.run(
                            "git rev-parse HEAD", self.ws).out.strip(),
                        blocked=[row for row in blocked if row != "M0 bootstrap"],
                    )
                    dash.print(f"  [green]■ gate is green — features begin ({status})[/]")
                else:
                    dash.task(0, 0, "done")

            # ---- foundation: deterministic design ground truth (icon, etc.) -----
            self._foundation(dash, goal)

            # ---- the grind: every task keeps the gate green ---------------------
            done = 0
            for mi, m in enumerate(plan.milestones, 1):
                dash.print(f"[bold {CLAY}]━━ M{mi}/{len(plan.milestones)}: {m.title} ━━[/]")
                for ti, t in enumerate(m.tasks, 1):
                    done += 1
                    task_key = f"{mi}.{ti}"
                    if resume and self._task_is_resumably_done(task_key, t):
                        dash.task(mi, ti, "done")
                        dash.print(f"  [dim]↳ {task_key} already green at its recorded commit[/]")
                        continue
                    blocked = [
                        row for row in blocked
                        if not row.startswith(f"{task_key} ")
                    ]
                    decision = self._gatekeep(dash, watcher, f"{mi}.{ti} {t.title}")
                    if decision == "skip":
                        dash.print(f"  [yellow]⏭ skipped by you:[/] {mi}.{ti} {t.title}")
                        blocked.append(f"{mi}.{ti} {t.title} (skipped)")
                        dash.task(mi, ti, "blocked")
                        records = dict(self.state.get("task_records") or {})
                        records[task_key] = {
                            "status": "skipped",
                            "fingerprint": self._task_fingerprint(t),
                            "head": tools.run("git rev-parse HEAD", self.ws).out.strip(),
                        }
                        processed_count, _green_count = self._task_counts(plan, records)
                        self._write_state(
                            task_records=records, blocked=blocked,
                            tasks_done=processed_count,
                        )
                        continue
                    if decision == "quit":
                        dash.print("  [yellow]■ stopped by you — green work is committed; --resume continues[/]")
                        watcher.stop()
                        self._write_state(outcome="user_stop", tokens=atom.tokens)
                        return
                    dash.task(mi, ti, "run")
                    self._write_state(
                        active_task=task_key,
                        active_task_fingerprint=self._task_fingerprint(t),
                    )
                    dash.print(f"[bold]▶ {mi}.{ti} {t.title}[/]  [dim]({done}/{total} · {atom.tokens} tok · {(time.time() - t0) / 60:.0f}m)[/]")
                    if self._refresh_gate():   # project may have materialised a gate since the last task
                        dash.print(f"  [green]● verify gate now active:[/] [dim]{self.gate_disp}[/]")
                        dash.gate = self.gate
                    verify = t.verify.strip()
                    if self.gate:
                        verify = f"({verify}) && ({self.gate})" if verify else self.gate
                    spec = TaskSpec(
                        goal=f"{t.title}\n{t.description}".strip(),
                        verify_cmd=verify,
                        files=t.files or None,
                        context=goal,
                    )
                    status = self._run_task(atom, spec, dash)
                    if status != "blocked":
                        self._verify_new_gate(dash, atom, goal)   # this task may have created the gate
                    if status == "blocked":
                        blocked.append(f"{mi}.{ti} {t.title}")
                        dash.task(mi, ti, "blocked")
                        dash.print("  [red]✗ blocked[/] — reverted; continuing with the rest of the plan")
                        self._hook("blocked", t.title)
                    else:
                        dash.task(mi, ti, "done")
                        self._hook("task_green", t.title)
                    current_head = tools.run("git rev-parse HEAD", self.ws).out.strip()
                    records = dict(self.state.get("task_records") or {})
                    records[task_key] = {
                        "status": status,
                        "fingerprint": self._task_fingerprint(t),
                        "head": current_head,
                        "gate": self.gate_disp,
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    dash.set_tokens(atom.tokens)
                    processed_count, _green_count = self._task_counts(plan, records)
                    update = {
                        "tasks_done": processed_count, "blocked": blocked, "tokens": atom.tokens,
                        "task_records": records, "active_task": None,
                    }
                    if status != "blocked":
                        update["last_green_head"] = current_head
                    self._write_state(**update)
                    if atom.budget_exhausted:
                        dash.print(f"[red]■ run token budget reached[/] ({atom.tokens}) — stopping; resume with --resume")
                        self._write_state(outcome="budget_stop")
                        return

            # ---- report ---------------------------------------------------------
            mins = (time.time() - t0) / 60
            _processed_count, green_count = self._task_counts(
                plan, dict(self.state.get("task_records") or {}))
            dash.phase("plan complete")
            dash.print(f"[bold green]■ plan complete[/] · {green_count}/{total} tasks green · {atom.tokens} tok · {mins:.0f}m")
            if blocked:
                dash.print("[yellow]blocked tasks:[/]")
                for b in blocked:
                    dash.print(f"  [yellow]-[/] {b}")
            self._write_state(
                outcome="plan_complete", run_status="finishing",
                active_task=None, minutes=round(mins, 1))

        watcher.stop()

        # ---- hygiene: incremental builds can mask staleness — one clean build ----
        gradle_gates = [
            gate for gate in self.gates
            if "gradlew" in gate.command or re.search(
                r"(?:^|[;&|() ])gradle(?:\s|$)", gate.command)
        ]
        if gradle_gates:
            c.print("  [dim]hygiene: clean build (incremental-staleness check)[/]")
            clean_commands = []
            for gate in gradle_gates:
                rel = gate.root.relative_to(self.ws)
                clean = (
                    "./gradlew clean -q"
                    if "gradlew" in gate.command else "gradle clean -q"
                )
                if rel != Path("."):
                    clean = f"cd {shlex.quote(str(rel))} && ({clean})"
                clean_commands.append(f"({clean})")
            hygiene_gate = " && ".join(clean_commands)
            command = (
                f"({hygiene_gate}) && ({self.gate})"
                if self.gate else hygiene_gate
            )
            r = self._run_verified_command(command)
            self._write_state(
                hygiene_clean=bool(r.ok), hygiene_gate=hygiene_gate)
            c.print(
                "  [green]● clean build green[/]"
                if r.ok else
                "  [red]● clean build RED — remediation will see the clean gate[/]"
            )
        else:
            self._write_state(hygiene_clean=True, hygiene_gate="")

        # ---- finish fixed point: product, visual, runtime and semantic gates -----
        spec_green = False
        previous_finish_signature = None
        finish_rounds = max(1, int(getattr(self.cfg, "finish_rounds", 4)))
        qa_plan = Plan("finish quality", [Milestone(
            "finish gates", [Task(
                "audit the complete product",
                "Run deterministic product checks, visual inspection, the build/test gate, "
                "and final requirement validation on the same revision.",
            )],
        )])
        for finish_round in range(1, finish_rounds + 1):
            c.print(f"[bold {CLAY}]━━ finish pass {finish_round}/{finish_rounds} ━━[/]")
            with Dash(console=c, plan=qa_plan, gate=self.gate,
                      thought_log=self._dir() / "thoughts.jsonl") as qa_dash:
                self._product_audit_loop(goal, atom, qa_dash)
                self._visual_review_loop(goal, atom, qa_dash)
                delivery = self._delivery_manifest(goal)
                qa_dash.print(
                    f"  [{'green' if delivery.get('ready') else 'yellow'}]●[/] "
                    f"delivery manifest · "
                    f"{sum(bool(row.get('ready')) for row in delivery.get('deliverables') or [])}/"
                    f"{len(delivery.get('deliverables') or [])} ready · "
                    "[dim].spiral/delivery.json[/]"
                )
                self._write_state(
                    delivery_manifest=str(self._dir() / "delivery.json"),
                    delivery_ready=bool(delivery.get("ready")),
                )
            spec_green = self._validate_loop(goal, atom)
            if spec_green:
                break
            finish_signature = (
                tools.run("git rev-parse HEAD", self.ws).out.strip(),
                self.state.get("product_audit"), self.state.get("visual_review"),
                self.state.get("validation_status"), tuple(self.state.get("gaps") or []),
            )
            if finish_signature == previous_finish_signature:
                c.print("[yellow]■ finish plateau — the same evidence-backed gaps remain[/]")
                break
            previous_finish_signature = finish_signature
            if self.state.get("validation_status") not in {"quality-pending"}:
                break

        outcome = "complete" if spec_green else "finished_with_gaps"
        self._write_state(outcome=outcome, run_status=outcome, tokens=atom.tokens,
                          minutes=round((time.time() - t0) / 60, 1))
        _processed_count, green_count = self._task_counts(
            plan, dict(self.state.get("task_records") or {}))
        self._hook(
            "run_complete",
            f"{green_count}/{total} tasks green · "
            + ("spec green" if spec_green else "finish gaps remain"),
        )
        self._summary_card(atom, t0, green_count, blocked, total)
