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
import shutil
import subprocess
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
from spiral.config import Config, general_api_providers
from spiral.contracts import lint_contracts
from spiral.execution import (
    BudgetExceeded, EvidenceLevel, EvidenceRecord, OrchestrationPolicy,
    TaskEvidenceDAG, TaskState,
    evidence_level_for_command,
)
from spiral.ladder import is_python_gate, venv_prefix
from spiral.llm import Ollama
from spiral.runtime_control import checkpoint as runtime_checkpoint
from spiral.planner import (
    DeliverableManifestError, Milestone, Plan, Task, analyze_deliverables,
    coverage_gaps, critique_plan,
    default_output_globs, design_brief, design_tokens, enrich_deliverable_spec,
    enrich_product_spec,
    ensure_plan_coverage,
    extract_spec, lint_plan, make_plan, normalize_plan_requirements, parse_plan,
    plan_to_dict, repair_plan, sanitize_checks, validate_spec,
)
from spiral.ledger import Ledger
from spiral.repomap import build_relevant_repomap, build_repomap, list_files

CLAY = "rgb(217,119,87)"


class BuildIncomplete(RuntimeError):
    """A finite build run ended honestly, but required evidence still has debt."""

    def __init__(self, outcome: str, gaps: list[str] | tuple[str, ...] = (),
                 result_path: str = ".spiral/result.json"):
        self.outcome = str(outcome)
        self.gaps = tuple(str(gap) for gap in gaps if str(gap).strip())
        self.result_path = str(result_path)
        detail = f": {self.gaps[0]}" if self.gaps else ""
        super().__init__(f"build incomplete ({self.outcome}){detail}")


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
    if (
        (ws / "pyproject.toml").is_file() or (ws / "pytest.ini").is_file()
        or (ws / "requirements.txt").is_file() or (ws / "tests").is_dir()
        or next((p for p in ws.glob("*.py")), None) is not None
    ):
        # A ladder, not a single command: parse → load → run → test. `pytest -q`
        # alone answers "does this work?" with "there are no tests, so yes", which
        # is how a project that cannot be imported earns a green commit. The
        # exit-5 tolerance still applies while no test file exists, and the ladder
        # drops it the moment one does. See spiral/ladder.py.
        from spiral.ladder import python_ladder

        return python_ladder(ws)
    return ""


_TEST_FILE_RX = re.compile(
    r"(^|[./_-])(test|tests|spec)([._-]|$)|\.(test|spec)\.[jt]sx?$", re.I)
_SKIP_DIRS = {"node_modules", ".git", "dist", "build", "vendor", "__pycache__",
              ".spiral", ".venv", "venv", "target", "out", "coverage"}


def _orphan_test_files(ws: Path) -> dict[str, list[Path]]:
    """Test files present in the tree, grouped by language.

    A generated test suite that nothing ever executes is worse than no tests: it
    reads as evidence while proving nothing. A real run wrote a self-contradicting
    assertion (-40 and -60 for the same call), committed it, and called the project
    done — because the project had no manifest, so no test runner was ever detected
    and the artifact gate only checks that the file *parses*."""
    import os

    found: dict[str, list[Path]] = {"js": [], "py": []}
    for dirpath, dirnames, filenames in os.walk(ws):
        # prune in place — rglob would still WALK node_modules before skipping it
        dirnames[:] = [d for d in dirnames
                       if d not in _SKIP_DIRS and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(ws)
        in_test_dir = any(p.lower() in {"test", "tests", "spec", "__tests__"}
                          for p in rel_dir.parts)
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            if not (bool(_TEST_FILE_RX.search(name)) or in_test_dir):
                continue
            if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}:
                found["js"].append(path)
            elif suffix == ".py" and (path.stem.startswith("test_")
                                      or path.stem.endswith("_test") or in_test_dir):
                found["py"].append(path)
    return {k: sorted(v)[:40] for k, v in found.items() if v}


def _orphan_test_gate(ws: Path) -> str:
    """A command that RUNS the workspace's tests when no manifest-based gate does.
    Returns '' when there is nothing runnable — never a command that cannot pass for
    environmental reasons (a missing runner is not the project's fault)."""
    groups = _orphan_test_files(ws)
    parts: list[str] = []
    js = groups.get("js") or []
    if js and shutil.which("node"):
        # node >= 18 ships a test runner; older node has no built-in and would fail
        # for a reason no edit could fix, so verify support before adopting it.
        try:
            probe = subprocess.run(["node", "--test", "--test-name-pattern", "^$",
                                    "--test-only"], cwd=ws, capture_output=True,
                                   text=True, timeout=30)
            supported = "bad option" not in (probe.stderr or "").lower()
        except Exception:
            supported = False
        if supported:
            rel = sorted({str(p.relative_to(ws)) for p in js})
            parts.append("node --test " + " ".join(shlex.quote(r) for r in rel))
    py = groups.get("py") or []
    if py:
        try:
            import importlib.util

            has_pytest = importlib.util.find_spec("pytest") is not None
        except Exception:
            has_pytest = False
        if has_pytest:
            rel = sorted({str(p.relative_to(ws)) for p in py})
            parts.append(f"{shlex.quote(sys.executable)} -m pytest -q "
                         + " ".join(shlex.quote(r) for r in rel))
    return " && ".join(parts)


def _runs_tests(command: str) -> bool:
    """Does this gate command already execute a test suite?"""
    low = (command or "").lower()
    return any(tok in low for tok in (
        "pytest", "npm run test", " test", "--test", "gradlew test", "go test",
        "cargo test", "mvn test", "unittest", "vitest", "jest"))


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
        if is_python_gate(gate):
            venv_bin = (
                project_root / ".spiral" / "dependency-cache" / "python"
                / "venv" / "bin"
            )
            # `export`, not a `VAR=x cmd` prefix: the latter only reaches the
            # first command of a `&&` chain, so every rung after the first would
            # run against the wrong interpreter.
            gate = venv_prefix(venv_bin) + gate
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

    manifest_rows = list(rows)

    # Tests that no gate executes are not evidence. If the workspace carries a test
    # suite that nothing above runs, promote running it to a gate of its own — a
    # suite that fails is a RED gate, not an absent one.
    if not any(_runs_tests(row.command) for row in rows):
        orphan = _orphan_test_gate(ws)
        if orphan:
            key = (str(ws), orphan)
            if key not in seen:
                rows.append(GateSpec(ws, orphan, "tests"))
                seen.add(key)

    # structural integrity still applies when no manifest-based gate was found —
    # independent of whether an orphan test gate was just added
    if not manifest_rows:
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
        self.ol = Ollama(self.cfg.base_url, providers=self.cfg.providers)
        self.ol.configure_budget(
            wall_seconds=self.cfg.run_wall_budget_seconds,
            total_tokens=self.cfg.run_token_budget,
            model_calls=self.cfg.run_call_budget,
        )
        self.orchestration_policy = OrchestrationPolicy.from_values(
            self.cfg.complexity_tier, self.cfg.prefer_single_resident_model)
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

    def _prepare_owned_local_model(self, model: str) -> list[str]:
        """Make a deliberate local role switch using this run's exact receipts."""
        if not model or model in getattr(self.ol, "providers", {}):
            return []
        switch = getattr(self.ol, "evict_owned_local_models_except", None)
        return switch({model}) if callable(switch) else []

    def _external_git_approval(self) -> bool:
        from spiral.transactions import external_git_approval

        return external_git_approval()

    def _revision(self) -> str:
        """A durable tree identity without mutating Git in managed chat runs."""

        if self._external_git_approval():
            from spiral.transactions import workspace_fingerprint

            return workspace_fingerprint(self.ws)
        return tools.run("git rev-parse HEAD", self.ws).out.strip()

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
            # patterns get fixed by the same loop as compile errors.
            # `gate` is already a parenthesised `&&`-chain from _compose_gates, and shell
            # `&&` is associative, so append WITHOUT re-wrapping — re-wrapping produced a
            # leading `((…))` that zsh/sh parse as an arithmetic expression, breaking the
            # gate on every task with "bad math expression".
            gate = f"{gate} && ({shlex.quote(sys.executable)} -m spiral.footguns .)"
            disp += " +footguns"
            # a page whose <script> does not parse reports "9 verified, 0 errors"
            # from the artifact gate and earns a commit; the browser finds it far
            # too late. Composed like footguns rather than replacing anything.
            from spiral.ladder import compose as _compose, web_rungs

            scripts = web_rungs(self.ws)
            if scripts:
                # a BRACE group, not parens: compose() already returns a labelled
                # `(cmd) || {…}` chain, and wrapping that in `(` produces a leading
                # `((` — the arithmetic-evaluation trap this file was already bitten
                # by once. `{ …; }` groups without introducing it.
                gate = f"{gate} && {{ {_compose(scripts)}; }}"
                disp += " +scripts"
            # behaviour rungs, per ecosystem — the runtime probe for a page, a
            # --help check for a CLI, the in-process app drive for python (that
            # one lives in the ladder itself). The gate memo keeps repeat runs
            # free on an unchanged tree, which is what makes a browser-driving
            # rung affordable at task granularity.
            from spiral.ladder import behave_rungs

            behave = behave_rungs(self.ws)
            if behave:
                gate = f"{gate} && {{ {_compose(behave)}; }}"
                disp += " +behave"
        if self.cfg.extra_gate:
            # user-defined blocking gate (their linter/tests) — veto power on every task.
            # Checked on its own, before composition: `A && (B || true)` can still fail
            # through A, so the composed gate looks fine while B's veto is silently
            # worthless. The user asked for a blocking gate; one that cannot block is a
            # mistake worth stopping for.
            from spiral.harness_check import vacuous_gate as _vacuous

            why = _vacuous(self.cfg.extra_gate)
            if why:
                raise RuntimeError(
                    f"your extra_gate can never fail — {why}. It is configured as a "
                    "blocking gate but would veto nothing, so every task would pass it "
                    f"untested · extra_gate: {self.cfg.extra_gate[:160]}")
            gate = f"{gate} && ({self.cfg.extra_gate})" if gate else self.cfg.extra_gate
            disp += " +extra_gate"
        # the gate is the run's ground truth, so a gate that cannot PARSE is a
        # broken instrument, not a red verdict — `((…))` composed here once read
        # as shell arithmetic and every run aborted at bootstrap "fixing" healthy
        # code. A syntax check at composition time makes that class impossible
        # to ship: it costs one no-op shell fork and correctly attributes the
        # fault to the harness.
        if gate:
            import subprocess
            from spiral.command_broker import shell_executable

            parse = subprocess.run(
                [shell_executable(), "-n", "-c", gate],
                capture_output=True, text=True,
            )
            if parse.returncode != 0:
                raise RuntimeError(
                    "the composed build gate does not parse as shell — this is a "
                    f"spiral bug, not a project fault: {parse.stderr.strip()[:200]} "
                    f"· gate: {gate[:200]}")
            # parsing is only half of it. A gate that runs fine and cannot report a
            # failure is worse than one that will not start, because it reports green
            # and is believed.
            from spiral.harness_check import vacuous_gate as _vacuous

            vacuity = _vacuous(gate)
            if vacuity:
                raise RuntimeError(
                    f"the composed build gate can never fail — {vacuity}. Every task "
                    "would be scored green without being tested; this is a spiral bug "
                    f"· gate: {gate[:200]}")
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
                if deps.get("failure_kind") == "transient":
                    from spiral.harness_check import HarnessFault

                    raise HarnessFault(
                        "transient dependency setup failed after bounded retries; "
                        "no source-edit attempt was consumed: "
                        + str(deps.get("detail") or "")[:1200]
                    )
                return tools.RunResult(
                    "spiral dependency synchronization", 1,
                    str(deps.get("detail") or "dependency synchronization failed"),
                )
        started = time.monotonic()
        self.command_broker.environment.update(deps.get("environment") or {})
        _full = bool(getattr(self.cfg, "builder_full_access", False))
        result = self.command_broker.run(
            command, timeout=self.cfg.verify_timeout, on_line=on_line,
            purpose="verification-gate", allow_network=_full,
            allow_host_read=_full,
            require_sandbox=bool(getattr(
                self.cfg, "builder_require_sandbox", True)),
            full_access=_full,
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
        if self._external_git_approval():
            self._dir()
            if not resume:
                self.c.print(
                    "  [dim]Git approval boundary active — changes remain in the "
                    "working tree until you approve an exact Git action[/]"
                )
            return
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
                # bytecode is regenerated by every gate run, and `git add -A`
                # tracking it means the next task's transaction sees a dirty
                # workspace and refuses to commit
                "__pycache__/", "*.pyc",
                "local.properties"):
            if want not in lines:
                lines.append(want)
        gi.write_text("\n".join(lines) + "\n")
        snap = tools.run(
            "git add -A && git -c user.name=Spiral "
            "-c user.email=spiral@localhost commit -q "
            "-m 'spiral: pre-run snapshot' --allow-empty",
            self.ws,
        )
        if not snap.ok:
            raise RuntimeError(snap.out or "could not create the pre-run snapshot")

    @staticmethod
    def _task_fingerprint(task: Task) -> str:
        payload = {
            "title": task.title,
            "description": task.description,
            "files": task.files,
            "verify": task.verify,
            "requirements": task.requirements,
        }
        # Contract fields join the fingerprint only when a task HAS them. Adding a
        # field unconditionally rehashes every task, so an in-flight run resumes at
        # task 1 and redoes work it had already committed — which is what happened
        # the first time. A changed contract must invalidate its task; a new field
        # must not invalidate everyone else's.
        for field_name in ("exports", "imports"):
            value = getattr(task, field_name, []) or []
            if value:
                payload[field_name] = value
        payload = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _task_is_resumably_done(self, key: str, task: Task) -> bool:
        from spiral.transactions import is_ancestor

        row = (self.state.get("task_records") or {}).get(key) or {}
        if row.get("status") not in {"green", "escalated", "skipped"}:
            return False
        if row.get("fingerprint") != self._task_fingerprint(task):
            return False
        if self._external_git_approval():
            return True
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
    def _recovery_frontier(current_tasks: dict[str, Task], records: dict,
                           current_head: str) -> list[str]:
        """Blocked tasks worth one fresh pass after the repository has advanced.

        A later independent task can materialize the package, fixture, interface, or
        configuration an earlier task lacked.  Repeating immediately against the same
        tree is thrashing; retrying once against a different verified revision is useful
        new information.  The returned order is the original plan order.
        """
        return [
            key for key in current_tasks
            if str((records.get(key) or {}).get("status") or "") == "blocked"
            and str((records.get(key) or {}).get("head") or "") != current_head
        ]

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
                # a calculator "as a static web page" is a UI, and missing that
                # skipped the design stage, the deterministic foundation, and the
                # visual review for the most common way people phrase a web goal
                "web page", "webpage", "html page", "static site", "static page",
                "single page", "index.html", "web ui", "web interface",
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
        brief = getattr(self, "_capability_brief", "")
        if brief:
            out += ("\n\nCAPABILITIES FOR THIS BUILD (already resolved by the "
                    "harness — do not re-install, and do not plan around tools "
                    "listed as unavailable):\n" + brief[:1500])
        # OUTSIDE the design.md branch: the generated palette deserves announcing
        # whenever it exists, and requiring a design brief file to also exist made
        # the announcement silently skip (no design.md -> workers never told ->
        # the first task invented a second stylesheet next to the generated one)
        if (self.ws / "tokens.css").is_file() and self._project_kind(goal) != "android":
            from spiral.webfoundation import FAVICON_FILE, TOKENS_FILE, tokens_brief

            out += "\n\n" + tokens_brief([TOKENS_FILE, FAVICON_FILE])
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
        except DeliverableManifestError:
            # The analyst was available but twice returned a malformed or
            # semantically invalid manifest. Falling back here would erase the
            # requested product/tool evidence and let planning continue on a lie.
            raise
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

        # The deliverable analyst has now supplied stronger, typed evidence than
        # goal keywords alone (for example ``ffmpeg`` or ``ollama:qwen3:8b``).
        # Builder runs an explicit inspect/setup pass here, before draft planning
        # or any worker edit. ``spiral plan`` stays read-only; build() enables this
        # phase and snapshots any newly declared project dependency immediately
        # afterward so task transactions still begin from a clean checkpoint.
        if getattr(self, "_capability_setup_enabled", False):
            try:
                from spiral.capability import manifest_tool_families

                tool_families = manifest_tool_families(manifest)
            except Exception:
                tool_families = []
            self._resolve_capabilities(
                goal, tool_families=tool_families, setup=True,
                synchronize_projects="if-declared")

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
                design_model = (
                    self.cfg.planner.name if self.cfg.prefer_single_resident_model
                    else self.cfg.critic.name
                )
                self._prepare_owned_local_model(design_model)
                with Spinner("designing") as sp:
                    design, dres = design_brief(goal, spec, self.cfg, self.ol,
                                                progress=lambda k: sp.tick())
                used_design_model = str(
                    (getattr(dres, "raw", {}) or {}).get("spiral_role_model")
                    or design_model
                )
                if design:
                    design_f.write_text(design)
                    self.ledger.log("plan", phase="design", model=used_design_model,
                                    ptok=dres.prompt_tokens, ctok=dres.completion_tokens)
                    self.ledger.thinking("design", dres.thinking)
                    c.print(f"  [green]●[/] design brief · {len(design)} chars → .spiral/design.md · [dim]{dres.total_tokens} tok[/]")
                if used_design_model != self.cfg.planner.name:
                    self._prepare_owned_local_model(self.cfg.planner.name)
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

        normalize_plan_requirements(spec, plan)
        initial_lint = (lint_plan(plan, existing) + coverage_gaps(spec, plan)
                        + lint_contracts(plan, self.ws))
        critic_rounds = self.orchestration_policy.critic_rounds(
            deterministic_defects=len(initial_lint), task_count=plan.task_count,
            requested_rounds=self.cfg.plan_rounds,
        )
        reviews = []
        if critic_rounds == 0:
            reviews.append({
                "round": 0, "verdict": "not_warranted", "defects": [],
                "policy": "single resident model; deterministic checks found no elevated risk",
            })
            c.print("  [dim]○ independent critic not warranted by current risk/evidence policy[/]")
        for rnd in range(1, critic_rounds + 1):
            normalize_plan_requirements(spec, plan)
            lint = (lint_plan(plan, existing) + coverage_gaps(spec, plan)
                    + lint_contracts(plan, self.ws))
            for d in lint:
                c.print(f"     [yellow]lint:[/] {d}")
            if self.cfg.critic.name != self.cfg.planner.name:
                self._prepare_owned_local_model(self.cfg.critic.name)
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
            if self.cfg.critic.name != self.cfg.planner.name:
                self._prepare_owned_local_model(self.cfg.planner.name)
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
    def _write_evidence_result(self, *, outcome: str, blocked: list[str], atom: Atom) -> dict:
        dag = getattr(self, "_evidence_dag", TaskEvidenceDAG())
        extra: list[EvidenceRecord] = []
        if self.state.get("hygiene_clean") and self.state.get("hygiene_gate"):
            command = str(self.state.get("hygiene_gate") or self.gate)
            extra.append(EvidenceRecord(
                evidence_level_for_command(command),
                "clean, non-incremental project gate passed", command=command,
                artifact=self._revision(), source="Spiral deterministic gate",
            ))
        if self.state.get("spec_green"):
            extra.append(EvidenceRecord(
                EvidenceLevel.BEHAVIOR,
                "declared requirements passed the final validation fixed point",
                artifact=".spiral/validation.json", source="Spiral validation gate",
            ))
        report = dag.evidence_report([
            *blocked,
            *(str(gap) for gap in (self.state.get("gaps") or [])),
        ])
        report.records.extend(extra)
        evidence = report.to_dict()
        # Keep acceptance debt machine-readable rather than burying it inside a
        # prose uncertainty sentence. Hosts use this exact list to distinguish a
        # genuinely complete result from a finite run that needs resume/user help.
        evidence["required_gaps"] = list(dict.fromkeys(report.unresolved))
        payload = {
            "schema_version": 1,
            "kind": "spiral.build.handoff",
            "outcome": outcome,
            "revision": self._revision(),
            "evidence": evidence,
            "budget": getattr(getattr(atom, "ol", None), "budget", None).snapshot()
            if getattr(getattr(atom, "ol", None), "budget", None) is not None else {},
            "task_graph": ".spiral/task-evidence-dag.json",
        }
        target = self._dir() / "result.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(target)
        self._write_state(
            result_handoff=str(target), evidence_levels=evidence["levels"],
            uncertainty=evidence["uncertainty"],
        )
        return payload

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
        joint = getattr(atom.ol, "budget", None)
        joint_tokens = joint.total_tokens if joint is not None else atom.tokens
        lines.append(f"Σ [bold]{joint_tokens:,}[/] joint tok "
                     f"· {getattr(joint, 'calls', st['attempts'])} model call(s) "
                     f"· {st['esc_lanes']} escalation(s) · {mins:.0f}m wall")
        for m, tps in sorted(st["tps"].items()):
            med = sorted(tps)[len(tps) // 2]
            lines.append(f"[dim]{m}[/] · {len(tps)} gen · median {med:.0f} t/s")
        evidence = self.state.get("evidence_levels") or []
        uncertainty = (self.state.get("uncertainty") or {}).get("level", "unknown")
        lines.append(f"evidence: {', '.join(evidence) or 'none'} · uncertainty: [bold]{uncertainty}[/]")
        lines.append(f"≈ [bold]${cloud:.2f}[/] cloud-equivalent worker traffic · local compute is finite and budgeted")
        self.c.print(Panel("\n".join(lines), title=f"[{CLAY}]⠷ run summary[/]",
                           border_style=CLAY, padding=(0, 1)))

    # -- consult: a big-context API model reviews the WHOLE project at once -------
    def consult(self, question: str = "") -> None:
        """Send the entire project to a large-context API model and get its
        highest-value observations. Local models are scope-limited (per-task
        files); this spends the big model's context reading everything, then asks
        for a lot of insight in few output tokens — the cheap, high-leverage call."""
        general_providers = general_api_providers(self.cfg.providers)
        if not general_providers:
            self.c.print("  [yellow]no API provider configured.[/] Add one to "
                         "~/.config/spiral/config.json and export its api_key_env. See README.")
            return
        model = next(iter(general_providers))
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

        if opined and self.cfg.critic.name != self.cfg.planner.name:
            self._prepare_owned_local_model(self.cfg.critic.name)
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

        before = self._revision()
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
        if self.cfg.critic.name != self.cfg.planner.name:
            self._prepare_owned_local_model(self.cfg.worker.name)
        plan = Plan("close validation gaps", [Milestone("validation gaps", tasks)])
        with Dash(console=self.c, plan=plan, gate=self.gate,
                  thought_log=self._dir() / "thoughts.jsonl") as dash:
            for ti, t in enumerate(tasks, 1):
                if atom.budget_exhausted:
                    dash.print("[red]■ execution budget reached before next remediation task[/]")
                    break
                dash.task(1, ti, "run")
                dash.print(f"[bold]▶ V.{ti} {t.title}[/]")
                if self._refresh_gate():
                    dash.print(f"  [green]● verify gate now active:[/] [dim]{self.gate_disp}[/]")
                    dash.gate = self.gate
                verify = t.verify.strip()
                if self.gate:
                    verify = f"({verify}) && {self.gate}" if verify else self.gate
                spec_task = TaskSpec(
                    goal=f"{t.title}\n{t.description}".strip(),
                    verify_cmd=verify, files=t.files or None, context=goal,
                    exports=list(getattr(t, "exports", []) or []) or None,
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
                    dash.print("[red]■ execution budget reached during remediation[/]")
                    break
        changed = bool(before and self._revision() != before)
        if changed:
            updates = {"product_audit": "stale-after-remediation"}
            updates["last_green_head"] = self._revision()
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
                    f"({hygiene_gate}) && {self.gate}"
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
            # runtime regressions first go to the deterministic healer: the
            # harness holds the history and a probe, so which commit broke the
            # page is a binary search, not a judgment. A model repairing its own
            # syntax error flip-flopped for whole attempt budgets; the revert
            # costs zero tokens. Healed -> re-audit; refused -> remediation as
            # before (the healer is a fast path, never a gatekeeper).
            if (
                not self._external_git_approval()
                and any(str(row.get("id", "")).startswith("runtime-") for row in issues)
            ):
                from spiral.regress import heal, runtime_predicate

                try:
                    healing = heal(self.ws, runtime_predicate(self.ws))
                except Exception as exc:
                    healing = None
                    dash.print(f"  [yellow]○ healer unavailable[/] [dim]{exc}[/]")
                if healing and healing.healed:
                    dash.print(
                        f"  [green]⟲ healed by revert[/] [dim]{healing.detail}[/]")
                    self._write_state(last_green_head=self._revision())
                    continue          # re-audit the healed tree
                if healing and healing.guilty:
                    dash.print(f"  [yellow]○ healer declined:[/] [dim]{healing.detail}[/]")
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
            verify_cmd=self.gate, files=None, context=goal), dash,
            # ratchet for the same reason bootstrap does: a newly active gate
            # usually reports several STACKED signatures (a missing dependency
            # hiding a bad import), and clearing one of them is real progress.
            # Without it strict_green reverts a correct partial fix, and the next
            # task starts from the same red gate and rediscovers the same error.
            ratchet=True)

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
            ui.print("  [red]■ run execution budget reached; task was not started[/]")
            return "blocked"

        def run_lane(**kwargs) -> bool | None:
            try:
                return atom.run(spec, **kwargs)
            except BudgetExceeded as exc:
                atom.mark_budget_exceeded(exc)
                ui.print(
                    f"  [red]■ run {exc.dimension} budget cannot admit another model call[/] "
                    f"[dim]{str(exc)[:240]}[/]"
                )
                return None

        worker = run_lane(
            attempts=attempts, strict_green=strict, ratchet=ratchet,
            allow_done=allow_done, ui=ui, route=getattr(self, "_route", None),
        )
        if worker is None:
            return "blocked"
        if worker:
            return "green"
        if atom.budget_exhausted:
            ui.print("  [red]■ run execution budget reached; escalation suppressed[/]")
            return "blocked"
        ui.print(f"  [rgb(217,119,87)]⇑ escalating to {self.cfg.escalation.name}[/]")
        atom.run_stats["esc_lanes"] += 1
        escalation = run_lane(
            model=self.cfg.escalation.name,
            attempts=esc_attempts or self.cfg.escalation_attempts,
            strict_green=strict, ratchet=ratchet, allow_done=allow_done, ui=ui,
            diversity=False,  # the dense lane is the last resort — no second sampler
        )
        if escalation is None:
            return "blocked"
        if escalation:
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
        kind = self._project_kind(goal)
        if kind not in {"android", "web", "gui", "desktop", "game"}:
            return
        from spiral.transactions import TaskTransaction

        try:
            transaction = TaskTransaction.begin(
                self.ws, f"{kind} design foundation")
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
            if kind == "android":
                written = write_android_icon(self.ws, accent, bg, glyph)
                written += write_android_tokens(
                    self.ws, tokens)  # canonical palette resource
                label = f"launcher icon [bold]{glyph}[/] + palette"
            else:
                # the web counterpart: a token stylesheet whose text colours were
                # chosen by contrast arithmetic, plus a favicon, so readability and
                # a coherent palette are true on the first commit instead of being
                # remediated after the model guesses hex values
                from spiral.webfoundation import write_web_foundation

                written = write_web_foundation(self.ws, tokens)
                label = f"design tokens + favicon [bold]{glyph}[/]"
        except Exception:
            transaction.rollback(reason="foundation generation failed")
            transaction.close()
            raise
        if not written:
            transaction.close()
            return  # already wired — nothing to do
        if self.gate and not self._gate_green(dash):
            transaction.rollback(reason="foundation made the gate red")
            transaction.close()
            dash.print("  [yellow]○ foundation reverted — gate went red[/]")
            return
        try:
            _short_head, moved = transaction.commit(
                f"spiral: foundation - {kind} design ground truth")
        except Exception:
            transaction.rollback(reason="foundation commit failed")
            transaction.close()
            raise
        if not moved:
            transaction.close()
            return
        self._write_state(last_green_head=self._revision())
        transaction.close()
        dash.print(f"  [green]■ foundation:[/] {label} · {len(written)} files")

    def _gate_predicate(self):
        """The detected gate as a heal() predicate, re-derived per probe tree.

        Re-detection is the point: some commits induce a DIFFERENT gate (a
        package.json appearing switches it to npm), and the workspace's composed
        gate references .spiral scripts a worktree does not carry. A broker rooted
        at the probe tree keeps the sandbox honest about paths.
        """
        from spiral.command_broker import CommandBroker

        def gate_ok(tree) -> bool:
            tree = Path(tree)
            gate = detect_gate(tree)
            if not gate:
                return True
            _full = bool(getattr(self.cfg, "builder_full_access", False))
            result = CommandBroker(tree, self.cfg).run(
                gate, timeout=self.cfg.verify_timeout,
                purpose="verification-gate", allow_network=_full,
                allow_host_read=_full,
                require_sandbox=bool(getattr(
                    self.cfg, "builder_require_sandbox", True)),
                full_access=_full,
            )
            return result.result.code == 0
        return gate_ok

    def _resolve_capabilities(
        self, goal: str, tool_families: list[str] | None = None,
        *, setup: bool = False,
        synchronize_projects: bool | str = True,
    ):
        """Ask what this build needs that this machine lacks, and close the gap.

        Declaring the dependency beats installing it: the provisioning that already
        runs before every gate picks it up, inside the sandbox and against the
        install budget, and the repo ends up recording what it depends on. What
        cannot be declared — a system binary, a model — is reported with the exact
        command instead of being installed behind the user's back.
        """
        try:
            from spiral.capability import (
                Resolution, inspect_workspace, resolve, setup_capabilities,
                write_capabilities,
            )
        except Exception:
            return None
        try:
            if setup:
                outcome = setup_capabilities(
                    self.ws, goal, tool_families,
                    declare=True,
                    synchronize_projects=synchronize_projects,
                    tool_auto=bool(getattr(
                        self.cfg, "builder_tool_auto", True)),
                    full_access=bool(getattr(
                        self.cfg, "builder_full_access", False)),
                    timeout=int(getattr(self.cfg, "verify_timeout", 900)),
                    allow_scripts=bool(getattr(
                        self.cfg, "builder_allow_install_scripts", False)),
                    broker=self.command_broker,
                )
            else:
                outcome = resolve(self.ws, goal, tool_families)
                outcome.inspection = inspect_workspace(self.ws)
        except Exception as exc:
            if not setup:
                self.c.print(
                    f"  [yellow]○ capability check unavailable[/] [dim]{exc}[/]")
                return None
            # Setup is part of the build admission boundary, not optional advice.
            # Persist unexpected preflight failures through the same receipt/gate
            # path as ordinary provisioning failures so a host retry has evidence
            # and planning cannot continue without known prerequisites.
            outcome = Resolution()
            try:
                outcome.inspection = inspect_workspace(self.ws)
            except Exception:
                outcome.inspection = {}
            outcome.setup_reports.append({
                "kind": "capability-preflight",
                "ok": False,
                "changed": False,
                "failure_kind": "setup",
                "detail": (
                    f"capability preflight raised {type(exc).__name__}: {exc}"
                )[:2000],
            })
        write_capabilities(self.ws, outcome)
        self._capability_brief = outcome.brief()
        if outcome.declared:
            self._capability_tree_changed = True
        if outcome.declared:
            names = sorted({p for need in outcome.declared for p in need.packages})
            self.c.print(
                f"  [green]●[/] capability: declared {len(names)} dependency(ies) "
                f"this goal needs · [dim]{', '.join(names)}[/]")
        if outcome.acquired:
            names = sorted({
                need.binary or (need.packages[0] if need.packages else need.id)
                for need in outcome.acquired
            })
            self.c.print(
                f"  [green]●[/] capability: acquired and certified "
                f"{len(names)} prerequisite(s) · [dim]{', '.join(names)}[/]")
        for report in outcome.setup_reports:
            if report.get("ok"):
                continue
            kind = str(report.get("failure_kind") or "setup")
            detail = str(report.get("detail") or "setup did not complete")
            self.c.print(
                f"  [yellow]○ {kind} capability setup:[/] [dim]{detail[:220]}[/]")
        for need in outcome.blocked:
            self.c.print(
                f"  [yellow]○ missing capability:[/] {need.binary or need.id} "
                f"[dim]({need.why}) — install with: {need.install_hint}[/]")
        failed_setup = [
            report for report in outcome.setup_reports
            if not bool(report.get("ok"))
        ]
        if setup and failed_setup:
            from spiral.harness_check import HarnessFault

            first = failed_setup[0]
            failure_kind = str(first.get("failure_kind") or "setup")
            detail = str(first.get("detail") or "capability setup did not complete")
            raise HarnessFault(
                f"{failure_kind} capability setup failed before planning; "
                "no source-edit attempt was consumed. Durable details are in "
                f".spiral/capabilities.json: {detail[:1200]}"
            )
        return outcome

    def build(self, goal: str, resume: bool = False, approve: bool = False) -> None:
        from spiral.dash import Dash

        runtime_checkpoint()
        c = self.c
        t0 = time.time()
        self._preflight()
        self._capability_tree_changed = False
        # capability gap FIRST, so declared dependencies land in the pre-run
        # snapshot. Declaring after the snapshot would leave the workspace dirty
        # and every task transaction would refuse to commit.
        if not resume:
            self._resolve_capabilities(self._raw_goal(goal), setup=True)
        self._snapshot(resume=resume)
        self._capability_tree_changed = False
        if not resume:
            self.state = {"last_green_head": self._revision()}

        plan = self.load_plan() if resume else None
        if resume and not goal.strip():
            try:
                goal = str(json.loads(
                    (self._dir() / "plan.json").read_text()).get("goal") or "")
            except Exception:
                goal = str(self.state.get("goal") or "")
        goal = self._raw_goal(goal)
        if plan is None:
            # A resume without plan.json stopped before any executable task could
            # exist. Re-run analyst-derived typed setup just as a fresh pre-plan
            # pass would; otherwise an interruption during planning permanently
            # suppresses prerequisites that goal-only inspection could not infer.
            self._capability_setup_enabled = True
            try:
                plan = self.make_plan(goal)
            finally:
                self._capability_setup_enabled = False
            if self._capability_tree_changed:
                # Typed tool-family evidence may have declared a dependency that
                # goal-only preflight could not know. Freeze that deterministic
                # setup into the pre-run baseline before any task transaction.
                self._snapshot(resume=False)
                self._write_state(last_green_head=self._revision())
                self._capability_tree_changed = False
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

        # Planning and execution share one client so role prompts reuse residency
        # and every token/call/wall second lands in the same finite run ledger.
        atom = Atom(self.ws, self.cfg, console=c, ol=self.ol)

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
        self._evidence_dag = TaskEvidenceDAG.from_plan(plan)
        dag_path = self._dir() / "task-evidence-dag.json"
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
        for key, task in current_tasks.items():
            row = prior_records.get(key) or {}
            status = str(row.get("status") or "")
            # Blocked records are retried on resume, so they stay pending in the
            # fresh graph. Completed/skipped work is terminal and unlocks ordering.
            if status not in {"green", "escalated", "skipped"}:
                continue
            if not self._task_is_resumably_done(key, task):
                continue
            evidence = []
            command = str(getattr(task, "verify", "") or self.gate)
            if status in {"green", "escalated"} and command:
                evidence.append(EvidenceRecord(
                    evidence_level_for_command(command),
                    f"resumed task {key} retains a recorded passing gate",
                    artifact=str(row.get("head") or ""), command=command,
                    source=".spiral/state.json",
                ))
            self._evidence_dag.finish(
                key,
                TaskState.COMPLETE if status in {"green", "escalated"}
                else TaskState.SKIPPED,
                evidence,
            )
        self._evidence_dag.save(dag_path)
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
                    if status == "blocked" and not self._external_git_approval():
                        # the models cannot fix it — but if some COMMIT broke the
                        # gate, reverting that commit is a computation. Bootstrap
                        # was the one termination path with no healer behind it,
                        # and it aborts the entire run; try the cheap
                        # deterministic exit before the expensive one.
                        from spiral.regress import heal

                        try:
                            healing = heal(self.ws, self._gate_predicate())
                        except Exception as exc:
                            healing = None
                            dash.print(f"  [yellow]○ gate healer unavailable[/] [dim]{exc}[/]")
                        if healing and healing.healed:
                            dash.print(f"  [green]⟲ gate healed by revert[/] [dim]{healing.detail}[/]")
                            self._write_state(last_green_head=self._revision())
                            status = "green" if self._gate_green(dash) else "blocked"
                        elif healing and healing.detail:
                            dash.print(f"  [yellow]○ gate healer declined:[/] [dim]{healing.detail}[/]")
                    if status == "blocked":
                        dash.task(0, 0, "blocked")
                        dash.print("[red]■ bootstrap could not reach green — aborting run (nothing can be verified).[/]")
                        self._write_state(blocked=["M0 bootstrap"], tokens=atom.tokens, outcome="bootstrap_failed")
                        self._write_evidence_result(
                            outcome="bootstrap_failed", blocked=["M0 bootstrap"], atom=atom)
                        watcher.stop()
                        raise BuildIncomplete(
                            "bootstrap_failed", ["M0 bootstrap"],
                            str(self._dir() / "result.json"),
                        )
                    dash.task(0, 0, "done")
                    self._write_state(
                        last_green_head=self._revision(),
                        blocked=[row for row in blocked if row != "M0 bootstrap"],
                    )
                    dash.print(f"  [green]■ gate is green — features begin ({status})[/]")
                else:
                    dash.task(0, 0, "done")

            # ---- foundation: deterministic design ground truth (icon, etc.) -----
            self._foundation(dash, goal)
            # the goal-with-design text was composed BEFORE the foundation existed,
            # so its tokens/favicon brief silently skipped — and the first worker
            # then invented its own stylesheet instead of linking the generated
            # one. Recompose now that the files are on disk.
            goal = self._goal_with_design(raw_goal)

            # ---- the grind: every task keeps the gate green ---------------------
            done = 0
            for mi, m in enumerate(plan.milestones, 1):
                runtime_checkpoint()
                dash.print(f"[bold {CLAY}]━━ M{mi}/{len(plan.milestones)}: {m.title} ━━[/]")
                for ti, t in enumerate(m.tasks, 1):
                    runtime_checkpoint()
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
                            "head": self._revision(),
                        }
                        processed_count, _green_count = self._task_counts(plan, records)
                        self._write_state(
                            task_records=records, blocked=blocked,
                            tasks_done=processed_count,
                        )
                        self._evidence_dag.finish(task_key, TaskState.SKIPPED)
                        self._evidence_dag.save(dag_path)
                        continue
                    if decision == "quit":
                        dash.print("  [yellow]■ stopped by you — green work is committed; --resume continues[/]")
                        watcher.stop()
                        self._write_state(outcome="user_stop", tokens=atom.tokens)
                        self._write_evidence_result(
                            outcome="user_stop", blocked=blocked, atom=atom)
                        raise BuildIncomplete(
                            "user_stop", [*blocked, "run stopped before completion"],
                            str(self._dir() / "result.json"),
                        )
                    dash.task(mi, ti, "run")
                    self._evidence_dag.start(task_key)
                    self._evidence_dag.save(dag_path)
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
                        verify = f"({verify}) && {self.gate}" if verify else self.gate
                    spec = TaskSpec(
                        goal=f"{t.title}\n{t.description}".strip(),
                        verify_cmd=verify,
                        files=t.files or None,
                        context=goal,
                        exports=list(getattr(t, "exports", []) or []) or None,
                    )
                    status = self._run_task(atom, spec, dash)
                    if status != "blocked":
                        self._verify_new_gate(dash, atom, goal)   # this task may have created the gate
                    if (
                        status == "blocked" and self.gate
                        and not self._external_git_approval()
                        and not self._gate_green(dash)
                    ):
                        # the task did not fail on its own work — it inherited a
                        # RED GATE from an earlier commit (observed: a package
                        # manifest whose test script pointed at a file that never
                        # landed poisoned the gate for every task after it, and
                        # each burned both lanes against a fault that was not
                        # theirs). Which commit broke the gate is a computation:
                        # bisect with the gate as the predicate, revert, retry
                        # the task once. Refusals fall through to blocked.
                        from spiral.regress import heal
                        try:
                            healing = heal(self.ws, self._gate_predicate())
                        except Exception as exc:
                            healing = None
                            dash.print(f"  [yellow]○ gate healer unavailable[/] [dim]{exc}[/]")
                        if healing and healing.healed:
                            dash.print(f"  [green]⟲ gate healed by revert[/] [dim]{healing.detail}[/]")
                            self._write_state(last_green_head=self._revision())
                            status = self._run_task(atom, spec, dash)
                        elif healing and healing.detail:
                            dash.print(f"  [yellow]○ gate healer declined:[/] [dim]{healing.detail}[/]")
                    if status == "blocked":
                        blocked.append(f"{mi}.{ti} {t.title}")
                        dash.task(mi, ti, "blocked")
                        dash.print("  [red]✗ blocked[/] — reverted; continuing with the rest of the plan")
                        self._hook("blocked", t.title)
                    else:
                        dash.task(mi, ti, "done")
                        self._hook("task_green", t.title)
                    current_head = self._revision()
                    task_evidence = []
                    if status != "blocked" and verify:
                        task_evidence.append(EvidenceRecord(
                            evidence_level_for_command(verify),
                            f"task {task_key} verification passed",
                            artifact=current_head, command=verify,
                            source="Spiral task transaction",
                        ))
                    self._evidence_dag.finish(
                        task_key,
                        TaskState.BLOCKED if status == "blocked" else TaskState.COMPLETE,
                        task_evidence,
                    )
                    self._evidence_dag.save(dag_path)
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
                        dimension = getattr(atom.ol, "budget", None)
                        dimension = dimension.exhausted_dimension() if dimension is not None else "token"
                        dash.print(
                            f"[red]■ run {dimension or 'execution'} budget reached[/] "
                            "— stopping; resume with --resume")
                        self._write_state(outcome="budget_stop")
                        self._write_evidence_result(
                            outcome="budget_stop", blocked=blocked, atom=atom)
                        watcher.stop()
                        raise BuildIncomplete(
                            "budget_stop",
                            [*blocked, "finite execution budget reached"],
                            str(self._dir() / "result.json"),
                        )

            # ---- recovery frontier ---------------------------------------------
            # A blocked task is evidence debt, not a permanent verdict.  Once later
            # independent work has advanced the tree, give each blocked task one finite
            # fresh pass against that richer repository.  This is deliberately after the
            # ordinary grind (no racing or duplicate work) and before final validation.
            # Same-tree failures are not replayed, so deterministic environment faults do
            # not turn into another model loop.
            records = dict(self.state.get("task_records") or {})
            recovery_keys = self._recovery_frontier(
                current_tasks, records, self._revision())
            if recovery_keys and not atom.budget_exhausted:
                dash.phase("recovering blocked work")
                dash.print(
                    f"[bold {CLAY}]━━ recovery frontier · {len(recovery_keys)} task(s) ━━[/]"
                )
            for task_key in recovery_keys:
                runtime_checkpoint()
                if atom.budget_exhausted:
                    break
                task = current_tasks[task_key]
                mi, ti = (int(value) for value in task_key.split(".", 1))
                dash.task(mi, ti, "run")
                dash.print(
                    f"[bold]↻ {task_key} {task.title}[/] "
                    "[dim]repository advanced since the blocked attempt[/]"
                )
                if self._refresh_gate():
                    dash.gate = self.gate
                verify = task.verify.strip()
                if self.gate:
                    verify = f"({verify}) && {self.gate}" if verify else self.gate
                recovery_spec = TaskSpec(
                    goal=(
                        f"{task.title}\n{task.description}\n\n"
                        "RECOVERY PASS: earlier work was blocked, but the repository has "
                        "since advanced. Re-inspect the current files; preserve all green "
                        "work and finish only the remaining obligation."
                    ).strip(),
                    verify_cmd=verify,
                    files=task.files or None,
                    context=goal,
                    exports=list(getattr(task, "exports", []) or []) or None,
                )
                recovery_status = self._run_task(
                    atom, recovery_spec, dash,
                    attempts=max(1, int(getattr(
                        self.cfg, "builder_remediation_attempts", 3))),
                    esc_attempts=max(1, int(getattr(
                        self.cfg, "builder_remediation_escalation_attempts", 2))),
                )
                if recovery_status != "blocked":
                    self._verify_new_gate(dash, atom, goal)
                    blocked = [
                        row for row in blocked
                        if not row.startswith(f"{task_key} ")
                    ]
                    dash.task(mi, ti, "done")
                    current_head = self._revision()
                    records[task_key] = {
                        "status": recovery_status,
                        "fingerprint": self._task_fingerprint(task),
                        "head": current_head,
                        "gate": self.gate_disp,
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "recovered": True,
                    }
                    evidence = []
                    if verify:
                        evidence.append(EvidenceRecord(
                            evidence_level_for_command(verify),
                            f"recovery pass for task {task_key} verified on the advanced tree",
                            artifact=current_head, command=verify,
                            source="Spiral recovery frontier",
                        ))
                    self._evidence_dag.finish(
                        task_key, TaskState.COMPLETE, evidence)
                    self._evidence_dag.save(dag_path)
                    processed_count, _ = self._task_counts(plan, records)
                    self._write_state(
                        task_records=records, blocked=blocked,
                        tasks_done=processed_count, last_green_head=current_head,
                    )
                    dash.print("  [green]■ recovered and verified[/]")
                else:
                    dash.task(mi, ti, "blocked")
                    current_head = self._revision()
                    records[task_key] = {
                        **dict(records.get(task_key) or {}),
                        "status": "blocked",
                        "fingerprint": self._task_fingerprint(task),
                        # Advancing the recorded head is what makes this a finite
                        # retry.  A resume on the same tree must not replay the
                        # same failed recovery; another pass becomes eligible only
                        # after some later verified work changes the revision.
                        "head": current_head,
                        "gate": self.gate_disp,
                        "recovery_attempted_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S"),
                    }
                    self._write_state(
                        task_records=records, blocked=blocked,
                        active_task=None,
                    )
                    dash.print(
                        "  [yellow]○ still blocked on the advanced tree — retained for resume[/]"
                    )

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
                f"({hygiene_gate}) && {self.gate}"
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
            runtime_checkpoint()
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
                self._revision(),
                self.state.get("product_audit"), self.state.get("visual_review"),
                self.state.get("validation_status"), tuple(self.state.get("gaps") or []),
            )
            if finish_signature == previous_finish_signature:
                c.print("[yellow]■ finish plateau — the same evidence-backed gaps remain[/]")
                break
            previous_finish_signature = finish_signature
            if self.state.get("validation_status") not in {"quality-pending"}:
                break

        _processed_count, green_count = self._task_counts(
            plan, dict(self.state.get("task_records") or {}))
        terminal_dag = getattr(self, "_evidence_dag", TaskEvidenceDAG())
        evidence_debt = list(terminal_dag.evidence_report([
            *blocked,
            *(str(gap) for gap in (self.state.get("gaps") or [])),
        ]).unresolved)
        if self.state.get("delivery_ready") is False:
            evidence_debt.append("delivery manifest has unmet acceptance evidence")
        if green_count != total:
            evidence_debt.append(
                f"only {green_count}/{total} required tasks are green")
        if not spec_green:
            evidence_debt.append("final requirement validation is not green")
        evidence_debt = list(dict.fromkeys(evidence_debt))
        outcome = "complete" if not evidence_debt else "finished_with_gaps"
        self._write_state(outcome=outcome, run_status=outcome, tokens=atom.tokens,
                          minutes=round((time.time() - t0) / 60, 1))
        self._hook(
            "run_complete",
            f"{green_count}/{total} tasks green · "
            + ("spec green" if spec_green else "finish gaps remain"),
        )
        self._write_evidence_result(outcome=outcome, blocked=blocked, atom=atom)
        self._summary_card(atom, t0, green_count, blocked, total)
        if evidence_debt:
            raise BuildIncomplete(
                outcome, evidence_debt, str(self._dir() / "result.json"))
