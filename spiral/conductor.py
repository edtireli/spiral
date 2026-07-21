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
import sys
import time
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
    Milestone, Plan, Task, coverage_gaps, critique_plan, design_brief, design_tokens,
    extract_spec, lint_plan, make_plan, parse_plan, plan_to_dict, repair_plan,
    sanitize_checks, validate_spec,
)
from spiral.ledger import Ledger
from spiral.repomap import build_repomap, list_files

CLAY = "rgb(217,119,87)"


def detect_gate(ws: Path) -> str:
    """Deterministic build-gate detection. The gate is ground truth; prefer the
    strongest cheap-to-run signal the project offers."""
    if (ws / "gradlew").is_file():
        return "./gradlew assembleDebug"
    if (ws / "package.json").is_file():
        try:
            scripts = json.loads((ws / "package.json").read_text()).get("scripts", {})
            for key in ("test", "build", "typecheck", "lint"):
                if key in scripts:
                    return f"npm run {key} --silent"
        except Exception:
            pass
    if (ws / "Cargo.toml").is_file():
        return "cargo build --quiet"
    if (ws / "go.mod").is_file():
        return "go build ./..."
    if (ws / "pyproject.toml").is_file() or (ws / "pytest.ini").is_file() or (ws / "tests").is_dir():
        # exit 5 = "no tests collected" — a project with a pyproject but no tests yet
        # is not failing, it just has nothing to assert. Treat 5 as green so an early
        # greenfield project isn't a permanently-red gate; real failures (1) stay red.
        return "python -m pytest -q || [ $? -eq 5 ]"
    return ""


class Conductor:
    def __init__(self, workspace: str | Path = ".", cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.ws = Path(workspace).resolve()
        self.ol = Ollama(self.cfg.base_url)
        self.c = make_console()
        self._base_gate = ""
        self.gate = ""
        self.gate_disp = "none detected"
        self._refresh_gate()
        self.state: dict = {}
        self.ledger = Ledger(self.ws)

    def _refresh_gate(self) -> bool:
        """(Re)detect the build gate against the *current* workspace and rebuild the
        composed gate command. Spiral often starts on an empty repo and creates the
        project as it goes (a pyproject.toml / tests dir only appears mid-run), so the
        gate has to be re-detected as files materialise — detecting once at construction
        leaves every task unverified. Returns True when the detected gate changed."""
        base = detect_gate(self.ws)
        if base == self._base_gate and (self.gate or not base):
            return False
        self._base_gate = base
        gate = base
        disp = base or "none detected"
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
        (self._dir() / "state.json").write_text(json.dumps(self.state, indent=2))

    def _save_plan(self, goal: str, plan: Plan) -> None:
        (self._dir() / "plan.json").write_text(json.dumps({"goal": goal, "plan": plan_to_dict(plan)}, indent=2))

    def load_plan(self) -> Plan | None:
        f = self.ws / ".spiral" / "plan.json"
        if not f.is_file():
            return None
        return parse_plan(json.loads(f.read_text())["plan"])

    # -- snapshot ---------------------------------------------------------------
    def _snapshot(self) -> None:
        """Commit the current tree so green-to-green reverts have a floor and
        untracked pre-existing files can never be swept by a revert. Work happens
        on a spiral/run-* BRANCH — never on the user's branch; they merge when
        they're happy."""
        if not (self.ws / ".git").is_dir():
            tools.run("git init -q", self.ws)
        cur = tools.run("git rev-parse --abbrev-ref HEAD", self.ws).out.strip()
        if not cur.startswith("spiral/"):
            branch = f"spiral/run-{time.strftime('%Y%m%d-%H%M')}"
            tools.run(f"git checkout -q -b {branch}", self.ws)
            self.c.print(f"  [dim]working on branch [bold]{branch}[/bold] — your branch is untouched[/]")
        gi = self.ws / ".gitignore"
        lines = gi.read_text().splitlines() if gi.is_file() else []
        for want in (".spiral/", ".gradle/", "build/", "app/build/", "local.properties"):
            if want not in lines:
                lines.append(want)
        gi.write_text("\n".join(lines) + "\n")
        tools.run("git add -A && git commit -q -m 'spiral: pre-run snapshot' --allow-empty", self.ws)

    def _project_kind(self, goal: str) -> str:
        """Classify the product so the visual designer runs only when it applies —
        'invoked if needed'. Repo signals are ground truth; goal keywords are the
        fallback. Returns: android | ios | web | gui | other."""
        ws = self.ws
        g = goal.lower()

        def has(pattern: str) -> bool:
            try:
                return next((p for p in ws.rglob(pattern) if "build" not in p.parts), None) is not None
            except Exception:
                return False

        if has("AndroidManifest.xml") or ("android" in g and ("app" in g or "kotlin" in g)):
            return "android"
        if has("*.xcodeproj") or has("*.xcworkspace") or "swiftui" in g or ("ios" in g and "app" in g):
            return "ios"
        web_dep = False
        pkg = ws / "package.json"
        if pkg.is_file():
            try:
                txt = pkg.read_text().lower()
                web_dep = any(k in txt for k in ("react", "vue", "svelte", "next", "vite", "angular", "solid-js"))
            except Exception:
                pass
        if web_dep or has("index.html") or any(
            k in g for k in ("website", "web app", "web-app", "frontend", "landing page", "single-page")
        ):
            return "web"
        if any(k in g for k in ("gui", "desktop app", "tkinter", "pyqt", "qt app", "electron", "gtk", "javafx", "swing", "kivy")):
            return "gui"
        if "app" in g and any(k in g for k in ("screen", "button", "dashboard", "interface", "view", "page")):
            return "gui"
        return "other"

    @staticmethod
    def _is_ui(kind: str) -> bool:
        return kind in ("android", "ios", "web", "gui")

    def _goal_with_design(self, goal: str) -> str:
        """Append the design spec so planner and workers implement decisions,
        not vibes. Sits in the stable prompt prefix → KV-cache friendly."""
        f = self._dir() / "design.md"
        if not f.is_file():
            return goal
        # ~1.6k tokens riding every prompt, but it IS the product's taste — and it
        # sits in the stable prefix, so the KV cache pays for it once
        out = goal + "\n\nDESIGN SPECIFICATION (implement these decisions literally):\n" + f.read_text()[:6000]
        # if the palette was materialized, point every screen at the shared tokens
        if (self._dir() / "design_tokens.json").is_file() and self._project_kind(goal) == "android":
            names = ", ".join(f"@color/{n}" for n in TOKEN_COLORS)
            out += ("\n\nCANONICAL PALETTE — the app's colors are defined once in "
                    f"res/values/spiral_tokens.xml as {names}. Reference these for accent, "
                    "background, surface, and primary text; do not invent new color values.")
        return out

    # -- plan -------------------------------------------------------------------
    # pipeline: spec → draft → [lint → critic (different brain) → repair] × rounds
    def make_plan(self, goal: str) -> Plan:
        c = self.c
        repomap = build_repomap(self.ws)
        existing = set(list_files(self.ws))
        c.print(f"  [dim]gate: {self.gate_disp} · repo map: {len(repomap)} chars · planner {self.cfg.planner.name}[/]")

        with Spinner("extracting spec") as sp:
            spec, res = extract_spec(goal, self.cfg, self.ol, progress=lambda k: sp.tick())
            sp.update(tokens=res.total_tokens)
        self.ledger.log("plan", phase="spec", model=self.cfg.planner.name, ptok=res.prompt_tokens, ctok=res.completion_tokens)
        self.ledger.thinking("spec", res.thinking)
        checked = sum(1 for r in spec if r.get("check"))
        reveal(c,
               *(f"     [yellow]check lint:[/] [dim]{note}[/]" for note in sanitize_checks(spec)),
               f"  [green]●[/] spec: {len(spec)} requirements"
               + (f" · {checked} with executable checks" if checked else "")
               + f" · [dim]{res.total_tokens} tok[/]",
               *(f"     [dim]{r['id']} ({r.get('kind', 'feature') + (', check' if r.get('check') else '')}):[/] {r['text'][:90]}"
                 for r in spec),
               delay=0.06)
        (self._dir() / "spec.json").write_text(json.dumps(spec, indent=2))

        kind = self._project_kind(goal)
        design_f = self._dir() / "design.md"
        tokens_f = self._dir() / "design_tokens.json"
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
        goal = self._goal_with_design(goal)

        with Spinner("planning") as sp:
            plan, res = make_plan(goal, repomap, self.gate, self.cfg, self.ol, progress=lambda k: sp.tick())
            sp.update(tokens=res.total_tokens)
        self.ledger.log("plan", phase="draft", model=self.cfg.planner.name, ptok=res.prompt_tokens, ctok=res.completion_tokens)
        self.ledger.thinking("draft", res.thinking)
        c.print(f"  [green]●[/] draft plan · {plan.task_count} tasks · [dim]{res.total_tokens} tok[/]")

        reviews = []
        for rnd in range(1, self.cfg.plan_rounds + 1):
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
    def _summary_card(self, atom: Atom, t0: float, done: int, blocked: list, total: int) -> None:
        st = atom.run_stats
        mins = (time.time() - t0) / 60
        # what this run would have cost on a typical cloud API (Sonnet-class rates)
        cloud = st["ptok"] * 3 / 1e6 + st["ctok"] * 15 / 1e6
        lines = []
        spec_green = self.state.get("spec_green")
        verdict = ("[bold green]SPEC-GREEN[/]" if spec_green
                   else f"[yellow]{len(self.state.get('gaps', []))} spec gap(s) remain[/]" if spec_green is False
                   else "[dim]spec not validated[/]")
        lines.append(f"[bold]{done - len(blocked)}/{total}[/] tasks green · {len(blocked)} blocked · {verdict}")
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
        if f.is_file():
            return json.loads(f.read_text())
        with Spinner("extracting spec") as sp:
            spec, _ = extract_spec(goal, self.cfg, self.ol, progress=lambda k: sp.tick())
        sanitize_checks(spec)
        f.write_text(json.dumps(spec, indent=2))
        return spec

    VALIDATE_CHUNK = 7  # requirements per call — verdicts stay far from the token cap

    def validate_only(self, goal: str, rnd: int = 1) -> list[dict]:
        """One inspection pass: per-requirement verdicts from code, printed as a
        scoreboard. Requirements are judged in CHUNKS so no reply can truncate,
        and any requirement without a verdict is surfaced as 'unjudged' — silence
        must never read as coverage."""
        c = self.c
        spec = self._load_spec(goal)
        repomap = build_repomap(self.ws, max_file_bytes=10_000, max_total=120_000)
        det = [r for r in spec if r.get("check")]
        opined = [r for r in spec if not r.get("check")]
        judged_by = (f"{len(det)} by execution · {self.cfg.critic.name} judges the rest"
                     if det else f"{self.cfg.critic.name} judges code")
        c.print(f"[bold {CLAY}]━━ validation {rnd} · {len(spec)} requirements · {judged_by} ━━[/]")

        verdicts: list[dict] = []
        tok_total = 0
        # ---- executable acceptance checks first: exit codes, not opinions -------
        for r in det:
            with Spinner(f"check {r['id']}") as sp:
                v = tools.run(r["check"], self.ws, timeout=self.cfg.verify_timeout,
                              on_line=lambda ln: sp.update(detail=ln))
            self.ledger.log("check", id=r["id"], cmd=r["check"][:120], exit=v.code)
            if v.ok:
                verdicts.append({"id": r["id"], "status": "implemented", "check": r["check"],
                                 "evidence": f"acceptance check passed: {r['check'][:70]}"})
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
                    "fix": {"title": f"make the acceptance check for {r['id']} pass",
                            "description": (f"Requirement: {r['text']}. Its executable acceptance check "
                                            f"`{r['check']}` exits {v.code}. Check output tail: {tail}"),
                            "files": []},
                })

        self.ol.evict(self.cfg.planner.name)
        for i in range(0, len(opined), self.VALIDATE_CHUNK):
            batch = opined[i:i + self.VALIDATE_CHUNK]
            label = f"validating {batch[0]['id']}–{batch[-1]['id']}"
            try:
                with Spinner(label) as sp:
                    vs, res = validate_spec(
                        goal, batch, repomap, self.gate, self.cfg, self.ol,
                        progress=lambda k: (sp.tick(), sp.update(detail="reading code…" if k == "think" else "writing verdicts")),
                    )
                verdicts += vs
                tok_total += res.total_tokens
                self.ledger.thinking(f"validate{rnd}-{batch[0]['id']}", res.thinking)
            except Exception as e:
                c.print(f"  [yellow]○ batch {batch[0]['id']}–{batch[-1]['id']} failed:[/] [dim]{e}[/]")

        judged = {v.get("id") for v in verdicts}
        for r in spec:
            if r["id"] not in judged:
                verdicts.append({"id": r["id"], "status": "unjudged",
                                 "evidence": "validator returned no verdict — will re-judge next round"})

        marks = {"implemented": ("✓", "green"), "partial": ("◐", "yellow"),
                 "missing": ("✗", "red"), "unjudged": ("?", "yellow")}
        counts: dict[str, int] = {}
        order = {"implemented": 0, "partial": 1, "missing": 2, "unjudged": 3}
        board: list[str] = []
        for v in sorted(verdicts, key=lambda v: order.get(v.get("status"), 4)):
            m, style = marks.get(v.get("status"), ("?", "dim"))
            counts[v.get("status", "unjudged")] = counts.get(v.get("status", "unjudged"), 0) + 1
            board.append(f"  [{style}]{m} {v['id']}[/] [dim]{v.get('evidence', '')[:90]}[/]")
        reveal(c, *board,
               f"  [bold]spec: {counts.get('implemented', 0)}/{len(spec)} implemented[/] · "
               f"[yellow]{counts.get('partial', 0)} partial[/] · [red]{counts.get('missing', 0)} missing[/] · "
               f"[yellow]{counts.get('unjudged', 0)} unjudged[/] · [dim]{tok_total} tok[/]\n",
               delay=0.06)
        (self._dir() / "validation.json").write_text(json.dumps(verdicts, indent=2))
        self.ledger.log("validate", round=rnd, model=self.cfg.critic.name, tok=tok_total,
                        **{k: counts.get(k, 0) for k in marks})
        return verdicts

    def _remediate(self, goal: str, atom: Atom, verdicts: list[dict]) -> None:
        """Turn partial/missing verdicts into a remediation milestone and grind it
        through the same gated loop as any other work."""
        from spiral.dash import Dash

        tasks = []
        for v in verdicts:
            if v.get("status") == "implemented":
                continue
            fix = v.get("fix") or {}
            # carry the validator's evidence AND its fix so the worker knows what
            # is wrong, not just which requirement to "implement"
            desc = (
                f"Requirement {v['id']} is NOT met. "
                f"Validator evidence: {v.get('evidence', '(none)')}. "
                f"Required fix: {fix.get('description', 'implement the requirement fully')}"
            )
            tasks.append(Task(
                title=fix.get("title", f"implement {v['id']}"),
                description=desc,
                files=fix.get("files", []) or [],
                # a failed acceptance check becomes the task's own gate: the loop
                # drives the actual criterion to green, not a proxy for it
                verify=v.get("check", "") or "",
            ))
        if not tasks:
            return
        self.ol.evict(self.cfg.critic.name)  # workers take the lane back
        plan = Plan("close validation gaps", [Milestone("validation gaps", tasks)])
        with Dash(console=self.c, plan=plan, gate=self.gate) as dash:
            for ti, t in enumerate(tasks, 1):
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
                status = self._run_task(atom, spec_task, dash, allow_done=False)
                if status != "blocked":
                    self._verify_new_gate(dash, atom, goal)
                dash.task(1, ti, "blocked" if status == "blocked" else "done")
                if atom.tokens >= self.cfg.run_token_budget:
                    dash.print("[red]■ token budget reached during remediation[/]")
                    return

    def _validate_loop(self, goal: str, atom: Atom) -> None:
        """Validate → remediate, repeating while the gap count keeps dropping.
        Stop on SPEC-GREEN, on a plateau (a round that closes nothing net), or at
        the hard round cap. Remediation is whack-a-mole — fixing one gap can
        expose another — so 'no fixed count' is not the stop signal; 'no net
        progress' is."""
        prev_gaps = None
        for rnd in range(1, self.cfg.validate_rounds + 1):
            verdicts = self.validate_only(goal, rnd)
            gaps = [v for v in verdicts if v.get("status") != "implemented"]
            if not gaps:
                self.c.print("[bold green]■ SPEC-GREEN — every requirement implemented per validator[/]")
                self._write_state(spec_green=True)
                self._hook("spec_green", goal[:120])
                return
            if prev_gaps is not None and len(gaps) >= prev_gaps:
                self.c.print(f"[yellow]■ no net progress this round — stopping. {len(gaps)} gap(s) remain "
                             "(see .spiral/validation.json)[/]")
                self._write_state(spec_green=False, gaps=[v["id"] for v in gaps])
                return
            if rnd >= self.cfg.validate_rounds:
                self.c.print(f"[yellow]■ validation round cap reached — {len(gaps)} gap(s) remain "
                             "(see .spiral/validation.json)[/]")
                self._write_state(spec_green=False, gaps=[v["id"] for v in gaps])
                return
            prev_gaps = len(gaps)
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
        r = tools.run(self.gate, self.ws, timeout=self.cfg.verify_timeout,
                      on_line=lambda ln: ui.detail(ln))
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
        if atom.run(spec, attempts=attempts, strict_green=strict, ratchet=ratchet,
                    allow_done=allow_done, ui=ui, route=getattr(self, "_route", None)):
            return "green"
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
                tools.run(f"git checkout -- {q}", self.ws)
            else:
                (self.ws / rel).unlink(missing_ok=True)

    def _foundation(self, dash, goal: str) -> None:
        """Deterministic design ground truth before feature work. For an Android
        app, draw the launcher icon from the design tokens and wire the manifest —
        the fiddly, always-the-same plumbing a small model reliably gets wrong, so
        the app never ships the stock robot. Committed only if the gate stays green."""
        if self._project_kind(goal) != "android":
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
        written = write_android_icon(self.ws, accent, bg, glyph)
        written += write_android_tokens(self.ws, tokens)   # canonical palette resource
        if not written:
            return  # already wired — nothing to do
        if self.gate and not self._gate_green(dash):
            self._revert(written)
            dash.print("  [yellow]○ foundation reverted — gate went red[/]")
            return
        tools.run("git add -A && git commit -q -m 'spiral: foundation — launcher icon + palette'", self.ws)
        dash.print(f"  [green]■ foundation:[/] launcher icon [bold]{glyph}[/] + palette · {len(written)} files")

    def build(self, goal: str, resume: bool = False, approve: bool = False) -> None:
        from spiral.dash import Dash

        c = self.c
        t0 = time.time()
        self._preflight()
        self._snapshot()

        plan = self.load_plan() if resume else None
        if plan is None:
            plan = self.make_plan(goal)
        goal = self._goal_with_design(goal)
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

        blocked: list[str] = []
        total = plan.task_count
        self._write_state(goal=goal[:200], gate=self.gate, tasks_total=total, tasks_done=0, blocked=[])

        from spiral.keys import Watcher

        watcher = Watcher().start()
        # the cockpit: pinned plan panel + live status line for the whole grind
        with Dash(console=c, plan=plan, gate=self.gate) as dash:
            dash.mode = watcher.mode if watcher.enabled else ""
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
                    decision = self._gatekeep(dash, watcher, f"{mi}.{ti} {t.title}")
                    if decision == "skip":
                        dash.print(f"  [yellow]⏭ skipped by you:[/] {mi}.{ti} {t.title}")
                        blocked.append(f"{mi}.{ti} {t.title} (skipped)")
                        dash.task(mi, ti, "blocked")
                        continue
                    if decision == "quit":
                        dash.print("  [yellow]■ stopped by you — green work is committed; --resume continues[/]")
                        watcher.stop()
                        self._write_state(outcome="user_stop", tokens=atom.tokens)
                        return
                    dash.task(mi, ti, "run")
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
                    dash.set_tokens(atom.tokens)
                    self._write_state(tasks_done=done, blocked=blocked, tokens=atom.tokens)
                    if atom.tokens >= self.cfg.run_token_budget:
                        dash.print(f"[red]■ run token budget reached[/] ({atom.tokens}) — stopping; resume with --resume")
                        self._write_state(outcome="budget_stop")
                        return

            # ---- report ---------------------------------------------------------
            mins = (time.time() - t0) / 60
            dash.phase("plan complete")
            dash.print(f"[bold green]■ plan complete[/] · {done - len(blocked)}/{total} tasks green · {atom.tokens} tok · {mins:.0f}m")
            if blocked:
                dash.print("[yellow]blocked tasks:[/]")
                for b in blocked:
                    dash.print(f"  [yellow]-[/] {b}")
            self._write_state(outcome="plan_complete", minutes=round(mins, 1))

        watcher.stop()
        self._hook("run_complete", f"{done - len(blocked)}/{total} green")

        # ---- hygiene: incremental builds can mask staleness — one clean build ----
        if self.gate and "gradlew" in self.gate:
            c.print("  [dim]hygiene: clean build (incremental-staleness check)[/]")
            tools.run("./gradlew clean -q", self.ws, timeout=300)
            r = tools.run(self.gate, self.ws, timeout=self.cfg.verify_timeout)
            c.print(f"  {'[green]● clean build green[/]' if r.ok else '[red]● clean build RED — incremental build was lying; remediation will see it[/]'}")

        # ---- the validator: plan-done is a claim; the spec-audit is the verdict --
        self._validate_loop(goal, atom)
        self._write_state(outcome="complete", tokens=atom.tokens,
                          minutes=round((time.time() - t0) / 60, 1))
        self._summary_card(atom, t0, done, blocked, total)
