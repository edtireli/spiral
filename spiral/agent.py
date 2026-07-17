"""The v0 atom — a single task driven to green, unattended.

    edit → verify → fix → commit

The worker (qwen3.6:27b, thinking off, hard token cap) sees the project vision, the
goal, the verify command's current output, and the relevant files. It replies with
SEARCH/REPLACE blocks. We apply them, re-run verify, and either commit (green) or feed
the errors straight back — up to the attempt budget. Ground truth is the exit code.

When a task has no verify command, edits are applied once and committed *unverified* —
surfaced loudly, because a task with no gate is exactly where autonomy is blind.
"""
from __future__ import annotations

import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from spiral import tools
from spiral.config import Config
from spiral.ledger import Ledger
from spiral.skillpack import load_skills, match_skills, render_for_prompt
from spiral.theme import make_console
from spiral.edits import apply_edits, parse_edits
from spiral.llm import Ollama

SYSTEM = (
    "You are spiral's worker — a focused coding agent that edits code to make a "
    "verification command pass.\n\n"
    "Change code ONLY by emitting SEARCH/REPLACE blocks in EXACTLY this format:\n\n"
    "path/to/file.ext\n"
    "<<<<<<< SEARCH\n"
    "exact existing lines to find\n"
    "=======\n"
    "new lines that replace them\n"
    ">>>>>>> REPLACE\n\n"
    "Rules:\n"
    "- SEARCH must copy the current file text closely (whitespace is tolerated).\n"
    "- Each SEARCH section must be SHORT — at most ~12 lines. NEVER copy a whole file.\n"
    "- At most 3 blocks per reply. Fix the FIRST/most important error first; later "
    "rounds will request the rest.\n"
    "- For a NEW file, leave the SEARCH section empty.\n"
    "- Make the SMALLEST change that makes the verification command pass.\n"
    "- If the TASK is ALREADY fully implemented in the FILES shown, reply with "
    "exactly: ALREADY_DONE (nothing else).\n"
    "- If you must reference an identifier, signature, or file you cannot SEE in "
    "FILES, do NOT invent it. Reply exactly:  ASK: grep <name>   (or: ASK: file "
    "<path>) and nothing else. Max 2 ASKs per task, never repeat one.\n"
    "- Output ONLY blocks — no prose, no explanations, no code fences."
)

TEXT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".txt", ".md", ".json", ".toml", ".cfg",
    ".ini", ".sh", ".c", ".h", ".cpp", ".go", ".rs", ".java", ".kt", ".kts", ".gradle",
    ".xml", ".html", ".css", ".yaml", ".yml", ".properties",
}


@dataclass
class TaskSpec:
    goal: str
    verify_cmd: str
    files: list[str] | None = None  # relevant files; None = auto-discover small text files
    context: str = ""               # the project vision, pinned into every prompt


def _auto_files(ws: Path, limit: int = 12, max_bytes: int = 20_000) -> list[str]:
    out: list[str] = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
            continue
        if any(part in tools._SKIP_DIRS or part.startswith(".") for part in p.relative_to(ws).parts):
            continue
        if p.stat().st_size > max_bytes:
            continue
        out.append(str(p.relative_to(ws)))
        if len(out) >= limit:
            break
    return out


class Atom:
    def __init__(self, workspace: str | Path = ".", cfg: Config | None = None, console: Console | None = None):
        self.cfg = cfg or Config.load()
        self.ws = Path(workspace).resolve()
        self.ol = Ollama(self.cfg.base_url)
        self.c = console or make_console()
        self.tokens = 0  # cumulative, for the conductor's budget
        self.run_stats: dict = {"attempts": 0, "green": 0, "ptok": 0, "ctok": 0,
                                "tps": {}, "esc_lanes": 0}
        self.ledger = Ledger(self.ws)
        self.skills = load_skills(self.ws)

    # -- git -----------------------------------------------------------------
    def _ensure_git(self) -> None:
        if not (self.ws / ".git").is_dir():
            tools.run("git init -q && git add -A && git commit -q -m 'spiral: baseline' --allow-empty", self.ws)

    def _commit(self, msg: str) -> tuple[str, bool]:
        """Returns (head, moved). moved=False means the 'edits' changed nothing —
        an empty commit must never count as completed work."""
        prev = tools.run("git rev-parse --short HEAD", self.ws).out.strip()
        tools.run("git add -A", self.ws)
        tools.run(f"git commit -q -m {shlex.quote(msg)}", self.ws)
        head = tools.run("git rev-parse --short HEAD", self.ws).out.strip()
        return head, head != prev

    # -- prompt --------------------------------------------------------------
    def _read(self, rel: str, cap: int = 16_000) -> str:
        try:
            return (self.ws / rel).read_text(errors="replace")[:cap]
        except Exception:
            return ""

    def _prompt(self, task: TaskSpec, files: list[str], verify_out: str, apply_errs: str,
                skills_text: str = "", tried: list[str] | None = None,
                repo_answers: str = "") -> str:
        """Prompt layout is cache-conscious: stable content (project, task, files)
        FIRST, volatile content (verify output, apply errors) LAST. Ollama reuses
        the KV cache for the unchanged prefix between attempts — on a 10k-token
        prompt that skips most of the prompt-eval wait."""
        # build gates (gradle) dump huge logs; the actionable part is at the tail
        if len(verify_out) > 4000:
            verify_out = "…(earlier output truncated)\n" + verify_out[-4000:]
        parts: list[str] = []
        if task.context:
            parts += ["PROJECT — keep every change aligned to this vision:", task.context, ""]
        parts += [
            f"TASK: {task.goal}", "",
            f"VERIFY (must exit 0): {task.verify_cmd or '(none provided)'}", "",
        ]
        if skills_text:
            parts += ["CRAFT NOTES (follow these):", skills_text, ""]
        parts.append("FILES:")
        budget = 60_000  # chars across file bodies; num_ctx is real now (32k tokens)
        for rel in files:
            body = self._read(rel)
            if budget <= 0:
                parts.append(f"--- {rel} --- (omitted, context budget)")
                continue
            body = body[:budget]
            budget -= len(body)
            parts += [f"--- {rel} ---", body, ""]
        parts += ["CURRENT VERIFY OUTPUT:", verify_out or "(none)", ""]
        if repo_answers:
            parts += ["REPO ANSWERS (from your ASKs — ground truth, use these exact names):",
                      repo_answers, ""]
        if tried:
            parts += ["ALREADY TRIED THIS TASK (do NOT repeat these — take a DIFFERENT approach):",
                      *[f"- {t}" for t in tried[-6:]], ""]
        if apply_errs:
            parts += ["SOME EDITS FAILED TO APPLY — fix the SEARCH text:", apply_errs, ""]
        parts.append("Emit SEARCH/REPLACE blocks to make VERIFY exit 0.")
        return "\n".join(parts)

    def _absorb_error_files(self, verify_out: str, files: list[str], cap: int = 14) -> None:
        """Pull file paths out of build errors and add them to the worker's context —
        the build often breaks in files the task didn't list."""
        for m in re.finditer(r"[\w/][\w./-]*\.(?:kt|kts|java|xml|gradle|toml|properties)\b", verify_out):
            path = m.group(0)
            if path.startswith("/"):
                path = "/" + path.lstrip("/")  # gradle emits file:///Users/... URLs
            if path.startswith(str(self.ws)):
                rel = path[len(str(self.ws)) + 1:]
            elif path.startswith("/"):
                continue
            else:
                rel = path
            if rel not in files and (self.ws / rel).is_file():
                files.append(rel)
                if len(files) >= cap:
                    return

    # -- checkpoint ----------------------------------------------------------
    def _checkpoint(self, task: TaskSpec, verify: tools.RunResult) -> None:
        d = self.ws / ".spiral" / "scratch"
        d.mkdir(parents=True, exist_ok=True)
        (d / "last_fail.txt").write_text(
            f"goal: {task.goal}\nverify: {task.verify_cmd}\nexit: {verify.code}\n\n{verify.out}\n"
        )

    # -- the loop ------------------------------------------------------------
    def run(
        self,
        task: TaskSpec,
        model: str | None = None,
        attempts: int | None = None,
        strict_green: bool = False,
        ratchet: bool = False,
        ui=None,
    ) -> bool:
        """Drive one task to green. `model` overrides the worker (escalation);
        `strict_green` reverts the tree to the last commit when the task fails,
        so a failed task can never poison the next one. `ratchet` is bootstrap
        mode — the task STARTS red, so there is no green to protect: instead,
        every attempt that reduces the error count is BANKED as a checkpoint
        commit, and failure reverts only to the last checkpoint. `ui` is a Dash
        (shared cockpit) or None → SoloStatus one-liner."""
        from spiral.dash import SoloStatus

        owns_ui = ui is None
        if owns_ui:
            ui = SoloStatus().__enter__()
        try:
            return self._run(task, model, attempts, strict_green, ratchet, ui)
        finally:
            if owns_ui:
                ui.__exit__(None, None, None)

    @staticmethod
    def _error_sigs(out: str) -> set[str]:
        """Distinct error signatures in gate output — the ratchet's progress metric.

        Progress = previously-persistent errors DISAPPEARING, not the raw count
        shrinking: fixing file A lets the compiler advance into file B and reveal
        NEW errors, so the count can rise while real progress is made."""
        sigs = {
            ln.strip()[:160]
            for ln in out.splitlines()
            if re.search(r"^e: |error:|Unresolved reference|FAILURE:", ln.strip())
        }
        return sigs or {"(unrecognized failure)"}

    def _compact_tried(self, tried: list[str]) -> list[str]:
        """Microcompaction: when the
        attempt log grows long, the janitor model squeezes the older entries into
        one lesson line. The 1B rides alongside the big model — no swap."""
        if len(tried) <= 8:
            return tried
        older, recent = tried[:-4], tried[-4:]
        try:
            res = self.ol.chat(
                self.cfg.janitor.name,
                [{"role": "user", "content":
                    "Summarize these failed coding attempts in ONE short line: which "
                    "approaches were tried and must NOT be repeated.\n" + "\n".join(older)}],
                num_predict=120,
                num_ctx=self.cfg.janitor.num_ctx,
                keep_alive="10m",
            )
            line = " ".join(res.text.split())[:220]
            if line:
                return [f"(compacted history) {line}"] + recent
        except Exception:
            pass
        return tried[-8:]  # janitor unavailable → plain truncation

    def _hunt_symbols(self, err_line: str, files: list[str]) -> str:
        """When the same 'Unresolved reference' repeats, STOP the synonym roulette:
        grep the repo for the symbol (underscore/case-insensitive) and hand the
        model the actual definitions — plus pull those files into context."""
        m = re.search(r"[Uu]nresolved reference '?([A-Za-z_]\w*)'?", err_line)
        if not m:
            return ""
        sym = m.group(1)
        needle = sym.lower().replace("_", "")
        hits: list[str] = []
        for p in sorted(self.ws.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in TEXT_EXT:
                continue
            rel = p.relative_to(self.ws)
            if any(part in tools._SKIP_DIRS or part.startswith(".") for part in rel.parts):
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if needle in line.lower().replace("_", ""):
                        hits.append(f"  {rel}:{i}: {line.strip()[:110]}")
                        if str(rel) not in files and len(files) < 14:
                            files.append(str(rel))
                        break  # one hit per file is enough
            except Exception:
                continue
            if len(hits) >= 6:
                break
        if not hits:
            return (f"REPO FACTS: '{sym}' does not exist ANYWHERE in the repo — you must "
                    f"CREATE it (e.g. add the id/definition where it belongs), not reference it.")
        facts = ("REPO FACTS — actual occurrences related to "
                 f"'{sym}' (use the EXACT existing identifier, or add the missing one HERE):\n"
                 + "\n".join(hits))
        layouts = sorted({h.split(":")[0].strip() for h in hits if "/layout/" in h})
        if layouts:
            facts += (
                "\nVIEWBINDING SCOPE: an android:id belongs ONLY to its own layout's "
                f"binding class. Ids found in {', '.join(layouts)} are NOT reachable from "
                "another layout's binding (e.g. ActivityMainBinding). Either (a) reference "
                "them through that layout's own binding after inflating it, or (b) add "
                '<include android:id="@+id/x" layout="@layout/..."/> to the activity layout '
                "and use binding.x.viewId, or (c) define the view directly in the activity's "
                "own layout."
            )
        return facts

    def _clean_paths(self, out: str) -> str:
        """Relativize file:///abs/paths in tool output — kills terminal autolink
        noise and saves prompt tokens."""
        return out.replace(f"file://{self.ws}/", "").replace(f"{self.ws}/", "")

    def _run(self, task: TaskSpec, model: str | None, attempts: int | None, strict_green: bool, ratchet: bool, ui) -> bool:
        self._ensure_git()
        model_name = model or self.cfg.worker.name
        files = list(task.files) if task.files else _auto_files(self.ws)
        has_verify = bool(task.verify_cmd and task.verify_cmd.strip())

        # a task's declared artifacts must EXIST — a green gate only proves the
        # code that exists compiles, not that the feature was ever built
        def _missing() -> list[str]:
            return [f for f in (task.files or []) if not (self.ws / f).is_file()]

        if has_verify:
            ui.phase("verifying", model="gate")
            verify = tools.run(
                task.verify_cmd, self.ws, timeout=self.cfg.verify_timeout,
                on_line=lambda ln: ui.detail(ln),
            )
            missing = _missing()
            audit_mode = verify.ok and not missing
            if audit_mode:
                # green + files exist proves nothing about the BEHAVIOR this task
                # adds — audit instead of blind-skip (the 12/12 false-completion lesson)
                verify_out = (
                    "(the build gate is GREEN and this task's declared files exist. AUDIT the "
                    "FILES: if the TASK is already fully implemented, reply exactly ALREADY_DONE; "
                    "otherwise emit the minimal SEARCH/REPLACE blocks that complete the missing "
                    "behavior WITHOUT breaking the build.)"
                )
                ui.print("  [dim]○ gate green + artifacts exist — auditing implementation[/]")
            elif verify.ok and missing:
                verify_out = (
                    "(the build gate is GREEN — do not break it. But this task's artifacts "
                    f"do not exist yet and must be created: {', '.join(missing)})"
                )
                ui.print(f"  [yellow]○ gate green but artifacts missing:[/] [dim]{', '.join(missing)}[/]")
            else:
                verify_out = self._clean_paths(verify.out)
                self._absorb_error_files(verify_out, files)
        else:
            verify = None
            verify_out = "(no verify command — edits are applied without a ground-truth gate)"
            ui.print("  [yellow]⚠ no verify gate for this task — applying blind[/]")

        cards = match_skills(task.goal, self.skills, files=files)
        notes = next((c_ for c_ in self.skills if c_.name == "project-notes"), None)
        if notes and notes not in cards:
            cards.append(notes)  # user notes ALWAYS ride along
        skills_text = render_for_prompt(cards, budget=5_000) if cards else ""
        if cards:
            ui.print(f"  [dim]skills: {', '.join(c.name for c in cards)}[/]")

        apply_errs = ""
        budget = attempts or self.cfg.task_attempt_budget
        cap = self.cfg.worker_max_tokens
        no_parse_streak = 0
        best_sigs = self._error_sigs(verify_out) if (has_verify and ratchet) else set()
        stall = 0
        tried: list[str] = []          # per-task attempt memory — no more synonym roulette
        err_seen: dict[str, int] = {}  # repeated error sigs trigger the symbol hunter
        asks_used = 0
        asked: set[str] = set()
        repo_answers = ""
        scratch = self.ws / ".spiral" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while attempt < budget:
            attempt += 1
            ui.print(f"  [dim]— attempt {attempt}/{budget} · {model_name} —[/]")
            tried = self._compact_tried(tried)
            prompt = self._prompt(task, files, verify_out, apply_errs, skills_text, tried, repo_answers)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]

            ui.phase("building", model=model_name)
            t_gen = time.time()
            tail = ""

            def _delta(kind: str, piece: str) -> None:
                nonlocal tail
                ui.tick()
                if kind == "text":
                    tail = (tail + piece)[-400:]
                    m = re.findall(r"[\w][\w./-]*\.(?:kt|kts|xml|java|gradle|py|md)", tail)
                    if m:
                        ui.detail(f"✎ {max(m, key=len)}")

            spec_m = self.cfg.spec_for(model_name)
            res = self.ol.chat(
                model_name, msgs, think=False,
                num_predict=cap,
                num_ctx=spec_m.num_ctx,
                keep_alive=self.cfg.keep_alive,
                on_delta=_delta,
            )
            self.tokens += res.total_tokens
            gen_s = round(time.time() - t_gen, 1)
            (scratch / "last_reply.txt").write_text(res.text)  # always inspectable

            ask = re.match(r"^ASK:\s*(grep|file)\s+(.+?)\s*$", res.text.strip()[:300], re.I | re.M)
            if ask:
                what, q = ask.group(1).lower(), ask.group(2).strip()
                key = f"{what}:{q}"
                if asks_used >= 2 or key in asked:
                    apply_errs = "ASK budget exhausted or repeated — emit SEARCH/REPLACE edits with what you have."
                    ui.print(f"  [yellow]⌕ ask rejected ({'repeat' if key in asked else 'budget'}):[/] [dim]{what} {q[:50]}[/]")
                    continue
                asked.add(key)
                asks_used += 1
                attempt -= 1  # asks are cheap (no gate run) — they don't consume attempts
                if what == "grep":
                    answer = tools.grep(self.ws, q, max_hits=12)
                    self._absorb_error_files(answer, files)
                else:
                    answer = self._read(q, cap=6_000) or f"(no such file: {q})"
                    if (self.ws / q).is_file() and q not in files and len(files) < 14:
                        files.append(q)
                repo_answers += f"\n--- ASK {what} {q} ---\n{answer[:3000]}\n"
                n_hits = answer.count(chr(10)) + 1 if answer else 0
                ui.print(f"  [rgb(217,119,87)]⌕ ask:[/] {what} [bold]{q[:60]}[/] [dim]→ {n_hits} line(s) fed back[/]")
                self.ledger.log("ask", task=task.goal[:60], what=what, q=q[:80], lines=n_hits)
                continue

            blocks = parse_edits(res.text)
            if not blocks and "ALREADY_DONE" in res.text[:80]:
                ui.print("  [green]● audit: already implemented — nothing to do[/]")
                return True
            if not blocks:
                no_parse_streak += 1
                truncated = res.completion_tokens >= cap - 8
                preview = " ".join(res.text.split())[:70]
                why = "reply hit token cap mid-block" if truncated else "no SEARCH/REPLACE markers found"
                ui.print(f"  [yellow]○[/] no edits parsed — {why} ([dim]{res.completion_tokens} tok[/])")
                if preview:
                    ui.print(f"     [dim]reply starts: {preview!r}[/]")
                if truncated:
                    cap = min(12288, cap + 2048)  # give the next attempt room
                    apply_errs = (
                        "Your reply was CUT OFF at the token limit before any block closed. "
                        "Emit ONE small block: SEARCH of at most 12 lines around the first error only."
                    )
                else:
                    apply_errs = (
                        "You produced no valid SEARCH/REPLACE blocks. Reply ONLY with blocks in the "
                        "exact format — no prose, no code fences, SEARCH sections of ≤12 lines."
                    )
                if no_parse_streak >= 3:
                    ui.print("  [yellow]⇥ 3 unparseable replies in a row — stopping this lane early[/]")
                    break
                continue
            no_parse_streak = 0

            results = apply_edits(self.ws, blocks)
            applied = [r for r in results if r.ok]
            failed = [r for r in results if not r.ok]
            apply_errs = "\n".join(
                f"- {r.path}: {r.reason}"
                + (
                    f"\n  The ACTUAL file text closest to your SEARCH is:\n{r.hint}\n"
                    "  Copy the file text above EXACTLY into your next SEARCH — do not "
                    "write code from memory."
                    if r.hint else ""
                )
                for r in failed
            )
            for r in applied:
                if r.path not in files:
                    files.append(r.path)
            edits_desc = ", ".join(f"{r.path}({r.how})" for r in applied) or "none"

            if not has_verify:
                if applied:
                    head, moved = self._commit(f"spiral: {task.goal[:60]}")
                    ui.print(f"  [yellow]●[/] edits: {edits_desc} · [dim]{res.total_tokens} tok[/] · [yellow]unverified[/]")
                    ui.print(f"  [green]✔ committed[/] [dim]{head}[/] [yellow](no gate)[/]")
                    return True
                ui.print("  [yellow]○[/] no edits applied")
                continue

            ui.phase("verifying", model="gate")
            verify = tools.run(
                task.verify_cmd, self.ws, timeout=self.cfg.verify_timeout,
                on_line=lambda ln: ui.detail(ln),
            )
            verify_out = self._clean_paths(verify.out)
            self._absorb_error_files(verify_out, files)
            mark = "[green]●[/]" if verify.ok else "[red]●[/]"
            ui.print(f"  {mark} edits: {edits_desc} · verify exit {verify.code} · [dim]{res.total_tokens} tok[/]")
            self.ledger.log(
                "attempt", task=task.goal[:80], model=model_name, attempt=attempt,
                ptok=res.prompt_tokens, ctok=res.completion_tokens, gen_s=gen_s,
                tps=round(res.completion_tokens / gen_s, 1) if gen_s > 0 else None,
                applied=len(applied), failed=len(failed), verify_exit=verify.code,
            )
            st = self.run_stats
            st["attempts"] += 1
            st["ptok"] += res.prompt_tokens
            st["ctok"] += res.completion_tokens
            if verify.ok:
                st["green"] += 1
            if gen_s > 0:
                st["tps"].setdefault(model_name, []).append(res.completion_tokens / gen_s)
            if failed:
                ui.print(f"  [yellow]○[/] {len(failed)} block(s) didn't apply")
            if not verify.ok:
                err = next(
                    (ln.strip() for ln in verify_out.splitlines()
                     if re.search(r"^e: |error:|Unresolved|FAILURE:|Caused by", ln)),
                    "",
                )
                if err:
                    ui.print(f"     [dim]gate says: {err[:110]}[/]")
                # attempt memory: what was tried, what it produced
                tried.append(f"attempt {attempt} [{model_name}]: {edits_desc} → {err[:90] or f'exit {verify.code}'}")
                # repeated identical error → hunt the symbol in the repo
                sig = err[:120]
                err_seen[sig] = err_seen.get(sig, 0) + 1
                if sig and err_seen[sig] == 2:
                    facts = self._hunt_symbols(err, files)
                    if facts:
                        apply_errs = (apply_errs + "\n\n" if apply_errs else "") + facts
                        ui.print("  [rgb(217,119,87)]⌕ symbol hunt — feeding repo facts back[/]")

            if verify.ok:
                if audit_mode and not applied:
                    ui.print("  [yellow]○ gate was already green and no edit landed — not accepting as done[/]")
                    continue
                still_missing = _missing()
                if still_missing:
                    verify_out = (
                        "(build gate is GREEN — do not break it. But these declared files "
                        f"still do not exist and must be created: {', '.join(still_missing)})"
                    )
                    ui.print(f"  [yellow]○ green but artifacts still missing:[/] [dim]{', '.join(still_missing)}[/]")
                    continue
                head, moved = self._commit(f"spiral: {task.goal[:60]}")
                if audit_mode and not moved:
                    ui.print("  [yellow]○ edits applied but changed NOTHING — not accepting as done[/]")
                    apply_errs = (
                        "Your last edits were no-ops (the file content did not change). "
                        "Make a REAL change that implements the task."
                    )
                    continue
                ui.print(f"  [green]✔ committed[/] [dim]{head}[/]")
                return True

            if ratchet:
                now_sigs = self._error_sigs(verify_out)
                resolved = best_sigs - now_sigs
                fresh = now_sigs - best_sigs
                if resolved:
                    head, _ = self._commit(
                        f"spiral: progress checkpoint — {len(resolved)} error(s) resolved, {len(fresh)} newly revealed"
                    )
                    ui.print(
                        f"  [rgb(217,119,87)]⚑ progress banked[/] [dim]{head} · resolved {len(resolved)}, revealed {len(fresh)}, remaining {len(now_sigs)}[/]"
                    )
                    best_sigs = now_sigs
                    stall = 0
                else:
                    stall += 1
                    if stall >= 3:
                        ui.print("  [yellow]⇥ no errors resolved in 3 attempts — stopping this lane[/]")
                        break

        ui.print("  [red]✗ budget exhausted[/] — checkpoint saved")
        if verify is not None:
            self._checkpoint(task, verify)
        if ratchet:
            tools.run("git reset --hard -q && git clean -fdq", self.ws)
            ui.print("  [yellow]⟲ reverted to last progress checkpoint[/] (banked work kept)")
        elif strict_green:
            tools.run("git reset --hard -q && git clean -fdq", self.ws)
            ui.print("  [yellow]⟲ reverted to last green commit[/] (green-to-green invariant)")
        return False
