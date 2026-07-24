"""The live cockpit — follow the plan and the work in real time.

TTY: a pinned panel (the plan with ✓/▶/○/✗ markers showing exactly where the run
is) plus an animated braille status line (phase · model · live tokens · elapsed ·
detail). Event lines print ABOVE the pinned region, like a transcript.

Non-TTY (piped/background): timestamped heartbeat lines — a log must never look
dead.

SoloStatus is the plan-less fallback (single `spiral do` tasks): same interface,
rendered as the one-line spinner.
"""
from __future__ import annotations

import collections
import json
import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group

from spiral.theme import make_console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from spiral.banner import CLASSIC, Spinner

CLAY = "rgb(217,119,87)"


class Dash:
    HEARTBEAT_S = 8.0

    def __init__(self, console: Console | None = None, plan=None, gate: str = "",
                 thought_log: str | Path | None = None):
        self.c = console or make_console()
        self.plan = plan
        self.gate = gate
        self.status: dict[tuple[int, int], str] = {}  # (mi,ti) -> run|done|blocked
        self._phase = "starting"
        self._model = ""
        self._detail = ""
        self._idea = ""
        self._ideas: collections.deque[dict[str, str]] = collections.deque(maxlen=200)
        self.thoughts_expanded = False
        self._thought_log = Path(thought_log) if thought_log else None
        if self._thought_log:
            try:
                self._thought_log.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self._thought_log = None
        self._tokens = 0
        self._t0 = time.time()
        self._phase_t0 = time.time()
        self._done = 0
        self._blocked = 0
        self._stop = threading.Event()
        self._live: Live | None = None
        self._thread: threading.Thread | None = None
        self.mode = ""      # 'auto'/'step' — shown in the status line when set
        self._paused = False
        self._samples: collections.deque = collections.deque(maxlen=60)  # (t, tok) for live t/s
        self._done_times: list[float] = []                               # green timestamps for ETA

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "Dash":
        if self.c.is_terminal:
            self._live = Live(self._render(), console=self.c, refresh_per_second=12,
                              vertical_overflow="crop")
            self._live.__enter__()
            self._thread = threading.Thread(target=self._anim_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._live:
            self._live.update(self._render(final=True))
            self._live.__exit__(*exc)

    def _anim_loop(self) -> None:
        while not self._stop.wait(0.08):
            if self._paused:
                continue
            try:
                self._live.update(self._render())
            except Exception:
                pass

    def _hb_loop(self) -> None:
        while not self._stop.wait(self.HEARTBEAT_S):
            stamp = time.strftime("%H:%M:%S")
            tok = f"{self._tokens / 1000:.1f}k tok" if self._tokens else "…"
            line = f"  ⠿ [{stamp}] {self._phase}"
            if self._model:
                line += f" [{self._model}]"
            line += f" · {tok} · {time.time() - self._phase_t0:.0f}s · Σ{(time.time() - self._t0) / 60:.0f}m"
            if self._detail:
                line += f" · {self._detail}"
            if self._idea:
                line += f" · idea: {self._idea[:120]}"
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    # -- mutations ---------------------------------------------------------------
    def phase(self, name: str, model: str = "") -> None:
        self._phase, self._model = name, model
        self._phase_t0 = time.time()
        self._detail = ""
        self._samples.clear()  # t/s is per-generation, not across phases

    def tick(self, n: int = 1) -> None:
        self._tokens += n
        self._samples.append((time.time(), self._tokens))

    def _tps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        (t0, k0), (t1, k1) = self._samples[0], self._samples[-1]
        dt = t1 - t0
        return (k1 - k0) / dt if dt > 0.5 else 0.0

    def set_tokens(self, n: int) -> None:
        self._tokens = n

    def detail(self, s: str) -> None:
        self._detail = (s or "").strip()[-70:]

    def idea(self, s: str) -> None:
        """Pinned high-level working note.

        This is intentionally a short hypothesis/status summary, not a raw chain of
        thought transcript. The cockpit needs the useful bit: what the model is
        currently trying or why it changed direction.
        """
        text = " ".join((s or "").replace("```", "").split())
        if not text:
            return
        text = text[:700]
        self._idea = text[:360]
        if self._ideas and self._ideas[-1].get("text") == text:
            return
        entry = {
            "ts": time.strftime("%H:%M:%S"),
            "phase": self._phase,
            "model": self._model,
            "text": text,
        }
        self._ideas.append(entry)
        if self._thought_log:
            try:
                with self._thought_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                self._thought_log = None

    def toggle_thoughts(self) -> None:
        self.thoughts_expanded = not self.thoughts_expanded

    def thought(self, piece: str, *, label: str = "thinking") -> None:
        """Convert a reasoning chunk into a safe, terse topic summary."""
        low = (piece or "").lower()
        topics = []
        for key, name in (
            ("error", "the current error"),
            ("import", "imports/symbols"),
            ("file", "the relevant files"),
            ("test", "the failing test"),
            ("gate", "the verification gate"),
            ("layout", "layout"),
            ("mobile", "mobile responsiveness"),
            ("contrast", "contrast"),
            ("overlap", "overlap/clipping"),
            ("citation", "citation grounding"),
            ("proof", "proof structure"),
            ("notation", "notation consistency"),
            ("novel", "novelty"),
            ("corpus", "corpus evidence"),
        ):
            if key in low and name not in topics:
                topics.append(name)
        if topics:
            self.idea(f"{label}: considering " + ", ".join(topics[:4]) + ".")
        else:
            self.idea(f"{label}: reasoning through the next decision.")

    def task(self, mi: int, ti: int, state: str) -> None:
        prev = self.status.get((mi, ti))
        self.status[(mi, ti)] = state
        if mi == 0:
            return  # M0 bootstrap shows in the panel but isn't a plan task (no 13/12)
        if state == "done" and prev != "done":
            self._done += 1
            self._done_times.append(time.time())
        if state == "blocked" and prev != "blocked":
            self._blocked += 1

    def print(self, *args, **kwargs) -> None:
        # with Live active on this console, prints land ABOVE the pinned region
        self.c.print(*args, **kwargs)

    def pause(self):
        """Context manager: suspend the pinned region for an interactive prompt."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self._paused = True
            if self._live:
                try:
                    self._live.stop()
                except Exception:
                    pass
            try:
                yield
            finally:
                if self._live:
                    try:
                        self._live.start(refresh=True)
                    except Exception:
                        pass
                self._paused = False
        return _cm()

    # -- render --------------------------------------------------------------------
    def _frame(self) -> str:
        return CLASSIC[int((time.time() - self._t0) / 0.08) % len(CLASSIC)]

    def _status_text(self, final: bool = False) -> Text:
        el = time.time() - self._phase_t0
        tok = f"{self._tokens / 1000:.1f}k tok" if self._tokens else "…"
        t = Text()
        t.append(f" {'●' if final else self._frame()} ", style=CLAY)
        t.append("spiral", style=f"bold {CLAY}")
        t.append(f" · {self._phase}", style="bold")
        if self._model:
            t.append(f" [{self._model}]", style="dim")
        t.append(f" · {tok} · {el:.0f}s", style="dim")
        tps = self._tps()
        if tps > 0:
            t.append(f" · {tps:.0f} t/s", style="dim")
        if self.mode:
            t.append(f" · {self.mode} ⇧⇥", style="dim" if self.mode == "auto" else "bold yellow")
        if self._detail:
            t.append(f"\n    {self._detail}", style="dim")
        return t

    _MARKS = {"done": ("✓", "green"), "run": ("▶", f"bold {CLAY}"), "blocked": ("✗", "red")}

    def _render(self, final: bool = False):
        parts = []
        if self._idea:
            parts.append(self._thoughts_panel())
        if self.plan is not None:
            rows: list[Text] = []
            if (0, 0) in self.status:
                mark, style = self._MARKS.get(self.status[(0, 0)], ("○", "dim"))
                rows.append(Text(f" {mark} M0 make the build gate pass", style=style))
            # Only EXPAND the active milestone's tasks — collapse the rest to header
            # lines. Keeps the pinned panel short enough to fit ANY terminal height
            # (a live region taller than the screen cascades — the cause of the spam).
            ms = self.plan.milestones
            active = next((mi for mi, m in enumerate(ms, 1)
                           if any(self.status.get((mi, ti)) == "run" for ti in range(1, len(m.tasks) + 1))), None)
            if active is None:
                active = next((mi for mi, m in enumerate(ms, 1)
                               if sum(1 for (a, _), s in self.status.items() if a == mi and s == "done") < len(m.tasks)), None)
            for mi, m in enumerate(ms, 1):
                mdone = sum(1 for (a, _), s in self.status.items() if a == mi and s == "done")
                done_all = mdone >= len(m.tasks)
                head = Text()
                head.append(f" ◆ M{mi} ", style=f"bold {CLAY}" if mi == active else ("green" if done_all else "dim"))
                head.append(m.title[:46], style="bold" if mi == active else "dim")
                head.append(f"  {mdone}/{len(m.tasks)}", style="dim")
                rows.append(head)
                if mi == active:
                    for ti, t in enumerate(m.tasks, 1):
                        s = self.status.get((mi, ti))
                        mark, style = self._MARKS.get(s, ("○", "dim"))
                        rows.append(Text(f"   {mark} {mi}.{ti} {t.title[:52]}", style=style))
            total = self.plan.task_count
            el = (time.time() - self._t0) / 60
            footer = (f"\n {self._done}/{total} green · {self._blocked} blocked · "
                      f"Σ {self._tokens / 1000:.1f}k tok · {el:.0f}m elapsed")
            remaining = total - self._done - self._blocked
            if len(self._done_times) >= 2 and remaining > 0:
                pace = (self._done_times[-1] - self._done_times[0]) / (len(self._done_times) - 1)
                footer += f" · eta ~{pace * remaining / 60:.0f}m"
            rows.append(Text(footer, style="dim"))
            parts.append(Panel(Group(*rows), title=f"[{CLAY}]⠷ plan[/]", border_style=CLAY, padding=(0, 1)))
        parts.append(self._status_text(final))
        return Group(*parts)

    def _thoughts_panel(self) -> Panel:
        if self.thoughts_expanded:
            rows = Text()
            for entry in list(self._ideas)[-8:]:
                rows.append(f"{entry.get('ts', '')} ", style="dim")
                phase = entry.get("phase") or "working"
                rows.append(f"{phase}: ", style=f"bold {CLAY}")
                rows.append(entry.get("text", ""), style="bold")
                rows.append("\n")
            if self._thought_log:
                rows.append(f"full log: {self._thought_log}", style="dim")
            title = f"[{CLAY}]thoughts[/] [dim]expanded · press t to collapse[/]"
            return Panel(rows, title=title, border_style=CLAY, padding=(0, 1))

        idea = Text()
        idea.append(self._idea, style="bold")
        title = f"[{CLAY}]thoughts[/] [dim]press t to expand[/]" if self._ideas else f"[{CLAY}]thoughts[/]"
        return Panel(idea, title=title, border_style=CLAY, padding=(0, 1))


class SoloStatus:
    """Same interface as Dash, no plan — renders the one-line Spinner.
    Used by standalone `spiral do` runs."""

    def __init__(self) -> None:
        self._sp: Spinner | None = None
        self.c = make_console()

    def __enter__(self) -> "SoloStatus":
        return self

    def __exit__(self, *exc) -> None:
        self._close()

    def _close(self) -> None:
        if self._sp:
            self._sp.__exit__(None, None, None)
            self._sp = None

    def phase(self, name: str, model: str = "") -> None:
        self._close()
        self._sp = Spinner(name).__enter__()

    def tick(self, n: int = 1) -> None:
        if self._sp:
            self._sp.tick(n)

    def set_tokens(self, n: int) -> None:
        if self._sp:
            self._sp.update(tokens=n)

    def detail(self, s: str) -> None:
        if self._sp:
            self._sp.update(detail=s)

    def idea(self, s: str) -> None:
        if self._sp:
            self._sp.update(detail=(" ".join((s or "").split()))[:70])

    def thought(self, piece: str, *, label: str = "thinking") -> None:
        self.idea(f"{label}: reasoning through the next decision.")

    def task(self, mi: int, ti: int, state: str) -> None:
        pass

    def print(self, *args, **kwargs) -> None:
        self._close()  # don't paint over the spinner line
        self.c.print(*args, **kwargs)
