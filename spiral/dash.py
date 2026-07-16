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

import sys
import threading
import time

from rich.console import Console, Group

from spiral.theme import make_console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from spiral.banner import CLASSIC, Spinner

CLAY = "rgb(217,119,87)"


class Dash:
    HEARTBEAT_S = 8.0

    def __init__(self, console: Console | None = None, plan=None, gate: str = ""):
        self.c = console or make_console()
        self.plan = plan
        self.gate = gate
        self.status: dict[tuple[int, int], str] = {}  # (mi,ti) -> run|done|blocked
        self._phase = "starting"
        self._model = ""
        self._detail = ""
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

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "Dash":
        if self.c.is_terminal:
            self._live = Live(self._render(), console=self.c, refresh_per_second=12)
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
            line += f" · {tok} · {time.time() - self._phase_t0:.0f}s"
            if self._detail:
                line += f" · {self._detail}"
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    # -- mutations ---------------------------------------------------------------
    def phase(self, name: str, model: str = "") -> None:
        self._phase, self._model = name, model
        self._phase_t0 = time.time()
        self._detail = ""

    def tick(self, n: int = 1) -> None:
        self._tokens += n

    def set_tokens(self, n: int) -> None:
        self._tokens = n

    def detail(self, s: str) -> None:
        self._detail = (s or "").strip()[-70:]

    def task(self, mi: int, ti: int, state: str) -> None:
        prev = self.status.get((mi, ti))
        self.status[(mi, ti)] = state
        if mi == 0:
            return  # M0 bootstrap shows in the panel but isn't a plan task (no 13/12)
        if state == "done" and prev != "done":
            self._done += 1
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
        if self.mode:
            t.append(f" · {self.mode} ⇧⇥", style="dim" if self.mode == "auto" else "bold yellow")
        if self._detail:
            t.append(f"\n    {self._detail}", style="dim")
        return t

    _MARKS = {"done": ("✓", "green"), "run": ("▶", f"bold {CLAY}"), "blocked": ("✗", "red")}

    def _render(self, final: bool = False):
        parts = []
        if self.plan is not None:
            rows: list[Text] = []
            if (0, 0) in self.status:
                mark, style = self._MARKS.get(self.status[(0, 0)], ("○", "dim"))
                rows.append(Text(f" {mark} M0 make the build gate pass", style=style))
            for mi, m in enumerate(self.plan.milestones, 1):
                mdone = sum(1 for (a, _), s in self.status.items() if a == mi and s == "done")
                head = Text()
                head.append(f" ◆ M{mi} ", style=f"bold {CLAY}")
                head.append(m.title[:52], style="bold")
                head.append(f"  {mdone}/{len(m.tasks)}", style="dim")
                rows.append(head)
                for ti, t in enumerate(m.tasks, 1):
                    s = self.status.get((mi, ti))
                    mark, style = self._MARKS.get(s, ("○", "dim"))
                    rows.append(Text(f"   {mark} {mi}.{ti} {t.title[:56]}", style=style))
            total = self.plan.task_count
            el = (time.time() - self._t0) / 60
            rows.append(Text(
                f"\n {self._done}/{total} green · {self._blocked} blocked · {el:.0f}m elapsed",
                style="dim",
            ))
            parts.append(Panel(Group(*rows), title=f"[{CLAY}]⠷ plan[/]", border_style=CLAY, padding=(0, 1)))
        parts.append(self._status_text(final))
        return Group(*parts)


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

    def task(self, mi: int, ti: int, state: str) -> None:
        pass

    def print(self, *args, **kwargs) -> None:
        self._close()  # don't paint over the spinner line
        self.c.print(*args, **kwargs)
