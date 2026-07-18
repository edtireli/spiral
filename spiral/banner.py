"""Terminal identity — a braille spiral generated the same way as the logomark,
plus a warm-gradient wordmark. `python -m spiral.banner` to see it.
"""
from __future__ import annotations

import math
import sys
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

CLAY = (217, 119, 87)       # #D97757 — spiral clay
CLAY_DEEP = (184, 84, 58)
CLAY_LITE = (235, 166, 128)

_BRAILLE_BASE = 0x2800
_DOT = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}


STYLES = ("spiral", "galaxy", "uzumaki")


def spiral_braille(cols: int = 20, rows: int = 9, turns: float = 2.4,
                   progress: float = 1.0, style: str = "spiral") -> list[str]:
    """Rasterize a spiral into a braille grid — same curve as the GUI mark.

    ``progress`` (0..1) draws only the inner portion, so animating it 0→1→0 makes
    the mark wind outward then retract — the "working" animation. ``style``:
    spiral (log, default) · galaxy (two log arms) · uzumaki (dense archimedean).
    """
    width, height = cols * 2, rows * 4
    cells = [[0] * cols for _ in range(rows)]
    cx, cy = width / 2, height / 2
    r_fit = min(width, height) / 2 - 1
    p = max(0.0, min(1.0, progress))

    def plot(r: float, ang: float) -> None:
        px, py = int(round(cx + r * math.cos(ang))), int(round(cy + r * math.sin(ang)))
        if 0 <= px < width and 0 <= py < height:
            cells[py // 4][px // 2] |= _DOT[(px % 2, py % 4)]

    if style == "uzumaki":
        tmax = turns * 1.7 * 2 * math.pi        # denser, hypnotic
        c = r_fit / tmax
        t = 0.0
        while t <= tmax * p:
            plot(c * t, t)
            t += 0.015
    else:
        tmax = turns * 2 * math.pi
        a = 0.9
        b = math.log(r_fit / a) / tmax
        arms = (0.0, math.pi) if style == "galaxy" else (0.0,)
        for off in arms:
            t = 0.0
            while t <= tmax * p:
                plot(a * math.exp(b * t), t + off)
                t += 0.02
    return ["".join(chr(_BRAILLE_BASE + cells[r][c]) for c in range(cols)) for r in range(rows)]


_WORDMARK_FALLBACK = ["s p i r a l"]


def _wordmark() -> list[str]:
    try:
        from pyfiglet import Figlet
        return Figlet(font="slant").renderText("spiral").rstrip("\n").splitlines()
    except Exception:
        return _WORDMARK_FALLBACK


def _lerp(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _rgb(c: tuple) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


CLASSIC = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """One-line 'working' indicator: braille spinner + phase + live tokens + elapsed
    + a detail tail (current file / gradle task / thinking).

    TTY: animates in place at ~12fps. Non-TTY (piped, background): emits a
    timestamped heartbeat line every few seconds — a background run must never
    look dead.
    """

    HEARTBEAT_S = 8.0

    def __init__(self, phase: str = "working", frames: str = CLASSIC):
        self._frames = frames
        self._phase = phase
        self._tokens = 0
        self._detail = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.time()
        self._tty = sys.stdout.isatty()

    def update(self, phase: str | None = None, tokens: int | None = None, detail: str | None = None) -> None:
        if phase is not None:
            self._phase = phase
        if tokens is not None:
            self._tokens = tokens
        if detail is not None:
            self._detail = detail.strip()[-64:]

    def tick(self, n: int = 1) -> None:
        self._tokens += n

    def _line(self) -> str:
        tok = f"{self._tokens / 1000:.1f}k tok" if self._tokens else "…"
        el = time.time() - self._t0
        base = f"{self._phase} · {tok} · {el:.0f}s"
        return f"{base} · {self._detail}" if self._detail else base

    def _render_tty(self, frame: str) -> None:
        clay, dim, rst = "\x1b[38;2;217;119;87m", "\x1b[2m", "\x1b[0m"
        sys.stdout.write(f"\r{clay}{frame}{rst} {clay}spiral{rst} {dim}· {self._line()}{rst}\x1b[K")
        sys.stdout.flush()

    def _loop(self) -> None:
        if self._tty:
            sys.stdout.write("\x1b[?25l")
            i = 0
            while not self._stop.is_set():
                self._render_tty(self._frames[i % len(self._frames)])
                i += 1
                time.sleep(0.08)
        else:
            # heartbeat mode: real newline-terminated lines for logs/monitors
            while not self._stop.wait(self.HEARTBEAT_S):
                stamp = time.strftime("%H:%M:%S")
                sys.stdout.write(f"  ⠿ [{stamp}] {self._line()}\n")
                sys.stdout.flush()

    def __enter__(self) -> "Spinner":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if self._tty:
            sys.stdout.write("\r\x1b[K\x1b[?25h")
            sys.stdout.flush()


def _type_in(text: str, p: float, start: float, end: float) -> str:
    """Slice `text` by eased progress within [start, end] — the wordmark and
    tagline type themselves in alongside the spiral's draw instead of popping."""
    if p >= end:
        return text
    if p <= start:
        return ""
    return text[: round(len(text) * (p - start) / (end - start))]


def _banner_frame(progress: float, tagline: str, style: str = "spiral"):
    """Compact banner: 4-row spiral beside tracked wordmark + tagline. Every
    element cascades with `progress` — nothing arrives all at once."""
    from rich.console import Group

    sp = spiral_braille(cols=9, rows=4, turns=2.2, progress=progress, style=style)
    lines = []
    for i, ln in enumerate(sp):
        t = Text("  " + ln, style=_rgb(CLAY))
        if i == 1:
            t.append("   " + _type_in("s p i r a l", progress, 0.20, 0.85),
                     style=f"bold {_rgb(CLAY)}")
        elif i == 2:
            t.append("   " + _type_in(tagline, progress, 0.55, 1.0), style="dim")
        lines.append(t)
    return Group(*lines)


def _current_style() -> str:
    try:
        from spiral.config import Config
        st = Config.load().spiral_style
        return st if st in STYLES else "spiral"
    except Exception:
        return "spiral"


def print_banner(console: Console | None = None, tagline: str = "local autonomous coder · on-device",
                 style: str | None = None, hold: float = 0.4) -> None:
    """Compact launch banner. On a TTY the spiral draws itself in once (~1.1s),
    settles briefly, and then the CLI's opening lines cascade in beneath it
    (theme.reveal) — the mark gets its moment without the elements slamming
    down all at once. Piped output gets the static banner only, no delay."""
    console = console or Console()
    style = style or _current_style()
    tty = sys.stdout.isatty()
    console.print()
    if tty:
        steps = 26
        with Live(_banner_frame(0.02, tagline, style), console=console,
                  refresh_per_second=30, transient=True) as live:
            for i in range(steps):
                p = (1 - math.cos(math.pi * (i + 1) / steps)) / 2  # eased 0→1
                live.update(_banner_frame(max(p, 0.02), tagline, style))
                time.sleep(1.1 / steps)
    console.print(_banner_frame(1.0, tagline, style))
    console.print()
    if tty and hold > 0:
        time.sleep(hold)


def play_draw(cycles: int = 3, period: float = 2.6, fps: int = 30) -> None:
    """Play the in→out braille draw in the terminal — the 'working' animation.

    (In v0's atom this becomes a live spinner that runs while the worker generates.)
    """
    console = Console()
    steps = int(cycles * period * fps)
    delay = 1.0 / fps
    with Live(console=console, refresh_per_second=fps, transient=True) as live:
        for i in range(steps):
            phase = 2 * math.pi * (i / (period * fps))
            p = (1 - math.cos(phase)) / 2
            lines = spiral_braille(progress=max(p, 0.001))
            live.update(Text("\n".join("  " + ln for ln in lines), style=_rgb(CLAY)))
            time.sleep(delay)


def play_vortex(seconds: float = 8.0, fps: int = 20) -> None:
    """Easter egg: the letters of 'spiral' ride the spiral itself, rotating.
    `python -m spiral.banner --vortex`"""
    w, h = 58, 20
    cx, cy = w / 2, h / 2
    tmax = 2.3 * 2 * math.pi
    a = 1.2
    b = math.log((min(w / 1.9, h / 0.85) / 2 - 1) / a) / tmax
    console = Console()

    def frame(phase: float) -> Text:
        grid = [[" "] * w for _ in range(h)]
        pts: list[tuple[int, int]] = []
        t = 0.0
        while t <= tmax:
            r = a * math.exp(b * t)
            x = int(round(cx + r * math.cos(t + phase) * 1.9))
            y = int(round(cy + r * math.sin(t + phase) * 0.85))
            if 0 <= x < w and 0 <= y < h and (not pts or pts[-1] != (x, y)):
                pts.append((x, y))
            t += 0.03
        for x, y in pts:
            grid[y][x] = "·"
        n = len(pts)
        for i, ch in enumerate("spiral"):
            x, y = pts[min(int(n * 0.35) + int(i * (n * 0.6) / 5), n - 1)]
            grid[y][x] = ch
        return Text("\n".join("  " + "".join(row) for row in grid), style=_rgb(CLAY))

    with Live(frame(0.0), console=console, refresh_per_second=fps, transient=True) as live:
        steps = int(seconds * fps)
        for i in range(steps):
            live.update(frame(i * 0.06))
            time.sleep(1 / fps)


if __name__ == "__main__":
    import sys

    if "--vortex" in sys.argv:
        play_vortex()
    elif "--animate" in sys.argv:
        play_draw()
    print_banner()
