"""spiral's terminal theme — one meaning per color, no accidental rainbow.

Palette contract:
  clay    — brand, structure, orchestration (milestones, task headers, spinner,
            banked progress, escalation: the state machine talking)
  green   — ground truth PASSED only (gate green, committed)
  red     — ground truth FAILED only (gate red, blocked, abort)
  yellow  — warnings needing eyes (blocks didn't apply, unparseable reply, no gate)
  dim     — all metadata (tokens, timers, paths, gate output)

Rich's auto-highlighter (cyan numbers, colored paths) is DISABLED everywhere —
that plus terminal file:// autolinks was most of the visual noise.
"""
from __future__ import annotations

import sys
import time

from rich.console import Console

CLAY = "rgb(217,119,87)"
OK = "green"
FAIL = "red"
WARN = "yellow"
META = "dim"


def make_console(**kwargs) -> Console:
    return Console(highlight=False, **kwargs)


def reveal(console: Console, *renderables, delay: float = 0.07) -> None:
    """Print lines one at a time with a small stagger, so the UI settles in
    beneath the banner instead of slamming down all at once. Piped output gets
    everything immediately — pacing is a TTY courtesy, never a log tax."""
    tty = sys.stdout.isatty()
    for r in renderables:
        console.print(r)
        if tty and delay > 0:
            time.sleep(delay)
