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

from rich.console import Console

CLAY = "rgb(217,119,87)"
OK = "green"
FAIL = "red"
WARN = "yellow"
META = "dim"


def make_console(**kwargs) -> Console:
    return Console(highlight=False, **kwargs)
