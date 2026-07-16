"""The flight recorder — every model call, attempt, and verdict as JSONL.

Forensics-first principle: with per-attempt records (model, tokens, duration,
tok/s, edits, verify exit) and captured THINKING from the planning-family calls,
any regression is attributable after the fact — which is what makes it safe to
evolve many parts of spiral at once. Also the data source for the future bandit
lane router.

    .spiral/ledger.jsonl           one JSON object per line
    .spiral/scratch/thinking-*.txt reasoning transcripts of plan/critic/validate
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Ledger:
    def __init__(self, ws: str | Path):
        self.dir = Path(ws) / ".spiral"
        self.path = self.dir / "ledger.jsonl"

    def log(self, kind: str, **rec) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **rec}
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # the recorder must never crash the flight

    def thinking(self, phase: str, text: str | None) -> None:
        if not text:
            return
        try:
            d = self.dir / "scratch"
            d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            (d / f"thinking-{phase}-{stamp}.txt").write_text(text)
        except Exception:
            pass
