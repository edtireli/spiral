"""Smoke test — prove the local pipe end to end: reach Ollama, hit the worker
model with thinking off and a hard cap, report tokens and throughput.

    python scripts/smoke.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.config import Config  # noqa: E402
from spiral.llm import Ollama  # noqa: E402


def main() -> int:
    cfg = Config.load()
    ol = Ollama(cfg.base_url)

    version = ol.health()
    if not version:
        print(f"✗ ollama unreachable at {cfg.base_url}")
        return 1
    print(f"● ollama {version} · worker={cfg.worker.name} (think={cfg.worker.think})")

    t0 = time.time()
    res = ol.chat(
        cfg.worker.name,
        [{"role": "user", "content": "Reply with exactly two words: spiral online"}],
        think=cfg.worker.think,
        num_predict=32,
    )
    dt = time.time() - t0

    print(f"→ {res.text.strip()!r}")
    tps = res.completion_tokens / dt if dt > 0 else 0.0
    print(f"  {res.prompt_tokens} in / {res.completion_tokens} out · {tps:.1f} tok/s · {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
