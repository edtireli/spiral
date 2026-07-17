"""spiral chat — a plain conversation with the local thinking model.

Streams token by token; the model's reasoning is shown dimmed, the answer in
normal weight. History is kept for the session and trimmed to fit the model's
context window. /exit or Ctrl-D to quit.
"""
from __future__ import annotations

import sys

import httpx

from spiral.config import Config
from spiral.llm import Ollama
from spiral.theme import make_console

CLAY = "\x1b[38;2;217;119;87m"
DIM = "\x1b[2m"
RST = "\x1b[0m"


def _fit(history: list[dict], budget_chars: int) -> list[dict]:
    """Drop whole oldest turns until the transcript fits the budget, always
    keeping the most recent exchange. Cheap char proxy for tokens — a long chat
    should degrade by forgetting the start, never by overflowing num_ctx."""
    total = sum(len(m["content"]) for m in history)
    while len(history) > 2 and total > budget_chars:
        total -= len(history.pop(0)["content"])
    return history


def chat(first: str = "", model: str | None = None) -> None:
    cfg = Config.load()
    ol = Ollama(cfg.base_url, providers=cfg.providers)
    model = model or cfg.planner.name
    c = make_console()
    c.print(f"  [dim]chat · {model} · reasoning shown dimmed · /exit or Ctrl-D to quit[/]\n")

    # leave ~40% of the window for the reply; ~4 chars/token is a safe proxy
    budget = int(cfg.planner.num_ctx * 4 * 0.6)
    history: list[dict] = []
    thinks = model in getattr(ol, "providers", {}) or True  # assumed; corrected lazily on a 400

    while True:
        if first:
            user, first = first, ""
            c.print(f"[bold rgb(217,119,87)]you ›[/] {user}")
        else:
            try:
                user = c.input("[bold rgb(217,119,87)]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                c.print("\n  [dim]bye[/]\n")
                return
        if not user:
            continue
        if user in ("/exit", "/quit", "exit", "quit"):
            c.print("  [dim]bye[/]\n")
            return

        history.append({"role": "user", "content": user})
        history = _fit(history, budget)
        sys.stdout.write(f"{CLAY}spiral ›{RST} ")
        sys.stdout.flush()
        in_think = False

        def on(kind: str, piece: str) -> None:
            nonlocal in_think
            if kind == "think":
                if not in_think:
                    sys.stdout.write("\n" + DIM)
                    in_think = True
                sys.stdout.write(piece)
            else:
                if in_think:
                    sys.stdout.write(RST + "\n\n")
                    in_think = False
                sys.stdout.write(piece)
            sys.stdout.flush()

        def generate(think: bool):
            return ol.chat(model, history, think=think, num_predict=cfg.planner_max_tokens,
                           num_ctx=cfg.planner.num_ctx, keep_alive=cfg.keep_alive, on_delta=on)

        try:
            try:
                res = generate(thinks)
            except httpx.HTTPStatusError as e:
                # some models have no reasoning channel → Ollama 400s. Learn once,
                # retry plainly; the 400 lands before any tokens, so nothing is lost.
                if thinks and e.response is not None and e.response.status_code == 400:
                    thinks = False
                    res = generate(False)
                else:
                    raise
        except KeyboardInterrupt:
            sys.stdout.write(RST + "\n  [interrupted]\n")
            history.pop()
            continue
        if in_think:
            sys.stdout.write(RST)
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        history.append({"role": "assistant", "content": res.text})
