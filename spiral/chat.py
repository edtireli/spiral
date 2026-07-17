"""spiral chat — a plain conversation with the local thinking model.

Streams token by token; the model's reasoning is shown dimmed, the answer in
normal weight. History is kept for the session. /exit or Ctrl-D to quit.
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


def _supports_think(ol: Ollama, model: str, cfg) -> bool:
    """Ollama returns 400 when asked to think on a model without a reasoning
    channel. Providers (API models) handle their own reasoning, so assume yes."""
    if model in getattr(ol, "providers", {}):
        return True
    try:
        ol.chat(model, [{"role": "user", "content": "hi"}], think=True,
                num_predict=1, keep_alive=cfg.keep_alive)
        return True
    except httpx.HTTPStatusError:
        return False
    except Exception:
        return True  # network/other errors surface on the real call, not here


def chat(first: str = "", model: str | None = None) -> None:
    cfg = Config.load()
    ol = Ollama(cfg.base_url, providers=cfg.providers)
    model = model or cfg.planner.name
    c = make_console()

    # not every model exposes a reasoning channel; probe once and degrade quietly.
    thinks = _supports_think(ol, model, cfg)
    tag = "thinking shown dimmed" if thinks else "no reasoning channel"
    c.print(f"  [dim]chat · {model} · {tag} · /exit or Ctrl-D to quit[/]\n")

    history: list[dict] = []
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

        try:
            res = ol.chat(model, history, think=thinks, num_predict=cfg.planner_max_tokens,
                          num_ctx=cfg.planner.num_ctx, keep_alive=cfg.keep_alive, on_delta=on)
        except KeyboardInterrupt:
            sys.stdout.write(RST + "\n  [interrupted]\n")
            history.pop()
            continue
        if in_think:
            sys.stdout.write(RST)
        sys.stdout.write("\n\n")
        sys.stdout.flush()
        history.append({"role": "assistant", "content": res.text})
