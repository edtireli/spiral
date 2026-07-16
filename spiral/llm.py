"""Ollama client — the local backend behind the swappable seam.

Deliberately thin: chat (blocking + streaming), thinking toggle, hard token cap,
and token accounting from Ollama's own eval counts. No provider lock-in leaks
past this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    thinking: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Ollama:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 1200.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def evict(self, model: str) -> None:
        """Explicitly unload a model (keep_alive=0 on an empty generate) — swap
        discipline beats letting two 20GB models thrash under RAM pressure."""
        try:
            self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
            )
        except Exception:
            pass

    def health(self) -> str | None:
        """Return the server version, or None if unreachable."""
        try:
            r = self._client.get(f"{self.base_url}/api/version")
            r.raise_for_status()
            return r.json().get("version")
        except Exception:
            return None

    def _payload(
        self,
        model: str,
        messages: list[dict],
        *,
        think: bool,
        num_predict: int | None,
        temperature: float,
        stop: list[str] | None,
        fmt: Any | None,
        num_ctx: int | None = None,
        keep_alive: Any | None = None,
    ) -> dict:
        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict
        # CRITICAL: Ollama's server default context is 4096 regardless of the
        # model's native window — an unset num_ctx silently TRUNCATES long
        # prompts (system prompt first). Always pass it.
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if stop:
            options["stop"] = stop
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "think": think,
            "options": options,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive  # stop 5-min idle unloads mid-run
        if fmt is not None:
            payload["format"] = fmt  # "json" or a JSON-schema dict for structured output
        return payload

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        think: bool = False,
        num_predict: int | None = None,
        temperature: float = 0.2,
        stop: list[str] | None = None,
        fmt: Any | None = None,
        on_delta: Any | None = None,
        num_ctx: int | None = None,
        keep_alive: Any | None = None,
    ) -> ChatResult:
        """One call, two modes. Without on_delta: blocking. With on_delta: streams,
        calling on_delta(kind, piece) per chunk (kind: 'think' | 'text') so a UI can
        tick tokens live — the difference between a CLI that feels dead and alive."""
        import json as _json

        payload = self._payload(
            model, messages, think=think, num_predict=num_predict,
            temperature=temperature, stop=stop, fmt=fmt,
            num_ctx=num_ctx, keep_alive=keep_alive,
        )
        if on_delta is None:
            payload["stream"] = False
            r = self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {}) or {}
            return ChatResult(
                text=msg.get("content", ""),
                thinking=msg.get("thinking"),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                raw=data,
            )

        payload["stream"] = True
        text_parts: list[str] = []
        think_parts: list[str] = []
        last: dict = {}
        with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = _json.loads(line)
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    think_parts.append(msg["thinking"])
                    on_delta("think", msg["thinking"])
                if msg.get("content"):
                    text_parts.append(msg["content"])
                    on_delta("text", msg["content"])
                if chunk.get("done"):
                    last = chunk
        return ChatResult(
            text="".join(text_parts),
            thinking="".join(think_parts) or None,
            prompt_tokens=last.get("prompt_eval_count", 0),
            completion_tokens=last.get("eval_count", 0),
            raw=last,
        )

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        *,
        think: bool = False,
        num_predict: int | None = None,
        temperature: float = 0.2,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield content deltas as they arrive (for the live TUI later)."""
        import json

        payload = self._payload(
            model, messages, think=think, num_predict=num_predict,
            temperature=temperature, stop=stop, fmt=None,
        )
        payload["stream"] = True
        with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    yield piece
