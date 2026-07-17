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
    """Local Ollama client, plus a routing seam: any model named in `providers`
    is dispatched to an OpenAI-compatible HTTP endpoint instead (e.g. a frontier
    reasoning model for the critic/escalation role while the worker stays local).
    Providers keep API keys in env vars, never in the config file."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 1200.0,
                 providers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        if providers is None:
            try:
                from spiral.config import Config
                providers = Config.load().providers
            except Exception:
                providers = {}
        self.providers = providers or {}

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

        if model in self.providers:
            return self._openai_chat(
                self.providers[model], model, messages, num_predict=num_predict,
                temperature=temperature, stop=stop, fmt=fmt, on_delta=on_delta,
            )

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

    # -- OpenAI-compatible provider path (remote reasoning models) ---------------
    def _openai_chat(
        self, provider: dict, model: str, messages: list[dict], *,
        num_predict: int | None, temperature: float, stop: list[str] | None,
        fmt: Any | None, on_delta: Any | None,
    ) -> ChatResult:
        import json as _json
        import os
        import time

        base = provider["base_url"].rstrip("/")
        key = os.environ.get(provider.get("api_key_env", "OPENAI_API_KEY"), "")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        body: dict[str, Any] = {"model": model, "messages": messages}
        # some reasoning models FIX temperature (kimi-k3: only 1) — provider wins
        body["temperature"] = provider["temperature"] if "temperature" in provider else temperature
        if num_predict:
            body["max_tokens"] = max(num_predict, 1024)  # reasoning models spend tokens thinking
        if stop:
            body["stop"] = stop
        if fmt is not None:
            body["response_format"] = {"type": "json_object"}  # JSON mode; schema enforced by our parser

        retries = int(provider.get("retries", 5))
        for attempt in range(1, retries + 1):
            try:
                if on_delta is None:
                    r = self._client.post(f"{base}/chat/completions", headers=headers, json=body)
                    if r.status_code == 200:
                        d = r.json()
                        msg = (d.get("choices") or [{}])[0].get("message", {}) or {}
                        u = d.get("usage", {}) or {}
                        return ChatResult(
                            text=msg.get("content", "") or "",
                            thinking=msg.get("reasoning_content"),
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            raw=d,
                        )
                else:
                    text_parts, think_parts, usage = [], [], {}
                    sbody = {**body, "stream": True, "stream_options": {"include_usage": True}}
                    with self._client.stream("POST", f"{base}/chat/completions", headers=headers, json=sbody) as r:
                        if r.status_code != 200:
                            r.read()
                        else:
                            for line in r.iter_lines():
                                if not line or not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                chunk = _json.loads(data)
                                if chunk.get("usage"):
                                    usage = chunk["usage"]
                                delta = (chunk.get("choices") or [{}])[0].get("delta", {}) or {}
                                if delta.get("reasoning_content"):
                                    think_parts.append(delta["reasoning_content"])
                                    on_delta("think", delta["reasoning_content"])
                                if delta.get("content"):
                                    text_parts.append(delta["content"])
                                    on_delta("text", delta["content"])
                            return ChatResult(
                                text="".join(text_parts),
                                thinking="".join(think_parts) or None,
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                            )
                # non-200 (both modes fall through here)
                err = r.json().get("error", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
                if r.status_code == 429 or "overload" in str(err.get("type", "")).lower():
                    time.sleep(min(2 ** attempt, 20))
                    continue
                return ChatResult(text="", prompt_tokens=0, completion_tokens=0,
                                  raw={"error": err or r.text[:300], "status": r.status_code})
            except (httpx.TimeoutException, httpx.TransportError):
                time.sleep(min(2 ** attempt, 20))
        return ChatResult(text="", prompt_tokens=0, completion_tokens=0, raw={"error": "provider unavailable after retries"})

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
