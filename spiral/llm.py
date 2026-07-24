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
                think=think,
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
        fmt: Any | None, on_delta: Any | None, think: bool,
    ) -> ChatResult:
        import json as _json
        import os
        import time

        base = provider["base_url"].rstrip("/")
        key_env = provider.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env, "")
        if not key:
            return ChatResult(text="", prompt_tokens=0, completion_tokens=0,
                              raw={"error": f"missing ${key_env}", "status": 401})
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        body: dict[str, Any] = {"model": model, "messages": messages}
        # some reasoning models FIX temperature (kimi-k3: only 1) — provider wins
        body["temperature"] = provider["temperature"] if "temperature" in provider else temperature
        is_kimi = "moonshot" in base.lower() or model.lower().startswith("kimi-")
        is_kimi_k3 = model.lower().startswith("kimi-k3")
        if is_kimi_k3:
            # K3 always reasons. Its API uses max_completion_tokens (max_tokens is
            # deprecated), and an 8k cap can be consumed entirely before final JSON.
            # Low effort keeps structured calls concise; free-form research gets room
            # for the deeper pass. Both remain overridable in provider configuration.
            body["reasoning_effort"] = provider.get(
                "reasoning_effort_thinking" if think else "reasoning_effort_structured",
                "max" if think else "low",
            )
            requested = max(int(num_predict or 0), 1024)
            # A huge provider default is not a sensible unattended default: one blank
            # reasoning-only reply could consume it before Spiral gets a usable answer.
            # Give consequential calls a real reasoning window, then recover at low
            # effort if the provider still reaches the cap without final content.
            default_floor = 131_072 if think else 32_768
            floor = int(provider.get(
                "min_completion_tokens_thinking" if think
                else "min_completion_tokens_structured",
                default_floor,
            ))
            body[provider.get("completion_token_field", "max_completion_tokens")] = max(
                requested, floor)
        elif is_kimi:
            # Kimi 2.x exposes an explicit thinking switch.
            body["thinking"] = {"type": "enabled" if think else "disabled"}
            if num_predict:
                body[provider.get("completion_token_field", "max_completion_tokens")] = max(
                    num_predict, 1024)
        elif num_predict:
            body[provider.get("completion_token_field", "max_tokens")] = max(
                num_predict, 1024)
        if stop:
            body["stop"] = stop
        if fmt is not None:
            body["response_format"] = {"type": "json_object"}  # JSON mode; schema enforced by our parser

        retries = max(1, int(provider.get("retries", 5)))
        blank_retries = max(0, int(provider.get("blank_retries", 1)))
        last_error: dict[str, Any] = {}
        spent_prompt = 0
        spent_completion = 0
        blank_count = 0
        base_messages = [dict(message) for message in messages]
        for attempt in range(1, retries + 1):
            if attempt > 1 and last_error.get("empty_response"):
                recovery = (
                    "The previous provider attempt emitted reasoning but no final answer. "
                    "Return the final answer now without restarting the analysis. "
                    + ("Emit one complete JSON object only." if fmt is not None
                       else "Be concise and directly answer the original request.")
                )
                body["messages"] = [
                    *base_messages,
                    {"role": "user", "content": recovery},
                ]
                if is_kimi_k3:
                    body["reasoning_effort"] = provider.get(
                        "reasoning_effort_recovery", "low")
            try:
                if on_delta is None:
                    r = self._client.post(f"{base}/chat/completions", headers=headers, json=body)
                    if r.status_code == 200:
                        d = r.json()
                        choice = (d.get("choices") or [{}])[0]
                        msg = choice.get("message", {}) or {}
                        u = d.get("usage", {}) or {}
                        spent_prompt += int(u.get("prompt_tokens", 0) or 0)
                        spent_completion += int(u.get("completion_tokens", 0) or 0)
                        content = msg.get("content", "") or ""
                        finish_reason = choice.get("finish_reason")
                        if content.strip():
                            return ChatResult(
                                text=content,
                                thinking=msg.get("reasoning_content"),
                                prompt_tokens=spent_prompt,
                                completion_tokens=spent_completion,
                                raw={
                                    **d,
                                    "finish_reason": finish_reason,
                                    "provider_attempts": attempt,
                                },
                            )
                        last_error = {
                            "error": "provider returned no final content",
                            "status": 200,
                            "finish_reason": finish_reason,
                            "empty_response": True,
                            "provider_attempts": attempt,
                        }
                        blank_count += 1
                        if attempt < retries and blank_count <= blank_retries:
                            continue
                        return ChatResult(
                            text="", prompt_tokens=spent_prompt,
                            completion_tokens=spent_completion, raw=last_error)
                else:
                    text_parts, think_parts, usage = [], [], {}
                    finish_reason = None
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
                                choice = (chunk.get("choices") or [{}])[0]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                                delta = choice.get("delta", {}) or {}
                                if delta.get("reasoning_content"):
                                    think_parts.append(delta["reasoning_content"])
                                    on_delta("think", delta["reasoning_content"])
                                if delta.get("content"):
                                    text_parts.append(delta["content"])
                                    on_delta("text", delta["content"])
                            spent_prompt += int(usage.get("prompt_tokens", 0) or 0)
                            spent_completion += int(usage.get("completion_tokens", 0) or 0)
                            content = "".join(text_parts)
                            if content.strip():
                                return ChatResult(
                                    text=content,
                                    thinking="".join(think_parts) or None,
                                    prompt_tokens=spent_prompt,
                                    completion_tokens=spent_completion,
                                    raw={
                                        "finish_reason": finish_reason,
                                        "provider_attempts": attempt,
                                    },
                                )
                            last_error = {
                                "error": "provider returned no final content",
                                "status": 200,
                                "finish_reason": finish_reason,
                                "empty_response": True,
                                "provider_attempts": attempt,
                            }
                            blank_count += 1
                            if attempt < retries and blank_count <= blank_retries:
                                continue
                            return ChatResult(
                                text="", prompt_tokens=spent_prompt,
                                completion_tokens=spent_completion,
                                raw=last_error)
                # non-200 (both modes fall through here)
                err: dict[str, Any] = {}
                if r.status_code != 200:
                    err = r.json().get("error", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
                    last_error = {"error": err or r.text[:300], "status": r.status_code}
                if r.status_code == 429 or "overload" in str(err.get("type", "")).lower():
                    time.sleep(min(2 ** attempt, 20))
                    continue
                return ChatResult(text="", prompt_tokens=spent_prompt,
                                  completion_tokens=spent_completion,
                                  raw=last_error)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = {"error": f"{type(e).__name__}: {e}", "status": 0}
                time.sleep(min(2 ** attempt, 20))
        return ChatResult(text="", prompt_tokens=spent_prompt,
                          completion_tokens=spent_completion,
                          raw=last_error or {"error": "provider unavailable after retries"})

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
