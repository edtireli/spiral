"""Ollama client — the local backend behind the swappable seam.

Deliberately thin: chat (blocking + streaming), thinking toggle, hard token cap,
and token accounting from Ollama's own eval counts. No provider lock-in leaks
past this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import contextlib
import httpx
import json
import os
import threading
import time
from pathlib import Path

from spiral.execution import BudgetExceeded, BudgetLimits, RunBudget


# One CLI process is one admitted Spiral run.  Track only exact local model names
# that process actually sends to Ollama; terminal cleanup must never discover and
# sweep somebody else's resident set.
_owned_local_models_lock = threading.Lock()
_owned_local_models: set[tuple[str, str]] = set()
_track_owned_local_models = False


def begin_owned_local_model_run() -> None:
    """Start one process-wide ownership receipt at the shared CLI boundary."""
    global _track_owned_local_models
    with _owned_local_models_lock:
        _owned_local_models.clear()
        _track_owned_local_models = True


def _remember_owned_local_model(base_url: str, model: str) -> None:
    if not model:
        return
    with _owned_local_models_lock:
        if _track_owned_local_models:
            _owned_local_models.add((base_url.rstrip("/"), model))


def release_owned_local_models(
    *, client_factory=None, timeout_seconds: float = 3.0,
) -> list[tuple[str, str]]:
    """Unload only the exact Ollama models used by the current CLI run.

    Long orchestration keeps its configured residency between steps.  This cleanup
    happens once, after success, failure, or interruption, while the process can still
    acquire the shared inference lease.  Hard-offline tests clear the receipt without
    constructing a client or opening a socket.
    """
    global _track_owned_local_models
    with _owned_local_models_lock:
        owned = sorted(_owned_local_models)
        _owned_local_models.clear()
        _track_owned_local_models = False
    if os.environ.get("SPIRAL_OFFLINE_TESTS"):
        return []
    factory = client_factory or Ollama
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    released: list[tuple[str, str]] = []
    by_endpoint: dict[str, list[str]] = {}
    for base_url, model in owned:
        by_endpoint.setdefault(base_url, []).append(model)
    for base_url, models in by_endpoint.items():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            client = factory(base_url, timeout=max(0.001, remaining), providers={})
        except Exception:
            continue
        try:
            for model in models:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    if client.evict(model, timeout=remaining):
                        released.append((base_url, model))
                except Exception:
                    # Terminal cleanup is best-effort and must never replace the
                    # run's real success/failure with a secondary unload error.
                    continue
        finally:
            try:
                client.close()
            except Exception:
                pass
    return released


class OfflineModelAccess(RuntimeError):
    """A test attempted to reach an inference service in hard-offline mode."""


def _reject_offline_request(request: httpx.Request) -> httpx.Response:
    raise OfflineModelAccess(
        f"SPIRAL_OFFLINE_TESTS forbids model-service access: {request.method} {request.url}"
    )

try:  # POSIX is the supported local-model host; keep imports harmless elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - Windows has no Ollama host integration
    fcntl = None


def _prompt_token_reserve(messages: list[dict]) -> int:
    """Conservative pre-dispatch prompt allowance for a hard total-token ceiling.

    Byte length upper-bounds ordinary byte-level tokenisation. The fixed framing margin
    covers chat role markers and provider wrappers that are not present in ``content``.
    This intentionally favours stopping early over silently exceeding the run ledger.
    """
    try:
        encoded = json.dumps(
            messages, ensure_ascii=False, separators=(",", ":"), default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = str(messages).encode("utf-8", errors="replace")
    return len(encoded) + 64 * len(messages) + 128


class InferenceLeaseTimeout(TimeoutError):
    pass


class InferenceLease:
    """Cross-process lease held only while a local inference response is active.

    The file and its <=4 KiB JSON owner record intentionally match SpiralChat's
    ``spiral-compute.lease`` flock protocol.  The controller supplies the exact
    shared path; no inherited descriptor is needed, so compilation, tests, file I/O,
    and human steering never pin the model lane.
    """

    PATH_ENV = "SPIRAL_MODEL_LEASE_PATH"
    OWNER_ENV = "SPIRAL_MODEL_LEASE_OWNER"
    TIMEOUT_ENV = "SPIRAL_MODEL_LEASE_TIMEOUT_SECONDS"

    def __init__(self, path: str | os.PathLike[str] | None, *, owner: dict | None = None,
                 timeout: float = 1200.0, poll: float = 0.05):
        self.path = Path(path).expanduser() if path else None
        self.owner = dict(owner or {})
        self.timeout = max(0.0, float(timeout))
        self.poll = max(0.01, float(poll))
        self._thread_lock = threading.Lock()

    @property
    def priority_dir(self) -> Path | None:
        return self.path.parent / "spiral-compute.priority" if self.path else None

    def _interactive_waiting(self) -> bool:
        """Return true for a live host-chat ticket and prune bounded stale tickets."""
        directory = self.priority_dir
        if directory is None:
            return False
        try:
            entries = list(directory.iterdir())
        except OSError:
            return False
        now = time.time()
        waiting = False
        for ticket in entries:
            if not ticket.name.startswith("interactive-") or ticket.suffix != ".json":
                continue
            try:
                stat = ticket.stat()
                payload = json.loads(ticket.read_text(encoding="utf-8")[:4096])
                pid = int(payload.get("pid") or 0)
                stale = now - stat.st_mtime > 300
                if pid > 0 and not stale:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        stale = True
                    except PermissionError:
                        pass
                if stale or pid <= 0:
                    ticket.unlink(missing_ok=True)
                else:
                    waiting = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                try:
                    if now - ticket.lstat().st_mtime > 300:
                        ticket.unlink(missing_ok=True)
                except OSError:
                    pass
        return waiting

    @classmethod
    def from_env(cls) -> "InferenceLease":
        raw_owner = os.environ.get(cls.OWNER_ENV, "")
        try:
            owner = json.loads(raw_owner) if raw_owner else {}
        except (TypeError, ValueError):
            owner = {"label": raw_owner[:200]}
        if not isinstance(owner, dict):
            owner = {"label": str(owner)[:200]}
        owner.setdefault("type", "spiral_inference")
        for env, key in (
            ("SPIRALCHAT_RUN_ID", "run_id"),
            ("SPIRALCHAT_ATTEMPT", "attempt"),
            ("SPIRALCHAT_REVISION", "revision"),
        ):
            if os.environ.get(env) and key not in owner:
                owner[key] = os.environ[env]
        try:
            timeout = float(os.environ.get(cls.TIMEOUT_ENV, "1200"))
        except ValueError:
            timeout = 1200.0
        return cls(os.environ.get(cls.PATH_ENV), owner=owner, timeout=timeout)

    @contextlib.contextmanager
    def hold(self, *, model: str, operation: str = "chat",
             timeout: float | None = None):
        if self.path is None or fcntl is None:
            yield
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._thread_lock:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            acquired = False
            try:
                os.chmod(self.path, 0o600)
                wait_for = self.timeout if timeout is None else min(self.timeout, max(0.0, timeout))
                deadline = time.monotonic() + wait_for
                while True:
                    if self._interactive_waiting():
                        if time.monotonic() >= deadline:
                            raise InferenceLeaseTimeout(
                                f"interactive chat kept local model priority for {wait_for:g}s")
                        time.sleep(min(self.poll, max(0.0, deadline - time.monotonic())))
                        continue
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        # Close the race where a chat ticket arrived between the check
                        # above and flock. Background calls yield at this inference edge.
                        if self._interactive_waiting():
                            fcntl.flock(fd, fcntl.LOCK_UN)
                            acquired = False
                            time.sleep(self.poll)
                            continue
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise InferenceLeaseTimeout(
                                f"local model lane stayed busy for {wait_for:g}s")
                        time.sleep(min(self.poll, max(0.0, deadline - time.monotonic())))
                record = {
                    "pid": os.getpid(),
                    "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    **self.owner,
                    "model": model,
                    "operation": operation,
                }
                raw = json.dumps(record, separators=(",", ":")).encode("utf-8")[:4096]
                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                view = memoryview(raw)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("could not persist model-lane owner")
                    view = view[written:]
                os.fsync(fd)
                yield
            finally:
                if acquired:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(fd)


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

    @property
    def spent_on_thinking(self) -> bool:
        """The model reasoned until the budget ran out and never began answering.

        ``num_predict`` is a cap on EVERYTHING a reasoning model emits, thinking
        included — so a budget sized for an answer is one a thinker can exhaust
        before it writes a word. The reply then comes back with prose in
        ``thinking``, an empty ``content``, and ``done_reason == "length"``.

        Reproduced against a local qwen3.6: with ``num_predict`` 16 and thinking on,
        content comes back empty and done_reason "length" while ``thinking`` holds
        the opening of a reasoning trace; with thinking off the same prompt answers
        in two tokens and stops. It surfaces as "returned no text (empty response)",
        which reads like a broken model rather than a budget that cut it off.
        """
        return bool(
            not (self.text or "").strip()
            and (self.thinking or "").strip()
            and (self.raw or {}).get("done_reason") == "length"
        )


class Ollama:
    """Local Ollama client, plus a routing seam: any model named in `providers`
    is dispatched to an OpenAI-compatible HTTP endpoint instead (e.g. a frontier
    reasoning model for the critic/escalation role while the worker stays local).
    Providers keep API keys in env vars, never in the config file."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 1200.0,
                 providers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self._no_think: set[str] = set()   # families that reject the thinking toggle
        self._timeout = timeout
        # The offline suite must be a physical no-network boundary, not merely a
        # convention that individual live tests remember to honor. A forgotten
        # health/residency probe must fail locally before opening a socket.
        transport = (
            httpx.MockTransport(_reject_offline_request)
            if os.environ.get("SPIRAL_OFFLINE_TESTS") else None
        )
        self._client = httpx.Client(timeout=timeout, transport=transport)
        loaded_cfg = None
        if providers is None:
            try:
                from spiral.config import Config
                loaded_cfg = Config.load()
                providers = loaded_cfg.providers
            except Exception:
                providers = {}
        self.providers = providers or {}
        self.inference_lease = InferenceLease.from_env()
        if loaded_cfg is None:
            try:
                from spiral.config import Config
                loaded_cfg = Config.load()
            except Exception:
                loaded_cfg = None
        limits = BudgetLimits.for_tier(
            getattr(loaded_cfg, "complexity_tier", "standard"))
        if loaded_cfg is not None:
            limits = BudgetLimits(
                int(getattr(loaded_cfg, "run_wall_budget_seconds", limits.wall_seconds)),
                int(getattr(loaded_cfg, "run_token_budget", limits.total_tokens)),
                int(getattr(loaded_cfg, "run_call_budget", limits.model_calls)),
            )
        self.budget = RunBudget(limits)

    def close(self) -> None:
        close = getattr(getattr(self, "_client", None), "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Ollama":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def configure_budget(self, *, wall_seconds: int, total_tokens: int,
                         model_calls: int, reset: bool = True) -> RunBudget:
        """Attach the finite run ledger used by every role sharing this client."""
        limits = BudgetLimits(int(wall_seconds), int(total_tokens), int(model_calls))
        if reset or not hasattr(self, "budget"):
            self.budget = RunBudget(limits)
        elif self.budget.limits != limits:
            # Never silently reset spent work when a conductor hands this client to
            # Atom. Narrow limits in place while retaining the joint accounting.
            self.budget.limits = limits
        return self.budget

    def _provider_minimum_completion(self, model: str, *, think: bool) -> int:
        """Minimum completion window this configured provider path will accept."""
        provider = self.providers.get(model)
        if not provider:
            return 1
        base = str(provider.get("base_url") or "")
        is_kimi = "moonshot" in base.lower() or model.lower().startswith("kimi-")
        is_kimi_k3 = model.lower().startswith("kimi-k3")
        if is_kimi_k3:
            default_floor = 131_072 if think else 32_768
            value = provider.get(
                "min_completion_tokens_thinking" if think
                else "min_completion_tokens_structured",
                default_floor,
            )
        elif is_kimi:
            value = provider.get("min_completion_tokens", 1024)
        else:
            value = provider.get("min_completion_tokens", 1024)
        return max(1, int(value))

    def _remaining_request_timeout(self) -> float:
        """HTTP timeout bounded by the run's remaining wall-clock allowance."""
        remaining = max(
            0.0,
            float(self.budget.limits.wall_seconds) - self.budget.elapsed_seconds,
        )
        if remaining <= 0:
            raise BudgetExceeded("wall", self.budget.snapshot())
        return max(0.001, min(float(getattr(self, "_timeout", 1200.0)), remaining))

    def evict(self, model: str, *, timeout: float | None = None) -> bool:
        """Explicitly unload a model (keep_alive=0 on an empty generate) — swap
        discipline beats letting two 20GB models thrash under RAM pressure."""
        try:
            lease = getattr(self, "inference_lease", None)
            budget = getattr(self, "budget", None)
            remaining = (
                budget.snapshot()["remaining"]["wall_seconds"]
                if budget is not None else None
            )
            deadline = (
                time.monotonic() + max(0.0, float(timeout))
                if timeout is not None else None
            )
            lease_timeout = remaining
            if deadline is not None:
                lease_timeout = max(0.0, deadline - time.monotonic())
                if remaining is not None:
                    lease_timeout = min(float(remaining), lease_timeout)
            scope = (lease.hold(model=model, operation="evict", timeout=lease_timeout)
                     if lease is not None else contextlib.nullcontext())
            with scope:
                request_timeout = (
                    max(0.001, deadline - time.monotonic())
                    if deadline is not None else None
                )
                if deadline is not None and request_timeout <= 0.001:
                    return False
                response = self._client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": 0},
                    **({"timeout": request_timeout} if request_timeout is not None else {}),
                )
                response.raise_for_status()
            return True
        except Exception:
            return False

    def evict_owned_local_models_except(
        self, keep: set[str], log=None,
    ) -> list[str]:
        """Evict only prior local models receipted by this CLI run.

        This is the safe role-switch seam for an intentional multi-model run.  It
        never asks Ollama what else is resident, so another app's model cannot be
        mistaken for memory owned by Spiral.  Without an active run receipt it is
        deliberately a no-op.
        """
        endpoint = self.base_url.rstrip("/")
        with _owned_local_models_lock:
            claimed = {
                (base_url, model) for base_url, model in _owned_local_models
                if _track_owned_local_models
                and base_url == endpoint
                and model not in keep
            }
            # Claim each receipt before the POST.  A successful A→B switch must
            # not send another empty generate for A on every later B call (which
            # some Ollama versions satisfy by loading A just to unload it again).
            _owned_local_models.difference_update(claimed)
        evicted: list[str] = []
        for receipt in sorted(claimed):
            _, model = receipt
            try:
                released = self.evict(model)
            except BaseException:
                with _owned_local_models_lock:
                    if _track_owned_local_models:
                        _owned_local_models.add(receipt)
                raise
            if not released:
                with _owned_local_models_lock:
                    if _track_owned_local_models:
                        _owned_local_models.add(receipt)
                continue
            evicted.append(model)
            if log:
                log(model)
        return evicted

    def resident(self) -> list[str]:
        """Model names Ollama currently holds in memory (/api/ps), newest first.

        Needed because spiral is no longer the only thing on this machine talking
        to Ollama: the phone chat keeps its own model loaded, and a second 18 GB
        model landing beside it is a Metal OOM, not a slow run."""
        try:
            return self.resident_strict()
        except Exception:
            return []

    def resident_strict(self) -> list[str]:
        """Return resident names or fail if admission cannot prove the state.

        UI/health callers may use ``resident()`` as a best-effort display.  Model
        admission must use this strict variant: timeout, HTTP error, TLS failure,
        or an unknown response shape cannot safely mean "no models resident".
        """
        response = self._client.get(f"{self.base_url}/api/ps")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("Ollama /api/ps returned an unknown response shape")
        names: list[str] = []
        for item in payload["models"]:
            if not isinstance(item, dict):
                raise ValueError("Ollama /api/ps returned a malformed model entry")
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ollama /api/ps returned a model without a name")
            names.append(name)
        return names

    def health(self) -> str | None:
        """Return the server version, or None if unreachable."""
        try:
            r = self._client.get(f"{self.base_url}/api/version")
            r.raise_for_status()
            return r.json().get("version")
        except Exception:
            return None

    def models(self) -> list[str]:
        """Installed model names, or [] if unreachable — callers treat an empty
        list as 'could not check' rather than 'nothing installed'."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", []) if m.get("name")]
        except Exception:
            return []

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
            "options": options,
        }
        if model not in getattr(self, "_no_think", set()):
            payload["think"] = think
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
        _recovering: bool = False,
    ) -> ChatResult:
        """Budgeted public model call; all roles share this one finite ledger."""
        if not hasattr(self, "budget"):
            self.budget = RunBudget(BudgetLimits.for_tier("standard"))
        minimum = self._provider_minimum_completion(model, think=think)
        requested = num_predict
        if model in self.providers and requested is not None:
            requested = max(int(requested), minimum)
        capped = self.budget.begin_call(
            requested,
            prompt_token_reserve=_prompt_token_reserve(messages),
            minimum_completion_tokens=minimum,
        )
        accounted_before = self.budget.total_tokens
        try:
            result = self._chat_impl(
                model, messages, think=think, num_predict=capped,
                temperature=temperature, stop=stop, fmt=fmt, on_delta=on_delta,
                num_ctx=num_ctx, keep_alive=keep_alive, _recovering=_recovering,
            )
        except BaseException:
            # A disconnected/timed-out provider may have completed the inference even
            # though no usage record arrived. Charge the entire admitted request. A
            # nested thinking-recovery path records its two attempts itself below.
            if self.budget.total_tokens == accounted_before:
                self.budget.record(_prompt_token_reserve(messages), capped)
            raise
        self.budget.record(result.prompt_tokens, result.completion_tokens)
        if isinstance(result.raw, dict):
            result.raw.setdefault("spiral_budget", self.budget.snapshot())
        return result

    def _chat_impl(
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
        _recovering: bool = False,
    ) -> ChatResult:
        """One call, two modes. Without on_delta: blocking. With on_delta: streams,
        calling on_delta(kind, piece) per chunk (kind: 'think' | 'text') so a UI can
        tick tokens live — the difference between a CLI that feels dead and alive.

        A reasoning model that spends the whole budget thinking is retried once
        with thinking off — see ``_answer_or_recover``."""
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
            lease = getattr(self, "inference_lease", None)
            remaining_wall = self.budget.snapshot()["remaining"]["wall_seconds"]
            scope = (lease.hold(model=model, timeout=remaining_wall)
                     if lease is not None else contextlib.nullcontext())
            with scope:
                request_timeout = self._remaining_request_timeout()
                _remember_owned_local_model(self.base_url, model)
                r = self._client.post(
                    f"{self.base_url}/api/chat", json=payload,
                    timeout=request_timeout,
                )
                if r.status_code == 400 and payload.pop("think", None) is not None:
                    # Rejected before inference: retrying the transport shape stays
                    # inside the same inference-scoped lease and logical call.
                    r = self._client.post(
                        f"{self.base_url}/api/chat", json=payload,
                        timeout=self._remaining_request_timeout(),
                    )
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {}) or {}
            return self._answer_or_recover(
                ChatResult(
                    text=msg.get("content", ""),
                    thinking=msg.get("thinking"),
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    raw=data,
                ),
                model, messages, think=think, num_predict=num_predict,
                temperature=temperature, stop=stop, fmt=fmt, on_delta=None,
                num_ctx=num_ctx, keep_alive=keep_alive, recovering=_recovering,
            )

        payload["stream"] = True
        text_parts: list[str] = []
        think_parts: list[str] = []
        last: dict = {}
        # not every family accepts the thinking toggle (gemma rejects what qwen
        # requires). A 400 arrives before any evaluation, so retrying without
        # `think` costs nothing; the model is remembered so later calls skip the
        # round-trip entirely. A plain loop — two tries at most.
        lease = getattr(self, "inference_lease", None)
        remaining_wall = self.budget.snapshot()["remaining"]["wall_seconds"]
        scope = (lease.hold(model=model, timeout=remaining_wall)
                 if lease is not None else contextlib.nullcontext())
        with scope:
            request_timeout = self._remaining_request_timeout()
            _remember_owned_local_model(self.base_url, model)
            for attempt in (1, 2):
                response = self._client.send(
                    self._client.build_request(
                        "POST", f"{self.base_url}/api/chat", json=payload,
                        timeout=request_timeout),
                    stream=True)
                if (response.status_code == 400 and attempt == 1
                        and payload.get("think") is not None):
                    response.close()
                    self._no_think.add(model)
                    payload.pop("think", None)
                    continue
                break
            with contextlib.closing(response) as r:
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
        return self._answer_or_recover(
            ChatResult(
                text="".join(text_parts),
                thinking="".join(think_parts) or None,
                prompt_tokens=last.get("prompt_eval_count", 0),
                completion_tokens=last.get("eval_count", 0),
                raw=last,
            ),
            model, messages, think=think, num_predict=num_predict,
            temperature=temperature, stop=stop, fmt=fmt, on_delta=on_delta,
            num_ctx=num_ctx, keep_alive=keep_alive, recovering=_recovering,
        )

    def _answer_or_recover(self, result: ChatResult, model: str,
                           messages: list[dict], *, think: bool, recovering: bool,
                           **call) -> ChatResult:
        """Return the reply, or buy an answer from a model that only thought.

        A reasoning model handed an answer-sized ``num_predict`` can spend all of
        it reasoning and return empty content. Retrying with a bigger budget only
        moves the wall — the thinking expands to fill it — so the retry turns
        thinking OFF, which is what actually guarantees the model starts writing
        immediately. Once, and only for a reply that produced literally nothing:
        a short answer is not this, and must not pay for a second call.

        The token cost of the wasted call is preserved in the returned result, so
        budget accounting still sees what was really spent.
        """
        if recovering or not think or not result.spent_on_thinking:
            return result
        requested = call.get("num_predict")
        minimum = self._provider_minimum_completion(model, think=False)
        if model in self.providers and requested is not None:
            requested = max(int(requested), minimum)
        try:
            capped = self.budget.begin_call(
                requested,
                prompt_token_reserve=_prompt_token_reserve(messages),
                minimum_completion_tokens=minimum,
                unrecorded_tokens=result.total_tokens,
            )
        except BaseException:
            # The first inference did happen. If recovery cannot even be admitted,
            # preserve its observed usage before the outer wrapper handles the error.
            self.budget.record(result.prompt_tokens, result.completion_tokens)
            raise
        try:
            again = self._chat_impl(
                model, messages, think=False, _recovering=True,
                **{**call, "num_predict": capped},
            )
        except BaseException:
            self.budget.record(
                result.prompt_tokens + _prompt_token_reserve(messages),
                result.completion_tokens + capped,
            )
            raise
        return ChatResult(
            text=again.text,
            thinking=result.thinking,
            prompt_tokens=result.prompt_tokens + again.prompt_tokens,
            completion_tokens=result.completion_tokens + again.completion_tokens,
            raw={**(again.raw or {}), "spiral_recovered_from_thinking": True},
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
        minimum_completion = self._provider_minimum_completion(model, think=think)
        completion_field = provider.get(
            "completion_token_field",
            "max_completion_tokens" if is_kimi else "max_tokens",
        )
        completion_cap = int(num_predict or 0)
        if completion_cap < minimum_completion:
            raise BudgetExceeded(
                "token",
                self.budget.snapshot(),
                detail=(
                    f"{model} requires at least {minimum_completion} completion "
                    f"tokens, but the run ledger admitted only {completion_cap}"
                ),
            )
        if is_kimi_k3:
            # K3 always reasons. Its API uses max_completion_tokens (max_tokens is
            # deprecated), and an 8k cap can be consumed entirely before final JSON.
            # Low effort keeps structured calls concise; free-form research gets room
            # for the deeper pass. Both remain overridable in provider configuration.
            body["reasoning_effort"] = provider.get(
                "reasoning_effort_thinking" if think else "reasoning_effort_structured",
                "max" if think else "low",
            )
            # A huge provider default is not a sensible unattended default: one blank
            # reasoning-only reply could consume it before Spiral gets a usable answer.
            # Give consequential calls a real reasoning window, then recover at low
            # effort if the provider still reaches the cap without final content.
            body[completion_field] = completion_cap
        elif is_kimi:
            # Kimi 2.x exposes an explicit thinking switch.
            body["thinking"] = {"type": "enabled" if think else "disabled"}
            body[completion_field] = completion_cap
        else:
            body[completion_field] = completion_cap
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
        initial_completion_cap = completion_cap
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
            attempt_prompt_reserve = _prompt_token_reserve(body["messages"])
            if attempt > 1:
                try:
                    body[completion_field] = self.budget.begin_retry(
                        initial_completion_cap,
                        prompt_token_reserve=attempt_prompt_reserve,
                        minimum_completion_tokens=minimum_completion,
                        unrecorded_tokens=spent_prompt + spent_completion,
                    )
                except BudgetExceeded as exc:
                    return ChatResult(
                        text="", prompt_tokens=spent_prompt,
                        completion_tokens=spent_completion,
                        raw={
                            **last_error,
                            "error": str(exc),
                            "spiral_budget_exhausted": exc.dimension,
                        },
                    )
            attempt_completion_cap = int(body[completion_field])
            try:
                if on_delta is None:
                    r = self._client.post(
                        f"{base}/chat/completions", headers=headers, json=body,
                        timeout=self._remaining_request_timeout(),
                    )
                    if r.status_code == 200:
                        d = r.json()
                        choice = (d.get("choices") or [{}])[0]
                        msg = choice.get("message", {}) or {}
                        u = d.get("usage", {}) or {}
                        spent_prompt += max(0, int(
                            u.get("prompt_tokens", attempt_prompt_reserve)
                            if u.get("prompt_tokens") is not None
                            else attempt_prompt_reserve
                        ))
                        spent_completion += max(0, int(
                            u.get("completion_tokens", attempt_completion_cap)
                            if u.get("completion_tokens") is not None
                            else attempt_completion_cap
                        ))
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
                    with self._client.stream(
                        "POST", f"{base}/chat/completions", headers=headers,
                        json=sbody, timeout=self._remaining_request_timeout(),
                    ) as r:
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
                            spent_prompt += max(0, int(
                                usage.get("prompt_tokens", attempt_prompt_reserve)
                                if usage.get("prompt_tokens") is not None
                                else attempt_prompt_reserve
                            ))
                            spent_completion += max(0, int(
                                usage.get("completion_tokens", attempt_completion_cap)
                                if usage.get("completion_tokens") is not None
                                else attempt_completion_cap
                            ))
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
                if r.status_code >= 500:
                    spent_prompt += attempt_prompt_reserve
                    spent_completion += attempt_completion_cap
                return ChatResult(text="", prompt_tokens=spent_prompt,
                                  completion_tokens=spent_completion,
                                  raw=last_error)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # The provider may have completed a timed-out request even though no
                # usage record reached us. Charge the full admitted request before any
                # retry; otherwise several ambiguous attempts can exceed the hard run
                # ceiling while each looks free locally.
                spent_prompt += attempt_prompt_reserve
                spent_completion += attempt_completion_cap
                last_error = {"error": f"{type(e).__name__}: {e}", "status": 0}
                time.sleep(min(2 ** attempt, 20))
            except BaseException:
                # Keyboard interruption, malformed provider data, and other unexpected
                # exits can happen after the remote model has consumed the request. Keep
                # earlier metered attempts and charge this ambiguous one at its admitted
                # maximum before preserving the original exception.
                self.budget.record(
                    spent_prompt + attempt_prompt_reserve,
                    spent_completion + attempt_completion_cap,
                )
                raise
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
        """Yield local content deltas under the same lease and joint budget."""
        import json

        if not hasattr(self, "budget"):
            self.budget = RunBudget(BudgetLimits.for_tier("standard"))
        prompt_reserve = _prompt_token_reserve(messages)
        capped = self.budget.begin_call(
            num_predict,
            prompt_token_reserve=prompt_reserve,
            minimum_completion_tokens=1,
        )
        payload = self._payload(
            model, messages, think=think, num_predict=capped,
            temperature=temperature, stop=stop, fmt=None,
        )
        payload["stream"] = True
        lease = getattr(self, "inference_lease", None)
        remaining_wall = self.budget.snapshot()["remaining"]["wall_seconds"]
        scope = (lease.hold(model=model, timeout=remaining_wall)
                 if lease is not None else contextlib.nullcontext())
        last: dict[str, Any] = {}
        try:
            with scope:
                request_timeout = self._remaining_request_timeout()
                _remember_owned_local_model(self.base_url, model)
                with self._client.stream(
                        "POST", f"{self.base_url}/api/chat", json=payload,
                        timeout=request_timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("done"):
                            last = chunk
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
        finally:
            self.budget.record(
                last.get("prompt_eval_count", prompt_reserve),
                last.get("eval_count", capped),
            )
