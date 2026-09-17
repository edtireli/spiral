"""Replay completed planning stages, never tool effects or completion evidence.

A stage becomes reusable only when its caller returns normally after validation.
On resume the same planner/parser runs again over its saved public responses. A
different request, model binding, or planner implementation requires new inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from spiral.execution import BudgetExceeded
from spiral.llm import ChatResult
from spiral.runtime_control import checkpoint

_MAX_BYTES = 1024 * 1024
_MAX_CALLS = 12


def _encoded(value):
    # Preserve schema property order: constrained decoders can depend on it.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class PlanningStage:
    def __init__(self, workspace: Path, name: str, client, *, replay=False,
                 implementation: str, on_event=lambda **event: None):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
            raise ValueError("invalid planning stage")
        self.client, self.name = client, name
        self.workspace = workspace.resolve()
        self.path = self.workspace / ".spiral" / "planning-checkpoints" / (name + ".json")
        self.implementation = implementation
        self.on_event = on_event
        self.entries = []
        self.saved = []
        self.replay = replay

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _confined(self):
        return (self.path.resolve().is_relative_to(self.workspace)
                and not self.path.is_symlink())

    def __enter__(self):
        if self.replay and self._confined():
            try:
                if self.path.stat().st_size <= _MAX_BYTES:
                    record = json.loads(self.path.read_bytes())
                    entries = record["entries"]
                    if (record["version"] == 1 and isinstance(entries, list)
                            and len(entries) <= _MAX_CALLS
                            and record["sha256"] == hashlib.sha256(_encoded(entries)).hexdigest()):
                        self.saved = entries
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return self

    def _binding(self, model):
        from spiral.context_measurement import ContextMeasurementUnavailable
        factory = getattr(self.client, "source_tokenizer", None)
        if not callable(factory):
            return None
        try:
            meter = factory(model)
            if (meter is not None and meter.model == model
                    and isinstance(getattr(meter, "inference_identity", None), str)):
                meter._check()
                return meter.inference_identity
        except (OSError, ValueError, ContextMeasurementUnavailable):
            pass
        # A mutable model tag alone cannot authenticate a saved model response.
        return None

    def chat(self, model, messages, **options):
        checkpoint()
        budget = getattr(self.client, "budget", None)
        if budget is not None and budget.exhausted:
            raise BudgetExceeded(budget.exhausted_dimension(), budget.snapshot())
        binding = self._binding(model)
        key = None
        if binding:
            request = {"implementation": self.implementation, "model": model,
                       "binding": binding, "messages": messages,
                       "options": {k: v for k, v in options.items() if k != "on_delta"}}
            key = hashlib.sha256(_encoded(request)).hexdigest()
        index = len(self.entries)
        saved = self.saved[index] if index < len(self.saved) else None
        if key and isinstance(saved, dict) and saved.get("request_sha256") == key:
            reply = saved.get("reply")
            if (isinstance(reply, dict) and isinstance(reply.get("text"), str)
                    and isinstance(reply.get("done_reason"), str)):
                self.entries.append(saved)
                self.on_event(stage=self.name, outcome="replayed", request_sha256=key,
                              original_tokens=saved.get("original_tokens"), inference_called=False)
                # No token callbacks: this is a restored response, not live streaming.
                return ChatResult(reply["text"], 0, 0, raw={
                    "done_reason": reply["done_reason"],
                    "planning_checkpoint": {"stage": self.name, "request_sha256": key,
                                            "original_tokens": saved.get("original_tokens")},
                })
        # A changed prefix invalidates the remainder of this saved stage.
        self.saved = []
        result = self.client.chat(model, messages, **options)
        self.entries.append({"request_sha256": key, "reply": {
            "text": result.text, "done_reason": result.done_reason,
        }, "original_tokens": {"prompt": result.prompt_tokens,
                               "completion": result.completion_tokens}})
        return result

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            # Preserve public failed proposals for diagnosis without making them
            # reusable. The last good checkpoint remains untouched.
            try:
                self._archive_failure(exc_type, exc)
            except Exception:
                # Optional diagnostic storage/logging must not replace the
                # original rejection, Stop or exhausted-budget exception.
                pass
            return False
        if (not self.entries or len(self.entries) > _MAX_CALLS
                or any(not entry["request_sha256"] for entry in self.entries)
                or not self._confined()):
            return False
        content = _encoded({"version": 1, "entries": self.entries,
                           "sha256": hashlib.sha256(_encoded(self.entries)).hexdigest()})
        if len(content) > _MAX_BYTES:
            return False
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self._confined():
                return False
            fd, temporary = tempfile.mkstemp(prefix=".planning-", dir=self.path.parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self.on_event(stage=self.name, outcome="checkpointed", calls=len(self.entries))
        except OSError:
            # Durable acceleration is optional; it cannot turn a valid plan into failure.
            self.on_event(stage=self.name, outcome="storage_unavailable")
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
        return False

    def _archive_failure(self, exc_type, exc):
        if not self.entries:
            return
        temporary = None
        try:
            content = _encoded({"version": 1, "stage": self.name,
                "implementation": self.implementation, "reusable": False,
                "error_type": exc_type.__name__, "error": str(exc)[:2000],
                "entries": self.entries})
            if len(self.entries) > _MAX_CALLS or len(content) > _MAX_BYTES:
                self.on_event(stage=self.name, outcome="failure_archive_unavailable", reason="size_bound")
                return
            digest = hashlib.sha256(content).hexdigest()
            path = self.workspace / ".spiral/planning-failures" / f"{self.name}-{digest}.json"
            def confined():
                return path.resolve().is_relative_to(self.workspace) and not path.is_symlink()
            if not confined():
                raise ValueError("failure archive escapes workspace")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not confined():
                raise ValueError("failure archive escapes workspace")
            fd, temporary = tempfile.mkstemp(prefix=".failed-plan-", dir=path.parent)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self.on_event(stage=self.name, outcome="failure_archived", calls=len(self.entries),
                path=path.relative_to(self.workspace).as_posix(), sha256=digest, reusable=False)
        except (OSError, TypeError, ValueError):
            self.on_event(stage=self.name, outcome="failure_archive_unavailable")
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
