"""Explicit host-selected local model identity, shared across all engine seats."""
from __future__ import annotations

import os
import re


class ModelBindingError(ValueError):
    """A task must not silently change its selected model or provider."""


def selected_local_model() -> str:
    value = os.environ.get("SPIRAL_REQUIRED_LOCAL_MODEL", "")
    if value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", value) is None:
        raise ModelBindingError("SPIRAL_REQUIRED_LOCAL_MODEL is not an exact model name")
    return value


def require_selected_local_model(model: str, providers: dict) -> None:
    required = selected_local_model()
    if not required:
        return
    canonical = lambda name: name.removeprefix("registry.ollama.ai/library/").removesuffix(":latest")
    if canonical(model) != canonical(required):
        raise ModelBindingError(f"task is bound to local model {required!r}; refusing {model!r}")
    if model in providers or required in providers:
        raise ModelBindingError("selected local model must not be redirected to an API provider")
