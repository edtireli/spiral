"""`dict.get(key, default)` does not do what the call site usually means.

    {"arxiv_id": None}.get("arxiv_id", "paper")   ->   None,  not "paper"

The default applies when the key is *absent*, not when it is present and null. Every
source spiral reads — model-authored JSON, Semantic Scholar, Europe PMC, Crossref —
emits explicit nulls routinely, so the default silently fails to apply and the next
subscript or method call raises. That exact trap has now cost this project three
separate crashes: `{"data": null}` from the citation graph, `{"arxiv_id": null}` from a
reading note (which killed a 50-minute research run), and a chain of near-misses.

``pick`` is the version that means what the call site intends: treat null like missing.
"""
from __future__ import annotations

from typing import Any


def pick(mapping: Any, key: str, default: Any = "") -> Any:
    """``mapping[key]`` when it is neither missing nor None, else ``default``."""
    if not isinstance(mapping, dict):
        return default
    value = mapping.get(key)
    return default if value is None else value


def pick_str(mapping: Any, key: str, default: str = "") -> str:
    """``pick`` coerced to str — for the very common 'label it, then slice it' case."""
    value = pick(mapping, key, default)
    return value if isinstance(value, str) else (default if value is None else str(value))


def pick_list(mapping: Any, key: str) -> list:
    """``pick`` coerced to a list, so ``for x in pick_list(...)`` is always safe."""
    value = pick(mapping, key, [])
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [] if value == "" else [value]


def pick_dict(mapping: Any, key: str) -> dict:
    """``pick`` coerced to a dict, so chained ``.get`` never lands on None."""
    value = pick(mapping, key, {})
    return value if isinstance(value, dict) else {}
