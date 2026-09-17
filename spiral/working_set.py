"""Size source views using the selected vocabulary, without claiming admission.

All non-source prompt sections stay intact. Only the source-body allowance varies;
omitted originals remain reachable. Final chat-template and memory admission still
belong to the generation backend, not to these raw-text measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class WorkingSetTooSmall(RuntimeError):
    pass


@dataclass(frozen=True)
class SizedWorkingSet:
    prompt: str
    pages: tuple[Any, ...]
    body_characters: int
    raw_text_tokens: int
    context_tokens: int
    output_reserved: int
    framing_reserved: int
    trials: int


def size_working_set(render: Callable, meter, system: str, *,
                     context_tokens: int, output_reserved: int,
                     max_characters: int, checkpoint=lambda: None) -> SizedWorkingSet:
    if any(type(value) is not int or value < 0 for value in
           (context_tokens, output_reserved, max_characters)):
        raise ValueError("working-set capacity must use non-negative integer bounds")
    if context_tokens == 0 or context_tokens > 1_048_576:
        raise ValueError("invalid configured worker context")
    # This is explicitly a reserve, never a claim to have measured a template.
    # It also leaves space for host budget metadata added after Atom renders.
    framing = 2048
    available = context_tokens - output_reserved - framing
    if available <= 0:
        raise WorkingSetTooSmall("Selected context cannot reserve output and chat framing; settings were not enlarged.")
    high = min(4_000_000, max_characters)
    trials = 0

    def measure(body):
        nonlocal trials
        checkpoint()
        prompt, pages = render(body)
        # Bound every helper item even for four-byte Unicode. Chunk boundaries
        # mean the SUM is a packing measurement, not exact rendered-chat tokens.
        chunks = [text[start:start+125_000] for text in (system, prompt)
                  for start in range(0, len(text), 125_000)]
        counts = meter.count(chunks)
        if (len(counts) != len(chunks) or any(type(n) is not int or n < 0 for n in counts)):
            raise ValueError("invalid raw-text counter result")
        trials += 1
        return SizedWorkingSet(prompt, tuple(pages), body, sum(counts),
                               context_tokens, output_reserved, framing, trials)

    full = measure(high)
    if full.raw_text_tokens <= available:
        return full
    best = measure(0)
    if best.raw_text_tokens > available:
        raise WorkingSetTooSmall(
            f"Task/control evidence alone measures {best.raw_text_tokens} raw text tokens; "
            f"only {available} remain after output/framing reserves. Split the task or "
            "choose a supported larger context; no task constraints were silently removed.")
    low = 0
    # Bounded search is conservative if tokenization is nonmonotonic. Return only
    # an actually measured fitting candidate, not an interpolated token claim.
    for _ in range(8):
        if high - low <= 256:
            break
        middle = (low + high) // 2
        candidate = measure(middle)
        if candidate.raw_text_tokens <= available:
            low, best = middle, candidate
        else:
            high = middle
    return SizedWorkingSet(best.prompt, best.pages, best.body_characters,
                           best.raw_text_tokens, context_tokens, output_reserved, framing, trials)
