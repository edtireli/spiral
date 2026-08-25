"""Shared train/runtime contract for the academic paper-structure adapter.

The authorization system message is deliberately separate from the prompt the
model sees.  Stage-two MLX completion rows contain one user message rendered by
``format_structure_prompt``; the serving process validates the system marker,
then removes it before applying the model chat template.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


BRIEF_TO_BLUEPRINT_TASK = "brief_to_blueprint"
MIN_BLUEPRINT_SECTIONS = 2
MAX_BLUEPRINT_SECTIONS = 12
BRIEF_TO_BLUEPRINT_INSTRUCTION = (
    "Construct the paper blueprint observed for this title and bounded abstract brief."
)
STRUCTURE_REQUEST_SYSTEM_MARKER = (
    "This call is only for the section outline and word budget."
)
STRUCTURE_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["paper_counts", "sections"],
}
STRUCTURE_EVIDENCE_CONSTRAINT = "observed_paper_structure_only"
STRUCTURE_DISCIPLINES = frozenset({
    "theoretical_physics",
    "particle_phenomenology",
    "biomedical_science",
})
STRUCTURE_GENRES = frozenset({
    "theory_article",
    "phenomenology_article",
    "biomedical_article",
    "empirical_biomedical_article",
    "systematic_review",
    "narrative_review",
    "clinical_case",
})

_PROMPT_PREFIX = "Complete one academic paper-structure task from the evidence below.\nTask: "
_PROMPT_MIDDLE = (
    "\nReturn exactly one JSON object and nothing else: no Markdown fence, "
    "commentary, or invented evidence. Preserve explicit word budgets and "
    "the requested response schema.\n\nInput JSON:\n"
)
_SPACE = re.compile(r"\s+")


class StructurePromptError(ValueError):
    """A runtime prompt is not an exact instance of the training contract."""


def canonical_json_text(value: Any) -> str:
    """Return the exact compact JSON representation used by stage-two rows."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def format_structure_prompt(task_type: str, prompt_input: Mapping[str, Any]) -> str:
    """Render the model-visible user prompt shared by training and inference."""

    task = str(task_type or "").strip()
    if not task:
        raise StructurePromptError("structure task_type must be non-empty")
    if not isinstance(prompt_input, Mapping) or not prompt_input:
        raise StructurePromptError("structure input must be a non-empty object")
    return (
        _PROMPT_PREFIX
        + task
        + _PROMPT_MIDDLE
        + canonical_json_text(dict(prompt_input))
        + "\n"
    )


def bounded_abstract_brief(value: str, *, maximum_words: int = 32) -> str:
    """Mirror the compiler's whitespace-bounded abstract brief."""

    if not isinstance(maximum_words, int) or isinstance(maximum_words, bool) or maximum_words <= 0:
        raise ValueError("maximum_words must be a positive integer")
    words = re.findall(r"\S+", _SPACE.sub(" ", str(value or "")).strip())
    return " ".join(words[:maximum_words])


def brief_to_blueprint_input(
    *, title: str, abstract_brief: str, discipline: str, genre: str,
) -> dict[str, Any]:
    """Build the exact input object learned by ``brief_to_blueprint`` rows."""

    selected_title = _SPACE.sub(" ", str(title or "")).strip()
    selected_brief = bounded_abstract_brief(abstract_brief)
    if not selected_title:
        raise StructurePromptError("brief_to_blueprint title must be non-empty")
    if not selected_brief:
        raise StructurePromptError("brief_to_blueprint abstract_brief must be non-empty")
    if discipline not in STRUCTURE_DISCIPLINES:
        raise StructurePromptError("brief_to_blueprint discipline is unsupported")
    if genre not in STRUCTURE_GENRES:
        raise StructurePromptError("brief_to_blueprint genre is unsupported")
    return {
        "instruction": BRIEF_TO_BLUEPRINT_INSTRUCTION,
        "context": {
            "discipline": discipline,
            "genre": genre,
            "title": selected_title,
            "abstract_brief": selected_brief,
        },
        "constraints": {"evidence": STRUCTURE_EVIDENCE_CONSTRAINT},
        "response_schema": dict(STRUCTURE_RESPONSE_SCHEMA),
    }


def parse_brief_to_blueprint_prompt(prompt: str) -> dict[str, Any]:
    """Parse and round-trip an exact runtime instance of the trained envelope."""

    if not isinstance(prompt, str) or not prompt.startswith(_PROMPT_PREFIX):
        raise StructurePromptError("structure prompt has the wrong contract prefix")
    task_and_input = prompt[len(_PROMPT_PREFIX):]
    task, separator, raw_input = task_and_input.partition(_PROMPT_MIDDLE)
    if not separator or task != BRIEF_TO_BLUEPRINT_TASK:
        raise StructurePromptError("structure runtime accepts brief_to_blueprint only")
    try:
        prompt_input = json.loads(raw_input)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructurePromptError("structure prompt Input JSON is invalid") from exc
    if not isinstance(prompt_input, dict):
        raise StructurePromptError("structure prompt Input JSON must be an object")
    expected_keys = {"instruction", "context", "constraints", "response_schema"}
    if set(prompt_input) != expected_keys:
        raise StructurePromptError("brief_to_blueprint input fields are incompatible")
    context = prompt_input.get("context")
    if not isinstance(context, dict) or set(context) != {
        "discipline", "genre", "title", "abstract_brief",
    }:
        raise StructurePromptError("brief_to_blueprint context fields are incompatible")
    rebuilt = brief_to_blueprint_input(
        title=context.get("title", ""),
        abstract_brief=context.get("abstract_brief", ""),
        discipline=context.get("discipline", ""),
        genre=context.get("genre", ""),
    )
    if prompt_input != rebuilt:
        raise StructurePromptError("brief_to_blueprint input values are incompatible")
    if prompt != format_structure_prompt(BRIEF_TO_BLUEPRINT_TASK, rebuilt):
        raise StructurePromptError("structure prompt is not canonically encoded")
    return rebuilt
