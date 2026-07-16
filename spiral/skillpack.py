"""spiral's skill library — distilled expertise as on-demand context.

On-demand craft loading: each skill is a markdown file with
frontmatter (name + trigger description) and a body of craft — checklists,
idioms, playbooks, pitfalls. A frontier model authors them once; local models
apply them forever, free. Skills load per-task only when they match, so the
worker's context stays lean.

Sources, in override order: <workspace>/.spiral/skills/ (project-specific),
then the built-in pack shipped in spiral/skills/.

Routing is keyword-overlap for now; upgrade path is embeddings via the
locally-installed nomic-embed-text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_BUILTIN = Path(__file__).parent / "skills"
_STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "so",
    "it", "its", "is", "are", "be", "that", "this", "then", "when", "use",
}


@dataclass
class SkillCard:
    name: str
    description: str
    body: str
    path: Path


def _parse(path: Path) -> SkillCard | None:
    try:
        text = path.read_text()
    except Exception:
        return None
    m = re.match(r"(?s)^---\n(.*?)\n---\n(.*)$", text)
    if not m:
        return None
    front, body = m.group(1), m.group(2).strip()
    fields = dict(
        (k.strip(), v.strip())
        for k, _, v in (line.partition(":") for line in front.splitlines() if ":" in line)
    )
    if "name" not in fields or "description" not in fields:
        return None
    return SkillCard(fields["name"], fields["description"], body, path)


def load_skills(workspace: str | Path | None = None) -> list[SkillCard]:
    cards: dict[str, SkillCard] = {}
    dirs = [_BUILTIN]
    if workspace:
        dirs.append(Path(workspace) / ".spiral" / "skills")  # later wins → overrides
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                card = _parse(f)
                if card:
                    cards[card.name] = card
    return list(cards.values())


def _tokens(text: str) -> set[str]:
    """Lowercase, split on non-alphanumerics, cheap plural fold."""
    out = set()
    for w in re.findall(r"[a-z][a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


# deterministic routing: file extensions are stronger evidence than prose overlap
_EXT_ROUTES = {
    "android-kotlin": (".kt", ".kts", ".xml", "androidmanifest", "gradle"),
    "dark-ui-design": ("colors.xml", "themes.xml", "styles.xml", "layout/"),
}


def match_skills(
    task_text: str,
    cards: list[SkillCard],
    files: list[str] | None = None,
    top: int = 2,
    min_overlap: int = 2,
) -> list[SkillCard]:
    """Rank skills for a task: file-extension routes are decisive, keyword overlap
    between task text and skill name+description breaks the rest.

    (v2: the conductor tags skills per task at plan time — model-as-router, the
    model-as-router over skill descriptions; this function stays as the fallback.)
    """
    task = _tokens(task_text)
    haystack = " ".join(files or []).lower() + " " + task_text.lower()
    scored = []
    for c in cards:
        overlap = len(task & _tokens(c.name + " " + c.description))
        if any(ext in haystack for ext in _EXT_ROUTES.get(c.name, ())):
            overlap += min_overlap  # decisive boost
        if overlap >= min_overlap:
            scored.append((overlap, c.name, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top]]


def render_for_prompt(cards: list[SkillCard], budget: int = 6_000) -> str:
    """Concatenate skill bodies for prompt injection, inside a char budget."""
    parts: list[str] = []
    for c in cards:
        body = c.body[: max(0, budget)]
        budget -= len(body)
        parts.append(f"## SKILL: {c.name}\n{body}")
        if budget <= 0:
            break
    return "\n\n".join(parts)
