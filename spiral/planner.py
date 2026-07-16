"""The conductor's brain — decompose a raw goal into milestones and tasks,
then reflect on its own plan before anything executes.

Both passes run Qwen in plan-mode: thinking ON, but the answer constrained to a
JSON schema, so it MUST emit a structured plan and stop instead of thinking its
budget away.

Gate philosophy: the conductor injects the project build gate (e.g. gradle) into
every task automatically — the planner is told NOT to invent shallow existence
checks, and task.verify is reserved for genuine EXTRA checks (unit tests etc.).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from spiral.config import Config
from spiral.llm import ChatResult, Ollama

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "understanding": {"type": "string"},
        "milestones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "goal": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "files": {"type": "array", "items": {"type": "string"}},
                                "verify": {"type": "string"},
                            },
                            "required": ["title", "description"],
                        },
                    },
                },
                "required": ["title", "tasks"],
            },
        },
    },
    "required": ["understanding", "milestones"],
}

PLANNER_SYSTEM = (
    "You are spiral's CONDUCTOR — the orchestrator of a local coding agent that will "
    "execute your plan task by task, unattended.\n\n"
    "Given a project GOAL and the current REPO, produce an execution PLAN:\n"
    "- Break the work into ordered MILESTONES, each into small concrete CODING TASKS a "
    "junior agent can finish in one sitting, each touching at most ~3 files.\n"
    "- Every class, screen, layout, or resource that any task references must be CREATED "
    "by that task or an earlier one. Never reference future or imaginary components.\n"
    "- Order tasks so the project builds after every single task.\n"
    "- Account for what already exists in the repo — extend and repair it, don't restart.\n"
    "- A mandatory BUILD GATE runs automatically after every task; do NOT write shallow "
    "file-existence or grep checks into 'verify'. Use 'verify' ONLY for a genuine extra "
    "check (e.g. a unit test command), else leave it empty.\n"
    "- 'description' must carry the full intent for that task (the executing agent sees "
    "only the task, not this conversation): name exact files, classes, ids, behaviors.\n"
    "Return ONLY JSON matching the schema."
)

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["feature", "quality", "constraint"]},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["requirements"],
}

SPEC_SYSTEM = (
    "You are spiral's ANALYST. Extract every concrete commitment from the project GOAL "
    "into a requirements checklist.\n"
    "- Each requirement is atomic and checkable by looking at the finished product.\n"
    "- ids R1..Rn. kind: feature (user-visible behavior), quality (style/feel/voice), "
    "constraint (platform/tech).\n"
    "- Do NOT invent requirements the goal doesn't state; do NOT merge distinct features "
    "into one requirement.\n"
    "Return ONLY JSON matching the schema."
)

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "where": {"type": "string"},
                    "issue": {"type": "string"},
                    "fix_hint": {"type": "string"},
                },
                "required": ["issue"],
            },
        },
    },
    "required": ["verdict", "defects"],
}

CRITIC_SYSTEM = (
    "You are spiral's PLAN CRITIC — a senior reviewer with a different brain than the "
    "planner. You review a PLAN before an unattended junior agent executes it. Output "
    "DEFECTS ONLY — never a plan.\n\n"
    "Hunt, in priority order:\n"
    "1. COVERAGE — map every REQUIREMENT id to at least one task. Name every unmapped id.\n"
    "2. PHANTOMS — tasks referencing classes/files/ids/resources that no earlier task "
    "creates and the REPO does not contain.\n"
    "3. ORDER — any point where the project would not compile after a task.\n"
    "4. VAGUENESS — tasks a junior would have to guess at (missing file names, class "
    "names, ids, or behaviors).\n"
    "5. CONSISTENCY — two tasks that would invent rival versions of the same concept.\n"
    "Include the LINT findings you agree with. verdict='pass' only with zero material "
    "defects. Be specific: cite task numbers in 'where'."
)

DESIGNER_SYSTEM = (
    "You are spiral's DESIGN DIRECTOR. Produce a CONCRETE design specification that "
    "junior executing agents will implement LITERALLY. Decisions, never options:\n"
    "1. CONCEPT: one sentence stating the product's feel and the single idea every "
    "decision serves.\n"
    "2. PALETTE: named tokens with exact hex values and one usage rule each. ONE "
    "accent; semantic colors reserved for meaning; dark surfaces #0A0A0A floor, "
    "elevation = lighter surface; body-text contrast ≥ 4.5:1.\n"
    "3. TYPE & SPACING: a modular scale (base 16, ×1.25 steps), exactly two weights, "
    "and a 4/8 spacing grid — every gap a multiple of 4-8; touch targets ≥ 48.\n"
    "4. SCREENS: each screen top-to-bottom — structure, components, and their FIVE "
    "states (default/focus/pressed/disabled/loading) plus the EMPTY and ERROR states.\n"
    "5. HIERARCHY: name each screen's ONE primary action; everything else quieter.\n"
    "6. VOICE: the product's tone with at least 6 verbatim sample strings; buttons "
    "say what they do.\n"
    "7. MOTION: each animation with duration (micro 100-150ms, transitions 200-300ms, "
    "nothing >800ms), easing (out for entrances), and what it teaches the user.\n"
    "Restraint law: for every element, if removing it loses no meaning — remove it. "
    "Markdown, under 2000 words."
)


def design_brief(goal: str, spec: list[dict], cfg: Config | None = None,
                 ol: Ollama | None = None, progress=None) -> tuple[str, ChatResult]:
    """One-time concrete design spec (critic thinking, fallback ladder). Free-form
    markdown — no JSON constraint, so the ladder checks length not parse."""
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    reqs = "\n".join(f"{r['id']}: {r['text']}" for r in spec)
    msgs = [
        {"role": "system", "content": DESIGNER_SYSTEM},
        {"role": "user", "content": f"GOAL:\n{goal}\n\nREQUIREMENTS:\n{reqs}\n\nWrite the design specification."},
    ]
    res = None
    for m, th in ((cfg.critic.name, cfg.critic.think), (cfg.critic.name, False), (cfg.planner.name, False)):
        res = ol.chat(m, msgs, think=th, num_predict=cfg.planner_max_tokens, temperature=0.6,
                      num_ctx=cfg.spec_for(m).num_ctx, keep_alive=cfg.keep_alive,
                      on_delta=(lambda kind, piece: progress(kind)) if progress else None)
        if len(res.text.strip()) > 400:
            return res.text.strip(), res
    return (res.text.strip() if res else ""), res


VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["implemented", "partial", "missing"]},
                    "evidence": {"type": "string"},
                    "fix": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "files": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required": ["id", "status"],
            },
        }
    },
    "required": ["verdicts"],
}

VALIDATOR_SYSTEM = (
    "You are spiral's VALIDATOR — the final inspector, a different brain from the "
    "builder. Judge each REQUIREMENT against the CODE alone. Never trust plans, task "
    "titles, or commit messages — if the code doesn't show it, it doesn't exist.\n"
    "- implemented: fully realized AND reachable/wired. A function that nothing calls "
    "does NOT count. A screen no navigation reaches does NOT count.\n"
    "- partial: some of it exists but is incomplete or unwired.\n"
    "- missing: no meaningful trace in the code.\n"
    "Cite evidence (file paths / symbols) for every verdict. If a file ends with a "
    "'…(truncated)' marker, content may exist beyond it — do not judge unseen content "
    "as missing; say so in the evidence instead. For every partial or missing "
    "requirement, provide one small concrete fix task: exact files and exactly "
    "what to add or wire. Return ONLY JSON matching the schema."
)


REPAIR_SYSTEM = (
    "You are spiral's CONDUCTOR. A senior critic reviewed your plan and found DEFECTS. "
    "Apply every defect's fix to the plan — add missing tasks, reorder, sharpen vague "
    "descriptions, remove phantoms. Change nothing that isn't defective. "
    "Return the FULL corrected plan as JSON in the same schema."
)


@dataclass
class Task:
    title: str
    description: str
    files: list[str] = field(default_factory=list)
    verify: str = ""


@dataclass
class Milestone:
    title: str
    tasks: list[Task]
    goal: str = ""


@dataclass
class Plan:
    understanding: str
    milestones: list[Milestone]

    @property
    def task_count(self) -> int:
        return sum(len(m.tasks) for m in self.milestones)


def plan_to_dict(plan: Plan) -> dict:
    return {
        "understanding": plan.understanding,
        "milestones": [
            {
                "title": m.title,
                "goal": m.goal,
                "tasks": [
                    {"title": t.title, "description": t.description, "files": t.files, "verify": t.verify}
                    for t in m.tasks
                ],
            }
            for m in plan.milestones
        ],
    }


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a : b + 1])
        raise


def _close_json(prefix: str) -> str:
    """Compute the closers a truncated JSON prefix needs (string + brackets)."""
    stack: list[str] = []
    in_str = esc = False
    for ch in prefix:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = in_str
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    return ('"' if in_str else "") + "".join(reversed(stack))


def _salvage_json(text: str) -> dict | None:
    """Repair JSON truncated mid-emission (thinking ate the token budget):
    progressively trim the tail and close open strings/brackets."""
    a = text.find("{")
    if a < 0:
        return None
    s = text[a:]
    for cut in range(len(s), max(len(s) - 4000, 1), -80):
        prefix = s[:cut].rstrip().rstrip(",")
        try:
            return json.loads(prefix + _close_json(prefix))
        except json.JSONDecodeError:
            continue
    return None


def _plan_chat(
    system: str,
    user: str,
    cfg: Config,
    ol: Ollama,
    temperature: float,
    schema: dict | None = None,
    model: str | None = None,
    think: bool = True,
    fallback_model: str | None = None,
    progress=None,
) -> ChatResult:
    """Structured planning call with a fallback ladder. Thinking mode can consume
    the whole num_predict budget and return EMPTY content (the original
    think-forever disease, at the conductor level). Ladder: think → think (retry)
    → think OFF [→ fallback model, think OFF]. The think-off rungs cannot ramble,
    so this never returns empty."""
    name = model or cfg.planner.name
    ladder: list[tuple[str, bool]] = [(name, think), (name, think), (name, False)]
    if fallback_model and fallback_model != name:
        ladder.append((fallback_model, False))
    res = None
    for m, th in ladder:
        res = ol.chat(
            m,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            think=th,
            num_predict=cfg.planner_max_tokens,
            temperature=temperature,
            fmt=schema or PLAN_SCHEMA,
            num_ctx=cfg.spec_for(m).num_ctx,
            keep_alive=cfg.keep_alive,
            on_delta=(lambda kind, piece: progress(kind)) if progress else None,
        )
        if not res.text.strip():
            continue  # thinking ate the whole budget — next rung
        # the reply must PARSE to leave the ladder: truncated JSON gets salvaged
        # (close open brackets), and if unsalvageable the next rung retries —
        # think-off rungs spend the whole budget on JSON and cannot truncate
        try:
            _extract_json(res.text)
            return res
        except json.JSONDecodeError:
            data = _salvage_json(res.text)
            if data is not None:
                res.text = json.dumps(data)
                return res
    raise RuntimeError(
        f"planner produced no parseable JSON on {len(ladder)} attempts "
        f"(last reply: {res.completion_tokens} tok, starts {res.text[:80]!r})"
    )


def parse_plan(data: dict) -> Plan:
    milestones = []
    for m in data.get("milestones", []):
        tasks = [
            Task(t["title"], t.get("description", ""), t.get("files", []) or [], t.get("verify", "") or "")
            for t in m.get("tasks", [])
        ]
        milestones.append(Milestone(m["title"], tasks, m.get("goal", "")))
    return Plan(data.get("understanding", ""), milestones)


def _gate_line(gate: str) -> str:
    if gate:
        return f"MANDATORY BUILD GATE (runs after every task): {gate}\n\n"
    return "NOTE: no build gate was detected in this repo — tasks with empty 'verify' run unverified.\n\n"


def make_plan(goal: str, repomap: str, gate: str = "", cfg: Config | None = None, ol: Ollama | None = None, progress=None) -> tuple[Plan, ChatResult]:
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    user = f"{_gate_line(gate)}GOAL:\n{goal}\n\nREPO:\n{repomap}\n\nProduce the execution plan as JSON."
    res = _plan_chat(PLANNER_SYSTEM, user, cfg, ol, temperature=0.3, progress=progress)
    return parse_plan(_extract_json(res.text)), res


def extract_spec(goal: str, cfg: Config | None = None, ol: Ollama | None = None, progress=None) -> tuple[list[dict], ChatResult]:
    """GOAL prose → atomic requirements checklist. Coverage stops being vibes and
    becomes a mechanical diff: every Rn maps to a task, or it's a defect."""
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    res = _plan_chat(SPEC_SYSTEM, f"GOAL:\n{goal}\n\nExtract the requirements checklist as JSON.",
                     cfg, ol, temperature=0.2, schema=SPEC_SCHEMA, progress=progress)
    return _extract_json(res.text).get("requirements", []), res


def lint_plan(plan: Plan, existing_files: set[str]) -> list[str]:
    """Deterministic, zero-token plan checks — ground truth before opinion,
    even at plan level."""
    defects: list[str] = []
    seen_files: set[str] = set(existing_files)
    seen_titles: set[str] = set()
    creation_verbs = ("create", "add", "new", "write", "implement", "generate")
    for mi, m in enumerate(plan.milestones, 1):
        for ti, t in enumerate(m.tasks, 1):
            tag = f"task {mi}.{ti} '{t.title}'"
            if len(t.files) > 3:
                defects.append(f"{tag}: touches {len(t.files)} files — split it (≤3).")
            if len(t.description) < 40:
                defects.append(f"{tag}: description too thin to execute without guessing.")
            if re.search(r"\b(grep|ls|test -f|find)\b", t.verify):
                defects.append(f"{tag}: shallow existence-check verify '{t.verify}' — not a gate.")
            low = (t.title + " " + t.description).lower()
            for f in t.files:
                if f not in seen_files and not any(v in low for v in creation_verbs):
                    defects.append(f"{tag}: edits '{f}' which no repo file or earlier task provides.")
                seen_files.add(f)
            key = t.title.strip().lower()
            if key in seen_titles:
                defects.append(f"{tag}: duplicate task title — likely rival implementations.")
            seen_titles.add(key)
    return defects


def critique_plan(
    goal: str, spec: list[dict], repomap: str, plan: Plan, lint: list[str],
    gate: str = "", cfg: Config | None = None, ol: Ollama | None = None, progress=None,
) -> tuple[str, list[dict], ChatResult]:
    """Different-brain review (dense critic model, thinking, defects-only output).
    Falls back to the planner model if the critic can't produce."""
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    reqs = "\n".join(f"{r['id']}: {r['text']} ({r.get('kind', 'feature')})" for r in spec)
    user = (
        f"{_gate_line(gate)}GOAL:\n{goal}\n\nREQUIREMENTS:\n{reqs}\n\nREPO:\n{repomap}\n\n"
        f"PLAN:\n{json.dumps(plan_to_dict(plan), indent=1)}\n\n"
        f"LINT FINDINGS (deterministic):\n" + ("\n".join(lint) or "(none)") +
        "\n\nReturn your verdict and defect list as JSON."
    )
    res = _plan_chat(CRITIC_SYSTEM, user, cfg, ol, temperature=0.2, schema=CRITIC_SCHEMA,
                     model=cfg.critic.name, think=cfg.critic.think,
                     fallback_model=cfg.planner.name, progress=progress)
    data = _extract_json(res.text)
    return data.get("verdict", "revise"), data.get("defects", []), res


def validate_spec(
    goal: str, spec: list[dict], repomap: str, gate: str = "",
    cfg: Config | None = None, ol: Ollama | None = None, progress=None,
) -> tuple[list[dict], ChatResult]:
    """Final inspection: per-requirement verdicts judged from CODE only.
    Runs on the critic model (different brain), with the planner as fallback."""
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    reqs = "\n".join(f"{r['id']}: {r['text']} ({r.get('kind', 'feature')})" for r in spec)
    user = (
        f"{_gate_line(gate)}GOAL:\n{goal}\n\nREQUIREMENTS:\n{reqs}\n\n"
        f"CODE (current repo state):\n{repomap}\n\n"
        "Return per-requirement verdicts as JSON."
    )
    res = _plan_chat(VALIDATOR_SYSTEM, user, cfg, ol, temperature=0.2, schema=VALIDATE_SCHEMA,
                     model=cfg.critic.name, think=cfg.critic.think,
                     fallback_model=cfg.planner.name, progress=progress)
    return _extract_json(res.text).get("verdicts", []), res


def repair_plan(
    goal: str, plan: Plan, defects: list[dict], gate: str = "",
    cfg: Config | None = None, ol: Ollama | None = None, progress=None,
) -> tuple[Plan, ChatResult]:
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    dl = "\n".join(f"- [{d.get('where', '?')}] {d['issue']} → {d.get('fix_hint', '')}" for d in defects)
    user = (
        f"{_gate_line(gate)}GOAL:\n{goal}\n\n"
        f"YOUR PLAN:\n{json.dumps(plan_to_dict(plan), indent=1)}\n\nDEFECTS:\n{dl}\n\n"
        "Return the full corrected plan as JSON."
    )
    res = _plan_chat(REPAIR_SYSTEM, user, cfg, ol, temperature=0.2, progress=progress)
    return parse_plan(_extract_json(res.text)), res
