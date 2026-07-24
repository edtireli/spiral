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

from spiral.appicon import GLYPHS
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
                                "requirements": {"type": "array", "items": {"type": "string"}},
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
    "- Deliver the whole usable product described by the requirements, not a demo, "
    "landing-page shell, happy-path mock, or minimal compile-green scaffold. Plan the "
    "primary workflow end to end, then the surrounding navigation/configuration, real "
    "data path, failure recovery, tests, packaging, and finish work that make it usable.\n"
    "- No task may leave TODOs, placeholder controls, fake success states, dead routes, "
    "sample-only data, or an unwired component unless the requirement explicitly asks for a mock.\n"
    "- Select proven libraries for established domain logic, charts, icons, editors, "
    "parsers, physics, or protocols. Include a dependency/tool investigation task before "
    "implementation when the correct choice is not established by the repo.\n"
    "- This is the Builder, not the academic Research engine. Investigate implementation "
    "references as needed, but do not invent novelty claims, literature-review milestones, "
    "theorems, or a research paper unless the user's build goal explicitly requests that artifact.\n"
    "- If the product has a user interface and a DESIGN SPECIFICATION is provided, "
    "milestone 1 establishes the FOUNDATION — the color tokens, type scale, spacing, "
    "and shared component styles from the spec — before any feature screen, so every "
    "screen inherits one coherent look.\n"
    "- UI plans implement the actual working experience first. They include responsive "
    "layouts, keyboard/focus behavior, loading/empty/error states, real visual assets when "
    "the domain needs them, and screenshot-based polish. Plot/dashboard plans include "
    "units, labels, legends, accessible colors, interaction, and export.\n"
    "- Account for what already exists in the repo — extend and repair it, don't restart.\n"
    "- A mandatory BUILD GATE runs automatically after every task; do NOT write shallow "
    "file-existence or grep checks into 'verify'. Use 'verify' ONLY for a genuine extra "
    "check (e.g. a unit test command), else leave it empty.\n"
    "- 'description' must carry the full intent for that task (the executing agent sees "
    "only the task, not this conversation): name exact files, classes, ids, behaviors.\n"
    "- Each task lists the requirement ids it advances in 'requirements'. Every requirement "
    "must map to at least one implementation or verification task.\n"
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
                    "check": {"type": "string"},
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
    "- Preserve every explicit commitment. Also infer only the ordinary product obligations "
    "without which the requested artifact would be visibly incomplete: the end-to-end primary "
    "workflow, relevant failure/empty/loading states, accessibility, tests, and runnable delivery. "
    "Do not invent unrelated features or a different product.\n"
    "- If (and only if) a requirement can be verified by RUNNING something, add 'check': "
    "one shell command that exits 0 exactly when the requirement is met — run a test "
    "file, invoke the CLI/binary and inspect its output, execute the program. A check "
    "must OBSERVE BEHAVIOR: commands that merely assert files or text exist (grep, ls, "
    "test -f, find, cat) are NOT checks — omit 'check' when only that is possible.\n"
    "Return ONLY JSON matching the schema."
)

ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_id": {"type": "string"},
        "deliverables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "web", "android", "ios", "desktop", "cli", "service",
                            "library", "simulation", "plot", "image", "video", "audio",
                            "document", "presentation", "dataset", "notebook", "3d",
                            "game", "firmware", "infrastructure", "formal-proof", "other",
                        ],
                    },
                    "description": {"type": "string"},
                    "root_hint": {"type": "string"},
                    "output_globs": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "visual": {"type": "boolean"},
                    "interactive": {"type": "boolean"},
                    "acceptance_evidence": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "tool_families": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["id", "kind", "description", "visual", "interactive"],
            },
        },
    },
    "required": ["primary_id", "deliverables"],
}

FILE_DELIVERABLE_KINDS = {
    "plot", "image", "video", "audio", "document", "presentation",
    "dataset", "notebook", "3d", "formal-proof",
}


def default_output_globs(kind: str) -> list[str]:
    """Give file deliverables a deterministic destination when analysis omits one."""

    return ["output/*"] if str(kind) in FILE_DELIVERABLE_KINDS else []


ARTIFACT_SYSTEM = (
    "You are spiral's DELIVERABLE ANALYST. Translate the user's goal into the actual "
    "artifacts that must exist at completion. Do not force a visual request into a web "
    "app, or a program into a GUI. A goal may require several deliverables, such as a "
    "service plus client, paper plus code, simulation plus plots, or model plus dataset. "
    "For each deliverable state its medium, whether it is visual/interactive, likely "
    "workspace root, exact relative output globs that identify finished outputs rather "
    "than source assets (for example output/report.pdf or dist/*.png), tool families "
    "needed, and concrete acceptance evidence. For a code project, leave output_globs "
    "empty unless the goal actually requires a built package/export or an existing build "
    "convention yields one; never declare the workspace, src/, app/, lib/, or another "
    "source tree as a finished output. Do not add a paper, novelty review, or academic "
    "classification to an ordinary software build. Evidence "
    "must describe opening, running, parsing, measuring, testing, rendering, or inspecting "
    "the artifact rather than merely checking that a file exists. Return JSON only."
)

# a check that only asserts presence is not a check — the exact failure class the
# green-gate/12-of-12 lesson was bought with, reintroduced at spec level
_PRESENCE_CHECK = re.compile(r"^\s*(grep|ls|find|test|stat|cat|head|tail|\[)\b")


def sanitize_checks(spec: list[dict]) -> list[str]:
    """Deterministic guard over model-authored acceptance checks: drop presence-style
    commands and anything the denylist refuses. Mutates spec in place; returns one
    note per dropped check so silence never reads as coverage."""
    from spiral import tools

    notes: list[str] = []
    for r in spec:
        cmd = (r.get("check") or "").strip()
        if not cmd:
            r.pop("check", None)
            continue
        if _PRESENCE_CHECK.match(cmd):
            notes.append(f"{r.get('id', '?')}: presence-style check dropped: {cmd[:60]}")
            r.pop("check", None)
        elif tools.is_dangerous(cmd):
            notes.append(f"{r.get('id', '?')}: check hits the denylist, dropped: {cmd[:60]}")
            r.pop("check", None)
        else:
            r["check"] = cmd
    return notes


def product_profile(goal: str, project_kind: str = "other") -> str:
    """Classify the requested deliverable for deterministic completeness rules."""

    text = (goal or "").lower()
    if any(word in text for word in (
            "plot", "chart", "visualization", "visualisation", "data viz", "dashboard")):
        return "plot"
    if any(word in text for word in (
            "simulation", "simulator", "numerical model", "monte carlo")):
        return "simulation"
    if project_kind in {"visualization", "plot"}:
        return "plot"
    if project_kind in {"android", "ios", "web", "gui", "desktop", "game"}:
        return "ui"
    if project_kind in {"image", "video", "audio", "3d"}:
        return "visual-media"
    if project_kind in {"document", "presentation"}:
        return "document"
    if project_kind == "dataset":
        return "data"
    if project_kind == "formal-proof":
        return "formal"
    if project_kind in {"infrastructure", "firmware"}:
        return "systems"
    if any(word in text for word in ("command-line", "command line", " cli ", "terminal tool")):
        return "cli"
    if any(word in text for word in ("rest api", "graphql", "web service", "backend service", "server")):
        return "service"
    if any(word in text for word in ("library", "package", "sdk", "module")):
        return "library"
    return "general"


def _is_product_build(goal: str, project_kind: str) -> bool:
    text = f" {(goal or '').lower()} "
    actions = (" build ", " create ", " make ", " develop ", " design ", " implement ")
    artifacts = (
        " app ", " application ", " website ", " site ", " tool ", " program ",
        " dashboard ", " game ", " simulator ", " simulation ", " cli ", " service ",
        " api ", " library ", " package ", " plot ", " chart ", " visualization ",
    )
    return project_kind in {
        "android", "ios", "web", "gui", "desktop", "visualization", "plot",
        "image", "video", "audio", "document", "presentation", "dataset",
        "notebook", "3d", "game", "firmware", "infrastructure", "formal-proof",
    } or (
        any(word in text for word in actions) and any(word in text for word in artifacts)
    )


def enrich_product_spec(goal: str, spec: list[dict], project_kind: str = "other") -> list[dict]:
    """Add a conservative, deterministic definition of done for product requests.

    The model extracts the user's explicit commitments. This pass supplies only the
    ordinary completion obligations implied by the artifact type, preventing a request
    for an app/tool/plot from quietly shrinking into a skeleton.
    """

    rows = [dict(row) for row in (spec or []) if isinstance(row, dict)]
    if not _is_product_build(goal, project_kind):
        return rows
    profile = product_profile(goal, project_kind)
    if profile == "visual-media":
        baseline: list[tuple[str, str, str]] = [
            (
                "feature", "artifact-completeness",
                "The final media contains the complete requested composition and real content, "
                "with no placeholder copy, missing assets, temporary marks, or unfinished regions.",
            ),
            (
                "quality", "media-delivery",
                "The final media decodes correctly at its intended dimensions, duration or "
                "resolution and is exported in an inspectable standard format.",
            ),
            (
                "constraint", "media-reproducibility",
                "Editable source or a deterministic generation procedure, asset provenance, "
                "fonts and exact export settings are retained so the artifact can be revised.",
            ),
            (
                "quality", "media-inspection",
                "The exported artifact is independently inspected at its intended size for "
                "legibility, hierarchy, clipping, unwanted margins, visual defects and content accuracy.",
            ),
        ]
    elif profile == "document":
        baseline = [
            (
                "feature", "document-completeness",
                "The document or presentation contains the complete requested argument, content, "
                "figures, tables, references and supporting material with no placeholder sections.",
            ),
            (
                "quality", "document-delivery",
                "The final document renders without clipped, overflowing, blank, malformed or "
                "inconsistently styled pages/slides, and references and numbering resolve.",
            ),
            (
                "constraint", "document-reproducibility",
                "Editable source, cited assets/data and a reproducible standard-format export are retained.",
            ),
        ]
    elif profile == "data":
        baseline = [
            (
                "feature", "dataset-contract",
                "The delivered dataset has the complete requested records and fields, a documented "
                "schema, units, types, missing-value semantics and provenance.",
            ),
            (
                "quality", "dataset-validation",
                "Machine-run validation checks schema, constraints, duplicates, ranges, encoding "
                "and representative values, and reports failures without silently dropping data.",
            ),
            (
                "constraint", "dataset-reproducibility",
                "The transformation or collection procedure is reproducible and records input "
                "versions, parameters and output checksums.",
            ),
        ]
    elif profile == "formal":
        baseline = [
            (
                "feature", "formal-completeness",
                "Every requested statement is represented precisely and proved without admitted "
                "goals, placeholders, accidental stronger assumptions or untracked axioms.",
            ),
            (
                "quality", "formal-certificate",
                "Every theorem claimed as verified is accepted by the declared proof checker from "
                "a clean environment, with assumptions, axioms and exact source retained.",
            ),
            (
                "constraint", "formal-reproducibility",
                "The prover, library versions, build command and dependency lock are recorded so "
                "the certificate can be checked independently.",
            ),
        ]
    else:
        baseline = [
            (
                "feature", "product-depth",
                "Every primary workflow implied by the goal works end to end through real "
                "domain logic and data paths; no TODOs, placeholder screens, dead controls, "
                "hard-coded fake success, or sample-only implementation remains.",
            ),
            (
                "quality", "failure-recovery",
                "Invalid input, boundary cases, and operational failures produce actionable "
                "feedback and a recovery or retry path without silently losing user work.",
            ),
            (
                "quality", "behavioral-verification",
                "Automated behavioral tests cover the primary success path plus meaningful "
                "boundary and failure cases, and the clean build/test command passes.",
            ),
            (
                "constraint", "runnable-delivery",
                "A fresh checkout has reproducible setup, run, test, and packaging instructions "
                "with safe example configuration and no dependency on undocumented local state.",
            ),
        ]
    if profile in {"ui", "plot"} or project_kind in {"android", "ios", "web", "gui"}:
        baseline += [
            (
                "feature", "complete-interaction-states",
                "Every intended view is reachable and relevant controls have working default, "
                "focus/hover, pressed, disabled, loading, empty, and error behavior without "
                "layout shift or dead ends.",
            ),
            (
                "quality", "responsive-accessible-ui",
                "The interface remains usable without clipping or overlap on mobile, desktop, "
                "and wide viewports, supports keyboard navigation and visible focus, labels "
                "interactive controls, and maintains readable contrast.",
            ),
            (
                "quality", "domain-specific-visual-finish",
                "The finished interface uses a coherent domain-specific visual system, real "
                "content and assets where needed, familiar icons, stable component dimensions, "
                "and polished hierarchy rather than a generic card-grid or landing-page shell.",
            ),
        ]
    if profile == "plot":
        baseline.append((
            "feature", "plot-semantics-export",
            "Plots expose meaningful labels, units, legends or direct annotations, accessible "
            "series distinction, inspectable values, and reproducible export of the figure and data.",
        ))
    elif profile == "simulation":
        baseline.append((
            "feature", "simulation-reproducibility",
            "Simulation parameters are validated and recorded, stochastic runs accept an explicit "
            "seed, numerical invariants or reference cases are tested, and results can be exported.",
        ))
    elif profile == "cli":
        baseline.append((
            "quality", "cli-contract",
            "The CLI has discoverable help, validated arguments, useful stdout/stderr, stable "
            "non-zero exit codes on failure, non-interactive operation, and configuration precedence tests.",
        ))
    elif profile == "service":
        baseline.append((
            "quality", "service-contract",
            "The service validates requests, returns documented status/error shapes, handles "
            "timeouts and shutdown, keeps secrets out of source, and has integration-level contract tests.",
        ))
    elif profile == "library":
        baseline.append((
            "quality", "library-contract",
            "The public API has stable types, examples, boundary/error semantics, focused tests, "
            "and packaging metadata sufficient for another project to consume it.",
        ))

    def terms(text: str) -> set[str]:
        return {
            word for word in re.findall(r"[a-z]{5,}", text.lower())
            if word not in _STOP
        }

    held = [terms(str(row.get("text") or "")) for row in rows]
    next_id = max(
        [int(match.group(1)) for row in rows
         if (match := re.fullmatch(r"R(\d+)", str(row.get("id") or "")))] or [0]
    ) + 1
    for kind, audit, text in baseline:
        wanted = terms(text)
        if any(len(wanted & existing) / max(1, len(wanted | existing)) >= 0.45
               for existing in held):
            continue
        rows.append({
            "id": f"R{next_id}",
            "text": text,
            "kind": kind,
            "origin": "inferred-product-baseline",
            "audit": audit,
        })
        held.append(wanted)
        next_id += 1
    return rows


def enrich_deliverable_spec(spec: list[dict], manifest: dict) -> list[dict]:
    """Give every declared output an explicit, independently validated requirement."""

    rows = [dict(row) for row in (spec or []) if isinstance(row, dict)]
    next_id = max(
        [int(match.group(1)) for row in rows
         if (match := re.fullmatch(r"R(\d+)", str(row.get("id") or "")))] or [0]
    ) + 1
    existing_deliverables = {
        str(row.get("deliverable")) for row in rows if row.get("deliverable")
    }
    for deliverable in manifest.get("deliverables") or []:
        if not isinstance(deliverable, dict):
            continue
        identifier = str(deliverable.get("id") or f"D{next_id}")
        if identifier in existing_deliverables:
            continue
        description = str(deliverable.get("description") or "").strip()
        evidence = [
            str(item).strip()
            for item in (deliverable.get("acceptance_evidence") or [])
            if str(item).strip()
        ]
        text = (
            f"Deliver {identifier} as a complete {deliverable.get('kind', 'artifact')}: "
            f"{description or 'the requested output'}."
        )
        output_globs = [
            str(item) for item in (deliverable.get("output_globs") or [])
            if str(item).strip()
        ]
        if output_globs:
            text += (
                " Finished outputs must resolve these exact workspace-relative "
                "patterns: " + ", ".join(output_globs) + "."
            )
        if evidence:
            text += " Acceptance evidence: " + "; ".join(evidence) + "."
        rows.append({
            "id": f"R{next_id}",
            "text": text,
            "kind": "feature",
            "origin": "deliverable-manifest",
            "deliverable": identifier,
        })
        next_id += 1
    return rows

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
    "6. PRODUCT DEPTH — reject compile-green scaffolds, placeholder data/actions, dead "
    "navigation, happy-path-only workflows, absent error recovery, or a plan that implements "
    "the central feature without the surrounding structure needed to use it.\n"
    "7. CRAFT — for UI/plots, ensure the design brief, responsive/accessibility work, real "
    "assets, interaction states, chart semantics, and visual QA are actually scheduled. For "
    "all products, require behavioral tests and reproducible run/package work.\n"
    "Include the LINT findings you agree with. verdict='pass' only with zero material "
    "defects. Be specific: cite task numbers in 'where'."
)

DESIGNER_SYSTEM = (
    "You are spiral's DESIGN DIRECTOR. Produce a concrete, implementation-ready design "
    "for the ACTUAL requested medium: interactive UI, plot, static image, video, document, "
    "presentation, notebook, or 3D artifact. Never turn one medium into another merely "
    "because web UI is familiar. Design for this domain and audience; do not apply a "
    "fashionable house style. Decisions, never option lists:\n"
    "1. PRODUCT/ARTIFACT MODEL: identify the audience, use context, content hierarchy and "
    "the single visual idea that supports the work. For interactive products, identify the "
    "repeated primary workflow and make the first screen the usable product, not marketing. "
    "For static/sequential media, specify the viewing distance, dimensions, duration/pages/"
    "slides and the intended reading order.\n"
    "2. COMPOSITION: for UI, specify each view, navigation, density, stable responsive "
    "constraints and what remains visible on mobile/desktop/wide. For images, documents, "
    "slides and video, specify the exact canvas/page/frame grid, margins, crop/safe areas, "
    "sequence and export variants. Operational tools should be quiet and scan-friendly. "
    "Do not use cards for page sections or nest cards inside cards.\n"
    "3. VISUAL SYSTEM: named color tokens with exact hex values and contrast rules; a balanced "
    "neutral foundation, restrained brand accents, and semantic colors used only for meaning. "
    "Choose light, dark, or mixed surfaces from the subject matter. Avoid generic purple/blue "
    "gradients, beige/brown monotones, decorative blobs, glassmorphism, and one-hue palettes.\n"
    "4. TYPE & SPACE: concrete families/fallbacks, a restrained modular scale (never viewport-"
    "scaled type), weights, line lengths, 4/8 spacing, container widths, grid tracks, aspect "
    "ratios, and minimum 44px targets. Text and controls must not clip or shift layout.\n"
    "5. COMPONENTS & STATES: for interactive work enumerate shared components and relevant "
    "hover/focus/pressed/disabled/loading/empty/error/success/offline states. Use familiar "
    "icons and native control forms. For non-interactive work, instead define recurring page/"
    "frame motifs, caption/figure/table treatment, transitions and continuity rules.\n"
    "6. REAL CONTENT & ASSETS: name the real images, maps, diagrams, plots, media, or domain "
    "objects required. Do not substitute atmospheric stock imagery or decorative SVGs where the "
    "user must inspect the actual thing. Specify source/licensing or generation needs.\n"
    "7. DATA & PLOTS: when present, specify units, uncertainty, legends/direct labels, accessible "
    "series distinctions beyond color, hover and keyboard inspection, zoom/filter controls, empty "
    "data behavior, and figure/data export.\n"
    "8. VOICE & MOTION: concise domain-appropriate strings for real states and actions. Do not add "
    "visible prose explaining the interface. Motion is purposeful, respects reduced motion, uses "
    "100-150ms micro feedback and 200-300ms transitions, and never blocks work.\n"
    "9. ACCEPTANCE: list observable checks appropriate to the medium: no overlap/clipping, "
    "contrast and real asset rendering everywhere; keyboard/focus/responsive behavior for UI; "
    "crop, bleed, page/slide consistency, caption legibility, timing, dimensions and export "
    "integrity for static or sequential media.\n"
    "FOUNDATION FIRST: tokens, typography, layout primitives, icons, and shared controls precede "
    "feature views. Restraint law: remove anything whose absence loses no meaning. Markdown, under "
    "2400 words."
)

TOKENS_SCHEMA = {
    "type": "object",
    "properties": {
        "accent": {"type": "string", "description": "the single brand color, #RRGGBB"},
        "background": {"type": "string", "description": "the darkest surface, #RRGGBB"},
        "surface": {"type": "string", "description": "one step lighter than background"},
        "on_dark": {"type": "string", "description": "primary text on dark surfaces"},
        "icon": {
            "type": "object",
            "properties": {
                "glyph": {"type": "string", "enum": list(GLYPHS)},
                "background": {"type": "string"},
                "foreground": {"type": "string"},
            },
            "required": ["glyph"],
        },
    },
    "required": ["accent", "background", "icon"],
}

TOKENS_SYSTEM = (
    "You are spiral's DESIGN DIRECTOR distilling a design brief into machine-usable "
    "tokens. Output JSON only.\n"
    "- accent: the ONE brand color as #RRGGBB (match the brief's accent).\n"
    "- background: the primary canvas #RRGGBB chosen for this domain; surface: the adjacent "
    "raised or grouped-content surface; on_dark: primary text when a dark surface is used.\n"
    "- icon.glyph: pick the SINGLE mark from the allowed set that best fits the product's "
    "concept (e.g. an eye for surveillance, a lock for privacy, a bubble for chat, a "
    "spiral by default). icon colors must remain legible at small sizes and belong to the "
    "brief's palette.\n"
    "Return ONLY JSON matching the schema."
)


def design_tokens(goal: str, spec: list[dict], brief: str = "", cfg: Config | None = None,
                  ol: Ollama | None = None, progress=None) -> tuple[dict, ChatResult]:
    """Distill the prose brief into concrete tokens (accent/surfaces + icon choice)
    that the harness turns into real theme values and a launcher icon. Cheap,
    schema-constrained, think-off — small reliable output, not another essay."""
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    reqs = "\n".join(f"{r['id']}: {r['text']}" for r in spec)
    user = (
        f"GOAL:\n{goal}\n\nREQUIREMENTS:\n{reqs}\n\n"
        f"DESIGN BRIEF:\n{brief[:4000]}\n\nReturn the design tokens as JSON."
    )
    res = _plan_chat(TOKENS_SYSTEM, user, cfg, ol, temperature=0.2, schema=TOKENS_SCHEMA,
                     think=False, max_tokens=min(cfg.planner_max_tokens, 2048),
                     progress=progress)
    return _extract_json(res.text), res


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
    "- A scaffold, placeholder, TODO, dead control, hard-coded fake result, unreachable route, "
    "or happy-path-only implementation is partial or missing, never implemented.\n"
    "- Judge the finished workflow, not file volume. Verify inputs reach real domain logic, "
    "outputs are inspectable/exportable where appropriate, failures recover cleanly, and "
    "tests exercise behavior rather than presence.\n"
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
    requirements: list[str] = field(default_factory=list)


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
                    {"title": t.title, "description": t.description, "files": t.files,
                     "verify": t.verify, "requirements": t.requirements}
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
    max_tokens: int | None = None,
    progress=None,
) -> ChatResult:
    """Structured planning call with a fallback ladder. Thinking mode can consume
    the whole num_predict budget and return EMPTY content (the original
    think-forever disease, at the conductor level). Ladder: think → think (retry)
    → think OFF [→ fallback model, think OFF]. The think-off rungs cannot ramble,
    so this never returns empty."""
    name = model or cfg.planner.name
    # Remote reasoning calls already perform one low-effort final-answer recovery
    # inside the provider adapter. Repeating the identical expensive rung here can
    # spend tens of thousands of tokens without adding a new recovery strategy.
    ladder: list[tuple[str, bool]] = (
        [(name, think), (name, False)]
        if name in getattr(ol, "providers", {})
        else [(name, think), (name, think), (name, False)]
    )
    if fallback_model and fallback_model != name:
        ladder.append((fallback_model, False))
    res = None
    for m, th in ladder:
        res = ol.chat(
            m,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            think=th,
            num_predict=max_tokens or cfg.planner_max_tokens,
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
            Task(t["title"], t.get("description", ""), t.get("files", []) or [],
                 t.get("verify", "") or "", t.get("requirements", []) or [])
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
                     cfg, ol, temperature=0.2, schema=SPEC_SCHEMA,
                     max_tokens=min(cfg.planner_max_tokens, 8192),
                     progress=progress)
    return _extract_json(res.text).get("requirements", []), res


def analyze_deliverables(
    goal: str, spec: list[dict], repomap: str = "",
    cfg: Config | None = None, ol: Ollama | None = None, progress=None,
) -> tuple[dict, ChatResult]:
    cfg = cfg or Config.load()
    ol = ol or Ollama(cfg.base_url)
    requirements = "\n".join(
        f"{row.get('id')}: {row.get('text')}" for row in spec)
    user = (
        f"GOAL:\n{goal}\n\nREQUIREMENTS:\n{requirements}\n\n"
        f"CURRENT REPOSITORY SIGNALS:\n{repomap[:20000] or '(empty)'}\n\n"
        "Return the deliverable manifest."
    )
    res = _plan_chat(
        ARTIFACT_SYSTEM, user, cfg, ol, temperature=0.15,
        schema=ARTIFACT_SCHEMA, think=cfg.planner.think,
        max_tokens=min(cfg.planner_max_tokens, 6144), progress=progress,
    )
    data = _extract_json(res.text)
    rows = [
        row for row in data.get("deliverables", [])
        if isinstance(row, dict) and row.get("id") and row.get("kind")
    ]
    for row in rows:
        globs = []
        for raw in row.get("output_globs") or []:
            pattern = str(raw).strip().removeprefix("./")
            if (
                pattern and not pattern.startswith(("/", "~"))
                and ".." not in Path(pattern).parts
            ):
                globs.append(pattern)
        row["output_globs"] = (
            list(dict.fromkeys(globs))[:12]
            or default_output_globs(str(row.get("kind") or "other"))
        )
        row["root_hint"] = str(row.get("root_hint") or ".").strip() or "."
    if not rows:
        rows = [{
            "id": "D1", "kind": "other", "description": goal[:300],
            "root_hint": ".", "visual": False, "interactive": False,
            "output_globs": default_output_globs("other"),
            "acceptance_evidence": [], "tool_families": [],
        }]
    primary = str(data.get("primary_id") or rows[0]["id"])
    if primary not in {str(row["id"]) for row in rows}:
        primary = str(rows[0]["id"])
    return {"schema_version": 1, "primary_id": primary, "deliverables": rows}, res


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


_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "that", "this", "is",
    "are", "be", "should", "must", "will", "can", "when", "each", "every", "into",
    "from", "user", "users", "app", "screen", "page", "view", "button", "which",
    "their", "them", "they", "have", "has", "show", "shows", "display", "using", "use",
}


def _terms(text: str) -> set[str]:
    """Distinctive lowercase words (≥4 chars, not stopwords) — the fingerprint of
    a requirement that a covering task would almost certainly echo."""
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in _STOP}


def coverage_gaps(spec: list[dict], plan: Plan) -> list[str]:
    """Deterministic coverage: a requirement whose distinctive terms appear in NO
    task is very likely forgotten. Zero tokens, conservative (flags only when none
    of the terms match anywhere) — coverage becomes a mechanical diff, not vibes.
    The most common 'logical gap' is a requirement nobody planned for; this catches
    it before execution instead of at the final spec audit."""
    tasks = [task for milestone in plan.milestones for task in milestone.tasks]
    declared = {
        str(identifier)
        for task in tasks for identifier in (task.requirements or [])
        if str(identifier).strip()
    }
    known = {str(row.get("id") or "") for row in spec}
    gaps: list[str] = []
    if declared:
        for unknown in sorted(declared - known):
            gaps.append(f"plan maps a task to unknown requirement {unknown}")
        for row in spec:
            if str(row.get("id") or "") not in declared:
                gaps.append(
                    f"requirement {row.get('id', '?')} is UNCOVERED by explicit task mapping: "
                    f"\"{row.get('text', '')[:90]}\"")
        return gaps

    haystack = " ".join(
        f"{t.title} {t.description}" for m in plan.milestones for t in m.tasks
    ).lower()
    for r in spec:
        terms = _terms(r.get("text", ""))
        if terms and not any(term in haystack for term in terms):
            missed = ", ".join(sorted(terms)[:3])
            gaps.append(f"requirement {r.get('id', '?')} may be UNCOVERED: "
                        f"\"{r.get('text', '')[:70]}\" — no task mentions {missed}")
    return gaps


def normalize_plan_requirements(spec: list[dict], plan: Plan) -> int:
    """Convert requirement prose emitted by a planner back to canonical ``R<n>`` IDs."""

    known = {str(row.get("id") or "").strip().lower(): str(row.get("id") or "")
             for row in spec}
    texts = {
        re.sub(r"\W+", " ", str(row.get("text") or "").lower()).strip(): str(row.get("id") or "")
        for row in spec
    }
    changed = 0
    for milestone in plan.milestones:
        for task in milestone.tasks:
            normalized: list[str] = []
            for raw in task.requirements or []:
                value = str(raw).strip()
                canonical = known.get(value.lower())
                plain = re.sub(r"\W+", " ", value.lower()).strip()
                if not canonical:
                    canonical = texts.get(plain)
                if not canonical and len(plain) >= 16:
                    matches = [
                        identifier for text, identifier in texts.items()
                        if plain in text or text in plain
                    ]
                    canonical = matches[0] if len(set(matches)) == 1 else None
                if canonical and canonical not in normalized:
                    normalized.append(canonical)
                if canonical != value:
                    changed += 1
            task.requirements = normalized
    return changed


def ensure_plan_coverage(spec: list[dict], plan: Plan) -> int:
    """Add an explicit task for every requirement the planner did not map.

    Lexical similarity is deliberately not evidence of coverage. A planner must name
    the canonical requirement id (or its full normalized prose); otherwise the
    deterministic repair creates a task carrying the complete requirement.
    """

    normalize_plan_requirements(spec, plan)
    tasks = [task for milestone in plan.milestones for task in milestone.tasks]
    declared = {
        str(identifier) for task in tasks for identifier in (task.requirements or [])
    }
    additions: list[Task] = []
    for row in spec:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in declared:
            continue
        text = str(row.get("text") or f"Complete requirement {identifier}")
        additions.append(Task(
            title=f"Complete {identifier}: {text[:64]}",
            description=(
                f"Implement requirement {identifier} in full: {text}. Add behavioral "
                "coverage and integrate it with every affected deliverable. Do not "
                "treat build success or file presence as proof of this behavior."
            ),
            verify=str(row.get("check") or ""),
            requirements=[identifier],
        ))
        declared.add(identifier)
    if additions:
        plan.milestones.append(Milestone("Acceptance coverage", additions))
    return len(additions)


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
                     fallback_model=cfg.planner.name,
                     max_tokens=min(cfg.planner_max_tokens, 8192),
                     progress=progress)
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
                     fallback_model=cfg.planner.name,
                     max_tokens=min(cfg.planner_max_tokens, 6144),
                     progress=progress)
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
