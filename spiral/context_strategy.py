"""Opt-in context-aware planning, not model admission or a second work owner.

The host supplies a measured inventory and a worker budget. The model chooses
task boundaries; this module checks their declared evidence footprint. Dispatch
still needs exact rendered-prompt tokenization and current memory admission.
There are no domain/model-name rules and no automatic context/model changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class EvidenceUnit:
    path: str
    tokens: int
    sha256: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanningEnvelope:
    model: str
    tokenizer_identity: str
    context_tokens: int
    output_reserve: int
    control_reserve: int
    units: tuple[EvidenceUnit, ...]

    def __post_init__(self):
        if not self.model or not self.tokenizer_identity:
            raise ValueError("exact model and tokenizer identities are required")
        for value in (self.context_tokens, self.output_reserve, self.control_reserve):
            if type(value) is not int or value <= 0:
                raise ValueError("context and reserves must be positive integer tokens")
        if self.context_tokens > 1048576 or self.source_budget <= 0:
            raise ValueError("invalid context envelope")
        paths = set()
        for unit in self.units:
            path = PurePosixPath(unit.path)
            if (not unit.path or path.is_absolute() or ".." in path.parts or
                    str(path) != unit.path or unit.path in paths):
                raise ValueError("inventory requires unique normalized relative paths")
            if type(unit.tokens) is not int or unit.tokens <= 0:
                raise ValueError("inventory requires positive measured token counts")
            if len(unit.sha256) != 64 or any(c not in "0123456789abcdef" for c in unit.sha256):
                raise ValueError("inventory requires content digests")
            paths.add(unit.path)
        if not paths or any(dep not in paths for unit in self.units for dep in unit.dependencies):
            raise ValueError("inventory must contain every declared dependency")

    @property
    def source_budget(self):
        return self.context_tokens - self.output_reserve - self.control_reserve

    def footprint(self, paths):
        inventory = {unit.path: unit for unit in self.units}
        paths = tuple(paths)
        scope = set(paths)
        unknown = scope - inventory.keys()
        if unknown:
            raise ValueError("unmeasured paths: " + ", ".join(sorted(unknown)))
        # Direct dependencies are included whole. A future execution may retrieve
        # contracts instead, but may not silently call a summary the original file.
        scope.update(dep for path in paths for dep in inventory[path].dependencies)
        return sorted(scope), sum(inventory[path].tokens for path in scope)

    def prompt_data(self):
        return {
            "schema": "spiral.planning-context.v1", "model": self.model,
            "tokenizer_identity": self.tokenizer_identity,
            "worker_context_tokens": self.context_tokens,
            "output_reserved_tokens": self.output_reserve,
            "control_reserved_tokens": self.control_reserve,
            "source_budget_tokens": self.source_budget,
            "whole_inventory_tokens": sum(unit.tokens for unit in self.units),
            "count_scope": "sum of separately tokenized units, not a dispatch receipt",
            "inventory": [{"path": unit.path, "tokens": unit.tokens,
                           "sha256": unit.sha256, "dependencies": list(unit.dependencies)}
                          for unit in self.units],
        }


CONTEXT_PLANNING_GUIDANCE = (
    "- Choose coherent task boundaries using the host's WORKER CONTEXT ENVELOPE. "
    "There is no fixed file-count cap. Prefer fewer complete dependency-coherent "
    "tasks when their working sets fit; split when capacity, risk or verification "
    "requires it. A larger available budget is not an instruction to fill it.\n"
    "- List the exact files each task will modify in files, and additional files "
    "it must read in context_reads (including newly introduced dependencies). "
    "For this measured-inventory planning mode, the declared working set is "
    "those edit/read files plus their direct "
    "dependencies, deduplicated. Its measured source-token sum must fit "
    "source_budget_tokens. Count dependency files even when another task edits them.\n"
    "- Preserve every requirement and explicit interfaces between split tasks. "
    "Do not reduce the requested product to fit a context window. If an indivisible "
    "working set does not fit, state the capacity problem; do not invent a smaller "
    "token count. Newly discovered files require measurement before execution.\n"
    "- Counts are planning evidence only: execution must separately admit the "
    "actual rendered prompt, memory, permissions, output headroom and cancellation.\n"
)


class ContextPlanRejected(ValueError):
    """Retain the model's proposal and specific defects for ordinary replanning."""
    def __init__(self, plan, report, attempts=()):
        self.plan = plan
        self.report = report
        self.attempts = tuple(attempts)
        super().__init__("context plan rejected: " + "; ".join(report["defects"]))


def declaration_defects(data):
    """Do not rely on a provider honoring JSON schema before a permissive parser."""
    defects = []
    milestones = data.get("milestones") if isinstance(data, dict) else None
    if not isinstance(milestones, list) or not milestones:
        return ["missing explicit milestones"]
    index = 0
    for milestone in milestones:
        tasks = milestone.get("tasks") if isinstance(milestone, dict) else None
        if not isinstance(tasks, list) or not tasks:
            defects.append("missing explicit tasks")
            continue
        for task in tasks:
            for key in ("files", "context_reads"):
                paths = task.get(key) if isinstance(task, dict) else None
                if (not isinstance(paths, list) or (key == "files" and not paths) or
                        any(not isinstance(p, str) or not p or p != p.strip() for p in paths)):
                    defects.append(f"task {index}: invalid or missing explicit {key}")
            index += 1
    return defects


def plan_context_report(plan, envelope):
    """No model calls, task edits, automatic repairs or execution permission."""
    rows, defects = [], []
    for index, task in enumerate(task for milestone in plan.milestones for task in milestone.tasks):
        reads = list(getattr(task, "context_reads", []))
        row = {"task": index, "files": list(task.files), "context_reads": reads}
        if not task.files:
            defects.append(f"task {index}: no declared measured files")
        try:
            scope, tokens = envelope.footprint([*task.files, *reads])
            row.update(evidence_files=scope, source_tokens=tokens, fits=tokens <= envelope.source_budget)
            if not row["fits"]:
                defects.append(f"task {index}: {tokens} source tokens exceed {envelope.source_budget}")
        except ValueError as exc:
            row.update(fits=False, error=str(exc))
            defects.append(f"task {index}: {exc}")
        rows.append(row)
    if not rows:
        defects.append("no tasks")
    return {"valid": not defects, "defects": defects, "tasks": rows,
            "source_budget_tokens": envelope.source_budget,
            "execution_admitted": False, "semantic_completeness_verified": False}
