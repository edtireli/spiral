"""Finite execution policy and evidence-carrying orchestration primitives.

This module is deliberately model-independent.  Builder and Research can share the
same bounded run ledger and the same serialisable task/evidence graph without either
engine knowing how the other prompts a model.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ComplexityTier(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


@dataclass(frozen=True)
class BudgetLimits:
    wall_seconds: int
    total_tokens: int
    model_calls: int

    @classmethod
    def for_tier(cls, tier: str | ComplexityTier) -> "BudgetLimits":
        try:
            selected = ComplexityTier(str(getattr(tier, "value", tier)).lower())
        except ValueError:
            selected = ComplexityTier.STANDARD
        return {
            ComplexityTier.QUICK: cls(30 * 60, 80_000, 24),
            ComplexityTier.STANDARD: cls(2 * 60 * 60, 350_000, 96),
            ComplexityTier.DEEP: cls(6 * 60 * 60, 1_200_000, 320),
            ComplexityTier.EXHAUSTIVE: cls(12 * 60 * 60, 3_000_000, 720),
        }[selected]


class BudgetExceeded(RuntimeError):
    def __init__(self, dimension: str, snapshot: dict[str, Any], detail: str = ""):
        self.dimension = dimension
        self.snapshot = snapshot
        self.detail = detail
        super().__init__(detail or f"run {dimension} budget exhausted")


class RunBudget:
    """One wall/token/call ledger shared by every role in an orchestration run."""

    def __init__(self, limits: BudgetLimits, *, clock=time.monotonic,
                 paused_clock=None):
        if min(limits.wall_seconds, limits.total_tokens, limits.model_calls) <= 0:
            raise ValueError("wall, token, and call budgets must all be finite and positive")
        self.limits = limits
        self._clock = clock
        if paused_clock is None:
            # Import lazily so the model-independent ledger stays usable on its own.
            # A disabled runtime controller returns zero and has no background work.
            from spiral.runtime_control import paused_seconds
            paused_clock = paused_seconds
        self._paused_clock = paused_clock
        self._paused_at_start = max(0.0, float(paused_clock()))
        self.started_at = clock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed_seconds(self) -> float:
        paused = max(
            0.0,
            float(self._paused_clock()) - self._paused_at_start,
        )
        return max(0.0, self._clock() - self.started_at - paused)

    @property
    def exhausted(self) -> bool:
        return bool(self.exhausted_dimension())

    def exhausted_dimension(self) -> str:
        if self.elapsed_seconds >= self.limits.wall_seconds:
            return "wall"
        if self.total_tokens >= self.limits.total_tokens:
            return "token"
        if self.calls >= self.limits.model_calls:
            return "call"
        return ""

    def begin_call(
        self,
        requested_completion_tokens: int | None,
        *,
        prompt_token_reserve: int = 0,
        minimum_completion_tokens: int = 1,
        unrecorded_tokens: int = 0,
    ) -> int:
        """Admit one provider request and return its hard completion-token cap.

        Prompt usage is known exactly only after a response. Callers reserve a
        conservative upper bound before dispatch so the advertised *total* token budget
        remains a real request-time ceiling rather than a post-hoc statistic.
        ``unrecorded_tokens`` covers an earlier attempt whose usage will be recorded when
        the logical chat returns.
        """
        dimension = self.exhausted_dimension()
        if dimension:
            raise BudgetExceeded(dimension, self.snapshot())
        reserve = max(0, int(prompt_token_reserve or 0))
        pending = max(0, int(unrecorded_tokens or 0))
        minimum = max(1, int(minimum_completion_tokens or 1))
        available = self.limits.total_tokens - self.total_tokens - pending - reserve
        if available < minimum:
            raise BudgetExceeded(
                "token",
                self.snapshot(),
                detail=(
                    f"run token budget cannot fit this model call: {available} "
                    f"completion token(s) remain after reserving {reserve} for the "
                    f"prompt, but the provider requires at least {minimum}"
                ),
            )
        requested = (
            available if requested_completion_tokens is None
            else int(requested_completion_tokens)
        )
        allocation = min(requested, available)
        if allocation < minimum:
            raise BudgetExceeded(
                "token",
                self.snapshot(),
                detail=(
                    f"requested completion cap {allocation} is below the provider "
                    f"minimum of {minimum} token(s)"
                ),
            )
        self.calls += 1
        return allocation

    def begin_retry(
        self,
        requested_completion_tokens: int | None = None,
        **admission,
    ) -> int:
        """Admit and account for a provider retry hidden inside one logical chat."""
        return self.begin_call(requested_completion_tokens, **admission)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.completion_tokens += max(0, int(completion_tokens or 0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "limits": asdict(self.limits),
            "used": {
                "wall_seconds": round(self.elapsed_seconds, 3),
                "total_tokens": self.total_tokens,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "model_calls": self.calls,
            },
            "remaining": {
                "wall_seconds": max(0, round(self.limits.wall_seconds - self.elapsed_seconds, 3)),
                "total_tokens": max(0, self.limits.total_tokens - self.total_tokens),
                "model_calls": max(0, self.limits.model_calls - self.calls),
            },
            "exhausted": self.exhausted_dimension() or None,
        }


class EvidenceLevel(str, Enum):
    COMPILE = "compile"
    TEST = "test"
    BEHAVIOR = "behavior"
    SOURCE = "source"
    HUMAN = "human"


class UncertaintyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EvidenceRecord:
    level: EvidenceLevel
    claim: str
    outcome: str = "pass"
    artifact: str = ""
    command: str = ""
    source: str = ""
    observed_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["level"] = self.level.value
        return row


def evidence_level_for_command(command: str) -> EvidenceLevel:
    text = f" {command.lower()} "
    if any(token in text for token in (
        " pytest", " unittest", " test ", " tests", "gradlew test", "npm test",
        "cargo test", "go test", "xcodebuild test", "ctest",
    )):
        return EvidenceLevel.TEST
    if any(token in text for token in (
        " run ", "curl ", "playwright", "selenium", "maestro", "smoke",
        "integration", "e2e", "artifact_gate", "spec_gate",
    )):
        return EvidenceLevel.BEHAVIOR
    return EvidenceLevel.COMPILE


@dataclass
class EvidenceReport:
    records: list[EvidenceRecord] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        passed = {record.level.value for record in self.records if record.outcome == "pass"}
        coverage = {level.value: level.value in passed for level in EvidenceLevel}
        reasons = list(dict.fromkeys(str(value) for value in self.unresolved if str(value).strip()))
        if reasons:
            uncertainty = UncertaintyLevel.HIGH
        elif coverage[EvidenceLevel.BEHAVIOR.value] and (
            coverage[EvidenceLevel.TEST.value] or coverage[EvidenceLevel.SOURCE.value]
        ):
            uncertainty = UncertaintyLevel.LOW
        elif any(coverage.values()):
            uncertainty = UncertaintyLevel.MEDIUM
            reasons.append("evidence does not independently cover both behavior and tests/sources")
        else:
            uncertainty = UncertaintyLevel.HIGH
            reasons.append("no verification evidence was recorded")
        return {
            "schema_version": 1,
            "levels": [level.value for level in EvidenceLevel if coverage[level.value]],
            "coverage": coverage,
            "records": [record.to_dict() for record in self.records],
            "uncertainty": {"level": uncertainty.value, "reasons": reasons},
        }


class TaskKind(str, Enum):
    BUILD = "build"
    RESEARCH = "research"
    VERIFY = "verify"
    HANDOFF = "handoff"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class TaskNode:
    id: str
    title: str
    kind: TaskKind = TaskKind.BUILD
    dependencies: list[str] = field(default_factory=list)
    required_evidence: list[EvidenceLevel] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    evidence: list[EvidenceRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind.value,
            "dependencies": list(self.dependencies),
            "required_evidence": [level.value for level in self.required_evidence],
            "state": self.state.value,
            "evidence": [record.to_dict() for record in self.evidence],
            "metadata": dict(self.metadata),
        }


class TaskEvidenceDAG:
    """Typed, serialisable DAG with deterministic single-node claiming.

    Execution is intentionally serial even when independent nodes exist.  A local
    27B model stays resident once, while dependencies/evidence remain explicit and
    a future host may safely schedule more concurrency without changing the file
    contract.
    """

    def __init__(self, nodes: Iterable[TaskNode] = ()):
        self.nodes = {node.id: node for node in nodes}
        self._validate()

    @classmethod
    def from_plan(cls, plan: Any) -> "TaskEvidenceDAG":
        nodes: list[TaskNode] = []
        previous = ""
        for mi, milestone in enumerate(getattr(plan, "milestones", []), 1):
            for ti, task in enumerate(getattr(milestone, "tasks", []), 1):
                command = str(getattr(task, "verify", "") or "")
                required = [evidence_level_for_command(command)] if command else []
                node = TaskNode(
                    id=f"{mi}.{ti}", title=str(getattr(task, "title", "")),
                    dependencies=[previous] if previous else [],
                    required_evidence=required,
                    metadata={
                        "milestone": str(getattr(milestone, "title", "")),
                        "files": list(getattr(task, "files", []) or []),
                        "requirements": list(getattr(task, "requirements", []) or []),
                        "exports": list(getattr(task, "exports", []) or []),
                        "imports": list(getattr(task, "imports", []) or []),
                    },
                )
                nodes.append(node)
                previous = node.id
        return cls(nodes)

    @classmethod
    def research_pipeline(cls, topic: str) -> "TaskEvidenceDAG":
        stages = [
            ("research.sources", "discover and ground sources", TaskKind.RESEARCH, EvidenceLevel.SOURCE),
            ("research.question", "form a scoped question", TaskKind.RESEARCH, EvidenceLevel.SOURCE),
            ("research.verify", "verify claims and replications", TaskKind.VERIFY, EvidenceLevel.BEHAVIOR),
            ("research.handoff", "produce evidence-bearing handoff", TaskKind.HANDOFF, EvidenceLevel.SOURCE),
        ]
        nodes, previous = [], ""
        for node_id, title, kind, evidence in stages:
            nodes.append(TaskNode(
                node_id, title, kind, [previous] if previous else [], [evidence],
                metadata={"topic": topic},
            ))
            previous = node_id
        return cls(nodes)

    def _validate(self) -> None:
        for node in self.nodes.values():
            missing = set(node.dependencies) - set(self.nodes)
            if missing:
                raise ValueError(f"{node.id} has missing dependencies: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("task graph contains a dependency cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in self.nodes[node_id].dependencies:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in self.nodes:
            visit(node_id)

    def ready(self) -> list[TaskNode]:
        # Serial dependencies encode ordering. A blocked task remains explicit
        # evidence debt but must not deadlock the conductor's documented
        # "continue with the rest of the plan" behavior.
        terminal = {TaskState.COMPLETE, TaskState.BLOCKED, TaskState.SKIPPED}
        return [
            node for node in self.nodes.values()
            if node.state == TaskState.PENDING
            and all(self.nodes[dep].state in terminal for dep in node.dependencies)
        ]

    def start(self, node_id: str) -> TaskNode:
        node = self.nodes[node_id]
        if node not in self.ready():
            raise RuntimeError(f"task {node_id} is not ready")
        node.state = TaskState.RUNNING
        return node

    def finish(self, node_id: str, state: TaskState | str,
               evidence: Iterable[EvidenceRecord] = ()) -> TaskNode:
        node = self.nodes[node_id]
        selected = TaskState(str(getattr(state, "value", state)))
        if selected not in {TaskState.COMPLETE, TaskState.BLOCKED, TaskState.SKIPPED}:
            raise ValueError("a finished node must be complete, blocked, or skipped")
        node.state = selected
        node.evidence.extend(evidence)
        return node

    def evidence_report(self, unresolved: Iterable[str] = ()) -> EvidenceReport:
        records = [record for node in self.nodes.values() for record in node.evidence]
        missing = []
        for node in self.nodes.values():
            passed = {record.level for record in node.evidence if record.outcome == "pass"}
            for required in node.required_evidence:
                if required not in passed:
                    missing.append(f"{node.id} lacks required {required.value} evidence")
            if node.state in {TaskState.BLOCKED, TaskState.SKIPPED}:
                missing.append(f"{node.id} is {node.state.value}")
            elif node.state in {TaskState.PENDING, TaskState.RUNNING}:
                missing.append(f"{node.id} is not complete ({node.state.value})")
        return EvidenceReport(records, [*unresolved, *missing])

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "execution": "serial", "nodes": [
            node.to_dict() for node in self.nodes.values()
        ]}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(temporary, target)


@dataclass(frozen=True)
class OrchestrationPolicy:
    complexity: ComplexityTier = ComplexityTier.STANDARD
    prefer_single_resident_model: bool = True

    @classmethod
    def from_values(cls, complexity: str, prefer_single_resident_model: bool = True):
        try:
            tier = ComplexityTier(str(complexity).lower())
        except ValueError:
            tier = ComplexityTier.STANDARD
        return cls(tier, bool(prefer_single_resident_model))

    def critic_rounds(self, *, deterministic_defects: int, task_count: int,
                      requested_rounds: int) -> int:
        """Buy independent criticism only when observable risk warrants a swap."""
        if deterministic_defects:
            warranted = 1 if self.complexity in {ComplexityTier.QUICK, ComplexityTier.STANDARD} else 2
        elif self.complexity in {ComplexityTier.DEEP, ComplexityTier.EXHAUSTIVE}:
            warranted = 1 if task_count <= 12 else 2
        elif task_count > 12:
            warranted = 1
        else:
            warranted = 0
        return max(0, min(int(requested_rounds), warranted))

    def escalation_warranted(self, *, uncertainty: str, evidence_gap: bool) -> bool:
        return bool(evidence_gap or uncertainty in {"high", "unknown"})
