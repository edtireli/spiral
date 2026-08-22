"""The live cockpit for `spiral prose`.

Same instrument as `build` and `research`: a pinned plan showing which stage is running,
a spinner with the model and live token rate, and an "idea" line saying *why* the current
stage exists. A rewrite pass is long and mostly silent otherwise — segment after segment
handed to a model — and a silent long-running job is indistinguishable from a hung one.

The plan is fixed rather than model-authored, because prose editing genuinely is the
same five stages every time.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from spiral.dash import Dash
from spiral.planner import Milestone, Plan, Task


def prose_plan(*, with_corpus: bool, rewriting: bool, deep: bool = False,
               beef_up: bool = False, restructure: bool = False,
               paper_audit: bool = False, tex_project: bool = False) -> Plan:
    """The stages this invocation will actually run — a plan listing work that will be
    skipped is a plan you stop trusting."""
    milestones = [
        Milestone("read the document", [
            Task("parse structure", "Split into editable prose and protected scaffolding "
                                    "(math, preamble, headings, figures)."),
            Task("measure prose", "Sentence length, hedging, passive rate, citation and "
                                  "equation density."),
            Task("detect AI tells", "Match against the mined Wikipedia catalogue plus "
                                    "structural and formatting patterns."),
        ]),
    ]
    if deep:
        milestones.append(Milestone("field research", [
            Task("classify article", "Use length, section, and citation evidence to decide "
                                     "whether scholarly retrieval belongs in this run."),
            Task("gather close full text", "Search the relevant scholarly channels, fetch "
                                           "primary text, deduplicate, and rank by topic."),
            Task("mine field profile", "Persist the selected-paper manifest and derive "
                                       "structural, lexical, and quantitative targets."),
        ]))
    elif with_corpus:
        milestones.append(Milestone("field template", [
            Task("mine exemplars", "Read the corpus and derive quantitative style bands."),
            Task("score against the band", "Report where this document sits outside the "
                                           "field's measured range."),
        ]))
    if rewriting:
        milestones.append(Milestone("rewrite", [
            Task("rewrite segments", "One paragraph at a time; the scorer decides each."),
            Task("guard the facts", "Reject any rewrite that drops numbers, citations or "
                                    "equations."),
        ]))
        output_tasks = [
            Task("write the document", "Back through its own structure; original untouched."),
        ]
        if restructure:
            output_tasks.append(Task(
                "align section arc",
                "Move intact native section blocks toward the corpus-observed order.",
            ))
        if beef_up:
            output_tasks.append(Task(
                "add cited detail",
                "Add only evidence-anchored claims that pass citation and entailment checks.",
            ))
        if paper_audit:
            output_tasks.append(Task(
                "audit the paper",
                "Write a separate corpus-grounded report of high-level improvements.",
            ))
        if tex_project:
            output_tasks.append(Task(
                "compile TeX copy",
                "Run latexmk in the copied project and surface unresolved build errors.",
            ))
        output_tasks.append(Task(
            "re-measure", "Report the score after every accepted operation.",
        ))
        milestones.append(Milestone("write out", output_tasks))
    return Plan(
        "measure the writing, then change it only where the measurement says so",
        milestones)


class ProseProgress:
    """Thin, explicit wrapper over :class:`Dash`.

    Deliberately not string-sniffing like the research UI: prose calls these methods
    directly, so a renamed log line can never silently stop advancing the plan."""

    def __init__(self, console: Console, *, with_corpus: bool = False,
                 rewriting: bool = False, deep: bool = False,
                 beef_up: bool = False, restructure: bool = False,
                 paper_audit: bool = False,
                 tex_project: bool = False,
                 thought_log: str | Path | None = None):
        self.dash = Dash(console=console,
                         plan=prose_plan(with_corpus=with_corpus, rewriting=rewriting,
                                         deep=deep, beef_up=beef_up,
                                         restructure=restructure,
                                         paper_audit=paper_audit,
                                         tex_project=tex_project),
                         gate="raw tell removal + field distance + content invariants",
                         thought_log=thought_log)
        self._active: tuple[int, int] | None = None
        self._watcher = None

    def __enter__(self) -> "ProseProgress":
        self.dash.__enter__()
        self.dash.mode = "auto"
        self.dash.phase("reading")
        try:
            from spiral.keys import Watcher

            self._watcher = Watcher().start()
            self._watcher.on_key("t", self.dash.toggle_thoughts)
            self._watcher.on_key("T", self.dash.toggle_thoughts)
        except Exception:
            self._watcher = None
        return self

    def __exit__(self, *exc) -> None:
        if self._active:
            self.dash.task(*self._active, "done")
        if self._watcher:
            self._watcher.stop()
        self.dash.__exit__(*exc)

    # -- plan movement ------------------------------------------------------
    def stage(self, mi: int, ti: int, *, phase: str = "", idea: str = "",
              detail: str = "", model: str = "") -> None:
        if self._active and self._active != (mi, ti):
            self.dash.task(*self._active, "done")
        self._active = (mi, ti)
        self.dash.task(mi, ti, "run")
        if phase:
            self.dash.phase(phase, model)
        if idea:
            self.dash.idea(idea)
        if detail:
            self.dash.detail(detail)

    def done(self, mi: int, ti: int) -> None:
        self.dash.task(mi, ti, "done")
        if self._active == (mi, ti):
            self._active = None

    def blocked(self, mi: int, ti: int) -> None:
        self.dash.task(mi, ti, "blocked")
        if self._active == (mi, ti):
            self._active = None

    # -- passthrough --------------------------------------------------------
    def detail(self, s: str) -> None:
        self.dash.detail(s)

    def idea(self, s: str) -> None:
        self.dash.idea(s)

    def tokens(self, n: int) -> None:
        self.dash.set_tokens(n)

    def print(self, *a, **k) -> None:
        self.dash.print(*a, **k)
