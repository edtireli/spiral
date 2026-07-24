"""Live cockpit for ``spiral research --solve``.

The research loop has the same problem as a long build: silence or raw breadcrumbs
make it feel stuck even when it is thinking, fetching, verifying, or writing. This
adapter reuses the main ``Dash`` pinned panel with a research-shaped plan.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from spiral.dash import Dash
from spiral.planner import Milestone, Plan, Task


def research_plan() -> Plan:
    return Plan(
        "research loop: deepen sources, propose checkable claims, verify, referee, write",
        [
            Milestone("corpus coverage", [
                Task("choose field/search plan", "Pick arXiv categories and focused search queries."),
                Task("gather keyword sources", "Fetch and ingest arXiv source material."),
                Task("snowball citation graph", "Follow references/citations and fill co-citation holes."),
                Task("read and validate corpus", "Ground notes in source anchors and pass deterministic coverage gates."),
            ]),
            Milestone("claim loop", [
                Task("discover/referee angles", "Schedule by information gain, deep-read, and probe counterfactual angle families."),
                Task("verify and replicate", "Verify claims, then regenerate required results from blinded briefs."),
                Task("bound prior art", "Run diversified searches and issue a scoped novelty-boundary certificate."),
                Task("reflect and gate", "Close the shared obligation graph with supervisor and deterministic checks."),
            ]),
            Milestone("paper", [
                Task("derive corpus style guide", "Extract section arc and mathematical register."),
                Task("draft and revise sections", "Write section-by-section, then coherence-pass."),
                Task("paper referee audit", "Check structure, citations, assumptions, and overclaiming."),
                Task("compile and bundle", "Compile a proof-carrying paper, checkpoint it, and register living-paper refresh rules."),
            ]),
        ],
    )


class ResearchProgress:
    """Callable UI callback consumed by :class:`ResearchLoop`.

    It accepts the existing text breadcrumbs, updates the pinned plan/status region,
    and prints a concise transcript above it.
    """

    def __init__(self, console: Console, workdir: str | Path = "spiral-research"):
        self.workdir = Path(workdir)
        self.dash = Dash(console=console, plan=research_plan(), gate="verified claims",
                         thought_log=self.workdir / "thoughts.jsonl")
        self._active: tuple[int, int] | None = None
        self._watcher = None

    def __enter__(self) -> "ResearchProgress":
        self.dash.__enter__()
        self.dash.mode = "auto"
        self.dash.phase("starting research")
        try:
            from spiral.keys import Watcher

            self._watcher = Watcher().start()
            self._watcher.on_key("t", self.dash.toggle_thoughts)
            self._watcher.on_key("T", self.dash.toggle_thoughts)
        except Exception:
            self._watcher = None
        return self

    def __exit__(self, *exc) -> None:
        if self._watcher:
            self._watcher.stop()
        self.dash.__exit__(*exc)

    def _task(self, mi: int, ti: int, state: str = "run", *, phase: str | None = None,
              detail: str | None = None) -> None:
        if self._active and self._active != (mi, ti) and self.dash.status.get(self._active) == "run":
            self.dash.task(*self._active, "done")
        self._active = (mi, ti)
        self.dash.task(mi, ti, state)
        if phase:
            self.dash.phase(phase)
        if detail:
            self.dash.detail(detail)

    def _done(self, mi: int, ti: int) -> None:
        self.dash.task(mi, ti, "done")
        if self._active == (mi, ti):
            self._active = None

    def __call__(self, msg: str) -> None:
        s = (msg or "").strip()
        if not s:
            return

        if s.startswith("search plan"):
            self._task(1, 1, "done", phase="search plan", detail=s)
            self.dash.idea("Choosing the field boundary and search terms before reading papers; bad categories make the whole corpus drift.")
            self.dash.print(f"  [green]●[/] {s}")
            return
        if s.startswith("── round"):
            self.dash.phase(s.replace("──", "").strip())
            self.dash.print(f"[bold rgb(217,119,87)]{s}[/]")
            return
        if s.startswith("gather ·"):
            self._task(1, 2, "run", phase="gathering corpus", detail=s)
            self.dash.idea("Building the local paper corpus from primary sources; these papers become both evidence and writing exemplars.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("+ "):
            self.dash.detail(s)
            self.dash.print(f"    [green]+[/] [dim]{s[2:]}[/]")
            return
        if s.startswith("graph ·"):
            self._task(1, 3, "run", phase="citation graph", detail=s)
            self.dash.idea("Following references and citations to expose foundational papers keyword search would miss.")
            return
        if s.startswith("corpus ·"):
            self._done(1, 2)
            self._done(1, 3)
            self.dash.phase("corpus coverage")
            self.dash.detail(s)
            self.dash.idea("Corpus frontier updated; next step is asking whether the gathered field coverage is actually enough.")
            self.dash.print(f"  [green]●[/] {s}")
            return
        if s.startswith("assess ·") or "sufficient" in s.lower():
            self._task(1, 4, "run", phase="assessing corpus", detail=s)
            self.dash.idea("Checking for missing definitions, constructions, or prior results before proposing claims.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("coverage ·"):
            self._task(1, 4, "run", phase="coverage gate", detail=s)
            self.dash.idea("Measuring source health, query diversity, usable primary text, topical coverage, and citation-frontier status.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("information gain ·"):
            self._task(1, 4, "run", phase="information scheduling", detail=s)
            self.dash.idea("Ranking search actions from uncovered terms, retrieval health, redundancy, and measured yield.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("read ·"):
            self._task(1, 4, "run", phase="reading corpus", detail=s)
            self.dash.idea("Compressing the paper corpus into reusable notes so the angle scout can read broadly without losing context.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("ideas ·"):
            self._done(1, 4)
            self._task(2, 1, "run", phase="idea families", detail=s)
            self.dash.idea("The thinking model is clustering the reading notes into possible research-question families.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("counterfactuals ·"):
            self._task(2, 1, "run", phase="counterfactual lab", detail=s)
            self.dash.idea("Changing one assumption or limit at a time to expose boundary cases, method transfers, and possible obstructions.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.strip().startswith("zoom ·"):
            self._task(2, 1, "run", phase="deep reading", detail=s)
            self.dash.idea("Zooming into the papers most relevant to a candidate family before committing to an angle.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.strip().startswith("angle ·"):
            self._done(1, 4)
            self._task(2, 1, "run", phase="angle scouting", detail=s)
            self.dash.idea("Checking candidate research questions against prior art and rejecting ones that are already known or too thin.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("proposal") or s.startswith("refine ·") or s.startswith("basis ·"):
            self._done(1, 4)
            self._task(2, 1, "run", phase="proposal referee", detail=s)
            self.dash.idea("Drafting a concrete research move, then forcing it through novelty, rigor, interest, and corpus-basis checks.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("✓") or s.startswith("✗"):
            self._done(2, 1)
            self._task(2, 2, "run", phase="verifying claims", detail=s)
            self.dash.idea("The model has proposed; now deterministic tools decide which claims survive.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("novelty ·"):
            self._done(2, 2)
            self._task(2, 3, "run", phase="prior art search", detail=s)
            self.dash.idea("Checking whether verified progress is actually new or just rediscovered literature.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("novelty boundary ·"):
            self._task(2, 3, "run", phase="novelty boundary", detail=s)
            self.dash.idea("Recording exactly which queries, sources, dates, nearest priors, and claim scope support the bounded novelty statement.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("reflect ·"):
            self._done(2, 3)
            self._task(2, 4, "run", phase="supervisor reflection", detail=s)
            self.dash.idea("Supervisor is deciding whether to continue, pivot, declare solved, or write down an honest limitation.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("completion ·"):
            self._done(2, 4)
            self.dash.phase("completion gate")
            self.dash.idea("A supervisor verdict cannot finish the run unless required claims, evidence strength, corpus coverage, and novelty-search health all pass.")
            self.dash.print(f"  [yellow]○[/] {s}")
            return
        if s.startswith("no verified") or s.startswith("exhausted"):
            self._done(2, 4)
            self.dash.phase("research decision")
            self.dash.print(f"  [yellow]○[/] {s}")
            return
        if s.startswith("write · style"):
            self._done(2, 4)
            self._task(3, 1, "run", phase="paper style guide", detail=s)
            self.dash.idea("Extracting the paper’s section arc, notation habits, vocabulary, and theorem/proof rhythm from the corpus.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("write · outline") or s.startswith("draft ·") or s.startswith("revise ·"):
            self._done(3, 1)
            self._task(3, 2, "run", phase="writing paper", detail=s)
            self.dash.idea("Writing section-by-section from verified findings, then revising for notation consistency and mathematical flow.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("audit ·"):
            self._done(3, 2)
            self._task(3, 3, "run", phase="paper audit", detail=s)
            self.dash.idea("Referee pass: checking citations, assumptions, proof claims, and whether the paper overstates what was verified.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("compile ·") or s.startswith("write · bundle"):
            self._done(3, 3)
            self._task(3, 4, "run", phase="compile and bundle", detail=s)
            self.dash.idea("Final gate: the paper must compile and include reproducibility artifacts before it is considered done.")
            self.dash.print(f"  [dim]{s}[/]")
            return
        if s.startswith("model ·"):
            self.dash.phase("model call")
            self.dash.detail(s)
            self.dash.print(f"  [yellow]○[/] {s}")
            return

        self.dash.detail(s)
        self.dash.print(f"  [dim]{s}[/]")
