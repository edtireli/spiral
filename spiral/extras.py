"""Small quality-of-life commands: note · stats · rewind."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from spiral import tools
from spiral.theme import CLAY


def add_note(c: Console, workspace: str, text: str) -> None:
    """Project wisdom the workers ALWAYS see (rides every prompt as a skill)."""
    d = Path(workspace).resolve() / ".spiral" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "project-notes.md"
    if not f.is_file():
        f.write_text(
            "---\n"
            "name: project-notes\n"
            "description: Notes recorded by the developer — conventions, decisions, warnings. Always relevant.\n"
            "---\n# Project notes\n"
        )
    with f.open("a") as fh:
        fh.write(f"- {text}\n")
    c.print(f"  [green]●[/] noted — workers will see it on every task [dim]({f})[/]")


def show_stats(c: Console, workspace: str) -> None:
    f = Path(workspace).resolve() / ".spiral" / "ledger.jsonl"
    if not f.is_file():
        c.print("  [dim]no ledger yet — run something first[/]")
        return
    rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    attempts = [r for r in rows if r.get("kind") == "attempt"]
    plans = [r for r in rows if r.get("kind") == "plan"]
    vals = [r for r in rows if r.get("kind") == "validate"]

    tok = sum(r.get("ptok", 0) + r.get("ctok", 0) for r in rows)
    c.print(f"\n[bold {CLAY}]spiral stats[/] [dim]{f.parent.parent.name}[/]\n")
    c.print(f"  ledger rows      {len(rows)}  ·  total tokens {tok:,}")
    c.print(f"  attempts         {len(attempts)}  ·  green {sum(1 for a in attempts if a.get('verify_exit') == 0)}"
            f"  ·  red {sum(1 for a in attempts if a.get('verify_exit') not in (0, None))}")

    per_model: dict[str, list] = defaultdict(list)
    for a in attempts:
        if a.get("tps"):
            per_model[a["model"]].append(a["tps"])
    for m, tps in sorted(per_model.items()):
        c.print(f"  {m:24s} {len(tps):3d} attempts · median {sorted(tps)[len(tps) // 2]:.1f} tok/s")

    if plans:
        c.print(f"  plan-family calls {len(plans)} ({', '.join(sorted({p.get('phase', '?') for p in plans}))})")
    for v in vals[-3:]:
        c.print(f"  validation r{v.get('round')}: [green]{v.get('implemented', 0)}✓[/] "
                f"[yellow]{v.get('partial', 0)}◐[/] [red]{v.get('missing', 0)}✗[/] "
                f"[yellow]{v.get('unjudged', 0)}?[/]")
    c.print()


def rewind(c: Console, workspace: str, n: int | None) -> None:
    """List spiral's task checkpoints on the current spiral/ branch; optionally
    hard-reset to one. Refuses to touch non-spiral branches."""
    ws = Path(workspace).resolve()
    branch = tools.run("git branch --show-current", ws).out.strip()
    if not branch.startswith("spiral/"):
        c.print(f"  [red]refusing[/] — current branch [bold]{branch}[/] is yours, not spiral's. "
                "Rewind only operates on spiral/run-* branches.")
        return
    log = tools.run("git log --oneline -15", ws).out.splitlines()
    if n is None:
        c.print(f"\n[bold {CLAY}]checkpoints on {branch}[/] [dim](spiral rewind <n> to reset)[/]\n")
        for i, ln in enumerate(log):
            c.print(f"  [{'bold' if i == 0 else 'dim'}]{i:2d}[/]  {ln}")
        c.print()
        return
    if not (0 < n < len(log)):
        c.print(f"  [red]no checkpoint {n}[/] (0-{len(log) - 1} shown)")
        return
    sha = log[n].split()[0]
    ans = input(f"  hard-reset {branch} to {n}: '{log[n][:70]}'? [y/N] ").strip().lower()
    if ans != "y":
        c.print("  [dim]unchanged[/]")
        return
    tools.run(f"git reset --hard -q {sha}", ws)
    c.print(f"  [green]⟲ rewound[/] to {sha} — later checkpoints still recoverable via git reflog")
