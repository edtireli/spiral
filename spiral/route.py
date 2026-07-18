"""Signature routing — the harness learns from its own ledger.

Every attempt records the error signature it faced and whether it cleared it.
Folded across runs, those records say which signatures the fast lane actually
solves and which it only burns attempts on before the dense model lands them.
mine() builds the per-signature stats; decide() turns them into a verdict: a
signature the worker has repeatedly failed and only escalation has cleared is
routed straight to escalation next time it appears. The weights never change —
the harness around them gets smarter with every run.

`spiral distill` prints the table, writes .spiral/route.json, and appends hard
signatures to the learned-fixes skill.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

MIN_FAILS = 3  # worker failures on a signature before it can be ruled hard


def norm_sig(err: str) -> str:
    """A signature stable ACROSS runs: the same mistake lands on different lines
    every time, so line:col and hex addresses are stripped before matching."""
    s = re.sub(r"(\.[A-Za-z]{1,4}):\d+(?::\d+)?", r"\1", err.strip())
    s = re.sub(r"0x[0-9a-fA-F]+", "0x_", s)
    return s[:120]


@dataclass
class SigStat:
    worker_fail: int = 0
    worker_green: int = 0
    esc_fail: int = 0
    esc_green: int = 0
    tasks: set = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.worker_fail + self.worker_green + self.esc_fail + self.esc_green


def mine(ledger_path: str | Path, worker: str, escalation: str) -> dict[str, SigStat]:
    """Fold the ledger's attempt records into per-signature outcome stats.
    Attempts by models that are neither the current worker nor the current
    escalation (old crews, API tiers) are skipped — conservative by design."""
    path = Path(ledger_path)
    stats: dict[str, SigStat] = {}
    if not path.is_file():
        return stats
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        sig = norm_sig(rec.get("sig") or "")
        model = rec.get("model", "")
        if rec.get("kind") != "attempt" or not sig or model not in (worker, escalation):
            continue
        st = stats.setdefault(sig, SigStat())
        green = rec.get("verify_exit") == 0
        if model == escalation:
            st.esc_green += green
            st.esc_fail += not green
        else:
            st.worker_green += green
            st.worker_fail += not green
        if rec.get("task"):
            st.tasks.add(str(rec["task"])[:60])
    return stats


def decide(sig: str, stats: dict[str, SigStat]) -> bool:
    """True → send this signature straight to the escalation lane: the worker
    has never cleared it in MIN_FAILS+ tries and escalation has. A single
    worker win keeps the signature in the fast lane — routing must never
    take work away from a lane that can do it."""
    st = stats.get(norm_sig(sig))
    return bool(st and st.worker_green == 0 and st.worker_fail >= MIN_FAILS and st.esc_green >= 1)


def ensure_learned_fixes(ws: str | Path) -> Path:
    """The learned-fixes skill file, created with its frontmatter on first use."""
    d = Path(ws) / ".spiral" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "learned-fixes.md"
    if not f.is_file():
        f.write_text(
            "---\n"
            "name: learned-fixes\n"
            "description: Escalation-model fixes from THIS repo — error signatures the "
            "fast lane could not solve and the repairs that worked. Use when build errors "
            "or kotlin gradle android failures resemble these.\n"
            "---\n# Learned fixes (auto-distilled)\n"
        )
    return f


def _harvest_hard(ws: Path, hard: dict[str, SigStat]) -> int:
    """Append newly-ruled hard signatures to learned-fixes.md (once each)."""
    if not hard:
        return 0
    f = ensure_learned_fixes(ws)
    have = f.read_text()
    n = 0
    with f.open("a") as fh:
        for sig, st in sorted(hard.items()):
            if sig in have:
                continue
            tasks = ", ".join(sorted(t for t in st.tasks if t))[:160]
            fh.write(
                f"\n## hard signature\n`{sig}`\n"
                f"The fast lane failed this {st.worker_fail}x (tasks: {tasks or '?'}); "
                "only escalation cleared it. spiral now routes it straight to the dense model.\n"
            )
            n += 1
    return n


def distill(console: Console, workspace: str) -> None:
    """`spiral distill` — the deterministic ledger miner. No model calls: the
    report, the routing table, and the skill entries all come from records."""
    from spiral.config import Config

    cfg = Config.load()
    ws = Path(workspace).resolve()
    stats = mine(ws / ".spiral" / "ledger.jsonl", cfg.worker.name, cfg.escalation.name)
    if not stats:
        console.print("  [yellow]no attempt records with signatures in .spiral/ledger.jsonl[/] "
                      "[dim]— run spiral build first[/]\n")
        return
    hard = {s: st for s, st in stats.items() if decide(s, stats)}
    console.print(f"  [bold]{len(stats)}[/] signature(s) in the ledger · "
                  f"[bold]{len(hard)}[/] ruled hard → routed straight to escalation\n")
    for sig, st in sorted(stats.items(), key=lambda kv: -kv[1].total)[:12]:
        verdict = ("[rgb(217,119,87)]→ escalation[/]" if sig in hard
                   else "[green]worker[/]      " if st.worker_green
                   else "[dim]undecided[/]   ")
        console.print(f"  {verdict} [dim]w {st.worker_green}✓/{st.worker_fail}✗ · "
                      f"e {st.esc_green}✓/{st.esc_fail}✗[/]  {sig[:74]}")
    (ws / ".spiral").mkdir(parents=True, exist_ok=True)
    (ws / ".spiral" / "route.json").write_text(json.dumps(
        {s: {"worker_green": st.worker_green, "worker_fail": st.worker_fail,
             "esc_green": st.esc_green, "esc_fail": st.esc_fail,
             "route": "escalation" if s in hard else "worker"}
         for s, st in sorted(stats.items())}, indent=2))
    added = _harvest_hard(ws, hard)
    console.print("\n  [dim]routing table → .spiral/route.json"
                  + (f" · {added} hard signature(s) appended to learned-fixes.md" if added else "")
                  + "[/]\n")
