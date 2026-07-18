"""Entry point.

    spiral                                   banner + ollama health
    spiral do   "<goal>" --verify "<cmd>"    one task → green (the atom)
    spiral plan  --goal-file F [--dir D]      show the conductor's decomposition (cheap)
    spiral build --goal-file F [--dir D]      plan, then execute the whole project
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from spiral.banner import print_banner
from spiral.theme import make_console, reveal
from spiral.config import Config
from spiral.llm import Ollama


def _health(console: Console) -> None:
    cfg = Config.load()
    version = Ollama(cfg.base_url).health()
    if version:
        reveal(console, f"  [green]●[/green] ollama {version}  ·  worker [bold]{cfg.worker.name}[/bold]\n")
    else:
        reveal(console, f"  [red]●[/red] ollama unreachable at {cfg.base_url}\n")


def _info_line(console: Console, workspace: str, *extra: str) -> None:
    """One useful line instead of echoing `dir .`: workspace · branch · gate.
    Extra lines cascade in after it (theme.reveal) instead of slamming down."""
    from spiral import tools
    from spiral.conductor import detect_gate

    ws = Path(workspace).resolve()
    parts = [f"[bold]{ws.name}[/]"]
    if (ws / ".git").is_dir():
        branch = tools.run("git branch --show-current", ws).out.strip()
        if branch:
            parts.append(branch)
    parts.append(f"gate: {detect_gate(ws) or 'none'}")
    reveal(console, "  [dim]▸[/] " + " [dim]·[/] ".join(parts) + "\n", *extra)


def _apply_tier(cfg, console, tier):
    """Remap roles onto the configured API model. None → local crew (default);
    'boost' → escalation + critic/validator/designer on the API; 'api' → all."""
    if not tier:
        return cfg
    if not cfg.providers:
        raise SystemExit(f"--{tier} needs an API provider in ~/.config/spiral/config.json "
                         "(see README: providers). Also export its api_key_env.")
    model = next(iter(cfg.providers))
    targets = (cfg.worker, cfg.planner, cfg.escalation, cfg.critic) if tier == "api" else (cfg.escalation, cfg.critic)
    for spec in targets:
        spec.name = model
    reveal(console, f"  [rgb(217,119,87)]◆ {tier}[/] — {model} on: "
           + (", ".join(sorted({'worker','planner','escalation','critic/validator'})) if tier == "api"
              else "escalation, critic/validator") + "\n")
    return cfg


def _load_goal(args) -> str:
    if getattr(args, "goal_pos", None):
        return args.goal_pos
    if getattr(args, "goal", None):
        return args.goal
    if getattr(args, "goal_file", None):
        return Path(args.goal_file).read_text()
    # no goal given — reuse the one from the last run in this workspace
    import json

    stored = Path(getattr(args, "dir", ".")) / ".spiral" / "plan.json"
    if stored.is_file():
        goal = json.loads(stored.read_text()).get("goal", "")
        if goal:
            make_console().print("  [dim]▸ reusing the goal from the last run (.spiral/plan.json)[/]\n")
            return goal
    raise SystemExit('give spiral a goal:  spiral build "make me a …"  (or --goal-file F)')


def main() -> None:
    parser = argparse.ArgumentParser(prog="spiral", add_help=True)
    sub = parser.add_subparsers(dest="cmd")

    do = sub.add_parser("do", help="drive a single task to green")
    do.add_argument("goal")
    do.add_argument("--verify", required=True, help="command that must exit 0")
    do.add_argument("--dir", default=".")
    do.add_argument("--file", action="append", help="relevant file (repeatable)")

    for name, helptext in (
        ("plan", "show the conductor's decomposition"),
        ("build", "plan, execute, validate — the full autonomous run"),
        ("validate", "judge the current code against the goal's spec (read-only)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("goal_pos", nargs="?", metavar="goal",
                       help="the goal, right here in quotes — no file needed")
        p.add_argument("--goal")
        p.add_argument("--goal-file")
        p.add_argument("--dir", default=".")
        if name == "build":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--approve", action="store_true",
                           help="show the plan and wait for confirmation before executing")
            p.add_argument("--boost", action="store_true",
                           help="local worker + API model for the reasoning roles (escalation, critic, validator)")
            p.add_argument("--api", action="store_true",
                           help="run the entire crew on the configured API model")

    srch = sub.add_parser("search", help="fast web search — ranked results, no synthesis")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=8)
    srch.add_argument("--sci", action="store_true", help="also include arXiv results")

    res = sub.add_parser("research", help="gather sources across web/arXiv/PubMed and synthesize a cited answer")
    res.add_argument("query")
    res.add_argument("-k", type=int, default=6, help="sources per channel")
    res.add_argument("--deep", action="store_true", help="more sources, follow links, longer thinking synthesis")
    res.add_argument("--sci", action="store_true", help="include arXiv + PubMed")
    res.add_argument("--dir", default=".", help="where to save the report")

    tune = sub.add_parser("tune", help="size context windows to this machine (KV math)")
    tune.add_argument("--apply", action="store_true")
    tune.add_argument("--wired", action="store_true", help="also raise the GPU wired-memory limit (sudo)")

    doc = sub.add_parser("doctor", help="health check: ollama, models, tune, gate, git, disk")
    doc.add_argument("--dir", default=".")

    sub.add_parser("setup", help="first-run: detect Ollama + pull a RAM-matched model crew")

    cons = sub.add_parser("consult", help="ask a big-context API model to review the whole project")
    cons.add_argument("question", nargs="?", default="")
    cons.add_argument("--dir", default=".")

    ch = sub.add_parser("chat", help="talk to the local thinking model (reasoning shown dimmed)")
    ch.add_argument("message", nargs="?", default="", help="optional first message; omit for an empty prompt")
    ch.add_argument("--model", help="override the model (default: the planner/thinking model)")

    sty = sub.add_parser("style", help="set the banner spiral shape: spiral · galaxy · uzumaki")
    sty.add_argument("name", nargs="?", help="omit to preview all three")

    note = sub.add_parser("note", help="record project wisdom the workers will always see")
    note.add_argument("text")
    note.add_argument("--dir", default=".")

    st = sub.add_parser("stats", help="run history from the ledger: tokens, tok/s, outcomes")
    st.add_argument("--dir", default=".")

    dst = sub.add_parser("distill", help="mine the ledger: signature routing table + learned fixes")
    dst.add_argument("--dir", default=".")

    rw = sub.add_parser("rewind", help="list task checkpoints; rewind the spiral branch to one")
    rw.add_argument("n", nargs="?", type=int, help="checkpoint number to rewind to")
    rw.add_argument("--dir", default=".")

    args = parser.parse_args()
    console = make_console()
    print_banner(console)   # shows for every command, holds ~1s, then work follows

    if args.cmd == "tune":
        from spiral.tune import main as tune_main

        raise SystemExit(tune_main())

    if args.cmd == "doctor":
        from spiral.doctor import main as doctor_main

        raise SystemExit(doctor_main(args.dir))

    if args.cmd == "setup":
        from spiral.setup import main as setup_main

        raise SystemExit(setup_main())

    if args.cmd == "consult":
        from spiral.conductor import Conductor

        _info_line(console, args.dir)
        Conductor(workspace=args.dir).consult(args.question)
        return

    if args.cmd == "chat":
        from spiral.chat import chat

        chat(args.message, model=args.model)
        return

    if args.cmd == "style":
        from spiral.banner import STYLES, spiral_braille, _rgb, CLAY as _CL, _current_style
        import json as _json

        if args.name and args.name in STYLES:
            f = Path.home() / ".config" / "spiral" / "config.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            d = _json.loads(f.read_text()) if f.is_file() else {}
            d["style"] = args.name
            f.write_text(_json.dumps(d, indent=2))
            console.print(f"  [green]●[/] banner style set to [bold]{args.name}[/]\n")
        else:
            if args.name:
                console.print(f"  [yellow]unknown style '{args.name}'[/] — choose one of:\n")
            cur = _current_style()
            for st in STYLES:
                console.print(f"  [bold]{st}[/]" + ("  [dim](current)[/]" if st == cur else ""))
                for ln in spiral_braille(cols=10, rows=4, turns=2.2, style=st):
                    console.print(f"    [{_rgb(_CL)}]{ln}[/]")
                console.print()
            console.print("  set with: [bold]spiral style <name>[/]\n")
        return

    if args.cmd == "note":
        from spiral.extras import add_note

        add_note(console, args.dir, args.text)
        return

    if args.cmd == "stats":
        from spiral.extras import show_stats

        show_stats(console, args.dir)
        return

    if args.cmd == "distill":
        from spiral.route import distill as distill_cmd

        _info_line(console, args.dir)
        distill_cmd(console, args.dir)
        return

    if args.cmd == "rewind":
        from spiral.extras import rewind

        rewind(console, args.dir, args.n)
        return

    if args.cmd == "search":
        from spiral.research import search as _search, arxiv as _arxiv

        hits = _search(args.query, k=args.k)
        if args.sci:
            hits += _arxiv(args.query, k=max(4, args.k // 2))
        for i, h in enumerate(hits, 1):
            tag = f"[{h.source}] " if h.source != "web" else ""
            console.print(f"  [bold rgb(217,119,87)]{i:>2}[/]  {tag}{h.title}")
            console.print(f"      [dim]{h.url}[/]")
            if h.snippet:
                console.print(f"      [dim]{h.snippet[:200]}[/]")
        console.print()
        return

    if args.cmd == "research":
        import re as _re
        from rich.panel import Panel
        from spiral.banner import Spinner
        from spiral.research import gather, synthesize

        chans = "web + arXiv + PubMed" if args.sci else "web"
        console.print(f"  [dim]gathering sources ({chans}){' · deep' if args.deep else ''}[/]")
        with Spinner("searching") as sp:
            hits = gather(args.query, k=args.k, sci=args.sci, follow=1 if args.deep else 0,
                          on=lambda u: sp.update(detail=u[:60]))
        if not hits:
            console.print("  [yellow]no usable sources found[/]\n")
            return
        console.print(f"  [green]●[/] {len(hits)} sources")
        with Spinner("synthesizing" + (" · thinking" if args.deep else "")) as sp:
            answer, used, res = synthesize(args.query, hits, deep=args.deep, on=lambda: sp.tick())
        console.print(Panel(answer.strip() or "(no answer)", title="[rgb(217,119,87)]⭷ research[/]",
                            border_style="rgb(217,119,87)", padding=(0, 1)))
        slug = _re.sub(r"[^a-z0-9]+", "-", args.query.lower())[:40].strip("-") or "answer"
        out = Path(args.dir) / f"research-{slug}.md"
        srcs = "\n".join(f"[{i}] {h.title} — {h.url}" for i, h in enumerate(used, 1))
        out.write_text(f"# {args.query}\n\n{answer}\n\n## Sources\n{srcs}\n")
        console.print(f"  [dim]{res.prompt_tokens:,} in / {res.completion_tokens:,} out · saved to {out}[/]\n")
        return

    if args.cmd == "do":
        from spiral.agent import Atom, TaskSpec

        _info_line(console, args.dir,
                   f"  goal   [bold]{args.goal}[/]",
                   f"  verify [dim]{args.verify}[/]\n")
        ok = Atom(workspace=args.dir).run(TaskSpec(args.goal, args.verify, args.file))
        raise SystemExit(0 if ok else 1)

    if args.cmd in ("plan", "build", "validate"):
        from spiral.conductor import Conductor

        goal = _load_goal(args)
        _info_line(console, args.dir)
        cfg = _apply_tier(Config.load(), console,
                          "api" if getattr(args, "api", False) else ("boost" if getattr(args, "boost", False) else None))
        cond = Conductor(workspace=args.dir, cfg=cfg)
        if args.cmd == "plan":
            cond.show_plan(cond.make_plan(goal))
        elif args.cmd == "validate":
            cond.validate_only(goal)
        else:
            cond.build(goal, resume=getattr(args, "resume", False),
                       approve=getattr(args, "approve", False))
        return

    _health(console)


def entry() -> None:
    try:
        main()
    except KeyboardInterrupt:
        make_console().print(
            "\n  [rgb(217,119,87)]⠿ interrupted[/] — green work is committed, banked "
            "checkpoints kept. [dim]Resume with the same command + --resume[/]\n"
        )
        raise SystemExit(130)


if __name__ == "__main__":
    entry()
