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
from spiral.theme import make_console
from spiral.config import Config
from spiral.llm import Ollama


def _health(console: Console) -> None:
    cfg = Config.load()
    version = Ollama(cfg.base_url).health()
    if version:
        console.print(f"  [green]●[/green] ollama {version}  ·  worker [bold]{cfg.worker.name}[/bold]\n")
    else:
        console.print(f"  [red]●[/red] ollama unreachable at {cfg.base_url}\n")


def _info_line(console: Console, workspace: str) -> None:
    """One useful line instead of echoing `dir .`: workspace · branch · gate."""
    from spiral import tools
    from spiral.conductor import detect_gate

    ws = Path(workspace).resolve()
    parts = [f"[bold]{ws.name}[/]"]
    if (ws / ".git").is_dir():
        branch = tools.run("git branch --show-current", ws).out.strip()
        if branch:
            parts.append(branch)
    parts.append(f"gate: {detect_gate(ws) or 'none'}")
    console.print("  [dim]▸[/] " + " [dim]·[/] ".join(parts) + "\n")


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
    console.print(f"  [rgb(217,119,87)]◆ {tier}[/] — {model} on: "
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

    res = sub.add_parser("research", help="search the web and read top hits (GET-only)")
    res.add_argument("query")
    res.add_argument("-k", type=int, default=3, help="pages to read")

    tune = sub.add_parser("tune", help="size context windows to this machine (KV math)")
    tune.add_argument("--apply", action="store_true")
    tune.add_argument("--wired", action="store_true", help="also raise the GPU wired-memory limit (sudo)")

    doc = sub.add_parser("doctor", help="health check: ollama, models, tune, gate, git, disk")
    doc.add_argument("--dir", default=".")

    sub.add_parser("setup", help="first-run: detect Ollama + pull a RAM-matched model crew")

    cons = sub.add_parser("consult", help="ask a big-context API model to review the whole project")
    cons.add_argument("question", nargs="?", default="")
    cons.add_argument("--dir", default=".")

    sty = sub.add_parser("style", help="set the banner spiral shape: spiral · galaxy · uzumaki")
    sty.add_argument("name", nargs="?", help="omit to preview all three")

    note = sub.add_parser("note", help="record project wisdom the workers will always see")
    note.add_argument("text")
    note.add_argument("--dir", default=".")

    st = sub.add_parser("stats", help="run history from the ledger: tokens, tok/s, outcomes")
    st.add_argument("--dir", default=".")

    rw = sub.add_parser("rewind", help="list task checkpoints; rewind the spiral branch to one")
    rw.add_argument("n", nargs="?", type=int, help="checkpoint number to rewind to")
    rw.add_argument("--dir", default=".")

    args = parser.parse_args()
    console = make_console()

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

        print_banner(console)
        _info_line(console, args.dir)
        Conductor(workspace=args.dir).consult(args.question)
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

    if args.cmd == "rewind":
        from spiral.extras import rewind

        rewind(console, args.dir, args.n)
        return

    if args.cmd == "research":
        from spiral.research import research

        for h in research(args.query, k=args.k):
            console.print(f"[bold rgb(217,119,87)]▸ {h.title}[/]\n  [dim]{h.url}[/]")
            if h.snippet:
                console.print(f"  {h.snippet}")
            if h.text:
                console.print(f"  [dim]{h.text[:500]}…[/]\n")
        return

    if args.cmd == "do":
        from spiral.agent import Atom, TaskSpec

        print_banner(console)
        _info_line(console, args.dir)
        console.print(f"  goal   [bold]{args.goal}[/]")
        console.print(f"  verify [dim]{args.verify}[/]\n")
        ok = Atom(workspace=args.dir).run(TaskSpec(args.goal, args.verify, args.file))
        raise SystemExit(0 if ok else 1)

    if args.cmd in ("plan", "build", "validate"):
        from spiral.conductor import Conductor

        goal = _load_goal(args)
        print_banner(console)
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

    print_banner(console)
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
