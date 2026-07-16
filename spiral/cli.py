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

    res = sub.add_parser("research", help="search the web and read top hits (GET-only)")
    res.add_argument("query")
    res.add_argument("-k", type=int, default=3, help="pages to read")

    tune = sub.add_parser("tune", help="size context windows to this machine (KV math)")
    tune.add_argument("--apply", action="store_true")
    tune.add_argument("--wired", action="store_true", help="also raise the GPU wired-memory limit (sudo)")

    args = parser.parse_args()
    console = make_console()

    if args.cmd == "tune":
        from spiral.tune import main as tune_main

        raise SystemExit(tune_main())

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
        cond = Conductor(workspace=args.dir)
        if args.cmd == "plan":
            cond.show_plan(cond.make_plan(goal))
        elif args.cmd == "validate":
            cond.validate_only(goal)
        else:
            cond.build(goal, resume=getattr(args, "resume", False))
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
