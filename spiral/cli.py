"""Entry point.

    spiral                                   banner + ollama health
    spiral do   "<goal>" --verify "<cmd>"    one task → green (the atom)
    spiral plan  --goal-file F [--dir D]      show the conductor's decomposition (cheap)
    spiral build --goal-file F [--dir D]      plan, then execute the whole project
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path

from rich.console import Console

from spiral.banner import print_banner
from spiral.theme import make_console, reveal
from spiral.config import Config, general_api_providers
from spiral.llm import (
    Ollama,
    begin_owned_local_model_run,
    release_owned_local_models,
)


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


def _apply_tier(cfg, console, tier, api_key: str | None = None):
    """Remap roles onto the configured API model. None → local crew (default);
    'boost' → escalation + critic/validator/designer on the API; 'api' → all.

    ``api_key`` (from ``--api KEY``) is injected into the provider's configured
    ``api_key_env`` for THIS process only — never written to any file."""
    if not tier:
        return cfg
    general_providers = general_api_providers(cfg.providers)
    if not general_providers:
        raise SystemExit(f"--{tier} needs an API provider in ~/.config/spiral/config.json "
                         "(planner-only/publication-only academic routes do not count). "
                         "Also export its api_key_env.")
    model = next(iter(general_providers))
    if api_key:
        import os
        key_env = (general_providers.get(model) or {}).get(
            "api_key_env", "OPENAI_API_KEY")
        os.environ[key_env] = api_key
        reveal(console, f"  [dim]◆ api key → ${key_env} (this process only)[/]\n")
    targets = (cfg.worker, cfg.planner, cfg.escalation, cfg.critic) if tier == "api" else (cfg.escalation, cfg.critic)
    for spec in targets:
        spec.name = model
    reveal(console, f"  [rgb(217,119,87)]◆ {tier}[/] — {model} on: "
           + (", ".join(sorted({'worker','planner','escalation','critic/validator'})) if tier == "api"
              else "escalation, critic/validator") + "\n")
    return cfg


def _apply_uncensored(console) -> None:
    """Point the GENERATING seats at the abliterated model for this process only.

    Set as SPIRAL_<ROLE> environment overrides rather than on one Config object:
    every subcommand loads its own Config, several of them deep inside the run
    (style_tool, research_loop), and the env vars are the seam all of them
    already read.

    The JUDGING seats — critic and research_auditor — deliberately keep their
    stock models. A run graded by the model whose checks were removed is a run
    with no grader, and the same-family-rubber-stamp finding that put gemma in
    the critic seat applies twice over here.
    """
    import os

    from spiral.config import Config

    model = Config.load().uncensored_model
    installed = Ollama(Config.load().base_url).models()
    if installed and model not in installed:
        raise SystemExit(
            f"--uncensored wants '{model}', which ollama does not have.\n"
            f"    install it, or point elsewhere with SPIRAL_UNCENSORED_MODEL=<name>."
        )
    # Persist the user's explicit mode choice across Config.load() boundaries.
    # ResearchLoop uses this marker to keep publication prose on the abliterated
    # generating seats instead of silently diverting it through the academic LoRA.
    os.environ["SPIRAL_UNCENSORED_ACTIVE"] = "1"
    for role in ("worker", "planner", "escalation"):
        os.environ[f"SPIRAL_{role.upper()}"] = model
    reveal(console, f"  [rgb(217,119,87)]◆ uncensored[/] — {model} on: "
                    "worker, planner, escalation [dim](judges stay stock)[/]\n")


def _free_foreign_models(console, cfg=None) -> None:
    """Read-only residency preflight; never evict a model this run does not own.

    The historical name is retained for callers, but the destructive sweep is
    intentionally gone.  A foreign resident can make a second large model fail
    with Metal OOM on a 32 GB Mac; stopping with an actionable message preserves
    ownership and is safer than silently unloading another app's model.
    """
    # Unit/offline verification is not allowed to wake or inspect a user's
    # resident models. This guard avoids even a loopback probe.
    if os.environ.get("SPIRAL_OFFLINE_TESTS"):
        return

    cfg = cfg or Config.load()

    def connection_refused(error: BaseException) -> bool:
        """Recognise only a definitive empty auxiliary socket, not ambiguity."""
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ConnectionRefusedError) or (
                getattr(current, "errno", None) in {61, 111}
            ):
                return True
            current = current.__cause__ or current.__context__
        text = str(error).lower()
        return "connection refused" in text and ("errno 61" in text or "errno 111" in text)
    # Inspect every loopback spelling, without mutating any of them. macOS resolves
    # `localhost` to ::1 and 127.0.0.1 to a DIFFERENT socket, and a machine can end
    # up running two ollama servers at once (the menu-bar app starts its own beside
    # a LaunchAgent-managed one). They keep separate memory pools.
    preexisting: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    for url in dict.fromkeys([cfg.base_url,
                              "http://127.0.0.1:11434", "http://localhost:11434"]):
        client = Ollama(url, providers=cfg.providers)
        try:
            try:
                names = client.resident_strict()
            except Exception as exc:
                # A refused auxiliary loopback address proves no Ollama process is
                # listening there. Every other failure is ambiguous and admission
                # must stop rather than pretending the resident set is empty.
                is_auxiliary = url.rstrip("/") != cfg.base_url.rstrip("/")
                if is_auxiliary and connection_refused(exc):
                    names = []
                else:
                    failures.append((url.rstrip("/"),
                                     f"{type(exc).__name__}: {exc}"))
                    names = []
            preexisting.extend((url.rstrip("/"), name) for name in names)
        finally:
            client.close()
    if failures:
        details = "\n".join(
            f"    - {url}: {error}" for url, error in sorted(set(failures))
        )
        raise SystemExit(
            "Spiral could not prove Ollama residency is safe:\n"
            f"{details}\n"
            "No model was unloaded. Restore the endpoint, then retry."
        )
    if preexisting:
        details = "\n".join(
            f"    - {name} at {url}" for url, name in sorted(set(preexisting))
        )
        raise SystemExit(
            "Spiral found model(s) resident before this run began:\n"
            f"{details}\n"
            "Even a configured name may belong to the phone or another app, so Spiral "
            "will not claim or evict it. Finish that work or unload the named model "
            "explicitly, then retry."
        )


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


def _admit_model_command(args, console):
    """Apply one strict local-residency gate before any model-bearing branch."""
    command = getattr(args, "cmd", None)
    model_bearing = command in {"chat", "do", "plan", "build", "validate"}
    if command == "prose":
        model_bearing = bool(
            getattr(args, "rewrite", False)
            or getattr(args, "deep", False)
            or getattr(args, "beef_up", False)
            or getattr(args, "restructure", False)
            or getattr(args, "audit", False)
        )
    if command == "research":
        non_model_view = bool(
            getattr(args, "taste_like", None)
            or getattr(args, "taste_dislike", None)
            or getattr(args, "history", False)
            or getattr(args, "audit", False)
            or getattr(args, "graph", False)
        )
        model_bearing = not non_model_view
    if not model_bearing:
        return None

    cfg = Config.load()
    tier = None
    if command in {"build", "research"}:
        tier = ("api" if getattr(args, "api", None)
                else "boost" if getattr(args, "boost", False) else None)
    elif command == "prose" and getattr(args, "api", None):
        tier = "api"
    cfg = _apply_tier(
        cfg, console, tier, api_key=getattr(args, "api", None),
    )
    _free_foreign_models(console, cfg)
    return cfg


def _load_research_topic(args, console: Console) -> str:
    """Research query loader with first-class resume support."""
    import json

    q = (getattr(args, "query", None) or "").strip()
    if getattr(args, "resume", False) or getattr(args, "refresh", False):
        state_f = Path(args.dir) / "spiral-research" / "state.json"
        if not state_f.is_file():
            raise SystemExit(f"no research state to resume at {state_f}")
        try:
            topic = (json.loads(state_f.read_text()).get("topic") or "").strip()
        except Exception as e:
            raise SystemExit(f"could not read research state at {state_f}: {e}")
        if not topic:
            raise SystemExit(f"research state at {state_f} has no topic")
        if q and q != topic:
            console.print("  [yellow]○[/] ignoring supplied query; resuming topic from state.json")
        return topic
    if not q:
        raise SystemExit('give spiral a research query, or use: spiral research --solve --resume')
    state_f = Path(args.dir) / "spiral-research" / "state.json"
    if state_f.is_file():
        console.print("  [yellow]○[/] starting fresh; existing research state ignored "
                      "([bold]--resume[/] continues it)")
    return q


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
        p.add_argument("--uncensored", action="store_true",
                       help="run the generating seats on the abliterated model "
                            "(judging seats stay stock)")
        if name == "build":
            access = p.add_mutually_exclusive_group()
            access.add_argument(
                "--full-access", action="store_true",
                help="lift the workspace sandbox: the model's shell reaches the WHOLE "
                     "machine (any path, network on) and may modify spiral itself. Only "
                     "disk-wipe/halt/fork-bomb and separately approved Git mutations "
                     "stay blocked.",
            )
            access.add_argument(
                "--workspace-only", action="store_true",
                help="authoritatively confine this run to --dir, overriding any persistent "
                     "full-access or sandbox setting",
            )
        if name == "build":
            p.add_argument("--resume", action="store_true")
            p.add_argument(
                "--reference", action="append", default=[], metavar="PATH",
                help="grant read-only access to one existing project, directory, or "
                     "document as untrusted reference data (repeatable)",
            )
            p.add_argument("--approve", action="store_true",
                           help="show the plan and wait for confirmation before executing")
            p.add_argument("--boost", action="store_true",
                           help="local worker + API model for the reasoning roles (escalation, critic, validator)")
            p.add_argument("--api", metavar="API_KEY", default=None,
                           help="run the entire crew on the configured API model; takes the "
                                "provider API key (e.g. --api \"$MOONSHOT_API_KEY\") — held in "
                                "this process's env only, never stored")
            p.add_argument("--visual-url",
                           help="URL to screenshot for local vision-model UI review")
            p.add_argument("--vision-model",
                           help="Ollama vision model for UI review (default: first configured model with vision)")
            p.add_argument("--no-visual-review", action="store_true",
                           help="disable screenshot + vision UI review for this build")
            p.add_argument("--auto-repos", dest="builder_auto_repos", action="store_true",
                           default=None, help="allow credential-free public GitHub reference clones")
            p.add_argument("--no-auto-repos", dest="builder_auto_repos", action="store_false",
                           help="disable public GitHub reference acquisition for this build")
            p.add_argument(
                "--allow-install-scripts", action="store_true",
                help="allow package lifecycle/source-build scripts (off by default)",
            )
            p.add_argument(
                "--token-budget", type=int, default=None,
                help="override the finite complexity-tier Builder token ceiling",
            )

    srch = sub.add_parser("search", help="fast web search — ranked results, no synthesis")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=8)
    srch.add_argument("--sci", action="store_true", help="also include arXiv results")

    res = sub.add_parser("research", help="gather sources across web/arXiv/bioRxiv/medRxiv/PubMed and synthesize a cited answer")
    res.add_argument("query", nargs="?")
    res.add_argument("-k", type=int, default=6, help="sources per channel")
    res.add_argument("--deep", action="store_true", help="more sources, follow links, longer thinking synthesis")
    res.add_argument("--sci", action="store_true", help="include arXiv + PubMed")
    res.add_argument("--dir", default=".", help="where to save the report")
    res.add_argument("--solve", action="store_true",
                     help="iterative loop: gather a source corpus (physics/math on arXiv; "
                          "neuroscience/biology/medicine on bioRxiv/medRxiv/Europe PMC/PubMed/"
                          "Crossref — chosen automatically per topic), propose CHECKABLE claims, "
                          "verify them (sympy/lean/numeric for quantitative claims, "
                          "corpus evidence-grounding for empirical ones), search prior art, "
                          "repeat until solved or a new open question is found, then write a "
                          "cited LaTeX paper")
    res.add_argument("--resume", action="store_true",
                     help="resume the previous --solve research run from spiral-research/state.json")
    res.add_argument("--refresh", action="store_true",
                     help="resume a completed living paper and reopen its literature/novelty horizon")
    res.add_argument("--rounds", type=int, default=None,
                     help="max research rounds with --solve (default: until solved/exhausted)")
    res.add_argument("--token-budget", type=int, default=None,
                     help="override the finite complexity-tier total model-token ceiling")
    res.add_argument("--api", metavar="API_KEY", default=None,
                     help="run the research reasoning (proposals, novelty critique, reflection, "
                          "write-up) on the API model; takes the provider API key "
                          "(e.g. --api \"$MOONSHOT_API_KEY\") — held in this process's env only, "
                          "never stored; verification stays local & deterministic")
    res.add_argument("--boost", action="store_true",
                     help="API model for the critic/reflection roles, local planner")
    res.add_argument("--uncensored", action="store_true",
                     help="run the generating seats on the abliterated model "
                          "(judging seats stay stock)")
    res.add_argument("--verification", action="store_true",
                     help="force literal verification-note mode; default is novelty/research")
    res.add_argument("--auto-repos", action="store_true",
                     help="allow workbench certificates to clone public GitHub repos into the "
                          "certificate workspace; failed repos are cleaned up")
    res.add_argument("--no-blind-replication", action="store_true",
                     help="disable independent blinded certificate regeneration (strictly on by default)")
    res.add_argument("--no-counterfactuals", action="store_true",
                     help="disable the one-assumption-at-a-time discovery lab")
    res.add_argument("--no-research-git", action="store_true",
                     help="disable the private content-addressed research checkpoint history")
    res.add_argument("--graph", action="store_true",
                     help="render the existing --solve research map to research-graph.html and exit")
    res.add_argument("--history", action="store_true",
                     help="show private research checkpoints without starting a model run")
    res.add_argument("--audit", action="store_true",
                     help="verify obligations, novelty, proof bundle, and living-paper freshness")
    res.add_argument("--taste-like", metavar="ANGLE",
                     help="teach the local taste profile a research angle you value")
    res.add_argument("--taste-dislike", metavar="ANGLE",
                     help="teach the local taste profile an angle you do not value")
    res.add_argument("--author", default="", help="paper author byline for --solve write-up")
    res.add_argument("--association", default="", help="paper affiliation/association for --solve write-up")
    res.add_argument("--refine", nargs="?", const=".", default=None, metavar="DIR",
                     help="refine an existing LaTeX project in DIR (default: here): learn the "
                          "field's literature and style, rebuild the paper in a NEW folder "
                          "(original untouched) with corpus-verified enrichment, and emit a "
                          "submittable PDF plus a blue-edit latexdiff PDF")

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

    pr = sub.add_parser(
        "prose", help="measure writing for AI tells; optionally rewrite until it reads human")
    pr.add_argument("file", help="text, markdown, LaTeX or PDF to measure")
    pr.add_argument("--corpus", metavar="DIR",
                    help="directory of exemplar papers (.tex/.txt/.md/.pdf) to mine a field "
                         "writing template from, and score the file against")
    pr.add_argument("--rewrite", action="store_true",
                    help="have the model rewrite it; the deterministic scorer decides whether "
                         "the rewrite is kept (it must lower the AI-tell score without changing "
                         "numbers, citations or equations)")
    pr.add_argument("--deep", action="store_true",
                    help="implies --rewrite; for article-like documents, research closely "
                         "matching full-text papers and mine their field style first")
    pr.add_argument("--beef-up", action="store_true",
                    help="implies --deep; add bounded corpus-supported academic detail with "
                         "validated sentence-level citations")
    pr.add_argument("--restructure", action="store_true",
                    help="implies --deep; reorder intact sections toward the corpus-mined "
                         "rhetorical arc without regenerating their content")
    pr.add_argument("--audit", action="store_true",
                    help="implies --deep; write a separate corpus-grounded high-level "
                         "paper audit (structure, novelty, evidence, claims, reproducibility)")
    pr.add_argument("--uncensored", action="store_true",
                    help="run the generating seats on the abliterated model "
                         "(judging seats stay stock)")
    pr.add_argument("--field", default="", metavar="NAME",
                    help="override the detected research field used by --deep")
    pr.add_argument("--query", action="append", default=[], metavar="SEARCH",
                    help="override automatic --deep literature queries (repeatable)")
    pr.add_argument("--out", help="where to write the copy (default: alongside, *_spiral; "
                                  "the source path and its aliases are always rejected)")
    pr.add_argument("--rounds", type=int, default=5, help="rewrite attempts (default 5)")
    pr.add_argument("--model", default="", help="override the rewriting model")
    pr.add_argument("--api", metavar="API_KEY", default=None,
                    help="use the configured API model for the rewrite")

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
    if getattr(args, "cmd", None) == "research":   # the one place research branding appears
        print_banner(console, tagline="local autonomous researcher · on-device", research=True)
    else:
        print_banner(console)   # shows for every command, holds ~1s, then work follows

    # Before any dispatch: every subcommand below loads its own Config, and this
    # sets the env overrides they all read.
    if getattr(args, "uncensored", False):
        _apply_uncensored(console)

    # This is the single admission boundary for every branch that can dispatch a
    # model. Read-only/status/help commands never touch Ollama residency.
    model_cfg = _admit_model_command(args, console)
    if model_cfg is not None:
        setattr(args, "_spiral_model_cfg", model_cfg)

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

        chat(args.message, model=args.model, cfg=model_cfg)
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

    if args.cmd == "prose":
        from spiral.style_tool import run_style

        raise SystemExit(run_style(console, args))

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

    if args.cmd == "research" and (getattr(args, "taste_like", None)
                                    or getattr(args, "taste_dislike", None)):
        import json
        from spiral.research_strategy import LocalTasteModel

        rdir = Path(args.dir) / "spiral-research"
        state_path = rdir / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        text_value = getattr(args, "taste_like", None) or getattr(args, "taste_dislike", None)
        profile = LocalTasteModel(rdir, state.get("topic") or text_value)
        outcome = "accepted" if getattr(args, "taste_like", None) else "rejected"
        profile.observe({"question": text_value, "corpus_basis": []}, outcome)
        console.print(
            f"  [green]●[/] local taste updated · [bold]{outcome}[/] · "
            f"[dim]{profile.path}[/]\n")
        return

    if args.cmd == "research" and getattr(args, "history", False):
        from spiral.research_history import ResearchGit

        rdir = Path(args.dir) / "spiral-research"
        rows = ResearchGit(rdir).log(limit=40)
        if not rows:
            raise SystemExit(f"no research checkpoints found at {rdir / '.research-git'}")
        for row in rows:
            parents = ",".join(parent[:10] for parent in row.get("parents", [])) or "root"
            console.print(
                f"  [bold rgb(217,119,87)]{row['commit'][:10]}[/] "
                f"[dim]{row.get('date', '')} · parent {parents}[/]  {row.get('subject', '')}")
        console.print()
        return

    if args.cmd == "research" and getattr(args, "audit", False):
        import json
        from spiral.epistemic import ObligationGraph
        from spiral.research_history import ResearchGit
        from spiral.research_provenance import (
            LivingPaper, NoveltyBoundaryCertificate, ProofCarryingPaper,
        )

        rdir = Path(args.dir) / "spiral-research"
        state_path = rdir / "state.json"
        if not state_path.is_file():
            raise SystemExit(f"no research state found at {state_path}")
        state = json.loads(state_path.read_text())
        graph = ObligationGraph(rdir, state.get("topic") or "research")
        result_report = graph.report("result")
        publication_report = graph.report("publication")
        expository = (state.get("active_proposal") or {}).get("mode") == "expository"
        novelty = ({"valid": True, "issues": ["not required for expository mode"]}
                   if expository else
                   NoveltyBoundaryCertificate.validate(rdir / "novelty-boundary.json"))
        proof = ProofCarryingPaper.validate(rdir / "writeup" / "proof-carrying-manifest.json")
        living = LivingPaper.inspect(rdir / "living-paper.json", rdir)
        history = ResearchGit(
            rdir,
            enabled=bool(state.get("research_commit") or (rdir / ".research-git").is_dir()),
        ).verify()
        rows = [
            ("result obligations", result_report.get("ready"),
             f"{result_report.get('required_count', 0)} required · {len(result_report.get('blockers') or [])} blockers"),
            ("publication obligations", publication_report.get("ready"),
             f"{publication_report.get('required_count', 0)} required · {len(publication_report.get('blockers') or [])} blockers"),
            ("event hash chain", (result_report.get("event_chain") or {}).get("valid"),
             f"{(result_report.get('event_chain') or {}).get('entries', 0)} entries"),
            ("novelty boundary", novelty.get("valid"), "; ".join(novelty.get("issues") or ["valid"])),
            ("proof bundle", proof.get("valid"), "; ".join(proof.get("issues") or ["valid"])),
            ("living paper", living.get("current"), "; ".join(living.get("issues") or ["current"])),
            ("research history", history.get("valid"),
             "; ".join(history.get("issues") or [
                 f"{history.get('commit_count', 0)} immutable checkpoints"])),
        ]
        for label, ok, detail in rows:
            mark = "[green]●[/]" if ok else "[yellow]○[/]"
            console.print(f"  {mark} {label:24s} [dim]{detail}[/]")
        console.print(f"\n  [dim]obligations: {graph.markdown_path}[/]\n")
        if not all(bool(ok) for _, ok, _ in rows):
            raise SystemExit(1)
        return

    if args.cmd == "research" and getattr(args, "graph", False):
        from spiral.research_graph import write_graph_view_from_files
        rdir = Path(args.dir) / "spiral-research"
        try:
            view = write_graph_view_from_files(rdir)
        except FileNotFoundError:
            raise SystemExit(f"no research map found at {rdir / 'research-map.json'}")
        console.print(f"  [green]●[/] graph: [dim]{view['html']}[/] "
                      f"({view['nodes']} nodes · {view['edges']} edges)")
        return

    if args.cmd == "research" and getattr(args, "refine", None):
        from spiral.research_refine import RefineError, RefineRun
        cfg = model_cfg
        target = Path(args.refine).resolve()
        console.print(f"  [dim]refine · {target}[/]")
        run = RefineRun(target, cfg=cfg,
                        on=lambda m: console.print(f"  [dim]▸ {m}[/]"))
        try:
            art = run.run()
        except RefineError as e:
            raise SystemExit(f"refine: {e}")
        except KeyboardInterrupt:
            console.print("  [yellow]interrupted[/] — partial artifacts in "
                          f"{run.out}")
            raise SystemExit(130)
        console.print(f"  [green]●[/] refined tex: [dim]{art['tex']}[/]")
        if art["pdf"]:
            console.print(f"  [green]●[/] submittable pdf: [dim]{art['pdf']}[/]")
        else:
            console.print("  [yellow]○[/] PDF did not compile — main.tex retained"
                          + (f" [dim]{art['compile_log'][:160]}[/]" if art.get("compile_log") else ""))
        if art["diff_pdf"]:
            console.print(f"  [green]●[/] blue-edit diff pdf: [dim]{art['diff_pdf']}[/]")
        console.print(f"  [dim]suggestions: {art['suggestions']}[/]")
        console.print(f"  [dim]report: {art['report']}[/]\n")
        return

    if args.cmd == "research" and getattr(args, "solve", False):
        from spiral.research_loop import ResearchLoop, ResearchModelError
        from spiral.research_ui import ResearchProgress
        query = _load_research_topic(args, console)
        label = ("refresh · " if getattr(args, "refresh", False)
                 else "resume · " if getattr(args, "resume", False) else "")
        console.print(f"  [dim]{label}{query[:100]}{'…' if len(query) > 100 else ''}[/]")
        cfg = model_cfg
        if getattr(args, "auto_repos", False):
            cfg.research_repo_auto = True
        if getattr(args, "no_blind_replication", False):
            cfg.research_blind_replication = False
        if getattr(args, "no_counterfactuals", False):
            cfg.research_counterfactuals = False
        if getattr(args, "no_research_git", False):
            cfg.research_git = False
        loop = None
        research_dir = Path(args.dir) / "spiral-research"
        with ResearchProgress(console, workdir=research_dir) as progress:
            loop = ResearchLoop(query, workdir=research_dir, cfg=cfg,
                                ui=progress, resume=(getattr(args, "resume", False)
                                                     or getattr(args, "refresh", False)),
                                mode="expository" if getattr(args, "verification", False) else None,
                                refresh=getattr(args, "refresh", False))
            try:
                state = loop.run(max_rounds=args.rounds, token_budget=args.token_budget)
            except KeyboardInterrupt:
                try:
                    loop._save()
                    loop.corpus.save()
                    loop._save_map()
                except Exception:
                    pass
                progress.dash.print("  [yellow]interrupted[/] — research state saved; resume with "
                                    "[bold]spiral research --solve --resume[/]")
                progress.dash.print(f"  [dim]state: {loop.dir / 'state.json'}[/]\n")
                raise SystemExit(130)
            except ResearchModelError as e:
                progress.dash.print(f"  [red]model error:[/] {e}")
                progress.dash.print(f"  [dim]journal/map so far: {loop.dir}[/]\n")
                raise SystemExit(2)
            n_ok = sum(1 for f in state.findings if f.get("ok"))
            n_qual = sum(1 for f in state.findings
                         if f.get("ok") and f.get("strength") in {"formal", "exact", "computational"})
            progress.dash.print(f"  [green]●[/] status [bold]{state.status}[/] · {state.round} rounds · "
                                f"{n_qual} qualifying findings ({n_ok} successful runs) · {state.tokens:,} tok")
            map_md = loop.dir / "research-map.md"
            progress.dash.print(f"  [dim]map: {map_md}[/]")
            graph_html = loop.dir / "research-graph.html"
            if graph_html.is_file():
                progress.dash.print(f"  [dim]graph: {graph_html}[/]")
            progress.dash.print(f"  [dim]obligations: {loop.obligations.markdown_path}[/]")
            if not state.completion.get("ready"):
                failed = [name for name, ok in (state.completion.get("checks") or {}).items() if not ok]
                progress.dash.print("  [yellow]paper skipped: completion gate has not passed[/]"
                                    + (f" [dim]({', '.join(failed[:4])})[/]" if failed else "") + "\n")
                return
            try:
                art = loop.write(author=args.author, association=args.association)
            except KeyboardInterrupt:
                progress.dash.print(
                    "  [yellow]interrupted during writing[/] — drafts and audits retained; "
                    "resume with [bold]spiral research --solve --resume[/]")
                progress.dash.print(f"  [dim]write-up: {loop.dir / 'writeup'}[/]\n")
                raise SystemExit(130)
            except (ResearchModelError, RuntimeError) as e:
                progress.dash.print(f"  [red]paper gate stopped:[/] {e}")
                progress.dash.print(
                    f"  [dim]draft/audits retained: {loop.dir / 'writeup'}[/]")
                progress.dash.print(
                    "  [dim]resume retries writing without repeating solved research: "
                    "spiral research --solve --resume[/]\n")
                raise SystemExit(3)
            tail = f" · pdf {art['pdf']}" if art.get("pdf") else " (no LaTeX toolchain — .tex written)"
            progress.dash.print(f"  [dim]write-up: {art['tex']}{tail}[/]\n")
            return

    if args.cmd == "research":
        if getattr(args, "resume", False) or getattr(args, "refresh", False):
            raise SystemExit("research --resume is only available with --solve")
        query = _load_research_topic(args, console)
        import re as _re
        from rich.panel import Panel
        from spiral.banner import Spinner
        from spiral.research import gather, synthesize

        chans = "web + arXiv + PubMed" if args.sci else "web"
        console.print(f"  [dim]gathering sources ({chans}){' · deep' if args.deep else ''}[/]")
        with Spinner("searching") as sp:
            hits = gather(query, k=args.k, sci=args.sci, follow=1 if args.deep else 0,
                          on=lambda u: sp.update(detail=u[:60]))
        if not hits:
            console.print("  [yellow]no usable sources found[/]\n")
            return
        console.print(f"  [green]●[/] {len(hits)} sources")
        with Spinner("synthesizing" + (" · thinking" if args.deep else "")) as sp:
            answer, used, res = synthesize(
                query, hits, cfg=model_cfg, deep=args.deep, on=lambda: sp.tick(),
            )
        console.print(Panel(answer.strip() or "(no answer)", title="[rgb(217,119,87)]⭷ research[/]",
                            border_style="rgb(217,119,87)", padding=(0, 1)))
        slug = _re.sub(r"[^a-z0-9]+", "-", query.lower())[:40].strip("-") or "answer"
        out = Path(args.dir) / f"research-{slug}.md"
        srcs = "\n".join(f"[{i}] {h.title} — {h.url}" for i, h in enumerate(used, 1))
        out.write_text(f"# {query}\n\n{answer}\n\n## Sources\n{srcs}\n")
        console.print(f"  [dim]{res.prompt_tokens:,} in / {res.completion_tokens:,} out · saved to {out}[/]\n")
        return

    if args.cmd == "do":
        from spiral.agent import Atom, TaskSpec

        _info_line(console, args.dir,
                   f"  goal   [bold]{args.goal}[/]",
                   f"  verify [dim]{args.verify}[/]\n")
        ok = Atom(workspace=args.dir, cfg=model_cfg).run(
            TaskSpec(args.goal, args.verify, args.file),
        )
        raise SystemExit(0 if ok else 1)

    if args.cmd in ("plan", "build", "validate"):
        from spiral.conductor import Conductor

        goal = _load_goal(args)
        _info_line(console, args.dir)
        cfg = model_cfg
        if os.environ.get("SPIRALCHAT_EXTERNAL_GIT_APPROVAL") == "1":
            # Clone/fetch/pull/commit/push are controller capabilities in a
            # SpiralChat-managed run. The model may inspect Git, but it cannot
            # acquire or mutate a repository through Builder's automatic lane.
            cfg.builder_repo_auto = False
        if getattr(args, "visual_url", None):
            cfg.visual_review_url = args.visual_url
        if getattr(args, "vision_model", None):
            cfg.vision_model = args.vision_model
        if getattr(args, "no_visual_review", False):
            cfg.visual_review = False
        if getattr(args, "workspace_only", False):
            # A controller-approved project-only run must win over persistent user
            # configuration. Merely omitting --full-access is not sufficient because
            # Config.load() may have read builder_full_access=true or disabled the
            # mandatory sandbox in ~/.config/spiral/config.json.
            cfg.builder_full_access = False
            cfg.builder_require_sandbox = True
            cfg.builder_allow_install_scripts = False
            cfg.builder_tool_auto = False
        elif getattr(args, "full_access", False):
            cfg.builder_full_access = True
            reveal(console,
                   "  [rgb(217,119,87)]◆ full access[/] — the model's shell reaches the whole "
                   "machine: any path, network on, spiral's own source included.\n"
                   "  [dim]disk-wipe · halt · fork-bomb · separately approved Git "
                   "mutations stay blocked. "
                   "runs unsandboxed by design.[/]\n")
        if getattr(args, "builder_auto_repos", None) is not None:
            cfg.builder_repo_auto = bool(args.builder_auto_repos)
        if os.environ.get("SPIRALCHAT_EXTERNAL_GIT_APPROVAL") == "1":
            cfg.builder_repo_auto = False
        if args.cmd == "build":
            from spiral.command_broker import (
                canonical_reference_roots, require_reference_identities,
            )

            try:
                cfg.builder_reference_roots = canonical_reference_roots(
                    getattr(args, "reference", []) or [], workspace=args.dir,
                )
                encoded_identities = os.environ.get(
                    "SPIRALCHAT_REFERENCE_IDENTITIES")
                if encoded_identities is not None:
                    require_reference_identities(
                        cfg.builder_reference_roots, encoded_identities)
            except ValueError as exc:
                raise SystemExit(f"invalid --reference: {exc}") from exc
        if getattr(args, "allow_install_scripts", False):
            if getattr(args, "workspace_only", False):
                raise SystemExit("--allow-install-scripts requires --full-access")
            cfg.builder_allow_install_scripts = True
        if getattr(args, "token_budget", None) is not None:
            cfg.builder_token_budget = max(1, int(args.token_budget))
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
    from spiral.harness_check import HarnessFault

    # The host stops detached jobs with SIGTERM. Translate that request into a
    # normal Python unwind so the same exact-owned-model cleanup used for Ctrl-C
    # runs before process exit. SIGKILL remains intentionally uncatchable.
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def terminate(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
    begin_owned_local_model_run()
    try:
        try:
            main()
        except HarnessFault as fault:
            # Not a build failure. The gate could not measure anything, so stopping and
            # saying why is the honest outcome — reporting the code as red would be a lie
            # about work that was never tested.
            make_console().print(
                f"\n  [red]■ stopped: the build gate is not usable[/]\n\n{fault}\n\n"
                "  [dim]Nothing here is a verdict on your code. Fix the gate or its "
                "environment, then resume with the same command + --resume[/]\n"
            )
            raise SystemExit(3)
        except KeyboardInterrupt:
            make_console().print(
                "\n  [rgb(217,119,87)]⠿ interrupted[/] — green work is committed, banked "
                "checkpoints kept. [dim]Resume with the same command + --resume[/]\n"
            )
            raise SystemExit(130)
    finally:
        try:
            release_owned_local_models()
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    entry()
