"""spiral setup — first-run bootstrap for a bare machine.

Detects Ollama and installed models; if the crew is missing, recommends one
sized to this machine's RAM and (with consent) pulls it and writes the model
config. Never installs system software or downloads models without a yes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from spiral.config import Config
from spiral.llm import Ollama
from spiral.theme import CLAY, make_console

CONFIG_PATH = Path.home() / ".config" / "spiral" / "config.json"

# crews of real, widely-available Ollama models, keyed by the RAM ceiling they fit.
# roles: worker/planner (fast) · escalation/critic (stronger) · janitor (tiny).
# sizes are approximate GB for the download-budget preview.
CREWS = [
    (20, "starter", {
        "worker": "qwen2.5-coder:7b", "escalation": "qwen2.5-coder:7b",
        "critic": "qwen2.5-coder:7b", "janitor": "llama3.2:1b",
    }),
    (40, "standard", {
        "worker": "qwen2.5-coder:14b", "escalation": "qwen2.5-coder:32b",
        "critic": "qwen2.5-coder:32b", "janitor": "llama3.2:1b",
    }),
    (9999, "pro", {
        "worker": "qwen2.5-coder:32b", "escalation": "qwen2.5-coder:32b",
        "critic": "qwen2.5-coder:32b", "janitor": "llama3.2:3b",
    }),
]
SIZE_GB = {
    "qwen2.5-coder:7b": 4.7, "qwen2.5-coder:14b": 9.0, "qwen2.5-coder:32b": 20.0,
    "llama3.2:1b": 1.3, "llama3.2:3b": 2.0,
}


def _ram_gb() -> float:
    try:
        out = subprocess.run("sysctl -n hw.memsize", shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        if out.isdigit():
            return int(out) / 2**30
    except Exception:
        pass
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 16.0


def _installed_models() -> set[str]:
    try:
        out = subprocess.run("ollama list", shell=True, capture_output=True, text=True, timeout=10).stdout
        return {ln.split()[0] for ln in out.splitlines()[1:] if ln.split()}
    except Exception:
        return set()


def pick_crew(ram: float):
    for ceiling, name, crew in CREWS:
        if ram <= ceiling:
            return name, crew
    return CREWS[-1][1], CREWS[-1][2]


def main() -> int:
    c = make_console()
    cfg = Config.load()
    c.print(f"\n[bold {CLAY}]spiral setup[/] — first-run bootstrap\n")

    if shutil.which("ollama") is None:
        c.print("  [red]✗ Ollama is not installed.[/] spiral runs models through it.\n")
        c.print("  Install it, then re-run [bold]spiral setup[/]:")
        c.print("    [dim]brew install ollama[/]        (Homebrew)")
        c.print("    [dim]curl -fsSL https://ollama.com/install.sh | sh[/]   (script)")
        c.print("  Docs: https://ollama.com/download\n")
        return 1

    if not Ollama(cfg.base_url).health():
        c.print("  [yellow]○ Ollama is installed but not running.[/] Start it in another terminal:")
        c.print("    [dim]ollama serve[/]\n  then re-run [bold]spiral setup[/].\n")
        return 1

    have = _installed_models()
    ram = _ram_gb()
    # A tuned/user-selected crew is authoritative. Re-running setup must never
    # silently replace a working newer model stack with the generic RAM preset.
    if CONFIG_PATH.is_file():
        tier = "configured"
        crew = {
            "worker": cfg.worker.name,
            "escalation": cfg.escalation.name,
            "critic": cfg.critic.name,
            "janitor": cfg.janitor.name,
        }
    else:
        tier, crew = pick_crew(ram)
    c.print(f"  detected [bold]{ram:.0f} GB[/] RAM → recommended crew: [bold {CLAY}]{tier}[/]\n")
    for role, model in crew.items():
        present = model in have
        mark = "[green]✓ installed[/]" if present else f"[dim]{SIZE_GB.get(model, '?')} GB download[/]"
        c.print(f"    {role:11s} [bold]{model}[/]  {mark}")

    to_pull = []
    for m in dict.fromkeys(crew.values()):  # unique, order-preserved
        if m not in have:
            to_pull.append(m)
    total = sum(SIZE_GB.get(m, 0) for m in to_pull)

    if not to_pull:
        c.print("\n  [green]● the whole crew is already installed.[/]")
    else:
        c.print(f"\n  [bold]{len(to_pull)} model(s) to download · ~{total:.0f} GB total[/]")
        import sys
        if not sys.stdin.isatty():
            c.print("  [dim](non-interactive — re-run in a terminal to pull)[/]\n")
            return 0
        ans = input(f"  pull the {tier} crew now? [y/N] ").strip().lower()
        if ans != "y":
            c.print("  [dim]skipped — nothing downloaded[/]\n")
            return 0
        for m in to_pull:
            c.print(f"\n  [bold {CLAY}]⇣ pulling {m}[/] …")
            rc = subprocess.run(f"ollama pull {m}", shell=True).returncode
            if rc != 0:
                c.print(f"  [red]✗ failed to pull {m}[/] — check the name / network")
                return 1

    # write the crew into the persistent config (merge, don't clobber)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CONFIG_PATH.is_file():
        try:
            existing = json.loads(CONFIG_PATH.read_text())
        except Exception:
            existing = {}
    models = existing.get("models", {})
    models.update({"worker": crew["worker"], "planner": crew["worker"],
                   "escalation": crew["escalation"], "critic": crew["critic"],
                   "janitor": crew["janitor"]})
    existing["models"] = models
    CONFIG_PATH.write_text(json.dumps(existing, indent=2))
    c.print(f"\n  [green]● crew saved[/] → {CONFIG_PATH}")
    c.print("  next: [bold]spiral tune[/] (size context to your RAM), then [bold]spiral build \"…\"[/]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
