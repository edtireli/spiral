"""spiral doctor — one-stop health check: is this machine ready to build?

    ● ok   ○ warning   ✗ broken
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from spiral.config import Config
from spiral.llm import Ollama
from spiral.theme import CLAY, make_console


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main(workspace: str = ".") -> int:
    c = make_console()
    cfg = Config.load()
    ws = Path(workspace).resolve()
    c.print(f"\n[bold {CLAY}]spiral doctor[/]\n")
    problems = 0

    def row(ok: bool | None, label: str, detail: str) -> None:
        nonlocal problems
        mark = "[green]●[/]" if ok else ("[yellow]○[/]" if ok is None else "[red]✗[/]")
        if ok is False:
            problems += 1
        c.print(f"  {mark} {label:16s} [dim]{detail}[/]")

    version = Ollama(cfg.base_url).health()
    row(bool(version), "ollama", f"v{version} at {cfg.base_url}" if version else f"unreachable at {cfg.base_url}")

    installed = {ln.split()[0] for ln in _sh("ollama list").splitlines()[1:] if ln.split()}
    for role, spec in (("worker", cfg.worker), ("escalation", cfg.escalation),
                       ("critic", cfg.critic), ("janitor", cfg.janitor)):
        row(spec.name in installed, f"model:{role}", f"{spec.name} · ctx {spec.num_ctx:,}"
            + ("" if spec.name in installed else " — NOT INSTALLED (ollama pull it)"))

    tuned = (Path.home() / ".config" / "spiral" / "config.json").is_file()
    kv = _sh("launchctl getenv OLLAMA_KV_CACHE_TYPE")
    row(True if (tuned and kv) else None, "tune",
        f"calibrated · KV {kv}" if (tuned and kv) else "untuned — run `spiral tune` (models may page)")

    from spiral.conductor import detect_gate
    gate = detect_gate(ws)
    row(bool(gate) or None, "gate", gate or "none detected in this directory — runs would be unverified")

    row((ws / ".git").is_dir() or None, "git", "repo" if (ws / ".git").is_dir() else "not a repo — spiral will init one")

    free_gb = shutil.disk_usage(ws).free / 2**30
    row(free_gb > 20, "disk", f"{free_gb:.0f} GB free")

    from spiral.skillpack import load_skills
    skills = load_skills(ws)
    row(bool(skills), "skills", ", ".join(s.name for s in skills) or "none")

    state = ws / ".spiral" / "state.json"
    if state.is_file():
        import json
        s = json.loads(state.read_text())
        row(True, "last run", f"{s.get('outcome', '?')} · {s.get('tokens', 0):,} tok · {s.get('ts', '')}")

    c.print()
    if any(spec.name not in installed for spec in (cfg.worker, cfg.escalation)):
        c.print("  [yellow]→ missing models?[/] run [bold]spiral setup[/] to pull a RAM-matched crew\n")
    if problems:
        c.print(f"  [red]{problems} problem(s) need attention[/]\n")
        return 1
    c.print("  [green]ready to build[/]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
