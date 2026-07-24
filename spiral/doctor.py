"""spiral doctor — one-stop health check: is this machine ready to build?

    ● ok   ○ warning   ✗ broken
"""
from __future__ import annotations

import shutil
import subprocess
import os
import sys
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
                       ("critic", cfg.critic),
                       ("research auditor", cfg.research_auditor),
                       ("janitor", cfg.janitor)):
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

    node = shutil.which("node")
    npm = shutil.which("npm")
    row(True if node and npm else None, "builder:node",
        f"{_sh('node --version')} · npm {_sh('npm --version')}" if node and npm
        else "node/npm unavailable — JavaScript products cannot be provisioned")

    browser_cache = Path.home() / ".cache" / "spiral" / "playwright"
    chromium = ""
    old_browser_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    try:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as runtime:
            candidate = Path(runtime.chromium.executable_path)
            chromium = str(candidate) if candidate.is_file() else ""
    except Exception:
        chromium = ""
    finally:
        if old_browser_path is None:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old_browser_path
    if not chromium:
        chromium = next((
            str(path) for path in (
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                Path("/usr/bin/google-chrome"), Path("/usr/bin/chromium"),
                Path("/usr/bin/chromium-browser"),
            ) if path.is_file()
        ), "")
    row(True if chromium else None, "builder:browser",
        chromium or "Chromium not cached — first UI build installs it automatically")

    try:
        from spiral.visual_review import choose_vision_model

        vision = choose_vision_model(cfg, Ollama(cfg.base_url))
    except Exception:
        vision = ""
    row(True if vision else None, "builder:vision",
        vision or "no installed model reports vision capability")

    from spiral.verify_math import _lean_exe, lean_available

    lean = _lean_exe()
    lean_version = ""
    if lean and lean_available():
        lean_version = _sh(f'"{lean}" --version').splitlines()[0]
    row(True if lean_version else None, "research:lean",
        lean_version or "not available — theorem claims fall back to weaker backends")

    singular = shutil.which("Singular") or shutil.which("singular")
    for candidate in (
        "/opt/homebrew/opt/singular/bin/Singular",
        "/usr/local/opt/singular/bin/Singular",
    ):
        if not singular and Path(candidate).is_file():
            singular = candidate
    singular_version = _sh(f'"{singular}" --version').splitlines()[0] if singular else ""
    row(True if singular_version else None, "research:CAS",
        singular_version or "Singular unavailable — primary decomposition is limited")

    latex = shutil.which("latexmk") or shutil.which("pdflatex")
    row(True if latex else None, "research:LaTeX",
        str(latex or "no latexmk/pdflatex — PDF compilation unavailable"))

    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    row(True if sandbox else None, "research:sandbox",
        "offline, host-readable, workdir-write-only" if sandbox
        else "no supported OS sandbox; workbench manifest will warn")

    try:
        from spiral.toolsmith import Toolsmith

        profile = Toolsmith(ws)
        capabilities = profile.scan()
        available_count = sum(1 for value in capabilities.values() if value.get("available"))
        attempts = sum(int(value.get("attempts", 0)) for value in profile.state.get("tools", {}).values())
        recipes = len(profile.state.get("recipes", []))
        row(True if available_count else None, "toolsmith",
            f"{available_count}/{len(capabilities)} toolchains · {attempts} observed runs · {recipes} reusable recipes")
    except Exception as exc:
        row(None, "toolsmith", f"profile unavailable: {exc}")

    strict = all((
        cfg.research_obligation_graph,
        cfg.research_blind_replication,
        cfg.research_information_scheduler,
        cfg.research_git,
        cfg.research_living_papers,
    ))
    row(True if strict else None, "research:kernel",
        "obligations + scheduling + blind replication + history + living papers"
        if strict else "one or more strict epistemic-kernel features disabled")

    for provider, settings in cfg.providers.items():
        key_env = str(settings.get("api_key_env") or "")
        present = bool(key_env and os.environ.get(key_env))
        row(True if present else None, f"api:{provider}"[:16],
            f"{key_env} is set" if present else f"configured; {key_env or 'API key env'} is not set")

    from spiral.skillpack import load_skills
    skills = load_skills(ws)
    row(bool(skills), "skills", ", ".join(s.name for s in skills) or "none")

    state = ws / ".spiral" / "state.json"
    if state.is_file():
        import json
        s = json.loads(state.read_text())
        row(True, "last run", f"{s.get('outcome', '?')} · {s.get('tokens', 0):,} tok · {s.get('ts', '')}")

    c.print()
    if any(spec.name not in installed
           for spec in (cfg.worker, cfg.escalation, cfg.research_auditor)):
        c.print("  [yellow]→ missing models?[/] run [bold]spiral setup[/] to pull a RAM-matched crew\n")
    if problems:
        c.print(f"  [red]{problems} problem(s) need attention[/]\n")
        return 1
    c.print("  [green]ready to build[/]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
