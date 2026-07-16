"""spiral tune — measure the machine, size the context windows to fit it.

Context is KV-cache RAM. This command reads what the machine actually has
(unified RAM, Ollama's KV quantization env, each model's weight size), computes
the affordable num_ctx per model, and writes it to ~/.config/spiral/config.json
(read by Config.load at startup). Dry-run by default; --apply also sets the
Ollama env (flash attention + q8_0 KV = HALF the KV RAM, ~zero quality loss).

The math: budget = usable_ram − weights − headroom;  num_ctx = budget / kv_per_token.
kv_per_token comes from a measured catalog (estimates for unknown models), scaled
by the KV cache type (f16 1.0 · q8_0 0.5 · q4_0 0.25).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from spiral.config import Config
from spiral.theme import CLAY, make_console

# measured/estimated f16 KV bytes per token (both K and V, all layers)
KV_CATALOG = {
    "qwen3.6:latest": 100_000,   # 36B-A3B MoE
    "qwen3:30b-a3b": 98_000,
    "qwen3.6:27b": 245_000,      # dense
}
KV_DEFAULT = 160_000
QFACTOR = {"f16": 1.0, "": 1.0, "q8_0": 0.5, "q4_0": 0.25}
HEADROOM_GB = 6.0     # non-wired RAM macOS+gradle keep (outside the GPU budget)
GPU_OVERHEAD_GB = 1.5  # compute buffers etc. inside the wired budget
CONFIG_PATH = Path.home() / ".config" / "spiral" / "config.json"


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def ram_gb() -> float:
    out = _sh("sysctl -n hw.memsize")
    return int(out) / 2**30 if out.isdigit() else 32.0


def kv_type() -> str:
    return _sh("launchctl getenv OLLAMA_KV_CACHE_TYPE") or ""


def flash_on() -> bool:
    return _sh("launchctl getenv OLLAMA_FLASH_ATTENTION") == "1"


def wired_limit_gb() -> float:
    out = _sh("sysctl -n iogpu.wired_limit_mb")
    if out.isdigit() and int(out) > 0:
        return int(out) / 1024
    return ram_gb() * 0.70  # macOS default GPU wired cap ≈ 70%


def weights_gb(model: str) -> float:
    for line in _sh("ollama list").splitlines():
        if line.split()[0:1] == [model]:
            for tok_ in line.split():
                if tok_.replace(".", "").isdigit() and "GB" in line:
                    pass
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "GB":
                    return float(parts[i - 1])
    return 20.0


def native_ctx(model: str) -> int:
    for line in _sh(f"ollama show {model}").splitlines():
        if "context length" in line:
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                return int(digits)
    return 131072


def calibrate(model: str, ktype: str, wired_gb: float | None = None) -> dict:
    kv_tok = KV_CATALOG.get(model, KV_DEFAULT) * QFACTOR.get(ktype, 1.0)
    budget_gb = (wired_gb or wired_limit_gb()) - weights_gb(model) - GPU_OVERHEAD_GB
    raw = int(budget_gb * 1e9 / kv_tok) if budget_gb > 0 else 8192
    ctx = max(8192, min(raw, native_ctx(model)))
    ctx = (ctx // 4096) * 4096  # round down to 4k multiple
    return {
        "model": model, "weights_gb": weights_gb(model),
        "kv_per_tok_kb": round(kv_tok / 1000, 1), "budget_gb": round(budget_gb, 1),
        "num_ctx": ctx,
    }


def main() -> int:
    apply = "--apply" in sys.argv
    wired = "--wired" in sys.argv
    c = make_console()
    cfg = Config.load()
    ktype = kv_type()
    target_ktype = ktype or "q8_0"

    c.print(f"\n[bold {CLAY}]spiral tune[/] — context windows sized to this machine\n")
    c.print(f"  unified RAM      {ram_gb():.0f} GB · GPU wired limit ≈ {wired_limit_gb():.0f} GB")
    c.print(f"  flash attention  {'on' if flash_on() else '[yellow]off[/]'}")
    c.print(f"  KV cache type    {ktype or '[yellow]f16 (default — 2x the RAM of q8_0)[/]'}\n")

    wired_now = wired_limit_gb()
    wired_target = ram_gb() - HEADROOM_GB if wired else wired_now
    models = sorted({cfg.worker.name, cfg.escalation.name, cfg.critic.name, cfg.planner.name})
    plans = [calibrate(m, target_ktype, wired_target) for m in models]
    for p in plans:
        c.print(f"  [bold]{p['model']}[/]")
        c.print(f"    weights {p['weights_gb']:.0f} GB · KV {p['kv_per_tok_kb']} KB/tok ({target_ktype}) "
                f"· budget {p['budget_gb']} GB → [bold {CLAY}]num_ctx {p['num_ctx']:,}[/]")
    if wired and wired_target > wired_now:
        c.print(f"  [dim](assumes --wired raises the GPU limit {wired_now:.0f} → {wired_target:.0f} GB)[/]")

    # ---- advisor: recommend, then ask ----------------------------------------
    if not apply and sys.stdin.isatty():
        tight = any(calibrate(m, target_ktype, wired_now)["budget_gb"] < 2.0 for m in models)
        rec = "2" if tight else "1"
        why = ("a model is at/over the GPU wired limit — raising it is the SAFE move "
               "(reverts on reboot, 6 GB stays reserved for macOS)" if tight
               else "context budgets are healthy; KV quantization alone doubles them")
        c.print(f"\n  [bold {CLAY}]advisor:[/] option {rec} — {why}\n")
        c.print("  [1] q8_0 KV + flash attention           (env only)")
        c.print("  [2] option 1 + raise GPU wired limit    (adds sudo sysctl, reverts on reboot)")
        c.print("  [n] do nothing\n")
        choice = input("  apply which? [1/2/n] ").strip().lower()
        if choice == "1":
            apply, wired = True, False
        elif choice == "2":
            apply, wired = True, True
        else:
            c.print("  [dim]nothing changed[/]\n")
            return 0
        # recompute the plans for what was actually chosen
        wired_target = ram_gb() - HEADROOM_GB if wired else wired_now
        plans = [calibrate(m, target_ktype, wired_target) for m in models]
    elif not apply:
        c.print(f"\n  [dim]dry run — nothing changed. Apply with:[/] spiral tune --apply"
                f" [dim](+ --wired to raise the GPU limit)[/]\n")
        return 0

    # ---- apply: ollama env + spiral config overlay ---------------------------
    _sh("launchctl setenv OLLAMA_FLASH_ATTENTION 1")
    _sh(f"launchctl setenv OLLAMA_KV_CACHE_TYPE {target_ktype}")
    c.print(f"\n  [green]●[/] ollama env set: flash attention + KV {target_ktype}")

    if wired:
        target_mb = int((ram_gb() - HEADROOM_GB) * 1024)
        c.print(f"  [yellow]sudo needed[/] to raise GPU wired limit to {target_mb} MB (reverts on reboot):")
        subprocess.run(f"sudo sysctl iogpu.wired_limit_mb={target_mb}", shell=True)

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CONFIG_PATH.is_file():
        try:
            existing = json.loads(CONFIG_PATH.read_text())
        except Exception:
            existing = {}
    existing.update({"num_ctx": {p["model"]: p["num_ctx"] for p in plans}, "kv_type": target_ktype})
    CONFIG_PATH.write_text(json.dumps(existing, indent=2))
    c.print(f"  [green]●[/] wrote {CONFIG_PATH}")
    c.print("\n  [bold]restart ollama to activate:[/] pkill ollama && ollama serve &")
    c.print("  [dim](do this between runs — a restart unloads models mid-flight)[/]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
