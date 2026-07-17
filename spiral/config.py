"""Configuration — backend is a swappable seam; local-first defaults.

Model strategy (32 GB unified memory, hardware-honest):
  - ONE resident model, qwen3:30b-a3b (MoE, 3B active): conductor duties with
    thinking ON, worker duties with thinking OFF. Same weights → planning,
    reflection, and re-planning cost zero model swaps, and the worker runs 3-4x
    faster than a dense 27B — wall-clock is set by worker turns.
  - qwen3.6:27b (dense) is the ESCALATION model: swapped in only when a task
    stalls, where slower-but-smarter earns its load time.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    num_ctx: int = 16384
    think: bool = False


@dataclass
class Config:
    # backend seam — local-first. "ollama" today; another provider could slot in.
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"

    # Conductor/worker: qwen3.6:latest = the 3.6-gen 36B MoE (A3B class) — two
    # generations newer than qwen3:30b-a3b at the same ~3B-active speed. RAM:
    # 23GB weights + ~2.5GB KV @24k → ~25.5GB footprint; num_ctx trimmed from
    # 32k to keep headroom for gradle + macOS on 32GB. If the first run swaps,
    # fall back to qwen3:30b-a3b here.
    planner: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:latest", num_ctx=24576, think=True)
    )
    worker: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:latest", num_ctx=24576, think=False)
    )
    escalation: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:27b", num_ctx=16384, think=False)
    )
    # plan critic: the DENSE model, thinking, but emitting only a short defect
    # list — model diversity catches what self-review can't, and the small output
    # neutralizes its think-forever risk
    critic: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:27b", num_ctx=16384, think=True)
    )

    def spec_for(self, model_name: str) -> ModelSpec:
        """The ModelSpec whose name matches — so per-model num_ctx follows the
        model wherever it's used (worker vs escalation lanes)."""
        for spec in (self.worker, self.escalation, self.planner, self.critic, self.janitor):
            if spec.name == model_name:
                return spec
        return self.worker
    # optional janitor for later phases (compaction / done-checks)
    janitor: ModelSpec = field(
        default_factory=lambda: ModelSpec("llama3.2:1b", num_ctx=8192, think=False)
    )

    # model residency: how long Ollama keeps a model loaded after a request.
    # Without this, the 5-min idle default unloads mid-gradle-verify and every
    # attempt pays a 30-60s reload.
    keep_alive: str = "45m"

    # budgets — the guardrails that keep an autonomous run bounded
    # output cap per reply (num_predict) — NOT the context window (that's num_ctx,
    # set by `tune`). A ceiling, not a target: a short reply still ends early, so
    # a higher cap only rescues replies that would otherwise truncate mid-block.
    worker_max_tokens: int = 8192
    planner_max_tokens: int = 16384    # thinking + a whole-app plan; thinking alone can eat 8k
    task_attempt_budget: int = 6       # edit→verify cycles before escalation
    escalation_attempts: int = 4       # extra cycles on the stronger model
    bootstrap_attempts: int = 12       # first-green repair gets a longer leash
    plan_rounds: int = 2               # lint→critic→repair cycles before execution
    validate_rounds: int = 4           # max validate→remediate cycles; stops early on a plateau
    run_token_budget: int = 4_000_000  # global ceiling for a whole run
    verify_timeout: int = 900          # seconds; real build gates (gradle) are slow

    # user-defined extra gate welded into every task's verify (your own linter,
    # tests, anything) — set "extra_gate" in ~/.config/spiral/config.json
    extra_gate: str = ""

    # remote OpenAI-compatible providers, keyed by model id. Any role set to one
    # of these model ids is dispatched to the endpoint instead of Ollama. API keys
    # live in env vars (api_key_env), never here. e.g.:
    #   "providers": {"kimi-k3": {"base_url": "https://api.moonshot.ai/v1",
    #                             "api_key_env": "MOONSHOT_API_KEY", "temperature": 1}}
    providers: dict = field(default_factory=dict)

    # theme — clay brand + a hacker triad mapped to verify-loop states
    clay: str = "#D97757"          # brand / prompt / the mark
    spiral_style: str = "spiral"   # banner shape: spiral · galaxy · uzumaki
    live_green: str = "#35f0a0"    # tests green / task committed
    working_amber: str = "#ffb000" # generating / verifying
    fail_red: str = "#ff5c57"      # verify failed / stuck

    @classmethod
    def load(cls) -> "Config":
        """Defaults → config-file overlay → env vars. Models are fully swappable
        without touching code:

          env:   SPIRAL_WORKER / SPIRAL_PLANNER / SPIRAL_ESCALATION /
                 SPIRAL_CRITIC / SPIRAL_JANITOR / SPIRAL_BASE_URL
          file:  ~/.config/spiral/config.json →
                 {"models": {"worker": "...", ...}, "num_ctx": {...}, "hooks": {...}}
        """
        cfg = cls()
        try:
            import json
            import os
            from pathlib import Path

            roles = {"planner": cfg.planner, "worker": cfg.worker,
                     "escalation": cfg.escalation, "critic": cfg.critic, "janitor": cfg.janitor}

            f = Path.home() / ".config" / "spiral" / "config.json"
            overlay = json.loads(f.read_text()) if f.is_file() else {}
            for role, name in overlay.get("models", {}).items():
                if role in roles:
                    roles[role].name = str(name)
            for role, spec in roles.items():
                env = os.environ.get(f"SPIRAL_{role.upper()}")
                if env:
                    spec.name = env
                if spec.name in overlay.get("num_ctx", {}):
                    spec.num_ctx = int(overlay["num_ctx"][spec.name])
            cfg.base_url = os.environ.get("SPIRAL_BASE_URL", overlay.get("base_url", cfg.base_url))
            cfg.extra_gate = overlay.get("extra_gate", cfg.extra_gate)
            cfg.spiral_style = os.environ.get("SPIRAL_STYLE", overlay.get("style", cfg.spiral_style))
            cfg.worker_max_tokens = int(overlay.get("worker_max_tokens", cfg.worker_max_tokens))
            cfg.providers = overlay.get("providers", cfg.providers)
        except Exception:
            pass  # a broken overlay must never break spiral
        return cfg
