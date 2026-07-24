"""Configuration — backend is a swappable seam; local-first defaults.

Model strategy (32 GB unified memory, hardware-honest):
  - ONE resident qwen3.6:35b-a3b model for planning, reading, writing, vision and
    ordinary criticism. Thinking is toggled by role; sharing weights avoids a
    23 GB ↔ 17 GB model swap in the middle of every research round.
  - qwen3.6:27b (dense) is the ESCALATION model only: swapped in when a task
    genuinely stalls, where slower-but-denser reasoning can earn its load time.
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
    # Ordinary critic shares the resident MoE weights. Model diversity comes from
    # the optional API critic (--boost/--api) or the dense escalation lane, rather
    # than paying a model swap for every corpus/proposal/referee call.
    critic: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:latest", num_ctx=24576, think=True)
    )
    # Independent research adjudicator. CLI API tiers intentionally do not remap
    # this role: proposal/basis/scope reviews should not become self-review merely
    # because planner, worker, and critic all use one frontier provider.
    research_auditor: ModelSpec = field(
        default_factory=lambda: ModelSpec("qwen3.6:27b", num_ctx=16384, think=False)
    )

    def spec_for(self, model_name: str) -> ModelSpec:
        """The ModelSpec whose name matches — so per-model num_ctx follows the
        model wherever it's used (worker vs escalation lanes)."""
        for spec in (
            self.worker, self.escalation, self.planner, self.critic,
            self.research_auditor, self.janitor,
        ):
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
    plan_rounds: int = 3               # lint→critic→repair cycles before execution
    validate_rounds: int = 8           # max validate→remediate cycles; stops early on a true plateau
    # Applied automatically when any main role is metered. Purely local research has
    # no implicit token stop; the CLI's explicit --token-budget still applies.
    run_token_budget: int = 4_000_000
    # Explicit Builder ceiling. Zero means unlimited when all active roles are local;
    # metered roles still inherit run_token_budget.
    builder_token_budget: int = 0
    verify_timeout: int = 900          # seconds; real build gates (gradle) are slow
    # best-of-N at the worker lane's exit: sampled candidates judged by the gate.
    # Local tokens are free and the gate is a deterministic judge — brute force
    # is spent exactly where a metered agent would economize. 0 disables.
    diversity_samples: int = 3

    # Worker research: repo/file/web/browser ASKs do not consume edit attempts.
    # For all action-count limits in this section, zero means unlimited. Web is
    # GET-only through spiral.research and each result is persisted under
    # .spiral/research/ for audit.
    ask_budget: int = 32
    web_research: bool = True
    web_research_budget: int = 24
    web_research_k: int = 8
    builder_repo_auto: bool = True
    builder_repo_budget: int = 3
    builder_repo_max_mb: int = 500
    # Remote package code is acquired automatically, but arbitrary lifecycle/build
    # hooks stay off unless the user explicitly accepts that execution boundary.
    builder_allow_install_scripts: bool = False
    builder_tool_auto: bool = True
    builder_tool_install_budget: int = 6
    builder_shell_budget: int = 24
    builder_vision_budget: int = 8
    builder_browser_budget: int = 8
    builder_shell_timeout: int = 300
    builder_require_sandbox: bool = True
    vision_model: str = ""
    visual_review: bool = True
    visual_review_url: str = ""
    visual_review_rounds: int = 3
    visual_review_timeout: int = 45
    product_audit_rounds: int = 3
    finish_rounds: int = 4
    builder_remediation_batch: int = 6
    builder_remediation_attempts: int = 3
    builder_remediation_escalation_attempts: int = 2
    # Vision-capable thinking models can consume a 2k allowance before emitting
    # their JSON defect report; keep enough room for reasoning plus the verdict.
    visual_review_max_tokens: int = 8192
    research_repo_auto: bool = False
    research_repo_budget: int = 1
    research_repo_max_mb: int = 750
    research_cleanup_failed_repos: bool = True
    research_tool_auto: bool = True
    research_tool_install_budget: int = 4
    # Public scientific data is acquired only by the typed Research data broker.
    # The model never receives an unrestricted networked shell. The broker resolves
    # catalog metadata first, computes the exact selected byte total, keeps a disk
    # reserve, resumes partial files, hashes content, and records licences/versions.
    research_data_auto: bool = True
    research_data_catalog_limit: int = 18
    research_data_max_gb: float = 20.0
    research_data_reserve_gb: float = 8.0
    research_data_file_limit: int = 20_000
    research_data_timeout: int = 3600
    research_data_sources: list[str] = field(
        default_factory=lambda: ["openneuro", "allen", "neuromaps", "zenodo"])
    research_notes_model: str = ""
    research_search_results_per_query: int = 8
    research_reading_limit: int = 60
    research_deep_read_limit: int = 8
    research_deep_chunk_limit: int = 10
    research_min_grounded_notes: int = 6
    research_min_grounded_deep_reads: int = 2
    research_min_papers: int = 10
    research_min_usable_texts: int = 6
    research_min_relevant_papers: int = 5
    research_min_relevant_usable_primary_texts: int = 4
    research_min_unique_queries: int = 3
    research_min_healthy_searches: int = 2
    research_min_relevant_query_families: int = 2
    research_min_topic_term_coverage: float = 0.45
    research_min_graph_success_rate: float = 0.60
    # Epistemic kernel and discovery policy. These gates are strict by default for
    # original research and are bypassed only by explicit expository verification mode.
    research_obligation_graph: bool = True
    research_blind_replication: bool = True
    research_replication_attempts: int = 2
    research_counterfactuals: bool = True
    research_counterfactual_limit: int = 6
    research_information_scheduler: bool = True
    research_information_gain_floor: float = 0.04
    research_plateau_patience: int = 8
    # Stalled rounds (flat corpus, dead instruments, only instrument checks failing)
    # before discovery degrades explicitly instead of vetoing forever.
    research_stall_patience: int = 3
    research_taste_model: bool = True
    research_git: bool = True
    research_living_papers: bool = True
    research_living_recheck_days: int = 30

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

            roles = {
                "planner": cfg.planner,
                "worker": cfg.worker,
                "escalation": cfg.escalation,
                "critic": cfg.critic,
                "research_auditor": cfg.research_auditor,
                "janitor": cfg.janitor,
            }

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
            cfg.run_token_budget = int(
                overlay.get("run_token_budget", cfg.run_token_budget))
            cfg.builder_token_budget = int(
                overlay.get("builder_token_budget", cfg.builder_token_budget))
            cfg.diversity_samples = int(overlay.get("diversity_samples", cfg.diversity_samples))
            cfg.ask_budget = int(overlay.get("ask_budget", cfg.ask_budget))
            cfg.web_research = bool(overlay.get("web_research", cfg.web_research))
            cfg.web_research_budget = int(overlay.get("web_research_budget", cfg.web_research_budget))
            cfg.web_research_k = int(overlay.get("web_research_k", cfg.web_research_k))
            cfg.builder_repo_auto = bool(
                overlay.get("builder_repo_auto", cfg.builder_repo_auto))
            cfg.builder_repo_budget = int(
                overlay.get("builder_repo_budget", cfg.builder_repo_budget))
            cfg.builder_repo_max_mb = int(
                overlay.get("builder_repo_max_mb", cfg.builder_repo_max_mb))
            cfg.builder_allow_install_scripts = bool(overlay.get(
                "builder_allow_install_scripts", cfg.builder_allow_install_scripts))
            cfg.builder_tool_auto = bool(overlay.get(
                "builder_tool_auto", cfg.builder_tool_auto))
            cfg.builder_tool_install_budget = int(overlay.get(
                "builder_tool_install_budget", cfg.builder_tool_install_budget))
            cfg.builder_shell_budget = int(overlay.get(
                "builder_shell_budget", cfg.builder_shell_budget))
            cfg.builder_vision_budget = int(overlay.get(
                "builder_vision_budget", cfg.builder_vision_budget))
            cfg.builder_browser_budget = int(overlay.get(
                "builder_browser_budget", cfg.builder_browser_budget))
            cfg.builder_shell_timeout = int(overlay.get(
                "builder_shell_timeout", cfg.builder_shell_timeout))
            cfg.builder_require_sandbox = bool(overlay.get(
                "builder_require_sandbox", cfg.builder_require_sandbox))
            cfg.vision_model = os.environ.get("SPIRAL_VISION", overlay.get("vision_model", cfg.vision_model))
            cfg.visual_review = bool(overlay.get("visual_review", cfg.visual_review))
            cfg.visual_review_url = os.environ.get(
                "SPIRAL_VISUAL_URL", overlay.get("visual_review_url", cfg.visual_review_url))
            cfg.visual_review_rounds = int(overlay.get("visual_review_rounds", cfg.visual_review_rounds))
            cfg.visual_review_timeout = int(overlay.get("visual_review_timeout", cfg.visual_review_timeout))
            cfg.product_audit_rounds = int(
                overlay.get("product_audit_rounds", cfg.product_audit_rounds))
            cfg.finish_rounds = int(overlay.get("finish_rounds", cfg.finish_rounds))
            cfg.builder_remediation_batch = int(overlay.get(
                "builder_remediation_batch", cfg.builder_remediation_batch))
            cfg.builder_remediation_attempts = int(overlay.get(
                "builder_remediation_attempts", cfg.builder_remediation_attempts))
            cfg.builder_remediation_escalation_attempts = int(overlay.get(
                "builder_remediation_escalation_attempts",
                cfg.builder_remediation_escalation_attempts))
            cfg.visual_review_max_tokens = int(
                overlay.get("visual_review_max_tokens", cfg.visual_review_max_tokens))
            cfg.research_repo_auto = bool(overlay.get("research_repo_auto", cfg.research_repo_auto))
            cfg.research_repo_budget = int(overlay.get("research_repo_budget", cfg.research_repo_budget))
            cfg.research_repo_max_mb = int(overlay.get("research_repo_max_mb", cfg.research_repo_max_mb))
            cfg.research_cleanup_failed_repos = bool(
                overlay.get("research_cleanup_failed_repos", cfg.research_cleanup_failed_repos))
            cfg.research_tool_auto = bool(
                overlay.get("research_tool_auto", cfg.research_tool_auto))
            cfg.research_tool_install_budget = int(overlay.get(
                "research_tool_install_budget",
                cfg.research_tool_install_budget))
            cfg.research_data_auto = bool(overlay.get(
                "research_data_auto", cfg.research_data_auto))
            cfg.research_data_catalog_limit = int(overlay.get(
                "research_data_catalog_limit", cfg.research_data_catalog_limit))
            cfg.research_data_max_gb = float(overlay.get(
                "research_data_max_gb", cfg.research_data_max_gb))
            cfg.research_data_reserve_gb = float(overlay.get(
                "research_data_reserve_gb", cfg.research_data_reserve_gb))
            cfg.research_data_file_limit = int(overlay.get(
                "research_data_file_limit", cfg.research_data_file_limit))
            cfg.research_data_timeout = int(overlay.get(
                "research_data_timeout", cfg.research_data_timeout))
            configured_sources = overlay.get(
                "research_data_sources", cfg.research_data_sources)
            if isinstance(configured_sources, list):
                cfg.research_data_sources = [
                    str(source).strip().lower() for source in configured_sources
                    if str(source).strip()
                ]
            cfg.research_notes_model = os.environ.get(
                "SPIRAL_RESEARCH_NOTES_MODEL",
                overlay.get("research_notes_model", cfg.research_notes_model),
            )
            cfg.research_search_results_per_query = int(overlay.get(
                "research_search_results_per_query",
                cfg.research_search_results_per_query))
            cfg.research_reading_limit = int(
                overlay.get("research_reading_limit", cfg.research_reading_limit))
            cfg.research_deep_read_limit = int(
                overlay.get("research_deep_read_limit", cfg.research_deep_read_limit))
            cfg.research_deep_chunk_limit = int(
                overlay.get("research_deep_chunk_limit", cfg.research_deep_chunk_limit))
            cfg.research_min_grounded_notes = int(overlay.get(
                "research_min_grounded_notes", cfg.research_min_grounded_notes))
            cfg.research_min_grounded_deep_reads = int(overlay.get(
                "research_min_grounded_deep_reads", cfg.research_min_grounded_deep_reads))
            cfg.research_min_papers = int(
                overlay.get("research_min_papers", cfg.research_min_papers))
            cfg.research_min_usable_texts = int(
                overlay.get("research_min_usable_texts", cfg.research_min_usable_texts))
            cfg.research_min_relevant_papers = int(
                overlay.get("research_min_relevant_papers", cfg.research_min_relevant_papers))
            cfg.research_min_relevant_usable_primary_texts = int(overlay.get(
                "research_min_relevant_usable_primary_texts",
                cfg.research_min_relevant_usable_primary_texts))
            cfg.research_min_unique_queries = int(
                overlay.get("research_min_unique_queries", cfg.research_min_unique_queries))
            cfg.research_min_healthy_searches = int(
                overlay.get("research_min_healthy_searches", cfg.research_min_healthy_searches))
            cfg.research_min_relevant_query_families = int(overlay.get(
                "research_min_relevant_query_families",
                cfg.research_min_relevant_query_families))
            cfg.research_min_topic_term_coverage = float(overlay.get(
                "research_min_topic_term_coverage", cfg.research_min_topic_term_coverage))
            cfg.research_min_graph_success_rate = float(overlay.get(
                "research_min_graph_success_rate", cfg.research_min_graph_success_rate))
            cfg.research_obligation_graph = bool(overlay.get(
                "research_obligation_graph", cfg.research_obligation_graph))
            cfg.research_blind_replication = bool(overlay.get(
                "research_blind_replication", cfg.research_blind_replication))
            cfg.research_replication_attempts = int(overlay.get(
                "research_replication_attempts", cfg.research_replication_attempts))
            cfg.research_counterfactuals = bool(overlay.get(
                "research_counterfactuals", cfg.research_counterfactuals))
            cfg.research_counterfactual_limit = int(overlay.get(
                "research_counterfactual_limit", cfg.research_counterfactual_limit))
            cfg.research_information_scheduler = bool(overlay.get(
                "research_information_scheduler", cfg.research_information_scheduler))
            cfg.research_information_gain_floor = float(overlay.get(
                "research_information_gain_floor", cfg.research_information_gain_floor))
            cfg.research_plateau_patience = int(overlay.get(
                "research_plateau_patience", cfg.research_plateau_patience))
            cfg.research_stall_patience = int(overlay.get(
                "research_stall_patience", cfg.research_stall_patience))
            cfg.research_taste_model = bool(overlay.get(
                "research_taste_model", cfg.research_taste_model))
            cfg.research_git = bool(overlay.get("research_git", cfg.research_git))
            cfg.research_living_papers = bool(overlay.get(
                "research_living_papers", cfg.research_living_papers))
            cfg.research_living_recheck_days = int(overlay.get(
                "research_living_recheck_days", cfg.research_living_recheck_days))
            cfg.providers = overlay.get("providers", cfg.providers)
        except Exception:
            pass  # a broken overlay must never break spiral
        return cfg
