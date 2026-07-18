# Changelog

## 0.1.0 — 2026-07-18

First public release.

- The atom: edit → verify → fix → commit against a detected build gate, with
  SEARCH/REPLACE edits, fuzzy apply, and per-task attempt budgets.
- The conductor: spec extraction → plan → deterministic lint + coverage check →
  different-brain critic → repair → bootstrap (ratchet semantics: progress
  checkpoints while red) → green-to-green task grind → clean build → spec
  validation → remediation, until SPEC-GREEN or a plateau.
- Executable acceptance checks: requirements can carry a behavior-observing
  shell command; validation judges those by exit code and asks a model only
  about the rest; failed checks gate their own remediation tasks.
- Diversity round: best-of-N candidates at spread temperatures when the worker
  lane exhausts on a red gate; the gate judges, green commits, ratchet banks.
- Escalation lane with automatic distillation of wins into a learned-fixes
  skill; signature routing sends error classes the worker has never beaten
  straight to the dense model; `spiral distill` mines the ledger on demand.
- Context acquisition: per-task file assignment, static symbol index
  (viewBinding-aware), error-path absorption, ASK protocol, repo symbol hunt
  on repeated errors, janitor-compacted attempt memory.
- Designer stage for UI projects: one-time design brief, distilled tokens, and
  a deterministic Android launcher icon + palette wired before feature work.
- Safety: dedicated `spiral/run-*` branch, shell denylist (no `rm -rf`, no
  `sudo`, no `git push`, no network egress from the worker), GET-only research
  module as the single web door, progress-based lane termination, resumable
  state.
- Cockpit: pinned plan panel, live tok/s + ETA, Shift-Tab auto/step modes,
  heartbeat lines when piped; banner with selectable spiral styles; cascading
  opening lines.
- Commands: build/plan/validate/do, setup/tune/doctor, stats/distill/note/
  rewind, search/research/chat/consult, style. Tiered crew via OpenAI-compatible
  providers (`--boost`, `--api`), keys via environment variables only.
