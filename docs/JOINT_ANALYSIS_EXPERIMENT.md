# Joint requirements and deliverable analysis

Status: **opt-in experiment; sequential analysis remains the default**. No live
latency or task-success improvement is claimed from the offline tests below.

Set `planning_analysis_mode` to `joint` in the existing Spiral config overlay, or
set `SPIRAL_PLANNING_ANALYSIS_MODE=joint` for the engine process. Explicit
`sequential` restores the original two-stage flow. Invalid values fail rather than
silently running a different experiment. A supervising host must actually pass the
setting to its child or expose the same config overlay: setting an environment
variable in an unrelated terminal does not change an already-running app.

## What changes

The model returns one object containing `requirements`, `primary_id` and
`deliverables`. The combined schema reuses both existing schemas. The exact authored
goal is not sliced; repository context retains its existing 20,000-character bound
and untrusted-data status. No keyword, task-size threshold, site name or particular
dataset selects this route. The selected planner model, context size, residency
setting and shared call/token/time budgets are unchanged.

The first logical analysis uses the existing planner thinking preference and
finite planning ladder. A corrective analysis uses the existing answer-only
correction policy. The joint output ceiling is the smaller of the planner ceiling
and 10,240 tokens, the sum of the previous 4,096/6,144 analysis ceilings. This is
not a larger global run budget. Budget/transport failures propagate; no extra
model is substituted.

Both views must pass shared structural validation and the existing deliverable
and typed-prerequisite checks before either is accepted. Joint analysis rejects
malformed JSON, incomplete output and even closed JSON whose backend finish reason
reports a token limit. It never salvages a partial contract or silently falls back
to a heuristic manifest. There are at most two analysis rounds, using the existing
finite ladder within each round. Failed attempts can still write diagnostic ledger
records, but not accepted spec/artifact manifests or new analysis-derived
prerequisite setup. The preexisting build entry still synchronizes previously
saved project manifests before planning; this experiment does not suppress that
independent setup path.

Normalization is shared with sequential analysis, not duplicated in a weaker
experimental implementation. The common validators now explicitly reject missing
or incorrectly typed required fields even when a backend ignores its JSON grammar.
Requirements keep their original IDs and text; empty/duplicate/ill-typed checklists
are rejected rather than silently accepted. Neither structural validation nor a
model-derived checklist proves that every semantic commitment was understood.

After successful analysis, existing check sanitization, canonical enrichment,
prerequisite setup, visual-design stages, draft planning, risk-based criticism,
repair, coverage checks, persistence and final execution/verification gates remain
in place. The experiment does not remove task decomposition or independent checks.

## Measurement boundary

Both modes now record `plan_attempt` entries for these analysis stages. Each records
the model, requested thinking/token cap, elapsed wall time, logical attempt/round,
parse outcome and bounded numeric token/backend metrics when available. No new
prompt, answer or reasoning content is included in these records. A parsed response
is not necessarily a validated contract.

The shared LLM layer may perform inner transport or thinking recovery. The attempt
wall clock covers that logical call; its returned token accounting follows the
existing layer. Backend duration/cache metrics are labeled `returned_response_only`
and must not be mistaken for complete per-HTTP-attempt accounting. The thinking
recovery flag is retained when supplied. Missing metrics remain missing, not zero.
These diagnostic records do not change or double-charge the authoritative budget.

Ollama distinguishes total prompt tokens from optional cached prompt tokens;
`prompt_eval_duration` covers uncached evaluation. See the
[primary usage contract](https://docs.ollama.com/api/usage). Do not infer a cache-hit
rate from total input tokens alone or attribute slow loading to prompt caching.

A valid joint response saves one successful analysis dispatch and repeated prompt
evaluation. It does not remove required output generation. Larger combined replies
may instead increase omission, truncation or correction costs on complex goals.
Keep sequential as default until serialized held-out comparisons show a benefit
without worse independently verified outcomes. Record time to first worker, total
task time, every attempt, backend metrics, model identity and the original acceptance
results. No baseline failure becomes a pass because a phase used fewer calls.

## Offline verification

The focused joint/planner/prerequisite/capability/pipeline suite passed **114 tests
in 2.83 seconds** using the existing engine venv. New tests cover one valid joint
dispatch, exact long goals/model/options, finite thinking fallback records, strict
truncation/malformed rejection, shared validators/materialization, no premature
setup/manifests, unchanged budget exceptions, a 44-requirement canonical tail through
the real conductor/draft/critic seam, and explicit config/default handling. The first
run found a test-fixture `BudgetLimits` constructor error; that fixture was corrected
before the passing run. No inference, installation, dependency acquisition or Git
operation was used by this verification.

The first full engine suite exited zero with **1,123 passed in
147.48 seconds**, but it was **not fully offline**: an existing test automatically
called local Ollama when its executable was installed. That live-test gate now
requires both `SPIRAL_LIVE_OLLAMA_TESTS=1` and an exact
`SPIRAL_LIVE_OLLAMA_MODEL`; it never discovers or selects another model.
`SPIRAL_OFFLINE_TESTS=1` overrides both and is explicitly set for the rerun.
It emitted nonfatal Playwright teardown/pending-task diagnostics,
which were not hidden or counted as additional tests. This is source
verification, not a real joint-analysis speed or end-to-end task acceptance result.

After the review fix rejecting numeric `primary_id` values in both modes and the
explicit live-test gate, the focused suite (also including thinking recovery/gate
tests) passed **134 tests, 1 skipped in 2.69 seconds**. The full rerun with
`SPIRAL_OFFLINE_TESTS=1 .venv/bin/python -m pytest -q` passed **1,132 tests,
1 skipped in 85.40 seconds**. The skipped case is the explicitly gated live Ollama
probe. Nonfatal Playwright teardown/pending-task diagnostics remained visible.
These corrected results do not provide live acceptance evidence for joint analysis.
