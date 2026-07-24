<p align="center">
  <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/spiral.gif" alt="spiral" width="680"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-D97757?style=flat-square" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-D97757?style=flat-square" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/backend-Ollama-D97757?style=flat-square" alt="Ollama"/>
  <img src="https://img.shields.io/badge/macOS%20%C2%B7%20Linux-D97757?style=flat-square" alt="macOS · Linux"/>
</p>

<p align="center">
  An autonomous coding agent that runs on local models. It turns a goal into a
  requirements checklist, implements each requirement against your project's real
  build or test command, and verifies the result before reporting done.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#example-run">Example run</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#principles">Principles</a>
</p>

<p align="center"><img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/divider.svg" width="520" alt=""/></p>

spiral is a command-line coding agent that runs entirely on local models through
[Ollama](https://ollama.com). Given a goal, it extracts a list of requirements,
plans the work, and implements each task against your project's actual build or
test command. Changes that pass are committed to git; changes that fail are
reverted. When the plan is finished, the code is checked against each
requirement — by running the requirement's acceptance check where one exists,
and by a separate model where none does — and spiral keeps working until all
are met or it reports which ones remain. No API keys and no network calls to a
model provider.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Install

```bash
pipx install spiral-coder     # isolated, puts `spiral` on your PATH (recommended)
# or
pip install spiral-coder
```

Either way you get a global `spiral` command. From a clone, `pip install -e .`
installs it in editable mode.

Requires Python 3.11+, [Ollama](https://ollama.com), and at least one local
model. Apple Silicon with 32 GB+ of unified memory is recommended for the
default model set; smaller machines can run a smaller crew (see
[`spiral setup`](#commands)).

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Quickstart

```bash
spiral setup                      # first run: detect Ollama, pull a RAM-matched model crew
spiral tune                       # once per machine: size model context windows to your RAM
spiral build "make me a pomodoro TUI in python, with tests"
```

spiral runs on a dedicated git branch (`spiral/run-*`), leaving your working
branch untouched. It commits each verified step and prints a summary when the
run ends:

```
╭──────────────────────────── ⠷ run summary ────────────────────────────╮
│ 11/12 tasks green · 1 blocked · SPEC-GREEN                             │
│ Σ 174,787 tok (141k in / 33k out) · 34 attempts · 2 escalations · 57m  │
│ qwen3.6:latest · 28 gen · median 30 t/s                                │
│ ≈ $0.93 of equivalent cloud API · $0.00 spent                          │
╰────────────────────────────────────────────────────────────────────────╯
```

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Example run

<p align="center">
  <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/demo.gif" alt="a recorded run: gate detection, run branch, spec, plan, approval, first task" width="680"/>
</p>

<p align="center"><sub>A recorded run: gate detection, the run branch, the spec, the milestone
plan, approval, and the first task under the live plan panel. Model waits are time-lapsed.</sub></p>

If the project does not build at the start, spiral repairs it first. Each
attempt that reduces the number of errors is committed, so progress is kept even
if a later attempt fails or the run is stopped:

```
━━ M0: bootstrap — make the build gate pass ━━
  — attempt 1/12 · qwen3.6:latest —
  ● edits: BigBrotherEyeView.kt(exact) · verify exit 1
     gate says: e: MainActivity.kt:116:21 Unresolved reference 'messageText'
  ⚑ progress banked ec7fbc2 · resolved 5, revealed 4, remaining 5
  — attempt 2/12 —
  ● edits: MainActivity.kt(exact) · verify exit 0
  ✔ committed fc2ed05
  ■ gate is green — features begin
```

After the plan finishes, the code is checked against each requirement.
Requirements with an executable check are judged by exit code; the rest by a
separate model. Unmet requirements become new tasks:

```
━━ validation 1 · 27 requirements · 4 by execution · qwen3.6:27b judges the rest ━━
  ✓ R2  acceptance check passed: python -m pytest tests/test_timer.py -q
  ✓ R4  activity_login.xml binds et_name; LoginActivity validates input
  ◐ R10 btnSend calls sendMessage(), but the scan is never triggered
  ✗ R14 no siren playback found anywhere in the code
  spec: 16/27 implemented · 8 partial · 1 missing · 2 unjudged
▶ V.1 implement R14 …
```

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> How it works

```
 spec extraction ──▶ design brief ──▶ plan ──▶ critic review ──▶ repair
 (+ acceptance     (UI projects only:         (a separate model
    checks)         tokens + a real icon)      reviews the plan)
       ▼
 bootstrap the gate to green            each resolved error is committed, so
       │                                repair converges across attempts
       ▼
 foundation: design system + launcher icon    (deterministic, for UI apps)
       │
       ▼
 implement tasks against the gate       matched skills · attempt memory ·
       │                                ASK protocol · symbol search ·
       │                                diversity round · escalation ·
       │                                signature routing · reused fixes
       ▼
 product audit ──▶ mobile/desktop/wide browser QA ──▶ clean build
       ▲                                              │
       └──── spec validation ◀── remediation ◀────────┘  (fixed point)
```

**Verification.** A task is complete only when it passes every check that
applies: the build or test gate (compiles, tests pass), a footgun lint welded
into that gate (patterns that compile but crash at runtime), an artifact check
(the files the task declared exist), a behavior audit (the task actually
changed something relevant), and the final spec validation (the requirement is
implemented). You can append your own check with `extra_gate`.
For full product requests, Spiral also rejects production TODOs/placeholders,
ceremonial tests, missing run instructions, unlabeled or non-exportable plots,
and unreproducible simulations. A clean deterministic audit is evidence against
these known failure classes, not a proof of subjective product quality.

**Acceptance checks.** At spec time each requirement can get one shell command
that exits 0 exactly when the requirement is met — run a test, invoke the CLI,
execute the program. A lint drops presence-style commands (`grep`, `ls`,
`test -f`) and anything on the denylist. Validation runs these checks first and
only asks a model about requirements that have none; a failed check becomes a
remediation task gated on the check itself.

**Diversity round.** When the worker exhausts its attempts on a red gate,
spiral samples N fresh candidates (default 3) at spread temperatures and runs
the gate on each. A green candidate is committed; during bootstrap the best red
candidate is banked if it resolved errors. The round never runs on a green
gate, so a no-op candidate cannot pass as done.

**Context acquisition.** Local models reference identifiers they have not been
shown, and do not reliably notice something is missing. spiral supplies context
in stages rather than relying on the model to ask:

1. the planner assigns relevant files to each task;
2. a static symbol index (types, members, layout-id → binding class) rides the prompt;
3. file paths named in build errors are added automatically;
4. the worker can request files, web evidence, or a public GitHub reference repository;
5. acquired repos are shallow, credential-free, commit/license recorded, and never executed;
6. a repeated identical error triggers symbol and official-doc research;
7. a task that exhausts its attempts escalates to the stronger model.

**Learning across runs.** Every attempt is logged to `.spiral/ledger.jsonl`
with the error signature it faced (normalized, so line numbers don't split
them) and whether it cleared it. When escalation solves something the worker
could not, the fix is appended to `learned-fixes.md` for later runs. A
signature the worker has repeatedly failed and only escalation has solved is
routed straight to escalation next time. `spiral distill` prints the table and
writes `.spiral/route.json`.

**Safety.** Work happens on a `spiral/run-*` branch; you merge. The model's
shell blocks destructive commands even in full-auto — `rm -rf`, `sudo`,
`mkfs`, `dd`, `git push`, `git reset --hard` — and blocks raw `curl`/`wget`.
The worker can still research the internet when it needs to: it may ask
`ASK: web <query>`, and repeated gate failures trigger automatic web research.
Those lookups go through the research module: GET-only, size-capped, saved under
`.spiral/research/`, and treated as source material rather than instructions.
It may also ask `ASK: repo <https://github.com/owner/repo>`; Builder records the
exact commit, size, tree, README, and license in `.spiral/tools/`, does not run
the clone, and removes partial/failed acquisitions. JavaScript and declarative
Python dependencies are synchronized before gates in credential-scrubbed local
caches. Package lifecycle hooks and Python source builds remain disabled unless
`--allow-install-scripts` (or the matching config key) is explicitly enabled.
For UI projects, `spiral build` also runs screenshot-based visual QA before the
final spec audit: Playwright captures mobile, desktop, and wide views; DOM/runtime,
keyboard, request, canvas-pixel, overflow, clipping, labeling, and target-size
checks run before a local vision model reviews domain fit and visual craft. The
Chromium runtime is installed automatically into a shared Spiral cache. Serious
defects become ordinary gated remediation tasks. Ctrl-C
stops cleanly; committed work is kept and `--resume` continues.

The live cockpit includes a pinned `thoughts` panel above the plan. It shows an
explicit working note: the current error, source lookup, candidate question,
rejection reason, visual-review focus, or next decision. Press `t` during a TTY
run to expand/collapse recent notes. The hash-chained decision trail is appended
to `.spiral/thoughts.jsonl` for builds and `spiral-research/thoughts.jsonl` for
research; `model-calls.jsonl` records the exact replayable prompts/evidence packets,
final model outputs, routing, usage, whether deep reasoning was requested, and a
length/hash (not the contents) of any private reasoning channel returned. These are
auditable scientific records, not a claim to expose a model's private hidden chain of
thought. Question discovery, angle selection, proposal critique, supervisor reflection,
and the first paper referee use the deep-reasoning lane; citation and claim-row
classification use the concise structured lane.

`spiral research --solve` uses the same principle for papers: the model proposes,
but SymPy/Lean/numeric/workbench certificates decide. Workbench certificates can
run Python, Lean/Lake, Sage, Singular, Rust, Go, Julia, R, Java, Swift, and
multi-step C/C++ compile/run bundles when those local toolchains are installed.
Public GitHub repos are cloned only when `--auto-repos` or
`research_repo_auto` is enabled; clones live inside the certificate workspace and
are removed again when the certificate fails. On macOS, model-authored certificate
commands run in an offline OS sandbox: user/volume data is unreadable except for the
exact certificate and installed runtime roots, writes are confined to the certificate
directory, and network sockets are denied. Dependency and Git acquisition happen
beforehand as separate recorded operations; Python dependencies are restricted to a
known research-package set, installed from binary wheels under a scrubbed environment.
Failed execution output is sent only to a local repair model, even under `--api`. On platforms
without an available OS sandbox the manifest says so explicitly; command screening
alone is not presented as isolation.

Data-driven Research uses a separate typed scientific-data broker rather than giving
generated code a networked shell. It searches OpenNeuro, Allen, the curated neuromaps
PET/brain-map registry, and Zenodo metadata,
pins accessions/releases/licences/citations, resolves the complete selected file list
and byte total, preserves a free-disk reserve, resumes partial downloads, hashes every
file, and then hard-links immutable cached data into `_data/ALIAS` inside the offline
certificate. A statistical analysis plan is locked before execution. Spatial nulls,
multiple-testing policy, held-out validation, causal scope, participant linkage,
coordinate-space registration and cross-species bridges are explicit gates. Exploratory
analyses may guide the next round, but cannot earn confirmatory evidence or unlock a
paper result.

The default research mode is question discovery and bounded novelty, not forcing
the literal prompt into a paper. A shared obligation graph carries user intent,
questions, assumptions, falsifiers, claims, evidence, replications, novelty scope,
and final artifacts through every phase. Its control flow is:

1. Plan several independent search routes and retrieve primary text.
2. Gate corpus readiness using relevant usable primary text (the same papers must
   satisfy both tests), topic coverage, distinct healthy query families that
   retrieved relevant records, and current citation-graph closure. Large graphs are
   audited in deterministic 30-paper batches and cannot be called saturated until
   every current seed has appeared in a healthy closed batch. Capture-recapture is
   reported as a diagnostic, never treated as proof that the literature is complete.
3. Rank search and reading actions by measured information gain, write
   source-anchored notes across the corpus, cluster idea families, then deep-read
   the strongest and nearest-prior papers.
4. Generate candidate questions plus one-change counterfactuals (boundary cases,
   singular limits, method transfers, and possible obstructions), search each
   proposed novelty move, and reject candidates that are known, thin, or untestable.
5. Use a transparent machine-local taste profile only to order admissible angles;
   it never overrides source, novelty, or verification gates. Commit one bounded
   question only after an exact-anchor basis audit and proposal
   referee. "Our documented search did not locate X" is allowed; an unsupported
   "X is the first" is not.
6. Derive self-contained claims with assumptions and falsifiers. After the first
   certificate passes, give a different local model a blinded brief without the
   original proof/code/output and require a method-distinct qualifying replication.
7. Issue a signed novelty-boundary certificate containing the exact claim scope,
   queries, source health, nearest results, primary-text reads, date, and the explicit
   warning that a bounded search is not proof of global absence or priority.
8. Recheck prior art, run supervisor reflection, and either loop, stop on a
   observable plateau, or enter writing only when the completion gate is green.
9. Infer a corpus-conditioned paper blueprint, notation table, equation map, and
   vocabulary guide; draft sections; then gate coherence, exact citation support,
   claim scope, final semantic review, abstract-last consistency, and a fresh
   reproducible LaTeX compile.
10. Release a proof-carrying paper whose sentence-level claims point to evidence,
    checkpoint the full lineage in a private research Git object database, and write
    a living-paper manifest that reopens literature/novelty obligations when local
    evidence changes or the recheck horizon expires.

Reasoning and rendering are separate lanes in the writer. Thinking calls choose the
outline, adjudicate evidence, and issue referee decisions; bounded non-thinking calls
apply a specified full-text transformation so hidden deliberation cannot consume the
entire output allowance. Every late rewrite is transactional: it replaces the last
green draft only after structure, citation, and claim-scope audits all pass again.

No finite search can prove open-world novelty. The run therefore preserves the
databases, queries, dates, source hashes, exact anchors, rejected angles, and
coverage report needed to state exactly what was and was not established.

**Models.** Each role can be set to any Ollama model:

| role | default | purpose |
|---|---|---|
| worker / planner | `qwen3.6:latest` (MoE, ~3B active) | plans and implements tasks |
| escalation | `qwen3.6:27b` (dense) | retries a task the worker could not finish |
| critic / validator / designer | `qwen3.6:latest`, thinking | ordinary reviews without a model swap; difficult semantic audits escalate to the dense model |
| research auditor | `qwen3.6:27b` (dense) | independent basis, claim-scope, and paper adjudication; remains local under `--boost`/`--api` |
| janitor | `llama3.2:1b` | summarizes attempt history to keep prompts short |

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Commands

| command | description |
|---|---|
| `spiral build "goal"` | plan, implement, validate, and remediate |
| `spiral build --resume` | continue a previous run |
| `spiral build --approve` | print the plan and wait for confirmation before running |
| `spiral build --boost` | local worker; escalation and critic/validator on the configured API provider |
| `spiral build --api` | run the entire crew on the configured API provider |
| `spiral build --visual-url URL` | screenshot this URL for local vision-model UI review |
| `spiral build --vision-model MODEL` | use a specific Ollama vision model for UI review |
| `spiral build --no-visual-review` | disable screenshot + vision UI review for one build |
| `spiral build --auto-repos` | allow credential-free public GitHub reference acquisition (default) |
| `spiral build --no-auto-repos` | disable public GitHub reference acquisition |
| `spiral build --allow-install-scripts` | permit third-party package lifecycle/source-build code for one build |
| `spiral plan "goal"` | show the decomposition without running it |
| `spiral validate` | check existing code against the goal's spec (read-only) |
| `spiral do "task" --verify "cmd"` | run a single task against one verify command |
| `spiral setup` | detect Ollama and pull a model crew sized to this machine |
| `spiral tune` | size model context windows to available memory |
| `spiral tune --wired` | also raise the macOS GPU wired-memory limit (sudo; reverts on reboot) |
| `spiral doctor` | check Ollama, models, tuning, gate, git, and disk |
| `spiral stats` | token counts, per-model throughput, and outcomes from the run log |
| `spiral distill` | mine the ledger: signature routing table + new learned-fixes entries |
| `spiral note "text"` | add a note that is included in every worker prompt |
| `spiral rewind [n]` | list task checkpoints and reset the run branch to one |
| `spiral style [name]` | set the banner shape: `spiral`, `galaxy`, or `uzumaki` |
| `spiral search "query"` | fast ranked web results, no synthesis (`--sci` adds arXiv) |
| `spiral research "query"` | gather web/arXiv/PubMed sources and synthesize a cited answer (`--deep`, `--sci`) |
| `spiral research "topic" --solve` | iterative novelty loop: gather corpus, snowball citations, verify claims, write paper |
| `spiral research --solve --resume` | resume an interrupted research loop |
| `spiral research --solve --refresh` | reopen a completed living paper for evidence and literature refresh |
| `spiral research --solve --verification` | force literal verification-note mode instead of novelty mode |
| `spiral research --solve --auto-repos` | allow public GitHub repos in workbench certificates, with failure cleanup |
| `spiral research --solve --no-blind-replication` | explicitly disable the default blind-replication gate |
| `spiral research --solve --no-counterfactuals` | disable neighboring-hypothesis generation |
| `spiral research --solve --no-research-git` | disable the private research checkpoint store |
| `spiral research --solve --token-budget N` | set an explicit run token ceiling; local runs otherwise have no implicit token limit |
| `spiral research --graph` | render an existing `spiral-research/research-map.json` to `research-graph.html` |
| `spiral research --history` | show the private content-addressed research checkpoint lineage |
| `spiral research --audit` | verify obligation/event chains, novelty boundary, proof bundle, and living-paper freshness |
| `spiral research --taste-like "angle"` | teach the machine-local taste profile a research direction you value |
| `spiral research --taste-dislike "angle"` | teach the machine-local taste profile a direction to de-emphasize |
| `spiral chat ["message"]` | talk to the local thinking model; reasoning shown dimmed |
| `spiral consult ["question"]` | send the whole project to a big-context API model for review |

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Live controls

During a run:

- **⇧ Tab** — switch between `auto` and `step` mode; the current mode is shown in the status line.
- **t** — expand/collapse the pinned recent-decisions panel.
- **step mode** — pause at each task: `enter` to run it, `s` to skip, `a` to return to auto, `q` to stop.
- **Ctrl-C** — stops cleanly. Committed work is kept and `--resume` continues from there.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Configuration

Models can be set per shell or persistently.

```bash
export SPIRAL_WORKER=qwen3.6:latest
export SPIRAL_ESCALATION=qwen3.6:27b
export SPIRAL_BASE_URL=http://localhost:11434
```

`~/.config/spiral/config.json` is written by `spiral setup` and `spiral tune`,
and can be edited directly:

```json
{
  "models":     { "worker": "qwen3.6:latest", "critic": "qwen3.6:latest", "escalation": "qwen3.6:27b", "research_auditor": "qwen3.6:27b" },
  "num_ctx":    { "qwen3.6:latest": 28672, "qwen3.6:27b": 57344 },
  "extra_gate": "ktlint app/src",
  "diversity_samples": 3,
  "visual_review": true,
  "visual_review_url": "",
  "vision_model": "qwen3.6:35b-a3b",
  "builder_repo_auto": true,
  "builder_repo_budget": 3,
  "builder_repo_max_mb": 500,
  "builder_allow_install_scripts": false,
  "finish_rounds": 4,
  "research_repo_auto": false,
  "research_repo_budget": 1,
  "research_repo_max_mb": 750,
  "research_data_auto": true,
  "research_data_catalog_limit": 18,
  "research_data_max_gb": 20,
  "research_data_reserve_gb": 8,
  "research_data_file_limit": 20000,
  "research_data_sources": ["openneuro", "allen", "neuromaps", "zenodo"],
  "research_notes_model": "qwen3.6:latest",
  "research_search_results_per_query": 8,
  "research_reading_limit": 60,
  "research_deep_read_limit": 8,
  "research_blind_replication": true,
  "research_replication_attempts": 2,
  "research_counterfactuals": true,
  "research_information_scheduler": true,
  "research_plateau_patience": 8,
  "research_git": true,
  "research_living_papers": true,
  "research_living_recheck_days": 30,
  "providers": {
    "kimi-k3": { "base_url": "https://api.moonshot.ai/v1", "api_key_env": "MOONSHOT_API_KEY" }
  },
  "hooks": {
    "run_complete": "osascript -e 'display notification \"$SPIRAL_INFO\" with title \"spiral\"'",
    "spec_green":   "say 'spec green'",
    "blocked":      "say 'spiral is stuck'"
  }
}
```

- `extra_gate` — a command appended to every task's gate. If it exits non-zero, the task is not complete.
- `diversity_samples` — candidates in the best-of-N round at the worker lane's exit (default 3, max 5, 0 disables).
- `providers` — OpenAI-compatible endpoints, keyed by model id. Any role set to one of these ids is served by that endpoint instead of Ollama. The API key is read from the environment variable named in `api_key_env` and never written to disk. `--boost` and `--api` remap roles onto the first provider; without them everything runs local.
- `run_token_budget` — safety ceiling used automatically when a main research role is metered. It is not applied to an all-local research run unless `--token-budget` is supplied explicitly; wall time, RAM, and disk remain real costs.
- `visual_review` — enables screenshot-based UI review for web/static UI targets. Set `visual_review_url`, `.spiral/visual_url`, or `SPIRAL_VISUAL_URL` when the app needs a specific running URL.
- `builder_repo_*` — controls credential-free, non-executing public GitHub reference acquisition. Failed and oversized clones are removed.
- `builder_allow_install_scripts` — permits package lifecycle hooks and Python source builds. The default false still installs npm/pnpm/yarn/bun dependencies with hooks disabled and Python dependencies from binary wheels.
- `finish_rounds` — bounds the final product/visual/runtime/spec fixed-point loop (default 4); exact repeated evidence stops it early.
- `research_repo_auto` — lets research workbench certificates clone public GitHub repos into their local certificate directory. The default is false; `--auto-repos` enables it for one run.
- `research_data_*` — controls typed public catalog discovery and scientific-data
  acquisition. Limits are checked against the resolved selection before transfer;
  zero-trust model execution remains offline.
- `research_notes_model` — optional local model for broad per-paper reading notes. If unset, research uses the local worker model even when `--api` routes the main reasoning roles to an API provider.
- `research_search_results_per_query` — breadth requested from each independent keyword route before citation-graph expansion (default 8).
- `research_reading_limit` / `research_deep_read_limit` — cap broad paper notes and later zoom-in reads so long corpora stay context-manageable.
- `research_blind_replication` / `research_replication_attempts` — require a
  solution-hidden, method-distinct independent certificate for every required
  original-research claim and bound its regeneration attempts.
- `research_counterfactuals` — probes source-adjacent boundary cases, changed
  assumptions, method transfers, and no-go routes before committing an angle.
- `research_information_*` / `research_plateau_patience` — rank actions and stop
  only from observed retrieval yield, health, redundancy, coverage, and sustained
  lack of qualifying evidence.
- `research_git` — records metadata, notes, decisions, certificates, audits, and
  papers in `spiral-research/.research-git` without touching an enclosing Git repo.
- `research_living_papers` / `research_living_recheck_days` — hash the completed
  evidence envelope and reopen it after local drift or the literature recheck horizon.
- `research_min_*` coverage settings — deterministic lower bounds for papers,
  usable text, relevant papers, independent query families, lexical topic
  coverage, grounded notes/deep reads, and citation-graph health. They are
  stopping criteria for this documented protocol, not estimates of universal
  literature completeness.
- `hooks` — commands run on the events `task_green`, `blocked`, `run_complete`, and `spec_green`. `$SPIRAL_EVENT` and `$SPIRAL_INFO` are set in the environment.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Project knowledge

spiral takes project-specific guidance from four sources:

1. **Skills** — markdown files loaded per task when they match. Included:
   `android-kotlin`, `dark-ui-design`, `design-principles`, `dependency-medic`.
   Add your own in `<project>/.spiral/skills/`.
2. **Notes** — `spiral note "text"` appends to a file that is included in every
   worker prompt.
3. **Reused fixes** — when the escalation model solves something the worker
   could not, the error and the fix are appended to
   `.spiral/skills/learned-fixes.md` so the worker can reuse it on later runs.
4. **Signature routing** — error signatures the worker has never beaten skip
   its lane on later runs (see `spiral distill`).

For UI work (Android, iOS, web, desktop — detected from the repo and goal), a
**design brief** with concrete values (color tokens, type sizes, spacing,
motion, sample copy) is generated once per project, or written by hand at
`.spiral/design.md`, and included in every prompt. The brief is distilled into
`.spiral/design_tokens.json`, and for an Android app spiral draws a launcher
icon from those tokens and wires the manifest before feature work.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Run artifacts

Everything a run decides or learns is written to `.spiral/` in the target repo:

| file | contents |
|---|---|
| `plan.json` · `state.json` · `spec.json` | goal, task graph, run state, requirements + checks |
| `ledger.jsonl` | every model call and attempt: tokens, tok/s, edits, verify exit, error signature |
| `thoughts.jsonl` | visible decision log behind the pinned `thoughts` panel |
| `validation.json` · `route.json` | latest per-requirement verdicts · signature routing table |
| `design.md` · `design_tokens.json` | the design brief and its tokens (UI projects) |
| `product-audit.json` | deterministic scaffold/test/delivery/plot/simulation finish checks |
| `visual-review/` | three-viewport screenshots, DOM/runtime audits, manifests, and vision reports |
| `dependency-cache/` · `tools/` | dependency manifests/caches · inspected public repos with commit/license records |
| `skills/learned-fixes.md` | fixes distilled from escalation wins |
| `scratch/` | reasoning transcripts, last raw reply, last failure |

`spiral research --solve` writes its own run directory (`spiral-research/` by
default): `research-map.json`, `research-map.md`, and `research-graph.html`
show both the search/citation frontier and a switchable reasoning/obligation layer;
the HTML graph supports wheel zoom, drag pan, fit reset, search highlighting, and
double-click focus; `journal.md` records every round;
`thoughts.jsonl` and `model-calls.jsonl` preserve the explicit decision and model-call records;
`coverage-latest.json` records every corpus gate and its evidence;
`notes/papers/` holds cached per-paper reading notes; `notes/deep/` holds
zoomed-in notes for papers selected by idea families; `notes/idea-families-*.json`
records the candidate research-question families; `counterfactuals/` records
one-change neighboring hypotheses; `epistemic/` contains the obligation graph and
its hash-chained mutation log; `strategy/` contains information-gain and transparent
taste records; `.research-git/` stores private checkpoints; `certificates/` holds
environment-locked executable analyses, manifests, and blind replications;
`data/catalog.json`, `data/plans/`, `data/cache/`, and `data/runs/` hold catalog
results, preregistered contracts, hashed source data, and provenance manifests;
`novelty-boundary.json`
scopes the literature claim; `writeup/style-guide.md`,
`writeup/writing-blueprint.md`, notation/equation/vocabulary maps,
`writeup/paper-audit.json`, `body-latest.tex`, and `paper.tex` show how the final
paper was structured and checked. `writeup/proof-carrying-manifest.json` maps paper
claims and artifacts to hashes/evidence; `living-paper.json` defines refresh rules.
Failed writing attempts retain these artifacts
instead of disappearing behind a generic model error.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Principles

<img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/dot.svg" width="13" alt=""/> **Completion is verified, not claimed.** It is decided by exit codes, file existence, and the spec check. A passing build is not the same as an implemented feature.

<img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/dot.svg" width="13" alt=""/> **Progress is committed incrementally.** During repair, each reduction in error count is a commit, so an interrupted or failed run resumes from the last improvement.

<img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/dot.svg" width="13" alt=""/> **Lanes stop on lack of progress.** A task stops after three attempts that resolve no new error, rather than repeating the same failure to a fixed limit.

<img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/dot.svg" width="13" alt=""/> **The control loop is deterministic.** Models generate edits and plans; ordering, verification, retries, and git are handled in code. This is what lets small models complete large tasks.

<img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/dot.svg" width="13" alt=""/> **Missing output is surfaced.** Unjudged requirements are marked, truncated JSON is repaired, and empty commits are rejected.

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Extras

```bash
python -m spiral.banner --vortex        # animated banner
python experiments/sinks_test.py        # context-overflow test
python scripts/record_demo.py --dir .   # re-record assets/demo.gif from a real run (needs pyte)
```

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Requirements

- macOS on Apple Silicon (32 GB+ unified memory recommended) or Linux
- [Ollama](https://ollama.com) with at least one local model
- Python 3.11+ and git

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Roadmap

Language-server diagnostics as a fast gate between builds; spec-time dry runs
for acceptance checks (a check that passes before the work exists proves
nothing); interrupting a single attempt without stopping the run; parallel
tasks via git worktrees; an emulator launch gate for Android; a JSON event
stream and a CI action.

<p align="center"><img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/divider.svg" width="520" alt=""/></p>

<p align="center">
  MIT · Built by <b>Edis Devin Tireli</b> · Ph.D. Fellow, University of Copenhagen
</p>
