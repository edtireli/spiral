<p align="center">
  <img src="assets/spiral.gif" alt="spiral" width="680"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-D97757?style=flat-square" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-D97757?style=flat-square" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/backend-Ollama-D97757?style=flat-square" alt="Ollama"/>
  <img src="https://img.shields.io/badge/macOS%20%C2%B7%20Linux-D97757?style=flat-square" alt="macOS · Linux"/>
</p>

<p align="center">
  An autonomous coding agent that runs on local models. It turns a goal into a
  requirements checklist, implements each requirement against your project's
  real build or test command, and verifies the result before reporting done.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#a-run-from-the-inside">A run, from the inside</a> ·
  <a href="#what-counts-as-done">What counts as done</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#safety">Safety</a> ·
  <a href="#limits">Limits</a>
</p>

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

## <img src="assets/mark.svg" width="21" alt=""/> Why spiral exists

A local model will confidently write broken code and then declare itself done.
That single observation shapes everything here: **spiral never trusts the
model's opinion of "done."** Ground truth is your compiler, your test suite,
your program's actual exit codes. The engine is a tight loop —

    edit → run the gate → read the errors → fix → green → git commit → next task

— in which models only *generate*: edits, plans, verdicts. Ordering, verification,
retries, and git are handled by deterministic code. That division of labor is
what lets small local models (a ~3B-active MoE doing most of the work) finish
large projects: the model doesn't need to be right, it needs to be *checkable*.

Everything runs on your machine through [Ollama](https://ollama.com). No API
keys, no code leaving the room, no metered tokens. For unpublished research
code and anything you can't legally send to a cloud model, this isn't a
preference — it's the only kind of agent you're allowed to run.

## <img src="assets/mark.svg" width="21" alt=""/> Install

```bash
pipx install spiral-coder     # isolated, puts `spiral` on your PATH (recommended)
# or
pip install spiral-coder
```

Either way you get a global `spiral` command. From a clone, `pip install -e .`
installs it in editable mode.

Requirements:

- macOS on Apple Silicon (32 GB+ unified memory recommended for the default
  model set) or Linux; smaller machines can run a smaller crew
- [Ollama](https://ollama.com) and git
- Python 3.11+

First-run setup, once per machine:

```bash
spiral setup      # detect Ollama + installed models; offer a crew sized to your RAM
spiral tune       # size model context windows to your memory (KV-cache math)
spiral doctor     # health check: ollama, models, tuning, gate, git, disk
```

`setup` never downloads anything without a yes. `tune` matters more than it
looks: Ollama's default context window is 4,096 tokens *regardless of the
model*, which silently truncates long prompts — `tune` computes what your RAM
can actually hold and writes it to the config. `tune --wired` additionally
raises the macOS GPU wired-memory limit (sudo; reverts on reboot) so Ollama can
use more of your unified memory.

## <img src="assets/mark.svg" width="21" alt=""/> Quickstart

```bash
cd your-project
spiral build "make me a pomodoro TUI in python, with tests"
```

spiral works on a dedicated git branch (`spiral/run-*`), never on yours. Each
verified step is a commit; you merge when you're happy. Stop any time with
Ctrl-C — committed work is kept, and the same command with `--resume` continues
where it left off. When the run ends you get one card that tells the story:

```
╭──────────────────────────── ⠷ run summary ────────────────────────────╮
│ 11/12 tasks green · 1 blocked · SPEC-GREEN                             │
│ Σ 174,787 tok (141k in / 33k out) · 34 attempts · 2 escalations · 57m  │
│ qwen3.6:latest · 28 gen · median 30 t/s                                │
│ ≈ $0.93 of equivalent cloud API · $0.00 spent                          │
╰────────────────────────────────────────────────────────────────────────╯
```

Useful variants:

```bash
spiral plan "goal"            # show the decomposition without running it
spiral build --approve        # print the plan, wait for your yes, then run
spiral build --resume         # continue a previous run (goal is remembered)
spiral do "add a --json flag" --verify "python -m pytest -q"   # one task, one gate
```

## <img src="assets/mark.svg" width="21" alt=""/> A run, from the inside

This section walks through every phase of `spiral build`, with the output you
will actually see. The pipeline:

```
 spec extraction ──▶ design brief ──▶ plan ──▶ critic review ──▶ repair
       │             (UI projects only:         (a separate model
       │              tokens + a real icon)      reviews the plan)
       ▼
 bootstrap the gate to green            each resolved error is committed, so
       │                                repair converges across attempts
       ▼
 foundation: design system + launcher icon    (deterministic, for UI apps)
       │
       ▼
 implement tasks against the gate       matched skills · attempt memory ·
       │                                ASK protocol · symbol search ·
       │                                diversity round · escalation
       ▼
 clean build ──▶ spec validation ──▶ remediation ──▶ SPEC-GREEN
```

### 1. The goal becomes a checklist

An analyst pass extracts every concrete commitment from your goal into atomic
requirements — this is what "done" will be measured against. Where a
requirement can be verified by *running something*, the analyst attaches an
executable acceptance check: one shell command that exits 0 exactly when the
requirement is met. A deterministic lint discards checks that merely assert
files exist (`grep`, `ls`, `test -f` — a file existing proves nothing about
behavior) and anything the safety denylist refuses:

```
     check lint: R9: presence-style check dropped: grep -q Timer main.py
  ● spec: 12 requirements · 4 with executable checks · 1,842 tok
     R1 (feature): 25-minute work sessions alternate with 5-minute breaks
     R2 (feature, check): the timer state machine is covered by pytest tests
     R3 (quality): the TUI redraws without flicker
     ...
```

### 2. UI projects get a design brief first

If the product has a user interface (detected from the repo and the goal — a
CLI or library skips this), a design pass writes a concrete specification:
color tokens with hex values, a type scale, spacing rules, sample copy, motion
durations. It is generated once, saved to `.spiral/design.md` (write your own
there to override), and pinned into every prompt — so screens implement
*decisions*, not per-task improvisation. The brief is distilled to
`.spiral/design_tokens.json`, and for Android spiral deterministically draws a
launcher icon from those tokens and wires the manifest before feature work —
the app ships a real icon without a small model hand-writing adaptive-icon XML.

### 3. The plan is drafted, linted, and reviewed by a different brain

The planner (thinking mode, constrained to a JSON schema so it must emit a plan
and stop) decomposes the goal into milestones and small tasks, each touching at
most ~3 files. Two deterministic passes run before any model critique: a plan
lint (tasks that touch too many files, thin descriptions, shallow verify
commands, references to files nothing creates) and a coverage check (a
requirement whose distinctive terms appear in no task is probably forgotten).
Then a *different model* — the dense critic — reviews the plan against the
requirements and the repo, and the planner repairs what it finds. Model
diversity catches what self-review cannot.

```
  ● draft plan · 7 tasks · 3,120 tok
     lint: task 2.1 'Wire timer': edits 'ui/timer.py' which no repo file or earlier task provides.
  ● critic 1 (qwen3.6:27b): revise · 2 defects · 2,933 tok
     ✗ [task 2.1] references TimerWidget before any task creates it
  ● repaired → 8 tasks
```

### 4. Bootstrap: the gate must be green before features begin

spiral auto-detects your build gate — `./gradlew assembleDebug`, `npm run
test`, `cargo build`, `go build ./...`, `python -m pytest -q` — and if the
project starts red, repairs it first. Bootstrap uses **ratchet semantics**:
there is no green to protect yet, only progress to keep, so every attempt that
resolves error signatures is banked as a checkpoint commit. A failed attempt
reverts only to the last checkpoint; progress compounds across attempts, model
lanes, and even interrupted runs:

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

### 5. The grind: every task keeps the gate green

Each task runs in the atom loop: the worker sees the project vision, the task,
the relevant files, and the gate's current output; it replies with
SEARCH/REPLACE edit blocks; spiral applies them, re-runs the gate, and either
commits (green) or feeds the errors straight back. A task only commits green —
so integration debt cannot accumulate silently — and a failed task reverts to
the last green commit, so it can never poison the next one.

Context reaches the worker in stages, because local models reference things
they haven't seen and don't notice they're missing something:

1. the planner assigns relevant files to each task;
2. a static symbol index (types, members, layout-id → viewBinding class) rides
   the prompt, so the worker reads what exists instead of guessing;
3. file paths named in build errors are pulled into context automatically;
4. the worker can ask — `ASK: grep <name>` or `ASK: file <path>` — instead of
   inventing an identifier (two asks per task, and asks don't cost attempts);
5. a repeated identical error triggers a repo-wide symbol hunt that feeds the
   actual definitions back;
6. an attempt-memory list of what was already tried (compacted by a 1B janitor
   model when it grows long) prevents the synonym roulette.

### 6. Stuck tasks: diversity first, then escalation

When the worker lane exhausts its attempts on a red gate, spiral does not give
up serially — it fails in parallel directions. The **diversity round** samples
N fresh candidates (default 3) at spread temperatures from the same prompt and
lets the gate judge each one. Sampling is nearly free on local hardware, and
the gate is a deterministic judge that cannot be argued with:

```
  ⚄ diversity round — 3 candidates, the gate judges
  ○ candidate 1 (t=0.7): no edits parsed
  ● candidate 2 (t=1.0): TimerEngine.kt(fuzzy) · exit 1 · 4 sig(s)
  ✔ candidate 3 (t=1.3) is green — committed 9c41f2a
```

A green candidate completes the task. During bootstrap, the best red candidate
is banked as a checkpoint if it resolved errors. The round only runs while the
gate is red — on a green-but-incomplete gate a do-nothing candidate would
"win", which is precisely the false completion the audit exists to prevent.

If diversity doesn't land it either, the task **escalates** to the dense model
for a few attempts. If that fails too, the tree reverts, the task is recorded
as blocked, and the run continues — one wedge never deadlocks a whole run.
Blocked tasks are listed in the final report.

### 7. Validation: execution first, opinion second

"Plan complete" is a claim, not a result. After a clean build, spiral audits
the code against the requirement checklist. Requirements with an executable
acceptance check are judged **by running the check** — an exit code, not an
opinion. Only requirements without a usable check are judged by the validator
model (a different brain from the builder, instructed to trust only code — an
unreachable screen or an uncalled function does not count):

```
━━ validation 1 · 12 requirements · 4 by execution · qwen3.6:27b judges the rest ━━
  ✓ R2  acceptance check passed: python -m pytest tests/test_timer.py -q
  ✓ R4  activity_login.xml binds et_name; LoginActivity validates input
  ◐ R10 btnSend calls sendMessage(), but the scan is never triggered
  ✗ R7  acceptance check failed (exit 1): AssertionError: log file not written
  spec: 9/12 implemented · 2 partial · 1 missing · 0 unjudged
```

Every unmet requirement becomes a remediation task carrying the validator's
evidence — and if it had a failing acceptance check, **that check joins the
task's gate**, so the loop drives the actual criterion to green rather than a
proxy for it. Validate → remediate repeats while the gap count drops; it stops
at SPEC-GREEN, on a plateau, or at the round cap. A requirement the validator
returned no verdict for is surfaced as `unjudged` — silence never reads as
coverage.

### 8. The harness learns across runs

Every model call, attempt, and verdict is recorded in `.spiral/ledger.jsonl` —
the flight recorder. Three mechanisms feed it back:

- **Learned fixes** — when escalation solves something the worker could not,
  the error and the winning repair are appended to
  `.spiral/skills/learned-fixes.md`; the worker sees the recipe on later runs.
- **Signature routing** — each attempt records the (normalized: line numbers
  stripped) error signature it faced and whether it cleared it. A signature the
  worker has failed 3+ times and never beaten, which escalation *has* beaten,
  is routed straight to escalation on later runs, skipping the doomed attempts:

  ```
    ⇒ router: 2 known hard signature(s) will skip the worker lane
    ...
    ⇒ known hard signature — routing straight to the escalation lane
       e: MainActivity.kt Unresolved reference 'bindingScan'
  ```

- **`spiral distill`** — mines the ledger on demand: prints the per-signature
  table, writes `.spiral/route.json`, and appends newly-ruled hard signatures
  to learned-fixes. No model calls; everything comes from records.

The model weights never change. The system around them gets smarter with use.

## <img src="assets/mark.svg" width="21" alt=""/> What counts as done

No single check is the target — a task or requirement is complete only when it
passes every layer that applies, because each layer catches a failure class the
others miss:

| layer | catches |
|---|---|
| build / test gate (auto-detected, runs on every task) | code that doesn't compile or fails tests |
| footgun lint (welded into the gate) | patterns that compile fine and crash at runtime — `Handler()` with no Looper, `parseColor` without `#` |
| `extra_gate` (yours: a linter, a test suite) | whatever you decide must never regress |
| declared-artifact existence | "green gate" runs where the feature's files were never created |
| behavior audit + no-op rejection | edits that change nothing; ALREADY_DONE claims on unmet requirements; empty commits |
| executable acceptance checks | requirements that *run* — judged by exit code, not by any model |
| spec validation + remediation | implemented-but-unwired code, forgotten requirements, partial features |

The design premise behind the layering: a gate can be *gamed* even when it
can't be lied to, and the more optimization pressure the loop applies, the more
that matters. Layered independent checks make the degenerate pass — stub the
function, delete the failing caller — much harder than the honest one.

## <img src="assets/mark.svg" width="21" alt=""/> Safety

Unattended autonomy is only safe when the blast radius is bounded by
construction, not by trust:

- **Your branch is untouched.** Work happens on `spiral/run-*`; you merge.
  `spiral rewind` refuses to operate on any branch that isn't spiral's.
- **The shell denylist.** The model's shell refuses destructive operations even
  in full-auto — `rm -rf`, `sudo`, `mkfs`, `dd`, fork bombs, `shutdown`,
  `git reset --hard`, redirects into `/dev` or `~` — with exit 126. It also
  blocks `git push` (spiral can commit, never publish) and `curl`/`wget`: the
  worker has **no network egress**.
- **One door to the web.** Only the research module fetches: GET-only,
  http(s)-only, size-capped, tags stripped. Fetched content is treated as
  untrusted reference data — summarized into briefs, never executed, never
  followed as instructions.
- **Interrupts are safe.** Ctrl-C keeps all committed work and banked
  checkpoints; `--resume` continues. Budgets (per-task attempts, global tokens,
  verify timeouts) bound every loop, and lanes stop on *lack of progress*
  rather than grinding a fixed count of identical failures.

## <img src="assets/mark.svg" width="21" alt=""/> Models

Each role can be any Ollama model; defaults target a 32 GB Apple Silicon Mac:

| role | default | purpose |
|---|---|---|
| worker / planner | `qwen3.6:latest` (MoE, ~3B active) | plans (thinking on) and implements (thinking off) — same weights, zero swap cost |
| escalation | `qwen3.6:27b` (dense) | retries a task the worker could not finish |
| critic / validator / designer | `qwen3.6:27b`, thinking | reviews the plan, judges the spec, writes the design brief — deliberately a different model than the worker |
| janitor | `llama3.2:1b` | compacts attempt history so prompts stay short |

`spiral setup` offers a crew sized to your RAM (7B-class on small machines up
to 32B-class on large ones) and pulls it with your consent. Models are kept
resident (`keep_alive: 45m`) so a mid-run reload never costs you a minute, and
spiral explicitly evicts one large model before loading another rather than
letting two thrash under memory pressure.

## <img src="assets/mark.svg" width="21" alt=""/> Commands

| command | description |
|---|---|
| `spiral build "goal"` | plan, implement, validate, and remediate — the full run |
| `spiral build --resume` | continue a previous run (the goal is remembered) |
| `spiral build --approve` | print the plan and wait for confirmation before running |
| `spiral build --boost` | keep the worker local; run the reasoning roles (escalation, critic/validator) on the configured API provider |
| `spiral build --api` | run the entire crew on the configured API provider |
| `spiral plan "goal"` | show the decomposition without running it |
| `spiral validate` | judge the current code against the goal's spec (read-only) |
| `spiral do "task" --verify "cmd"` | drive a single task against one verify command |
| `spiral setup` | first run: detect Ollama, offer a RAM-matched model crew |
| `spiral tune` | size model context windows to this machine's memory |
| `spiral tune --wired` | also raise the macOS GPU wired-memory limit (sudo; reverts on reboot) |
| `spiral doctor` | health check: ollama, models, tuning, gate, git, disk |
| `spiral stats` | tokens, per-model throughput, and outcomes from the ledger |
| `spiral distill` | mine the ledger: signature routing table + new learned-fixes entries |
| `spiral note "text"` | record project wisdom the workers will always see |
| `spiral rewind [n]` | list task checkpoints; reset the spiral branch to one |
| `spiral search "query"` | fast ranked web results, no synthesis (`--sci` adds arXiv) |
| `spiral research "query"` | gather web/arXiv/PubMed sources, synthesize a cited answer (`--deep`, `--sci`) |
| `spiral chat ["message"]` | talk to the local thinking model; reasoning shown dimmed |
| `spiral consult ["question"]` | send the whole project to a big-context API model for review |
| `spiral style [name]` | set the banner shape: `spiral`, `galaxy`, or `uzumaki` |

## <img src="assets/mark.svg" width="21" alt=""/> Live controls

During a run:

- **⇧ Tab** — cycle between `auto` and `step` mode, live; the current mode is shown in the status line.
- **step mode** — pause at every task boundary: `enter` run · `s` skip · `a` back to auto · `q` stop.
- **Ctrl-C** — stops cleanly; committed work and banked checkpoints are kept; `--resume` continues.

A pinned panel shows the plan and current position while log lines scroll
above it; the status line carries live tokens/s and an ETA. Piped or
backgrounded output degrades to timestamped heartbeat lines — a log must never
look dead.

## <img src="assets/mark.svg" width="21" alt=""/> Configuration

Environment variables override per shell:

```bash
export SPIRAL_WORKER=qwen3.6:latest       # also: SPIRAL_PLANNER, SPIRAL_ESCALATION,
export SPIRAL_ESCALATION=qwen3.6:27b      #       SPIRAL_CRITIC, SPIRAL_JANITOR
export SPIRAL_BASE_URL=http://localhost:11434
export SPIRAL_STYLE=galaxy
```

`~/.config/spiral/config.json` is written by `spiral setup` / `spiral tune`
and can be edited directly. Every supported key:

```json
{
  "models":     { "worker": "qwen3.6:latest", "escalation": "qwen3.6:27b",
                  "critic": "qwen3.6:27b", "planner": "qwen3.6:latest",
                  "janitor": "llama3.2:1b" },
  "num_ctx":    { "qwen3.6:latest": 28672, "qwen3.6:27b": 57344 },
  "worker_max_tokens": 8192,
  "extra_gate": "ktlint app/src",
  "diversity_samples": 3,
  "style": "spiral",
  "base_url": "http://localhost:11434",
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

- `num_ctx` — per-model context windows; write these with `spiral tune`, not by
  guessing (an unset window silently truncates prompts at 4,096 tokens).
- `worker_max_tokens` — the per-reply output cap (`num_predict`). A ceiling,
  not a target; spiral raises it temporarily when a reply truncates mid-block.
- `extra_gate` — a command appended to every task's gate. Non-zero exit means
  the task is not complete. This is your veto.
- `diversity_samples` — candidates in the best-of-N diversity round (default 3,
  max 5, 0 disables).
- `providers` — OpenAI-compatible endpoints, keyed by model id. Any role set to
  one of these ids is served by that endpoint instead of Ollama. The API key is
  read from the environment variable named in `api_key_env` and is never
  written to disk. `spiral build --boost` remaps escalation + critic/validator
  onto the first provider while the worker stays local; `--api` remaps the
  whole crew. Without these flags everything runs local — the default never
  needs a key.
- `hooks` — shell commands fired on `task_green`, `blocked`, `run_complete`,
  and `spec_green`, with `$SPIRAL_EVENT` and `$SPIRAL_INFO` set. Notifications,
  says, CI pings — your call; hooks can never break the run.

## <img src="assets/mark.svg" width="21" alt=""/> Project knowledge

Three ways to teach spiral your project, all persistent:

1. **Skills** — markdown files with a name + trigger description, loaded into
   the prompt only when a task matches. Four ship built-in (`android-kotlin`,
   `dark-ui-design`, `design-principles`, `dependency-medic`); add your own
   under `<project>/.spiral/skills/`. A frontier model can author a skill once;
   local models apply it forever, free.
2. **Notes** — `spiral note "we use Result<T>, never exceptions"` appends to a
   skill that rides *every* worker prompt.
3. **Learned fixes + routing** — automatic, from the ledger (see
   [the learning section](#8-the-harness-learns-across-runs)).

For UI projects, `.spiral/design.md` is the taste file: write it yourself and
spiral will implement your design decisions instead of generating a brief.

## <img src="assets/mark.svg" width="21" alt=""/> Run artifacts

Everything a run decides or learns lives in `.spiral/` in the target repo —
plain files, made to be read:

| file | contents |
|---|---|
| `plan.json` · `state.json` | the goal + task graph; live run state (`--resume` reads both) |
| `spec.json` | the requirements checklist, including acceptance checks |
| `plan_reviews.json` | critic verdicts and defects, per round |
| `ledger.jsonl` | the flight recorder — one line per model call, attempt, check, verdict: model, tokens, tok/s, edits, verify exit, error signature |
| `route.json` | per-signature routing verdicts (written by `spiral distill`) |
| `validation.json` | the latest per-requirement verdicts |
| `design.md` · `design_tokens.json` | the design brief and its distilled tokens (UI projects) |
| `skills/learned-fixes.md` · `skills/project-notes.md` | distilled escalation wins · your notes |
| `scratch/` | reasoning transcripts (`thinking-*.txt`), the last raw reply, the last failure |
| `consult.md` | the last whole-project consult report |

`spiral stats` summarizes the ledger; reading `ledger.jsonl` directly is the
fastest way to see exactly what a run did and why.

## <img src="assets/mark.svg" width="21" alt=""/> Limits

Honesty about the boundary is part of the design. spiral shines at grinding a
clearly-specified, verifiable implementation: scaffolding, CRUD, refactors,
ports, filling in a defined API, test-backed features. It is weak at
architecture-heavy or genuinely creative design work — keep yourself in the
scoping loop (`--approve` exists for exactly this). And the system is only as
honest as its acceptance criteria: where no check or test can exist, the final
verdict is still a model's judgment of code, which is an opinion. The plan
approval prompt is the one gate a human operates — it is the check on *intent*,
and no amount of machinery below it can substitute.

Wall-clock is the real cost of local autonomy. Tokens are free; an hour is an
hour. spiral spends attempts generously because they cost nothing but time —
schedule long runs accordingly (overnight is this tool's natural habitat).

## <img src="assets/mark.svg" width="21" alt=""/> Roadmap

Language-server diagnostics as a fast gate between builds; spec-time
falsifiability runs for acceptance checks (a check that passes before the work
exists proves nothing); interrupting a single attempt without stopping the run;
parallel tasks via git worktrees; an emulator launch gate for Android (the app
must start, not just compile); a JSON event stream and a CI action.

## <img src="assets/mark.svg" width="21" alt=""/> Extras

```bash
python -m spiral.banner --vortex     # animated banner
python experiments/sinks_test.py     # context-overflow test
```

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

<p align="center">
  MIT · Built by <b>Edis Devin Tireli</b> · Ph.D. Fellow, University of Copenhagen
</p>
