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

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

spiral is a command-line coding agent that runs entirely on local models through
[Ollama](https://ollama.com). Given a goal, it extracts a list of requirements,
plans the work, and implements each task against your project's actual build or
test command. Changes that pass are committed to git; changes that fail are
reverted. When the plan is finished, a separate model checks the code against
each requirement, and spiral keeps working until they are all met or it reports
which ones remain. No API keys and no network calls to a model provider.

## <img src="assets/mark.svg" width="21" alt=""/> Install

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

## <img src="assets/mark.svg" width="21" alt=""/> Quickstart

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

## <img src="assets/mark.svg" width="21" alt=""/> Example run

During a run, a status panel shows the plan and current position while log
lines scroll above it:

```
  ✔ committed f6e840f
  — attempt 2/6 · qwen3.6:latest —
╭──────────────── ⠷ plan ────────────────╮
│  ✓ M0 make the build gate pass         │
│  ◆ M1 Login screen  2/3                │
│    ✓ 1.1 Create LoginActivity          │
│    ▶ 1.2 Wire registration             │
│    ○ 1.3 Update manifest               │
│  3/7 green · 0 blocked · Σ 52.4k tok   │
│  · 14m elapsed · eta ~9m               │
╰─────────────────────────────────────────╯
 ⠹ spiral · building [qwen3.6:latest] · 48.2k tok · 12s · 30 t/s · auto ⇧⇥
    ✎ app/src/main/java/LoginActivity.kt
```

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

After the plan finishes, a separate model checks the code against each
requirement. Requirements that are missing or incomplete become new tasks:

```
━━ validation 1 · qwen3.6:27b judges code vs 27 requirements ━━
  ✓ R4  activity_login.xml binds et_name; LoginActivity validates input
  ✓ R15 TtsHelper sets STREAM volume to max via AudioManager
  ◐ R10 btnSend calls sendMessage(), but the scan is never triggered
  ✗ R14 no siren playback found anywhere in the code
  spec: 16/27 implemented · 8 partial · 1 missing · 2 unjudged
▶ V.1 implement R14 …
```

## <img src="assets/mark.svg" width="21" alt=""/> How it works

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
       │                                escalation · reused past fixes
       ▼
 clean build ──▶ spec validation ──▶ remediation ──▶ SPEC-GREEN
```

**Verification.** A task is complete only when it passes every check that
applies: the build or test gate (compiles, tests pass), a footgun lint welded
into that gate (deterministic patterns that compile fine but crash at runtime —
a `Handler()` with no Looper, `parseColor` without `#`), an artifact check (the
files the task declared exist), a behavior audit (the task actually changed
something relevant), and the final spec validation (the requirement is
implemented). You can append your own check with `extra_gate`.

**Diversity.** When the worker lane exhausts its attempts, spiral samples N
fresh candidates (default 3) at spread temperatures from the same prompt and
lets the gate judge each one — sampling is nearly free on local hardware, and
the gate is a deterministic judge that cannot be argued with. A green candidate
is committed and completes the task; during bootstrap the best red candidate is
banked as a checkpoint when it resolves errors. This runs only while the gate is
red, so a no-op candidate can never fake a win. `diversity_samples` in the
config sizes N (0 disables).

**Containment.** Runs happen on a dedicated `spiral/run-*` branch, and the
model's shell refuses genuinely destructive operations even in full-auto:
`rm -rf`, `sudo`, `git push`, `git reset --hard`, and disk or network commands
(`mkfs`, `dd`, `curl`, `wget`) are blocked by a denylist and fail with exit 126.
The only door to the network is the research module — GET-only, size-capped,
and fetched content is treated as reference data, never as instructions.

**Context acquisition.** Local models often reference identifiers or files they
have not been shown, and do not reliably notice when they are missing something.
spiral supplies context in stages rather than relying on the model to ask:

1. the planner assigns relevant files to each task;
2. the worker can request more with `ASK: grep <name>` or `ASK: file <path>`;
3. file paths named in build errors are added to the prompt automatically;
4. a repeated identical error triggers a repo search that returns the real
   definitions;
5. a task that exhausts its attempts escalates to the stronger model.

**Models.** Each role can be set to any Ollama model:

| role | default | purpose |
|---|---|---|
| worker / planner | `qwen3.6:latest` (MoE, ~3B active) | plans and implements tasks |
| escalation | `qwen3.6:27b` (dense) | retries a task the worker could not finish |
| critic / validator / designer | `qwen3.6:27b`, thinking | reviews the plan, validates the spec, writes the design brief; runs on a different model than the worker |
| janitor | `llama3.2:1b` | summarizes attempt history to keep prompts short |

## <img src="assets/mark.svg" width="21" alt=""/> Commands

| command | description |
|---|---|
| `spiral build "goal"` | plan, implement, validate, and remediate |
| `spiral build --resume` | continue a previous run |
| `spiral build --approve` | print the plan and wait for confirmation before running |
| `spiral build --boost` | keep the worker local; run the reasoning roles (escalation, critic/validator) on the configured API provider |
| `spiral build --api` | run the entire crew on the configured API provider |
| `spiral plan "goal"` | show the decomposition without running it |
| `spiral validate` | check existing code against the goal's spec (read-only) |
| `spiral do "task" --verify "cmd"` | run a single task against one verify command |
| `spiral setup` | detect Ollama and pull a model crew sized to this machine |
| `spiral tune` | size model context windows to available memory |
| `spiral tune --wired` | also raise the macOS GPU wired-memory limit so Ollama can use more unified RAM (sudo; reverts on reboot) |
| `spiral doctor` | check Ollama, models, tuning, gate, git, and disk |
| `spiral stats` | token counts, per-model throughput, and outcomes from the run log |
| `spiral note "text"` | add a note that is included in every worker prompt |
| `spiral rewind [n]` | list task checkpoints and reset the run branch to one |
| `spiral style [name]` | set the banner shape: `spiral`, `galaxy`, or `uzumaki` |
| `spiral search "query"` | fast ranked web results, no synthesis (`--sci` adds arXiv) |
| `spiral research "query"` | gather web/arXiv/PubMed sources and synthesize a cited answer (`--deep`, `--sci`) |
| `spiral chat ["message"]` | talk to the local thinking model; reasoning shown dimmed |
| `spiral consult ["question"]` | send the whole project to a big-context API model for review |

## <img src="assets/mark.svg" width="21" alt=""/> Live controls

During a run:

- **⇧ Tab** — switch between `auto` and `step` mode; the current mode is shown in the status line.
- **step mode** — pause at each task: `enter` to run it, `s` to skip, `a` to return to auto, `q` to stop.
- **Ctrl-C** — stops cleanly. Committed work is kept and `--resume` continues from there.

## <img src="assets/mark.svg" width="21" alt=""/> Configuration

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
  "models":     { "worker": "qwen3.6:latest", "critic": "qwen3.6:27b" },
  "num_ctx":    { "qwen3.6:latest": 28672, "qwen3.6:27b": 57344 },
  "extra_gate": "ktlint app/src",
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
- `diversity_samples` — candidates sampled in the best-of-N diversity round at the worker lane's exit (default 3, max 5, 0 disables).
- `providers` — OpenAI-compatible endpoints, keyed by model id. Any role set to
  one of these ids is served by that endpoint instead of Ollama. The API key is
  read from the environment variable named in `api_key_env` and is never written
  to disk. `spiral build --boost` remaps the reasoning roles (escalation,
  critic/validator) onto the first provider while the worker stays local;
  `--api` remaps the whole crew. Without these flags everything runs local —
  the default never needs a key.
- `hooks` — commands run on the events `task_green`, `blocked`, `run_complete`, and `spec_green`. `$SPIRAL_EVENT` and `$SPIRAL_INFO` are set in the environment.

## <img src="assets/mark.svg" width="21" alt=""/> Project knowledge

spiral takes project-specific guidance from three sources:

1. **Skills** — markdown files loaded per task when they match. Included:
   `android-kotlin`, `dark-ui-design`, `design-principles`, `dependency-medic`.
   Add your own in `<project>/.spiral/skills/`.
2. **Notes** — `spiral note "text"` appends to a file that is included in every
   worker prompt.
3. **Reused fixes** — when the escalation model solves something the worker
   could not, the error and the fix are appended to
   `.spiral/skills/learned-fixes.md` so the worker can reuse it on later runs.

For UI work (Android, iOS, web, desktop — detected from the repo and goal; a
CLI, TUI, or library skips it), a **design brief** with concrete values (color
tokens, type sizes, spacing, motion durations, sample copy) is generated once
per project, or written by hand at `.spiral/design.md`, and included in every
prompt so implementation follows fixed decisions instead of choices made per
task. The brief is distilled into `.spiral/design_tokens.json`, and for an
Android app spiral draws a **launcher icon** from those tokens and wires the
manifest before feature work — so the app ships a real icon, not the stock
robot, without a small model having to hand-write adaptive-icon XML.

## <img src="assets/mark.svg" width="21" alt=""/> Run artifacts

Everything a run learns or decides is written to `.spiral/` in the target repo:

| file | contents |
|---|---|
| `plan.json` | the goal and task graph; `--resume` and goal reuse read it |
| `ledger.jsonl` | the flight recorder — one line per model call and attempt: model, tokens, tok/s, edits, verify exit, verdicts |
| `scratch/thinking-*.txt` | reasoning transcripts from the plan, critic, and validation calls |
| `validation.json` | the latest spec verdict for each requirement |
| `design.md` · `design_tokens.json` | the design brief and its distilled tokens (UI projects) |
| `skills/learned-fixes.md` | fixes the escalation model found, reused by the worker on later runs |

`spiral stats` summarizes the ledger; reading `ledger.jsonl` directly is the
fastest way to see exactly what a run did and why.

## <img src="assets/mark.svg" width="21" alt=""/> Principles

<img src="assets/dot.svg" width="13" alt=""/> **Completion is verified, not claimed.** It is decided by exit codes, file existence, and the spec check. A passing build is not the same as an implemented feature.

<img src="assets/dot.svg" width="13" alt=""/> **Progress is committed incrementally.** During repair, each reduction in error count is a commit, so an interrupted or failed run resumes from the last improvement.

<img src="assets/dot.svg" width="13" alt=""/> **Lanes stop on lack of progress.** A task stops after three attempts that resolve no new error, rather than repeating the same failure to a fixed limit.

<img src="assets/dot.svg" width="13" alt=""/> **The control loop is deterministic.** Models generate edits and plans; ordering, verification, retries, and git are handled in code. This is what lets small models complete large tasks.

<img src="assets/dot.svg" width="13" alt=""/> **Missing output is surfaced.** Unjudged requirements are marked, truncated JSON is repaired, and empty commits are rejected.

## <img src="assets/mark.svg" width="21" alt=""/> Extras

```bash
python -m spiral.banner --vortex     # animated banner
python experiments/sinks_test.py     # context-overflow test
```

## <img src="assets/mark.svg" width="21" alt=""/> Requirements

- macOS on Apple Silicon (32 GB+ unified memory recommended) or Linux
- [Ollama](https://ollama.com) with at least one local model
- Python 3.11+ and git

## <img src="assets/mark.svg" width="21" alt=""/> Roadmap

Language-server diagnostics as a fast gate between builds; model routing based on
the run log; interrupting a single attempt without stopping the run; parallel
tasks via git worktrees; a JSON event stream and a CI action.

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

<p align="center">
  MIT · Built by <b>Edis Devin Tireli</b> · Ph.D. Fellow, University of Copenhagen
</p>
