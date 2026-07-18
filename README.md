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
 clean build ──▶ spec validation ──▶ remediation ──▶ SPEC-GREEN
```

**Verification.** A task is complete only when it passes every check that
applies: the build or test gate (compiles, tests pass), a footgun lint welded
into that gate (patterns that compile but crash at runtime), an artifact check
(the files the task declared exist), a behavior audit (the task actually
changed something relevant), and the final spec validation (the requirement is
implemented). You can append your own check with `extra_gate`.

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
4. the worker can request more with `ASK: grep <name>` or `ASK: file <path>`;
5. a repeated identical error triggers a repo search that returns the real definitions;
6. a task that exhausts its attempts escalates to the stronger model.

**Learning across runs.** Every attempt is logged to `.spiral/ledger.jsonl`
with the error signature it faced (normalized, so line numbers don't split
them) and whether it cleared it. When escalation solves something the worker
could not, the fix is appended to `learned-fixes.md` for later runs. A
signature the worker has repeatedly failed and only escalation has solved is
routed straight to escalation next time. `spiral distill` prints the table and
writes `.spiral/route.json`.

**Safety.** Work happens on a `spiral/run-*` branch; you merge. The model's
shell blocks destructive commands even in full-auto — `rm -rf`, `sudo`,
`mkfs`, `dd`, `git push`, `git reset --hard` — and blocks `curl`/`wget`, so
the worker has no network access. The only door to the web is the research
module: GET-only, size-capped, fetched content treated as data, never as
instructions. Ctrl-C stops cleanly; committed work is kept and `--resume`
continues.

**Models.** Each role can be set to any Ollama model:

| role | default | purpose |
|---|---|---|
| worker / planner | `qwen3.6:latest` (MoE, ~3B active) | plans and implements tasks |
| escalation | `qwen3.6:27b` (dense) | retries a task the worker could not finish |
| critic / validator / designer | `qwen3.6:27b`, thinking | reviews the plan, validates the spec, writes the design brief; runs on a different model than the worker |
| janitor | `llama3.2:1b` | summarizes attempt history to keep prompts short |

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Commands

| command | description |
|---|---|
| `spiral build "goal"` | plan, implement, validate, and remediate |
| `spiral build --resume` | continue a previous run |
| `spiral build --approve` | print the plan and wait for confirmation before running |
| `spiral build --boost` | local worker; escalation and critic/validator on the configured API provider |
| `spiral build --api` | run the entire crew on the configured API provider |
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
| `spiral chat ["message"]` | talk to the local thinking model; reasoning shown dimmed |
| `spiral consult ["question"]` | send the whole project to a big-context API model for review |

## <img src="https://raw.githubusercontent.com/edtireli/spiral/main/assets/mark.svg" width="21" alt=""/> Live controls

During a run:

- **⇧ Tab** — switch between `auto` and `step` mode; the current mode is shown in the status line.
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
  "models":     { "worker": "qwen3.6:latest", "critic": "qwen3.6:27b" },
  "num_ctx":    { "qwen3.6:latest": 28672, "qwen3.6:27b": 57344 },
  "extra_gate": "ktlint app/src",
  "diversity_samples": 3,
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
| `validation.json` · `route.json` | latest per-requirement verdicts · signature routing table |
| `design.md` · `design_tokens.json` | the design brief and its tokens (UI projects) |
| `skills/learned-fixes.md` | fixes distilled from escalation wins |
| `scratch/` | reasoning transcripts, last raw reply, last failure |

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
