<p align="center">
  <img src="assets/spiral.gif" alt="spiral" width="680"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-D97757?style=flat-square" alt="MIT"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-D97757?style=flat-square" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/runs-100%25%20local-D97757?style=flat-square" alt="100% local"/>
  <img src="https://img.shields.io/badge/API%20cost-%240.00-D97757?style=flat-square" alt="$0.00"/>
</p>

<p align="center">
  <b>Give it a goal. Walk away. Come back to SPEC-GREEN.</b><br/>
  An autonomous coding agent that plans, designs, builds, verifies, and validates —
  entirely on your own hardware.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#sixty-seconds">Quickstart</a> ·
  <a href="#what-a-run-looks-like">Demo</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#commands">Commands</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#philosophy">Philosophy</a>
</p>

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

spiral is a local-first autonomous coding CLI. It decomposes your goal into a
spec, writes itself a design brief, plans against a critic's review, then grinds
tasks **green-to-green** — every change verified by your project's real build
gate, every green task a git commit, every failure banked as progress instead of
lost. At the end, a validator audits the code against every requirement and
keeps working until the spec is green. No API keys. No cloud. No babysitting.

## <img src="assets/mark.svg" width="21" alt=""/> Install

```bash
pip install spiral-coder
```

Requires Python 3.11+, [Ollama](https://ollama.com), and a capable local model
(Apple Silicon with 32 GB+ recommended — see [Requirements](#requirements)).

## <img src="assets/mark.svg" width="21" alt=""/> Sixty seconds

```bash
spiral doctor                     # is this machine ready?
spiral tune                       # once: calibrate context windows to your RAM (advisor asks first)
spiral build "make me a pomodoro TUI in python, with tests"
```

Then leave. spiral works on its own branch (`spiral/run-*` — yours is never
touched), commits every verified step, and ends with the receipts:

```
╭──────────────────────────── ⠷ run summary ────────────────────────────╮
│ 11/12 tasks green · 1 blocked · SPEC-GREEN                             │
│ Σ 174,787 tok (141k in / 33k out) · 34 attempts · 2 escalations · 57m  │
│ qwen3.6:latest · 28 gen · median 30 t/s                                │
│ ≈ $0.93 of cloud API · spent $0.00 · your hardware, your tokens        │
╰────────────────────────────────────────────────────────────────────────╯
```

## <img src="assets/mark.svg" width="21" alt=""/> What a run looks like

A pinned cockpit shows the plan with live markers while events scroll above it:

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

Broken projects get repaired before feature work — progress is **banked**, never
lost, even across runs:

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

And "done" is never taken on faith — a different model audits the code against
every extracted requirement, and gaps become new work:

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
       │                (concrete hexes,          (a different model
       │                 type scale, voice)        hunts plan defects)
       ▼
 bootstrap the gate to green            error-set ratchet: every resolved
       │                                error is a commit — marathons converge
       ▼
 grind tasks green-to-green             skills in-prompt · attempt memory ·
       │                                ASK protocol · symbol hunter ·
       │                                escalation lane · auto-distilled fixes
       ▼
 clean-build hygiene ──▶ chunked spec validation ──▶ remediation ──▶ ● SPEC-GREEN
```

**The tower of verification.** Each layer catches the lie the previous one
can't see: the build gate proves code compiles → artifact checks prove files
exist → behavior audits prove tasks did something → the spec validator proves
the *product* exists → your `extra_gate` proves whatever you demand.

**The five-layer context system.** Small local models can't reliably know what
they don't know, so context acquisition is mechanical, not metacognitive:
plan-time file assignment → **ASK** (`ASK: grep <name>` when referencing
something unseen) → error-path harvesting → the symbol hunter (repeated error →
repo facts injected) → escalation to the stronger model.

**The crew.** Every role is a swappable plug:

| role | default | job |
|---|---|---|
| worker / planner | `qwen3.6:latest` (MoE, ~3B active) | fast: plans, builds, repairs |
| escalation | `qwen3.6:27b` (dense) | takes over stuck tasks |
| critic / validator / designer | `qwen3.6:27b`, thinking | a different brain reviews plans, judges specs, writes design briefs |
| janitor | `llama3.2:1b` | compacts attempt history |

## <img src="assets/mark.svg" width="21" alt=""/> Commands

| command | what it does |
|---|---|
| `spiral build "goal"` | the full autonomous run: plan → build → validate → remediate |
| `spiral build --resume` | continue where a previous run stopped |
| `spiral build --approve` | show the plan, wait for your y/N before executing |
| `spiral plan "goal"` | decomposition only — see what it *would* do |
| `spiral validate` | judge existing code against the goal's spec (read-only) |
| `spiral do "task" --verify "cmd"` | one task, one gate — the minimal loop |
| `spiral tune` | calibrate context windows to this machine (KV math + advisor) |
| `spiral doctor` | readiness check: ollama, models, tune, gate, git, disk |
| `spiral stats` | ledger analytics: tokens, per-model t/s, outcomes |
| `spiral note "text"` | record project wisdom — injected into every worker prompt |
| `spiral rewind [n]` | list task checkpoints; roll the spiral branch back |
| `spiral research "query"` | GET-only web search + page reading |

## <img src="assets/mark.svg" width="21" alt=""/> Live controls

While a run is grinding:

- **⇧ Tab** — cycle `auto ↔ step`, live (shown in the status line)
- **step mode** — pauses at every task boundary: `enter` run · `s` skip · `a` back to auto · `q` stop cleanly
- **Ctrl-C** — always safe: green work is committed, banked checkpoints kept, `--resume` continues

## <img src="assets/mark.svg" width="21" alt=""/> Configuration

Models are plugs — swap per-shell or persistently:

```bash
export SPIRAL_WORKER=qwen3.6:latest
export SPIRAL_ESCALATION=qwen3.6:27b
export SPIRAL_BASE_URL=http://localhost:11434
```

`~/.config/spiral/config.json` (written by `spiral tune`, extended by you):

```json
{
  "models":     { "worker": "qwen3.6:latest", "critic": "qwen3.6:27b" },
  "num_ctx":    { "qwen3.6:latest": 28672, "qwen3.6:27b": 57344 },
  "extra_gate": "ktlint app/src",
  "hooks": {
    "run_complete": "osascript -e 'display notification \"$SPIRAL_INFO\" with title \"spiral\"'",
    "spec_green":   "say 'spec green'",
    "blocked":      "say 'spiral is stuck'"
  }
}
```

- **`extra_gate`** — your own command welded into every task's gate. If it exits
  non-zero, the task isn't done. Veto power.
- **`hooks`** — shell commands fired at `task_green` / `blocked` /
  `run_complete` / `spec_green`, with `$SPIRAL_EVENT` and `$SPIRAL_INFO` set.

## <img src="assets/mark.svg" width="21" alt=""/> Teaching spiral

spiral learns your project three ways:

1. **Skills** — markdown craft files loaded per-task when they match. Ships with
   `android-kotlin`, `dark-ui-design`, `design-principles`, `dependency-medic`.
   Drop your own in `<project>/.spiral/skills/`.
2. **Notes** — `spiral note "never bump compileSdk to fix code errors"` rides
   every prompt from then on.
3. **Auto-distillation** — when the escalation model solves what the fast lane
   couldn't, the error signature and winning fix are appended to
   `.spiral/skills/learned-fixes.md`. The expensive model teaches the cheap one,
   permanently.

Taste is specification: a **design brief** (exact palette tokens, type scale,
motion timings in ms, verbatim microcopy) is generated once per project — or
written by you at `.spiral/design.md` — and rides every prompt. The executor
implements decisions, not vibes.

## <img src="assets/mark.svg" width="21" alt=""/> Philosophy

Every rule below was bought with a real failure:

<img src="assets/dot.svg" width="13" alt=""/> **Never trust the model's opinion of "done."** Exit codes, artifacts, audits,
  and spec verdicts — a green gate is not a built feature.
<img src="assets/dot.svg" width="13" alt=""/> **Progress is banked.** Bootstrap repairs commit checkpoints per resolved
  error; a failed marathon resumes where it stopped.
<img src="assets/dot.svg" width="13" alt=""/> **Terminate on progress, not budgets.** A lane stops after 3 attempts without
  a newly-resolved error — never wastes 12 attempts saying the same thing.
<img src="assets/dot.svg" width="13" alt=""/> **The harness drives; models only generate.** That inversion is why ~3B active
  parameters can do this job at all.
<img src="assets/dot.svg" width="13" alt=""/> **Silence must never read as coverage.** Unjudged requirements are shouted,
  truncated JSON is salvaged, empty commits are rejected.

## <img src="assets/mark.svg" width="21" alt=""/> Extras

```bash
python -m spiral.banner --vortex     # you'll see
python experiments/sinks_test.py     # what survives context overflow?
```

## <img src="assets/mark.svg" width="21" alt=""/> Requirements

- macOS on Apple Silicon (32 GB+ unified memory recommended) or Linux
- [Ollama](https://ollama.com) with at least one strong local model
- Python 3.11+ · git

## <img src="assets/mark.svg" width="21" alt=""/> Roadmap

LSP fast-gate (instant diagnostics between full builds) · evidence-based lane
routing from the ledger · esc-to-interrupt mid-attempt · parallel tasks via git
worktrees · `--json` event stream + CI action.

<p align="center"><img src="assets/divider.svg" width="520" alt=""/></p>

<p align="center">
  MIT · Built by <b>Edis Devin Tireli</b> · Ph.D. Fellow, University of Copenhagen
</p>
