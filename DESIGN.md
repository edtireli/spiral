# spiral — design

A local-first, autonomous coding CLI. You hand it a scoped project; it plans,
writes, verifies, and commits in a loop until the scope's acceptance criteria are
green — or it hits a budget, checkpoints, and lets you resume. Modern agent-CLI
UX, hacker aesthetic. Targets a 32 GB Apple Silicon Mac against Ollama.

## The one principle that makes it work

A local 27B will confidently write broken code and then declare itself done.
So **spiral never trusts the model's opinion of "done."** Ground truth is the
compiler / linter / test suite. The engine is a tight loop:

    edit → run it → read the errors → fix → green → git commit → next task

The compiler is the intelligence that catches the model's mistakes. "Done" is
**executable acceptance criteria defined at scope time**, not vibes. Progress is
`criteria passed ÷ total`, a real number.

## Brains — one model, two modes

The smartest local brain does both jobs; we don't waste it on a 4B planner.

| Mode  | Model         | Thinking | Output            | Budget            |
|-------|---------------|----------|-------------------|-------------------|
| Plan  | qwen3.6:27b   | ON       | structured (JSON) | capped, must emit |
| Build | qwen3.6:27b   | OFF      | tool calls        | hard num_predict  |

Forcing plan-mode into a JSON schema is the fix for "thinks all its tokens away":
it must emit the schema and stop. Build-mode runs `/no_think` with a hard cap,
~5 tools, one task, and sees only the 2–3 relevant files.

**Backend is swappable, local-first.** Ollama today behind a thin seam; another
provider can slot in for a single hard step if ever wanted. Never required.
Later: `llama3.2:1b` as a janitor (compaction / done-checks); `qwen3:30b-a3b`
(MoE, 3B active = fast) as an optional quick planner.

## State on disk — `.spiral/` in the target repo

- `PLAN.md` — human-readable task graph
- `STATE.json` — scope %, current task, budgets spent (makes runs resumable)
- `criteria.json` — executable acceptance criteria = the definition of done
- `repomap.txt` — compressed symbol map (aider-style)
- `scratch/` — per-task notes + checkpoints

## Autonomy is safe because of git

Everything runs in a git-checkpointed workspace; every green task is a commit.
Walk away, come back to a clean log, revert any bad step. A denylist blocks
genuinely destructive shell ops even in full-auto mode.

## Budgets & resumption

Per-task token + attempt budgets; global token + wall-clock budget. On a
budget-hit the worker checkpoints to `scratch/` and the conductor re-issues a
continuation with a summary — fresh small context each leg, disk holds memory.
Kill anytime; `STATE.json` resumes across sessions.

## Failure modes → containment

| Failure mode                          | Containment                                            |
|---------------------------------------|--------------------------------------------------------|
| Bad decomposition derails the run     | plan in think-mode + structured output; revisable; human approves plan once |
| Declares "done" prematurely           | executable acceptance criteria; not done until green   |
| Loops / thrashes one file             | loop detection + per-task attempt budget → re-plan     |
| Drifts, forgets the goal              | goal + criteria pinned into every prompt               |
| Hallucinates APIs, writes broken code | the verify loop — run after every edit, feed errors back |
| Malformed edits (small models)        | search/replace edit format + fuzzy apply + validate every edit |
| Runs forever / unbounded cost         | global token + wall-clock + attempt budgets; resumable |

## Specialist roles (the crew)

One generalist model hits specialized failure classes; each gets a dedicated role
with its own mechanism, not a smarter generalist:

| Role | Brain | Job |
|---|---|---|
| conductor | a3b, think ON, schema-forced | decompose, reflect on own plan, re-plan |
| worker | a3b, think OFF, hard cap | edit→verify→fix against the gate |
| escalation | dense 27B, think OFF | one stuck task at a time, then back to a3b |
| **medic (v2)** | signature router + playbook skill + probe tools | dependency/toolchain hell |
| **researcher (v2)** | a3b + research.py (GET-only web) | live knowledge: current versions, API docs, unfamiliar errors |
| janitor (v2) | llama 1B | compaction, done-checks, log hygiene |

## The skill library (spiral/skills/ + <ws>/.spiral/skills/)

The on-demand skills mechanism: each skill is markdown with frontmatter
(name + trigger description) and a body of distilled craft — checklists, idioms,
playbooks. A frontier model authors them ONCE; local models apply them forever,
free. Loaded per-task only on match (skillpack.py: extension routes + keyword
overlap now; conductor-tags-tasks as model-router in v2), so worker context stays
lean. Seed pack: android-kotlin (cross-file coherence, TTS/siren/animation
patterns), dark-ui-design (surfaces, spacing, dystopian voice), dependency-medic
(error-signature → fix table, probe-first, pin-to-observed).

Skills are frozen knowledge; the researcher is live knowledge. research.py is the
ONLY door to the web (worker shell still denies curl/wget): GET-only, http(s)-only,
size-capped, tag-stripped. Fetched content is UNTRUSTED DATA — summarized into
briefs under .spiral/research/, fed to models as reference, never as instructions,
never executed. Consumers: conductor at plan time (unknown tech), medic (version
matrices it can't observe locally), escalation (stuck on an unfamiliar API).

Medic design: build errors have a nearly-closed vocabulary (`class file major
version N`, `AAR metadata check failed`, `Duplicate class`, `Could not resolve`).
Router classifies red gates: code errors → worker; dependency signatures → medic,
which probes ground truth FIRST (java -version, ls sdk dirs), matches signature →
playbook strategy, and pins to versions observed on the machine — never
hallucinated-latest. The playbook is distilled once by a frontier model into a
local artifact; local models apply it forever. Version-matrix knowledge is
tabular fact, not reasoning — exactly where local models hallucinate worst and
where blind edit-iteration is slowest (each attempt costs a full resolve).

## Hard-won invariants (each bought with a real failure)

- **Green-to-green:** the detected build gate is injected into every task; commits
  only on green; failed tasks revert to the last green commit. Presence checks
  (grep/ls) are not gates — they passed 8 tasks of non-compiling code once.
- **Ratchet before first green:** green-to-green semantics only exist AFTER a
  green baseline. Bootstrap starts red — there is no green to protect, only
  progress to keep. Every attempt that reduces the distinct-error count banks a
  checkpoint commit; failure reverts to the last checkpoint; progress compounds
  across attempts, model lanes, and runs. Learned the hard way: a 27B ground
  through an error stack for 4 attempts and a blanket revert threw it all away.
- **Progress-based termination, not fixed budgets, for marathons:** stop a lane
  after 3 attempts without error-count improvement (and after 3 consecutive
  unparseable replies). Fixed budgets either waste attempts on a stalled lane or
  starve a converging one — 12 identical no-parse failures once burned 48 minutes.
- **Planner fallback ladder:** think → think → think-OFF. Thinking mode can eat the
  entire num_predict and return empty content (the original think-forever disease,
  at conductor level). The last rung cannot ramble. Reflection failures fall back
  to the draft plan — an enhancement must never lose the run.
- **Bootstrap boundary:** the loop can't fix a gate that won't run. Humans/frontier
  provide a *launching* compiler (wrapper, JDK, sdk path); spiral owns every error
  the gate emits from then on — including making it pass the first time.
- **One wedge never deadlocks the run:** stuck → escalate to dense model → revert +
  mark blocked → continue. Blocked list is part of the final report.
- **A green gate is not a built feature:** the build proves that what EXISTS
  compiles — nothing more. The moment M0 went green, every feature task
  auto-passed as "already green" and a run declared 12/12 with zero features
  built (an APK that crashes on launch). Tasks complete only when the gate is
  green AND their declared artifacts exist AND (for edits to existing files) an
  audit attempt confirms the behavior — until the acceptance loop audits the
  whole spec.

## Phasing

- **v0 — the atom:** `edit→verify→fix→commit` on a single task. ✓ proven
- **v1 — conductor:** decompose → reflect → bootstrap-to-green → grind
  green-to-green with escalation; resumable state. ✓ built, under live test
- **v2 — the crew:** medic (dependency playbook), janitor (compaction), loop
  detection, adaptive re-planning mid-run, Textual dashboard.
- **v3 — polish:** parallel tasks via git worktrees, optional swap-in provider,
  milestone-boundary taste review (human or judge model), aesthetic pass.

## Honest limits

Shines at grinding a clearly-specified, test-backed implementation (scaffolding,
CRUD, refactors, ports, filling a defined API). Weak at architecture-heavy or
creative design — keep the human in the scoping loop. Only as honest as its
acceptance criteria.
