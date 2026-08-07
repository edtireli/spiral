# Hardening spiral

A plan for making the builder robust enough to hand any project and trust the
outcome. Everything in here is grounded in defects actually observed on
2026-07-26 — one day of adversarial testing against a single FastAPI goal and a
single calculator goal found seven blocking defects and a dozen structural gaps,
which is the strongest possible argument that the next day of testing will find
more. The plan's spine is the project's own thesis, applied without mercy to the
project itself:

> Everything unverified will eventually be wrong, including the verifiers.

## What one day of testing proved

Worth recording, because each of these is a *class*, not an incident:

1. **The gate itself can be the bug.** `((…))` composed by `_refresh_gate` was
   arithmetic, not a subshell — every run aborted at bootstrap. The published
   0.1.7 still ships this. A gate that cannot pass is indistinguishable, to the
   loop, from code that is truly broken: the harness spent full budgets "fixing"
   healthy code.
2. **The sandbox can starve the gate.** Denying `stat` on `$HOME` broke pytest's
   rootdir walk (`Operation not permitted: /Users/edt/Desktop`); denying
   spiral's own interpreter returned exit 127. Both made the gate permanently
   red for reasons no edit could address.
3. **The harness can trip its own guards.** The dependency-cache venv's
   `bin/python` symlinks crashed the escaping-symlink check with an uncaught
   RuntimeError; tracked `__pycache__` made `_dirty()` refuse every transaction;
   redirecting the build log into the workspace did the same.
4. **Green can be vacuous.** `pytest -q || [ $? -eq 5 ]` called a non-importable
   application green for two committed milestones. The artifact gate called a
   page whose `<script>` does not parse "9 verified, 0 errors".
5. **Wording can be satisfied while behaviour regresses.** A remediation task
   titled "handle division by zero with an inline error state" cited the
   requirement in a comment, passed every gate, and made the display show
   `null`. Only clicking the buttons caught it.
6. **Paraphrase drift is enforced drift.** The spec extractor turned "index.html
   plus its own CSS and JS" into "a single index.html containing all HTML, CSS
   and JavaScript" — and the coverage machinery *enforced the paraphrase*,
   inlining 686 duplicate lines.
7. **Unvalidated classification steers the work.** A test suite labelled
   `kind: web, visual: true` made delivery demand visual evidence a test runner
   cannot have, and made the acceptance milestone build an HTML test page
   instead of a runnable suite.
8. **The verifiers were wrong too.** The first bench rubric accepted the word
   "undefined" as proof of an error message (rewarding the defect it existed to
   catch); the first runtime probe silently no-op'd on a browser-version
   mismatch; the first fingerprint change broke resume for every in-flight run;
   the first walk-back search skipped over the good commit. All four were found
   by tests, not by reading.

## Principles the plan enforces

- **A check that cannot fail is not a check.** Every rung, audit, and probe must
  have a demonstrated failing case in the test suite. (The exit-5 tolerance is
  the canonical violation: it made the test rung unfailable on any testless
  project.)
- **A check that cannot run must say so, loudly, in the report.** Silent
  exclusion reads as a pass. The bench prints EXCLUDED rows and reduced
  coverage; the runtime probe writes its note into the audit. Everything else
  must follow.
- **Deterministic before model, always.** Any repair that can be a computation
  (revert of a bisected commit, a dependency declaration, a parse rejection at
  edit time) must run before a model gets to spend tokens on it.
- **One privileged path per side effect.** Dependency acquisition goes through
  provisioning; nothing else installs. Edits go through `apply_edits`; nothing
  else writes source. Every new capability that wants a side effect must reuse
  the existing path or argue in review why not.
- **Every guard added to the harness gets the harness's own treatment**: a test
  that shows it firing, a test that shows it *not* firing on legitimate input,
  and a printed trace when it acts.

## The plan

### Tier 0 — before anything else (hours)

- [ ] Commit the working tree. Sixteen fixes and ten modules exist only as
      uncommitted changes; a `git checkout` away from vanishing.
- [x] Reinstall the CLI editable so the machine runs the fixed code.
- [ ] Bump the version. 0.1.7 is published broken; the number must move so
      `doctor` and upgrade tooling can tell users. Do NOT publish until the
      bench passes at threshold (below).
- [ ] Delete or quarantine the published 0.1.7 wheel's known-fatal path with a
      release note, whenever the next publish happens.

### Tier 1 — close the observed classes completely (days)

Each item generalises a defect found today from the instance to the class.

- [ ] **Gate self-test.** Before the first task of every run, execute the
      composed gate against a known-trivial fixture (empty temp project) and
      against `true`/`false` substitution. A gate that cannot go green on
      trivial input, or that reports shell parse errors, aborts the run with
      "the gate is broken" — never "the code is broken". This turns defect
      class 1 and 2 from budget-burners into instant, correctly-attributed
      failures.
- [ ] **Sandbox conformance suite.** One test per starved capability observed:
      stat on ancestors, exec of spiral's interpreter, read of the package
      tree, write to TMPDIR, localhost bind while network is denied. Run it in
      CI on macOS; on Linux, the bwrap equivalents.
- [ ] **Ladder parity for every ecosystem the gate detector claims.** Python
      has parse→load→run→test; web has script-parse and the runtime probe. The
      other ~20 ecosystems in `_detect_gate_here` still have single-command
      gates with unknown vacuousness. Minimum bar per ecosystem: a "can it
      possibly fail on an empty project" test, and a load/smoke rung where the
      runtime allows one (node `--check`+import, `cargo check`, `go vet`,
      gradle `compileDebugKotlin` before assemble).
- [ ] **Spec fidelity: requirements carry their source span.** Each requirement
      stores the goal text it derives from; a deterministic containment check
      flags requirements whose key nouns/constraints do not appear in the span
      ("single file" vs "plus its own CSS and JS"). Flagged requirements are
      demoted to advisory — they cannot drive coverage tasks until a human or a
      different-family model confirms them. This is the fix for defect class 6,
      and it is cheap.
- [ ] **Deliverable manifest validation, second half.** `sanitize_deliverables`
      now reconciles kind/visual/globs with descriptions. Remaining: validate
      `primary_id` semantics, and cross-check the manifest against the plan
      (a deliverable no task's files could produce is a plan defect).
- [ ] **Dedupe remediation batches by fault, not by evidence string.** Done for
      the runtime probe; audit the other issue producers for the same
      multiplication.

### Tier 2 — make quality the default, not the demand (days)

- [ ] **Promote the design foundation to every UI ecosystem.** Web now gets
      AA-by-construction tokens + favicon; Android had icon + palette. Missing:
      desktop (Tk/Qt stylesheet equivalent), and a "foundation brief" line in
      every worker prompt for those ecosystems (web has it; verify Android's
      still fires after the refactor).
- [ ] **Quality floor for non-web surfaces.** `quality.py` covers web files
      only. Android XML has the same computable floor: hardcoded colours vs
      `@color/`, missing `contentDescription`, missing pressed/focused state
      lists, dp/sp misuse. Same for a future desktop pack.
- [ ] **Hazard library growth.** The runtime probe knows calculator hazards
      (divide-by-zero, double operator, double decimal). Generalise the
      mechanism: hazards declared per role-vocabulary (forms: submit empty,
      submit twice; lists: delete while filtered; navigation: back after
      error). Each hazard is only armed when the page's controls match — the
      existing pattern, extended.
- [ ] **Bench as release gate.** `bench/run.py --all` must pass a threshold
      (start: 85% per probe, no `runtime-*` failures) before any publish.
      Wire it as a manual release step now, CI later. Grow the probe set to
      one per capability pack; the three existing probes stay pinned as
      regression anchors.
- [ ] **Score history.** Persist each scorecard with the git sha of spiral that
      produced it (`bench/history.jsonl`). The question "did spiral get better
      this month" must be answerable with a plot, not a recollection.

### Tier 3 — the loop itself (week)

- [ ] **Behavioural acceptance checks in the spec.** The bench's
      click-and-assert format, available to the spec extractor: a requirement
      may carry `probe: {click: [...], expect_any: [...], reject_any: [...]}`
      instead of a shell check. Validation then runs it after EVERY remediation
      touching that requirement — the direct fix for wording-satisfied
      behaviour-regressed, applied at the requirement level rather than only at
      finish.
- [ ] **Healer scope growth.** `regress.heal` reverts the guilty commit for
      runtime regressions. Extend the predicate menu: gate-red regressions
      (same bisect, predicate = the gate), quality-floor regressions (predicate
      = audit_ui count not increased). Always fast-path, never gatekeeper.
- [ ] **Critic diversity, measured.** A different-family critic (gemma) reviews
      plans; keep qwen as planner. Measure on the pinned plans that qwen
      self-approved with zero requirement mappings: the acceptance criterion
      for the critic role is that it catches those. If it does not, the model
      is not the fix and the deterministic lint set grows instead.
- [ ] **Escalation-lane parse guard.** The 27B lane died to "3 unparseable
      replies" — prose with no SEARCH/REPLACE. The harness already knows the
      expected format; a malformed reply should cost a retry with a terser
      format reminder before it costs a lane. Cheap, bounded (one reprompt),
      and addresses the single most expensive observed waste (a 12-attempt
      lane lost in 3).
- [ ] **Budget attribution report.** Every run ends with tokens-by-phase and
      tokens-by-outcome (landed / reverted / unparseable). Today's runs wasted
      the large majority of tokens on work that was reverted; that number must
      be on the summary card where it hurts.

### Tier 4 — capability packs as the organising unit (weeks)

The end state the redesign points at: a pack per domain that owns

    detect      files/goal patterns that identify the domain
    ladder      the rungs, each with a demonstrated failing case
    foundation  deterministic ground truth written before feature work
    floor       computable quality checks (audit issues)
    hazards     runtime probe sequences
    skills      discipline docs, ecosystem-gated
    needs       DOMAIN_PACKAGES/BINARIES entries + certificates
    probe       one bench probe as the pack's acceptance test

Packs to build, in order of leverage: `python-web` (mostly exists — assemble),
`static-web` (exists — assemble), `python-cli`, `android` (exists implicitly —
extract), `node-web`, `python-ml` (torch + dataset handling + a training-run
smoke: loss decreases over 20 steps on synthetic data). A goal that matches no
pack gets the generic ladder plus a logged gap — and "which packs are missing"
becomes a measurable roadmap instead of a feeling.

### Testing doctrine (applies to every tier)

1. **Every defect fixed today gets a name in the test suite.** Done for all
   sixteen; keep the invariant.
2. **Mutation-test the guards quarterly.** Flip a guard off (`ratchet=False`,
   remove the clamp, re-wrap the gate in parens) and assert the suite fails.
   A guard whose removal nothing notices is dead weight. `test_gate_detect`
   catching the `((` reintroduction within hours is the existence proof.
3. **Self-application.** spiral builds its own probes: run
   `spiral build` on the bench goals as the pre-release ritual, score with the
   bench, keep the transcripts. The tool that cannot build a calculator to 85%
   has no business claiming "any project".
4. **Chaos hour, monthly.** One hour of hostile inputs: goals in other
   languages, goals demanding the impossible, repos with submodules, repos
   with hooks, read-only files, 10k-file repos, a gate that hangs. Each crash
   becomes a named test.
5. **The bench never trusts spiral.** Scorecards run from a clean checkout of
   the built tree with an independent interpreter, as they do now. The moment
   a bench check imports spiral's own judgment, it stops being evidence.

## What "done" looks like

Not perfection — a contract:

- Any goal on the supported-pack list, from an empty repo, reaches its bench
  threshold unattended, or terminates early with a correctly-attributed,
  human-readable reason (gate broken / capability missing / requirement
  unsatisfiable).
- No silent no-ops anywhere: every check that did not run is named in the
  report it belongs to.
- Every model-facing failure costs bounded tokens: deterministic fast paths
  first, one format-reminder retry, ratchet everywhere partial progress is
  possible, budget attribution on the summary card.
- The published package passes its own bench from a fresh `pipx install` on a
  machine that is not this one.
