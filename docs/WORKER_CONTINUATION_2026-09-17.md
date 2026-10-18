# Durable worker continuation and measured working sets

## What changed

`worker_memory.py` records exact task/model-scoped worker events in a small SQLite
index. Content-addressed originals hold public replies, input snapshots, tool
results, verification output and rollback feedback. A restarted worker gets a
bounded recent view and can page older records with `ASK: history before N`, then
read the exact referenced evidence with `ASK: file`.

Proposals remain historical and untrusted. They are not replayed automatically.
A pending call is not a successful call; a failed or interrupted lane records its
rollback/archive location. Current task, source versions and a fresh verification
result take precedence. Changing the exact task contract or selected model uses a
separate history. The broader project context map can still find saved historical
material; explicit cross-task lineage is not inferred from similar prose.

The index supports concurrent inserts and commits only after the referenced event
exists. Corrupt or mismatched records are not presented as valid evidence. Records
are private workspace artifacts, not new external storage or model calls. This is
a retrieval mechanism, not a claim that a small model knows when every possible
fact is missing.

`working_set.py` now sizes actual worker source views with the optional trusted
vocabulary helper. It keeps task/control sections intact, varies only source-body
space and preserves addresses for omitted text. Full source prefixes can exceed
the previous 16K-per-file/36K-total-character limits when the measured envelope
allows it. The selected model and context setting do not change.

Sizing reserves the requested output plus 2,048 framing/host-metadata tokens.
It measures raw text chunks, not the backend's chat template. Every sizing receipt
says `execution_admitted: false`; exact backend admission remains mandatory where
supported. An undersized measured control envelope stops before inference. If the
optional helper is unavailable, the existing bounded path remains, with an explicit
unavailable ledger event. Production packaging/activation is still open.

## Real measurement and a preserved failure

The initial real larger-input measurement timed out: Python 3.11 pipe retries stopped
feeding input after a short polling timeout. The source now passes the bounded batch
through a private temporary input file, independently of cancellation polling.
A delayed-reader regression verifies a 600KB Unicode payload reaches the helper in
full. Timeout/Stop still reap only the owned vocabulary process.

With the exact Heretic vocabulary and the real worker renderer:

| Configured sizing envelope | Source characters visible | Measured raw text tokens | Sizing time |
| --- | ---: | ---: | ---: |
| 8,192 | 8,192 | 4,063 | 3.37s |
| 32,768 | 74,752 | 28,344 | 2.73s |
| 131,072 | 335,872 | 125,715 | 1.74s |

Each reserves 2,048 output and 2,048 framing tokens. The independent official CLI
count matched the two smaller raw-message sums; the largest chunk sum was two
tokens higher than the unchunked raw-message sum (125,713). No inference, KV-cache
allocation, 128K quality test or GSQ/700K validation took place.

Artifacts: `/Users/edt/spiralChat/build/context-tokenizer-20260917/` includes the
initial failure, successful `working-set-measurement.json` and both probe scripts.

## Observed retry-budget defect

The second real verification-source continuation explicitly requested zero
escalation attempts. `esc_attempts or default` incorrectly started a thinking call.
The owned trial was stopped at 427.76 seconds; both backend exits were observed,
unsettled inference was zero and the lane was free. The first normal patch still
failed verification, all original assertions remained, and the final project
remained 6/15 on the independent behavior comparison. It did not use context lookup.

Worker and escalation budgets now distinguish `None` from zero, reject invalid
limits and suppress disabled lanes before their work starts. The failed live run
is retained rather than relabeled as a successful no-escalation test.

## Evidence boundary

Regression tests exercise actual Atom restart, historical lookup, rollback,
current-source checks, task/model isolation, pending calls, malformed records,
source packing, context refusal and cancellation. Scripted test replies verify
mechanisms; they are not model-quality or native-UI evidence. Hour/day-long project
acceptance, paired-client continuation and installed deployment remain open.

Final full engine validation: **1,360 passed, one skipped, 88.02s**.
A separate two-process probe used actual sandboxed gates and scripted replies to
verify restart/retrieval/rollback with the original assertion unchanged; its result
is `/Users/edt/spiralChat/build/context-process-restart-20260917.json`.

## Relocatable source runtime

`scripts/stage_context_tokenizer.py` now stages the explicit helper and its four
non-system libraries, rewrites only copied load commands, removes build-machine
search paths, verifies ad-hoc signatures and writes a content manifest. It refuses
an existing output. This is development packaging, not application installation.

A relocated directory with spaces loaded the real Heretic vocabulary, with no
Homebrew library in its dynamic-loader trace. A frozen source host started against
its private empty database and shared machine compute lease; a separate real child
runtime exercised the production conductor inventory hook. Zero model calls or
user tasks were dispatched. The owned host then exited. Record:
`/Users/edt/spiralChat/build/context-source-host-stage-20260917/report.json`.
Controller-launched task/UI acceptance remains distinct from this startup smoke.
