# Measured source costs in normal planning

The conductor can now supply whole-file token costs and static import candidates
to the normal planner. A configured model context is kept distinct from measured
source size and from actual backend admission. This does not validate a 700K
context or make the failed project pass.

## Runtime hook

The trusted host may provide an absolute `SPIRAL_CONTEXT_TOKENIZER` executable.
It is never selected from the project being edited. With that hook present,
`Ollama.source_tokenizer()` resolves the **exact selected worker model** through
the local provider's read-only metadata endpoint. Unsupported/remote bindings
produce no local measurements. No model is substituted, evicted or generated.

`scripts/context_tokenizer.cpp` loads vocabulary only, without weights or a KV
cache, and measures a bounded batch in one process. Source bytes, file count,
stdout/stderr, wall time and cache size are bounded. Stop/deadline exceptions
propagate; the owned helper is reaped before the runtime activity ends. File and
executable identities are checked before/after use. The tokenizer identity binds
local filesystem versions; it is not a cryptographic verification of all model
weights or a proof that another provider uses an identical chat template.

On this development Mac, the helper was built outside the repositories with:

```sh
clang++ -std=c++17 -O2 scripts/context_tokenizer.cpp \
  -I/opt/homebrew/include -L/opt/homebrew/lib \
  -Wl,-rpath,/opt/homebrew/lib -lllama \
  -o /Users/edt/spiralChat/build/context-tokenizer-20260917/spiral-tokenize
```

That is a source-development artifact, not an installed app/runtime change.
Production helper packaging and activation remain separate work. No host or
user model settings were changed. The initial local build command had a linker
flag typo; its corrected build succeeded.

## Planning semantics

Raw source counts use no BOS and do not parse special tokens. Exact source
digests and static dependency candidates accompany the counts. The workspace
versions are rechecked after measurement. Full records are saved with content
addresses; large inventories get bounded directory summaries and links to the
complete paged records. Omitted view entries are explicit.

Normal planning receives this as sizing evidence. It can still create new files;
their future sizes are unmeasured. This **does not** claim strict whole-file
envelope admission for every task or alter the existing strict experimental
planner tests. The actual rendered chat, framing, control/history and output
must still pass the generation backend's admission. The owned backend already
counts its exact chat template; equivalent engine-wide dispatch/packing remains
an open milestone.

## Real measurement evidence

With `qwen3.8:27b-heretic`, the batch helper measured the four frozen project
files in 0.722 s. Counts **1139 / 1113 / 99 / 1233** exactly matched separate
calls to the installed official tokenizer. A real conductor measurement run
counted nine current text files in 0.479 s and made **zero generation calls**.
Empty text, Unicode, escaped/newline text and literal control-token spellings
also matched the official tokenizer. These are local timing observations,
not equal-load performance benchmarks.

Artifacts: `/Users/edt/spiralChat/build/context-tokenizer-20260917/measurement.json`
and `conductor-measurement.json` in the same directory.

The vocabulary-only mechanism was checked against the
[installed llama.cpp revision's tokenizer source](https://github.com/ggml-org/llama.cpp/blob/a935fbffe/tools/tokenize/tokenize.cpp).
No paper result is treated as a reproduced project-quality result.

## Subsequent worker integration

The same optional counter now sizes normal worker and diversity source views.
See `WORKER_CONTINUATION_2026-09-17.md` for real 8K/32K/128K sizing measurements,
the discovered large-input pipe defect and its fix, and exact limits. This closes
source packing plumbing, not backend capacity/quality admission or app activation.
