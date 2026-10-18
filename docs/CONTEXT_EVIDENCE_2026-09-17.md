# Context that can be navigated and checked

Priority: autonomous coding across limited working sets. This is source work,
not an installed-client or hour/day-long reliability claim.

## Research and the design decision

- [MemGPT](https://arxiv.org/abs/2310.08560) motivates hierarchical external
  memory and explicit paging. Keeping exact sources outside the prompt is useful;
  it does not establish that an arbitrary model knows when to retrieve them.
- [Recursive Language Models](https://arxiv.org/html/2512.24601v1) explores
  programmatic inspection of external context and recursive inference. Its
  frontier-model experiments and acknowledged inefficient context decisions do
  not prove equivalent results on the selected 27B model. Spiral should not copy
  unrestricted recursion, mixed-model calls or unbounded inference costs.
- [LocAgent](https://arxiv.org/html/2503.09089v1) motivates navigable file/symbol/
  import relationships. Its Python graph and fine-tuned localization results are
  distinct from general autonomous project completion. The first Spiral slice
  uses static Python definitions/imports and generic text search for other files.
- [Repoformer](https://arxiv.org/abs/2403.10059),
  [Self-RAG](https://arxiv.org/abs/2310.11511) and
  [FLARE](https://arxiv.org/abs/2305.06983) study selective/active retrieval.
  Learned retrieval tokens and token-confidence signals are not automatically
  calibrated in an arbitrary local coding model. Use observable source gaps as
  one trigger; allow explicit model lookup as another.
- [ACE](https://arxiv.org/abs/2510.04618) motivates incremental context records
  instead of repeatedly rewriting one lossy summary. Learned advice must remain
  subordinate to original requirements, current source and verification receipts.

The resulting design separates durable exact evidence, a rebuildable navigation
map, and the small current prompt. A receipt records a file hash and the exact
character range supplied. It proves visibility, not comprehension. Before a
patch, a missing/stale target triggers retrieval and another bounded model turn;
no proposed block is applied during that retrieval. The original verifier still
decides whether the result works.

## Milestones and current evidence

1. **Implemented, initial tests pass:** bounded fresh workspace map, file/symbol
   lookup, static local import neighbors, durable versioned map manifests,
   `ASK: context`, and current-prompt source coverage. No embedding service,
   extra model, or changes to the selected model/stream transport.
2. **Implemented, source validated:** missing/stale patch target
   admission; retrieval consumes the existing ASK budget and model call budget;
   repeated gaps cannot reset edit attempts forever. Diversity candidates use
   the same admission check. Structured diagnostic addresses can fetch omitted
   source before the next model call. Multiple requested regions remain together
   when they fit the working-set allowance. Small existing-file views continue
   normally. Fresh coverage after a frozen diff requires its captured file hash;
   a concurrent change cannot inherit that receipt.
3. **Pending:** real selected-model trial and independent project verification,
   retaining previous failures. The native comparison window remains the frozen
   earlier run waiting for approval; a screenshot is not evidence that new code
   has run there.
4. **Pending:** measured production planning/prompt admission and a full host/client
   project through two steers, compaction, Stop/restart/resume and authorized
   budget continuation. Preserve explicit incomplete results if the model fails.

## Limits of this slice

Exact content-addressed requirements, attempts and tool/verification results are
also searchable after compaction/restart. They are labelled historical and checked
against their saved digest; generated map manifests never index themselves.
Large maps are split into exact, addressable parts. Literal workspace file
arguments in the verification command enter the worker context, and their
directories become scoped lookup roots, including otherwise hidden fixtures.
This reads the checks; it grants no permission to alter them.

The map is bounded lexical/static navigation, not semantic understanding or a
complete call graph. Hidden files, symlinks, binary/oversized sources and files
beyond the scan limits are omitted. No match is never a global absence claim.
Exact stored histories remain addressable through the existing file-page tool;
this map does not yet index the host's entire conversational journal. Existing shell tools
retain their brokered authority; patch coverage is not a general shell-effect
proof. Static graph edges do not yet trigger dependency-closure admission.
Large-model capacity/quality probing and installed/phone parity remain separate
acceptance gates. No nominal 700K capability is inferred from these mechanisms.

## Source validation recorded so far

- Full engine after initial map/admission: 1,299 passed, one skipped, 89.32 s.
- Full engine after saved-history lookup, concurrent-change protection,
  multi-region retention and map paging: 1,309 passed, one skipped, 114.77 s.
- Subsequent verification-source integration: 67 focused tests passed, including
  23 new context-map tests. The preceding full count predates this last addition.
- Actual repository scan: 224 indexed text files, 0.885 s; symbol lookup 0.031 s.
  This is one local timing observation, not a broad performance benchmark.
