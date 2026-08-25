# Academic QLoRA runbook

The production profile is deliberately narrow: the immutable
`mlx-community/Qwen3.8-27B-4bit` revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff` in `qwen38_27b_q4.toml`, its exact
three-shard inventory, an Apple-silicon host, and the exact MLX versions in
`requirements-mlx.lock`. Qwen3.8 retains the `qwen3_5` MLX architecture name;
that technical name is expected and is not a Qwen3.5 checkpoint substitution.
The harness never downloads the 27B checkpoint. The existing local checkpoint
is `/Volumes/T5_EVO_EDT/qwen38-mlx`.

Create an isolated Python 3.12 environment; do not install this stack into
Spiral's application Python:

```sh
python3.12 -m venv ~/.venvs/spiral-academic-mlx
~/.venvs/spiral-academic-mlx/bin/python -m pip install --upgrade 'pip==26.2'
~/.venvs/spiral-academic-mlx/bin/python -m pip install \
  --requirement scripts/academic_finetune/requirements-mlx.lock
```

Prepare and inspect without loading weights:

```sh
python scripts/academic_finetune/train_qlora.py \
  --corpus /path/to/academic_corpus.jsonl \
  --data-dir /Volumes/T5_EVO_EDT/academic/data \
  --model /Volumes/T5_EVO_EDT/qwen38-mlx \
  --output /Volumes/T5_EVO_EDT/academic/run-001 \
  --python ~/.venvs/spiral-academic-mlx/bin/python
```

Add `--execute` only after the preflight receipt is green. The trainer acquires
`~/.spiralchat/spiral-compute.lease` for the entire child-process lifetime and
requires `/api/ps` to show no resident Ollama model. It never evicts another
model. Stop chat, voice, OCR, and other inference first.

The small audited model view defaults to the internal APFS cache at
`~/Library/Caches/SpiralAcademic`; 15+ GB model weights, dataset, checkpoints and
adapter output remain on the caller-selected disk. This avoids relying on
symlinks inside an exFAT T5 volume.

`--resume` selects the latest complete, content-hashed adapter checkpoint.
MLX-LM 0.31.3 does not serialize optimizer moments, so this is a deterministic
adapter-weight continuation, not a bit-identical uninterrupted optimizer run.
That limitation is recorded with every checkpoint.

For an end-to-end trainer smoke, pass `--smoke-model` with a small local 4-bit
MLX checkpoint and `--execute`. Smoke uses a 640-token sequence cap (the audited
corpus maximum is 591), and output is permanently placed below `SMOKE_ONLY/`;
that path cannot emit `spiral.academic-adapter.v1`.

Before committing to the full run, the exact 27B model can be tested for one to
four iterations without risking a 1,200-iteration launch or deployable output:

```sh
python scripts/academic_finetune/train_qlora.py \
  --data-dir /Volumes/T5_EVO_EDT/academic/data \
  --model /Volumes/T5_EVO_EDT/qwen38-mlx \
  --output /Volumes/T5_EVO_EDT/academic/feasibility-001 \
  --python "$HOME/Library/Application Support/SpiralAcademic/runtime/bin/python" \
  --feasibility-iters 1 --execute
```

The destination must not already exist. This mode uses the production base,
hybrid target paths, batch size, accumulation, checkpointing and 1,024-token
sequence setting, but writes only `FEASIBILITY_ONLY/` receipts and metrics. It
cannot publish an adapter manifest.

Every executed training mode writes:

- `training-metrics.jsonl`: append-only, machine-readable train/validation loss,
  learning rate, throughput, and peak-memory records;
- `loss-curves.html`: an atomic, dependency-free curve view that refreshes every
  two seconds when opened locally;
- `trainer.log`: the complete raw MLX-LM output.

NaN, infinity, negative loss, journal corruption, and duplicate event/iteration
records fail closed before an adapter can be published. To regenerate the view:

```sh
python -m scripts.academic_finetune.live_metrics \
  --metrics /path/to/run/training-metrics.jsonl \
  --output /path/to/run/loss-curves.html
```

After training, use `evaluate.py compare` to generate both arms greedily with the
same seed/config, calculate completion-only held-out NLL, score argument and
citation fidelity, and create separately keyed blind A/B packets.

For a completed training run, pass `--training-run /path/to/run` and write to
`/path/to/run/evaluation`.  A successful comparison then atomically publishes
`post-training-evaluation.json` and clears `post_training_validation_required` in
`training-status.json`.  If comparison artifacts finished but final status writing
was interrupted, `evaluate.py finalize` rehashes and finalizes those artifacts
without loading the model; it is safe and idempotent to repeat.

Serve the published adapter on the manifest-pinned loopback endpoint with:

```sh
spiral-academic-serve \
  --manifest /path/to/run/academic-adapter.manifest.json \
  --python "$HOME/Library/Application Support/SpiralAcademic/runtime/bin/python"
```

The content-addressed model view is derived from the manifest and the default
`~/Library/Caches/SpiralAcademic` cache; use `--model-view` only for an explicit
equivalent view. `GET /v1/spiral/identity` exposes the exact runtime attestation.
Each `POST /v1/chat/completions` rehashes the adapter, corpus, prepared splits,
and every base weight shard before loading. Generation occurs in a child process
under `~/.spiralchat/spiral-compute.lease`, requires Ollama to be empty, and the
child exits before the lease is released, so the 27B weights are never resident
between requests. The server binds only to the manifest's loopback port.

Requests may include `"adapter_strength": 1.0`. The value is a real in-memory
multiplier on every loaded LoRA module's trained scale, bounded to 0.0–2.0 in
0.05 steps. Thus 0.0 uses the base-model contribution, 1.0 leaves the trained
scale (32) unchanged, and values above one amplify the adapter. The server never
edits the authenticated adapter bundle, and every response separately echoes the
actual value as `spiral_adapter_strength` while the base runtime identity remains
immutable. An omitted field defaults to 1.0.

To retain image, tool, streaming and thinking capabilities on the complete
Qwen checkpoint, run the separate Ollama-compatible VLM lane:

Create its isolated, proven CPython 3.12 environment separately from the
MLX-LM training runtime:

```sh
VLM_RUNTIME="$HOME/Library/Application Support/SpiralAcademic/vlm-runtime"
uv venv --python 3.12 "$VLM_RUNTIME"
uv pip install \
  --python "$VLM_RUNTIME/bin/python3" \
  --requirement scripts/academic_finetune/requirements-vlm.lock
```

The lock pins the exact direct runtime set used by the live image, streaming,
thinking, tool-parser and adapter-strength tests. Do not install it over the
training environment: the VLM lane uses MLX 0.32.1, while the authenticated
MLX-LM training and text-serving receipt remains pinned to MLX 0.31.2.

Then start the service:

```sh
spiral-academic-vlm-serve \
  --manifest /path/to/run/academic-adapter.manifest.json \
  --model-root /Volumes/T5_EVO_EDT/qwen38-mlx \
  --python "$HOME/Library/Application Support/SpiralAcademic/vlm-runtime/bin/python3" \
  --lease-authority-token-file "$HOME/Library/Application Support/SpiralAcademic/lease-authority.token"
```

It binds to `127.0.0.1:8081`, serves `POST /api/chat`, and exposes the same
identity path at `GET /v1/spiral/identity`. The isolated VLM runtime lock pins
MLX 0.32.1, MLX-VLM 0.6.16 and Transformers 5.15.1 plus their proven direct
runtime dependencies. The token file must be a
non-symlink regular file owned by the serving user with mode 0600 and 32–512
printable ASCII bytes. The Spiral host sends it only in
`X-Spiral-Lease-Authority` while it already holds the shared compute flock; this
authenticated handoff prevents a recursive flock deadlock. A direct request with
no header acquires the flock itself. Both paths still require Ollama to be empty,
and the child is terminated and awaited before a standalone lease is released.

At startup the VLM service cryptographically verifies the full 15 GB shard
inventory plus its exact chat template, processor, tokenizer and vocabulary
frontend. Per-request and worker checks bind to that startup receipt using
device/inode/size/mtime/ctime metadata and rehash the small config/index files,
avoiding two redundant 15 GB reads before generation. Final responses publish
separate attestation, model-load and first-token timings. `adapter_strength` is
accepted at the top level on the same 0.0–2.0 grid and is echoed on every NDJSON
delta and final frame; the final frame also contains the immutable VLM identity.
