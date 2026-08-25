#!/usr/bin/env python3
"""Prepare and run the pinned Qwen3.8-27B academic MLX QLoRA job.

The default action is model-free preparation plus preflight.  The 27B model is
never downloaded, imported, or loaded unless the operator supplies a verified
local snapshot and the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import time
import uuid
from pathlib import Path

try:
    from .bound_training_data import (
        BoundTrainingDataError,
        create_bounded_training_data_view,
    )
    from .training_support import (
        CheckpointLedger,
        HarnessError,
        STRUCTURE_PROMPT_CONTRACT,
        TrainingComputeLease,
        TrainingRunLock,
        TrainingSupervisorJournal,
        adapter_bundle_digest,
        atomic_write_bytes,
        atomic_write_json,
        build_adapter_manifest,
        build_training_run_contract,
        create_text_training_view,
        create_training_only_dataset_view,
        ensure_training_run_contract,
        load_dataset_manifest,
        load_toml_config,
        mlx_training_command,
        model_weight_inventory,
        plan_training_retry,
        prepare_mlx_dataset,
        publish_adapter_bundle,
        python_package_versions,
        run_preflight,
        run_training_process,
        sha256_file,
        verify_ollama_empty,
        validate_parent_adapter_initialization,
        yaml_training_config,
    )
except ImportError:  # direct `python scripts/.../train_qlora.py`
    from bound_training_data import (  # type: ignore
        BoundTrainingDataError,
        create_bounded_training_data_view,
    )
    from training_support import (  # type: ignore
        CheckpointLedger,
        HarnessError,
        STRUCTURE_PROMPT_CONTRACT,
        TrainingComputeLease,
        TrainingRunLock,
        TrainingSupervisorJournal,
        adapter_bundle_digest,
        atomic_write_bytes,
        atomic_write_json,
        build_adapter_manifest,
        build_training_run_contract,
        create_text_training_view,
        create_training_only_dataset_view,
        ensure_training_run_contract,
        load_dataset_manifest,
        load_toml_config,
        mlx_training_command,
        model_weight_inventory,
        plan_training_retry,
        prepare_mlx_dataset,
        publish_adapter_bundle,
        python_package_versions,
        run_preflight,
        run_training_process,
        sha256_file,
        verify_ollama_empty,
        validate_parent_adapter_initialization,
        yaml_training_config,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "qwen38_27b_q4.toml"
DEFAULT_VIEW_CACHE = Path("~/Library/Caches/SpiralAcademic").expanduser()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Pinned, completion-only MLX QLoRA for Spiral academic adapters")
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument(
        "--corpus", type=Path,
        help="combined JSONL from an audited academic corpus compiler")
    result.add_argument("--data-dir", type=Path, required=True, help="prepared MLX split directory")
    result.add_argument("--model", type=Path, help="local pinned Qwen3.8-27B-4bit snapshot")
    result.add_argument("--output", type=Path, required=True, help="new run/output directory")
    result.add_argument("--model-view-cache", type=Path, default=DEFAULT_VIEW_CACHE,
                        help="small APFS cache for the audited text-only model view")
    result.add_argument("--execute", action="store_true", help="actually load MLX and train")
    result.add_argument("--resume", action="store_true", help="continue from the latest atomic adapter checkpoint")
    result.add_argument(
        "--supervise", action="store_true",
        help=("babysit signal-terminated MLX children with identity-locked, "
              "bounded automatic retries"))
    result.add_argument(
        "--max-retries", type=int, default=5,
        help="maximum automatic signal-crash retries in one supervisor session")
    result.add_argument(
        "--retry-base-seconds", type=float, default=5.0,
        help="initial supervisor retry delay")
    result.add_argument(
        "--retry-max-seconds", type=float, default=60.0,
        help="maximum supervisor retry delay (hard limit: 60 seconds)")
    result.add_argument(
        "--initialize-from-manifest", type=Path,
        help=("authenticate and initialize a fresh cumulative run from an immutable "
              "academic adapter manifest"))
    result.add_argument(
        "--safe-schedule", action="store_true",
        help=("token-gate every row, checkpoint every optimizer update, and "
              "evaluate held-out splits only after the trainer unloads"))
    result.add_argument("--lease-path", type=Path,
                        default=Path("~/.spiralchat/spiral-compute.lease").expanduser())
    result.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    result.add_argument("--python", default=sys.executable, help="Python containing pinned mlx/mlx-lm")
    smoke = result.add_argument_group("non-production live smoke")
    smoke.add_argument("--smoke-model", type=Path,
                       help="explicit small local MLX model; never publishes an academic adapter")
    smoke.add_argument("--smoke-iters", type=int, default=1)
    result.add_argument(
        "--feasibility-iters", type=int,
        help=("load the exact production 27B stack for only 1–4 iterations; "
              "writes FEASIBILITY_ONLY and can never publish an adapter"))
    return result


def _prepare(arguments: argparse.Namespace, config: dict) -> dict:
    if arguments.corpus is not None:
        return prepare_mlx_dataset(
            arguments.corpus, arguments.data_dir,
            require_trainable=arguments.smoke_model is None, config=config)
    return load_dataset_manifest(arguments.data_dir, config=config)


def _smoke_config(config: dict, iterations: int) -> dict:
    if iterations <= 0 or iterations > 8:
        raise HarnessError("--smoke-iters must be between 1 and 8")
    smoke = copy.deepcopy(config)
    training = smoke["training"]
    training.update({
        "grad_accumulation_steps": 1,
        "num_layers": 1,
        "iterations": iterations,
        "val_batches": 1,
        "steps_per_report": 1,
        "steps_per_eval": 1,
        "save_every": 1,
        # The audited 1,398-example corpus reaches 591 prompt+completion tokens.
        # A 640-token smoke exercises complete held-out targets instead of creating
        # a meaningless NaN validation loss through completion truncation.
        "max_seq_length": min(640, int(training["max_seq_length"])),
    })
    # Qwen2.5 and other smoke architectures do not have Qwen3.5 linear attention.
    smoke["lora"]["keys"] = ["self_attn.q_proj", "self_attn.v_proj"]
    return smoke


def _validate_smoke_model(path: Path) -> dict:
    if not path.is_dir():
        raise HarnessError(f"smoke model is not a local directory: {path}")
    try:
        model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid smoke model config: {exc}") from exc
    quant = model_config.get("quantization") or model_config.get("quantization_config") or {}
    if quant.get("bits") != 4:
        raise HarnessError("smoke model must itself be a 4-bit MLX checkpoint")
    _, shards = model_weight_inventory(path)
    return {
        "smoke_only": True,
        "model_path": str(path.resolve()),
        "model_type": model_config.get("model_type"),
        "architecture": model_config.get("architectures"),
        "weight_bytes": sum(item.stat().st_size for item in shards),
    }


def _run_smoke(arguments: argparse.Namespace, config: dict, dataset: dict) -> int:
    smoke_receipt = _validate_smoke_model(arguments.smoke_model)
    smoke_config = _smoke_config(config, arguments.smoke_iters)
    smoke_root = arguments.output / "SMOKE_ONLY"
    if smoke_root.exists():
        raise HarnessError(
            "smoke output already exists; use a fresh --output so metrics/checkpoints cannot mix")
    smoke_root.mkdir(parents=True)
    marker = {
        "schema_version": "spiral.academic-smoke-only.v1",
        "deployable": False,
        "reason": "Non-production trainer/runtime validation with an architecture override.",
        "model": smoke_receipt,
        "dataset_manifest_sha256": dataset.get("source_corpus_manifest_sha256"),
    }
    atomic_write_json(smoke_root / "SMOKE_ONLY.json", marker)
    work_adapter = smoke_root / "work-adapter"
    work_adapter.mkdir(exist_ok=True)
    yaml_path = smoke_root / "mlx_lora.yaml"
    atomic_write_bytes(
        yaml_path,
        yaml_training_config(
            smoke_config, arguments.smoke_model, arguments.data_dir, work_adapter,
            iterations=arguments.smoke_iters,
        ).encode("utf-8"),
    )
    command = mlx_training_command(yaml_path, python_executable=arguments.python)
    print("SMOKE_ONLY: no production adapter manifest can be emitted")
    print("command:", " ".join(command))
    if not arguments.execute:
        return 0
    versions = python_package_versions(arguments.python)
    if versions["mlx"] is None or versions["mlx_lm"] is None:
        raise HarnessError("the selected Python does not expose installed mlx and mlx-lm packages")
    ledger = CheckpointLedger(smoke_root / "checkpoints")
    smoke_run_id = "smoke-" + uuid.uuid4().hex
    metrics_path = smoke_root / "training-metrics.jsonl"
    print(f"live loss view: {smoke_root / 'loss-curves.html'}")
    print(f"machine metrics: {metrics_path}")
    lease = TrainingComputeLease(arguments.lease_path)
    lease.acquire(owner={"model": str(arguments.smoke_model), "smoke_only": True})
    with lease:
        verify_ollama_empty(arguments.ollama_url)
        result = run_training_process(
            command, work_adapter, ledger, inherited_lease_fd=lease.descriptor,
            metrics_path=metrics_path, metrics_mode="smoke_only",
            metrics_run_id=smoke_run_id,
            trainer_log_path=smoke_root / "trainer.log")
    if result:
        raise HarnessError(f"MLX smoke trainer exited with status {result}")
    if (work_adapter / "adapters.safetensors").is_file():
        ledger.capture(
            work_adapter / "adapters.safetensors", arguments.smoke_iters,
            adapter_config=work_adapter / "adapter_config.json")
    atomic_write_json(smoke_root / "SMOKE_COMPLETE.json", {
        **marker,
        "completed": True,
        "academic_adapter_manifest_emitted": False,
    })
    return 0


def _safe_runtime_config(config: dict) -> tuple[dict, dict]:
    runtime = copy.deepcopy(config)
    training = runtime["training"]
    total_steps = int(training["iterations"])
    accumulation = int(training["grad_accumulation_steps"])
    save_every = min(8, int(training["save_every"]), total_steps)
    save_every -= save_every % accumulation
    if save_every <= 0:
        save_every = accumulation
    training["save_every"] = save_every
    # The safe data view has no valid split, so MLX-LM cannot take its forced
    # iteration-1/final validation branches. Keep an out-of-horizon value in the
    # generated YAML as defense in depth and as an explicit operational receipt.
    training["steps_per_eval"] = total_steps + 1
    return runtime, {
        "mode": "checkpoint_only_training",
        "save_every": save_every,
        "steps_per_eval": total_steps + 1,
        "in_process_validation": False,
        "held_out_evaluation": "separate_process_after_training",
        "note": "Held-out evaluation is isolated; the bounded data receipt gates backward-pass size.",
    }


def _configured_runtime_schedule(config: dict) -> dict:
    training = config["training"]
    return {
        "mode": "configured_in_process_validation",
        "save_every": training["save_every"],
        "steps_per_eval": training["steps_per_eval"],
        "val_batches": training["val_batches"],
        "in_process_validation": True,
        "held_out_evaluation": "mlx_lm_training_process",
    }


def _require_unpartitioned_structure_view(
    receipt: dict, dataset_manifest: dict | None = None,
) -> None:
    output = receipt.get("output")
    if not isinstance(output, dict) or (
        output.get("partitioned_source_rows") != 0
        or output.get("derived_rows") != 0
    ):
        raise HarnessError(
            "academic structure rows must fit the token boundary exactly; "
            "automatic prose partitioning is forbidden")
    if dataset_manifest is None:
        return
    source_gate = dataset_manifest.get("source_exact_training_token_gate")
    source_tokenizer = (
        source_gate.get("tokenizer") if isinstance(source_gate, dict) else None
    )
    runtime_tokenizer = receipt.get("tokenizer")
    gate = receipt.get("gate")
    if (
        not isinstance(source_gate, dict)
        or not isinstance(source_tokenizer, dict)
        or not isinstance(runtime_tokenizer, dict)
        or source_tokenizer.get("identity") != runtime_tokenizer.get("identity")
        or not isinstance(gate, dict)
        or source_gate.get("max_sequence_length") != gate.get("max_sequence_length")
    ):
        raise HarnessError(
            "academic structure compiler and training view must use the exact same "
            "tokenizer identity and sequence boundary")


def _select_adapter_initialization(previous, parent_adapter):
    """Prefer this run's durable checkpoint over its immutable stage-one parent."""

    if previous is not None:
        return previous.config_path, previous.path
    if parent_adapter is not None:
        return parent_adapter.adapter_config_path, parent_adapter.weights_path
    return None, None


def _resume_command(arguments: argparse.Namespace) -> str:
    if arguments.model is None:
        raise HarnessError("cannot construct a resume command without --model")
    argv = [
        str(arguments.python), "-u", "-m", "scripts.academic_finetune.train_qlora",
        "--config", str(arguments.config.resolve()),
        "--data-dir", str(arguments.data_dir.resolve()),
        "--model", str(arguments.model.resolve()),
        "--output", str(arguments.output.resolve()),
        "--model-view-cache", str(arguments.model_view_cache.resolve()),
        "--lease-path", str(arguments.lease_path.resolve()),
        "--ollama-url", str(arguments.ollama_url),
        "--python", str(arguments.python),
    ]
    if arguments.safe_schedule:
        argv.append("--safe-schedule")
    if arguments.initialize_from_manifest is not None:
        argv.extend([
            "--initialize-from-manifest",
            str(arguments.initialize_from_manifest.resolve()),
        ])
    if getattr(arguments, "supervise", False):
        argv.extend([
            "--supervise",
            "--max-retries", str(arguments.max_retries),
            "--retry-base-seconds", str(arguments.retry_base_seconds),
            "--retry-max-seconds", str(arguments.retry_max_seconds),
        ])
    argv.extend(["--resume", "--execute"])
    repository_root = HERE.parents[1]
    return (
        f"cd {shlex.quote(str(repository_root))} && "
        f"/usr/bin/caffeinate -ims {shlex.join(argv)}"
    )


def _checkpoint_status(checkpoint) -> dict | None:
    if checkpoint is None:
        return None
    return {
        "step": checkpoint.step,
        "weights": str(checkpoint.path),
        "weights_sha256": checkpoint.sha256,
        "adapter_config": str(checkpoint.config_path),
        "adapter_config_sha256": checkpoint.config_sha256,
        "bundle_sha256": checkpoint.bundle_sha256,
        "receipt_sha256": checkpoint.receipt_sha256,
    }


def _write_training_status(
    output: Path, *, state: str, run_contract: dict, schedule: dict,
    resume_command: str, total_steps: int, checkpoint=None,
    attempt_id: str | None = None, error: str | None = None,
    supervisor_status: dict | None = None,
) -> dict:
    completed_steps = checkpoint.step if checkpoint is not None else 0
    status = {
        "schema_version": "spiral.academic-training-status.v1",
        "state": state,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_identity": run_contract["run_identity"],
        "attempt_id": attempt_id,
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "remaining_steps": max(0, total_steps - completed_steps),
        "latest_checkpoint": _checkpoint_status(checkpoint),
        "resume_available": checkpoint is not None and completed_steps < total_steps,
        "restart_available": checkpoint is None and completed_steps < total_steps,
        "recovery_mode": (
            "checkpoint_resume" if checkpoint is not None else "fresh_restart"),
        "resume_command": resume_command,
        "runtime_schedule": schedule,
        "resume_semantics": run_contract["resume_semantics"],
        "post_training_validation_required": schedule.get("in_process_validation") is False,
    }
    if error is not None:
        status["error"] = error
    if supervisor_status is not None:
        status["supervisor"] = {
            key: supervisor_status.get(key) for key in (
                "schema_version", "state", "run_identity", "session_id", "pid",
                "max_retries", "retries_used", "attempt_id", "recovery_mode",
                "next_retry_at_epoch", "last_exit_status", "last_signal",
                "last_event_sequence", "event_ledger",
            ) if key in supervisor_status
        }
    atomic_write_json(output / "training-status.json", status)
    return status


def _publish_from_checkpoint(
    arguments: argparse.Namespace, config: dict, dataset: dict, base_receipt: dict,
    versions: dict, checkpoint, *, training_data_receipt: dict | None = None,
    parent_adapter=None,
) -> Path:
    adapter_dir = arguments.output / "adapter"
    if adapter_dir.exists():
        bundle_digest, required_files = adapter_bundle_digest(adapter_dir)
        if bundle_digest != checkpoint.bundle_sha256:
            raise HarnessError(
                "existing published adapter does not match the completed checkpoint")
    else:
        adapter_dir, bundle_digest, required_files = publish_adapter_bundle(
            checkpoint.path.parent, arguments.output)
    if bundle_digest != checkpoint.bundle_sha256:
        raise HarnessError("published adapter does not match the completed checkpoint")
    adapter_manifest_path = arguments.output / "academic-adapter.manifest.json"
    manifest = build_adapter_manifest(
        config=config,
        base_receipt=base_receipt,
        dataset_manifest=dataset,
        dataset_manifest_path=arguments.data_dir / "dataset_manifest.json",
        adapter_manifest_path=adapter_manifest_path,
        adapter_dir=adapter_dir,
        bundle_digest=bundle_digest,
        required_files=required_files,
        package_versions=versions,
        training_data_receipt=training_data_receipt,
        parent_adapter=parent_adapter,
    )
    if adapter_manifest_path.exists():
        try:
            existing = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"existing adapter manifest is corrupt: {exc}") from exc
        if existing != manifest:
            raise HarnessError("existing adapter manifest does not match the completed run")
    else:
        atomic_write_json(adapter_manifest_path, manifest)
    return adapter_manifest_path


def _prior_training_status(output: Path, run_identity: str) -> dict | None:
    path = output / "training-status.json"
    if not path.exists():
        return None
    if not path.is_file():
        raise HarnessError("training status is not a file")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"existing training status is corrupt: {exc}") from exc
    if (
        not isinstance(status, dict)
        or status.get("schema_version") != "spiral.academic-training-status.v1"
        or status.get("run_identity") != run_identity
    ):
        raise HarnessError("existing training status belongs to another run identity")
    return status


def _production(arguments: argparse.Namespace, config: dict, dataset: dict) -> int:
    arguments.output.mkdir(parents=True, exist_ok=True)
    run_lock = TrainingRunLock(arguments.output / ".training-run.lock")
    run_lock.acquire(owner={"output": str(arguments.output.resolve())})
    with run_lock:
        return _production_locked(arguments, config, dataset, run_lock=run_lock)


def _production_locked(
    arguments: argparse.Namespace, config: dict, dataset: dict,
    *, run_lock: TrainingRunLock,
) -> int:
    if arguments.model is None:
        raise HarnessError(
            "--model must point to the pinned local 27B snapshot; remote IDs are intentionally not downloaded")
    if (arguments.output / "FEASIBILITY_ONLY").exists():
        raise HarnessError("refusing to reuse a FEASIBILITY_ONLY directory for production training")
    versions = python_package_versions(arguments.python)
    report = run_preflight(
        config, arguments.model, arguments.data_dir, arguments.output,
        package_versions=versions)
    atomic_write_json(arguments.output / "preflight.json", report.as_dict())
    if not report.ok:
        report.require_ok()
    base_receipt = report.facts["base_model"]
    structure_mode = (
        config["profile"]["prompt_contract"] == STRUCTURE_PROMPT_CONTRACT)
    if structure_mode and arguments.initialize_from_manifest is None:
        raise HarnessError(
            "the academic structure profile requires --initialize-from-manifest")
    if structure_mode and not arguments.safe_schedule:
        raise HarnessError(
            "the academic structure profile requires --safe-schedule so every "
            "JSON target is token-gated without truncation or partitioning")
    parent_adapter = None
    if arguments.initialize_from_manifest is not None:
        parent_adapter = validate_parent_adapter_initialization(
            arguments.initialize_from_manifest, config, base_receipt)
    model_view = create_text_training_view(
        arguments.model, arguments.model_view_cache, base_receipt)
    total_steps = int(config["training"]["iterations"])
    runtime_config = config
    trainer_data_dir = arguments.data_dir
    training_data_receipt = None
    if arguments.safe_schedule:
        runtime_config, schedule = _safe_runtime_config(config)
        sequence_limit = int(runtime_config["training"]["max_seq_length"])
        if sequence_limit > 512:
            raise HarnessError(
                "the robust 32-GB Qwen3.8 path requires max_seq_length <= 512")
        try:
            trainer_data_dir, training_data_receipt = create_bounded_training_data_view(
                arguments.data_dir,
                arguments.output,
                arguments.model,
                max_sequence_length=sequence_limit,
                dataset_manifest=dataset,
            )
        except BoundTrainingDataError as exc:
            raise HarnessError(f"bounded training-data gate failed: {exc}") from exc
        if structure_mode:
            _require_unpartitioned_structure_view(training_data_receipt, dataset)
        schedule["training_data_view"] = str(trainer_data_dir)
        schedule["training_data_view_identity"] = training_data_receipt["view_identity"]
        schedule["maximum_training_tokens"] = training_data_receipt["gate"][
            "maximum_total_tokens"]
        schedule["all_source_completions_preserved"] = training_data_receipt[
            "preservation"]["all_source_completions_preserved"]
        if structure_mode:
            schedule["structure_overflow_policy"] = "reject_never_partition"
    else:
        schedule = _configured_runtime_schedule(config)
    run_contract = build_training_run_contract(
        config,
        base_receipt,
        dataset,
        training_data_receipt=training_data_receipt,
        parent_adapter=parent_adapter,
    )
    ensure_training_run_contract(arguments.output / "training-run.json", run_contract)
    ledger = CheckpointLedger(
        arguments.output / "checkpoints", run_identity=run_contract["run_identity"])
    latest = ledger.latest()
    if not arguments.resume and latest is not None:
        raise HarnessError("checkpoints already exist; pass --resume explicitly")
    if latest is not None and latest.step > total_steps:
        raise HarnessError("latest checkpoint exceeds the configured iteration count")
    resume_command = _resume_command(arguments)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    print(f"live loss view: {arguments.output / 'loss-curves.html'}")
    print(f"machine metrics: {arguments.output / 'training-metrics.jsonl'}")
    print(f"durable status: {arguments.output / 'training-status.json'}")
    print("exact resume command:", resume_command)
    supervisor = None
    supervisor_status = None
    retries_used = 0
    if arguments.supervise:
        session_id = uuid.uuid4().hex
        supervisor = TrainingSupervisorJournal(
            arguments.output,
            run_identity=run_contract["run_identity"],
            session_id=session_id,
            max_retries=arguments.max_retries,
            base_delay_seconds=arguments.retry_base_seconds,
            maximum_delay_seconds=arguments.retry_max_seconds,
        )
        prior = _prior_training_status(
            arguments.output, run_contract["run_identity"])
        supervisor.append(
            "session_started",
            prior_state=prior.get("state") if prior else None,
            prior_attempt_id=prior.get("attempt_id") if prior else None,
            prior_error=prior.get("error") if prior else None,
            checkpoint=_checkpoint_status(latest),
            recovery_mode=(
                "checkpoint_resume" if latest is not None else "fresh_restart"),
        )
        if prior is not None and prior.get("state") in {
            "starting", "running", "retry_wait", "failed", "interrupted",
        }:
            supervisor.append(
                "prior_status_reconciled",
                prior_state=prior.get("state"),
                prior_attempt_id=prior.get("attempt_id"),
                prior_error=prior.get("error"),
                checkpoint=_checkpoint_status(latest),
                recovery_mode=(
                    "checkpoint_resume" if latest is not None else "fresh_restart"),
            )
        supervisor_status = supervisor.write_status(
            state="preparing", retries_used=0,
            recovery_mode=(
                "checkpoint_resume" if latest is not None else "fresh_restart"),
        )

    while True:
        # A retry always re-reads and authenticates the ledger. At step zero this
        # remains a fresh start; once a checkpoint exists it becomes a resume.
        latest = ledger.latest()
        previous = latest if (arguments.resume or retries_used > 0) else None
        completed_steps = previous.step if previous else 0
        if completed_steps > total_steps:
            raise HarnessError("latest checkpoint exceeds the configured iteration count")
        if previous is not None and completed_steps == total_steps:
            manifest_path = _publish_from_checkpoint(
                arguments, config, dataset, base_receipt, versions, previous,
                training_data_receipt=training_data_receipt,
                parent_adapter=parent_adapter)
            if supervisor is not None:
                supervisor.append(
                    "session_completed", checkpoint=_checkpoint_status(previous))
                supervisor_status = supervisor.write_status(
                    state="completed", retries_used=retries_used,
                    checkpoint=_checkpoint_status(previous),
                    recovery_mode="checkpoint_resume",
                )
            _write_training_status(
                arguments.output, state="completed", run_contract=run_contract,
                schedule=schedule, resume_command=resume_command,
                total_steps=total_steps, checkpoint=previous,
                supervisor_status=supervisor_status)
            print(f"published immutable academic adapter: {manifest_path}")
            return 0

        attempt_id = uuid.uuid4().hex
        recovery_mode = (
            "checkpoint_resume" if previous is not None else "fresh_restart")
        work_adapter = arguments.output / ".work" / f"adapter-{attempt_id}"
        work_adapter.mkdir(parents=True)
        initialization_config, initialization_weights = _select_adapter_initialization(
            previous, parent_adapter)
        if initialization_config is not None:
            atomic_write_bytes(
                work_adapter / "adapter_config.json",
                initialization_config.read_bytes())
        yaml_path = arguments.output / ".work" / f"mlx-lora-{attempt_id}.yaml"
        atomic_write_bytes(
            yaml_path,
            yaml_training_config(
                runtime_config, model_view, trainer_data_dir, work_adapter,
                iterations=total_steps - completed_steps,
                resume_adapter=initialization_weights,
            ).encode("utf-8"),
        )
        command = mlx_training_command(
            yaml_path, python_executable=arguments.python)
        print("command:", " ".join(command))
        if supervisor is not None:
            supervisor.append(
                "attempt_started", attempt_id=attempt_id,
                retries_used=retries_used, recovery_mode=recovery_mode,
                checkpoint=_checkpoint_status(previous),
                remaining_steps=total_steps - completed_steps,
            )
            supervisor_status = supervisor.write_status(
                state="prepared", retries_used=retries_used,
                attempt_id=attempt_id, recovery_mode=recovery_mode,
                checkpoint=_checkpoint_status(previous),
            )
        _write_training_status(
            arguments.output, state="prepared", run_contract=run_contract,
            schedule=schedule, resume_command=resume_command,
            total_steps=total_steps, checkpoint=previous,
            attempt_id=attempt_id, supervisor_status=supervisor_status)
        if not arguments.execute:
            print("preflight complete; pass --execute to load the pinned 27B model")
            return 0

        lease = TrainingComputeLease(arguments.lease_path)
        if supervisor is not None:
            supervisor_status = supervisor.write_status(
                state="starting", retries_used=retries_used,
                attempt_id=attempt_id, recovery_mode=recovery_mode,
                checkpoint=_checkpoint_status(previous),
            )
        _write_training_status(
            arguments.output, state="starting", run_contract=run_contract,
            schedule=schedule, resume_command=resume_command,
            total_steps=total_steps, checkpoint=previous,
            attempt_id=attempt_id, supervisor_status=supervisor_status)
        try:
            lease.acquire(owner={
                "model": config["base_model"]["model_id"],
                "revision": config["base_model"]["revision"],
                "run_id": run_contract["run_identity"],
                "attempt_id": attempt_id,
            })
            if supervisor is not None:
                supervisor_status = supervisor.write_status(
                    state="running", retries_used=retries_used,
                    attempt_id=attempt_id, recovery_mode=recovery_mode,
                    checkpoint=_checkpoint_status(previous),
                )
            _write_training_status(
                arguments.output, state="running", run_contract=run_contract,
                schedule=schedule, resume_command=resume_command,
                total_steps=total_steps, checkpoint=previous,
                attempt_id=attempt_id, supervisor_status=supervisor_status)
            with lease:
                verify_ollama_empty(arguments.ollama_url)
                result = run_training_process(
                    command, work_adapter, ledger,
                    cumulative_offset=completed_steps,
                    inherited_lease_fd=lease.descriptor,
                    inherited_run_lock_fd=run_lock.descriptor,
                    metrics_path=arguments.output / "training-metrics.jsonl",
                    metrics_mode="production", metrics_run_id=attempt_id,
                    trainer_log_path=arguments.output / "trainer.log")
        except BaseException as exc:
            durable = ledger.latest()
            if supervisor is not None:
                supervisor.append(
                    "attempt_interrupted", attempt_id=attempt_id,
                    retries_used=retries_used, error=str(exc),
                    checkpoint=_checkpoint_status(durable),
                    automatic_retry=False,
                )
                supervisor_status = supervisor.write_status(
                    state="interrupted", retries_used=retries_used,
                    attempt_id=attempt_id, recovery_mode=(
                        "checkpoint_resume" if durable is not None
                        else "fresh_restart"),
                    checkpoint=_checkpoint_status(durable), error=str(exc),
                )
            _write_training_status(
                arguments.output, state="interrupted", run_contract=run_contract,
                schedule=schedule, resume_command=resume_command,
                total_steps=total_steps, checkpoint=durable,
                attempt_id=attempt_id, error=str(exc),
                supervisor_status=supervisor_status)
            print(
                "training interrupted; durable checkpoint step: "
                f"{durable.step if durable else 0}", file=sys.stderr)
            print(f"resume with: {resume_command}", file=sys.stderr)
            raise

        if result:
            durable = ledger.latest()
            decision = plan_training_retry(
                result, retries_used=retries_used,
                max_retries=arguments.max_retries if supervisor is not None else 0,
                base_delay_seconds=arguments.retry_base_seconds,
                maximum_delay_seconds=arguments.retry_max_seconds,
                checkpoint=durable,
            )
            error = f"MLX-LM trainer exited with status {result}"
            if supervisor is not None:
                supervisor.append(
                    "attempt_failed", attempt_id=attempt_id,
                    retries_used=retries_used, exit_status=result,
                    signal=decision.signal_name, reason=decision.reason,
                    checkpoint=_checkpoint_status(durable),
                    automatic_retry=decision.retry,
                    recovery_mode=decision.recovery_mode,
                )
            if decision.retry:
                retries_used += 1
                delay = float(decision.delay_seconds or 0)
                next_retry_at = time.time() + delay
                if supervisor is not None:
                    supervisor.append(
                        "retry_scheduled", attempt_id=attempt_id,
                        retries_used=retries_used,
                        delay_seconds=delay,
                        next_retry_at_epoch=next_retry_at,
                        recovery_mode=decision.recovery_mode,
                        checkpoint=_checkpoint_status(durable),
                    )
                    supervisor_status = supervisor.write_status(
                        state="retry_wait", retries_used=retries_used,
                        attempt_id=attempt_id,
                        recovery_mode=decision.recovery_mode,
                        checkpoint=_checkpoint_status(durable),
                        next_retry_at_epoch=next_retry_at,
                        last_exit_status=result,
                        last_signal=decision.signal_name,
                        error=error,
                    )
                _write_training_status(
                    arguments.output, state="retry_wait",
                    run_contract=run_contract, schedule=schedule,
                    resume_command=resume_command, total_steps=total_steps,
                    checkpoint=durable, attempt_id=attempt_id, error=error,
                    supervisor_status=supervisor_status)
                print(
                    f"trainer {decision.signal_name or result} at durable step "
                    f"{durable.step if durable else 0}; {decision.recovery_mode} "
                    f"in {delay:g}s (retry {retries_used}/{arguments.max_retries})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue

            if supervisor is not None:
                supervisor_status = supervisor.write_status(
                    state="failed", retries_used=retries_used,
                    attempt_id=attempt_id,
                    recovery_mode=decision.recovery_mode,
                    checkpoint=_checkpoint_status(durable),
                    last_exit_status=result,
                    last_signal=decision.signal_name,
                    error=error,
                )
            _write_training_status(
                arguments.output, state="failed", run_contract=run_contract,
                schedule=schedule, resume_command=resume_command,
                total_steps=total_steps, checkpoint=durable,
                attempt_id=attempt_id, error=error,
                supervisor_status=supervisor_status)
            print(
                "training failed; durable checkpoint step: "
                f"{durable.step if durable else 0}", file=sys.stderr)
            print(f"resume with: {resume_command}", file=sys.stderr)
            raise HarnessError(error)

        try:
            final_weights = work_adapter / "adapters.safetensors"
            final_config = work_adapter / "adapter_config.json"
            completed = ledger.capture(
                final_weights, total_steps, adapter_config=final_config)
            adapter_manifest_path = _publish_from_checkpoint(
                arguments, config, dataset, base_receipt, versions, completed,
                training_data_receipt=training_data_receipt,
                parent_adapter=parent_adapter)
        except BaseException as exc:
            durable = ledger.latest()
            if supervisor is not None:
                supervisor.append(
                    "finalization_interrupted", attempt_id=attempt_id,
                    retries_used=retries_used, error=str(exc),
                    checkpoint=_checkpoint_status(durable),
                )
                supervisor_status = supervisor.write_status(
                    state="interrupted", retries_used=retries_used,
                    attempt_id=attempt_id, recovery_mode=(
                        "checkpoint_resume" if durable is not None
                        else "fresh_restart"),
                    checkpoint=_checkpoint_status(durable), error=str(exc),
                )
            _write_training_status(
                arguments.output, state="interrupted", run_contract=run_contract,
                schedule=schedule, resume_command=resume_command,
                total_steps=total_steps, checkpoint=durable,
                attempt_id=attempt_id, error=str(exc),
                supervisor_status=supervisor_status)
            print(f"finalization interrupted; resume with: {resume_command}",
                  file=sys.stderr)
            raise
        if supervisor is not None:
            supervisor.append(
                "session_completed", attempt_id=attempt_id,
                retries_used=retries_used,
                checkpoint=_checkpoint_status(completed),
            )
            supervisor_status = supervisor.write_status(
                state="completed", retries_used=retries_used,
                attempt_id=attempt_id, recovery_mode="checkpoint_resume",
                checkpoint=_checkpoint_status(completed),
            )
        _write_training_status(
            arguments.output, state="completed", run_contract=run_contract,
            schedule=schedule, resume_command=resume_command,
            total_steps=total_steps, checkpoint=completed,
            attempt_id=attempt_id, supervisor_status=supervisor_status)
        print(f"published immutable academic adapter: {adapter_manifest_path}")
        return 0


def _feasibility(arguments: argparse.Namespace, config: dict, dataset: dict) -> int:
    iterations = arguments.feasibility_iters
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 4:
        raise HarnessError("--feasibility-iters must be between 1 and 4")
    if arguments.model is None:
        raise HarnessError("27B feasibility requires the exact local --model snapshot")
    if arguments.output.exists():
        raise HarnessError(
            "feasibility output must be a new path; refusing to touch an existing production/run directory")
    feasibility_root = arguments.output / "FEASIBILITY_ONLY"
    feasibility_root.mkdir(parents=True)
    versions = python_package_versions(arguments.python)
    report = run_preflight(
        config, arguments.model, arguments.data_dir, feasibility_root,
        package_versions=versions)
    atomic_write_json(feasibility_root / "preflight.json", report.as_dict())
    report.require_ok()
    base_receipt = report.facts["base_model"]
    model_view = create_text_training_view(
        arguments.model, arguments.model_view_cache, base_receipt)
    run_id = "feasibility-" + uuid.uuid4().hex
    marker = {
        "schema_version": "spiral.academic-27b-feasibility.v1",
        "deployable": False,
        "academic_adapter_manifest_allowed": False,
        "iterations": iterations,
        "run_id": run_id,
        "base_model": base_receipt,
        "training_contract": {
            "exact_production_config": True,
            "exact_hybrid_lora_targets": list(config["lora"]["keys"]),
            "batch_size": config["training"]["batch_size"],
            "grad_accumulation_steps": config["training"]["grad_accumulation_steps"],
            "gradient_checkpointing": config["training"]["grad_checkpoint"],
            "max_seq_length": config["training"]["max_seq_length"],
        },
    }
    atomic_write_json(feasibility_root / "FEASIBILITY_ONLY.json", marker)
    work_adapter = feasibility_root / ".work-adapter"
    work_adapter.mkdir()
    yaml_path = feasibility_root / "mlx_lora.yaml"
    atomic_write_bytes(
        yaml_path,
        yaml_training_config(
            config, model_view, arguments.data_dir, work_adapter,
            iterations=iterations).encode("utf-8"))
    command = mlx_training_command(yaml_path, python_executable=arguments.python)
    metrics_path = feasibility_root / "training-metrics.jsonl"
    print("FEASIBILITY_ONLY: exact 27B load test; no adapter manifest can be emitted")
    print("command:", " ".join(command))
    print(f"live loss view: {feasibility_root / 'loss-curves.html'}")
    print(f"machine metrics: {metrics_path}")
    if not arguments.execute:
        return 0
    ledger = CheckpointLedger(feasibility_root / "checkpoints")
    lease = TrainingComputeLease(arguments.lease_path)
    lease.acquire(owner={
        "model": config["base_model"]["model_id"],
        "revision": config["base_model"]["revision"],
        "run_id": run_id,
        "operation": "academic_27b_feasibility",
        "feasibility_only": True,
    })
    with lease:
        verify_ollama_empty(arguments.ollama_url)
        result = run_training_process(
            command, work_adapter, ledger,
            inherited_lease_fd=lease.descriptor,
            metrics_path=metrics_path, metrics_mode="feasibility_only",
            metrics_run_id=run_id,
            trainer_log_path=feasibility_root / "trainer.log")
    if result:
        raise HarnessError(f"MLX 27B feasibility trainer exited with status {result}")
    records = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_records = [row for row in records if row.get("event") == "train" and row.get("valid")]
    validation_records = [
        row for row in records if row.get("event") == "validation" and row.get("valid")]
    complete = {
        **marker,
        "completed": True,
        "weights_unloaded": True,
        "lease_released_after_child_exit": True,
        "training_metrics_sha256": sha256_file(metrics_path),
        "observed": {
            "last_train_loss": train_records[-1].get("train_loss") if train_records else None,
            "last_val_loss": validation_records[-1].get("val_loss") if validation_records else None,
            "peak_memory_gb": max(
                (float(row["peak_memory_gb"]) for row in train_records), default=None),
        },
        "academic_adapter_manifest_emitted": False,
    }
    atomic_write_json(feasibility_root / "FEASIBILITY_COMPLETE.json", complete)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        config = load_toml_config(arguments.config)
        dataset = _prepare(arguments, config)
        if arguments.smoke_model is not None and arguments.feasibility_iters is not None:
            raise HarnessError("--smoke-model and --feasibility-iters are mutually exclusive")
        if arguments.initialize_from_manifest is not None and (
            arguments.smoke_model is not None or arguments.feasibility_iters is not None
        ):
            raise HarnessError(
                "--initialize-from-manifest is production-only and cannot be used "
                "for smoke or feasibility runs")
        if arguments.supervise and not arguments.execute:
            raise HarnessError("--supervise requires --execute")
        if arguments.supervise and (
            arguments.smoke_model is not None
            or arguments.feasibility_iters is not None
        ):
            raise HarnessError("--supervise is production-only")
        if arguments.supervise:
            plan_training_retry(
                0, retries_used=0, max_retries=arguments.max_retries,
                base_delay_seconds=arguments.retry_base_seconds,
                maximum_delay_seconds=arguments.retry_max_seconds,
                checkpoint=None,
            )
        if arguments.smoke_model is not None:
            return _run_smoke(arguments, config, dataset)
        if arguments.feasibility_iters is not None:
            return _feasibility(arguments, config, dataset)
        return _production(arguments, config, dataset)
    except HarnessError as exc:
        print(f"academic QLoRA: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
