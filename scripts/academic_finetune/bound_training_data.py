"""Build a deterministic, tokenizer-gated, train-only MLX data view.

MLX-LM truncates examples at ``max_seq_length`` inside the trainer.  That is
not acceptable for completion-only academic supervision: truncation can
silently remove the end of the human completion.  This module applies the
same two ``tokenizer.apply_chat_template`` calls as MLX-LM's
``CompletionsDataset.process`` before training and replaces only oversized
paragraph rows with losslessly partitioned, contiguous sentence chunks.

The production tokenizer loader imports only MLX-LM's tokenizer utility.  It
never calls ``mlx_lm.load`` or opens a ``*.safetensors`` model-weight shard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.academic_finetune.text import (
    certainty,
    citation_count,
    citation_markers,
    claims_are_feasible,
    context_is_safe,
    neutral_claims,
    rhetorical_relation,
    sentences,
)
from scripts.academic_finetune.training_support import (
    format_academic_prompt,
    target_context_leakage_reason,
)


BOUND_TRAINING_DATA_SCHEMA = "spiral.academic-bounded-training-data.v1"
DERIVATION_SCHEMA = "spiral.academic-completion-chunk.v1"
_PREVIOUS_CONTEXT_START = "\n\nPrevious context:\n"
_PREVIOUS_CONTEXT_END = "\n\nClaims to express:\n"
_SPACE = re.compile(r"\s+")
_TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


class BoundTrainingDataError(RuntimeError):
    """A fail-closed error raised before an unsafe row reaches MLX-LM."""


@dataclass(frozen=True)
class SequenceMeasurement:
    """Exact lengths returned by MLX-LM's completion-dataset template calls."""

    total_tokens: int
    prompt_offset: int
    completion_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_offset": self.prompt_offset,
            "completion_tokens": self.completion_tokens,
        }


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalise_space(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _tokenizer_files_identity(model_path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name in _TOKENIZER_FILES:
        path = model_path / name
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not any(entry["path"] in {"tokenizer.json", "tokenizer.model"} for entry in files):
        raise BoundTrainingDataError(
            f"local tokenizer assets are missing from {model_path}"
        )
    contract = {
        "loader": "mlx_lm.tokenizer_utils.load",
        "local_only": True,
        "files": files,
    }
    return {
        **contract,
        "identity": _sha256_bytes(_canonical_json(contract)),
    }


def load_tokenizer_only(model_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load the local MLX tokenizer without importing or loading model weights."""

    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise BoundTrainingDataError(
            f"tokenizer model path is not a local directory: {model_path}"
        )
    identity = _tokenizer_files_identity(model_path)
    try:
        # Deliberately import tokenizer_utils directly.  mlx_lm.utils.load and
        # mlx_lm.load both construct the model and are forbidden in this path.
        from mlx_lm.tokenizer_utils import load as mlx_load_tokenizer
    except ImportError as exc:  # pragma: no cover - exercised in production runtime
        raise BoundTrainingDataError(
            "MLX-LM tokenizer support is not installed in this Python runtime"
        ) from exc
    try:
        tokenizer = mlx_load_tokenizer(
            model_path,
            tokenizer_config_extra={
                "local_files_only": True,
                "trust_remote_code": True,
            },
        )
    except Exception as exc:  # pragma: no cover - tokenizer library error detail varies
        raise BoundTrainingDataError(
            f"could not load the local tokenizer from {model_path}: {exc}"
        ) from exc
    return tokenizer, identity


def measure_completion_sequence(
    tokenizer: Any, row: Mapping[str, Any]
) -> SequenceMeasurement:
    """Mirror ``mlx_lm.tuner.datasets.CompletionsDataset.process`` exactly."""

    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BoundTrainingDataError("training row prompt must be a non-empty string")
    if not isinstance(completion, str) or not completion.strip():
        raise BoundTrainingDataError("training row completion must be a non-empty string")
    tools = row.get("tools", None)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_dict=False,
        )
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            add_generation_prompt=True,
            return_dict=False,
        )
        total_length = len(tokens)
        prompt_offset = len(prompt_tokens)
    except Exception as exc:
        raise BoundTrainingDataError(
            f"tokenizer.apply_chat_template failed for row {row.get('example_id')!r}: {exc}"
        ) from exc
    if total_length <= 0:
        raise BoundTrainingDataError("chat template returned an empty training sequence")
    if prompt_offset <= 0 or prompt_offset >= total_length:
        raise BoundTrainingDataError(
            "completion-only prompt offset must be positive and smaller than total tokens "
            f"(offset={prompt_offset}, total={total_length})"
        )
    return SequenceMeasurement(
        total_tokens=total_length,
        prompt_offset=prompt_offset,
        completion_tokens=total_length - prompt_offset,
    )


def _extract_original_context(prompt: str) -> str:
    before, separator, remainder = prompt.partition(_PREVIOUS_CONTEXT_START)
    del before
    if not separator:
        raise BoundTrainingDataError(
            "oversized row does not follow the attested academic prompt contract "
            "(Previous context marker missing)"
        )
    context, separator, _claims = remainder.partition(_PREVIOUS_CONTEXT_END)
    if not separator or not context.strip():
        raise BoundTrainingDataError(
            "oversized row does not contain a recoverable original context"
        )
    return context.strip()


def _derived_example_id(
    source_example_id: str,
    start_sentence: int,
    stop_sentence: int,
    completion: str,
) -> str:
    contract = (
        f"{DERIVATION_SCHEMA}\0{source_example_id}\0"
        f"{start_sentence}:{stop_sentence}\0{_sha256_bytes(completion.encode('utf-8'))}"
    )
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()


def _derived_row(
    source: Mapping[str, Any],
    *,
    source_example_id: str,
    completion: str,
    context: str,
    start_sentence: int,
    stop_sentence: int,
) -> dict[str, Any]:
    sentence_count = stop_sentence - start_sentence
    task_type = "paragraph" if sentence_count >= 2 else "sentence"
    claims = neutral_claims(
        completion,
        maximum=6,
        paragraph=task_type == "paragraph",
    )
    if not claims or not claims_are_feasible(claims, completion):
        raise BoundTrainingDataError(
            f"could not construct a non-leaking feasible plan for {source_example_id} "
            f"sentences {start_sentence + 1}-{stop_sentence}"
        )
    if not context_is_safe(context, completion):
        raise BoundTrainingDataError(
            f"derived context leaks completion for {source_example_id} "
            f"sentences {start_sentence + 1}-{stop_sentence}"
        )
    leakage_reason = target_context_leakage_reason(completion, context)
    if leakage_reason:
        raise BoundTrainingDataError(
            f"derived context leaks completion for {source_example_id}: {leakage_reason}"
        )
    prompt_row = {
        "task_type": task_type,
        "input": {
            "context": context,
            "claims": claims,
            "rhetorical_relation": rhetorical_relation(completion),
            "certainty": certainty(completion),
            "citation_count": citation_count(completion),
            "citation_slots": citation_markers(completion),
            "construction_method": "bounded_semantic_roles_v3",
        },
    }
    row = dict(source)
    row.update(
        {
            "completion": completion,
            "example_id": _derived_example_id(
                source_example_id,
                start_sentence,
                stop_sentence,
                completion,
            ),
            "prompt": format_academic_prompt(prompt_row),
            "task_type": task_type,
        }
    )
    return row


def _source_sentences(completion: str, example_id: str) -> list[str]:
    result = sentences(completion.strip())
    if not result:
        raise BoundTrainingDataError(
            f"oversized paragraph {example_id} has no sentence boundaries"
        )
    if _normalise_space(" ".join(result)) != _normalise_space(completion):
        raise BoundTrainingDataError(
            f"sentence splitter would not preserve completion {example_id} exactly"
        )
    return result


def _partition_oversized_paragraph(
    tokenizer: Any,
    source: Mapping[str, Any],
    *,
    max_sequence_length: int,
) -> tuple[list[tuple[dict[str, Any], SequenceMeasurement]], dict[str, Any]]:
    example_id = str(source.get("example_id", ""))
    if source.get("task_type") != "paragraph":
        raise BoundTrainingDataError(
            f"oversized non-paragraph row {example_id!r} cannot be shortened without "
            "truncating its completion"
        )
    completion = str(source["completion"]).strip()
    original_context = _extract_original_context(str(source["prompt"]))
    original_sentences = _source_sentences(completion, example_id)
    chosen: list[tuple[dict[str, Any], SequenceMeasurement]] = []
    cursor = 0
    while cursor < len(original_sentences):
        context = (
            original_context
            if cursor == 0
            else " ".join(original_sentences[max(0, cursor - 2) : cursor])
        )
        accepted: tuple[dict[str, Any], SequenceMeasurement, int] | None = None
        first_error: BoundTrainingDataError | None = None
        # Descending stop indices implement the deterministic longest-fitting
        # contiguous greedy partition requested by the training contract.
        for stop in range(len(original_sentences), cursor, -1):
            candidate_completion = " ".join(original_sentences[cursor:stop])
            try:
                candidate = _derived_row(
                    source,
                    source_example_id=example_id,
                    completion=candidate_completion,
                    context=context,
                    start_sentence=cursor,
                    stop_sentence=stop,
                )
                measurement = measure_completion_sequence(tokenizer, candidate)
            except BoundTrainingDataError as exc:
                if first_error is None:
                    first_error = exc
                continue
            if measurement.total_tokens <= max_sequence_length:
                accepted = candidate, measurement, stop
                break
        if accepted is None:
            detail = f": {first_error}" if first_error is not None else ""
            raise BoundTrainingDataError(
                f"one complete sentence from oversized paragraph {example_id!r} cannot "
                f"fit within {max_sequence_length} tokens without truncation{detail}"
            )
        candidate, measurement, stop = accepted
        chosen.append((candidate, measurement))
        cursor = stop

    derived_sentences: list[str] = []
    for row, _measurement in chosen:
        derived_sentences.extend(sentences(str(row["completion"])))
    if derived_sentences != original_sentences:
        raise BoundTrainingDataError(
            f"derived chunks do not preserve every sentence exactly once for {example_id}"
        )
    reconstructed = " ".join(str(row["completion"]) for row, _measurement in chosen)
    if _normalise_space(reconstructed) != _normalise_space(completion):
        raise BoundTrainingDataError(
            f"derived chunks do not reconstruct completion {example_id}"
        )
    mapping = {
        "source_example_id": example_id,
        "action": "partitioned",
        "source_completion_sha256": _sha256_bytes(completion.encode("utf-8")),
        "source_completion_normalized_sha256": _sha256_bytes(
            _normalise_space(completion).encode("utf-8")
        ),
        "reconstructed_completion_normalized_sha256": _sha256_bytes(
            _normalise_space(reconstructed).encode("utf-8")
        ),
        "source_sentence_count": len(original_sentences),
        "output_example_ids": [row["example_id"] for row, _measurement in chosen],
        "output_sentence_ranges": [
            {
                "start": sum(
                    len(sentences(str(previous[0]["completion"])))
                    for previous in chosen[:index]
                ),
                "stop": sum(
                    len(sentences(str(previous[0]["completion"])))
                    for previous in chosen[: index + 1]
                ),
            }
            for index in range(len(chosen))
        ],
        "output_task_types": [row["task_type"] for row, _measurement in chosen],
        "output_measurements": [
            measurement.as_dict() for _row, measurement in chosen
        ],
        "preservation": "all source sentences occur exactly once in original order",
    }
    return chosen, mapping


def _read_source_rows(path: Path) -> tuple[bytes, list[tuple[bytes, dict[str, Any]]]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BoundTrainingDataError(f"cannot read source training split {path}: {exc}") from exc
    if not payload or not payload.endswith(b"\n"):
        raise BoundTrainingDataError("source train.jsonl must be non-empty and newline terminated")
    result: list[tuple[bytes, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        if not line.strip():
            raise BoundTrainingDataError(
                f"source train.jsonl contains a blank line at {line_number}"
            )
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BoundTrainingDataError(
                f"invalid source training JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise BoundTrainingDataError(
                f"source training row {line_number} must be a JSON object"
            )
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise BoundTrainingDataError(
                f"source training row {line_number} has no example_id"
            )
        if example_id in seen_ids:
            raise BoundTrainingDataError(f"duplicate source example_id {example_id!r}")
        seen_ids.add(example_id)
        result.append((line, row))
    return payload, result


def _source_train_path(
    data_dir: Path, dataset_manifest: Mapping[str, Any] | None
) -> tuple[Path, str | None, int | None]:
    manifest = dataset_manifest
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest is None and manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BoundTrainingDataError(
                f"cannot read prepared dataset manifest {manifest_path}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise BoundTrainingDataError("prepared dataset manifest must be an object")
        manifest = loaded
    if manifest is None:
        return data_dir / "train.jsonl", None, None
    entry = manifest.get("splits", {}).get("train", {})
    if not isinstance(entry, Mapping):
        raise BoundTrainingDataError("prepared dataset manifest has no train split")
    relative = str(entry.get("path", "train.jsonl"))
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise BoundTrainingDataError("prepared train split path is unsafe")
    expected_hash = entry.get("sha256")
    expected_count = entry.get("count")
    return (
        data_dir / relative,
        str(expected_hash) if expected_hash is not None else None,
        int(expected_count) if isinstance(expected_count, int) else None,
    )


def _coerce_tokenizer_identity(value: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, str):
        if not value.strip():
            raise BoundTrainingDataError("tokenizer identity must not be empty")
        return {"identity": value.strip(), "loader": "injected"}
    result = dict(value)
    if not result:
        raise BoundTrainingDataError("tokenizer identity must not be empty")
    # Fail before writing anything if a fake or injected identity is not stable JSON.
    _canonical_json(result)
    if not isinstance(result.get("identity"), str) or not result["identity"]:
        contract = dict(result)
        result["identity"] = _sha256_bytes(_canonical_json(contract))
    return result


def create_bounded_training_data_view(
    data_dir: Path,
    output_root: Path,
    model_path: Path | None = None,
    *,
    max_sequence_length: int = 512,
    tokenizer: Any | None = None,
    tokenizer_identity: Mapping[str, Any] | str | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create an atomic content-addressed train-only view safe at ``max_seq_length``.

    ``tokenizer`` and ``tokenizer_identity`` are injectable for model-free tests.
    Production callers should pass only the pinned local ``model_path``; loading
    that tokenizer never loads any model-weight shard.
    """

    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length <= 1
    ):
        raise BoundTrainingDataError("max_sequence_length must be an integer greater than 1")
    data_dir = data_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_path, expected_hash, expected_count = _source_train_path(
        data_dir, dataset_manifest
    )
    if tokenizer is None:
        if model_path is None:
            raise BoundTrainingDataError(
                "model_path is required when no tokenizer is injected"
            )
        tokenizer, loaded_identity = load_tokenizer_only(model_path)
        if tokenizer_identity is not None:
            raise BoundTrainingDataError(
                "tokenizer_identity must not override the production tokenizer receipt"
            )
        identity = loaded_identity
    else:
        if tokenizer_identity is None:
            raise BoundTrainingDataError(
                "an injected tokenizer requires an explicit deterministic tokenizer_identity"
            )
        identity = _coerce_tokenizer_identity(tokenizer_identity)

    source_payload, source_rows = _read_source_rows(source_path)
    source_hash = _sha256_bytes(source_payload)
    if expected_hash is not None and source_hash != expected_hash:
        raise BoundTrainingDataError(
            "source train.jsonl bytes do not match the prepared dataset manifest"
        )
    if expected_count is not None and len(source_rows) != expected_count:
        raise BoundTrainingDataError(
            "source train.jsonl count does not match the prepared dataset manifest"
        )

    output_lines: list[bytes] = []
    unchanged_lines: list[bytes] = []
    derived_lines: list[bytes] = []
    mappings: list[dict[str, Any]] = []
    maximum_total = 0
    maximum_prompt = 0
    maximum_completion = 0
    source_maximum_total = 0
    for raw_line, row in source_rows:
        measurement = measure_completion_sequence(tokenizer, row)
        source_maximum_total = max(source_maximum_total, measurement.total_tokens)
        if measurement.total_tokens <= max_sequence_length:
            # Preserve bounded source rows byte-for-byte, including key order and
            # JSON escaping; this is stronger than preserving only their values.
            output_lines.append(raw_line)
            unchanged_lines.append(raw_line)
            mappings.append(
                {
                    "source_example_id": row["example_id"],
                    "action": "unchanged",
                    "source_completion_sha256": _sha256_bytes(
                        str(row["completion"]).encode("utf-8")
                    ),
                    "output_example_ids": [row["example_id"]],
                    "output_measurements": [measurement.as_dict()],
                }
            )
            measurements = [measurement]
        else:
            chunks, mapping = _partition_oversized_paragraph(
                tokenizer,
                row,
                max_sequence_length=max_sequence_length,
            )
            mappings.append(mapping)
            measurements = []
            for chunk, chunk_measurement in chunks:
                if chunk_measurement.total_tokens > max_sequence_length:
                    raise BoundTrainingDataError(
                        "internal error: derived row exceeds the configured token gate"
                    )
                line = _canonical_json(chunk)
                output_lines.append(line)
                derived_lines.append(line)
                measurements.append(chunk_measurement)
        for item in measurements:
            maximum_total = max(maximum_total, item.total_tokens)
            maximum_prompt = max(maximum_prompt, item.prompt_offset)
            maximum_completion = max(maximum_completion, item.completion_tokens)

    output_payload = b"".join(output_lines)
    if not output_payload:
        raise BoundTrainingDataError("bounded training view would be empty")
    output_count = len(output_lines)
    partitioned_mappings = [item for item in mappings if item["action"] == "partitioned"]
    unchanged_count = len(mappings) - len(partitioned_mappings)
    derived_count = len(derived_lines)
    mapping_hash = _sha256_bytes(_canonical_json(mappings))
    identity_contract = {
        "schema_version": BOUND_TRAINING_DATA_SCHEMA,
        "source_train_sha256": source_hash,
        "source_train_count": len(source_rows),
        "tokenizer_identity": identity["identity"],
        "max_sequence_length": max_sequence_length,
        "partition_policy": (
            "unchanged-if-bounded; otherwise longest-fitting contiguous sentence chunks; "
            "never truncate completion"
        ),
        "mapping_sha256": mapping_hash,
        "output_train_sha256": _sha256_bytes(output_payload),
    }
    view_identity = _sha256_bytes(_canonical_json(identity_contract))
    receipt = {
        "schema_version": BOUND_TRAINING_DATA_SCHEMA,
        "view_identity": view_identity,
        "identity_contract": identity_contract,
        "source": {
            "data_directory": str(data_dir),
            "train_path": str(source_path.resolve()),
            "train_sha256": source_hash,
            "train_count": len(source_rows),
            "maximum_total_tokens_before_transform": source_maximum_total,
        },
        "tokenizer": identity,
        "gate": {
            "method": "mlx_lm.CompletionsDataset.apply_chat_template parity",
            "max_sequence_length": max_sequence_length,
            "prompt_offset_rule": "0 < prompt_offset < total_tokens",
            "completion_policy": "never truncate",
            "maximum_total_tokens": maximum_total,
            "maximum_prompt_offset": maximum_prompt,
            "maximum_completion_tokens": maximum_completion,
        },
        "output": {
            "train_path": "train.jsonl",
            "train_sha256": _sha256_bytes(output_payload),
            "train_count": output_count,
            "exposed_files": ["train.jsonl"],
            "omitted_files": ["valid.jsonl", "test.jsonl"],
            "unchanged_source_rows": unchanged_count,
            "partitioned_source_rows": len(partitioned_mappings),
            "derived_rows": derived_count,
            "unchanged_rows_sha256": _sha256_bytes(b"".join(unchanged_lines)),
            "derived_rows_sha256": _sha256_bytes(b"".join(derived_lines)),
        },
        "preservation": {
            "mapping_sha256": mapping_hash,
            "all_source_completions_preserved": True,
            "partitioned_completion_check": (
                "normalized concatenation and exact ordered sentence equality"
            ),
        },
        "mappings": mappings,
    }

    parent = output_root / ".work" / "bounded-trainer-data"
    destination = parent / view_identity[:20]
    receipt_payload = _canonical_json(receipt)
    if destination.exists():
        if not destination.is_dir():
            raise BoundTrainingDataError(
                f"bounded trainer-data destination is not a directory: {destination}"
            )
        try:
            existing_receipt = (destination / "view.json").read_bytes()
        except OSError as exc:
            raise BoundTrainingDataError(
                f"cannot verify existing bounded trainer-data view: {exc}"
            ) from exc
        if existing_receipt != receipt_payload:
            raise BoundTrainingDataError(
                "existing bounded trainer-data receipt does not match this transformation"
            )
        train_path = destination / "train.jsonl"
        unexpected = [
            name for name in ("valid.jsonl", "test.jsonl") if (destination / name).exists()
        ]
        if (
            not train_path.is_file()
            or _sha256_file(train_path) != receipt["output"]["train_sha256"]
            or unexpected
        ):
            raise BoundTrainingDataError(
                "existing bounded trainer-data view failed its integrity gate"
            )
        return destination, receipt

    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    staging.mkdir()
    try:
        _atomic_bytes(staging / "train.jsonl", output_payload)
        _atomic_bytes(staging / "view.json", receipt_payload)
        _fsync_directory(staging)
        try:
            os.replace(staging, destination)
        except OSError:
            # A concurrent identical builder may have won the atomic publish.
            if not destination.is_dir():
                raise
            shutil.rmtree(staging, ignore_errors=True)
        _fsync_directory(parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not (destination / "train.jsonl").is_file():
        raise BoundTrainingDataError("atomic bounded trainer-data publish failed")
    return destination, receipt

