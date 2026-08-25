#!/usr/bin/env python3
"""Held-out evaluation and blind A/B packets for the academic adapter.

Exact sentence reproduction is reported only as a diagnostic.  The decision
surface is target NLL plus content, claim/argument, epistemic-certainty, and
citation fidelity against the actual held-out human prose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .training_support import (
        HarnessError,
        TrainingComputeLease,
        adapter_bundle_digest,
        atomic_write_bytes,
        atomic_write_json,
        format_academic_prompt,
        read_corpus_records,
        sha256_file,
        verify_ollama_empty,
    )
except ImportError:  # direct script execution
    # Prediction workers execute this file directly so they can use the exact
    # selected MLX Python without relying on an installed console entry point.
    # Put the repository root on sys.path before importing the package: the
    # training helpers also import the shared ``spiral`` prompt contract.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.academic_finetune.training_support import (  # type: ignore
        HarnessError,
        TrainingComputeLease,
        adapter_bundle_digest,
        atomic_write_bytes,
        atomic_write_json,
        format_academic_prompt,
        read_corpus_records,
        sha256_file,
        verify_ollama_empty,
    )


EVALUATION_SCHEMA = "spiral.academic-evaluation.v1"
PREDICTION_SCHEMA = "spiral.academic-predictions.v1"
NLL_SCHEMA = "spiral.academic-target-nll.v1"
BLIND_SCHEMA = "spiral.academic-blind-ab.v1"
POST_TRAINING_EVALUATION_SCHEMA = "spiral.academic-post-training-evaluation.v1"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
CITATION_RE = re.compile(
    r"\[C\d+\]|\[(?:\d+[a-z]?)(?:\s*[,;–-]\s*\d+[a-z]?)*\]|"
    r"\([A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'’-]+))?,\s*(?:19|20)\d{2}[a-z]?\)|"
    r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
    re.IGNORECASE,
)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "these", "this", "those", "to", "was", "were", "which", "with",
})
RELATION_MARKERS: dict[str, tuple[str, ...]] = {
    "contrast": ("however", "whereas", "although", "by contrast", "nevertheless", "but"),
    "causal": ("because", "therefore", "thus", "consequently", "owing to", "due to"),
    "cause": ("because", "therefore", "thus", "consequently", "owing to", "due to"),
    "limitation": ("however", "limited", "limitation", "caveat", "cannot", "uncertain"),
    "implication": ("implies", "suggests", "indicates", "therefore", "hence", "consequently"),
    "synthesis": ("collectively", "together", "overall", "taken together", "while", "both"),
    "comparison": ("whereas", "compared", "similarly", "in contrast", "relative to"),
    "elaboration": ("specifically", "in particular", "moreover", "furthermore", "namely"),
    "evidence": (
        "evidence", "observed", "measured", "reported", "shows", "demonstrates",
        "consistent with", "supports", "data indicate",
    ),
    "definition": (
        "defined as", "we define", "refers to", "denotes", "is called",
        "by definition", "meaning",
    ),
}
HEDGES = frozenset({
    "appear", "appears", "approximately", "could", "likely", "may", "might", "perhaps",
    "possibly", "potentially", "seem", "seems", "suggest", "suggests", "tentative",
})
STRONG_MARKERS = frozenset({
    "always", "certainly", "clearly", "conclusively", "demonstrate", "demonstrates",
    "establish", "establishes", "must", "necessarily", "proves", "will",
})


def _stem(token: str) -> str:
    token = token.casefold()
    for suffix in ("ization", "ational", "fulness", "ously", "ments", "ingly", "ation", "ities", "ness", "ment", "able", "ible", "ally", "ing", "ied", "ies", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)] + ("y" if suffix in {"ied", "ies"} else "")
    return token


def content_tokens(text: str) -> list[str]:
    return [
        _stem(token) for token in TOKEN_RE.findall(text.replace("-", " ").replace("–", " "))
        if token.casefold() not in STOPWORDS and len(token) > 1
    ]


def multiset_f1(reference: str, candidate: str) -> float:
    expected = Counter(content_tokens(reference))
    actual = Counter(content_tokens(candidate))
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def claim_coverage(claims: Sequence[str], candidate: str) -> float:
    candidate_set = set(content_tokens(candidate))
    scores = []
    for claim in claims:
        # Corpus claim slots use ``semantic role — proposition``.  Role/template
        # words are construction metadata, not content the writer should parrot.
        semantic_payload = claim.partition("—")[2].strip() or claim
        claim_set = set(content_tokens(semantic_payload))
        if claim_set:
            scores.append(len(claim_set & candidate_set) / len(claim_set))
    return statistics.fmean(scores) if scores else 1.0


def _contains_marker(text: str, marker: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", text.casefold()) is not None


def argument_score(relation: str, target: str, candidate: str) -> float:
    normalized = relation.casefold().replace("_", " ").replace("-", " ")
    matched_relation = next(
        ((name, markers) for name, markers in RELATION_MARKERS.items() if name in normalized),
        None,
    )
    marker_family = matched_relation[1] if matched_relation else ()
    if not marker_family:
        # For an unknown relation, matching a discourse connective used by the human
        # target is a conservative diagnostic rather than inventing a label ontology.
        marker_family = tuple(
            marker for markers in RELATION_MARKERS.values() for marker in markers
            if _contains_marker(target, marker)
        )
    if not marker_family:
        return 1.0
    target_markers = {marker for marker in marker_family if _contains_marker(target, marker)}
    candidate_markers = {marker for marker in marker_family if _contains_marker(candidate, marker)}
    # When the plan names a known relation, any connective from that semantic
    # family is valid; prose realization is one-to-many.  Unknown relations stay
    # conservative and are compared with connectives found in the human target.
    expected = set(marker_family) if matched_relation else (target_markers or set(marker_family))
    return min(1.0, len(candidate_markers & expected) / max(1, min(1, len(expected))))


def certainty_score(requested: str, target: str, candidate: str) -> float:
    requested = requested.casefold()
    target_tokens = set(TOKEN_RE.findall(target.casefold()))
    candidate_tokens = set(TOKEN_RE.findall(candidate.casefold()))
    target_hedged = bool(target_tokens & HEDGES)
    candidate_hedged = bool(candidate_tokens & HEDGES)
    target_strong = bool(target_tokens & STRONG_MARKERS)
    candidate_strong = bool(candidate_tokens & STRONG_MARKERS)
    wants_hedge = any(word in requested for word in ("tentative", "cautious", "uncertain", "hedged", "moderate"))
    wants_strong = any(word in requested for word in ("strong", "certain", "definitive"))
    if wants_hedge and candidate_strong and not candidate_hedged:
        return 0.0
    if wants_hedge:
        return 1.0 if candidate_hedged else 0.5
    if wants_strong:
        return 1.0 if candidate_strong == target_strong else 0.5
    return 1.0 if (candidate_hedged, candidate_strong) == (target_hedged, target_strong) else 0.5


def citations(text: str) -> list[str]:
    return [re.sub(r"\s+", "", value).casefold() for value in CITATION_RE.findall(text)]


def citation_diagnostics(target: str, candidate: str, expected_count: int, allowed_slots: Sequence[str]) -> dict[str, float | int]:
    target_citations = citations(target)
    candidate_citations = citations(candidate)
    target_set = set(target_citations)
    candidate_set = set(candidate_citations)
    allowed = {slot.casefold() for slot in allowed_slots}
    # Normalized training examples use [C#]. Legacy paper-style citations are still
    # compared to the held-out target and cannot be introduced from nowhere.
    permitted = allowed or target_set
    hallucinated = candidate_set - permitted
    required = allowed or target_set
    recall = len(candidate_set & required) / len(required) if required else (1.0 if not candidate_set else 0.0)
    count_score = max(0.0, 1.0 - abs(len(candidate_citations) - expected_count) / max(1, expected_count))
    precision = 1.0 if not candidate_set else (len(candidate_set - hallucinated) / len(candidate_set))
    fidelity = statistics.fmean((recall, count_score, precision))
    return {
        "fidelity": fidelity,
        "recall": recall,
        "count_score": count_score,
        "hallucinated_identifiers": len(hallucinated),
        "candidate_count": len(candidate_citations),
        "target_count": len(target_citations),
    }


def score_candidate(row: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    target = str(row["target"]).strip()
    prompt_input = row["input"]
    citation = citation_diagnostics(
        target,
        candidate,
        int(prompt_input["citation_count"]),
        list(prompt_input.get("citation_slots") or []),
    )
    return {
        "semantic_content_f1": multiset_f1(target, candidate),
        "claim_coverage": claim_coverage(list(prompt_input["claims"]), candidate),
        "argument_relation": argument_score(
            str(prompt_input["rhetorical_relation"]), target, candidate),
        "certainty_fidelity": certainty_score(str(prompt_input["certainty"]), target, candidate),
        "citation_fidelity": citation["fidelity"],
        "citation": citation,
        # Diagnostic only: academic realization is inherently one-to-many.
        "exact_match_diagnostic": candidate.strip() == target,
    }


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HarnessError(f"cannot read predictions {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{path}:{line_number}: invalid prediction JSON: {exc}") from exc
        if row.get("schema_version", PREDICTION_SCHEMA) != PREDICTION_SCHEMA:
            raise HarnessError(f"{path}:{line_number}: unsupported prediction schema")
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in predictions:
            raise HarnessError(f"{path}:{line_number}: missing or duplicate example_id")
        for arm in ("base", "adapter"):
            value = row.get(arm)
            if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
                raise HarnessError(f"{path}:{line_number}: {arm}.text must be a string")
            nll = value.get("target_nll")
            if nll is not None and (
                not isinstance(nll, (int, float)) or isinstance(nll, bool)
                or not math.isfinite(float(nll)) or float(nll) < 0
            ):
                raise HarnessError(f"{path}:{line_number}: {arm}.target_nll must be finite and non-negative")
        predictions[example_id] = row
    return predictions


def _mean_metric(rows: Sequence[Mapping[str, Any]], arm: str, metric: str) -> float:
    return statistics.fmean(float(row[arm][metric]) for row in rows)


def _load_nll_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read NLL report {path}: {exc}") from exc
    if report.get("schema_version") != NLL_SCHEMA:
        raise HarnessError("NLL report has the wrong schema")
    for arm in ("base", "adapter"):
        value = report.get(arm, {}).get("mean_target_nll")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise HarnessError(f"NLL report is missing {arm}.mean_target_nll")
    return report


def evaluate_predictions(
    corpus_path: Path, predictions_path: Path, output_dir: Path, *,
    seed: int = 24082026, nll_report_path: Path | None = None,
) -> dict[str, Any]:
    held_out = [row for row in read_corpus_records(corpus_path) if row["split"] == "test"]
    predictions = _load_predictions(predictions_path)
    expected_ids = {str(row["example_id"]) for row in held_out}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise HarnessError(
            f"predictions do not exactly cover held-out examples; missing={missing[:5]}, extra={extra[:5]}")
    scored: list[dict[str, Any]] = []
    per_record_nll = True
    for row in sorted(held_out, key=lambda value: value["example_id"]):
        prediction = predictions[str(row["example_id"])]
        entry: dict[str, Any] = {
            "example_id": row["example_id"],
            "source_stratum": row["source"]["stratum"],
            "task_type": row["task_type"],
        }
        for arm in ("base", "adapter"):
            text = str(prediction[arm]["text"])
            entry[arm] = score_candidate(row, text)
            entry[arm]["text"] = text
            nll = prediction[arm].get("target_nll")
            if nll is None:
                per_record_nll = False
            else:
                entry[arm]["target_nll"] = float(nll)
        scored.append(entry)
    external_nll = _load_nll_report(nll_report_path)
    if not per_record_nll and external_nll is None:
        raise HarnessError(
            "held-out target NLL is mandatory: provide it per prediction or with --nll-report")
    if per_record_nll:
        nll = {
            arm: statistics.fmean(float(row[arm]["target_nll"]) for row in scored)
            for arm in ("base", "adapter")
        }
        nll_source = "per_example"
    else:
        assert external_nll is not None
        nll = {arm: float(external_nll[arm]["mean_target_nll"]) for arm in ("base", "adapter")}
        nll_source = "aggregate_mlx_completion_only"

    primary_metrics = (
        "semantic_content_f1", "claim_coverage", "argument_relation",
        "certainty_fidelity", "citation_fidelity",
    )
    arms: dict[str, Any] = {}
    for arm in ("base", "adapter"):
        arms[arm] = {
            "mean_target_nll": nll[arm],
            **{metric: _mean_metric(scored, arm, metric) for metric in primary_metrics},
        }
    diagnostics = {
        arm: {
            "exact_match_rate": statistics.fmean(
                float(row[arm]["exact_match_diagnostic"]) for row in scored),
            "citation_hallucinations": sum(
                int(row[arm]["citation"]["hallucinated_identifiers"]) for row in scored),
        }
        for arm in ("base", "adapter")
    }
    summary = {
        "schema_version": EVALUATION_SCHEMA,
        "held_out_examples": len(scored),
        "corpus_sha256": sha256_file(corpus_path),
        "predictions_sha256": sha256_file(predictions_path),
        "nll_source": nll_source,
        "arms": arms,
        "delta_adapter_minus_base": {
            "mean_target_nll": nll["adapter"] - nll["base"],
            **{
                metric: arms["adapter"][metric] - arms["base"][metric]
                for metric in primary_metrics
            },
        },
        "diagnostics_not_selection_metrics": diagnostics,
        "selection_rule": (
            "Prefer lower held-out target NLL only when argument/citation diagnostics do not regress; "
            "confirm with the separately blinded human A/B packet. Exact match is never a gate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "evaluation_summary.json", summary)
    atomic_write_bytes(
        output_dir / "evaluation_examples.jsonl",
        b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for row in scored
        ),
    )
    _write_blind_packets(held_out, predictions, output_dir, seed=seed)
    return summary


def _write_blind_packets(
    held_out: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]],
    output_dir: Path, *, seed: int,
) -> None:
    packets: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for row in sorted(held_out, key=lambda value: value["example_id"]):
        example_id = str(row["example_id"])
        prediction = predictions[example_id]
        order_bit = hashlib.sha256(f"{seed}\0{example_id}".encode()).digest()[0] & 1
        arm_a, arm_b = (("base", "adapter") if order_bit == 0 else ("adapter", "base"))
        packet_id = hashlib.sha256(f"blind\0{seed}\0{example_id}".encode()).hexdigest()[:20]
        packets.append({
            "schema_version": BLIND_SCHEMA,
            "packet_id": packet_id,
            "task": {
                "task_type": row["task_type"],
                "context": row["input"]["context"],
                "claims": row["input"]["claims"],
                "rhetorical_relation": row["input"]["rhetorical_relation"],
                "certainty": row["input"]["certainty"],
                "citation_slots": row["input"].get("citation_slots", []),
            },
            "candidate_a": prediction[arm_a]["text"],
            "candidate_b": prediction[arm_b]["text"],
            "rubric": [
                "faithfully expresses every supplied claim",
                "constructs the requested logical relation",
                "uses appropriately restrained academic certainty",
                "preserves only the allowed citation slots",
                "reads as natural expert academic prose",
            ],
        })
        keys.append({
            "packet_id": packet_id,
            "example_id": example_id,
            "candidate_a": arm_a,
            "candidate_b": arm_b,
            "held_out_target": row["target"],
        })
    packet_payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in packets
    )
    key_payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in keys
    )
    atomic_write_bytes(output_dir / "blind_ab_packets.jsonl", packet_payload)
    atomic_write_bytes(output_dir / "blind_ab_answer_key.jsonl", key_payload)


def write_prediction_template(corpus_path: Path, destination: Path) -> None:
    held_out = [row for row in read_corpus_records(corpus_path) if row["split"] == "test"]
    payload = b"".join(
        (json.dumps({
            "schema_version": PREDICTION_SCHEMA,
            "example_id": row["example_id"],
            "prompt": format_academic_prompt(row),
            "base": {"text": "", "target_nll": None},
            "adapter": {"text": "", "target_nll": None},
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in sorted(held_out, key=lambda value: value["example_id"])
    )
    atomic_write_bytes(destination, payload)


def mlx_nll_command(
    *, python_executable: str, model_view: Path, data_dir: Path,
    adapter_dir: Path | None, max_sequence_length: int,
) -> list[str]:
    command = [
        python_executable, "-m", "mlx_lm", "lora",
        "--model", str(model_view),
        "--data", str(data_dir),
        "--test", "--test-batches", "-1",
        "--batch-size", "1",
        "--max-seq-length", str(max_sequence_length),
        "--mask-prompt",
    ]
    # MLX-LM 0.31.3 defaults this option to the literal ``adapters`` directory.
    # Its test-only base-model path is an explicitly empty value, not omission.
    command.extend(["--adapter-path", str(adapter_dir) if adapter_dir is not None else ""])
    return command


def _prediction_worker(
    *, model_view: Path, requests_path: Path, output_path: Path,
    arm: str, adapter_dir: Path | None, max_tokens: int, seed: int,
) -> None:
    """Heavy child path. Imported only inside the selected pinned MLX Python."""
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
    except ImportError as exc:  # pragma: no cover - exercised by explicit MLX smoke
        raise HarnessError(f"selected prediction Python cannot import MLX-LM: {exc}") from exc
    model, tokenizer = load(
        str(model_view),
        adapter_path=str(adapter_dir) if adapter_dir is not None else None,
        tokenizer_config={"trust_remote_code": True},
    )
    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line]
    outputs: list[dict[str, Any]] = []
    for request in requests:
        example_id = str(request["example_id"])
        example_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{example_id}".encode()).digest()[:4], "big")
        mx.random.seed(example_seed)
        messages = [{"role": "user", "content": str(request["prompt"])}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        text = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        if not isinstance(text, str):
            raise HarnessError(f"MLX returned no text for held-out example {example_id}")
        outputs.append({
            "schema_version": "spiral.academic-prediction-arm.v1",
            "example_id": example_id,
            "arm": arm,
            "text": text.strip(),
            "seed": example_seed,
            "decode": "greedy_argmax",
        })
    atomic_write_bytes(
        output_path,
        b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for row in outputs
        ),
    )


def _load_prediction_arm(path: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HarnessError(f"prediction worker did not publish {arm} output: {exc}") from exc
    for line in lines:
        row = json.loads(line)
        example_id = str(row.get("example_id") or "")
        if row.get("arm") != arm or not example_id or example_id in rows or not isinstance(row.get("text"), str):
            raise HarnessError(f"invalid or duplicate {arm} prediction worker output")
        rows[example_id] = row
    return rows


def prediction_worker_command(
    *, python_executable: str, model_view: Path, requests_path: Path,
    output_path: Path, arm: str, adapter_dir: Path | None,
    max_tokens: int, seed: int,
) -> list[str]:
    command = [
        python_executable, str(Path(__file__).resolve()), "_predict-worker",
        "--model-view", str(model_view), "--requests", str(requests_path),
        "--output", str(output_path), "--arm", arm,
        "--max-tokens", str(max_tokens), "--seed", str(seed),
    ]
    if adapter_dir is not None:
        command.extend(["--adapter", str(adapter_dir)])
    return command


def run_mlx_predictions(
    *, corpus_path: Path, python_executable: str, model_view: Path,
    adapter_dir: Path, output_path: Path, receipt_path: Path,
    lease_path: Path, ollama_url: str, max_tokens: int = 256,
    max_examples: int = 256, seed: int = 24082026,
) -> dict[str, Any]:
    if not 1 <= max_tokens <= 512:
        raise HarnessError("prediction max_tokens must be between 1 and 512")
    if not 1 <= max_examples <= 1024:
        raise HarnessError("prediction max_examples must be between 1 and 1024")
    held_out = sorted(
        (row for row in read_corpus_records(corpus_path) if row["split"] == "test"),
        key=lambda row: row["example_id"],
    )
    if not held_out:
        raise HarnessError("corpus has no held-out test examples")
    if len(held_out) > max_examples:
        raise HarnessError(
            f"held-out split has {len(held_out)} examples, above explicit bound {max_examples}; "
            "raise --max-examples deliberately rather than silently sampling")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requests_path = output_path.parent / f".{output_path.name}.requests.jsonl"
    request_payload = b"".join(
        (json.dumps({
            "example_id": row["example_id"],
            "prompt": format_academic_prompt(row),
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in held_out
    )
    atomic_write_bytes(requests_path, request_payload)
    arm_paths = {
        arm: output_path.parent / f".{output_path.name}.{arm}.jsonl"
        for arm in ("base", "adapter")
    }
    commands = {
        arm: prediction_worker_command(
            python_executable=python_executable,
            model_view=model_view,
            requests_path=requests_path,
            output_path=arm_paths[arm],
            arm=arm,
            adapter_dir=adapter_dir if arm == "adapter" else None,
            max_tokens=max_tokens,
            seed=seed,
        )
        for arm in ("base", "adapter")
    }
    lease = TrainingComputeLease(lease_path)
    lease.acquire(owner={"model": str(model_view), "operation": "academic_blind_predictions"})
    with lease:
        verify_ollama_empty(ollama_url)
        for arm in ("base", "adapter"):
            completed = subprocess.run(
                commands[arm], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                pass_fds=(lease.descriptor,) if lease.descriptor is not None else (),
            )
            if completed.returncode:
                log = output_path.parent / f"prediction-{arm}.log"
                atomic_write_bytes(log, completed.stdout.encode("utf-8"))
                raise HarnessError(
                    f"MLX {arm} prediction worker failed with status {completed.returncode}; see {log}")
    arms = {arm: _load_prediction_arm(path, arm) for arm, path in arm_paths.items()}
    expected_ids = {str(row["example_id"]) for row in held_out}
    if any(set(rows) != expected_ids for rows in arms.values()):
        raise HarnessError("prediction workers did not return exact held-out example coverage")
    predictions = [
        {
            "schema_version": PREDICTION_SCHEMA,
            "example_id": row["example_id"],
            "base": {"text": arms["base"][str(row["example_id"])]["text"], "target_nll": None},
            "adapter": {"text": arms["adapter"][str(row["example_id"])]["text"], "target_nll": None},
        }
        for row in held_out
    ]
    atomic_write_bytes(
        output_path,
        b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            for row in predictions
        ),
    )
    bundle_digest, required_files = adapter_bundle_digest(adapter_dir)
    receipt = {
        "schema_version": "spiral.academic-prediction-receipt.v1",
        "corpus_sha256": sha256_file(corpus_path),
        "held_out_count": len(held_out),
        "held_out_ids_sha256": hashlib.sha256(
            "\n".join(sorted(expected_ids)).encode("utf-8")).hexdigest(),
        "predictions_sha256": sha256_file(output_path),
        "base_model_view": str(model_view.resolve()),
        "adapter": {
            "path": str(adapter_dir.resolve()),
            "sha256": bundle_digest,
            "required_files": required_files,
        },
        "decode": {
            "strategy": "greedy_argmax",
            "temperature": 0.0,
            "seed": seed,
            "per_example_seed": "sha256(seed\\0example_id)[0:4]",
            "max_tokens": max_tokens,
            "chat_template": True,
            "enable_thinking": False,
            "identical_between_arms": True,
        },
    }
    atomic_write_json(receipt_path, receipt)
    for path in [requests_path, *arm_paths.values()]:
        try:
            path.unlink()
        except OSError:
            pass
    return receipt


def _parse_test_nll(output: str) -> float:
    matches = re.findall(r"Test loss\s+([0-9]+(?:\.[0-9]+)?)", output)
    if len(matches) != 1:
        raise HarnessError("could not identify exactly one MLX held-out Test loss")
    return float(matches[0])


def run_mlx_nll(
    *, python_executable: str, model_view: Path, adapter_dir: Path, data_dir: Path,
    output_path: Path, lease_path: Path, ollama_url: str, max_sequence_length: int = 1024,
) -> dict[str, Any]:
    commands = {
        "base": mlx_nll_command(
            python_executable=python_executable, model_view=model_view, data_dir=data_dir,
            adapter_dir=None, max_sequence_length=max_sequence_length),
        "adapter": mlx_nll_command(
            python_executable=python_executable, model_view=model_view, data_dir=data_dir,
            adapter_dir=adapter_dir, max_sequence_length=max_sequence_length),
    }
    lease = TrainingComputeLease(lease_path)
    lease.acquire(owner={"model": str(model_view), "operation": "academic_held_out_nll"})
    scores: dict[str, float] = {}
    with lease:
        verify_ollama_empty(ollama_url)
        for arm, command in commands.items():
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                pass_fds=(lease.descriptor,) if lease.descriptor is not None else (),
            )
            log_path = output_path.with_suffix(f".{arm}.log")
            atomic_write_bytes(log_path, completed.stdout.encode("utf-8"))
            if completed.returncode:
                raise HarnessError(
                    f"MLX {arm} NLL evaluation failed with status {completed.returncode}; "
                    f"see {log_path}")
            scores[arm] = _parse_test_nll(completed.stdout)
    report = {
        "schema_version": NLL_SCHEMA,
        "completion_only": True,
        "test_split_sha256": sha256_file(data_dir / "test.jsonl"),
        "model_view": str(model_view.resolve()),
        "adapter_path": str(adapter_dir.resolve()),
        "base": {"mean_target_nll": scores["base"]},
        "adapter": {"mean_target_nll": scores["adapter"]},
    }
    atomic_write_json(output_path, report)
    return report


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} must be a JSON object")
    return value


def _artifact_attestation(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve())
        size = path.stat().st_size
    except (OSError, ValueError) as exc:
        raise HarnessError(f"evaluation artifact is missing or outside its run: {path}") from exc
    return {"path": relative.as_posix(), "size_bytes": size, "sha256": sha256_file(path)}


def _assert_finite_metrics(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_metrics(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_metrics(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise HarnessError(f"{label} is not finite")


def finalize_post_training_evaluation(
    *, training_run: Path, evaluation_dir: Path, corpus_path: Path,
    model_view: Path, adapter_dir: Path, data_dir: Path,
) -> dict[str, Any]:
    """Bind a completed comparison to its training run and clear the validation gate.

    The receipt is written before ``training-status.json`` changes.  Re-running this
    function after an interruption is model-free and idempotent: every artifact is
    rehashed, an existing receipt must match exactly, and an already-finalized status
    is left byte-for-byte unchanged.
    """
    run = training_run.resolve()
    evaluation = evaluation_dir.resolve()
    if evaluation != run / "evaluation":
        raise HarnessError("post-training evaluation output must be <training-run>/evaluation")
    status_path = run / "training-status.json"
    manifest_path = run / "academic-adapter.manifest.json"
    status = _read_json_object(status_path, "training status")
    manifest = _read_json_object(manifest_path, "adapter manifest")
    if status.get("schema_version") != "spiral.academic-training-status.v1":
        raise HarnessError("training status has the wrong schema")
    if status.get("state") != "completed" or status.get("completed_steps") != status.get("total_steps"):
        raise HarnessError("post-training evaluation can finalize only a completed training run")
    run_identity = str(status.get("run_identity") or "")
    if len(run_identity) != 64:
        raise HarnessError("training status is missing its run identity")
    if manifest.get("schema_version") != "spiral.academic-adapter.v1":
        raise HarnessError("adapter manifest has the wrong schema")

    expected_adapter = (run / str(manifest.get("adapter", {}).get("path") or "")).resolve()
    if adapter_dir.resolve() != expected_adapter:
        raise HarnessError("evaluation adapter is not the adapter published by this run")
    bundle_digest, required_files = adapter_bundle_digest(expected_adapter)
    manifest_adapter = manifest.get("adapter", {})
    if bundle_digest != manifest_adapter.get("sha256") or required_files != manifest_adapter.get("required_files"):
        raise HarnessError("published adapter no longer matches its manifest")
    checkpoint = status.get("latest_checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("bundle_sha256") != bundle_digest:
        raise HarnessError("completed checkpoint does not match the published adapter")

    manifest_dataset = manifest.get("dataset", {})
    expected_dataset_manifest = (
        run / str(manifest_dataset.get("dataset_manifest_path") or "")
    ).resolve()
    supplied_dataset_manifest = (data_dir.resolve() / "dataset_manifest.json").resolve()
    if supplied_dataset_manifest != expected_dataset_manifest:
        raise HarnessError("evaluation data directory is not the manifest-bound dataset")
    if sha256_file(supplied_dataset_manifest) != manifest_dataset.get("manifest_sha256"):
        raise HarnessError("evaluation dataset manifest no longer matches the adapter manifest")
    if sha256_file(corpus_path) != manifest_dataset.get("source_corpus_file_sha256"):
        raise HarnessError("evaluation corpus no longer matches the adapter manifest")
    test_sha256 = sha256_file(data_dir / "test.jsonl")
    if test_sha256 != manifest_dataset.get("split_sha256", {}).get("test"):
        raise HarnessError("held-out test split no longer matches the adapter manifest")

    artifacts_by_name = {
        name: evaluation / name
        for name in (
            "predictions.jsonl",
            "prediction_receipt.json",
            "target_nll.json",
            "target_nll.base.log",
            "target_nll.adapter.log",
            "evaluation_summary.json",
            "evaluation_examples.jsonl",
            "blind_ab_packets.jsonl",
            "blind_ab_answer_key.jsonl",
        )
    }
    prediction_receipt = _read_json_object(
        artifacts_by_name["prediction_receipt.json"], "prediction receipt")
    nll_report = _load_nll_report(artifacts_by_name["target_nll.json"])
    assert nll_report is not None
    summary = _read_json_object(artifacts_by_name["evaluation_summary.json"], "evaluation summary")
    predictions_sha256 = sha256_file(artifacts_by_name["predictions.jsonl"])
    corpus_sha256 = sha256_file(corpus_path)
    if prediction_receipt.get("schema_version") != "spiral.academic-prediction-receipt.v1":
        raise HarnessError("prediction receipt has the wrong schema")
    if prediction_receipt.get("corpus_sha256") != corpus_sha256:
        raise HarnessError("prediction receipt is bound to a different corpus")
    if prediction_receipt.get("predictions_sha256") != predictions_sha256:
        raise HarnessError("predictions no longer match their receipt")
    prediction_adapter = prediction_receipt.get("adapter", {})
    if (
        prediction_adapter.get("sha256") != bundle_digest
        or prediction_adapter.get("required_files") != required_files
        or Path(str(prediction_adapter.get("path") or "")).resolve() != expected_adapter
    ):
        raise HarnessError("prediction receipt is bound to a different adapter")
    resolved_model_view = model_view.resolve()
    if Path(str(prediction_receipt.get("base_model_view") or "")).resolve() != resolved_model_view:
        raise HarnessError("prediction receipt is bound to a different model view")
    if (
        nll_report.get("test_split_sha256") != test_sha256
        or Path(str(nll_report.get("model_view") or "")).resolve() != resolved_model_view
        or Path(str(nll_report.get("adapter_path") or "")).resolve() != expected_adapter
    ):
        raise HarnessError("held-out NLL report is not bound to this run")
    if (
        summary.get("schema_version") != EVALUATION_SCHEMA
        or summary.get("corpus_sha256") != corpus_sha256
        or summary.get("predictions_sha256") != predictions_sha256
        or summary.get("held_out_examples") != prediction_receipt.get("held_out_count")
        or summary.get("nll_source") != "aggregate_mlx_completion_only"
    ):
        raise HarnessError("evaluation summary is not bound to the prediction/NLL receipts")
    for arm in ("base", "adapter"):
        if summary.get("arms", {}).get(arm, {}).get("mean_target_nll") != nll_report[arm]["mean_target_nll"]:
            raise HarnessError("evaluation summary and NLL report disagree")
    _assert_finite_metrics(summary.get("arms"), "evaluation_summary.arms")
    _assert_finite_metrics(summary.get("delta_adapter_minus_base"), "evaluation_summary.delta")

    artifact_attestations = [
        _artifact_attestation(artifacts_by_name[name], run)
        for name in sorted(artifacts_by_name)
    ]
    receipt_path = run / "post-training-evaluation.json"
    existing_receipt = (
        _read_json_object(receipt_path, "post-training evaluation receipt")
        if receipt_path.exists() else None
    )
    if existing_receipt is not None and (
        existing_receipt.get("schema_version") != POST_TRAINING_EVALUATION_SCHEMA
        or existing_receipt.get("run_identity") != run_identity
    ):
        raise HarnessError("existing post-training evaluation receipt belongs to another run")
    completed_at = (
        str(existing_receipt["completed_at"])
        if existing_receipt is not None
        else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    receipt = {
        "schema_version": POST_TRAINING_EVALUATION_SCHEMA,
        "completed_at": completed_at,
        "run_identity": run_identity,
        "training_checkpoint": dict(checkpoint),
        "adapter_manifest_sha256": sha256_file(manifest_path),
        "adapter": {
            "path": str(expected_adapter),
            "sha256": bundle_digest,
            "required_files": required_files,
        },
        "base_model_view": str(resolved_model_view),
        "corpus_sha256": corpus_sha256,
        "dataset_manifest_sha256": sha256_file(supplied_dataset_manifest),
        "test_split_sha256": test_sha256,
        "held_out_examples": summary["held_out_examples"],
        "decode": prediction_receipt.get("decode"),
        "result": {
            "arms": summary["arms"],
            "delta_adapter_minus_base": summary["delta_adapter_minus_base"],
            "selection_rule": summary.get("selection_rule"),
        },
        "artifacts": artifact_attestations,
    }
    if existing_receipt is not None:
        if existing_receipt != receipt:
            raise HarnessError("evaluation artifacts disagree with the existing durable receipt")
    else:
        atomic_write_json(receipt_path, receipt)

    receipt_sha256 = sha256_file(receipt_path)
    validation_status = {
        "state": "completed",
        "completed_at": completed_at,
        "receipt": receipt_path.name,
        "receipt_sha256": receipt_sha256,
        "evaluation_dir": "evaluation",
        "evaluation_summary_sha256": sha256_file(artifacts_by_name["evaluation_summary.json"]),
    }
    if not (
        status.get("post_training_validation_required") is False
        and status.get("post_training_validation") == validation_status
    ):
        status["post_training_validation_required"] = False
        status["post_training_validation"] = validation_status
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(status_path, status)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate Spiral's academic adapter")
    subparsers = result.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template", help="create held-out prediction template")
    template.add_argument("--corpus", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score", help="score completed base/adapter predictions")
    score.add_argument("--corpus", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--nll-report", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--seed", type=int, default=24082026)
    predict = subparsers.add_parser(
        "predict", help="generate exact held-out base and adapter arms under the shared lease")
    _add_prediction_arguments(predict)
    compare = subparsers.add_parser(
        "compare", help="generate, compute held-out NLL, score, and build blind A/B packets")
    _add_prediction_arguments(compare)
    finalize = subparsers.add_parser(
        "finalize", help="model-free verification and durable completion of a comparison")
    finalize.add_argument("--training-run", type=Path, required=True)
    finalize.add_argument("--corpus", type=Path, required=True)
    finalize.add_argument("--model-view", type=Path, required=True)
    finalize.add_argument("--adapter", type=Path, required=True)
    finalize.add_argument("--data-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    nll = subparsers.add_parser("nll", help="explicitly load MLX twice for held-out target NLL")
    nll.add_argument("--model-view", type=Path, required=True)
    nll.add_argument("--adapter", type=Path, required=True)
    nll.add_argument("--data-dir", type=Path, required=True)
    nll.add_argument("--output", type=Path, required=True)
    nll.add_argument("--python", default=sys.executable)
    nll.add_argument("--max-seq-length", type=int, default=1024)
    nll.add_argument("--lease-path", type=Path,
                     default=Path("~/.spiralchat/spiral-compute.lease").expanduser())
    nll.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    nll.add_argument("--execute", action="store_true")
    worker = subparsers.add_parser("_predict-worker", help=argparse.SUPPRESS)
    worker.add_argument("--model-view", type=Path, required=True)
    worker.add_argument("--adapter", type=Path)
    worker.add_argument("--requests", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--arm", choices=("base", "adapter"), required=True)
    worker.add_argument("--max-tokens", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    return result


def _add_prediction_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--model-view", type=Path, required=True)
    command.add_argument("--adapter", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--max-tokens", type=int, default=256)
    command.add_argument("--max-examples", type=int, default=256)
    command.add_argument("--max-seq-length", type=int, default=1024)
    command.add_argument("--seed", type=int, default=24082026)
    command.add_argument("--lease-path", type=Path,
                         default=Path("~/.spiralchat/spiral-compute.lease").expanduser())
    command.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    command.add_argument(
        "--training-run", type=Path,
        help="completed run to finalize after a successful compare (output must be RUN/evaluation)")
    command.add_argument("--execute", action="store_true")


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "template":
            write_prediction_template(arguments.corpus, arguments.output)
        elif arguments.command == "score":
            summary = evaluate_predictions(
                arguments.corpus, arguments.predictions, arguments.output_dir,
                seed=arguments.seed, nll_report_path=arguments.nll_report)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif arguments.command in {"predict", "compare"}:
            predictions_path = arguments.output_dir / "predictions.jsonl"
            receipt_path = arguments.output_dir / "prediction_receipt.json"
            if not arguments.execute:
                plan = {
                    "execute": False,
                    "model_view": str(arguments.model_view),
                    "adapter": str(arguments.adapter),
                    "corpus": str(arguments.corpus),
                    "decode": {"strategy": "greedy_argmax", "seed": arguments.seed,
                               "max_tokens": arguments.max_tokens},
                    "shared_lease": str(arguments.lease_path),
                    "compare_includes_nll_and_blind_packets": arguments.command == "compare",
                }
                print(json.dumps(plan, indent=2, sort_keys=True))
                print("dry run only; pass --execute to load the model")
            else:
                arguments.output_dir.mkdir(parents=True, exist_ok=True)
                receipt = run_mlx_predictions(
                    corpus_path=arguments.corpus,
                    python_executable=arguments.python,
                    model_view=arguments.model_view,
                    adapter_dir=arguments.adapter,
                    output_path=predictions_path,
                    receipt_path=receipt_path,
                    lease_path=arguments.lease_path,
                    ollama_url=arguments.ollama_url,
                    max_tokens=arguments.max_tokens,
                    max_examples=arguments.max_examples,
                    seed=arguments.seed,
                )
                if arguments.command == "compare":
                    nll_path = arguments.output_dir / "target_nll.json"
                    run_mlx_nll(
                        python_executable=arguments.python,
                        model_view=arguments.model_view,
                        adapter_dir=arguments.adapter,
                        data_dir=arguments.data_dir,
                        output_path=nll_path,
                        lease_path=arguments.lease_path,
                        ollama_url=arguments.ollama_url,
                        max_sequence_length=arguments.max_seq_length,
                    )
                    summary = evaluate_predictions(
                        arguments.corpus, predictions_path, arguments.output_dir,
                        seed=arguments.seed, nll_report_path=nll_path)
                    if arguments.training_run is not None:
                        finalize_post_training_evaluation(
                            training_run=arguments.training_run,
                            evaluation_dir=arguments.output_dir,
                            corpus_path=arguments.corpus,
                            model_view=arguments.model_view,
                            adapter_dir=arguments.adapter,
                            data_dir=arguments.data_dir,
                        )
                    print(json.dumps(summary, indent=2, sort_keys=True))
                else:
                    print(json.dumps(receipt, indent=2, sort_keys=True))
        elif arguments.command == "nll":
            commands = {
                arm: mlx_nll_command(
                    python_executable=arguments.python,
                    model_view=arguments.model_view,
                    data_dir=arguments.data_dir,
                    adapter_dir=arguments.adapter if arm == "adapter" else None,
                    max_sequence_length=arguments.max_seq_length,
                )
                for arm in ("base", "adapter")
            }
            if not arguments.execute:
                print(json.dumps(commands, indent=2))
                print("dry run only; pass --execute to load the model")
            else:
                report = run_mlx_nll(
                    python_executable=arguments.python,
                    model_view=arguments.model_view,
                    adapter_dir=arguments.adapter,
                    data_dir=arguments.data_dir,
                    output_path=arguments.output,
                    lease_path=arguments.lease_path,
                    ollama_url=arguments.ollama_url,
                    max_sequence_length=arguments.max_seq_length,
                )
                print(json.dumps(report, indent=2, sort_keys=True))
        elif arguments.command == "finalize":
            receipt = finalize_post_training_evaluation(
                training_run=arguments.training_run,
                evaluation_dir=arguments.output_dir,
                corpus_path=arguments.corpus,
                model_view=arguments.model_view,
                adapter_dir=arguments.adapter,
                data_dir=arguments.data_dir,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
        elif arguments.command == "_predict-worker":
            _prediction_worker(
                model_view=arguments.model_view,
                requests_path=arguments.requests,
                output_path=arguments.output,
                arm=arguments.arm,
                adapter_dir=arguments.adapter,
                max_tokens=arguments.max_tokens,
                seed=arguments.seed,
            )
        return 0
    except HarnessError as exc:
        print(f"academic evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
