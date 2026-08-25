from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.academic_finetune.evaluate_structure import (
    EXAMPLES_FILENAME,
    STRUCTURE_EVALUATION_SCHEMA,
    STRUCTURE_PREDICTION_SCHEMA,
    SUMMARY_FILENAME,
    StructureEvaluationError,
    evaluate_structure_predictions,
    main,
    parse_exact_json_object,
    score_structure_candidate,
)


def _row(example_id: str, task: str, target: object, required: list[str]) -> dict:
    return {
        "schema_version": "spiral.academic-paper-structure.v1",
        "example_id": example_id,
        "split": "test",
        "task_type": task,
        "source": {"stratum": "arxiv:hep-th"},
        "document": {"document_id": f"document:{example_id}"},
        "input": {
            "response_schema": {
                "type": "object",
                "properties": {key: {} for key in required},
                "required": required,
            }
        },
        "target": target,
    }


def _fixture_rows() -> list[dict]:
    return [
        _row("01-role", "recognize_role", {"role": "introduction"}, ["role"]),
        _row(
            "02-order",
            "order_structure",
            {
                "parent_id": "s2",
                "parent_path": [2],
                "ordered_section_ids": ["s2.1", "s2.2", "s2.3"],
            },
            ["parent_id", "parent_path", "ordered_section_ids"],
        ),
        _row(
            "03-budget",
            "budget_structure",
            {
                "section_words": 500,
                "section_budgets": [
                    {"id": "s1", "paragraphs": 2, "words": 200, "figures": 0, "tables": 0},
                    {"id": "s2", "paragraphs": 3, "words": 300, "figures": 1, "tables": 1},
                ],
            },
            ["section_words", "section_budgets"],
        ),
        _row(
            "04-restore",
            "restore_section",
            {
                "missing_section": {
                    "id": "s2.2",
                    "heading": "Effective action",
                    "path": [2, 2],
                    "role": "domain_development",
                    "words": 220,
                    "parent_id": "s2",
                }
            },
            ["missing_section"],
        ),
        _row(
            "05-repair",
            "repair_structure",
            {
                "parent_id": "paper",
                "parent_path": [],
                "sections": [
                    {"id": "s1", "heading": "Introduction", "path": [1], "role": "introduction", "words": 200},
                    {"id": "s2", "heading": "Results", "path": [2], "role": "results", "words": 300},
                ],
                "violation_labels": ["wrong_order", "wrong_role"],
            },
            ["parent_id", "parent_path", "sections", "violation_labels"],
        ),
        _row(
            "06-blueprint",
            "brief_to_blueprint",
            {
                "paper_counts": {
                    "abstract_words": 50,
                    "section_words": 500,
                    "section_paragraphs": 5,
                    "unsectioned_words": 0,
                    "figures": 1,
                    "tables": 1,
                },
                "sections": [
                    {"id": "s1", "heading": "Introduction", "role": "introduction", "words": 200},
                    {"id": "s2", "heading": "Results", "role": "results", "words": 300},
                ],
            },
            ["paper_counts", "sections"],
        ),
        _row("07-replay", "prose_replay", "Human prose target.", []),
    ]


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _predictions(rows: list[dict]) -> list[dict]:
    base_outputs = {
        "01-role": json.dumps({"role": "results"}),
        "02-order": json.dumps({
            "parent_id": "paper",
            "parent_path": [],
            "ordered_section_ids": ["s2.2", "s2.1", "s2.3"],
        }),
        "03-budget": json.dumps({
            "section_words": 500,
            "section_budgets": [
                {"id": "s1", "paragraphs": 2, "words": 250, "figures": 0, "tables": 0},
                {"id": "s2", "paragraphs": 3, "words": 200, "figures": 1, "tables": 1},
            ],
        }),
        "04-restore": '```json\n{"missing_section": {}}\n```',
        "05-repair": json.dumps({
            "parent_id": "paper",
            "parent_path": [],
            "sections": [
                {"id": "s2", "heading": "Results", "path": [1], "role": "introduction", "words": 260},
                {"id": "s1", "heading": "Introduction", "path": [2], "role": "introduction", "words": 240},
            ],
            "violation_labels": ["wrong_order"],
        }),
        "06-blueprint": json.dumps({
            "paper_counts": {
                "abstract_words": 50,
                "section_words": 500,
                "section_paragraphs": 5,
                "unsectioned_words": 0,
                "figures": 1,
                "tables": 1,
            },
            "sections": [
                {"id": "s2", "heading": "Findings", "role": "discussion", "words": 320},
                {"id": "s1", "heading": "Introduction", "role": "introduction", "words": 180},
            ],
        }),
        "07-replay": "An ordinary generated prose replay.",
    }
    return [
        {
            "schema_version": STRUCTURE_PREDICTION_SCHEMA,
            "example_id": row["example_id"],
            "base": base_outputs[row["example_id"]],
            "adapter": (
                "An adapter prose replay."
                if row["task_type"] == "prose_replay"
                else json.dumps(row["target"], sort_keys=True, separators=(",", ":"))
            ),
        }
        for row in rows
    ]


def test_exact_json_parser_rejects_wrappers_duplicates_nonobjects_and_nonfinite() -> None:
    assert parse_exact_json_object(" \n {\"role\":\"results\"} \n")[0] == {"role": "results"}
    for invalid in (
        '```json\n{"role":"results"}\n```',
        '{"role":"results"} trailing',
        '["results"]',
        '{"role":"results","role":"discussion"}',
        '{"score":NaN}',
    ):
        parsed, error = parse_exact_json_object(invalid)
        assert parsed is None and error


def test_structure_evaluation_scores_all_six_tasks_and_counts_replay(tmp_path: Path) -> None:
    rows = _fixture_rows()
    corpus = tmp_path / "corpus.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "evaluation"
    _jsonl(corpus, rows)
    _jsonl(predictions, _predictions(rows))

    summary = evaluate_structure_predictions(corpus, predictions, output)

    assert summary["schema_version"] == STRUCTURE_EVALUATION_SCHEMA
    assert summary["counts"]["examples"] == 7
    assert summary["counts"]["structure_examples"] == 6
    assert summary["counts"]["prose_replay_examples"] == 1
    adapter_tasks = summary["arms"]["adapter"]["by_task"]
    for task in (
        "recognize_role",
        "order_structure",
        "budget_structure",
        "restore_section",
        "repair_structure",
        "brief_to_blueprint",
    ):
        metrics = adapter_tasks[task]["metrics"]
        assert metrics["json_validity"] == 1.0
        assert metrics["required_key_validity"] == 1.0
        assert metrics["schema_validity"] == 1.0
        assert all(math.isfinite(value) for value in metrics.values())
    assert adapter_tasks["budget_structure"]["metrics"]["budget_total_compliance"] == 1.0
    assert adapter_tasks["budget_structure"]["metrics"]["word_mae"] == 0.0
    assert adapter_tasks["repair_structure"]["metrics"]["violation_label_f1"] == 1.0
    blueprint = adapter_tasks["brief_to_blueprint"]["metrics"]
    assert blueprint["blueprint_heading_overlap"] == 1.0
    assert blueprint["blueprint_role_overlap"] == 1.0
    assert blueprint["blueprint_order_overlap"] == 1.0
    replay = adapter_tasks["prose_replay"]
    assert replay == {"example_count": 1, "metrics": {"outputs_counted": 1}}
    assert "json_validity" not in replay["metrics"]

    base_tasks = summary["arms"]["base"]["by_task"]
    assert base_tasks["recognize_role"]["metrics"]["role_accuracy"] == 0.0
    assert base_tasks["restore_section"]["metrics"]["json_validity"] == 0.0
    assert base_tasks["budget_structure"]["metrics"]["budget_total_compliance"] == 0.0
    assert 0.0 < base_tasks["repair_structure"]["metrics"]["violation_label_f1"] < 1.0
    assert base_tasks["brief_to_blueprint"]["metrics"]["blueprint_order_overlap"] == 0.0

    summary_bytes = (output / SUMMARY_FILENAME).read_bytes()
    examples_bytes = (output / EXAMPLES_FILENAME).read_bytes()
    assert hashlib_sha256(examples_bytes) == summary["examples_sha256"]
    assert len(examples_bytes.splitlines()) == 7
    evaluate_structure_predictions(corpus, predictions, output)
    assert (output / SUMMARY_FILENAME).read_bytes() == summary_bytes
    assert (output / EXAMPLES_FILENAME).read_bytes() == examples_bytes


def hashlib_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_nested_schema_and_required_key_validity_are_distinct() -> None:
    row = _fixture_rows()[2]
    missing_observed_counts = json.dumps({
        "section_words": 500,
        "section_budgets": [
            {"id": "s1", "paragraphs": 2, "words": 200},
            {"id": "s2", "paragraphs": 3, "words": 300},
        ],
    })
    score = score_structure_candidate(row, missing_observed_counts)
    assert score["metrics"]["json_validity"] == 1.0
    assert score["metrics"]["required_key_validity"] == 1.0
    assert score["metrics"]["schema_validity"] == 0.0
    assert score["metrics"]["budget_total_compliance"] == 1.0


def test_prediction_coverage_duplicates_and_literal_string_contract_are_strict(
    tmp_path: Path,
) -> None:
    row = _fixture_rows()[0]
    corpus = tmp_path / "corpus.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(corpus, [row])

    _jsonl(predictions, [])
    with pytest.raises(StructureEvaluationError, match="exactly cover"):
        evaluate_structure_predictions(corpus, predictions, tmp_path / "missing")

    packet = _predictions([row])[0]
    _jsonl(predictions, [packet, packet])
    with pytest.raises(StructureEvaluationError, match="duplicate prediction"):
        evaluate_structure_predictions(corpus, predictions, tmp_path / "duplicate")

    wrong_shape = dict(packet)
    wrong_shape["base"] = {"text": wrong_shape["base"]}
    _jsonl(predictions, [wrong_shape])
    with pytest.raises(StructureEvaluationError, match="literal output string"):
        evaluate_structure_predictions(corpus, predictions, tmp_path / "wrong-shape")

    extra = dict(packet)
    extra["example_id"] = "extra"
    _jsonl(predictions, [packet, extra])
    with pytest.raises(StructureEvaluationError, match=r"extra=\['extra'\]"):
        evaluate_structure_predictions(corpus, predictions, tmp_path / "extra")


def test_cli_writes_the_same_model_free_artifacts(tmp_path: Path) -> None:
    rows = _fixture_rows()
    corpus = tmp_path / "corpus.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "cli-output"
    _jsonl(corpus, rows)
    _jsonl(predictions, _predictions(rows))
    assert main([
        "--corpus", str(corpus),
        "--predictions", str(predictions),
        "--output-dir", str(output),
    ]) == 0
    assert (output / SUMMARY_FILENAME).is_file()
    assert (output / EXAMPLES_FILENAME).is_file()
