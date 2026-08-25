from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.academic_finetune.bound_training_data import (
    BoundTrainingDataError,
    create_bounded_training_data_view,
    measure_completion_sequence,
)
from scripts.academic_finetune.text import (
    certainty,
    citation_count,
    citation_markers,
    neutral_claims,
    rhetorical_relation,
    sentences,
)
from scripts.academic_finetune.training_support import format_academic_prompt


class FakeTokenizer:
    """Word-count tokenizer with a stable chat-template overhead."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        copied = [dict(message) for message in messages]
        self.calls.append((copied, dict(kwargs)))
        length = 4 + sum(len(message["content"].split()) for message in messages)
        if len(messages) == 2:
            length += 1
        return list(range(length))


def _row(example_id: str, task_type: str, completion: str, context: str) -> dict:
    claims = neutral_claims(completion, paragraph=task_type == "paragraph")
    return {
        "completion": completion,
        "example_id": example_id,
        "prompt": format_academic_prompt(
            {
                "task_type": task_type,
                "input": {
                    "context": context,
                    "claims": claims,
                    "rhetorical_relation": rhetorical_relation(completion),
                    "certainty": certainty(completion),
                    "citation_count": citation_count(completion),
                    "citation_slots": citation_markers(completion),
                },
            }
        ),
        "source_stratum": "arxiv:hep-th",
        "task_type": task_type,
    }


def _write_train(directory: Path, lines: list[bytes]) -> None:
    directory.mkdir(parents=True)
    (directory / "train.jsonl").write_bytes(b"".join(lines))
    # These held-out files must never be copied into the bounded trainer view.
    (directory / "valid.jsonl").write_text("held out\n", encoding="utf-8")
    (directory / "test.jsonl").write_text("held out\n", encoding="utf-8")


def test_exact_gate_preserves_bounded_bytes_and_losslessly_greedy_chunks(tmp_path: Path):
    tokenizer = FakeTokenizer()
    bounded = _row(
        "bounded",
        "sentence",
        "The reference calculation remains stable throughout the sampled interval.",
        "An independent estimate fixes the normalization.",
    )
    paragraph_sentences = [
        "The scalar field remains stable because the boundary condition suppresses growing radial modes.",
        "However, weak nonlinear corrections may shift the resonant frequency at late times.",
        "The numerical solutions demonstrate that energy is conserved to approximately 2 percent.",
        "Consequently, the effective description remains reliable throughout the sampled parameter range.",
    ]
    original_context = (
        "Earlier calculations establish the reference solution without constraining "
        "the missing argument."
    )
    oversized = _row(
        "oversized",
        "paragraph",
        " ".join(paragraph_sentences),
        original_context,
    )
    # Noncanonical whitespace/key order demonstrates byte-for-byte preservation.
    bounded_line = (json.dumps(bounded, ensure_ascii=False) + "  \n").encode()
    oversized_line = (
        json.dumps(oversized, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    data_dir = tmp_path / "data"
    _write_train(data_dir, [bounded_line, oversized_line])

    view, receipt = create_bounded_training_data_view(
        data_dir,
        tmp_path / "run",
        tokenizer=tokenizer,
        tokenizer_identity={"identity": "fake-word-tokenizer-v1"},
        max_sequence_length=120,
    )
    output_lines = (view / "train.jsonl").read_bytes().splitlines(keepends=True)
    assert output_lines[0] == bounded_line
    output_rows = [json.loads(line) for line in output_lines]
    derived = output_rows[1:]

    # The descending greedy search chooses the longest fitting chunk each time:
    # two sentences, then one, then one at this deterministic fake-token budget.
    assert [row["task_type"] for row in derived] == [
        "paragraph",
        "sentence",
        "sentence",
    ]
    assert [len(sentences(row["completion"])) for row in derived] == [2, 1, 1]
    assert [
        sentence
        for row in derived
        for sentence in sentences(row["completion"])
    ] == paragraph_sentences
    assert "Previous context:\n" + original_context in derived[0]["prompt"]
    assert (
        "Previous context:\n" + " ".join(paragraph_sentences[:2])
        in derived[1]["prompt"]
    )
    assert (
        "Previous context:\n" + " ".join(paragraph_sentences[1:3])
        in derived[2]["prompt"]
    )
    assert len({row["example_id"] for row in output_rows}) == len(output_rows)
    assert all(len(row["example_id"]) == 64 for row in derived)

    measurements = [measure_completion_sequence(tokenizer, row) for row in output_rows]
    assert all(0 < item.prompt_offset < item.total_tokens <= 120 for item in measurements)
    assert all(item.completion_tokens > 0 for item in measurements)
    assert receipt["output"] == {
        **receipt["output"],
        "unchanged_source_rows": 1,
        "partitioned_source_rows": 1,
        "derived_rows": 3,
    }
    assert receipt["preservation"]["all_source_completions_preserved"] is True
    mapping = receipt["mappings"][1]
    assert mapping["source_completion_normalized_sha256"] == (
        mapping["reconstructed_completion_normalized_sha256"]
    )
    assert not (view / "valid.jsonl").exists()
    assert not (view / "test.jsonl").exists()
    assert hashlib.sha256((view / "train.jsonl").read_bytes()).hexdigest() == (
        receipt["output"]["train_sha256"]
    )

    # Every gate call is exactly the same call shape as MLX-LM's
    # CompletionsDataset.process implementation.
    assert tokenizer.calls
    for messages, kwargs in tokenizer.calls:
        assert kwargs["tools"] is None
        assert kwargs["return_dict"] is False
        if len(messages) == 1:
            assert kwargs["add_generation_prompt"] is True
        else:
            assert len(messages) == 2
            assert "add_generation_prompt" not in kwargs


def test_view_is_content_addressed_atomic_and_idempotent(tmp_path: Path):
    tokenizer = FakeTokenizer()
    row = _row(
        "bounded",
        "sentence",
        "The effective description remains accurate within the stated approximation.",
        "The preceding derivation defines the approximation.",
    )
    data_dir = tmp_path / "data"
    _write_train(data_dir, [(json.dumps(row) + "\n").encode()])
    arguments = dict(
        tokenizer=tokenizer,
        tokenizer_identity="fake-word-tokenizer-v1",
        max_sequence_length=512,
    )
    first_path, first_receipt = create_bounded_training_data_view(
        data_dir, tmp_path / "run", **arguments
    )
    second_path, second_receipt = create_bounded_training_data_view(
        data_dir, tmp_path / "run", **arguments
    )
    assert second_path == first_path
    assert second_receipt == first_receipt
    assert sorted(path.name for path in first_path.iterdir()) == [
        "train.jsonl",
        "view.json",
    ]
    assert json.loads((first_path / "view.json").read_text()) == first_receipt
    assert not list(first_path.parent.glob(".*.tmp-*"))


def test_overlong_sentence_fails_instead_of_truncating_completion(tmp_path: Path):
    tokenizer = FakeTokenizer()
    completion = (
        "The deliberately long scientific completion contains every required observation "
        "and therefore must remain wholly intact even when the configured sequence budget "
        "is much too small for the prompt and response together."
    )
    row = _row("long-sentence", "sentence", completion, "A prior result sets the scale.")
    data_dir = tmp_path / "data"
    _write_train(data_dir, [(json.dumps(row) + "\n").encode()])
    with pytest.raises(BoundTrainingDataError, match="cannot be shortened.*truncating"):
        create_bounded_training_data_view(
            data_dir,
            tmp_path / "run",
            tokenizer=tokenizer,
            tokenizer_identity="fake-word-tokenizer-v1",
            max_sequence_length=40,
        )
    assert not (tmp_path / "run" / ".work" / "bounded-trainer-data").exists()


def test_gate_rejects_invalid_completion_offset():
    class InvalidOffsetTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return [1, 2, 3]

    with pytest.raises(BoundTrainingDataError, match="prompt offset"):
        measure_completion_sequence(
            InvalidOffsetTokenizer(),
            {"example_id": "bad", "prompt": "prompt", "completion": "completion"},
        )

