from types import SimpleNamespace

import pytest

from spiral.review_context import select_review_batch, preserved_review_source


def pick(context, *, meter=True):
    rows = [{"id": f"R{i}"} for i in range(18)]
    return select_review_batch(rows, lambda batch: ("s" * (len(batch) * 1000), ["source.py"]),
        lambda batch, source: ["system", source],
        meter=SimpleNamespace(model="selected:exact", count=lambda texts: [len(t) for t in texts]) if meter else None,
        model="selected:exact", context_tokens=context, output_reserved=6144)


def test_larger_measured_capacity_groups_more_requirements_with_same_evidence():
    large, small = pick(32768), pick(14000)
    assert len(large.requirements) == 18 and len(small.requirements) == 4
    for result, capacity in [(large, 32768), (small, 14000)]:
        assert result.raw_tokens + result.framing_reserved + 6144 <= capacity
        assert result.selected_files == ["source.py"]


def test_missing_or_insufficient_measurement_does_not_invent_a_fitting_claim():
    assert len(pick(700000, meter=False).requirements) == 4
    assert pick(700000, meter=False).raw_tokens is None
    assert pick(8192).raw_tokens is None


@pytest.mark.parametrize("model,count", [("other:model", [1]), ("selected:exact", [True]),
                                         ("selected:exact", [-1]), ("selected:exact", [0])])
def test_bad_measurements_or_other_model_are_refused(model, count):
    with pytest.raises(ValueError):
        select_review_batch([{"id": "R1"}], lambda rows: ("source", []), lambda *a: ["input"],
            meter=SimpleNamespace(model=model, count=lambda _: count), model="selected:exact",
            context_tokens=32768, output_reserved=6144)


@pytest.mark.parametrize("counts", [[1], [1, 2, 3], None, "12"])
def test_incomplete_or_extra_measurements_cannot_certify_a_batch(counts):
    with pytest.raises(ValueError):
        select_review_batch([{"id": "R1"}], lambda rows: ("source", []),
            lambda *a: ["system", "user"],
            meter=SimpleNamespace(model="selected:exact", count=lambda _: counts),
            model="selected:exact", context_tokens=32768, output_reserved=6144)


@pytest.mark.parametrize("failure", ["omitted", "file truncation", "total truncation"])
def test_expansion_keeps_focused_evidence_even_when_tokens_fit(tmp_path, failure):
    for name in ("a", "b"):
        (tmp_path / name).write_text("source content")
    rows = list(range(8))
    def source_for(batch):
        selected = ["a"] if len(batch) <= 4 and batch[0] == 0 else ["b"]
        if len(batch) > 4:
            selected = ["a"] if failure == "omitted" else ["a", "b"]
        return "source text", selected
    view = select_review_batch(rows,
        lambda batch: preserved_review_source(tmp_path, batch, source_for,
            fallback_size=4, max_file_chars=5 if failure == "file truncation" else 100,
            max_total=20 if failure == "total truncation" else 100),
        lambda *a: ["system", "user"],
        meter=SimpleNamespace(model="same", count=lambda _: [1, 1]),
        model="same", context_tokens=32768, output_reserved=6144)
    assert len(view.requirements) == 4


def test_expansion_preserves_union_of_focused_evidence(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).write_text("source")
    def source_for(rows):
        return "all code", ["a", "b"] if len(rows) > 4 else ["a" if rows[0] == 0 else "b"]
    result = preserved_review_source(tmp_path, list(range(8)), source_for,
        fallback_size=4, max_file_chars=10, max_total=20)
    assert result == ("all code", ["a", "b"])
