"""Group reviews by measured input capacity without changing model or settings."""
from dataclasses import dataclass


class ReviewEvidenceTooLarge(ValueError):
    """A combined review would lose evidence from its smaller groups."""


def preserved_review_source(root, rows, source_for, *, fallback_size,
                            max_file_chars, max_total):
    """Enlarge only when every smaller group's selected file is present whole.

    Combining queries changes relevance ranking. Token fit alone cannot justify
    displacing the evidence that a smaller, more focused review would receive.
    """
    source, selected = source_for(rows)
    if len(rows) <= fallback_size:
        return source, selected
    required = set()
    for offset in range(0, len(rows), fallback_size):
        _, files = source_for(rows[offset:offset + fallback_size])
        required.update(files)
    try:
        sizes = [len((root / path).read_text(errors="replace")) for path in selected]
    except OSError as exc:
        raise ReviewEvidenceTooLarge("review evidence could not be inspected") from exc
    if (not required.issubset(selected) or sum(sizes) > max_total
            or any(size > max_file_chars for size in sizes)):
        raise ReviewEvidenceTooLarge("combined review would omit or truncate focused evidence")
    return source, selected


@dataclass(frozen=True)
class ReviewBatch:
    requirements: list
    source: str
    selected_files: list
    raw_tokens: int | None = None
    framing_reserved: int = 2048


def select_review_batch(requirements, source_for, prompt_for, *, meter=None,
                        model, context_tokens, output_reserved, fallback_size=4):
    def candidate(size):
        rows = requirements[:size]
        source, selected = source_for(rows)
        return ReviewBatch(rows, source, selected)

    fallback = min(fallback_size, len(requirements))
    if not requirements:
        raise ValueError("review requires at least one requirement")
    if meter is None:
        return candidate(fallback)
    if meter.model != model:
        raise ValueError("review counter must preserve the selected model")
    available = context_tokens - output_reserved - 2048
    # This reserves output space per verdict; complete JSON coverage is still
    # checked after inference. Counts never certify model quality or admission.
    size = min(len(requirements), 64, max(fallback, output_reserved // 256))
    while True:
        try:
            view = candidate(size)
        except ReviewEvidenceTooLarge:
            if size <= fallback:
                raise
            size = max(fallback, size // 2)
            continue
        texts = prompt_for(view.requirements, view.source)
        counts = meter.count(texts)
        if (not isinstance(counts, (list, tuple)) or len(counts) != len(texts)
                or not counts or any(type(n) is not int or n <= 0 for n in counts)):
            raise ValueError("invalid review token measurements")
        if sum(counts) <= available:
            return ReviewBatch(view.requirements, view.source, view.selected_files, sum(counts))
        if size <= fallback:
            # Preserve the conservative legacy group; the provider still owns
            # exact rendered-chat admission and may refuse it. No fitting claim.
            return view
        size = max(fallback, size // 2)
