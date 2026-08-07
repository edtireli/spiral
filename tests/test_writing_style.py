"""Writing style as measurement: corpus-mined templates and AI-tell detection.

Every number here is deterministic — same text in, same numbers out — because the
whole point is that a model may rewrite prose but only the counters decide whether the
rewrite actually moved toward the field's style.
"""
from spiral.writing_style import (
    StyleTemplate, ai_score, ai_tells, measure, mine_template, score_against,
)

AI_PROSE = """Additionally, this framework stands as a testament to the enduring
interplay between theory and practice. It is not only a robust methodology, but also a
vibrant tapestry of ideas, showcasing the meticulous care of its authors and
underscoring the importance of holistic thinking. Experts argue that the approach serves
as a pivotal turning point, reflecting a broader evolution. Despite these challenges,
the future outlook remains promising."""

PLAIN_PROSE = """We measured the decay rate at three temperatures. The 40 K sample
decayed with a time constant of 12.4 ms, about twice the 300 K value. We did not
reproduce the anomaly reported earlier; our sample had lower defect density, which
likely explains the difference. The remaining discrepancy at 120 K is unexplained."""


def test_ai_tells_separate_machine_prose_from_plain_prose():
    assert ai_score(AI_PROSE) > 50
    assert ai_score(PLAIN_PROSE) < 10
    assert ai_score(AI_PROSE) > ai_score(PLAIN_PROSE) * 5


def test_each_tell_is_quotable():
    """A tell with no example is an accusation; every hit must carry its evidence."""
    for tell in ai_tells(AI_PROSE):
        assert tell.examples and all(e.strip() for e in tell.examples)
        assert tell.explanation


def test_named_wikipedia_patterns_are_detected():
    found = {t.id for t in ai_tells(AI_PROSE)}
    for expected in ("significance-inflation", "negative-parallelism",
                     "copula-avoidance", "superficial-analysis",
                     "challenges-formula", "ai-vocabulary"):
        assert expected in found, f"missed {expected}"


def test_detector_is_deterministic():
    assert ai_score(AI_PROSE) == ai_score(AI_PROSE)
    assert [t.id for t in ai_tells(AI_PROSE)] == [t.id for t in ai_tells(AI_PROSE)]


def test_rewrite_that_removes_tells_scores_lower():
    """The property a rewrite tool needs: the number must move when prose improves."""
    before = ai_score(AI_PROSE)
    rewritten = ("This framework connects theory and practice. It is a methodology with "
                 "a wide range of ideas. The approach changed how the field works.")
    assert ai_score(rewritten) < before


def test_measure_reports_prose_shape_not_latex():
    tex = (r"\section{Results} We find $x = 2$ and \begin{equation}E=mc^2\end{equation} "
           r"holds \cite{einstein1905}. This may be relevant.")
    m = measure(tex)
    assert m.words > 5
    assert m.citations_per_1k > 0 and m.equations_per_1k > 0
    assert m.hedges_per_1k > 0            # "may"
    assert "mc^2" not in str(m.as_dict())  # math stripped from prose counts


def test_mine_template_gives_bands_and_section_order():
    corpus = [
        r"\section{Introduction} We study X, which may matter \cite{a}. "
        r"\section{Results} We find Z. " * 25,
        r"\section{Introduction} We consider Y \cite{b}. "
        r"\section{Results} Data show V. " * 25,
        r"\section{Introduction} Prior work \cite{c} suggests W. "
        r"\section{Results} We observe Q. " * 25,
    ]
    tpl = mine_template(corpus)
    assert tpl.sample_size == 3
    assert tpl.section_order[:2] == ["Introduction", "Results"]
    assert len(tpl.section_order) == len(set(tpl.section_order)), "duplicated sections"
    for name, (lo, med, hi) in tpl.targets.items():
        assert lo <= med <= hi, f"{name} band is not ordered"
    assert tpl.vocabulary


def test_empty_corpus_yields_empty_template_not_a_crash():
    tpl = mine_template([])
    assert tpl.sample_size == 0 and tpl.targets == {}
    assert isinstance(tpl, StyleTemplate)
    # scoring against an empty template reports no gaps rather than exploding
    rep = score_against(PLAIN_PROSE, tpl)
    assert rep["gaps"] == [] and rep["in_band"] is True


def test_score_against_reports_actionable_gaps():
    corpus = [r"\section{Introduction} We show a result \cite{a}. It may hold. " * 30] * 3
    tpl = mine_template(corpus)
    rep = score_against(AI_PROSE, tpl)
    assert rep["ai_score"] > 0
    assert rep["missing_sections"], "AI prose has no sections; template expects them"
    assert all(("raise" in g or "lower" in g) for g in rep["gaps"])


def test_template_markdown_renders():
    tpl = mine_template([r"\section{Intro} We test it \cite{x}. " * 30] * 3)
    md = tpl.markdown()
    assert "Writing template" in md and "| metric |" in md
