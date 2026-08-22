"""Writing style as measurement: corpus-mined templates and AI-tell detection.

Every number here is deterministic — same text in, same numbers out — because the
whole point is that a model may rewrite prose but only the counters decide whether the
rewrite actually moved toward the field's style.
"""
from spiral.writing_style import (
    StyleTemplate, ai_score, ai_tells, measure, mine_template, score_against,
    template_distance,
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


def test_formulaic_metadiscourse_and_future_work_are_detected():
    text = (
        "It is important to note that the protocol was fixed. The study provides a "
        "clear milestone and suggests directions for future work. These findings "
        "confirm the robustness of the method and highlight the need for continued "
        "investigation."
    )
    found = {tell.id for tell in ai_tells(text, per_1k=False)}
    assert {"formulaic-metadiscourse", "challenges-formula",
            "formulaic-need-claim", "significance-inflation",
            "formulaic-findings-claim", "ai-vocabulary"} <= found


def test_formulaic_conclusions_left_by_a_live_rewrite_are_detected():
    text = (
        "The study marks a milestone and suggests directions for future work. "
        "These results confirm the experimental paradigm’s reliability and highlight "
        "the need for further investigation. Further work is required."
    )
    found = {tell.id for tell in ai_tells(text, per_1k=False)}
    assert {"significance-inflation", "challenges-formula",
            "formulaic-findings-claim", "formulaic-need-claim"} <= found


def test_formulaic_conclusion_synonyms_do_not_evade_the_catalogue():
    text = (
        "The study provides a baseline for future research and outlines specific avenues for "
        "subsequent analysis. These results indicate the paradigm functions reliably "
        "while requiring further investigation. The experimental paradigm demonstrates "
        "reliability."
    )
    found = {tell.id for tell in ai_tells(text, per_1k=False)}
    assert {"significance-inflation", "challenges-formula",
            "formulaic-findings-claim", "formulaic-need-claim"} <= found

    later = {tell.id for tell in ai_tells(
        "This study establishes a baseline and outlines next steps. "
        "The experimental paradigm yields reliable data.", per_1k=False)}
    assert {"significance-inflation", "formulaic-findings-claim"} <= later


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


def test_template_distance_scores_vocabulary_and_section_arc_not_only_numbers():
    tpl = StyleTemplate(
        sample_size=5,
        vocabulary=["cortical", "stimulus", "adaptation", "amplitude"],
        section_order=["Introduction", "Methods", "Results"],
        targets={},
    )
    matching = (
        "# Introduction\n\nCortical stimulus adaptation was measured.\n\n"
        "# Methods\n\nWe recorded response amplitude.\n\n# Results\n\nThe response changed."
    )
    unrelated = (
        "# Discussion\n\nThe medieval manuscript contains several poems.\n\n"
        "# Appendix\n\nWe list the catalogue entries."
    )
    assert template_distance(matching, tpl) < template_distance(unrelated, tpl)
    assert template_distance(matching, tpl) == 0.0


def test_template_markdown_renders():
    tpl = mine_template([r"\section{Intro} We test it \cite{x}. " * 30] * 3)
    md = tpl.markdown()
    assert "Writing template" in md and "| metric |" in md


# ---------------------------------------------------------- mined + structural tells
# Verbatim examples from Wikipedia: Signs of AI writing. An earlier detector caught only
# 1 of these 10 — it implemented "not only X but Y" and missed the plain "not X, but Y"
# framing plus every formatting tell.
import pytest

from spiral.writing_style import ai_tells as _tells


@pytest.mark.parametrize("label,text,expect", [
    ("not X but Y",       "It is not a mirror but a portal to something else.", "not-x-but-y"),
    ("no X, no Y, just",  "There are no gods, no masters, just people.", "not-x-but-y"),
    ("it's not X it's Y", "It's not a failure, it's a redirection of the effort.", "not-x-but-y"),
    ("X rather than Y",   "prioritizing empirical consolidation rather than ideological purity",
                          "x-rather-than-y"),
    ("not only X but Y",  "not only a work of self-representation, but a visual document",
                          "negative-parallelism"),
    ("title case",        "## Impact of Technology and Digitalization\n\nbody", "title-case-headings"),
    ("inline-header list", "- **Heavy-Duty Rotary Saws**: Designed for tougher materials\n"
                           "- **Compact Trim Saws**: For finish work", "inline-header-list"),
    ("title as noun",     "**Catchment area (health)** refers to the geographic area.",
                          "title-as-proper-noun"),
    ("thematic break",    "text here\n\n---\n\n## Next Section\n", "thematic-break-before-heading"),
    ("skipped heading",   "## Top\n\nbody text\n\n#### Deep\n", "skipped-heading-level"),
])
def test_wikipedia_catalogued_tells_are_detected(label, text, expect):
    assert expect in {t.id for t in _tells(text)}, f"{label}: missed {expect}"


def test_density_tells_have_a_floor():
    """One bold word or one em dash in a long passage is not evidence; a page dense
    with them is. The floor is per 1000 words, so length matters, not raw count."""
    filler = "We measured the sample and recorded the result carefully. " * 30
    assert "boldface-overuse" not in {t.id for t in _tells(filler + "One **term** here.")}
    assert "em-dash-overuse" not in {t.id for t in _tells(filler + "A single — dash.")}
    heavy = "The **a** and **b** with **c** plus **d** and **e** and **f**."
    assert "boldface-overuse" in {t.id for t in _tells(heavy)}
    assert "em-dash-overuse" in {
        t.id for t in _tells("A — b — c — d — e — f — g — h.")}


def test_sentence_case_headings_are_not_flagged():
    assert "title-case-headings" not in {
        t.id for t in _tells("## Impact of technology and digitalization\n\nbody")}


def test_mined_phrase_tells_load_and_fire():
    """The phrase lists come from the page's own 'Words to watch' boxes."""
    from spiral.ai_tells import compile_tells, load

    mined = load()
    assert mined, "mined tell cache is missing"
    assert len(compile_tells(mined)) >= 8
    ids = {t.id for t in _tells(
        "It has been profiled in local media outlets and trade publications.")}
    assert any("notability" in i for i in ids)


def test_words_box_parser_survives_nested_citation_templates():
    """The live 2026 vocabulary box nests cite templates inside ``strong``.

    Stopping at the first closing braces retained only the first few words and silently
    dropped the rest of Wikipedia's list.
    """
    from spiral.ai_tells import mine_wikitext

    source = """
== Language ==
Words to watch: {{strong|''delve'',<ref name="x">{{cite web|title=X}}</ref>
''emphasizing'', {{citation needed|date=August 2026}} ''vibrant'' }}
"""
    mined = mine_wikitext(source)
    assert mined["Language"]["words_to_watch"] == [
        "delve", "emphasizing", "vibrant",
    ]


def test_wikipedia_placeholders_compile_to_real_patterns():
    from spiral.ai_tells import _phrase_to_regex
    import re

    article = re.compile(_phrase_to_regex("represents [a] shift"), re.I)
    date = re.compile(_phrase_to_regex("as of [date]"), re.I)
    assert article.search("represents a shift")
    assert article.search("represents the shift")
    assert date.search("as of 2026")
    assert date.search("as of August 13, 2026")


def test_wikipedia_attached_ellipsis_and_end_boundaries_compile_correctly():
    from spiral.ai_tells import _phrase_to_regex
    import re

    attached = re.compile(_phrase_to_regex("its... challenges... future"), re.I)
    leading = re.compile(_phrase_to_regex("...in local media"), re.I)
    key = re.compile(_phrase_to_regex("key"), re.I)
    assert attached.search("Despite its age, it faces several challenges before future work")
    assert leading.search("It has appeared widely in local media")
    assert key.search("a key result")
    assert not key.search("a keyboard result")


def test_plain_human_prose_stays_clean():
    """The whole thing is worthless if ordinary writing trips it."""
    human = ("We measured the decay rate at three temperatures. The 40 K sample decayed "
             "with a time constant of 12.4 ms, about twice the 300 K value. We did not "
             "reproduce the anomaly reported earlier.")
    assert _tells(human) == []


def test_docx_action_gate_ignores_typography_and_single_technical_word():
    from spiral.style_tool import _docx_actionable_tells

    assert not _docx_actionable_tells("The model used HC3-robust standard errors.")
    assert not _docx_actionable_tells("The programme was called “danmark”.")
    assert _docx_actionable_tells("Additionally, the response changed.")
    assert _docx_actionable_tells("Studies have shown that the response changes.")


def test_percentage_does_not_hide_tells_later_in_the_same_paragraph():
    text = (
        "Response amplitude decreased by 18.4% during the third block. "
        "This finding underscores the importance of checking the complete paragraph."
    )
    ids = {tell.id for tell in _tells(text)}
    assert "ai-vocabulary" in ids
    assert "significance-inflation" in ids


def test_high_density_vocabulary_is_not_a_one_word_blacklist():
    assert "high-density-of-ai-vocabulary-words" not in {
        t.id for t in _tells("The key estimate follows from the measured decay rate.")
    }
    dense = "The key result offers valuable context across the empirical landscape."
    assert "high-density-of-ai-vocabulary-words" in {t.id for t in _tells(dense)}
    assert "ai-vocabulary" in {t.id for t in _tells("We take a deep dive into the result.")}


def test_mined_and_handwritten_tells_do_not_double_count():
    """A mined section and a hand-written pattern often name the same habit
    ('Superficial analyses' vs superficial-analysis). Reporting both inflates the
    score and clutters the output."""
    ids = {t.id for t in _tells(AI_PROSE)}
    for hand, mined in [
        ("superficial-analysis", "superficial-analyses"),
        ("challenges-formula", "outline-like-conclusions-about-challenges-and-fu"),
        ("significance-inflation", "undue-emphasis-on-significance-legacy-and-broade"),
        ("ai-vocabulary", "high-density-of-ai-vocabulary-words"),
        ("copula-avoidance", "avoidance-of-basic-copulatives-is-are-phrases"),
    ]:
        assert not (hand in ids and mined in ids), f"double-counted {hand} + {mined}"


def test_mined_tells_still_add_coverage_where_handwritten_miss():
    """Deduping must suppress duplicates, not silence the mined catalogue."""
    ids = {t.id for t in _tells(
        "It has been profiled in local media outlets and trade publications.")}
    assert any("notability" in i for i in ids)
