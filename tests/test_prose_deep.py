"""Deep prose: article routing, corpus selection, and rewrite gates."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from spiral.documents import read_document, write_document
from spiral.prose_research import (
    ArticleAssessment, _matched_topic_anchors, _select_full_texts,
    build_deep_profile, detect_article, plan_article,
)
from spiral.research_corpus import Corpus, Paper
from spiral.style_tool import content_drift
from spiral.writing_style import ai_raw_score, ai_score


def test_long_unstructured_article_is_detected_without_headings_or_citations(tmp_path):
    text = "We measured the response and recorded the result. " * 155
    assessment = detect_article(tmp_path / "essay.txt", text)
    assert assessment.words >= 1200
    assert assessment.is_article is True


def test_short_note_does_not_trigger_literature_research(tmp_path):
    assessment = detect_article(
        tmp_path / "note.md",
        "## Result\n\nWe measured one sample and recorded the result.",
    )
    assert assessment.is_article is False


def test_field_and_query_overrides_shape_the_search_plan(tmp_path):
    plan = plan_article(
        tmp_path / "draft.md", "A manuscript about an experiment.",
        field_hint="clinical neuroscience",
        query_overrides=["cortical response trial", "cortical response trial"],
    )
    assert plan["field"] == "clinical neuroscience"
    assert plan["domain"] == "bio-med"
    assert plan["channels"] == ["europepmc", "pubmed", "crossref"]
    assert plan["queries"] == ["cortical response trial"]
    assert plan["planned_by"] == "user overrides"


def test_explicit_field_and_query_do_not_get_rewritten_by_planner(tmp_path):
    class _Planner:
        @property
        def planner(self):
            raise AssertionError("planner must not be consulted")

    class _LLM:
        def chat(self, *args, **kwargs):
            raise AssertionError("LLM must not be consulted")

    plan = plan_article(
        tmp_path / "draft.md", "A manuscript about an experiment.",
        cfg=_Planner(), ol=_LLM(), field_hint="visual neuroscience",
        query_overrides=["visual cortical repetition adaptation EEG"],
    )
    assert plan["field"] == "visual neuroscience"
    assert plan["queries"] == ["visual cortical repetition adaptation EEG"]
    assert plan["planned_by"] == "user overrides"


def test_automatic_plan_reuses_ready_manifest_without_replanning(tmp_path):
    from spiral.prose_research import profile_root

    article = tmp_path / "draft.md"
    text = "A clinical neuroscience manuscript about cortical oxygen metabolism."
    article.write_text(text)
    plan = {
        "title": "Cortical oxygen metabolism", "field": "clinical neuroscience",
        "summary": "", "terminology": ["oxygen"], "domain": "bio-med",
        "channels": ["pubmed"], "categories": [],
        "queries": ["cortical oxygen metabolism"], "planned_by": "model",
    }
    root = profile_root(article, text, plan)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "status": "ready",
        "document_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
        "search_plan": plan,
    }))

    class _NoPlanner:
        @property
        def planner(self):
            raise AssertionError("unchanged document must reuse its plan")

    reused = plan_article(article, text, cfg=_NoPlanner(), ol=object())
    assert reused == plan
    stable = profile_root(article, text) / "search-plan.json"
    assert json.loads(stable.read_text())["plan"] == plan


def test_raw_tell_score_cannot_be_lowered_by_padding():
    original = "This framework serves as a pivotal method."
    padded = original + " We describe the measured result in ordinary terms." * 20
    assert ai_raw_score(padded) == ai_raw_score(original)
    assert ai_score(padded) < ai_score(original)  # why density is report-only
    assert any("padded" in issue for issue in content_drift(original, padded))


def test_content_invariants_are_symmetric_and_preserve_multiplicity():
    original = (
        r"We did not observe 42 units twice: 42, with $x=2$ \citep[3]{known}. "
        r"See \ref{fig:a} and https://example.test/a."
    )
    changed = (
        r"We observed 42 units once, with $x=3$ \citep[3]{known,new}. "
        r"See \ref{fig:b} and https://example.test/b."
    )
    issues = content_drift(original, changed, min_ratio=0.5, max_ratio=2.0)
    joined = "\n".join(issues)
    assert "dropped numbers" in joined and "invented numbers" in joined
    assert "inline equations" in joined
    assert "citations" in joined
    assert "references/labels" in joined
    assert "URLs" in joined
    assert "negation cues changed" in joined
    assert any("quantities/units" in issue for issue in
               content_drift("The sample remained at 40 K.",
                             "The sample remained at 40 C."))
    assert any("proper names" in issue for issue in
               content_drift("The study was run at Oxford University.",
                             "The study was run at Stanford University."))
    assert not any("numbers" in issue for issue in
                   content_drift("The trial ended in 2024, after 3 runs.",
                                 "After 3 runs, the trial ended in 2024."))
    assert not any("proper names" in issue for issue in content_drift(
        "# Discussion\n\nDespite the limitation, the result held.",
        "# Discussion\n\nThe result held under the stated limitation.",
    ))


def test_content_gate_allows_concise_removal_of_boilerplate_when_facts_survive():
    original = (
        "Additionally, this groundbreaking framework serves as a pivotal method for "
        "assessing cortical responses in 42 adult participants. The study was conducted "
        "at Oxford University in 2024, and participants completed 3 blocks of 18 trials. "
        "It is important to note that the protocol did not include sedative medication [1]."
    )
    concise = (
        "This framework assessed cortical responses in 42 adult participants. The study "
        "was conducted at Oxford University in 2024, and participants completed 3 blocks "
        "of 18 trials. The protocol did not include sedative medication [1]."
    )
    assert len(concise.split()) < len(original.split()) * 0.75
    assert content_drift(original, concise) == []


def _paper(identifier: str, title: str, body: str, abstract: str) -> Paper:
    import hashlib

    return Paper(
        arxiv_id=identifier, title=title, abstract=abstract, text=body,
        body_source="tex", content_hash=hashlib.sha256(body.encode()).hexdigest(),
        source="arxiv",
    )


def test_full_text_selection_excludes_wrong_genres_and_populations():
    body = ("Methods. Adult patients completed cerebral MRI during hypoxia. "
            "Results. Cerebral oxygen metabolism was measured. ") * 30
    papers = [
        _paper("primary", "Cerebral oxygen metabolism after mild traumatic brain injury",
               body, "We studied adult patients with mTBI using MRI during hypoxia."),
        _paper("review", "Cerebral oxygen metabolism after concussion: a narrative review",
               body, "This narrative review summarizes MRI studies."),
        _paper("protocol", "Hypoxic MRI after mTBI: study protocol", body,
               "This study protocol describes planned recruitment."),
        _paper("animal", "Cerebral metabolism after mild brain injury in mice", body,
               "Mice underwent hypoxia and MRI."),
        _paper("neonatal", "MRI of neonatal hypoxic brain injury", body,
               "Neonatal patients underwent MRI after hypoxia."),
        _paper("overview", "Mild traumatic brain injury: mechanisms and outcomes",
               body, "Current evidence describes injury mechanisms and future work."),
    ]
    selected, _ = _select_full_texts(
        "adult post-concussion patients cerebral oxygen metabolism hypoxia MRI",
        papers,
        article_text="Adult patients with persistent post-concussion symptoms.",
    )
    assert [paper.bare_id for paper in selected] == ["primary"]


def test_primary_scholarship_gate_is_not_biomed_specific():
    physics = ("We derive the quantum lattice response and report the measured phase. "
               "Quantum lattice response determines the phase transition. ") * 30
    history = ("This article reconstructs maritime court practice from port records. "
               "Maritime court records document commercial practice. ") * 30
    review = ("This narrative review summarizes quantum lattice research. "
              "Quantum lattice response appears in prior studies. ") * 30
    selected_physics, _ = _select_full_texts(
        "quantum lattice response phase transition",
        [_paper("physics", "Quantum lattice response", physics,
                "We derive the quantum lattice response."),
         _paper("review", "Quantum lattice response: a narrative review", review,
                "This narrative review summarizes prior work.")],
    )
    selected_history, _ = _select_full_texts(
        "maritime court records commercial practice",
        [_paper("history", "Maritime court records", history,
                "This article reconstructs maritime court practice.")],
    )
    assert [paper.bare_id for paper in selected_physics] == ["physics"]
    assert [paper.bare_id for paper in selected_history] == ["history"]


def test_full_text_selection_deduplicates_provider_copies_by_title():
    tex = _paper(
        "2401.00001", "Carrollian amplitudes and celestial symmetries",
        (r"\section{Introduction} We derive Carrollian amplitudes for Yang-Mills "
         r"helicity and positive energy transforms. " * 35),
        "We derive Carrollian amplitudes for Yang-Mills helicity and positive energy.",
    )
    pdf = _paper(
        "doi:10.1000/carroll", "Carrollian Amplitudes and Celestial Symmetries",
        ("We present Carrollian amplitudes for Yang-Mills helicity through positive "
         "energy transforms. " * 35),
        "We present Carrollian amplitudes for Yang-Mills helicity and positive energy.",
    )
    pdf.body_source = "pdf"
    pdf.doi = "10.1000/carroll"
    selected, _rows = _select_full_texts(
        "Carrollian amplitudes Yang-Mills helicity positive energy transforms",
        [pdf, tex],
    )
    assert [paper.bare_id for paper in selected] == ["2401.00001"]


def test_short_topic_anchor_requires_every_identity_term():
    paper = _paper(
        "nontraumatic", "Non-traumatic brain injury in intensive care",
        "We studied non-traumatic brain injury in adult patients. " * 30,
        "We studied adult patients with non-traumatic brain injury.",
    )
    assert _matched_topic_anchors(["mild traumatic brain injury"], paper) == []


def test_deep_profile_keeps_only_close_unique_full_text_and_persists(tmp_path,
                                                                    monkeypatch):
    article = tmp_path / "draft.md"
    text = ("# Quantum widget dynamics\n\n" +
            "We study quantum widget dynamics in a driven lattice. " * 80)
    article.write_text(text)
    corpus = Corpus(tmp_path / "store")
    close_body = (r"\section{Introduction} Quantum widget dynamics in a driven lattice. "
                  r"\section{Results} We measure widget response. ") * 40
    duplicate_body = close_body
    unrelated_body = (r"\section{Introduction} Medieval poetry and manuscript history. "
                      r"\section{Discussion} Verse and palaeography. ") * 40
    for paper in (
        _paper("2401.00001", "Quantum widget dynamics", close_body,
               "Quantum widget response in a driven lattice."),
        _paper("2401.00002", "Quantum widget response", duplicate_body,
               "Quantum widget response in a driven lattice."),
        _paper("2401.00003", "Medieval verse", unrelated_body,
               "A catalogue of medieval poetry."),
    ):
        corpus.papers[paper.bare_id] = paper
    corpus.save()
    monkeypatch.setattr(corpus, "graph_deepen", lambda **kwargs: {"rounds": 0})

    plan = {
        "title": "Quantum widget dynamics", "field": "quantum materials",
        "summary": "Driven quantum widgets in a lattice",
        "terminology": ["quantum", "widget", "lattice"],
        "channels": [], "categories": [], "queries": [],
    }
    profile = build_deep_profile(
        article, text, assessment=ArticleAssessment(True, 5, ["test"], 900),
        plan=plan, corpus=corpus,
    )
    assert [row["id"] for row in profile.selected_papers] == ["2401.00001"]
    assert profile.template.sample_size == 1
    assert profile.manifest_path and profile.manifest_path.is_file()
    manifest = json.loads(profile.manifest_path.read_text())
    assert manifest["status"] == "ready"
    assert "without unrelated padding" in " ".join(manifest["warnings"])
    assert profile.manifest_path.with_name("style-guide.md").is_file()


def test_deep_profile_records_retrieval_failure_and_falls_back(tmp_path, monkeypatch):
    article = tmp_path / "draft.txt"
    text = "A long article body. " * 100
    article.write_text(text)
    corpus = Corpus(tmp_path / "empty-store")

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(corpus, "build", fail)
    profile = build_deep_profile(
        article, text, assessment=ArticleAssessment(True, 4, ["long"], 1200),
        plan={"title": "Widgets", "field": "physics", "summary": "",
              "terminology": ["widgets"], "channels": ["arxiv"],
              "categories": [], "queries": ["quantum widgets"]},
        corpus=corpus,
    )
    manifest = json.loads(profile.manifest_path.read_text())
    assert manifest["status"] == "fallback"
    assert profile.template.sample_size == 0
    assert "Wikipedia tells only" in profile.warning


def test_deep_profile_cache_is_scoped_to_field_and_query_plan(tmp_path, monkeypatch):
    article = tmp_path / "draft.md"
    text = ("# Widget response\n\n" +
            "We measured cortical widget response during stimulation. " * 80)
    article.write_text(text)
    corpus = Corpus(tmp_path / "store")
    body = (r"\section{Introduction} Cortical widget response during stimulation. "
            r"\section{Results} We measured widget response amplitude. ") * 40
    paper = _paper("2401.00009", "Cortical widget response", body,
                   "Cortical widget response during stimulation.")
    corpus.papers[paper.bare_id] = paper
    corpus.save()
    monkeypatch.setattr(corpus, "graph_deepen", lambda **kwargs: {"rounds": 0})
    assessment = ArticleAssessment(True, 5, ["test"], 900)
    first_plan = {
        "title": "Widget response", "field": "clinical neuroscience", "summary": "",
        "terminology": ["cortical", "widget", "response"], "channels": [],
        "categories": [], "queries": ["cortical widget response"],
    }
    second_plan = {
        **first_plan, "field": "materials science", "queries": ["lattice widget response"],
    }
    first = build_deep_profile(article, text, assessment=assessment,
                               plan=first_plan, corpus=corpus)
    second = build_deep_profile(article, text, assessment=assessment,
                                plan=second_plan, corpus=corpus)
    assert first.manifest_path != second.manifest_path
    first_manifest = json.loads(first.manifest_path.read_text())
    second_manifest = json.loads(second.manifest_path.read_text())
    assert first_manifest["plan_fingerprint"] != second_manifest["plan_fingerprint"]
    assert first_manifest["search_plan"]["field"] == "clinical neuroscience"
    assert second_manifest["search_plan"]["field"] == "materials science"


def test_deep_profile_does_not_skip_search_merely_because_cache_has_many_papers(
        tmp_path, monkeypatch):
    article = tmp_path / "draft.md"
    text = "# Quantum widget dynamics\n\n" + (
        "We measured quantum widget response in the driven lattice. " * 90
    )
    article.write_text(text)
    corpus = Corpus(tmp_path / "store")
    for index in range(8):
        body = (
            r"\section{Methods} We measured quantum widget response in the driven lattice. "
            r"\section{Results} Quantum widget dynamics changed under drive. "
        ) * 30
        paper = _paper(
            f"2401.10{index:03d}", f"Quantum widget lattice response {index}", body,
            "We measured quantum widget response in a driven lattice.",
        )
        corpus.papers[paper.bare_id] = paper
    corpus.save()
    calls = []

    def searched(query, **kwargs):
        calls.append(query)
        corpus.last_build_report = {
            "source_ok": True, "result_count": 8,
            "result_ids": list(corpus.papers),
        }
        return []

    monkeypatch.setattr(corpus, "build", searched)
    monkeypatch.setattr(corpus, "graph_deepen", lambda **kwargs: {"round_reports": []})
    plan = {
        "title": "Quantum widget dynamics", "field": "quantum materials",
        "summary": "Driven lattice response", "terminology": ["quantum widget"],
        "channels": ["arxiv"], "categories": [],
        "queries": ["quantum widget lattice response"],
    }
    build_deep_profile(
        article, text, assessment=ArticleAssessment(True, 5, ["test"], 900),
        plan=plan, corpus=corpus,
    )
    assert calls and calls[0] == "quantum widget lattice response"
    manifest_root = next((article.parent / ".spiral" / "prose").glob("*/manifest.json"))
    manifest = json.loads(manifest_root.read_text())
    assert manifest["research_map"]
    assert "coverage" in manifest


def test_jats_full_text_counts_as_primary_for_research_coverage():
    from spiral.research_quality import CoveragePolicy, corpus_quality_report

    papers = []
    for index in range(4):
        paper = _paper(
            f"pmc:{index}", f"Cortical oxygen metabolism imaging {index}",
            ("Cortical oxygen metabolism was measured using brain imaging. " * 30),
            "Cortical oxygen metabolism brain imaging.",
        )
        paper.body_source = "jats"
        papers.append(paper)
    report = corpus_quality_report(
        "cortical oxygen metabolism brain imaging", papers,
        {"searches": [], "graph_rounds": []},
        policy=CoveragePolicy(
            min_papers=1, min_usable_texts=1, min_relevant_papers=1,
            min_relevant_usable_primary_texts=1, min_unique_queries=0,
            min_healthy_searches=0, min_relevant_query_families=0,
            min_topic_term_coverage=0.1,
        ),
    )
    assert report["usable_primary_text_count"] == 4
    assert report["relevant_usable_primary_text_count"] == 4


class _UI:
    def __init__(self):
        self.dash = SimpleNamespace(c=SimpleNamespace())
        self.lines = []

    def stage(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(arg) for arg in args))

    def detail(self, *args, **kwargs):
        pass

    def tokens(self, *args, **kwargs):
        pass

    def done(self, *args, **kwargs):
        pass

    def blocked(self, *args, **kwargs):
        pass


class _ScriptedRewrite:
    providers = {}

    def __init__(self, replies):
        self.replies = list(replies)
        self.users = []

    def chat(self, model, messages, **kwargs):
        self.users.append(messages[-1]["content"])
        return SimpleNamespace(text=self.replies.pop(0), prompt_tokens=2,
                               completion_tokens=3)


def test_rewrite_compares_every_round_uses_feedback_and_blocks_source_overlap(
        tmp_path, monkeypatch):
    import spiral.config as config_module
    import spiral.llm as llm_module
    from spiral.config import Config
    from spiral.style_tool import _rewrite

    original = ("Additionally, this framework serves as a pivotal method for the "
                "analysis of all 42 carefully measured laboratory samples in practice.")
    copied = ("This framework is a direct method for the analysis of all 42 carefully "
              "measured laboratory samples in practice and in the study.")
    selected = ("We apply this framework directly to the analysis of all 42 carefully "
                "measured laboratory samples in practice during the study.")
    invented = selected.replace("42", "43")
    source = SimpleNamespace(
        bare_id="source-1", abstract="", text="Before. " + copied + " After.")
    scripted = _ScriptedRewrite([copied, selected, invented])
    cfg = Config()
    monkeypatch.setattr(config_module.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(llm_module, "Ollama", lambda *args, **kwargs: scripted)

    source_path = tmp_path / "draft.md"
    source_path.write_text(original)
    doc = read_document(source_path)
    out = tmp_path / "human.md"
    args = SimpleNamespace(api=None, model="test-model", rounds=3, out=str(out))
    before = {"ai_score": ai_score(doc.text())}
    result = _rewrite(
        _UI(), args, source_path, doc, None, before, 1,
        corpus_papers=[source], default_output_path=lambda path: out,
        write_document=write_document,
    )
    assert result == 0
    assert len(scripted.users) == 3, "first improvement must not stop best-of-N"
    assert "source-overlap rejection" in scripted.users[1]
    assert "EXACT PRESERVATION INVENTORY" in scripted.users[0]
    assert "numbers (same values and multiplicity): 42" in scripted.users[0]
    assert out.read_text() == selected


def test_rewrite_refines_an_improved_candidate_with_residual_tells(tmp_path,
                                                                   monkeypatch):
    import spiral.config as config_module
    import spiral.llm as llm_module
    from spiral.config import Config
    from spiral.style_tool import _rewrite

    original = (
        "Additionally, this intricate framework serves as a pivotal method for "
        "assessing all 42 measured laboratory samples during the study."
    )
    residual = (
        "The study marks a milestone in assessing all 42 measured laboratory "
        "samples with this framework."
    )
    clean = (
        "The study used this framework to directly assess all 42 measured laboratory "
        "samples."
    )
    scripted = _ScriptedRewrite([residual, clean])
    cfg = Config()
    monkeypatch.setattr(config_module.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(llm_module, "Ollama", lambda *args, **kwargs: scripted)
    source_path = tmp_path / "draft.md"
    source_path.write_text(original)
    doc = read_document(source_path)
    out = tmp_path / "human.md"
    args = SimpleNamespace(api=None, model="test-model", rounds=2, out=str(out))

    assert _rewrite(
        _UI(), args, source_path, doc, None, {"ai_score": ai_score(original)}, 1,
        corpus_papers=[], default_output_path=lambda path: out,
        write_document=write_document,
    ) == 0
    assert "TARGET PASSAGE:\n" + residual in scripted.users[1]
    assert out.read_text() == clean


def test_deep_implies_rewrite_even_for_a_short_non_article(tmp_path, monkeypatch):
    import spiral.config as config_module
    import spiral.llm as llm_module
    import spiral.prose_ui as prose_ui_module
    from spiral.config import Config
    from spiral.style_tool import run_style

    original = ("Additionally, this framework serves as a pivotal method for all "
                "42 carefully measured laboratory samples in practice.")
    rewritten = ("We apply this framework as the stated method for all 42 carefully "
                 "measured laboratory samples in practice.")
    source_path = tmp_path / "note.md"
    out = tmp_path / "note.human.md"
    source_path.write_text(original)
    scripted = _ScriptedRewrite([rewritten, rewritten, rewritten])
    cfg = Config()
    monkeypatch.setattr(config_module.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(llm_module, "Ollama", lambda *args, **kwargs: scripted)
    monkeypatch.setattr(prose_ui_module, "ProseProgress",
                        lambda *args, **kwargs: _UI())
    args = SimpleNamespace(
        file=str(source_path), deep=True, rewrite=False, corpus=None,
        field="", query=[], api=None, model="test-model", rounds=3,
        out=str(out),
    )
    assert run_style(SimpleNamespace(print=lambda *args, **kwargs: None), args) == 0
    assert out.read_text() == rewritten


def test_deep_docx_with_no_safe_candidate_writes_exact_audited_copy(
        tmp_path, monkeypatch):
    import spiral.config as config_module
    import spiral.llm as llm_module
    from spiral.config import Config
    from spiral.style_tool import _rewrite

    source_path = tmp_path / "clean.docx"
    source_path.write_bytes(b"opaque-word-package-for-copy-path")
    doc = SimpleNamespace(
        kind="docx", text=lambda: "Plain measured prose.", editable_segments=[],
        segments=[], with_replacements=lambda replacements: "Plain measured prose.",
    )
    cfg = Config()
    monkeypatch.setattr(config_module.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(llm_module, "Ollama", lambda *args, **kwargs: SimpleNamespace())
    out = tmp_path / "clean.edited.docx"
    args = SimpleNamespace(api=None, model="test-model", rounds=5, out=str(out),
                           deep=True)
    assert _rewrite(
        _UI(), args, source_path, doc, None, {"ai_score": 0.0}, 1,
        corpus_papers=[], default_output_path=lambda path: out,
        write_document=lambda *args, **kwargs: None,
    ) == 0
    assert out.read_bytes() == source_path.read_bytes()


def test_cli_help_and_deep_plan_are_wired(monkeypatch, capsys):
    import sys
    from spiral.cli import entry
    from spiral.prose_ui import prose_plan

    monkeypatch.setattr(sys, "argv", ["spiral", "prose", "--help"])
    try:
        entry()
    except SystemExit as exc:
        assert exc.code == 0
    help_text = capsys.readouterr().out
    assert "--deep" in help_text and "--field" in help_text and "--query" in help_text
    assert "--beef-up" in help_text and "--restructure" in help_text
    titles = [milestone.title for milestone in
              prose_plan(with_corpus=False, rewriting=True, deep=True).milestones]
    assert titles == ["read the document", "field research", "rewrite", "write out"]
    output_tasks = prose_plan(
        with_corpus=False, rewriting=True, deep=True, beef_up=True,
        restructure=True,
    ).milestones[-1].tasks
    assert [task.title for task in output_tasks] == [
        "write the document", "align section arc", "add cited detail", "re-measure",
    ]
