from __future__ import annotations

import json
from types import SimpleNamespace

from spiral.paper_audit import generate_paper_audit, write_paper_audit
from spiral.research_corpus import Paper
from spiral.writing_style import StyleTemplate


class _LLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(text=json.dumps(self.payload))


class _Config:
    keep_alive = "5m"

    @staticmethod
    def spec_for(model):
        return SimpleNamespace(num_ctx=16_384)


def _papers():
    return [Paper(
        arxiv_id="2401.00001", title="Carrollian amplitudes and positive cones",
        authors=["Ada Physicist"], published="2024", doi="10.1000/cones",
        abstract="We organize Carrollian amplitudes using positive energy cones.",
        text=("Introduction. We organize Carrollian amplitudes using positive energy "
              "cones and validate the construction. ") * 30,
        body_source="tex",
    )]


def test_audit_keeps_corpus_recommendations_only_with_valid_source_ids(tmp_path):
    payload = {
        "executive_summary": "The central result is clear but the comparison can be sharper.",
        "strengths": ["The theorem is stated before the numerical validation."],
        "recommendations": [
            {"priority": "high", "category": "novelty", "target": "Introduction",
             "action": "State the difference from existing positive-cone treatments explicitly.",
             "reason": "The close paper organizes the same background object but not this wall theorem.",
             "basis": "corpus", "source_ids": ["C1"]},
            {"priority": "high", "category": "evidence", "target": "Discussion",
             "action": "Add an unsupported comparison to a paper outside the held corpus.",
             "reason": "This recommendation has no selected source behind it.",
             "basis": "corpus", "source_ids": ["C99"]},
        ],
        "proposed_section_order": ["Introduction", "Main theorem", "Validation", "Discussion"],
        "do_not_change": ["Do not alter the theorem statement without scientific review."],
    }
    papers = _papers()
    audit = generate_paper_audit(
        _LLM(payload), _Config(), "model",
        r"\section{Introduction} Text. \section{Discussion} Text.", "tex",
        papers, [{"id": papers[0].bare_id, "relevance": 0.8}],
        StyleTemplate(sample_size=1), {"discovery_ready": True},
    )
    assert len(audit.recommendations) == 1
    assert audit.recommendations[0]["source_ids"] == ["C1"]

    markdown, machine = write_paper_audit(tmp_path / "main.tex", audit)
    assert "[C1]" in markdown.read_text()
    assert "automatically apply" in markdown.read_text()
    assert json.loads(machine.read_text())["recommendations"][0]["category"] == "novelty"


def test_audit_reads_complete_manuscript_and_flattens_structured_advisories(tmp_path):
    payload = {
        "executive_summary": "The full manuscript supports a focused editorial audit.",
        "strengths": [{"action": "Preserve the central theorem statement.",
                       "reason": "It defines the manuscript's contribution."}],
        "recommendations": [{
            "priority": "medium", "category": "readability", "target": "Discussion",
            "action": "Define the bullet notation before the final comparison.",
            "reason": "The notation appears before its prose explanation.",
            "basis": "manuscript", "source_ids": [],
        }],
        "proposed_section_order": ["Introduction", "Discussion"],
        "do_not_change": [{"action": "Do not alter the proof coefficients.",
                           "reason": "They require scientific review."}],
    }
    llm = _LLM(payload)
    tail = "END-OF-MANUSCRIPT-SENTINEL"
    papers = _papers()
    audit = generate_paper_audit(
        llm, _Config(), "model",
        r"\section{Introduction} Text. \section{Discussion} " + tail, "tex",
        papers, [{"id": papers[0].bare_id, "relevance": 0.8}],
        StyleTemplate(sample_size=1), {"discovery_ready": True},
    )
    sent = llm.calls[0][0][1][1]["content"]
    assert tail in sent
    assert "MANUSCRIPT (COMPLETE)" in sent
    assert audit.strengths == [
        "Preserve the central theorem statement. — It defines the manuscript's contribution."
    ]
    assert audit.do_not_change == [
        "Do not alter the proof coefficients. — They require scientific review."
    ]
    markdown, _machine = write_paper_audit(tmp_path / "main.tex", audit)
    assert "{'priority'" not in markdown.read_text()
