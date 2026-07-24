from __future__ import annotations

import json
import shutil
from pathlib import Path


def _closed_result_graph(root: Path):
    from spiral.epistemic import ObligationGraph

    graph = ObligationGraph(root, "Classify a bounded model")
    question = graph.ensure(
        "question", "Does the bounded model have the stated locus?",
        node_id="question:q", stage="result", required=True, status="in_progress")
    claim = graph.ensure(
        "claim", "The locus is x = 0", node_id="claim:c", stage="result",
        required=True, metadata={"requires_replication": True})
    assumption = graph.ensure(
        "assumption", "x is real", node_id="assumption:a", stage="result",
        required=True, status="declared")
    novelty = graph.ensure(
        "novelty", "Bound the prior-art status", node_id="novelty:n",
        stage="result", required=True)
    graph.link(claim, question, "answers")
    graph.link(claim, assumption, "depends_on")
    graph.link(novelty, question, "scopes")
    graph.add_evidence(
        claim, "exact residual is zero", evidence_kind="verification",
        verifier="sympy", node_id="verification:v")
    graph.set_status(claim, "supported", verifier="sympy")
    graph.add_evidence(
        novelty, "bounded healthy search", evidence_kind="novelty_certificate",
        verifier="protocol", relation="scopes", node_id="novelty-certificate:n")
    graph.set_status(novelty, "supported", verifier="protocol")
    decision = graph.ensure(
        "decision", "scope answered", node_id="decision:d", stage="result",
        status="supported")
    graph.link(decision, question, "supports")
    graph.set_status(question, "supported", verifier="completion gate")
    graph.save()
    return graph, claim


def test_obligation_graph_requires_independent_replication_and_verifies_event_chain(tmp_path):
    graph, claim = _closed_result_graph(tmp_path)

    blocked = graph.report("result")
    assert blocked["ready"] is False
    assert blocked["replication_gaps"] == [claim]

    graph.add_evidence(
        claim, "blind second derivation", evidence_kind="replication",
        verifier="lean", independent=True, relation="replicates",
        node_id="replication:r")
    graph.save()

    report = graph.report("result")
    assert report["ready"] is True
    assert report["event_chain"]["valid"] is True
    assert graph.markdown_path.is_file()
    assert graph.digest() == graph.compact()["digest"]


def test_obligation_event_chain_detects_tampering(tmp_path):
    graph, _ = _closed_result_graph(tmp_path)
    lines = graph.events_path.read_text().splitlines()
    row = json.loads(lines[-1])
    row["reason"] = "tampered"
    lines[-1] = json.dumps(row)
    graph.events_path.write_text("\n".join(lines) + "\n")

    assert graph.verify_event_chain()["valid"] is False
    assert graph.report("result")["ready"] is False


def test_information_scheduler_prefers_uncovered_nonduplicate_query(tmp_path):
    from types import SimpleNamespace

    from spiral.research_strategy import InformationGainScheduler

    scheduler = InformationGainScheduler(tmp_path, "tidal response Lovelock ladder symmetry")
    corpus = SimpleNamespace(papers={
        "a": SimpleNamespace(title="Lovelock perturbations", abstract="tidal response"),
    })
    research_map = {
        "searches": [{"query": "Lovelock tidal response", "retrieval": {"source_ok": True}}],
    }
    rows = scheduler.rank_queries(
        ["Lovelock tidal response", "ladder symmetry conserved operator"],
        research_map=research_map, coverage={"blocking_reasons": []}, corpus=corpus)

    assert rows[0]["query"] == "ladder symmetry conserved operator"
    assert rows[0]["score"] > rows[1]["score"]


def test_taste_model_only_learns_strongly_from_explicit_user_feedback(tmp_path):
    from spiral.research_strategy import LocalTasteModel

    profile = LocalTasteModel(tmp_path, "integrable models")
    profile.global_path = tmp_path / "global-taste.json"
    before = dict(profile.state["weights"])
    angle = {
        "question": "Classify an exact bounded locus with a symbolic certificate",
        "corpus_basis": ["paper-a", "paper-b"],
    }
    profile.observe(angle, "known")
    assert profile.state["weights"] == before

    profile.observe(angle, "accepted")
    assert profile.state["weights"] != before
    assert profile.global_path.is_file()
    assert any(value > 0 for value in profile.state["term_weights"].values())


def test_counterfactual_lab_rejects_drift_and_accepts_one_change(tmp_path):
    from spiral.research_strategy import CounterfactualLab

    lab = CounterfactualLab(tmp_path)
    parent = {"question": "Classify static tidal response in a fixed Lovelock black hole"}
    valid = {
        "parent_question": parent["question"],
        "question": "Classify static tidal response in the singular Lovelock coupling limit",
        "move": "singular_limit",
        "changed_assumption": "take the degenerate Lovelock coupling limit",
        "falsifier": "a connection coefficient without the predicted limiting pole",
        "first_check": "symbolically expand the connection coefficient",
    }
    drift = {**valid, "question": "Optimize a neural image classifier"}

    assert lab.validate(parent, valid)[0] is True
    assert lab.validate(parent, drift)[0] is False


def test_blind_replication_brief_hides_original_solution_and_requires_model_diversity():
    from spiral.research_replication import blind_brief, independent_enough

    original = {
        "kind": "workbench", "statement": "the residual vanishes",
        "assumptions": ["x is real"], "falsifier": "a nonzero residual",
        "method_family": "symbolic", "files": {"solve.py": "SECRET ORIGINAL CODE"},
        "cmd": "python solve.py", "proof": "SECRET PROOF",
    }
    brief = blind_brief(original, question="Does the residual vanish?")
    encoded = json.dumps(brief)
    assert "SECRET ORIGINAL CODE" not in encoded
    assert "SECRET PROOF" not in encoded
    assert "python solve.py" not in encoded

    replica = {
        "kind": "theorem", "statement": "the residual vanishes",
        "proof": "by ring", "method_family": "formal Lean proof",
    }
    same_model = independent_enough(
        original, "workbench", replica, "lean", "model-a", "model-a")
    other_model = independent_enough(
        original, "workbench", replica, "lean", "model-a", "model-b")
    assert same_model["independent"] is False
    assert other_model["independent"] is True


def test_replica_method_audit_uses_executed_code_not_declared_labels(tmp_path):
    from spiral.research_replication import inspect_replica_methods

    claim = {
        "files": {
            "symbolic.py": (
                "import sympy as s\n"
                "x=s.symbols('x')\n"
                "assert s.expand((x+1)**2-x**2-2*x-1) == 0\n"
                "print('METHOD_OK:symbolic')\n"
            ),
            "numeric.py": (
                "import mpmath as mp\n"
                "for x in [-2, 0, 3.5]:\n"
                "    residual=(x+1)**2-x**2-2*x-1\n"
                "    assert abs(residual) < mp.mpf('1e-30')\n"
                "print('METHOD_OK:numeric')\n"
            ),
        }
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "steps_run": [
            {"argv": ["python", "symbolic.py"], "returncode": 0},
            {"argv": ["python", "numeric.py"], "returncode": 0},
        ],
        "validation_evidence": {"methods": [
            {"name": "symbolic", "step": 0, "valid": True},
            {"name": "numeric", "step": 1, "valid": True},
        ]},
    }))

    audit = inspect_replica_methods(claim, str(manifest))
    assert audit["method_diversity"] is True
    assert audit["adversarial_falsifier_check"] is True

    claim["files"]["numeric.py"] = claim["files"]["symbolic.py"]
    repeated = inspect_replica_methods(claim, str(manifest))
    assert repeated["method_diversity"] is False


def _novelty_certificate(tmp_path: Path) -> dict:
    from spiral.research_provenance import NoveltyBoundaryCertificate

    return NoveltyBoundaryCertificate.build(
        tmp_path,
        question="Is x = 0 the only bounded locus?",
        proposal={"question": "Is x = 0 the only bounded locus?", "conventions": "x real"},
        findings=[{
            "ok": True, "claim_id": "claim-1", "strength": "exact", "backend": "sympy",
            "claim": {"statement": "x = 0 is a locus", "assumptions": ["x real"]},
        }],
        search_report={
            "queries": ["bounded locus", "classification obstruction"],
            "query_reports": [{"query": "bounded locus"}, {"query": "classification obstruction"}],
            "healthy_queries": 2, "healthy_query_families": 2,
            "sources_ok": ["arxiv", "semantic_scholar"],
        },
        priors=[{"title": "Nearby result", "identifier": "2401.00001", "year": 2024}],
        coverage={"novelty_ready": True, "paper_count": 12, "relevant_paper_count": 8},
    )


def test_novelty_boundary_is_bounded_signed_and_tamper_evident(tmp_path):
    from spiral.research_provenance import NoveltyBoundaryCertificate

    certificate = _novelty_certificate(tmp_path)
    assert certificate["valid"] is True
    assert NoveltyBoundaryCertificate.validate(certificate)["valid"] is True
    assert "not proof of global absence" in certificate["scope_statement"]

    path = Path(certificate["path"])
    value = json.loads(path.read_text())
    value["question"] = "tampered"
    path.write_text(json.dumps(value))
    assert NoveltyBoundaryCertificate.validate(path)["valid"] is False


def test_proof_carrying_and_living_paper_detect_artifact_drift(tmp_path):
    from spiral.research_provenance import LivingPaper, ProofCarryingPaper

    research = tmp_path / "research"
    out = research / "writeup"
    out.mkdir(parents=True)
    paper = out / "paper.tex"
    paper.write_text("verified paper")
    audit = out / "paper-audit.json"
    audit.write_text('{"stage":"complete"}')
    obligations = research / "epistemic" / "obligations.json"
    obligations.parent.mkdir(parents=True)
    obligations.write_text('{"nodes":{},"edges":[]}')
    corpus = research / "corpus" / "corpus.json"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"papers":[]}')
    novelty = _novelty_certificate(research)
    finding = {
        "ok": True, "claim_id": "claim-1", "strength": "exact", "backend": "sympy",
        "required": True, "obligation_id": "claim:c",
        "replication": {"passed": True, "status": "passed"},
        "claim": {"statement": "x = 0 is a locus", "assumptions": ["x real"]},
    }
    scope_packet = {"claims": [{"claim_id": "sentence-1", "sentence": "x = 0 is a locus"}]}
    scope_audit = {"claims": [{
        "claim_id": "sentence-1", "status": "verified", "evidence_id": "finding:claim-1",
    }]}
    proof = ProofCarryingPaper.build(
        out, findings=[finding], paper_files=[paper], audit_files=[audit],
        novelty_certificate=novelty["path"], obligation_graph=obligations,
        obligation_report={"ready": True}, citation_packet={"contexts": []},
        scope_packet=scope_packet, scope_audit=scope_audit,
        research_commit="abc123", integrity_chains={
            "thoughts": {"ok": True}, "model_calls": {"ok": True},
            "obligations": {"valid": True},
        })
    assert proof["valid"] is True
    assert ProofCarryingPaper.validate(proof["path"])["valid"] is True

    living = LivingPaper.create(
        research, out, topic="topic", question="question",
        proof_manifest=proof["path"], novelty_certificate=novelty["path"],
        obligation_graph=obligations, corpus_manifest=corpus,
        research_commit="abc123", recheck_days=30)
    assert LivingPaper.inspect(living["path"], research)["current"] is True

    paper.write_text("changed after release")
    assert ProofCarryingPaper.validate(proof["path"])["valid"] is False
    assert LivingPaper.inspect(living["path"], research)["stale"] is True


def test_research_git_checkpoints_without_touching_outer_repository(tmp_path):
    if not shutil.which("git"):
        return
    from spiral.research_history import ResearchGit

    root = tmp_path / "run"
    root.mkdir()
    (root / "state.json").write_text('{"round":1}')
    history = ResearchGit(root)
    first = history.checkpoint("first", phase="round")
    assert first["ok"] and first["commit"]
    assert not (root / ".git").exists()

    (root / "state.json").write_text('{"round":2}')
    second = history.checkpoint("second", phase="round")
    assert second["ok"] and second["commit"] != first["commit"]
    assert history.log()[0]["commit"] == second["commit"]
    verification = history.verify()
    assert verification["valid"] is True
    assert verification["commit_count"] == 2


def test_research_graph_has_switchable_epistemic_layer(tmp_path):
    from spiral.research_graph import build_graph_data, write_graph_view

    map_data = {
        "topic": "A topic", "searches": [], "graph_rounds": [],
        "epistemic": {
            "digest": "abc",
            "nodes": [{
                "id": "claim:c", "kind": "claim", "label": "A verified claim",
                "title": "A verified claim", "status": "supported", "required": True,
            }],
            "edges": [], "result_report": {"ready": True},
        },
    }
    data = build_graph_data(map_data)
    assert any(node.get("layer") == "epistemic" for node in data["nodes"])
    view = write_graph_view(map_data, None, tmp_path)
    html = Path(view["html"]).read_text()
    assert 'data-layer="epistemic"' in html
    assert "let layerMode = 'field'" in html
