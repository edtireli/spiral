"""Deterministic research scheduling, counterfactuals, and local taste memory.

The reasoning model proposes candidate actions.  This module decides their order
from observable quantities: uncovered topic terms, retrieval health, query
redundancy, source yield, grounding, checkability, and prior outcomes.  The local
taste model is intentionally only a tie-breaker among scientifically admissible
angles; it never overrides evidence or novelty gates.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path


def _tokens(value: str) -> set[str]:
    stop = {
        "about", "after", "again", "against", "also", "among", "and", "are",
        "been", "before", "being", "between", "could", "does", "every", "find",
        "for", "from", "have", "into", "more", "most", "not", "only", "other",
        "our", "should", "such", "than", "that", "the", "their", "then", "there",
        "these", "they", "this", "through", "under", "using", "was", "were",
        "what", "when", "where", "which", "while", "with", "would",
    }
    return {
        token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", str(value or "").lower())
        if token not in stop
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _load(path: Path, default):
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, type(default)):
                return value
        except Exception:
            pass
    return default


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class InformationGainScheduler:
    """Rank research actions by measured expected information gain."""

    def __init__(self, research_root: str | Path, topic: str):
        self.root = Path(research_root) / "strategy"
        self.path = self.root / "information-gain.json"
        self.topic = topic
        self.state = _load(self.path, {
            "schema_version": 1,
            "topic": topic,
            "observations": [],
            "rankings": [],
        })
        self.state.setdefault("observations", [])
        self.state.setdefault("rankings", [])

    @staticmethod
    def _corpus_blob(corpus) -> str:
        return " ".join(
            f"{getattr(p, 'title', '')} {getattr(p, 'abstract', '')}"
            for p in getattr(corpus, "papers", {}).values()
        ).lower()

    def rank_queries(self, queries: list[str], *, research_map: dict,
                     coverage: dict, corpus) -> list[dict]:
        clean = []
        seen = set()
        for value in queries:
            query = " ".join(str(value or "").split())
            if query and query.lower() not in seen:
                seen.add(query.lower())
                clean.append(query)
        previous = [
            str(item.get("query") or "") for item in research_map.get("searches", [])
            if item.get("query")
        ]
        previous_tokens = [_tokens(query) for query in previous]
        topic_tokens = _tokens(self.topic)
        corpus_blob = self._corpus_blob(corpus)
        covered_topic = {term for term in topic_tokens if term in corpus_blob}
        uncovered_topic = topic_tokens - covered_topic
        observations = self.state.get("observations", [])
        rows = []
        for query in clean:
            q_tokens = _tokens(query)
            redundancy = max((_jaccard(q_tokens, prior) for prior in previous_tokens), default=0.0)
            novelty = 1.0 - redundancy
            uncovered = len(q_tokens & uncovered_topic) / max(1, len(q_tokens))
            related = len(q_tokens & topic_tokens) / max(1, len(q_tokens))
            similar = [
                item for item in observations
                if _jaccard(q_tokens, _tokens(item.get("query", ""))) >= 0.45
            ]
            if similar:
                health = sum(1.0 if row.get("source_ok") else 0.0 for row in similar) / len(similar)
                yield_score = sum(
                    min(1.0, float(row.get("added", 0)) / max(1.0, float(row.get("k", 8))))
                    for row in similar
                ) / len(similar)
                uncertainty = 1.0 / math.sqrt(len(similar) + 1)
            else:
                health = 0.65
                yield_score = 0.5
                uncertainty = 1.0
            blocker_text = " ".join(coverage.get("blocking_reasons") or []).lower()
            blocker_boost = 0.0
            if any(term in blocker_text for term in q_tokens):
                blocker_boost = 1.0
            score = (
                0.28 * novelty
                + 0.22 * uncovered
                + 0.15 * related
                + 0.13 * health
                + 0.10 * yield_score
                + 0.07 * uncertainty
                + 0.05 * blocker_boost
            )
            if query.lower() in {p.lower() for p in previous}:
                score *= 0.35
            rows.append({
                "query": query,
                "score": round(score, 6),
                "components": {
                    "query_novelty": round(novelty, 4),
                    "uncovered_topic_terms": round(uncovered, 4),
                    "topic_relation": round(related, 4),
                    "historical_source_health": round(health, 4),
                    "historical_yield": round(yield_score, 4),
                    "exploration_uncertainty": round(uncertainty, 4),
                    "coverage_blocker_match": blocker_boost,
                },
                "uncovered_terms": sorted(q_tokens & uncovered_topic),
            })
        rows.sort(key=lambda row: (-row["score"], row["query"].lower()))
        self.state["rankings"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": "query",
            "coverage_digest": {
                "discovery_ready": bool(coverage.get("discovery_ready")),
                "novelty_ready": bool(coverage.get("novelty_ready")),
                "relevant_papers": int(coverage.get("relevant_paper_count") or 0),
            },
            "rows": rows,
        })
        self.state["rankings"] = self.state["rankings"][-100:]
        _save(self.path, self.state)
        return rows

    def observe_search(self, query: str, *, added: int, k: int, retrieval: dict) -> None:
        report = retrieval.get("fallback") if isinstance(retrieval.get("fallback"), dict) else retrieval
        self.state["observations"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query": " ".join(str(query or "").split()),
            "added": int(added),
            "k": int(k),
            "source_ok": report.get("source_ok") is True,
            "result_count": int(report.get("result_count") or 0),
        })
        self.state["observations"] = self.state["observations"][-500:]
        _save(self.path, self.state)

    def rank_families(self, families: list[dict], notes: list[dict]) -> list[dict]:
        grounded_ids = {
            str(note.get("arxiv_id") or "").replace("arXiv:", "").split("v")[0]
            for note in notes if note.get("grounded")
        }
        rows = []
        for family in families:
            paper_refs = [str(x) for x in (
                (family.get("deep_read_papers") or []) + (family.get("key_papers") or []))]
            grounded = sum(
                1 for ref in paper_refs if any(pid and pid in ref for pid in grounded_ids))
            question_count = len(family.get("question_seeds") or [])
            checks = " ".join(str(x) for x in (family.get("first_checks") or [])).lower()
            checkability = min(1.0, sum(
                word in checks for word in (
                    "symbolic", "numeric", "lean", "groebner", "certificate",
                    "residual", "substitution", "proof", "exact",
                )) / 3.0)
            risks = len(family.get("missing_or_risks") or [])
            score = (
                0.40 * min(1.0, grounded / 2.0)
                + 0.25 * checkability
                + 0.20 * min(1.0, question_count / 2.0)
                + 0.15 * (1.0 / (1.0 + risks))
            )
            rows.append({**family, "_information_score": round(score, 6)})
        return sorted(rows, key=lambda row: (-row["_information_score"], str(row.get("name", ""))))

    def plateau_report(self, *, patience: int, floor: float,
                       coverage_ready: bool) -> dict:
        observations = self.state.get("observations", [])[-max(1, patience):]
        if len(observations) < patience:
            return {"exhausted": False, "reason": "insufficient observations", "mean_gain": None}
        gains = [
            (min(1.0, row.get("added", 0) / max(1, row.get("k", 8)))
             if row.get("source_ok") else 0.0)
            for row in observations
        ]
        mean = sum(gains) / len(gains)
        healthy = sum(1 for row in observations if row.get("source_ok"))
        exhausted = bool(coverage_ready and healthy >= max(2, patience // 2) and mean < floor)
        return {
            "exhausted": exhausted,
            "reason": (
                "healthy recent searches yielded negligible new corpus information"
                if exhausted else "observable search frontier still has information value"
            ),
            "mean_gain": round(mean, 6),
            "healthy_observations": healthy,
            "window": len(observations),
            "floor": floor,
        }


class LocalTasteModel:
    """Small transparent online preference model stored only on this machine."""

    FEATURES = (
        "specificity", "exactness", "verification", "mechanism", "scope_control",
        "cross_method", "corpus_basis", "productive_negative", "excess_breadth",
    )

    DEFAULT_WEIGHTS = {
        "specificity": 0.85,
        "exactness": 0.75,
        "verification": 1.0,
        "mechanism": 0.75,
        "scope_control": 0.85,
        "cross_method": 0.55,
        "corpus_basis": 1.0,
        "productive_negative": 0.45,
        "excess_breadth": -0.9,
    }

    def __init__(self, research_root: str | Path, topic: str):
        self.path = Path(research_root) / "strategy" / "taste-profile.json"
        self.global_path = Path.home() / ".local" / "share" / "spiral" / "research-taste.json"
        global_state = _load(self.global_path, {})
        inherited = {
            "schema_version": 1,
            "weights": dict(global_state.get("weights") or self.DEFAULT_WEIGHTS),
            "term_weights": dict(
                global_state.get("term_weights") or global_state.get("topic_terms") or {}),
            "current_topic_terms": sorted(_tokens(topic)),
            "observations": [],
        }
        self.state = _load(self.path, inherited)
        self.state.setdefault("weights", dict(self.DEFAULT_WEIGHTS))
        self.state.setdefault("term_weights", dict(self.state.pop("topic_terms", {}) or {}))
        self.state["current_topic_terms"] = sorted(_tokens(topic))
        self.state.setdefault("observations", [])
        self._persist()

    def _persist(self, *, global_update: bool = False) -> None:
        _save(self.path, self.state)
        if global_update:
            _save(self.global_path, {
                "schema_version": 1,
                "weights": self.state.get("weights", {}),
                "term_weights": self.state.get("term_weights", {}),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "observation_count": len(self.state.get("observations", [])),
            })

    def features(self, angle: dict) -> dict[str, float]:
        question = str(angle.get("question") or "")
        blob = " ".join(str(value) for value in angle.values()).lower()
        words = question.split()
        corpus_basis = len(angle.get("corpus_basis") or [])
        specificity_terms = (
            "fixed", "dimension", "dimensional", "parameter", "algebra", "model",
            "sector", "locus", "boundary", "multipole", "group", "manifold",
        )
        exact_terms = ("exact", "classify", "theorem", "identity", "closed form", "no-go")
        verify_terms = (
            "certificate", "symbolic", "numeric", "substitution", "groebner",
            "lean", "proof", "residual", "integration", "code",
        )
        mechanism_terms = (
            "mechanism", "symmetry", "operator", "conserved", "structure",
            "why", "obstruction", "duality",
        )
        scope_terms = ("within", "bounded", "ansatz", "under", "assuming", "fixed")
        cross_terms = ("independent", "two methods", "symbolic and", "numeric and", "cross-check")
        negative_terms = ("no-go", "obstruction", "impossible", "nonexistence", "failure")
        return {
            "specificity": _clamp(sum(term in blob for term in specificity_terms) / 3),
            "exactness": _clamp(sum(term in blob for term in exact_terms) / 2),
            "verification": _clamp(sum(term in blob for term in verify_terms) / 3),
            "mechanism": _clamp(sum(term in blob for term in mechanism_terms) / 2),
            "scope_control": _clamp(sum(term in blob for term in scope_terms) / 2),
            "cross_method": _clamp(sum(term in blob for term in cross_terms) / 2),
            "corpus_basis": _clamp(corpus_basis / 3),
            "productive_negative": _clamp(sum(term in blob for term in negative_terms) / 2),
            "excess_breadth": _clamp(max(0, len(words) - 55) / 55),
        }

    def score(self, angle: dict) -> tuple[float, dict]:
        features = self.features(angle)
        weights = self.state.get("weights", self.DEFAULT_WEIGHTS)
        raw = sum(float(weights.get(key, 0.0)) * value for key, value in features.items())
        term_weights = self.state.get("term_weights", {})
        overlap = sum(float(term_weights.get(token, 0))
                      for token in _tokens(angle.get("question", "")))
        topic_bonus = max(-0.35, min(0.35, overlap / 30.0))
        return round(raw + topic_bonus, 6), {
            "features": features,
            "topic_bonus": round(topic_bonus, 6),
        }

    def rank(self, angles: list[dict]) -> list[dict]:
        rows = []
        for angle in angles:
            score, explanation = self.score(angle)
            rows.append({**angle, "_taste_score": score, "_taste_explanation": explanation})
        return sorted(rows, key=lambda row: (-row["_taste_score"], str(row.get("question", ""))))

    def observe(self, angle: dict, outcome: str) -> None:
        target = 1.0 if outcome in {"pursue", "accepted", "verified"} else -1.0
        features = self.features(angle)
        score, _ = self.score(angle)
        prediction = math.tanh(score / 2.5)
        error = target - prediction
        # Explicit user feedback owns taste. Scientific outcomes are retained as
        # context and only a verified result supplies a tiny utility calibration;
        # "known" or "thin" must not teach the system that the user dislikes an idea.
        rate = 0.12 if outcome in {"accepted", "rejected"} else 0.02 if outcome == "verified" else 0.0
        for key, value in features.items():
            old = float(self.state["weights"].get(key, 0.0))
            self.state["weights"][key] = round(_clamp(
                old + rate * error * value, -2.0, 2.0), 6)
        if outcome in {"accepted", "rejected"}:
            direction = 1 if outcome == "accepted" else -1
            for token in _tokens(angle.get("question", "")):
                old = int(self.state["term_weights"].get(token, 0))
                self.state["term_weights"][token] = max(-20, min(20, old + direction))
        self.state["observations"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question": str(angle.get("question") or "")[:800],
            "outcome": outcome,
            "features": features,
            "score_before": score,
        })
        self.state["observations"] = self.state["observations"][-500:]
        self._persist(global_update=outcome in {"accepted", "rejected"})


class CounterfactualLab:
    """Persist and validate neighboring hypotheses without promoting them to facts."""

    ALLOWED_MOVES = {
        "relax_assumption", "strengthen_assumption", "boundary_case",
        "singular_limit", "method_transfer", "dual_formulation",
        "obstruction", "negative_theorem", "parameter_extension",
    }

    def __init__(self, research_root: str | Path):
        self.root = Path(research_root) / "counterfactuals"
        self.root.mkdir(parents=True, exist_ok=True)

    def validate(self, parent: dict, candidate: dict) -> tuple[bool, list[str]]:
        issues = []
        question = " ".join(str(candidate.get("question") or "").split())
        move = str(candidate.get("move") or "")
        if not question:
            issues.append("missing question")
        if move not in self.ALLOWED_MOVES:
            issues.append("unrecognized counterfactual move")
        if not str(candidate.get("changed_assumption") or "").strip():
            issues.append("changed assumption is not explicit")
        if not str(candidate.get("falsifier") or "").strip():
            issues.append("falsifier is not explicit")
        if not str(candidate.get("first_check") or "").strip():
            issues.append("first check is not explicit")
        parent_tokens = _tokens(parent.get("question", ""))
        candidate_tokens = _tokens(question)
        relation = _jaccard(parent_tokens, candidate_tokens)
        if relation < 0.12:
            issues.append("candidate drifted away from its parent angle")
        if relation > 0.94:
            issues.append("candidate does not materially change the parent")
        return not issues, issues

    def save_round(self, round_no: int, parents: list[dict], candidates: list[dict]) -> Path:
        path = self.root / f"round-{round_no}.json"
        _save(path, {
            "schema_version": 1,
            "round": round_no,
            "parents": parents,
            "candidates": candidates,
        })
        return path
