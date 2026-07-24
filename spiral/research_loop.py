"""spiralʳᵉˢᵉᵃʳᶜʰ — the iterative research conductor.

The research analogue of spiral's build Conductor. A round is:

  gather  → read primary sources into the corpus
  read    → a cheap/local model writes cached paper notes; the thinking model clusters
            them into idea families and deep-reads selected papers
  propose → the thinking model selects a grounded research angle and states *checkable
            claims* (identities, solutions, numerical experiments) — never prose alone
  verify  → every claim is run through a deterministic tool (``verify_math`` /
            ``numeric_lab`` / Lean when present); refuted claims are dropped, survivors
            banked as findings
  novelty → surviving results are searched against the literature (``citations``)
  reflect → the model reads the verdicts + prior art and decides: continue, pivot,
            declare solved, or promote a *new* verified-open question
  persist → state is written so the loop resumes

It repeats — unbounded by default — until a question is answered with verified claims, a
genuinely new open question is found, or a round/token budget is hit. Then it writes a
cited LaTeX paper. The invariant, carried from spiral: **the model proposes; tools
decide.** A finding exists because a checker confirmed it, not because the model was
fluent.
"""

from __future__ import annotations

import base64
import json
import hashlib
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Finding:
    claim: dict
    ok: bool
    backend: str
    detail: str
    round: int
    question: str = ""
    claim_id: str = ""
    strength: str = "unverified"   # formal | exact | computational | empirical | executable
    required: bool = True
    replication: dict = field(default_factory=dict)
    obligation_id: str = ""


@dataclass
class ResearchState:
    topic: str
    question: str = ""
    round: int = 0
    status: str = "open"                 # open | solved | new_question | exhausted
    findings: list = field(default_factory=list)
    corpus_ids: list = field(default_factory=list)
    history: list = field(default_factory=list)   # per-round {action, reason, query}
    tokens: int = 0
    local_tokens: int = 0
    api_tokens: int = 0
    active_proposal: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    completion: dict = field(default_factory=dict)
    novelty_boundary: dict = field(default_factory=dict)
    obligation_report: dict = field(default_factory=dict)
    research_commit: str = ""
    living_paper: dict = field(default_factory=dict)
    data_resources: dict = field(default_factory=dict)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Local models often get the right structure but lose one closing brace after a
    multi-line code string. Parse with brace tracking first, then try a minimal
    balance repair before giving up.
    """
    start = text.find("{")
    if start < 0:
        return {}

    def _escape_controls_in_strings(frag: str) -> str:
        out = []
        in_str = False
        esc = False
        for ch in frag:
            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                elif ch == "\\":
                    out.append(ch)
                    esc = True
                elif ch == '"':
                    out.append(ch)
                    in_str = False
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\t":
                    out.append("\\t")
                elif ch == "\r":
                    out.append("\\r")
                else:
                    out.append(ch)
                continue
            out.append(ch)
            if ch == '"':
                in_str = True
        return "".join(out)

    def _loads(frag: str) -> dict:
        variants = [frag, _escape_controls_in_strings(frag)]
        variants += [re.sub(r",\s*([}\]])", r"\1", v) for v in variants]
        variants += [v.replace("'", '"') for v in variants]
        last_exc = None
        for variant in variants:
            try:
                return json.loads(variant)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        return {}

    def _loads_lenient(frag: str) -> dict:
        try:                              # tolerate trailing commas / single quotes
            return _loads(frag)
        except Exception:
            if depth > 0:
                return _loads(frag + ("}" * depth))
            raise

    depth = 0
    in_str = False
    esc = False
    last = len(text)
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last = i + 1
                break
    frag = text[start:last]
    try:
        return _loads_lenient(frag)
    except Exception:
        return {}


def _prior_objects(priors: list) -> list:
    from spiral.citations import Prior

    out = []
    for p in priors or []:
        if isinstance(p, Prior):
            out.append(p)
        elif isinstance(p, dict):
            try:
                out.append(Prior(**p))
            except TypeError:
                pass
    return out


class ResearchModelError(RuntimeError):
    """A reasoning backend failed in a way the loop cannot honestly paper over."""


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ResearchLoop:
    def __init__(self, topic: str, workdir="./spiral-research", cfg=None, ol=None, ui=None,
                 *, resume: bool = True, mode: str | None = None,
                 refresh: bool = False):
        from spiral.config import Config
        from spiral.llm import Ollama
        from spiral.research_corpus import Corpus
        self.cfg = cfg or Config.load()
        self.ol = ol or Ollama(self.cfg.base_url, providers=getattr(self.cfg, "providers", None))
        self.ui = ui
        self.resume_requested = bool(resume)
        self.refresh_requested = bool(refresh)
        self.mode_override = mode if mode in {"research", "expository"} else None
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.corpus = Corpus(self.dir / "corpus")
        if not resume:
            # Keep downloaded files on disk, but do not let an old topic's manifest leak
            # into a fresh run in the same workspace.
            self.corpus.papers.clear()
        self.state = (self._load() if resume else None) or ResearchState(topic=topic)
        self.map = self._load_map() if resume else {
            "topic": self.state.topic,
            "searches": [],
            "graph_rounds": [],
        }
        self._last_novelty_report: dict = {}
        self._thought_hash = self._load_thought_hash()
        self._model_call_hash = self._load_last_chain_hash(
            self.dir / "model-calls.jsonl")
        from spiral.epistemic import ObligationGraph
        from spiral.research_history import ResearchGit
        from spiral.research_strategy import (
            CounterfactualLab, InformationGainScheduler, LocalTasteModel,
        )
        from spiral.toolsmith import Toolsmith

        self.obligations = ObligationGraph(self.dir, self.state.topic)
        self.scheduler = InformationGainScheduler(self.dir, self.state.topic)
        self.taste = LocalTasteModel(self.dir, self.state.topic)
        self.counterfactual_lab = CounterfactualLab(self.dir)
        self.research_git = ResearchGit(
            self.dir, enabled=bool(getattr(self.cfg, "research_git", True)))
        self.toolsmith = Toolsmith(self.dir)
        from spiral.research_data import ScientificDataBroker

        self.data_broker = ScientificDataBroker(self.dir / "data", cfg=self.cfg)
        self._bootstrap_obligations()

    # -- persistence ---------------------------------------------------------
    def _statefile(self) -> Path:
        return self.dir / "state.json"

    def _load(self):
        f = self._statefile()
        if f.is_file():
            d = json.loads(f.read_text())
            st = ResearchState(**{k: d[k] for k in ("topic",) if k in d})
            for k, v in d.items():
                setattr(st, k, v)
            # Older research runs predate several structured state fields and may
            # persist them as JSON null.  Resume must migrate those files instead of
            # failing on the first ``.get``/iteration in the new control loop.
            for name in ("findings", "corpus_ids", "history"):
                if not isinstance(getattr(st, name, None), list):
                    setattr(st, name, [])
            for name in (
                    "active_proposal", "coverage", "completion", "novelty_boundary",
                    "obligation_report", "living_paper", "data_resources"):
                if not isinstance(getattr(st, name, None), dict):
                    setattr(st, name, {})
            for name in ("tokens", "local_tokens", "api_tokens", "round"):
                value = getattr(st, name, 0)
                try:
                    setattr(st, name, int(value or 0))
                except (TypeError, ValueError):
                    setattr(st, name, 0)
            return st
        return None

    def _save(self):
        try:
            self.obligations.save()
            self.state.obligation_report = self.obligations.report("result")
        except Exception:
            pass
        _atomic_text(
            self._statefile(), json.dumps(asdict(self.state), indent=2))

    # -- shared epistemic kernel --------------------------------------------
    def _bootstrap_obligations(self) -> None:
        """Migrate resumable pre-kernel runs into the obligation graph."""

        proposal = self.state.active_proposal if isinstance(self.state.active_proposal, dict) else {}
        if proposal.get("question") and proposal.get("claims"):
            self._register_proposal_obligations(proposal, migrate=True)
        for raw in self.state.findings:
            if not isinstance(raw, dict):
                continue
            finding = Finding(
                claim=raw.get("claim") or {}, ok=bool(raw.get("ok")),
                backend=str(raw.get("backend") or ""), detail=str(raw.get("detail") or ""),
                round=int(raw.get("round") or 0), question=str(raw.get("question") or ""),
                claim_id=str(raw.get("claim_id") or self._claim_id(raw.get("claim") or {})),
                strength=str(raw.get("strength") or "unverified"),
                required=bool(raw.get("required", True)),
                replication=raw.get("replication") if isinstance(raw.get("replication"), dict) else {},
                obligation_id=str(raw.get("obligation_id") or ""),
            )
            self._sync_finding_obligation(finding, migrate=True)
        self.obligations.save()

    def _checkpoint(self, label: str, *, phase: str, **metadata) -> str:
        try:
            result = self.research_git.checkpoint(label, phase=phase, metadata=metadata)
            commit = str(result.get("commit") or "")
            if commit:
                self.state.research_commit = commit
            return commit
        except Exception:
            return ""

    def _register_proposal_obligations(self, proposal: dict, *, migrate: bool = False) -> dict:
        if not proposal.get("question") or proposal.get("_no_proposal"):
            return {}
        question = str(proposal.get("question") or "").strip()
        old = self.state.active_proposal if isinstance(self.state.active_proposal, dict) else {}
        old_ids = old.get("_obligations") if isinstance(old.get("_obligations"), dict) else {}
        old_question_id = str(old_ids.get("question") or "")
        qid = self.obligations.ensure(
            "question", question,
            node_id=f"question:{hashlib.sha256(self._question_key(question).encode()).hexdigest()[:20]}",
            stage="result", required=True, status="in_progress",
            scope=str(proposal.get("scope") or proposal.get("conventions") or ""),
            metadata={"round": self.state.round, "mode": self._task_mode()},
        )
        if old_question_id and old_question_id != qid:
            self.obligations.set_status(
                old_question_id, "superseded", reason="a later vetted proposal replaced this question")
            for source, edge in self.obligations.incoming(old_question_id):
                if edge.get("relation") in {"answers", "scopes"} and source.get("required"):
                    self.obligations.set_status(
                        source["id"], "superseded",
                        reason="parent research question was superseded")
            self.obligations.link(qid, old_question_id, "supersedes")
        self.obligations.link(qid, self.obligations.objective_id, "answers")
        selected_candidate = str((proposal.get("_angle") or {}).get("_obligation_id") or "")
        if selected_candidate and self.obligations.node(selected_candidate):
            self.obligations.set_status(
                selected_candidate, "declared", reason="selected after angle and prior-art audits")
            self.obligations.link(qid, selected_candidate, "derived_from")

        novelty_id = ""
        if self._task_mode() != "expository":
            novelty_id = self.obligations.ensure(
                "novelty",
                f"Bound the prior-art status of: {question}",
                node_id=f"novelty:{hashlib.sha256(self._question_key(question).encode()).hexdigest()[:20]}",
                stage="result", required=True, status="open",
                scope="bounded documented literature search; never global proof of absence",
                verifier="novelty boundary certificate",
            )
            self.obligations.link(novelty_id, qid, "scopes")

        claim_ids = []
        for claim in proposal.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            cid = self._claim_id(claim)
            oid = f"claim:{cid}"
            statement = str(claim.get("statement") or claim.get("note") or cid)
            assumptions = [str(x) for x in (claim.get("assumptions") or []) if str(x).strip()]
            required = bool(claim.get("required", True))
            self.obligations.ensure(
                "claim", statement, node_id=oid, stage="result", required=required,
                status="open", assumptions=assumptions,
                verifier=str(claim.get("kind") or "deterministic verifier"),
                falsifier=str(claim.get("falsifier") or "nonzero residual or counterexample"),
                metadata={
                    "claim_id": cid,
                    "kind": claim.get("kind"),
                    "round": self.state.round,
                    "requires_replication": bool(
                        required and self._task_mode() != "expository"
                        and getattr(self.cfg, "research_blind_replication", True)),
                },
            )
            self.obligations.link(oid, qid, "answers")
            for assumption in assumptions:
                aid = self.obligations.ensure(
                    "assumption", assumption, stage="result", required=required,
                    status="declared", scope=statement,
                )
                self.obligations.link(oid, aid, "depends_on")
            falsifier = str(claim.get("falsifier") or "").strip()
            if falsifier:
                fid = self.obligations.ensure(
                    "falsifier", falsifier, stage="result", required=False,
                    status="declared", scope=statement,
                )
                self.obligations.link(fid, oid, "tests")
            claim["_obligation_id"] = oid
            claim_ids.append(oid)
        for stale_id in set(old_ids.get("claims") or []) - set(claim_ids):
            self.obligations.set_status(
                str(stale_id), "superseded", reason="proposal refinement replaced this claim")
        proposal["_obligations"] = {
            "question": qid, "novelty": novelty_id, "claims": claim_ids,
        }
        if not migrate:
            self._log_thought(
                "obligation-register",
                f"registered {len(claim_ids)} claim obligations for {question}",
                obligations=proposal["_obligations"],
            )
        self.obligations.save()
        return proposal["_obligations"]

    def _sync_finding_obligation(self, finding: Finding, *, migrate: bool = False) -> None:
        oid = finding.obligation_id or str(finding.claim.get("_obligation_id") or "")
        oid = oid or f"claim:{finding.claim_id or self._claim_id(finding.claim)}"
        finding.obligation_id = oid
        self.obligations.ensure(
            "claim", str(finding.claim.get("statement") or finding.claim.get("note") or oid),
            node_id=oid, stage="result", required=finding.required,
            metadata={
                "claim_id": finding.claim_id,
                "kind": finding.claim.get("kind"),
                "requires_replication": bool(
                    finding.required and self._task_mode() != "expository"
                    and getattr(self.cfg, "research_blind_replication", True)),
            },
        )
        relation = "supports" if finding.ok else "refutes"
        status = "supported" if finding.ok else "refuted"
        manifest = str(finding.claim.get("manifest") or "")
        evidence_id = self.obligations.add_evidence(
            oid,
            f"{finding.backend} verifier: {finding.detail[:1000]}",
            evidence_kind="verification", artifact=manifest,
            verifier=finding.backend, relation=relation, status="supported",
            metadata={
                "strength": finding.strength, "round": finding.round,
                "claim_id": finding.claim_id,
            },
            node_id=f"verification:{finding.claim_id}:{finding.round}",
        )
        self.obligations.set_status(
            oid, status, reason=finding.detail[:1000], verifier=finding.backend)
        if finding.replication.get("passed"):
            path = str(finding.replication.get("path") or "")
            rid = self.obligations.add_evidence(
                oid,
                f"Blind independent replication of {finding.claim_id}",
                evidence_kind="replication", artifact=path,
                verifier=str(finding.replication.get("backend") or "independent verifier"),
                independent=True, relation="replicates", status="supported",
                metadata=finding.replication,
                node_id=f"replication:{finding.claim_id}",
            )
            self.obligations.link(rid, evidence_id, "derived_from", metadata={"solution_hidden": True})
        if not migrate:
            self.obligations.save()

    def _sync_decision_obligations(self, proposal: dict, decision: dict) -> None:
        ids = proposal.get("_obligations") if isinstance(proposal.get("_obligations"), dict) else {}
        qid = str(ids.get("question") or "")
        if not qid:
            return
        if decision.get("action") in {"solved", "new_question"}:
            did = self.obligations.ensure(
                "decision",
                str(decision.get("reason") or decision.get("assessment") or "supervisor accepted result"),
                node_id=f"decision:{self.state.round}:{hashlib.sha256(qid.encode()).hexdigest()[:10]}",
                stage="result", required=False, status="supported",
                verifier="supervisor plus deterministic completion gate",
                metadata={"action": decision.get("action"), "round": self.state.round},
            )
            self.obligations.link(did, qid, "supports")
            self.obligations.set_status(
                qid, "supported", reason=str(decision.get("reason") or "answer accepted"),
                verifier="supervisor scope decision")
        elif decision.get("action") == "pivot":
            self.obligations.set_status(qid, "superseded", reason="supervisor requested a pivot")

    def _living_status(self) -> dict:
        from spiral.research_provenance import LivingPaper

        manifest = self.dir / "living-paper.json"
        status = LivingPaper.inspect(manifest, self.dir)
        if self.refresh_requested and manifest.is_file():
            status = {**status, "current": False, "stale": True,
                      "issues": ["explicit refresh requested", *status.get("issues", [])]}
        self.state.living_paper = status
        return status

    def _say(self, msg: str):
        if self.ui:
            self.ui(msg)

    def _load_thought_hash(self) -> str:
        return self._load_last_chain_hash(self.dir / "thoughts.jsonl")

    @staticmethod
    def _load_last_chain_hash(path: Path) -> str:
        if not path.is_file():
            return ""
        try:
            last = path.read_text(encoding="utf-8").splitlines()[-1]
            return str(json.loads(last).get("entry_hash") or "")
        except Exception:
            return ""

    def _log_thought(self, phase: str, text: str, **extra) -> None:
        """Persist explicit research deliberation artifacts.

        This is not hidden chain-of-thought. It is the visible audit trail: reading
        notes, candidate angles, rejection reasons, next searches, and supervisor
        decisions that make a long autonomous run followable.
        """
        text = " ".join(str(text or "").split())
        if not text:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": self.state.round,
            "phase": phase,
            "text": text[:4000],
            "extra": extra,
            "prev_hash": self._thought_hash,
        }
        try:
            canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
            entry["entry_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            with (self.dir / "thoughts.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._thought_hash = entry["entry_hash"]
        except Exception:
            pass

    def _audit_model_call(self, *, model: str, role: str, variant: str,
                          system: str, user: str, result=None, error: str = "",
                          reasoning_requested: bool = False) -> None:
        """Persist replayable prompts and public output, never private reasoning tokens."""

        raw = getattr(result, "raw", {}) or {}
        private_reasoning = str(getattr(result, "thinking", None) or "")
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": self.state.round,
            "model": model,
            "role": role,
            "variant": variant,
            "system_sha256": hashlib.sha256(system.encode("utf-8", "ignore")).hexdigest(),
            "user_sha256": hashlib.sha256(user.encode("utf-8", "ignore")).hexdigest(),
            "system_chars": len(system),
            "user_chars": len(user),
            "system_prompt": system,
            "user_prompt": user,
            "output": (getattr(result, "text", "") or ""),
            "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(result, "completion_tokens", 0) or 0),
            "finish_reason": raw.get("finish_reason") if isinstance(raw, dict) else None,
            "error": error,
            "reasoning_requested": bool(reasoning_requested),
            "private_reasoning_returned": bool(private_reasoning),
            "private_reasoning_chars": len(private_reasoning),
            "private_reasoning_sha256": (
                hashlib.sha256(private_reasoning.encode("utf-8", "ignore")).hexdigest()
                if private_reasoning else ""),
            "private_reasoning_recorded": False,
            "prev_hash": self._model_call_hash,
        }
        try:
            canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
            entry["entry_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            with (self.dir / "model-calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._model_call_hash = entry["entry_hash"]
        except Exception:
            pass

    # -- research map --------------------------------------------------------
    def _mapfile(self) -> Path:
        return self.dir / "research-map.json"

    def _load_map(self) -> dict:
        f = self._mapfile()
        if f.is_file():
            try:
                data = json.loads(f.read_text())
            except Exception:
                data = {}
        else:
            data = {}
        data.setdefault("topic", self.state.topic)
        if not isinstance(data.get("searches"), list):
            data["searches"] = []
        if not isinstance(data.get("graph_rounds"), list):
            data["graph_rounds"] = []
        return data

    def _map_markdown(self) -> str:
        lines = ["# Research map", "", f"Topic: {self.state.topic}", ""]
        if self.state.question:
            lines += [f"Working question: {self.state.question}", ""]
        lines += ["## Searches"]
        if not self.map.get("searches"):
            lines.append("(none recorded)")
        for s in self.map.get("searches", []):
            cats = ", ".join(s.get("categories") or []) or "unrestricted"
            added = ", ".join(s.get("added") or []) or "none"
            retrieval = s.get("retrieval") or {}
            health = "ok" if retrieval.get("source_ok") is True else (
                "failed" if retrieval.get("source_ok") is False else "unknown")
            lines.append(
                f"- round {s.get('round', 0)} | {cats} | `{s.get('query', '')}` | "
                f"source: {health} | results: {retrieval.get('result_count', '?')} | added: {added}")
        lines += ["", "## Citation Graph"]
        if not self.map.get("graph_rounds"):
            lines.append("(none recorded)")
        for g in self.map.get("graph_rounds", []):
            health = g.get("health") or {}
            lines.append(
                f"- round {g.get('research_round', 0)} | seeds: {len(g.get('seeds') or [])} | "
                f"edges: {g.get('edge_count', 0)} | added: {', '.join(g.get('added') or []) or 'none'}"
                + f" | source: {'ok' if health.get('coverage_valid') else 'invalid'}"
                + (" | saturated" if g.get("saturated") else "")
            )
            holes = g.get("holes") or []
            if holes:
                top = "; ".join(f"{h.get('id')} x{h.get('count')}" for h in holes[:8])
                lines.append(f"  holes: {top}")
        lines += ["", "## Coverage Gates"]
        if not self.map.get("coverage_reports"):
            lines.append("(none recorded)")
        for c in self.map.get("coverage_reports", []):
            lines.append(
                f"- round {c.get('round', 0)} | discovery: "
                f"{'ready' if c.get('discovery_ready') else 'blocked'} | novelty: "
                f"{'ready' if c.get('novelty_ready') else 'blocked'} | "
                f"papers: {c.get('paper_count', 0)} | relevant: {c.get('relevant_paper_count', 0)}"
            )
        lines += ["", "## Prior-art Protocols"]
        if not self.map.get("prior_art_searches"):
            lines.append("(none recorded)")
        for p in self.map.get("prior_art_searches", []):
            lines.append(
                f"- round {p.get('round', 0)} | {'ready' if p.get('ready') else 'invalid'} | "
                f"healthy queries: {p.get('healthy_queries', 0)} | independent families: "
                f"{p.get('healthy_query_families', 0)} | sources: "
                f"{', '.join(p.get('sources_ok') or []) or 'none'} | results: {p.get('result_count', 0)}"
            )
        lines += ["", "## Scientific Data"]
        data_resources = (
            self.state.data_resources
            if isinstance(self.state.data_resources, dict) else {})
        data_records = data_resources.get("records") or []
        if not data_records:
            lines.append("(no typed data-catalog search recorded)")
        for record in data_records[:30]:
            lines.append(
                f"- {record.get('source')}:{record.get('dataset_id')} | "
                f"{record.get('title', '')} | version: {record.get('version') or 'unknown'} | "
                f"licence: {record.get('license') or 'unresolved'}"
            )
        epistemic = self.map.get("epistemic") if isinstance(self.map.get("epistemic"), dict) else {}
        result_gate = epistemic.get("result_report") or {}
        publication_gate = epistemic.get("publication_report") or {}
        lines += ["", "## Epistemic obligations"]
        if not epistemic:
            lines.append("(none recorded)")
        else:
            lines.append(
                f"- result gate: {'ready' if result_gate.get('ready') else 'open'} | "
                f"required: {result_gate.get('required_count', 0)} | "
                f"blockers: {len(result_gate.get('blockers') or [])}")
            lines.append(
                f"- publication gate: {'ready' if publication_gate.get('ready') else 'open'} | "
                f"required: {publication_gate.get('required_count', 0)} | "
                f"blockers: {len(publication_gate.get('blockers') or [])}")
            lines.append(f"- graph digest: `{epistemic.get('digest', '')}`")
        return "\n".join(lines) + "\n"

    def _save_map(self):
        self.map["topic"] = self.state.topic
        self.map["question"] = self.state.question
        try:
            self.obligations.save()
            self.map["epistemic"] = self.obligations.compact()
        except Exception:
            pass
        _atomic_text(
            self._mapfile(), json.dumps(self.map, indent=2))
        _atomic_text(self.dir / "research-map.md", self._map_markdown())
        try:
            from spiral.research_graph import write_graph_view
            write_graph_view(self.map, self.corpus, self.dir)
        except Exception:
            # The JSON/Markdown map is the canonical audit log. The browser view is
            # convenience; a rendering hiccup must never interrupt a long research run.
            pass

    def _record_search(self, query: str, categories, added: list[str], k: int,
                       retrieval: dict | None = None):
        retrieval = retrieval or {}
        self.map.setdefault("searches", []).append({
            "round": self.state.round,
            "query": query,
            "categories": list(categories or []),
            "k": k,
            "added": added,
            "corpus_size": len(self.corpus.papers),
            "retrieval": retrieval,
        })
        try:
            self.scheduler.observe_search(
                query, added=len(added), k=k, retrieval=retrieval)
            sid = self.obligations.ensure(
                "search", f"Literature search: {query}",
                node_id=f"search:{self.state.round}:{hashlib.sha256(query.encode()).hexdigest()[:12]}",
                stage="discovery", required=False,
                status="supported" if retrieval.get("source_ok") is True else "blocked",
                verifier="retrieval source telemetry",
                metadata={
                    "query": query, "categories": list(categories or []),
                    "added": added, "retrieval": retrieval,
                },
            )
            self.obligations.link(sid, self.obligations.objective_id, "derived_from")
        except Exception:
            pass
        self._save_map()

    def _record_graph(self, report: dict):
        for g in report.get("round_reports", []):
            self.map.setdefault("graph_rounds", []).append({
                "research_round": self.state.round,
                **g,
                "corpus_size": len(self.corpus.papers),
            })
        self._save_map()

    # -- retrieval liveness -------------------------------------------------
    def _tried_queries(self) -> list[str]:
        return [str(s.get("query") or "")
                for s in (self.map.get("searches") or []) if s.get("query")]

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        # Hyphens split: 'higher-dimensional' and 'higher dimensional' are the same
        # search family, and hyphen variants are exactly how paraphrases sneak past.
        return set(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}",
                              str(query).lower().replace("-", " ")))

    def _query_is_novel(self, query: str, *, threshold: float = 0.75) -> bool:
        """True when ``query`` is not a token-level paraphrase of a tried search.

        A stalled run once re-issued 'kodama ishibashi master equations higher
        dimensional black holes' in six near-identical wordings, round after round —
        each a fresh arXiv request, none new information. Re-searching a family that
        already answered (or already failed) is not persistence, it is the loop
        confusing motion with progress."""
        terms = self._query_terms(query)
        if not terms:
            return False
        for tried in self._tried_queries():
            held = self._query_terms(tried)
            if held and len(terms & held) / len(terms | held) >= threshold:
                return False
        return True

    def _instrument_health(self, *, window: int = 8) -> dict:
        """Observable health of the retrieval instruments over the recent record —
        the evidence a stall decision is made from, never a model's impression."""
        searches = (self.map.get("searches") or [])[-window:]
        failures = results = added = 0
        for s in searches:
            retrieval = s.get("retrieval") or {}
            if retrieval.get("source_ok") is False:
                failures += 1
            results += int(retrieval.get("result_count") or 0)
            added += len(s.get("added") or [])
        graph_rounds = (self.map.get("graph_rounds") or [])[-2:]
        graph_added = sum(len(g.get("added") or []) for g in graph_rounds)
        graph_ok = any(
            int((g.get("health") or {}).get("successful_requests") or 0) > 0
            for g in graph_rounds)
        dead = bool(searches) and results == 0 and added == 0 and graph_added == 0
        return {
            "recent_searches": len(searches),
            "recent_search_failures": failures,
            "recent_search_results": results,
            "recent_search_added": added,
            "recent_graph_added": graph_added,
            "recent_graph_ok": graph_ok,
            "instruments_dead": dead,
        }

    @staticmethod
    def _data_driven_topic(topic: str) -> bool:
        terms = {
            "allen", "atlas", "bids", "connectome", "data", "dataset", "eeg",
            "expression", "fmri", "imaging", "microscopy", "mri", "neuroimaging",
            "nifti", "nwb", "openneuro", "pet", "receptor", "transcriptomic",
        }
        words = set(re.findall(r"[a-z][a-z0-9-]+", str(topic).lower()))
        return bool(words & terms)

    def _data_catalog_sketch(self, *, chars: int = 7000) -> str:
        report = self.state.data_resources
        if not isinstance(report, dict) or not report.get("records"):
            return "(no typed scientific-data resources discovered)"
        rows = []
        for record in (report.get("records") or [])[:18]:
            rows.append({
                key: record.get(key) for key in (
                    "source", "dataset_id", "title", "description", "version",
                    "doi", "license", "species", "modalities", "url", "metadata",
                )
            })
        return json.dumps(rows, ensure_ascii=False, indent=2)[:chars]

    def discover_data_resources(self, *, force: bool = False) -> dict:
        """Search typed public catalogs when the topic calls for empirical data."""

        if not self._data_driven_topic(self.state.topic):
            return {}
        current = self.state.data_resources
        healthy = (
            isinstance(current, dict)
            and bool(current.get("records"))
            and bool(current.get("healthy_sources"))
        )
        if healthy and not force:
            return current
        self._say("data · metadata-first catalog search")
        try:
            report = self.data_broker.discover(
                self.state.topic,
                sources=list(getattr(
                    self.cfg, "research_data_sources",
                    ["openneuro", "allen", "neuromaps", "zenodo"])),
                limit=max(3, int(getattr(
                    self.cfg, "research_data_catalog_limit", 18))),
            )
        except Exception as exc:
            report = {
                "query": self.state.topic, "records": [], "healthy_sources": [],
                "errors": {"broker": f"{type(exc).__name__}: {exc}"},
            }
        self.state.data_resources = report
        self.map["data_catalog"] = report
        self._log_thought(
            "data-catalog",
            f"{len(report.get('records') or [])} candidate public resources; "
            f"healthy sources: {', '.join(report.get('healthy_sources') or []) or 'none'}",
            report=report,
        )
        self._save_map()
        return report

    def _graph_seed_batch(self, limit: int = 30) -> list[str]:
        """Choose the next deterministic citation batch not yet closed.

        Semantic Scholar calls are deliberately bounded, so a large corpus must be
        covered across research rounds. A seed counts as closed only after a healthy,
        untruncated batch reported no unresolved co-citation holes.
        """
        from spiral.research_quality import rank_papers_for_topic

        ordered = [
            str(getattr(p, "bare_id", getattr(p, "arxiv_id", "")))
            for p in rank_papers_for_topic(
                self.state.topic, self.corpus.papers.values())
        ]
        closed: set[str] = set()
        for report in self.map.get("graph_rounds") or []:
            health = report.get("health") or report.get("graph_health") or {}
            if not (
                health.get("coverage_valid") is True
                and (report.get("batch_frontier_closed") is True
                     or report.get("saturated") is True)
                and not report.get("frontier_truncated")
                and not (report.get("unresolved_holes_after_round") or [])
            ):
                continue
            closed.update(
                str(seed).replace("arXiv:", "").split("v")[0]
                for seed in (health.get("successful_seeds") or [])
            )
        pending = [seed for seed in ordered if seed and seed not in closed]
        return (pending or ordered)[:max(1, limit)]

    # -- llm -----------------------------------------------------------------
    def _model_error(self, res) -> str:
        raw = getattr(res, "raw", {}) or {}
        err = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or err)[:240]
        if err:
            return str(err)[:240]
        status = raw.get("status") if isinstance(raw, dict) else None
        finish = raw.get("finish_reason") if isinstance(raw, dict) else None
        if status:
            return f"status {status}"
        if finish:
            return f"finish_reason {finish}"
        return "empty response"

    def _compact_user_prompt(self, user: str, *, max_chars: int = 18_000) -> str:
        """Shrink a model prompt after a blank/error response.

        Empty local completions often happen when a large TeX-heavy corpus digest nudges
        the model over its comfortable context. Preserve the task header and a sampled
        corpus sketch so the retry remains grounded but fits easily.
        """
        user = user or ""
        if len(user) <= max_chars:
            return user
        marker = "\n\nCORPUS:"
        if marker in user:
            head, corpus = user.split(marker, 1)
            chunks = [c.strip() for c in re.split(r"\n(?=\[\d+\])", corpus) if c.strip()]
            budget = max(4_000, max_chars - len(head) - 500)
            take = chunks[:18]
            per = max(300, budget // max(1, len(take)))
            body = "\n\n".join(c[:per].rstrip() for c in take)
            return (
                f"{head}{marker}\n"
                f"[compacted retry after empty model response; {len(take)}/{len(chunks)} corpus entries]\n\n"
                f"{body}"
            )[:max_chars]
        half = max_chars // 2
        return user[:half].rstrip() + "\n\n[... compacted retry ...]\n\n" + user[-half:].lstrip()

    def _think(self, system: str, user: str, think: bool | None = None, *, role: str = "planner",
               fmt=None, max_chars: int = 18_000, temperature: float = 0.4,
               num_predict: int | None = None,
               context_limit: int | None = None) -> tuple[str, int]:
        spec = getattr(self.cfg, role, self.cfg.planner)
        think = spec.think if think is None else think
        selected_ctx = min(spec.num_ctx, context_limit) if context_limit else spec.num_ctx
        attempts = [(spec.name, selected_ctx, role)]
        if spec.name in getattr(self.ol, "providers", {}):
            try:
                from spiral.config import Config
                fallback = Config().planner
                if fallback.name != spec.name:
                    fallback_ctx = min(fallback.num_ctx, context_limit) if context_limit else fallback.num_ctx
                    attempts.append((fallback.name, fallback_ctx, "local fallback"))
            except Exception:
                pass

        last = ""
        for model, num_ctx, label in attempts:
            variants = [("full", system, user)]
            compact = self._compact_user_prompt(user, max_chars=max_chars)
            if compact != user:
                variants.append(("compact", system + "\nIf the corpus is compacted, answer from the visible evidence only.", compact))
            else:
                variants.append(("retry", system + "\nYour previous response was empty. Return a concise valid answer now.", user))
            for variant, sys_msg, user_msg in variants:
                try:
                    res = self.ol.chat(
                        model,
                        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                        think=think, num_predict=num_predict or self.cfg.planner_max_tokens,
                        num_ctx=num_ctx, keep_alive=self.cfg.keep_alive, temperature=temperature,
                        fmt=fmt,
                    )
                except Exception as exc:
                    last = f"{label}:{model} failed ({type(exc).__name__}: {exc})"
                    self._audit_model_call(
                        model=model, role=role, variant=variant,
                        system=sys_msg, user=user_msg, error=last,
                        reasoning_requested=bool(think))
                    self._say(f"  model · {last}")
                    continue
                prompt_tokens = int(getattr(res, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(res, "completion_tokens", 0) or 0)
                used = prompt_tokens + completion_tokens
                self.state.tokens += used
                if model in getattr(self.ol, "providers", {}):
                    self.state.api_tokens += used
                else:
                    self.state.local_tokens += used
                self._audit_model_call(
                    model=model, role=role, variant=variant,
                    system=sys_msg, user=user_msg, result=res,
                    reasoning_requested=bool(think))
                text = (res.text or "").strip()
                if text:
                    if label == "local fallback":
                        self._say(f"  model · fell back to {model}")
                    elif variant != "full":
                        self._say(f"  model · recovered with {variant} prompt")
                    return text, getattr(res, "completion_tokens", 0) or 0
                last = f"{label}:{model} returned no text ({self._model_error(res)})"
                self._say(f"  model · {last}")
        raise ResearchModelError(last or "reasoning backend returned no text")

    def _think_json(self, system: str, user: str, *, role: str = "critic",
                    max_chars: int = 12_000, required: tuple[str, ...] = (),
                    max_tokens: int | None = None,
                    context_limit: int | None = None,
                    reasoning: bool = False) -> dict:
        """Structured research calls must either yield JSON or fail loudly in the log.

        Routine extraction/classification stays concise. Consequential scientific calls
        opt into ``reasoning=True`` and receive a larger local generation allowance; Kimi's
        provider adapter independently reserves enough completion room for its mandatory
        reasoning. Either lane still returns only the public JSON decision to the caller.
        """
        sys = (
            system
            + "\nReturn exactly one compact JSON object. No Markdown, no prose outside JSON, "
              "no derivation transcript."
        )
        predict = max_tokens or max(2048, min(self.cfg.planner_max_tokens, 8192))
        if reasoning:
            predict = max(predict, min(self.cfg.planner_max_tokens, 8192))
        try:
            text, _ = self._think(
                sys, user, think=reasoning, role=role, fmt="json", max_chars=max_chars,
                temperature=0.1,
                num_predict=predict,
                context_limit=context_limit,
            )
        except ResearchModelError as exc:
            self._say(f"  model · structured JSON failed ({str(exc)[:140]})")
            return {}
        data = _extract_json(text)
        if not isinstance(data, dict):
            self._say("  model · structured JSON malformed · compact retry")
            retry_user = self._compact_user_prompt(user, max_chars=max_chars)
            try:
                retry_text, _ = self._think(
                    sys + "\nYour previous object was malformed or truncated. Keep every string "
                    "concise and emit the complete JSON object now.",
                    retry_user, think=reasoning, role=role, fmt="json", max_chars=max_chars,
                    temperature=0.0,
                    num_predict=predict,
                    context_limit=context_limit,
                )
            except ResearchModelError:
                return {}
            data = _extract_json(retry_text)
            if not isinstance(data, dict):
                return {}
        if required and not all(str(data.get(k, "")).strip() for k in required):
            return {}
        return data

    def _local_vision_json(
        self, system: str, user: str, image_paths: list[str | Path],
        *, max_tokens: int = 3072,
    ) -> dict:
        """Run an auditable local-only vision review; never route artifacts to an API."""

        from spiral.visual_review import choose_vision_model

        model = choose_vision_model(self.cfg, self.ol)
        if not model or model in getattr(self.ol, "providers", {}):
            return {}
        paths = [Path(path) for path in image_paths if Path(path).is_file()][:8]
        if not paths:
            return {}
        images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in paths]
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user + "\nFILES: " + ", ".join(path.name for path in paths),
                "images": images,
            },
        ]
        for variant, reasoning in (("full", True), ("concise-retry", False)):
            try:
                res = self.ol.chat(
                    model, messages, think=reasoning, num_predict=max_tokens,
                    num_ctx=self.cfg.spec_for(model).num_ctx,
                    keep_alive=self.cfg.keep_alive, temperature=0.1, fmt="json",
                )
            except Exception as exc:
                self._audit_model_call(
                    model=model, role="local_vision", variant=variant,
                    system=system, user=user,
                    error=f"{type(exc).__name__}: {exc}",
                    reasoning_requested=reasoning,
                )
                continue
            used = int(getattr(res, "prompt_tokens", 0) or 0) + int(
                getattr(res, "completion_tokens", 0) or 0)
            self.state.tokens += used
            self.state.local_tokens += used
            self._audit_model_call(
                model=model, role="local_vision", variant=variant,
                system=system, user=user, result=res,
                reasoning_requested=reasoning,
            )
            data = _extract_json((res.text or "").strip())
            if data:
                data["_model"] = model
                data["_scope"] = "local vision advisory; no image left the device"
                return data
        return {}

    # -- reading notes / context management ---------------------------------
    @staticmethod
    def _safe_id(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "paper")).strip("_") or "paper"

    def _notes_root(self) -> Path:
        d = self.dir / "notes"
        (d / "papers").mkdir(parents=True, exist_ok=True)
        (d / "deep").mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _paper_source_hash(p) -> str:
        # Grounded notes authenticate the complete held source. A prefix hash lets a
        # changed appendix/late derivation survive cache validation, exactly where a
        # long mathematical paper may place conventions or proofs.
        sample = "\n".join([
            getattr(p, "title", ""),
            getattr(p, "abstract", ""),
            getattr(p, "text", "") or "",
        ])
        return hashlib.sha256(sample.encode("utf-8", "ignore")).hexdigest()

    @staticmethod
    def _paper_excerpt(p, *, max_chars: int = 12_000) -> str:
        text = getattr(p, "text", "") or getattr(p, "abstract", "") or ""
        title = getattr(p, "title", "") or getattr(p, "arxiv_id", "")
        abstract = getattr(p, "abstract", "") or ""
        parts = [f"TITLE: {title}", f"ABSTRACT:\n{abstract[:1800]}", "OPENING:\n" + text[:3600]]
        eqs = re.findall(
            r"(\\begin\{(?:equation|align|gather|multline)[^}]*\}.*?\\end\{(?:equation|align|gather|multline)[^}]*\}|\\\[.*?\\\])",
            text, flags=re.S,
        )
        if eqs:
            parts.append("REPRESENTATIVE EQUATIONS:\n" + "\n\n".join(e[:900] for e in eqs[:8]))
        sec = re.search(r"\\section\*?\{(?:Conclusion|Conclusions|Discussion|Summary)[^}]*\}(.*)", text, re.I | re.S)
        if sec:
            parts.append("ENDING/DISCUSSION:\n" + sec.group(1)[:3000])
        elif len(text) > 5000:
            parts.append("ENDING:\n" + text[-2600:])
        return "\n\n".join(parts)[:max_chars]

    def _notes_model_spec(self) -> tuple[str, int]:
        """Use a cheap/local model for broad reading even under --api unless configured."""
        configured = str(getattr(self.cfg, "research_notes_model", "") or "").strip()
        if configured:
            spec = self.cfg.spec_for(configured)
            return configured, spec.num_ctx
        if self.cfg.worker.name not in getattr(self.ol, "providers", {}):
            return self.cfg.worker.name, self.cfg.worker.num_ctx
        try:
            from spiral.config import Config

            local = Config().worker
            return local.name, local.num_ctx
        except Exception:
            return self.cfg.worker.name, self.cfg.worker.num_ctx

    def _think_json_model(self, model: str, num_ctx: int, system: str, user: str,
                          *, max_tokens: int = 2048, max_chars: int = 12_000) -> dict:
        try:
            res = self.ol.chat(
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": user[:max_chars]}],
                think=False, num_predict=max_tokens, num_ctx=num_ctx,
                keep_alive=self.cfg.keep_alive, temperature=0.1, fmt="json",
            )
        except Exception as exc:
            self._audit_model_call(
                model=model, role="notes", variant="json",
                system=system, user=user[:max_chars],
                error=f"{type(exc).__name__}: {exc}")
            self._say(f"  model · notes:{model} failed ({type(exc).__name__}: {str(exc)[:90]})")
            return {}
        prompt_tokens = int(getattr(res, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(res, "completion_tokens", 0) or 0)
        used = prompt_tokens + completion_tokens
        self.state.tokens += used
        if model in getattr(self.ol, "providers", {}):
            self.state.api_tokens += used
        else:
            self.state.local_tokens += used
        self._audit_model_call(
            model=model, role="notes", variant="json",
            system=system, user=user[:max_chars], result=res)
        data = _extract_json((res.text or "").strip())
        if not data:
            self._say(f"  model · notes:{model} returned no parseable JSON ({self._model_error(res)})")
        return data

    def _fallback_paper_note(self, p, source_hash: str) -> dict:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", (getattr(p, "abstract", "") or getattr(p, "text", ""))[:3000])
        keywords = list(dict.fromkeys(w.lower() for w in words if len(w) > 4))[:18]
        return {
            "arxiv_id": getattr(p, "arxiv_id", ""),
            "title": getattr(p, "title", "") or getattr(p, "arxiv_id", ""),
            "source_hash": source_hash,
            "role_in_corpus": "fallback metadata/abstract note; model summary unavailable",
            "main_results": [getattr(p, "abstract", "")[:500]] if getattr(p, "abstract", "") else [],
            "methods": [],
            "objects_equations": [],
            "assumptions_conventions": [],
            "relation_to_topic": "",
            "gaps_or_openings": [],
            "reusable_tools": [],
            "notation": [],
            "keywords": keywords,
            "confidence": "low",
            "evidence": [],
        }

    @staticmethod
    def _normalise_anchor(text: str) -> str:
        return " ".join(str(text or "").lower().split())

    def _validate_note_evidence(self, note: dict, excerpt: str, p) -> dict:
        """Ground model-written reading notes in exact spans from the supplied source."""

        haystack = self._normalise_anchor(excerpt)
        valid = []
        rejected = 0
        for item in (note.get("evidence") or []):
            if not isinstance(item, dict):
                rejected += 1
                continue
            anchor = " ".join(str(item.get("anchor") or "").split())[:500]
            normalized = self._normalise_anchor(anchor)
            if len(normalized) >= 12 and normalized in haystack:
                valid.append({
                    "supports": " ".join(str(item.get("supports") or "").split())[:500],
                    "anchor": anchor,
                })
            else:
                rejected += 1
        primary = bool(
            getattr(p, "tex_path", "") or getattr(p, "pdf_path", "")
            or str(getattr(p, "body_source", "")) in {"tex", "pdf"}
        )
        if len(valid) >= 3 and primary:
            confidence = "high"
        elif valid:
            confidence = "medium"
        else:
            confidence = "low"
        note["evidence"] = valid
        note["rejected_evidence_count"] = rejected
        note["confidence"] = confidence
        note["grounded"] = bool(valid)
        return note

    def _paper_note(self, p) -> dict:
        root = self._notes_root()
        pid = getattr(p, "bare_id", getattr(p, "arxiv_id", "paper"))
        path = root / "papers" / f"{self._safe_id(pid)}.json"
        source_hash = self._paper_source_hash(p)
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("source_hash") == source_hash and cached.get("schema_version") == 2:
                    return cached
            except Exception:
                pass
        model, num_ctx = self._notes_model_spec()
        system = (
            "You are making a research reading note for another model. Summarize this "
            "paper as source evidence for the user's research prompt. Return ONLY JSON "
            '{"arxiv_id":"...","title":"...","role_in_corpus":"...","main_results":["..."],'
            '"methods":["..."],"objects_equations":["..."],"assumptions_conventions":["..."],'
            '"relation_to_topic":"...","gaps_or_openings":["..."],"reusable_tools":["..."],'
            '"notation":["..."],"keywords":["..."],'
            '"evidence":[{"supports":"which note claim this supports",'
            '"anchor":"exact short phrase or equation copied from EXCERPT"}]}. '
            "Be compact, technical, and explicit about what the paper does NOT cover. "
            "Every substantive summary must have a source anchor; never invent an anchor."
        )
        excerpt = self._paper_excerpt(p, max_chars=12_000)
        user = (
            f"USER TOPIC:\n{self.state.topic}\n\n"
            f"PAPER ID: {getattr(p, 'arxiv_id', '')}\n"
            f"AUTHORS: {', '.join(getattr(p, 'authors', [])[:8])}\n"
            f"CATEGORIES: {', '.join(getattr(p, 'categories', []) or [])}\n\n"
            f"EXCERPT:\n{excerpt}"
        )
        note = self._think_json_model(model, num_ctx, system, user, max_tokens=2048, max_chars=16_000)
        if not note:
            note = self._fallback_paper_note(p, source_hash)
        note = self._validate_note_evidence(note, excerpt, p)
        note.setdefault("arxiv_id", getattr(p, "arxiv_id", ""))
        note.setdefault("title", getattr(p, "title", "") or getattr(p, "arxiv_id", ""))
        note["schema_version"] = 2
        note["source_hash"] = source_hash
        note["notes_model"] = model
        path.write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
        with (root / "reading-notes.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")
        self._log_thought(
            "paper-note",
            f"{note.get('arxiv_id', pid)}: {note.get('role_in_corpus') or note.get('title')}",
            note=note,
        )
        try:
            source_id = self.obligations.ensure(
                "source",
                f"{note.get('title') or getattr(p, 'title', pid)} ({note.get('arxiv_id', pid)})",
                node_id=f"source:{self._safe_id(str(pid))}",
                stage="discovery", required=False,
                status="supported" if note.get("grounded") else "blocked",
                verifier="exact source-anchor validation",
                provenance=[str(path)],
                metadata={
                    "arxiv_id": note.get("arxiv_id", ""),
                    "confidence": note.get("confidence", "low"),
                    "source_hash": source_hash,
                    "grounded_anchor_count": len(note.get("evidence") or []),
                },
            )
            self.obligations.link(source_id, self.obligations.objective_id, "derived_from")
            self.obligations.save()
        except Exception:
            pass
        return note

    def _ensure_reading_notes(self) -> list[dict]:
        from spiral.research_quality import rank_papers_for_topic

        papers = rank_papers_for_topic(self.state.topic, self.corpus.papers.values())[
            : max(1, int(getattr(self.cfg, "research_reading_limit", 60)))]
        notes = []
        if not papers:
            return notes
        self._say(f"read · corpus notes · {len(papers)} papers")
        for p in papers:
            note = self._paper_note(p)
            notes.append(note)
            self._say(f"  read · {note.get('arxiv_id', getattr(p, 'arxiv_id', 'paper'))[:24]}")
        return notes

    @staticmethod
    def _notes_digest(notes: list[dict], *, limit: int = 60, chars: int = 18_000) -> str:
        lines = []
        for i, n in enumerate(notes[:limit], 1):
            bits = [
                f"[{i}] {n.get('title') or n.get('arxiv_id')} ({n.get('arxiv_id', '')})",
                f"role: {n.get('role_in_corpus', '')}",
                "results: " + "; ".join(str(x) for x in (n.get("main_results") or [])[:3]),
                "methods: " + "; ".join(str(x) for x in (n.get("methods") or [])[:4]),
                "objects/equations: " + "; ".join(str(x) for x in (n.get("objects_equations") or [])[:4]),
                "gaps: " + "; ".join(str(x) for x in (n.get("gaps_or_openings") or [])[:3]),
                f"grounding: {n.get('confidence','low')} ({len(n.get('evidence') or [])} exact anchors)",
            ]
            lines.append("\n".join(bits))
        return "\n\n".join(lines)[:chars]

    def _idea_families(self, notes: list[dict]) -> list[dict]:
        root = self._notes_root()
        family_schema = (
            '{"families":[{"name":"...","interest":"...","question_seeds":["..."],'
            '"key_papers":["arXiv id/title", "..."],"deep_read_papers":["arXiv id", "..."],'
            '"missing_or_risks":["..."],"prior_art_queries":["..."],'
            '"first_checks":["symbolic/numeric/certificate idea", "..."]}]}'
        )
        batch_families: list[dict] = []
        if len(notes) > 16:
            model, num_ctx = self._notes_model_spec()
            batch_system = (
                "Cluster this BATCH of source-grounded paper notes into provisional research "
                "idea families. Preserve paper ids and evidence gaps. Return ONLY JSON "
                + family_schema + ". Do not claim novelty."
            )
            for start in range(0, len(notes), 12):
                batch = notes[start:start + 12]
                data = self._think_json_model(
                    model, num_ctx, batch_system,
                    f"TOPIC:\n{self.state.topic}\n\nNOTE BATCH:\n"
                    f"{self._notes_digest(batch, limit=12, chars=10_000)}",
                    max_tokens=2200, max_chars=14_000)
                for family in (data.get("families") or []):
                    if isinstance(family, dict):
                        family["source_batch"] = start // 12 + 1
                        batch_families.append(family)
        system = (
            "You are reading a large corpus through structured notes. Find idea families "
            "inside the user's prompt: clusters where the literature suggests an interesting "
            "gap, under-classified target, reusable method, or promising exact check. Return "
            "ONLY JSON " + family_schema + ". "
            "Prefer several plausible families over one premature commitment."
        )
        if batch_families:
            user = (
                f"TOPIC:\n{self.state.topic}\n\nPROVISIONAL FAMILIES FROM ALL NOTE BATCHES:\n"
                f"{json.dumps(batch_families, ensure_ascii=False)[:18_000]}\n\n"
                "Merge overlaps, retain disagreements/risks, and select deep reads across batches."
            )
        else:
            user = f"TOPIC:\n{self.state.topic}\n\nREADING NOTES:\n{self._notes_digest(notes)}"
        data = self._think_json(
            system,
            user,
            role="planner", max_chars=20_000, reasoning=True,
        )
        families = [f for f in (data.get("families") or []) if isinstance(f, dict)]
        families = families[:8]
        if getattr(self.cfg, "research_information_scheduler", True):
            families = self.scheduler.rank_families(families, notes)
        path = root / f"idea-families-round-{self.state.round}.json"
        path.write_text(json.dumps({
            "families": families,
            "provisional_batch_families": batch_families,
            "paper_note_count": len(notes),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        md = [f"# Idea families round {self.state.round}", ""]
        for f in families:
            md += [f"## {f.get('name','family')}", f.get("interest", ""), "",
                   "**Question seeds:** " + "; ".join(str(x) for x in (f.get("question_seeds") or [])),
                   "**Key papers:** " + "; ".join(str(x) for x in (f.get("key_papers") or [])),
                   "**First checks:** " + "; ".join(str(x) for x in (f.get("first_checks") or [])), ""]
        (root / f"idea-families-round-{self.state.round}.md").write_text("\n".join(md), encoding="utf-8")
        self._log_thought("idea-families", f"{len(families)} candidate idea families synthesized", families=families)
        for family in families:
            try:
                family_id = self.obligations.ensure(
                    "idea_family",
                    str(family.get("name") or family.get("interest") or "candidate family"),
                    node_id=f"idea:{self.state.round}:{hashlib.sha256(str(family.get('name', '')).encode()).hexdigest()[:12]}",
                    stage="discovery", required=False, status="in_progress",
                    scope=str(family.get("interest") or ""), metadata=family,
                )
                self.obligations.link(family_id, self.obligations.objective_id, "derived_from")
            except Exception:
                pass
        self.obligations.save()
        self._say(f"ideas · {len(families)} families from reading notes")
        return families

    def _find_paper(self, ref: str):
        ref_l = str(ref or "").lower()
        for p in self.corpus.papers.values():
            if p.bare_id.lower() in ref_l or p.arxiv_id.lower() in ref_l:
                return p
        for p in self.corpus.papers.values():
            title = (p.title or "").lower()
            if title and (title[:40] in ref_l or ref_l[:40] in title):
                return p
        return None

    def _paper_chunks(self, p, *, chunk_chars: int = 7_000) -> tuple[list[str], dict]:
        """Build a bounded, evenly sampled map pass over an entire paper body."""

        text = (getattr(p, "text", "") or getattr(p, "abstract", "") or "").strip()
        limit = max(1, int(getattr(self.cfg, "research_deep_chunk_limit", 10)))
        if not text:
            return [], {"total_chars": 0, "covered_chars": 0, "coverage_fraction": 0.0,
                        "chunk_count": 0, "sampling": "missing"}
        if len(text) <= chunk_chars * limit:
            chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
            sampling = "complete"
        else:
            max_start = max(0, len(text) - chunk_chars)
            starts = [round(i * max_start / max(1, limit - 1)) for i in range(limit)]
            chunks = [text[start:start + chunk_chars] for start in starts]
            sampling = "evenly-sampled"
        covered = min(len(text), sum(len(c) for c in chunks))
        return chunks, {
            "total_chars": len(text),
            "covered_chars": covered,
            "coverage_fraction": round(covered / max(1, len(text)), 4),
            "chunk_count": len(chunks),
            "sampling": sampling,
        }

    def _deep_read_family(self, family: dict, *, remaining: int | None = None,
                          seen_global: set[str] | None = None) -> list[dict]:
        refs = []
        for key in ("deep_read_papers", "key_papers"):
            refs += [str(x) for x in (family.get(key) or []) if x]
        papers = []
        seen = set()
        for ref in refs:
            p = self._find_paper(ref)
            if p and p.bare_id not in seen:
                seen.add(p.bare_id)
                papers.append(p)
        if papers:
            from spiral.research_quality import rank_papers_for_topic

            family_topic = " ".join([
                str(family.get("name") or ""), str(family.get("interest") or ""),
                *[str(x) for x in (family.get("question_seeds") or [])],
            ])
            ranked = rank_papers_for_topic(family_topic or self.state.topic, papers)
            order = {paper.bare_id: index for index, paper in enumerate(ranked)}
            papers.sort(key=lambda paper: order.get(paper.bare_id, len(order)))
        limit = max(1, int(getattr(self.cfg, "research_deep_read_limit", 8)))
        if remaining is not None:
            limit = max(0, min(limit, remaining))
        notes = []
        for p in papers[:limit]:
            pid = getattr(p, "bare_id", getattr(p, "arxiv_id", "paper"))
            if seen_global is not None and pid in seen_global:
                continue
            if seen_global is not None:
                seen_global.add(pid)
            source_hash = self._paper_source_hash(p)
            family_material = json.dumps({
                "name": family.get("name", ""),
                "question_seeds": family.get("question_seeds") or [],
                "interest": family.get("interest", ""),
            }, sort_keys=True, ensure_ascii=False)
            family_key = hashlib.sha256(family_material.encode("utf-8")).hexdigest()[:12]
            cache_stem = f"{self._safe_id(pid)}-{family_key}-{source_hash[:12]}"
            path = self._notes_root() / "deep" / f"{cache_stem}.json"
            if path.is_file():
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if (cached.get("schema_version") == 2
                            and cached.get("source_hash") == source_hash
                            and cached.get("family_key") == family_key):
                        notes.append(cached)
                        self._say(f"  zoom · {p.bare_id[:24]} · cached deep read")
                        continue
                except Exception:
                    pass
            chunk_root = self._notes_root() / "deep" / cache_stem
            chunk_root.mkdir(parents=True, exist_ok=True)
            chunks, coverage = self._paper_chunks(p)
            model, num_ctx = self._notes_model_spec()
            chunk_notes = []
            chunk_system = (
                "Extract a source-grounded deep-reading note from ONE paper chunk. Return ONLY JSON "
                '{"summary":"...","results":["..."],"equations":["..."],'
                '"methods":["..."],"limitations":["..."],'
                '"evidence":[{"supports":"...","anchor":"exact phrase/equation from CHUNK"}]}. '
                "Do not infer claims not visible in this chunk."
            )
            for index, chunk in enumerate(chunks, 1):
                chunk_user = (
                    f"TOPIC:\n{self.state.topic}\n\nIDEA FAMILY:\n{json.dumps(family)[:1800]}\n\n"
                    f"PAPER: {p.title} ({p.arxiv_id})\nCHUNK {index}/{len(chunks)}:\n{chunk}"
                )
                chunk_note = self._think_json_model(
                    model, num_ctx, chunk_system, chunk_user,
                    max_tokens=1600, max_chars=11_000)
                if not chunk_note:
                    chunk_note = {"summary": "chunk note unavailable", "evidence": []}
                chunk_note = self._validate_note_evidence(chunk_note, chunk, p)
                chunk_note.update({"chunk": index, "chunks": len(chunks)})
                (chunk_root / f"chunk-{index:02d}.json").write_text(
                    json.dumps(chunk_note, indent=2, ensure_ascii=False), encoding="utf-8")
                chunk_notes.append(chunk_note)
            system = (
                "Synthesize the source-grounded CHUNK NOTES into a deep reading for the "
                "selected idea family. Return ONLY JSON "
                '{"arxiv_id":"...","family_relevance":"...","exact_equations":["..."],'
                '"known_results_or_limits":["..."],"open_gap_for_this_project":["..."],'
                '"normalisations":["..."],"verification_hooks":["..."],'
                '"danger_of_being_known":"...","notes_for_proposal":"..."}. '
                "Use only grounded chunk notes. Mark any missing derivation or sampled-away "
                "material explicitly."
            )
            user = (
                f"TOPIC:\n{self.state.topic}\n\nIDEA FAMILY:\n{json.dumps(family)[:2500]}\n\n"
                f"PAPER:\n{p.title} ({p.arxiv_id})\n\n"
                f"SOURCE COVERAGE:\n{json.dumps(coverage)}\n\n"
                f"CHUNK NOTES:\n{json.dumps(chunk_notes, ensure_ascii=False)[:22000]}"
            )
            note = self._think_json(
                system, user, role="planner", max_chars=24_000, reasoning=True)
            if not note:
                note = {"arxiv_id": p.arxiv_id, "family_relevance": "deep note unavailable",
                        "notes_for_proposal": (p.abstract or p.text or "")[:800]}
            note["source_coverage"] = coverage
            note["arxiv_id"] = getattr(p, "arxiv_id", pid)
            note["schema_version"] = 2
            note["source_hash"] = source_hash
            note["family_key"] = family_key
            note["family_name"] = family.get("name", "")
            note["grounded_evidence"] = [
                evidence for chunk_note in chunk_notes for evidence in (chunk_note.get("evidence") or [])
            ][:40]
            primary = bool(
                getattr(p, "tex_path", "") or getattr(p, "pdf_path", "")
                or str(getattr(p, "body_source", "")) in {"tex", "pdf"}
            )
            note["grounded"] = bool(primary and note["grounded_evidence"])
            path.write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
            notes.append(note)
            try:
                source_id = f"source:{self._safe_id(str(pid))}"
                deep_id = self.obligations.ensure(
                    "deep_read",
                    f"Deep read of {p.title} for {family.get('name', 'idea family')}",
                    node_id=f"deep-read:{cache_stem}", stage="discovery",
                    required=False,
                    status="supported" if note.get("grounded") else "blocked",
                    verifier="chunk-level exact anchor validation",
                    provenance=[str(path)], metadata={
                        "source_coverage": coverage,
                        "family": family.get("name", ""),
                        "grounded_evidence_count": len(note.get("grounded_evidence") or []),
                    },
                )
                if self.obligations.node(source_id):
                    self.obligations.link(deep_id, source_id, "derived_from")
            except Exception:
                pass
            self._log_thought("deep-read", f"{p.arxiv_id}: {note.get('family_relevance','deep read')}", note=note)
            self._say(f"  zoom · {p.bare_id[:24]} · deep read")
        return notes

    # -- phases --------------------------------------------------------------
    def _task_mode(self) -> str:
        """Separate original-research runs from expository verification tasks.

        ``--solve`` is an execution mode, not a license to mutate a user's stated
        theorem into a novelty hunt. If the topic asks to verify/write/explain a given
        identity, corpus papers are background and style exemplars.
        """
        if self.mode_override:
            return self.mode_override
        t = self.state.topic.lower()
        novelty = (
            "discover", "previously-unknown", "previously unknown", "novel",
            "new theorem", "new identity", "open question", "original",
            "generalize", "generalisation", "generalization", "extend", "improve",
            "investigate", "explore", "research", "find a", "search for",
            "conjecture", "derive a new", "prove a new",
        )
        if any(p in t for p in novelty):
            return "research"

        has_literal_identity = bool(self._literal_identity_claims())
        verify = ("verify", "check", "show", "prove", "confirm")
        note = (
            "short mathematical note", "expository", "explain",
            "write a note", "using corpus citations only as background",
            "citations only as background", "identity",
        )
        if has_literal_identity and any(p in t for p in verify) and any(p in t for p in note):
            return "expository"
        return "research"

    def _literal_identity_claims(self) -> list[dict]:
        """Extract identities literally stated in the topic, if any.

        This gives simple verification prompts a deterministic backbone. The model may
        still write around the result, but it does not get to replace ``(x+1)^2 = ...``
        with a different research programme.
        """
        from spiral.verify_math import verify

        text = " ".join(self.state.topic.replace("≡", "=").replace("−", "-").split())
        text = (text.replace("$", "").replace("\\(", "").replace("\\)", "")
                    .replace("\\[", "").replace("\\]", ""))
        text = re.sub(r"\^\s*\{([^{}]+)\}", r"^\1", text)
        text = text.replace("\\cdot", "*").replace("\\times", "*")
        pattern = re.compile(
            r"(?P<lhs>\([^=,;:.]+?\)(?:\s*\^\s*[-+]?\d+)?|"
            r"[A-Za-z0-9_.]+(?:\s*(?:\*\*|\^|[*/+\-])\s*[A-Za-z0-9_.()]+)*)"
            r"\s*=\s*"
            r"(?P<rhs>[A-Za-z0-9_.+\-*/^(){}\s]+)"
        )
        claims: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for m in pattern.finditer(text):
            lhs = m.group("lhs").strip()
            rhs = re.split(r"\b(using|with|where|for)\b", m.group("rhs"), 1)[0]
            rhs = rhs.strip(" .,;:")
            if not lhs or not rhs or (lhs, rhs) in seen:
                continue
            claim = {
                "kind": "identity",
                "lhs": lhs,
                "rhs": rhs,
                "statement": f"For every value in a commutative ring where the expression is defined, {lhs} = {rhs}.",
                "assumptions": ["ordinary commutative polynomial arithmetic"],
                "falsifier": f"a substitution or symbolic expansion for which {lhs} - ({rhs}) is nonzero",
                "method_family": "exact symbolic algebra",
                "required": True,
                "note": f"literal identity from the prompt: {lhs} = {rhs}",
            }
            verdict = verify(claim)
            parse_problem = ("SympifyError", "SyntaxError", "TokenError", "invalid syntax")
            if not any(p in verdict.detail for p in parse_problem):
                seen.add((lhs, rhs))
                claims.append(claim)
        return claims

    def search_plan(self, n: int = 3) -> tuple[list[str], list[str]]:
        """Decide WHERE on arXiv to look and WHAT to type: the right subject categories
        plus focused keyword queries. 'Find the right place, then search there' — a
        number-theory identity belongs in math.NT/math.CO, a gauge anomaly in
        hep-th/hep-ph; an unrestricted search for a term like 'Ramanujan' drowns in
        string-theory hits. Returns ``(categories, queries)``."""
        if self._task_mode() == "expository" and self._literal_identity_claims():
            return ["math.HO"], [
                "binomial theorem proof",
                "polynomial expansion identity",
                "elementary algebraic identities",
            ][:n]
        system = (
            "Choose where on arXiv to search for the TOPIC and what to type. Reply ONLY "
            'JSON {"categories":["math.NT","math.CO"],"queries":["hypergeometric identity","..."]}. '
            "categories: 1-4 real arXiv category codes for the field (math.NT, math.CO, "
            "math.CA, math.AG, hep-th, hep-ph, gr-qc, quant-ph, cond-mat.stat-mech, cs.LG, …). "
            "queries: 2-4 short keyword phrases (3-6 words), no punctuation, no cat: prefixes."
        )
        data = self._think_json(
            system, f"TOPIC: {self.state.topic}", role="planner", reasoning=True)
        cats = [c.strip() for c in data.get("categories", []) if isinstance(c, str) and c.strip()][:4]
        qs = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        if not qs:                                   # deterministic fallback: salient keywords
            stop = {"discover", "previously", "unknown", "propose", "verify", "confirm",
                    "write", "against", "literature", "using", "which", "their", "these",
                    "that", "with", "find", "prove", "exact", "new"}
            words = re.findall(r"[A-Za-z]{4,}", self.state.topic)   # split hyphens/punct
            keys = [w for w in words if w.lower() not in stop][:6]
            qs = [" ".join(keys[:4])] if keys else [self.state.topic[:60]]
        return cats, qs[:n]

    def gather(self, query: str, k: int = 8, categories=None) -> int:
        self._say(f"gather · {('/'.join(categories) + ' · ') if categories else ''}{query[:50]}")
        added = self.corpus.build(query, k=k, categories=categories,
                                  on=lambda a: self._say(f"  + {a}"))
        retrieval = dict(getattr(self.corpus, "last_build_report", {}) or {})
        # Widen only when the category search itself returned nothing/failed.  A healthy
        # search whose results are already in the corpus is evidence, not an empty search.
        if not added and categories and not retrieval.get("result_count"):
            added = self.corpus.build(query, k=k, on=lambda a: self._say(f"  + {a} (unrestricted)"))
            retrieval = {
                "restricted": retrieval,
                "fallback": dict(getattr(self.corpus, "last_build_report", {}) or {}),
                "source_ok": bool((getattr(self.corpus, "last_build_report", {}) or {}).get("source_ok")),
                "result_count": int((getattr(self.corpus, "last_build_report", {}) or {}).get("result_count") or 0),
            }
        for p in added:
            if p.bare_id not in self.state.corpus_ids:
                self.state.corpus_ids.append(p.bare_id)
        self._record_search(query, categories, [p.bare_id for p in added], k, retrieval)
        return len(added)

    _CLAIM_SPEC = (
        'Claims must be machine-verifiable, one of: '
        'Every claim must include "required":true|false; all required claims must pass '
        'before the proposal can be complete. Every claim must also include '
        '"statement":"self-contained proposition", "assumptions":["..."], '
        '"falsifier":"specific residual/counterexample/failing criterion", and '
        '"method_family":"formal|symbolic|numeric|compiled|mixed description". '
        '{"kind":"identity","lhs":"<sympy>","rhs":"<sympy>","required":true,"note":"..."}, '
        '{"kind":"solution","equation":"<expr=expr>","var":"x","value":"<sympy>","note":"..."}, '
        '{"kind":"groebner","generators":["<poly>", "..."],"variables":["x","y"],'
        '"basis":["<poly>", "..."],"order":"lex","note":"..."}, '
        '{"kind":"ideal_membership","expr":"<poly>","generators":["<poly>", "..."],'
        '"variables":["x","y"],"order":"lex","note":"..."}, '
        '{"kind":"numeric","code":"<python printing True/False last line>","note":"..."}, '
        '{"kind":"theorem","statement":"<Lean thm sig>","proof":"<Lean tactics, e.g. by decide>","note":"..."}, '
        '{"kind":"workbench","files":{"check.py":"exact reproducible code"},'
        '"cmd":"python check.py","expect":"CERTIFICATE_OK","requirements":["sympy"],'
        '"tools":["brew singular"],'
        '"datasets":[{"source":"openneuro|neuromaps|zenodo|allen|neurovault|osf",'
        '"id":"stable accession","version":"pinned release","alias":"short_name",'
        '"include":["exact/path/or/glob"],"purpose":"role in analysis",'
        '"species":"human","modalities":["fMRI"]}],'
        '"analysis_plan":{"mode":"confirmatory|exploratory","hypothesis":"falsifiable",'
        '"primary_outcome":"one prespecified outcome","unit_of_analysis":"...",'
        '"inclusion_exclusion":"...","confounds":"...","missing_data":"...",'
        '"multiple_testing":"correction/family","spatial_null":"spin/permutation or N/A",'
        '"validation":"held-out or independent replication","replication":"...",'
        '"stopping_rule":"...","causal_scope":"association only unless identified"},'
        '"alignment":{"target_space":"...","registration":"transforms and QC",'
        '"resolution_policy":"...","species_bridge":"N/A or explicit bridge",'
        '"participant_linkage":"matched or unmatched group-level sources"},'
        '"validation":{"independent_methods":['
        '{"name":"symbolic","step":0,"marker":"METHOD_OK:symbolic"},'
        '{"name":"independent numeric","step":1,"marker":"METHOD_OK:numeric"}],'
        '"acceptance_criteria":['
        '{"name":"residual bound","step":1,"marker":"CRITERION_OK:residual"}]},'
        '"required":true,"note":"..."}. '
        'For compile/run certificates, use "steps":["c++ -std=c++17 check.cpp -o check","./check"] '
        'instead of cmd. Direct Rust, Go, Julia, R, Java, Swift, Lean, Sage, Singular, Python, '
        'C and C++ commands are supported when installed; any other installed executable '
        'may be used under the offline OS sandbox. If a missing established CLI is materially '
        'better than reimplementing it, request a typed registry/core install in "tools" as '
        '"python package", "node package", or "brew formula". Choose from the supplied empirical '
        'tool profile. If a public GitHub implementation would materially help, include '
        '"repos":[{"url":"https://github.com/owner/repo","ref":"optional commit/tag","purpose":"..."}]; '
        "repo acquisition runs only when the user/config enables auto-repos and failed repos are cleaned up. "
        "Use sympy syntax (** powers, * products, pi, I, exp, sin). A theorem claim is the "
        "strongest — prefer it when the statement is a clean formal proposition. Use groebner/"
        "ideal_membership for small exact polynomial certificates. For large classification/RG/"
        "Lax/Gröbner tasks, emit workbench claims with exact code that derives and verifies the "
        "result, installs local Python requirements only when needed, prints residuals/cases, "
        "and ends with CERTIFICATE_OK only if all checks pass. A workbench run without two "
        "machine-observed method/criterion markers from two distinct successful command steps "
        "is executable evidence, not an independently reproduced result. For empirical work, "
        "declare datasets only inside a workbench claim, read them from `_data/ALIAS`, and "
        "prespecify the complete analysis_plan before acquisition. Use metadata/derivatives or "
        "a justified subset before raw multi-GB data. Unmatched atlases and participant data "
        "permit spatial/ecological association only; spatial maps require an explicit spatial "
        "null, and cross-species or cross-space fusion requires an alignment contract. A data "
        "certificate must write `spiral-result.json` containing aggregate-only fields "
        "`estimand`, `estimate`, `uncertainty`, `sample_size`, and `diagnostics`; never write "
        "participant/subject rows or identifiers into that summary."
    )

    def _repair_workbench_claim(self, claim: dict, result, attempt: int) -> dict | None:
        """Ask the reasoning model to patch a failed executable certificate once or twice.

        The repair is still untrusted and goes back through the same workbench gate; this
        just gives the model a chance to do ordinary research-programming things like fix
        a missing import, add a local requirement, or strengthen an assertion. Execution
        output is host-local data, so it is never sent to a configured API provider.
        """
        system = (
            "Repair this failed executable research certificate. Reply ONLY one JSON "
            "workbench claim with files/cmd/expect/requirements/tools/note and preserve any "
            "datasets, analysis_plan, alignment, and validation contracts unchanged. Preserve the "
            "mathematical goal, prefer exact symbolic code, use steps for compile/run bundles, "
            "add local Python requirements only when necessary, and end by printing "
            "CERTIFICATE_OK only after every independent check passes. Do not use shell, "
            "network calls from certificate code, destructive file operations, absolute paths, "
            "or subprocesses. If a Python certificate exited "
            "with code 0 but printed no marker, check that the script's entry point runs "
            'under `if __name__ == "__main__":`. A data certificate must also write '
            "aggregate-only spiral-result.json with estimand, estimate, uncertainty, "
            "sample_size, and diagnostics."
        )
        user = (
            f"ATTEMPT: {attempt}\nFAILED CLAIM:\n{json.dumps(claim)[:5000]}\n\n"
            f"DETAIL:\n{getattr(result, 'detail', '')[:1200]}\n\n"
            f"RUN META:\n{json.dumps(getattr(result, 'extra', {}))[:800]}\n\n"
            f"STDOUT:\n{getattr(result, 'stdout', '')[-3000:]}\n\n"
            f"STDERR:\n{getattr(result, 'stderr', '')[-3000:]}"
        )
        providers = getattr(self.ol, "providers", {})
        repair_role = next((
            role for role in ("research_auditor", "worker", "planner", "escalation", "critic")
            if getattr(self.cfg, role).name not in providers
        ), "")
        if not repair_role:
            self._say(
                "  workbench · repair blocked: no local model available for private execution output")
            self._log_thought(
                "workbench-repair-blocked",
                "certificate output was retained locally because every configured model was remote",
                manifest=getattr(result, "manifest", ""), attempt=attempt)
            return None
        fixed = self._think_json(
            system, user, role=repair_role, max_chars=9_000, reasoning=True)
        if not isinstance(fixed, dict) or not isinstance(fixed.get("files"), dict):
            return None
        fixed["kind"] = "workbench"
        fixed.setdefault("cmd", claim.get("cmd") or claim.get("command") or "python check.py")
        fixed.setdefault("expect", claim.get("expect") or "CERTIFICATE_OK")
        fixed.setdefault("requirements", claim.get("requirements") or [])
        fixed.setdefault("tools", claim.get("tools") or [])
        fixed.setdefault("note", f"repaired certificate for {claim.get('note', 'workbench claim')}")
        for key in (
                "statement", "assumptions", "falsifier", "method_family", "required",
                "conventions", "acceptance_criteria", "datasets", "analysis_plan",
                "alignment", "validation"):
            if key in claim:
                fixed.setdefault(key, claim[key])
        fixed["_repair_attempt"] = attempt
        fixed["_repaired_from"] = claim.get("manifest") or claim.get("note") or "previous workbench attempt"
        return fixed

    def _proposal_brief(self) -> str:
        """Let the research model think in a useful artifact before JSON commitment."""
        system = (
            "You are the research lead. Write a concise research "
            "brief, not JSON: identify the most promising fixed target in the TOPIC, the "
            "specific novelty risk, the equations, datasets or objects likely needed, and the "
            "first machine-checkable claim or executable analysis certificate that should be "
            "attempted. For empirical work, separate exploratory discovery from confirmatory "
            "testing and name a held-out validation route. Keep it under 900 words. Do not ask "
            "the user questions."
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\n"
            f"WORKING QUESTION:\n{self.state.question or '(none yet)'}\n\n"
            f"CORPUS SKETCH:\n{self.corpus.summaries(limit=14, chars=520)}\n\n"
            f"TYPED PUBLIC DATA CATALOG:\n{self._data_catalog_sketch()}"
        )
        try:
            brief, _ = self._think(
                system, user, role="planner", max_chars=10_000, num_predict=4096,
                temperature=0.35,
            )
        except ResearchModelError as exc:
            self._say(f"  model · research brief failed ({str(exc)[:120]})")
            return ""
        brief = brief[:6000]
        root = self._notes_root()
        (root / f"research-brief-round-{self.state.round}.md").write_text(
            brief + "\n", encoding="utf-8")
        self._log_thought("research-brief", brief, model_role="planner")
        return brief

    def _draft_proposal(self) -> dict:
        if self._task_mode() == "expository":
            system = (
                "You are writing a careful mathematical verification note, not hunting for "
                "a novel theorem. Keep the TOPIC's stated identity/equation unchanged. Do not "
                "invent nilpotent, idempotent, deformation, CRT, noncommutative, or open-problem "
                "generalizations unless the user explicitly asked for them. Use the CORPUS only "
                "as background/style context. Reply ONLY JSON "
                '{"question":"...","reasoning":"...","claims":[...]}. ' + self._CLAIM_SPEC
            )
        else:
            system = (
                "You are a research engine doing ORIGINAL work, not a chatbot. "
                "The TOPIC is your directive: narrow it into one concrete, tractable "
                "sub-question only if the corpus supports a grounded route. Do NOT force "
                "a novelty story through thin evidence. If the corpus is not ready, reply "
                'JSON {"no_proposal":true,"reasoning":"why not yet","missing":["..."],'
                '"next_query":"short arXiv query"}. Otherwise give CHECKABLE claims that '
                "make progress on the question. Physics/maths use many conventions (metric "
                "signature, α' and generator normalisations, index placement) — STATE the "
                "conventions you adopt and use them consistently. Reply ONLY JSON "
                '{"question":"...","conventions":"...","reasoning":"...","claims":[...]}. '
                + self._CLAIM_SPEC
            )
        brief = "" if self._task_mode() == "expository" else self._proposal_brief()
        user = (
            f"TOPIC: {self.state.topic}\n"
            f"WORKING QUESTION: {self.state.question or '(commit to one now)'}\n\n"
            + (f"RESEARCH BRIEF:\n{brief}\n\n" if brief else "")
            + f"CORPUS:\n{self.corpus.summaries(limit=14, chars=520)}\n\n"
            + f"TYPED PUBLIC DATA CATALOG:\n{self._data_catalog_sketch()}"
        )
        return self._think_json(
            system, user, role="planner", max_chars=10_000, reasoning=True)

    def _discover_angles(self, n: int = 5) -> list[dict]:
        """Mine the corpus for candidate research questions inside the user's topic."""
        notes = self._ensure_reading_notes()
        families = self._idea_families(notes) if notes else []
        deep_notes: list[dict] = []
        deep_seen: set[str] = set()
        deep_limit = max(1, int(getattr(self.cfg, "research_deep_read_limit", 8)))
        for family in families[: max(1, min(3, int(getattr(self.cfg, "research_deep_read_limit", 8))))]:
            remaining = deep_limit - len(deep_seen)
            if remaining <= 0:
                break
            deep_notes.extend(self._deep_read_family(
                family, remaining=remaining, seen_global=deep_seen))
        brief = self._proposal_brief()
        system = (
            "Find candidate research angles WITHIN the user's TOPIC. Use the corpus as "
            "evidence, not decoration. Start from IDEA FAMILIES and DEEP READ NOTES; "
            "these are the context-managed reading pass over the larger corpus. Return ONLY JSON "
            '{"angles":[{"question":"specific research question","target":"fixed object/model",'
            '"why_interesting":"...","corpus_basis":["paper id/title", "..."],'
            '"novelty_risk":"what may already be known","check_plan":"first exact/symbolic/numeric check",'
            '"data_plan":{"datasets":["source:id"],"discovery_split":"...",'
            '"confirmation_split":"...","alignment_risk":"...","negative_controls":["..."]},'
            '"search_queries":["short prior-art query", "..."]}]}. '
            "Each question must be narrow enough to attack with certificates, but still "
            "aimed at the original prompt. For data-driven topics, prefer a falsifiable result "
            "that can be independently validated over an attractive unconstrained correlation; "
            "reject species/space/modality combinations that cannot be aligned honestly. Do not "
            "include generic survey questions."
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\n"
            + (f"RESEARCH BRIEF:\n{brief}\n\n" if brief else "")
            + f"IDEA FAMILIES:\n{json.dumps(families, indent=2)[:6000]}\n\n"
            + f"DEEP READ NOTES:\n{json.dumps(deep_notes, indent=2)[:7500]}\n\n"
            + f"BROAD READING NOTES:\n{self._notes_digest(notes, limit=20, chars=3500)}\n\n"
            + f"TYPED PUBLIC DATA CATALOG:\n{self._data_catalog_sketch()}"
        )
        data = self._think_json(
            system, user, role="planner", max_chars=19_000, reasoning=True)
        raw = data.get("angles") if isinstance(data, dict) else []
        angles = []
        for a in raw or []:
            if isinstance(a, dict) and str(a.get("question", "")).strip():
                a["question"] = " ".join(str(a["question"]).split())[:500]
                angles.append(a)
        if getattr(self.cfg, "research_counterfactuals", True):
            angles.extend(self._counterfactual_angles(angles, families, deep_notes))
        # Taste is a transparent tie-breaker only. Every selected angle still goes
        # through source-health, primary-text, novelty, basis, and certificate gates.
        if getattr(self.cfg, "research_taste_model", True):
            angles = self.taste.rank(angles)
        seen_questions = set()
        deduped = []
        for angle in angles:
            key = self._question_key(str(angle.get("question") or ""))
            if key and key not in seen_questions:
                seen_questions.add(key)
                deduped.append(angle)
        angles = deduped
        for angle in angles:
            try:
                aid = self.obligations.ensure(
                    "candidate_question", str(angle.get("question") or ""),
                    node_id=f"candidate:{hashlib.sha256(self._question_key(str(angle.get('question') or '')).encode()).hexdigest()[:20]}",
                    stage="discovery", required=False, status="in_progress",
                    scope=str(angle.get("target") or ""), metadata={
                        "novelty_risk": angle.get("novelty_risk", ""),
                        "check_plan": angle.get("check_plan") or angle.get("first_check", ""),
                        "counterfactual": bool(angle.get("_counterfactual")),
                        "taste_score": angle.get("_taste_score"),
                    },
                )
                self.obligations.link(aid, self.obligations.objective_id, "derived_from")
                angle["_obligation_id"] = aid
            except Exception:
                pass
        candidate_ids = {
            self._question_key(str(angle.get("question") or "")): angle.get("_obligation_id")
            for angle in angles if angle.get("_obligation_id")
        }
        for angle in angles:
            if not angle.get("_counterfactual") or not angle.get("_obligation_id"):
                continue
            parent_id = candidate_ids.get(
                self._question_key(str(angle.get("parent_question") or "")))
            if parent_id:
                self.obligations.link(angle["_obligation_id"], parent_id, "derived_from", metadata={
                    "move": angle.get("move"),
                    "changed_assumption": angle.get("changed_assumption"),
                })
        self.obligations.save()
        self._log_thought("candidate-angles", f"{len(angles)} candidate research angles discovered", angles=angles)
        return angles[:n]

    def _counterfactual_angles(self, parents: list[dict], families: list[dict],
                               deep_notes: list[dict]) -> list[dict]:
        """Probe useful neighbouring hypotheses before prematurely committing."""

        if not parents:
            return []
        limit = max(0, min(12, int(getattr(self.cfg, "research_counterfactual_limit", 6))))
        if not limit:
            return []
        system = (
            "You run a counterfactual discovery lab. Starting from grounded candidate "
            "research angles, alter exactly one assumption, limit, representation, or method "
            "at a time. Seek boundary cases, singular limits, transferred methods, structural "
            "obstructions, and honest no-go theorems. Do not claim truth or novelty. Return "
            "ONLY JSON "
            '{"counterfactuals":[{"parent_question":"exact parent question",'
            '"question":"neighbouring testable question",'
            '"move":"relax_assumption|strengthen_assumption|boundary_case|singular_limit|'
            'method_transfer|dual_formulation|obstruction|negative_theorem|parameter_extension",'
            '"changed_assumption":"one explicit change","why_informative":"...",'
            '"falsifier":"observable result that rejects it","first_check":"exact executable check",'
            '"corpus_basis":["paper id",...],"search_queries":["short query",...]}]}. '
            "Stay inside the user's topic and cite only ids visible in the supplied notes."
        )
        data = self._think_json(
            system,
            f"TOPIC:\n{self.state.topic}\n\nPARENT ANGLES:\n"
            f"{json.dumps(parents[:5], ensure_ascii=False)[:8000]}\n\n"
            f"IDEA FAMILIES:\n{json.dumps(families[:5], ensure_ascii=False)[:5000]}\n\n"
            f"DEEP READS:\n{json.dumps(deep_notes[:8], ensure_ascii=False)[:7000]}",
            role="planner", max_chars=22_000, reasoning=True,
        )
        candidates = []
        for raw in (data.get("counterfactuals") or [])[:limit * 2]:
            if not isinstance(raw, dict):
                continue
            parent = next((
                angle for angle in parents
                if self._question_key(str(angle.get("question") or ""))
                == self._question_key(str(raw.get("parent_question") or ""))
            ), None)
            if parent is None:
                continue
            valid, issues = self.counterfactual_lab.validate(parent, raw)
            record = {
                **raw,
                "target": parent.get("target", ""),
                "novelty_risk": (
                    "counterfactual hypothesis; novelty entirely unresolved until prior-art audit"),
                "check_plan": raw.get("first_check", ""),
                "_counterfactual": True,
                "_counterfactual_valid": valid,
                "_counterfactual_issues": issues,
            }
            if valid:
                candidates.append(record)
            if len(candidates) >= limit:
                break
        path = self.counterfactual_lab.save_round(self.state.round, parents, candidates)
        self._log_thought(
            "counterfactual-lab",
            f"{len(candidates)} source-adjacent counterfactuals survived structural validation",
            path=str(path), candidates=candidates,
        )
        self._say(f"counterfactuals · {len(candidates)} testable neighbours")
        return candidates

    @staticmethod
    def _dedupe_queries(queries: list[str], *, limit: int = 3) -> list[str]:
        out = []
        seen = set()
        for query in queries:
            q = " ".join(str(query or "").split()).strip()
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
            if len(out) >= limit:
                break
        return out

    def _prior_queries(self, question: str, extra: list[str] | None = None) -> list[str]:
        from spiral.research_quality import topic_terms

        lexical = " ".join(topic_terms(question, limit=9))
        broad = " ".join(topic_terms(self.state.topic, limit=8))
        return self._dedupe_queries([question, *(extra or []), lexical, broad], limit=3)

    def _prior_art_bundle(self, queries: list[str], *, k: int = 6) -> tuple[list, dict]:
        """Run diversified prior-art searches and retain source-health evidence."""

        from spiral.citations import prior_art
        from spiral.research_quality import query_family_count

        all_priors = []
        seen_titles = set()
        query_reports = []
        for query in self._dedupe_queries(queries, limit=3):
            report: dict = {}
            try:
                hits = prior_art(query, k=k, physics=True, report=report)
            except TypeError:
                # An older plugin/test provider cannot establish source health.  Preserve
                # its results, but never silently treat unknown health as a novelty check.
                hits = prior_art(query, k=k, physics=True)
                report = {
                    "query": query, "ready": False, "sources_ok": [],
                    "result_count": len(hits), "error": "source-health telemetry unavailable",
                }
            report.setdefault("query", query)
            query_reports.append(report)
            for prior in hits:
                key = prior.title.lower().strip()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    all_priors.append(prior)
        healthy_queries = sum(1 for r in query_reports if r.get("ready") is True)
        healthy_query_texts = [
            str(r.get("query") or "") for r in query_reports if r.get("ready") is True
        ]
        healthy_query_families = query_family_count(healthy_query_texts)
        sources_ok = sorted({s for r in query_reports for s in (r.get("sources_ok") or [])})
        bundle = {
            "queries": [r.get("query", "") for r in query_reports],
            "query_reports": query_reports,
            "healthy_queries": healthy_queries,
            "healthy_query_families": healthy_query_families,
            "sources_ok": sources_ok,
            "result_count": len(all_priors),
            "checks": {
                "multiple_healthy_queries": healthy_queries >= 2,
                "independent_query_families": healthy_query_families >= 2,
                "multiple_healthy_sources": len(sources_ok) >= 2,
            },
        }
        bundle["ready"] = all(bundle["checks"].values())
        self.map.setdefault("prior_art_searches", []).append({
            "round": self.state.round,
            **bundle,
        })
        self._save_map()
        return all_priors, bundle

    def _ground_nearby_priors(self, question: str, priors: list, *, limit: int = 2) -> dict:
        """Fetch and deeply read the closest identifiable prior-art papers.

        Search-result titles/abstracts are enough to reject obvious duplicates, but not to
        support an absence claim. A candidate that survives the first referee pass therefore
        triggers primary-text reads of its nearest arXiv neighbours before commitment.
        """

        from spiral.research_corpus import Paper, parse_arxiv_id
        from spiral.research_quality import topic_terms

        q_terms = set(topic_terms(question, limit=16))
        ranked = []
        seen = set()
        for prior in _prior_objects(priors):
            aid = parse_arxiv_id(prior.identifier or prior.url)
            if not aid:
                continue
            bare = Paper(arxiv_id=aid).bare_id
            if bare in seen:
                continue
            seen.add(bare)
            blob = f"{prior.title} {prior.abstract}".lower()
            overlap = sum(1 for term in q_terms if term in blob)
            ranked.append((overlap, prior.citations, prior, aid))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[3]))
        selected = ranked[:max(0, limit)]
        added = []
        papers = []
        broad_notes = []
        for _, _, prior, aid in selected:
            existed = self.corpus.has(aid)
            paper = self.corpus.add(Paper(
                arxiv_id=aid,
                title=prior.title,
                authors=list(prior.authors),
                abstract=prior.abstract,
                published=str(prior.year or ""),
                url=prior.url or f"https://arxiv.org/abs/{aid}",
            ), fetch=not existed)
            if not existed:
                added.append(paper.bare_id)
            papers.append(paper)
            broad_notes.append(self._paper_note(paper))
        if added:
            self.corpus.save()
            # The corpus changed after this round's graph report. Its old saturation result
            # cannot certify the enlarged corpus; the next round must evaluate it again.
            self.state.coverage["novelty_ready"] = False
            self.state.coverage.setdefault("warnings", []).append(
                "nearest prior-art papers were added after graph evaluation; rerun coverage")

        family = {
            "name": "nearest prior-art audit",
            "interest": f"Determine whether the closest retrieved papers answer: {question}",
            "question_seeds": [question],
            "deep_read_papers": [p.bare_id for p in papers],
            "key_papers": [p.bare_id for p in papers],
        }
        deep_notes = self._deep_read_family(family, remaining=len(papers)) if papers else []
        report = {
            "identifiable_prior_count": len(ranked),
            "selected_ids": [p.bare_id for p in papers],
            "new_corpus_ids": added,
            "grounded_broad_reads": sum(1 for note in broad_notes if note.get("grounded")),
            "grounded_deep_reads": sum(1 for note in deep_notes if note.get("grounded")),
            "deep_notes": deep_notes,
        }
        self._log_thought(
            "nearest-prior-deep-read",
            f"deep-read {len(deep_notes)}/{len(papers)} selected nearby prior papers",
            report=report,
        )
        return report

    def _audit_angle(self, angle: dict, priors: list, search_report: dict | None = None,
                     grounded_priors: dict | None = None) -> dict:
        """Decide whether a candidate is known, thin, or worth turning into claims."""
        from spiral.citations import novelty_digest

        if not (search_report or {}).get("ready"):
            return {
                "verdict": "thin",
                "novelty": "unresolved: prior-art retrieval protocol was not healthy",
                "basis": "source failure/insufficient query diversity is not evidence of absence",
                "checkability": str(angle.get("check_plan") or ""),
                "issues": ["prior-art search health gate failed"],
                "next_query": str((angle.get("search_queries") or [""])[0]),
                "search_report": search_report or {},
            }
        if (grounded_priors is not None
                and grounded_priors.get("identifiable_prior_count", 0) > 0
                and grounded_priors.get("grounded_deep_reads", 0) < 1):
            return {
                "verdict": "thin",
                "novelty": "unresolved: nearby identifiable prior art could not be deep-read",
                "basis": "a failed primary-text read is not evidence that the result is absent",
                "checkability": str(angle.get("check_plan") or ""),
                "issues": ["no source-grounded deep read of the closest identifiable prior"],
                "next_query": str((angle.get("search_queries") or [""])[0]),
                "search_report": search_report or {},
                "grounded_prior_report": grounded_priors,
            }

        system = (
            "You are the gap referee. Decide whether this candidate research angle should "
            "be pursued. Reply ONLY JSON "
            '{"verdict":"pursue|known|thin","novelty":"...","basis":"...",'
            '"checkability":"...","issues":["..."],"next_query":"short search if rejected"}. '
            "Use 'known' if prior art appears to already classify/answer it. Use 'thin' "
            "if the corpus does not support the bridge or the first check is vague. Use "
            "'pursue' only when it is within the topic, not apparently answered, and has a "
            "concrete certificate path."
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\n"
            f"ANGLE:\n{json.dumps(angle)[:2500]}\n\n"
            f"CORPUS:\n{self.corpus.summaries(limit=12, chars=380)}\n\n"
            + (f"GROUNDED NEAREST-PRIOR DEEP READS:\n"
               f"{json.dumps(grounded_priors, ensure_ascii=False)[:7000]}\n\n"
               if grounded_priors is not None else "")
            + f"{novelty_digest(_prior_objects(priors))}"
        )
        data = self._think_json(
            system, user, role="critic", max_chars=8_000, reasoning=True)
        verdict = str(data.get("verdict", "")).lower()
        if verdict not in {"pursue", "known", "thin"}:
            data["verdict"] = "thin"
        data["search_report"] = search_report or {}
        return data

    def _proposal_from_angle(self, angle: dict, priors: list) -> dict:
        from spiral.citations import novelty_digest

        system = (
            "Turn this accepted ANGLE into a checkable research proposal. Reply ONLY JSON "
            '{"question":"...","conventions":"...","reasoning":"...","claims":[...]}. '
            "The question must stay within the user's topic and the selected fixed target. "
            "Do not claim the whole ambitious programme is solved; produce the first "
            "verifiable step or exact sub-classification that would genuinely advance it. "
            "If no concrete certificate can be stated, reply JSON "
            '{"no_proposal":true,"reasoning":"why no checkable step yet","missing":["..."],'
            '"next_query":"short arXiv query"}. '
            + self._CLAIM_SPEC
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\n"
            f"ANGLE:\n{json.dumps(angle)[:3000]}\n\n"
            f"{novelty_digest(_prior_objects(priors))}\n\n"
            f"CORPUS:\n{self.corpus.summaries(limit=14, chars=430)}\n\n"
            f"TYPED PUBLIC DATA CATALOG:\n{self._data_catalog_sketch()}\n\n"
            f"LOCAL TOOL CAPABILITIES:\n{self.toolsmith.capability_brief()}\n\n"
            f"EMPIRICALLY SUCCESSFUL LOCAL CERTIFICATE RECIPES:\n"
            f"{self.toolsmith.recipe_brief(limit=6) or '(none recorded yet)'}"
        )
        data = self._think_json(
            system, user, role="planner", max_chars=9_000, reasoning=True)
        if data.get("no_proposal"):
            data["_no_proposal"] = True
            data.setdefault("claims", [])
        return data

    def _compact_proposal_retry(self, question_hint: str) -> dict:
        system = (
            "The previous proposal call failed to produce usable JSON. Make one compact, "
            "honest decision. If there is a grounded, checkable route, reply JSON "
            '{"question":"...","conventions":"...","reasoning":"...","claims":[...]}. '
            "If not, reply JSON "
            '{"no_proposal":true,"reasoning":"why not yet","missing":["..."],'
            '"next_query":"short arXiv query"}. '
            "Do not force novelty through thin evidence. For a hard physics classification, "
            "a valid first claim may be a workbench certificate deriving/checking a symbolic "
            "reduction, Frobenius recurrence, numerical matching, or known limiting case. "
            + self._CLAIM_SPEC
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\n"
            f"QUESTION HINT:\n{question_hint}\n\n"
            f"CORPUS SKETCH:\n{self.corpus.summaries(limit=10, chars=420)}\n\n"
            f"TYPED PUBLIC DATA CATALOG:\n{self._data_catalog_sketch()}"
        )
        data = self._think_json(
            system, user, role="planner", max_chars=7_500, reasoning=True)
        if not data:
            return {}
        if data.get("no_proposal"):
            data["_no_proposal"] = True
            data.setdefault("claims", [])
            return data
        data.setdefault("question", question_hint)
        data.setdefault("claims", [])
        data.setdefault("reasoning", "compact proposal retry after structured-output failure")
        return data

    def _fallback_question(self) -> str:
        """A deterministic question hint for retry prompts.

        This is not accepted as a result by itself; it just prevents retry prompts from
        being blank when the model has not yet committed to a grounded question.
        """
        first = re.split(r"(?<=[.?!])\s", self.state.topic.strip())[0]
        return " ".join((first or self.state.topic).split())[:240]

    def _critique_proposal(self, proposal: dict, priors: list,
                           search_report: dict | None = None) -> dict:
        """Vet the *proposal* against prior art + rigor BEFORE spending verification on
        it — the research analogue of a referee. Steers away from re-deriving 1962."""
        from spiral.citations import novelty_digest
        if not (search_report or {}).get("ready"):
            return {
                "verdict": "revise",
                "novelty": "unresolved because the prior-art protocol did not pass",
                "rigor": "source health is a prerequisite for novelty review",
                "interest": "not assessed",
                "issues": ["healthy diversified prior-art searches are missing"],
                "steer": "repair or diversify literature retrieval",
            }
        system = (
            "You are a hard-nosed referee vetting a research PROPOSAL before any work is "
            "done. Judge it on three axes and reply ONLY JSON "
            '{"verdict":"accept|revise","novelty":"...","rigor":"...","interest":"...",'
            '"issues":["..."],"steer":"..."}: '
            "(1) NOVELTY — is the question already answered in the PRIOR ART? If so, revise. "
            "(2) RIGOR — are the claims concrete and independently checkable, not vague? "
            "(3) INTEREST — would answering it matter? 'accept' only if all three hold."
        )
        user = (f"PROPOSAL question: {proposal.get('question','')}\n"
                f"claims: {json.dumps(proposal.get('claims', []))[:1200]}\n\n"
                f"{novelty_digest(_prior_objects(priors))}")
        return self._think_json(
            system, user, role="critic", max_chars=8_000, reasoning=True)

    def _basis_audit(self, proposal: dict, priors: list) -> dict:
        """Check whether a novelty proposal is actually grounded in the evidence trail.

        This is deliberately separate from verification. SymPy/Lean can decide whether
        a claim is true, but they cannot decide whether the *research move* came from
        the corpus or from a fluent hallucination.
        """
        from spiral.citations import novelty_digest

        paper_notes: dict[str, dict] = {}
        note_root = self.dir / "notes" / "papers"
        if note_root.is_dir():
            for path in note_root.glob("*.json"):
                try:
                    note = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                pid = str(note.get("arxiv_id") or path.stem).replace("arXiv:", "").split("v")[0]
                paper_notes[pid] = note

        from spiral.research_quality import rank_papers_for_topic

        ranked_corpus = rank_papers_for_topic(
            self.state.question or self.state.topic, self.corpus.papers.values())
        corpus_by_id = {p.bare_id: p for p in ranked_corpus}
        grounding_packet = []
        for paper in ranked_corpus[:30]:
            note = paper_notes.get(paper.bare_id) or {}
            if (note.get("schema_version") != 2 or not note.get("grounded")
                    or note.get("source_hash") != self._paper_source_hash(paper)):
                continue
            grounding_packet.append({
                "id": paper.bare_id,
                "title": paper.title,
                "main_results": (note.get("main_results") or [])[:3],
                "methods": (note.get("methods") or [])[:3],
                "gaps_or_openings": (note.get("gaps_or_openings") or [])[:3],
                "evidence": (note.get("evidence") or [])[:8],
            })

        system = (
            "You are the corpus-grounding referee for an autonomous mathematical research "
            "run. Decide whether the PROPOSAL is genuinely based on the CORPUS/PRIOR ART, "
            "or whether it is an unsupported creative pivot. Do not judge truth; judge "
            "evidence basis. Keyword overlap alone is not enough. Reply ONLY JSON "
            '{"verdict":"grounded|thin","basis":"...","evidence":[{"source":"corpus|prior",'
            '"id":"paper id or title","supports":"specific concept/bridge",'
            '"anchor":"exact supplied source phrase/equation"}],'
            '"missing":["..."],"searches":["short query",...]}. '
            "Use grounded only when the relevant objects, methods, or bridge are supported "
            "by the exact supplied anchors. A paper being topically related is not enough."
        )
        user = (
            f"TOPIC:\n{self.state.topic}\n\nPROPOSAL:\n{json.dumps(proposal)[:3000]}\n\n"
            "SOURCE-GROUNDED CORPUS PACKET:\n"
            f"{json.dumps(grounding_packet, ensure_ascii=False)[:16000]}\n\n"
            f"{novelty_digest(_prior_objects(priors))}"
        )
        basis_role = (
            "research_auditor"
            if self.cfg.research_auditor.name != self.cfg.critic.name else "critic")
        data = self._think_json(
            system, user, role=basis_role, max_chars=22_000, context_limit=16_384)
        if not data:
            data = {"verdict": "thin", "basis": "no parseable grounding audit",
                    "evidence": [], "missing": ["parseable basis audit"], "searches": []}

        def norm(value: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))

        def match_reference(reference: str, candidates: list[tuple[str, str]]):
            raw = str(reference or "").lower().replace("arxiv:", "")
            folded = norm(reference)
            for identifier, title in candidates:
                title_folded = norm(title)
                if identifier.lower() in raw or (len(folded) >= 10 and folded == title_folded):
                    return identifier
            return ""

        corpus_sources = [(p.bare_id, p.title or "") for p in corpus_by_id.values()]
        prior_sources = []
        prior_by_id = {}
        for index, prior in enumerate(_prior_objects(priors), 1):
            identifier = str(prior.identifier or f"prior-{index}")
            prior_sources.append((identifier, prior.title or ""))
            prior_by_id[identifier] = prior

        valid_evidence = []
        rejected_evidence = []
        for item in data.get("evidence") or []:
            if not isinstance(item, dict) or not str(item.get("id", "")).strip() \
                    or not str(item.get("supports", "")).strip() \
                    or not str(item.get("anchor", "")).strip():
                rejected_evidence.append({
                    "evidence": item, "reason": "missing id/support description/exact anchor"})
                continue
            source = str(item.get("source") or "").lower().strip()
            if source == "corpus":
                resolved = match_reference(item["id"], corpus_sources)
                note = paper_notes.get(resolved) or {}
                if not resolved:
                    rejected_evidence.append({"evidence": item, "reason": "corpus reference is not held"})
                    continue
                if not note.get("grounded") or not note.get("evidence"):
                    rejected_evidence.append({
                        "evidence": item,
                        "reason": f"corpus paper {resolved} has no source-grounded reading note",
                    })
                    continue
                paper = corpus_by_id[resolved]
                if (note.get("schema_version") != 2
                        or note.get("source_hash") != self._paper_source_hash(paper)):
                    rejected_evidence.append({
                        "evidence": item,
                        "reason": f"corpus paper {resolved} has a stale or unauthenticated reading note",
                    })
                    continue
                anchor = self._normalise_anchor(item.get("anchor", ""))
                supplied = {
                    self._normalise_anchor(e.get("anchor", ""))
                    for e in (note.get("evidence") or []) if isinstance(e, dict)
                }
                full_source = self._normalise_anchor("\n".join([
                    paper.title or "", paper.abstract or "", paper.text or "",
                ]))
                if len(anchor) < 12 or anchor not in supplied or anchor not in full_source:
                    rejected_evidence.append({
                        "evidence": item,
                        "reason": f"corpus anchor for {resolved} is not an exact authenticated span",
                    })
                    continue
                valid_evidence.append({
                    **item, "source": "corpus", "resolved_id": resolved,
                    "anchor_verified": True,
                })
            elif source == "prior":
                resolved = match_reference(item["id"], prior_sources)
                if not resolved:
                    rejected_evidence.append({"evidence": item, "reason": "prior reference was not retrieved"})
                    continue
                prior = prior_by_id[resolved]
                anchor = self._normalise_anchor(item.get("anchor", ""))
                prior_source = self._normalise_anchor(
                    f"{prior.title or ''} {prior.abstract or ''}")
                if len(anchor) < 12 or anchor not in prior_source:
                    rejected_evidence.append({
                        "evidence": item,
                        "reason": f"prior anchor for {resolved} is not in the retrieved record",
                    })
                    continue
                valid_evidence.append({
                    **item, "source": "prior", "resolved_id": resolved,
                    "anchor_verified": True,
                })
            else:
                rejected_evidence.append({"evidence": item, "reason": "source must be corpus or prior"})

        has_corpus_basis = any(e.get("source") == "corpus" for e in valid_evidence)
        grounded = (
            str(data.get("verdict", "")).lower() == "grounded"
            and bool(valid_evidence)
            and has_corpus_basis
            and not rejected_evidence
        )
        data["verdict"] = "grounded" if grounded else "thin"
        data["grounded"] = grounded
        data["evidence"] = valid_evidence
        data["rejected_evidence"] = rejected_evidence
        data["deterministic_checks"] = {
            "all_references_resolved": not rejected_evidence,
            "all_anchors_authenticated": all(
                e.get("anchor_verified") for e in valid_evidence),
            "has_source_grounded_corpus_basis": has_corpus_basis,
        }
        return data

    def _refine_proposal(self, proposal: dict, critique: dict, priors: list) -> dict:
        system = (
            "Revise the PROPOSAL to fix the referee's ISSUES: make it genuinely novel "
            "(distinct from the prior art), sharper, and more clearly checkable. Keep what "
            "worked. Reply ONLY JSON {\"question\":\"...\",\"reasoning\":\"...\",\"claims\":[...]}. "
            + self._CLAIM_SPEC
        )
        steer = ", ".join(p.title for p in _prior_objects(priors)[:6] if p.title)
        user = (f"PROPOSAL: {json.dumps(proposal)[:1500]}\n\nREFEREE: {json.dumps(critique)[:800]}\n\n"
                f"STEER AWAY FROM: {steer}")
        return self._think_json(
            system, user, role="planner", max_chars=8_000, reasoning=True)

    def propose(self, refine_rounds: int = 2) -> dict:
        """Draft a proposal, then iterate it against prior art + a referee critique until
        it is accepted or the refinement budget runs out — so what reaches verification is
        already vetted for novelty and rigor, not the model's first guess."""
        if self._task_mode() == "expository":
            literal = self._literal_identity_claims()
            if literal:
                first = literal[0]
                self._say("proposal · expository literal identity")
                proposal = {
                    "question": f"Verify the identity {first['lhs']} = {first['rhs']}.",
                    "reasoning": (
                        "The prompt asks for verification and a short note. The stated "
                        "identity is therefore checked directly; corpus material is used "
                        "as background rather than as a novelty target."
                    ),
                    "claims": literal,
                    "_vetted": True,
                    "mode": "expository",
                }
                self._register_proposal_obligations(proposal)
                return proposal
            proposal = self._draft_proposal()
            proposal.setdefault("question", self._fallback_question())
            proposal.setdefault("claims", [])
            proposal.setdefault("reasoning", "")
            proposal["_vetted"] = True
            proposal["mode"] = "expository"
            self._register_proposal_obligations(proposal)
            return proposal

        angles = self._discover_angles()
        proposal = {}
        rejected: list[dict] = []
        for i, angle in enumerate(angles, 1):
            q = str(angle.get("question", "")).strip()
            if not q:
                continue
            self._say(f"  angle · {i}/{len(angles)} · checking prior art · {q[:70]}")
            queries = self._prior_queries(
                q, [s for s in angle.get("search_queries", []) if isinstance(s, str)])
            priors, search_report = self._prior_art_bundle(queries, k=5)
            audit = self._audit_angle(angle, priors[:8], search_report)
            self._say(
                f"  angle · {audit.get('verdict','thin')} · "
                f"{str(audit.get('novelty') or audit.get('basis') or '')[:60]}"
            )
            self._log_thought(
                "angle-audit",
                f"{audit.get('verdict','thin')}: {q}",
                angle=angle,
                audit=audit,
                prior_art=[asdict(p) for p in priors[:8]],
            )
            if audit.get("verdict") != "pursue":
                if getattr(self.cfg, "research_taste_model", True):
                    self.taste.observe(angle, str(audit.get("verdict") or "rejected"))
                if angle.get("_obligation_id"):
                    self.obligations.set_status(
                        angle["_obligation_id"], "superseded",
                        reason=str(audit.get("novelty") or audit.get("basis") or "angle rejected"))
                rejected.append({"angle": angle, "audit": audit})
                continue
            grounded_priors = self._ground_nearby_priors(q, priors[:8])
            audit = self._audit_angle(angle, priors[:8], search_report, grounded_priors)
            self._say(
                f"  angle · deep prior audit · {audit.get('verdict','thin')} · "
                f"{grounded_priors.get('grounded_deep_reads', 0)} grounded reads"
            )
            self._log_thought(
                "angle-deep-prior-audit",
                f"{audit.get('verdict','thin')}: {q}",
                angle=angle, audit=audit, grounded_priors=grounded_priors,
            )
            if audit.get("verdict") != "pursue":
                if getattr(self.cfg, "research_taste_model", True):
                    self.taste.observe(angle, str(audit.get("verdict") or "rejected"))
                if angle.get("_obligation_id"):
                    self.obligations.set_status(
                        angle["_obligation_id"], "superseded",
                        reason=str(audit.get("novelty") or audit.get("basis") or "deep prior rejected angle"))
                rejected.append({"angle": angle, "audit": audit})
                continue
            if getattr(self.cfg, "research_taste_model", True):
                self.taste.observe(angle, "pursue")
            proposal = self._proposal_from_angle(angle, priors[:8])
            if proposal.get("question") and proposal.get("claims"):
                proposal["_angle"] = angle
                proposal["_angle_audit"] = audit
                proposal["_prior_art_report"] = search_report
                proposal["_grounded_prior_report"] = grounded_priors
                self._log_thought(
                    "proposal-commit",
                    f"committed proposal: {proposal.get('question')}",
                    proposal=proposal,
                    angle=angle,
                )
                break
            rejected.append({
                "angle": angle,
                "audit": {"verdict": "thin",
                          "next_query": proposal.get("next_query") or audit.get("next_query", ""),
                          "issues": proposal.get("missing") or ["no concrete machine-checkable claim"]},
            })
            proposal = {}

        if not proposal:
            retry = self._compact_proposal_retry(self.state.question or self._fallback_question())
            if retry.get("question") and retry.get("claims"):
                proposal = retry
            else:
                next_query = ""
                missing = []
                for r in rejected:
                    audit = r.get("audit") or {}
                    if not next_query:
                        next_query = str(audit.get("next_query") or "").strip()
                    missing += [str(x) for x in (audit.get("issues") or []) if x]
                if retry.get("next_query"):
                    next_query = retry["next_query"]
                if retry.get("missing"):
                    missing += [str(x) for x in retry.get("missing") if x]
                self._say("  proposal · no grounded checkable angle yet")
                no_prop = {
                    "_no_proposal": True,
                    "question": self.state.question,
                    "claims": [],
                    "reasoning": retry.get("reasoning")
                    or "No candidate angle survived novelty/basis/checkability review.",
                    "missing": missing[:8],
                    "next_query": next_query,
                    "rejected_angles": rejected[:5],
                }
                self._log_thought("no-proposal", no_prop["reasoning"], proposal=no_prop)
                return no_prop

        proposal.setdefault("claims", [])
        proposal.setdefault("reasoning", "")
        for _ in range(max(0, refine_rounds)):
            extra = list((proposal.get("_angle") or {}).get("search_queries") or [])
            priors, search_report = self._prior_art_bundle(
                self._prior_queries(proposal["question"], extra), k=6)
            proposal["_prior_art_report"] = search_report
            critique = self._critique_proposal(proposal, priors, search_report)
            self._say(f"  refine · {critique.get('verdict','?')} · {str(critique.get('novelty',''))[:40]}")
            if critique.get("verdict") == "accept":
                audit = self._basis_audit(proposal, priors)
                proposal["_basis_audit"] = audit
                self._say(f"  basis · {audit.get('verdict','thin')} · {str(audit.get('basis',''))[:50]}")
                if audit.get("grounded"):
                    proposal["_vetted"] = True
                    break
                critique = {
                    "verdict": "revise",
                    "novelty": "basis thin",
                    "rigor": "proposal lacks corpus-grounded support",
                    "interest": critique.get("interest", ""),
                    "issues": audit.get("missing") or ["novelty move is not grounded in the corpus"],
                    "steer": "; ".join(audit.get("searches") or []) or audit.get("basis", ""),
                }
            refined = self._refine_proposal(proposal, critique, priors)
            if refined.get("question"):
                # Proposal prose may change, but its corpus-mined search routes and
                # provenance remain part of the audit trail. Dropping them here can
                # collapse the next novelty check back to paraphrases of one query.
                for key in (
                    "_angle", "_angle_audit", "_grounded_prior_report",
                ):
                    if key in proposal and key not in refined:
                        refined[key] = proposal[key]
                proposal = refined
        if not proposal.get("_vetted"):
            audit = proposal.get("_basis_audit")
            proposal["claims"] = []
            note = (f"Basis audit did not ground this novelty move: "
                    f"{audit.get('basis','thin evidence')}") if audit else \
                "Proposal did not pass the novelty/rigor/basis referee yet."
            proposal["reasoning"] = f"{proposal.get('reasoning','')}\n\n{note}".strip()
            proposal["_no_proposal"] = True
            if audit:
                proposal["next_query"] = (audit.get("searches") or [""])[0]
        elif proposal.get("question") and proposal.get("claims"):
            self._register_proposal_obligations(proposal)
        return proposal

    def evaluate_corpus_quality(self, *, stalled_rounds: int = 0) -> dict:
        """Run the non-LLM retrieval/coverage gate and persist its evidence.

        ``stalled_rounds`` is the observable no-progress streak from the loop. When the
        retrieval instruments are provably dead and only instrument checks block
        discovery, the gate degrades EXPLICITLY (see
        :func:`spiral.research_quality.apply_stall_override`) instead of vetoing
        forever — soundness gates must not become liveness bugs."""

        from spiral.research_quality import (
            CoveragePolicy, apply_stall_override, corpus_quality_report, report_markdown)

        policy = CoveragePolicy(
            min_papers=int(getattr(self.cfg, "research_min_papers", 10)),
            min_usable_texts=int(getattr(self.cfg, "research_min_usable_texts", 6)),
            min_relevant_papers=int(getattr(self.cfg, "research_min_relevant_papers", 5)),
            min_relevant_usable_primary_texts=int(getattr(
                self.cfg, "research_min_relevant_usable_primary_texts", 4)),
            min_unique_queries=int(getattr(self.cfg, "research_min_unique_queries", 3)),
            min_healthy_searches=int(getattr(self.cfg, "research_min_healthy_searches", 2)),
            min_relevant_query_families=int(getattr(
                self.cfg, "research_min_relevant_query_families", 2)),
            min_topic_term_coverage=float(
                getattr(self.cfg, "research_min_topic_term_coverage", 0.45)),
            min_graph_success_rate=float(
                getattr(self.cfg, "research_min_graph_success_rate", 0.60)),
        )
        report = corpus_quality_report(
            self.state.topic,
            self.corpus.papers.values(),
            self.map,
            notes_root=self.dir / "notes",
            policy=policy,
        )
        health = self._instrument_health()
        report["instrument_health"] = health
        # A fired override is STICKY while the instruments stay dead: opening the gate
        # changes the progress signature, which resets the stall counter — without
        # stickiness the override would un-fire next round and discovery would
        # oscillate open/closed on a still-dead instrument.
        previous_override = (self.state.coverage or {}).get("stall_override") or {}
        effective_stalled = stalled_rounds
        if previous_override and health["instruments_dead"]:
            effective_stalled = max(
                stalled_rounds, int(previous_override.get("stalled_rounds") or 0))
        if effective_stalled and not report["discovery_ready"]:
            report = apply_stall_override(
                report,
                stalled_rounds=effective_stalled,
                patience=max(2, int(getattr(self.cfg, "research_stall_patience", 3))),
                instruments_dead=health["instruments_dead"],
                evidence=health,
            )
            if report.get("stall_override") and not previous_override:
                overridden = ", ".join(report["stall_override"]["overridden_checks"])
                self._say(
                    f"stall · instruments dead {effective_stalled} rounds — discovery "
                    f"opened in degraded mode ({overridden}; limitation recorded)")
                self._log_thought(
                    "stall-override",
                    "discovery opened in degraded mode; novelty still gated",
                    override=report["stall_override"],
                )
        report["round"] = self.state.round
        root = self.dir / "coverage"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"round-{self.state.round}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.dir / "coverage-latest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.dir / "coverage-latest.md").write_text(report_markdown(report), encoding="utf-8")
        self.map.setdefault("coverage_reports", []).append({
            "round": self.state.round,
            "discovery_ready": report["discovery_ready"],
            "novelty_ready": report["novelty_ready"],
            "paper_count": report["paper_count"],
            "relevant_paper_count": report["relevant_paper_count"],
            "blocking_reasons": report["blocking_reasons"],
            "path": str(path),
        })
        self.state.coverage = report
        self._save_map()
        label = "ready" if report["discovery_ready"] else "not ready"
        blockers = ", ".join(report["blocking_reasons"][:3])
        self._say(f"coverage · {label}" + (f" · {blockers}" if blockers else ""))
        self._log_thought(
            "coverage-gate",
            f"discovery {label}; novelty {'ready' if report['novelty_ready'] else 'not ready'}",
            report=report,
        )
        return report

    def assess_corpus(self) -> dict:
        """Ask the research model which concepts are missing — grounded and stateful.

        This is a search-steering judgment, not the sufficiency gate.  Observable corpus
        readiness is decided separately by :meth:`evaluate_corpus_quality`.

        The judge sees (1) a FULL id·title index of the corpus, (2) body excerpts of the
        most topic-RELEVANT papers — not the first-N in insertion order, which for a
        seeded corpus shows the seeds forever — and (3) its own previous verdict plus
        what arrived since, so gaps can be checked off instead of re-derived from
        nothing. A run once burned 23 identical critic calls declaring Kodama-Ishibashi
        'absent' while that exact paper sat in the corpus, just outside a 20-paper
        insertion-order window; every element here exists to make that impossible."""
        if self._task_mode() == "expository":
            self._say("assess · corpus used as background")
            return {"sufficient": True, "missing": [], "searches": []}
        from spiral.research_quality import rank_papers_for_topic
        ranked_ids = [
            p.bare_id for p in rank_papers_for_topic(
                self.state.topic, self.corpus.papers.values())
        ]
        history = self.map.get("corpus_assessments") or []
        last = history[-1] if history else {}
        known_before = set(last.get("known_ids") or [])
        now_ids = list(self.corpus.papers)
        new_ids = [pid for pid in now_ids if pid not in known_before]
        new_lines = [
            f"- {pid} · {(self.corpus.papers[pid].title or '(untitled)').strip()[:80]}"
            for pid in new_ids[:30]
        ]
        prev_missing = [str(m)[:220] for m in (last.get("missing") or [])][:8]
        tried = sorted({q.strip() for q in self._tried_queries() if q.strip()})
        system = (
            "You audit a LOCAL research corpus for the TOPIC. You are shown the FULL INDEX "
            "of every paper in the corpus, body excerpts of the most topic-relevant papers, "
            "the gaps you flagged last round, and the papers added since. Judge ONLY against "
            "the index and excerpts: call an item missing ONLY if no indexed paper plausibly "
            "covers it; when an indexed paper resolves a previously flagged gap, move it to "
            '"resolved" citing its id. Reply ONLY JSON {"sufficient":true|false,'
            '"resolved":[{"item":"...","ids":["..."]}],"missing":["..."],'
            '"searches":["short arXiv query",...]}. At most 6 missing items, most blocking '
            "first. searches: at most 4 short keyword queries, none a paraphrase of an "
            "ALREADY-TRIED query."
        )
        user = (
            f"TOPIC: {self.state.topic}\n\n"
            + (("PREVIOUSLY FLAGGED MISSING:\n- " + "\n- ".join(prev_missing) + "\n\n")
               if prev_missing else "")
            + ((f"ADDED SINCE LAST ASSESSMENT ({len(new_ids)} papers):\n"
                + "\n".join(new_lines) + "\n\n") if known_before and new_ids else "")
            + f"ALREADY-TRIED QUERIES:\n{'; '.join(tried[:40]) or '(none)'}\n\n"
            + f"FULL INDEX ({len(now_ids)} papers):\n{self.corpus.index(title_chars=76)}\n\n"
            + "CORPUS:\n"
            + self.corpus.summaries(limit=16, chars=700, ids=ranked_ids)
        )
        data = self._think_json(
            system, user, role="critic", max_chars=26_000, reasoning=True)
        record = {
            "round": self.state.round,
            "sufficient": bool(data.get("sufficient")),
            "missing": [str(m) for m in (data.get("missing") or [])][:8],
            "resolved": data.get("resolved") or [],
            "searches": [str(s).strip() for s in (data.get("searches") or [])
                         if isinstance(s, str) and s.strip()][:4],
            "paper_count": len(now_ids),
            "known_ids": now_ids,
        }
        self.map["corpus_assessments"] = (history + [record])[-8:]
        self._save_map()
        if not data.get("sufficient", True):
            resolved = len(record["resolved"])
            self._say("assess · corpus missing targeted concepts"
                      + (f" · {resolved} previous gap(s) resolved" if resolved else ""))
            self._log_thought(
                "corpus-assessment",
                "corpus missing targeted concepts",
                assessment=data,
            )
            for q in record["searches"][:3]:
                if self._query_is_novel(q):
                    self.gather(q, k=4)
                else:
                    self._say(f"  skip · paraphrase of a tried search · {q[:60]}")
        else:
            self._say("assess · no conceptual gap named")
            self._log_thought("corpus-assessment", "model named no conceptual gap", assessment=data)
        return data

    def _journal(self, proposal: dict, findings: list, decision: dict):
        """Append this round to a human-readable research log — the map of ideas explored,
        what verified/refuted, and why the loop continued — so a long run is inspectable."""
        j = self.dir / "journal.md"
        lines = [f"\n## Round {self.state.round}", f"**Question:** {self.state.question}"]
        if proposal.get("conventions"):
            lines.append(f"**Conventions:** {proposal['conventions'][:300]}")
        lines.append(f"**Idea:** {str(proposal.get('reasoning',''))[:400]}")
        audit = proposal.get("_basis_audit") or {}
        if audit:
            lines.append(f"**Basis audit:** {audit.get('verdict','thin')} — "
                         f"{str(audit.get('basis',''))[:300]}")
            for e in (audit.get("evidence") or [])[:3]:
                lines.append(f"  - {e.get('source','?')} {e.get('id','?')}: "
                             f"{str(e.get('supports',''))[:180]}")
        if proposal.get("_no_proposal") or proposal.get("no_proposal"):
            missing = "; ".join(str(x) for x in (proposal.get("missing") or [])[:6])
            if missing:
                lines.append(f"**Missing before proposal:** {missing}")
            if proposal.get("next_query"):
                lines.append(f"**Next search:** {proposal['next_query']}")
        rejected = proposal.get("rejected_angles") or []
        if rejected:
            lines.append("**Rejected angles:**")
            for r in rejected[:5]:
                angle = r.get("angle") or {}
                angle_audit = r.get("audit") or {}
                why = angle_audit.get("novelty") or angle_audit.get("basis") or angle_audit.get("issues") or ""
                lines.append(
                    f"- {angle_audit.get('verdict','thin')}: "
                    f"{str(angle.get('question',''))[:180]} — {str(why)[:220]}"
                )
        lines.append("**Claims:**")
        for f in findings:
            label = {
                "formal": "PROVED",
                "exact": "EXACTLY VERIFIED",
                "computational": "COMPUTATIONALLY REPRODUCED",
                "empirical": "NUMERICALLY OBSERVED",
                "executable": "EXECUTED (NOT INDEPENDENTLY VERIFIED)",
            }.get(f.strength, "refuted/unverified") if f.ok else "refuted/unverified"
            lines.append(f"- {'✓' if f.ok else '✗'} {label} [{f.backend}] "
                         f"{f.claim.get('note','')} — {f.detail[:90]}")
        lines.append(f"**Decision:** {decision.get('action','?')} — {str(decision.get('reason',''))[:300]}")
        with j.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    @staticmethod
    def _claim_id(claim: dict) -> str:
        stable = {
            k: v for k, v in (claim or {}).items()
            if not str(k).startswith("_") and k not in {"manifest"}
        }
        raw = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _finding(self, claim: dict, ok: bool, backend: str, detail: str,
                 *, strength: str) -> Finding:
        claim_id = str(claim.get("_logical_claim_id") or self._claim_id(claim))
        return Finding(
            claim=claim,
            ok=ok,
            backend=backend,
            detail=detail,
            round=self.state.round,
            question=self.state.question,
            claim_id=claim_id,
            strength=strength if ok else "unverified",
            required=bool(claim.get("required", True)),
            obligation_id=str(claim.get("_obligation_id") or ""),
        )

    @staticmethod
    def _workbench_strength(claim: dict, ok: bool) -> str:
        if not ok:
            return "unverified"
        try:
            manifest = json.loads(Path(str(claim.get("manifest") or "")).read_text(
                encoding="utf-8"))
            evidence = manifest.get("validation_evidence") or {}
            data_evidence = manifest.get("data_evidence") or {}
            data_ready = bool(
                data_evidence.get("not_applicable")
                or (
                    data_evidence.get("confirmatory_ready") is True
                    and data_evidence.get("result_summary_ready") is True
                ))
            if evidence.get("computationally_reproduced") is True and data_ready:
                return "computational"
        except Exception:
            pass
        return "executable"

    def _verify_workbench_claim(self, claim: dict) -> Finding:
        from spiral.research_workbench import run_workbench_claim

        current = self._normalise_workbench_claim(claim)
        current.setdefault("_logical_claim_id", self._claim_id(claim))
        if claim.get("_obligation_id"):
            current.setdefault("_obligation_id", claim["_obligation_id"])
        try:
            repairs = int(current.get("repair_rounds", 2))
        except Exception:
            repairs = 2
        repairs = max(0, min(2, repairs))
        last = None
        for attempt in range(repairs + 1):
            last = run_workbench_claim(
                current,
                self.dir / "certificates",
                timeout=float(current.get("timeout", 300)),
                allow_repos=bool(getattr(self.cfg, "research_repo_auto", False)),
                repo_budget=int(getattr(self.cfg, "research_repo_budget", 1)),
                repo_max_mb=int(getattr(self.cfg, "research_repo_max_mb", 750)),
                cleanup_failed_repos=bool(getattr(self.cfg, "research_cleanup_failed_repos", True)),
                allow_tools=bool(getattr(
                    self.cfg, "research_tool_auto", True)),
                tool_budget=int(getattr(
                    self.cfg, "research_tool_install_budget", 4)),
                allow_data=bool(getattr(
                    self.cfg, "research_data_auto", True)),
                data_root=self.dir / "data",
                data_cfg=self.cfg,
            )
            current["manifest"] = last.manifest
            result_summary = (
                last.extra.get("result_summary")
                if isinstance(last.extra, dict) else {})
            if result_summary:
                current["_result_summary"] = result_summary
            blocked = any(w in last.detail.lower() for w in ("blocked", "unsafe"))
            if last.ok or attempt >= repairs or blocked:
                detail = last.detail
                if result_summary:
                    detail += (
                        "; aggregate: "
                        + json.dumps(result_summary, ensure_ascii=False)[:1200]
                    )
                return self._finding(
                    current, last.ok, "workbench", detail[:1500],
                    strength=self._workbench_strength(current, last.ok))
            self._say("  repair · workbench certificate")
            fixed = self._repair_workbench_claim(current, last, attempt + 1)
            if not fixed:
                return self._finding(current, False, "workbench", last.detail[:300],
                                     strength="unverified")
            fixed["_logical_claim_id"] = current.get("_logical_claim_id")
            if current.get("_obligation_id"):
                fixed["_obligation_id"] = current["_obligation_id"]
            current = self._normalise_workbench_claim(fixed)
        return self._finding(
            current, False, "workbench",
            (last.detail if last else "workbench did not run")[:300], strength="unverified")

    def _normalise_workbench_claim(self, claim: dict) -> dict:
        """Tolerate a common malformed local-model shape.

        The intended schema has top-level ``cmd``/``expect``/``requirements`` and a
        ``files`` mapping. Smaller models sometimes put those metadata keys inside the
        files mapping after a long code string. Move them back before running.
        """
        out = dict(claim or {})
        files = out.get("files")
        if isinstance(files, dict):
            files = dict(files)
            for key in (
                "cmd", "command", "steps", "expect", "requirements",
                    "timeout", "repair_rounds", "repos", "tools", "datasets",
                    "analysis_plan", "alignment", "validation"):
                if key not in out and key in files:
                    out[key] = files.pop(key)
            out["files"] = files
        return out

    def _verify_replica_candidate(self, claim: dict, claim_id: str) -> dict:
        """Verify a blinded replica without entering the ordinary repair loop."""

        from spiral.numeric_lab import check_numeric_claim
        from spiral.research_workbench import run_workbench_claim
        from spiral.verify_math import verify

        kind = str(claim.get("kind") or "").lower()
        backend = ""
        detail = ""
        manifest = ""
        if kind in {"workbench", "certificate", "code_certificate"}:
            claim = self._normalise_workbench_claim(claim)
            result = run_workbench_claim(
                claim,
                self.dir / "certificates" / "blind-replication" / claim_id,
                timeout=float(claim.get("timeout", 300)),
                allow_repos=False,
                cleanup_failed_repos=True,
                allow_data=bool(getattr(
                    self.cfg, "research_data_auto", True)),
                data_root=self.dir / "data",
                data_cfg=self.cfg,
            )
            claim["manifest"] = result.manifest
            ok = result.ok
            backend = "workbench"
            detail = result.detail
            manifest = result.manifest
            strength = self._workbench_strength(claim, result.ok)
        elif kind == "numeric":
            result = check_numeric_claim(claim.get("code", ""))
            ok = result.ok
            backend = "numeric"
            detail = result.error or result.stdout
            strength = "empirical" if result.ok else "unverified"
        else:
            result = verify(claim)
            ok = result.ok
            backend = result.backend
            detail = result.detail
            if result.backend == "lean":
                strength = "formal"
            elif result.backend == "sympy" and result.kind != "numeric_equal":
                strength = "exact"
            elif result.ok:
                strength = "empirical"
            else:
                strength = "unverified"
        return {
            "ok": bool(ok), "backend": backend, "detail": str(detail)[:2000],
            "strength": strength, "manifest": manifest, "claim": claim,
        }

    def _blind_replicate(self, finding: Finding) -> dict:
        """Regenerate a required result while hiding its original solution artifact."""

        from spiral.research_replication import (
            blind_brief, independent_enough, inspect_replica_methods, write_report,
        )

        if (self._task_mode() == "expository"
                or not getattr(self.cfg, "research_blind_replication", True)
                or not finding.required):
            return {"status": "not_required", "passed": True, "independent": False}
        brief = blind_brief(
            finding.claim, question=self.state.question,
            conventions=str(self.state.active_proposal.get("conventions") or ""),
        )
        attempts = max(1, min(4, int(getattr(self.cfg, "research_replication_attempts", 2))))
        report = {
            "claim_id": finding.claim_id,
            "status": "failed",
            "passed": False,
            "brief": brief,
            "solution_hidden": True,
            "original_backend": finding.backend,
            "original_strength": finding.strength,
            "planner_model": self.cfg.planner.name,
            "replica_model": self.cfg.research_auditor.name,
            "attempts": [],
        }
        system = (
            "You are an independent replication researcher. You receive only a proposition, "
            "assumptions, conventions, falsifier, and acceptance criteria. You do NOT receive "
            "the original proof, code, output, or certificate. Re-derive it independently with "
            "a method different from ORIGINAL METHOD FAMILY. Return ONLY JSON "
            '{"approach":"concise public method summary","method_family":"specific distinct method",'
            '"claim":{...}}. The claim must be either a complete Lean theorem with a proof and '
            "no `sorry`, or a workbench certificate with complete files and commands. A workbench "
            "certificate must execute at least two genuinely distinct successful method steps, "
            "emit exact METHOD_OK: markers and a CRITERION_OK: marker declared in `validation`, "
            "and one method must actively search the supplied falsifier/boundary cases for a "
            "counterexample inside the stated domain before printing CERTIFICATE_OK. Do not "
            "return a bare identity that merely asks SymPy to "
            "recheck the supplied proposition. If the brief contains datasets, copy the frozen "
            "datasets, analysis_plan, and alignment objects into the replica claim, read only "
            "from `_data/ALIAS`, and write aggregate-only spiral-result.json with estimand, "
            "estimate, uncertainty, sample_size, and diagnostics. Do not use network access "
            "or repositories."
        )
        for attempt in range(1, attempts + 1):
            data = self._think_json(
                system,
                f"BLINDED REPLICATION BRIEF:\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n\n"
                f"LOCAL TOOL CAPABILITIES:\n{self.toolsmith.capability_brief()}\n\n"
                f"ATTEMPT: {attempt}/{attempts}",
                role="research_auditor", max_chars=12_000,
                max_tokens=min(self.cfg.planner_max_tokens, 8192),
                context_limit=16_384, reasoning=True,
            )
            replica_claim = data.get("claim") if isinstance(data.get("claim"), dict) else {}
            if data.get("method_family") and replica_claim:
                replica_claim["method_family"] = data["method_family"]
            if not replica_claim:
                report["attempts"].append({
                    "attempt": attempt, "ok": False,
                    "reason": "replication model returned no executable claim",
                })
                continue
            result = self._verify_replica_candidate(replica_claim, finding.claim_id)
            independence = independent_enough(
                finding.claim, finding.backend, result["claim"], result["backend"],
                self.cfg.planner.name, self.cfg.research_auditor.name,
            )
            method_audit = (
                inspect_replica_methods(result["claim"], result["manifest"])
                if result["backend"] == "workbench" else {
                    "method_diversity": True,
                    "adversarial_falsifier_check": True,
                    "reason": "formal/exact verifier covers the full encoded proposition",
                }
            )
            qualifying = result["strength"] in {"formal", "exact", "computational"}
            passed = bool(
                result["ok"] and qualifying and independence["independent"]
                and method_audit.get("method_diversity")
                and method_audit.get("adversarial_falsifier_check"))
            attempt_record = {
                "attempt": attempt,
                "approach": str(data.get("approach") or "")[:1200],
                "replica_claim": result["claim"],
                "backend": result["backend"],
                "strength": result["strength"],
                "detail": result["detail"],
                "manifest": result["manifest"],
                "independence": independence,
                "method_audit": method_audit,
                "qualifying": qualifying,
                "ok": passed,
            }
            report["attempts"].append(attempt_record)
            if passed:
                report.update({
                    "status": "passed", "passed": True,
                    "backend": result["backend"], "strength": result["strength"],
                    "manifest": result["manifest"], "independence": independence,
                    "method_audit": method_audit,
                    "approach": attempt_record["approach"],
                })
                break
        path = write_report(self.dir, finding.claim_id, report)
        report["path"] = str(path)
        self._say(
            f"  {'✓' if report['passed'] else '✗'} [blind replication] "
            f"{finding.claim.get('note', finding.claim_id)[:50]}")
        self._log_thought(
            "blind-replication",
            f"{report['status']}: {finding.claim_id}", report=report,
        )
        return report

    def verify_claims(self, claims: list) -> list[Finding]:
        from spiral.numeric_lab import check_numeric_claim
        from spiral.verify_math import verify
        out = []
        for c in claims or []:
            kind = str(c.get("kind", "")).lower()
            if kind == "numeric":
                r = check_numeric_claim(c.get("code", ""))
                fnd = self._finding(c, r.ok, "numeric", (r.error or r.stdout)[:200],
                                    strength=self._workbench_strength(c, r.ok)
                                    if c.get("validation") else "empirical")
            elif kind in {"workbench", "certificate", "code_certificate"}:
                fnd = self._verify_workbench_claim(c)
            else:
                v = verify(c)
                if v.backend == "lean":
                    strength = "formal"
                elif v.backend == "sympy" and v.kind != "numeric_equal":
                    strength = "exact"
                elif v.ok:
                    strength = "empirical"
                else:
                    strength = "unverified"
                fnd = self._finding(c, v.ok, v.backend, v.detail, strength=strength)
            self._say(f"  {'✓' if fnd.ok else '✗'} [{fnd.backend}] {c.get('note', kind)[:50]}")
            if fnd.ok:
                fnd.replication = self._blind_replicate(fnd)
            self._sync_finding_obligation(fnd)
            out.append(fnd)
        return out

    def novelty(self, question: str) -> list:
        self._say("novelty · searching prior art")
        extra = list((self.state.active_proposal.get("_angle") or {}).get("search_queries") or [])
        priors, report = self._prior_art_bundle(self._prior_queries(question, extra), k=8)
        report["corpus_novelty_ready"] = bool(self.state.coverage.get("novelty_ready"))
        report["ready"] = bool(report.get("ready") and report["corpus_novelty_ready"])
        self._last_novelty_report = report
        path = self.dir / f"novelty-search-round-{self.state.round}.json"
        path.write_text(json.dumps({
            "question": question,
            "report": report,
            "results": [asdict(p) for p in priors],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log_thought(
            "novelty-search",
            f"prior-art protocol {'ready' if report['ready'] else 'not ready'}; {len(priors)} records",
            report=report,
            results=[asdict(p) for p in priors],
        )
        prior_dicts = [asdict(p) for p in priors]
        try:
            from spiral.research_provenance import NoveltyBoundaryCertificate

            relevant = self._relevant_stored_findings(question)
            certificate = NoveltyBoundaryCertificate.build(
                self.dir, question=question,
                proposal=self.state.active_proposal,
                findings=relevant,
                search_report=report,
                priors=prior_dicts,
                coverage=self.state.coverage,
            )
            self.state.novelty_boundary = certificate
            ids = self.state.active_proposal.get("_obligations") or {}
            novelty_id = str(ids.get("novelty") or "")
            if novelty_id:
                eid = self.obligations.add_evidence(
                    novelty_id,
                    certificate.get("scope_statement", "bounded novelty search"),
                    evidence_kind="novelty_certificate",
                    artifact=certificate.get("path", ""),
                    verifier="deterministic novelty protocol",
                    relation="scopes",
                    status="supported" if certificate.get("valid") else "blocked",
                    metadata={
                        "certificate_sha256": certificate.get("certificate_sha256"),
                        "valid": certificate.get("valid"),
                        "as_of": certificate.get("as_of"),
                    },
                    node_id=f"novelty-certificate:{self.state.round}",
                )
                self.obligations.link(eid, novelty_id, "scopes")
                self.obligations.set_status(
                    novelty_id, "supported" if certificate.get("valid") else "blocked",
                    reason=certificate.get("scope_statement", ""),
                    verifier="novelty boundary certificate",
                )
                self.obligations.save()
            self._say(
                f"novelty boundary · {'valid' if certificate.get('valid') else 'blocked'}")
        except Exception as exc:
            self.state.novelty_boundary = {
                "valid": False, "error": f"{type(exc).__name__}: {exc}"}
        return prior_dicts

    def reflect(self, verified: list[Finding], priors: list) -> dict:
        from spiral.citations import novelty_digest
        confirmed = [f for f in verified if f.ok]
        if self._task_mode() == "expository":
            if confirmed:
                return {
                    "assessment": "The stated expository identity has been machine-verified.",
                    "novel": False,
                    "action": "solved",
                    "next_query": "",
                    "reason": "The prompt requested verification/write-up, not a novel prior-art claim.",
                }
            return {
                "assessment": "No stated expository claim has verified yet.",
                "novel": False,
                "action": "continue",
                "next_query": "",
                "reason": "Need a machine-verified version of the stated identity before writing.",
            }
        system = (
            "You are the research supervisor. Given the QUESTION, the VERIFIED claims "
            "(machine-checked — trust these), the REFUTED claims, and PRIOR ART, decide "
            "the next action. Reply with ONLY JSON: "
            '{"assessment":"...","novel":true|false,'
            '"action":"continue|solved|new_question|pivot",'
            '"next_query":"<SHORT keyword arXiv search, 3-6 words, to deepen the corpus>",'
            '"reason":"..."}. '
            "'solved' only if the confirmed claims actually answer the question AND prior "
            "art does not already contain it. 'new_question' if the work instead surfaced a "
            "verified-open question worth pursuing. Be honest: unverified is not solved."
        )
        digest = novelty_digest(_prior_objects(priors))
        user = (f"QUESTION: {self.state.question}\n\n"
                f"VERIFIED:\n" + "\n".join(f"- [{f.backend}] {f.claim.get('note','')}: {f.detail}" for f in confirmed) +
                f"\n\nREFUTED:\n" + "\n".join(f"- {f.claim.get('note','')}: {f.detail}" for f in verified if not f.ok) +
                f"\n\n{digest}")
        data = self._think_json(
            system, user, role="critic", max_chars=8_000, reasoning=True)
        if not data:
            return {"assessment": "no parseable supervisor response",
                    "action": "continue", "next_query": "",
                    "reason": "supervisor returned no parseable JSON"}
        if data.get("action") not in {"continue", "solved", "new_question", "pivot"}:
            data["action"] = "continue"
        return data

    @staticmethod
    def _question_key(question: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (question or "").lower()))

    def _relevant_stored_findings(self, question: str) -> list[dict]:
        key = self._question_key(question)
        return [
            f for f in self.state.findings
            if f.get("ok") and self._question_key(str(f.get("question") or "")) == key
        ]

    def completion_gate(self, proposal: dict, decision: dict) -> dict:
        """Deterministically decide whether a supervisor's `solved` may terminate.

        The model judges scientific scope; code enforces evidence strength, required-claim
        coverage, corpus readiness, and healthy novelty retrieval.
        """

        self._sync_decision_obligations(proposal, decision)
        relevant = self._relevant_stored_findings(self.state.question)
        passed_ids = {f.get("claim_id") for f in relevant if f.get("claim_id")}
        passed_notes = {
            " ".join(str((f.get("claim") or {}).get("note", "")).lower().split())
            for f in relevant
        }
        qualifying_strengths = {"formal", "exact", "computational"}
        qualifying = [f for f in relevant if f.get("strength") in qualifying_strengths]
        required = [c for c in (proposal.get("claims") or []) if c.get("required", True)]
        claim_contract_gaps = []
        for claim in required:
            missing = []
            if not str(claim.get("statement") or "").strip():
                missing.append("statement")
            if not isinstance(claim.get("assumptions"), list):
                missing.append("assumptions")
            if not str(claim.get("falsifier") or "").strip():
                missing.append("falsifier")
            if not str(claim.get("method_family") or "").strip():
                missing.append("method_family")
            if missing:
                claim_contract_gaps.append({
                    "claim": claim.get("note") or self._claim_id(claim),
                    "missing": missing,
                })
        missing_required = []
        weak_required = []
        by_id = {f.get("claim_id"): f for f in relevant if f.get("claim_id")}
        by_note = {
            " ".join(str((f.get("claim") or {}).get("note", "")).lower().split()): f
            for f in relevant
        }
        for claim in required:
            cid = self._claim_id(claim)
            note = " ".join(str(claim.get("note", "")).lower().split())
            found = by_id.get(cid) or by_note.get(note)
            if not found or (cid not in passed_ids and note not in passed_notes):
                missing_required.append(claim.get("note") or cid)
            elif found.get("strength") not in qualifying_strengths:
                weak_required.append({
                    "claim": claim.get("note") or cid,
                    "strength": found.get("strength", "unverified"),
                })

        expository = self._task_mode() == "expository"
        replication_gaps = [
            f.get("claim_id") for f in relevant
            if f.get("required", True)
            and f.get("strength") in qualifying_strengths
            and not (f.get("replication") or {}).get("passed")
        ] if not expository and getattr(self.cfg, "research_blind_replication", True) else []
        novelty_boundary_validation = {"valid": True, "issues": []}
        if not expository:
            try:
                from spiral.research_provenance import NoveltyBoundaryCertificate

                novelty_boundary_validation = NoveltyBoundaryCertificate.validate(
                    self.state.novelty_boundary)
            except Exception as exc:
                novelty_boundary_validation = {
                    "valid": False, "issues": [f"novelty certificate validation failed: {exc}"]}
        obligation_report = self.obligations.report("result")
        self.state.obligation_report = obligation_report
        from spiral.research_quality import reading_metrics, verify_jsonl_hash_chain

        reading = reading_metrics(self.dir / "notes")
        thought_chain = verify_jsonl_hash_chain(self.dir / "thoughts.jsonl")
        model_call_chain = verify_jsonl_hash_chain(self.dir / "model-calls.jsonl")
        if expository and not (self.dir / "model-calls.jsonl").is_file():
            model_call_chain = {
                "ok": True,
                "entries": 0,
                "head": "",
                "not_required": "literal expository result used no model call",
            }
        available_relevant = max(1, int(self.state.coverage.get("relevant_paper_count") or 0))
        required_notes = min(
            max(0, int(getattr(self.cfg, "research_min_grounded_notes", 6))),
            available_relevant,
        )
        required_deep = min(
            max(0, int(getattr(self.cfg, "research_min_grounded_deep_reads", 2))),
            available_relevant,
        )
        checks = {
            "supervisor_scope_decision": decision.get("action") in {"solved", "new_question"},
            "has_qualifying_evidence": bool(qualifying),
            "all_required_claims_passed": not missing_required,
            "required_evidence_is_independent": not weak_required,
            "corpus_discovery_ready": expository or bool(self.state.coverage.get("discovery_ready")),
            "corpus_novelty_ready": expository or bool(self.state.coverage.get("novelty_ready")),
            "prior_art_protocol_ready": expository or bool(self._last_novelty_report.get("ready")),
            "proposal_was_vetted": expository or bool(proposal.get("_vetted")),
            "required_claim_contracts_complete": not claim_contract_gaps,
            "source_grounded_reading_ready": (
                expository or reading["grounded_paper_notes"] >= required_notes),
            "grounded_deep_reading_ready": (
                expository or reading["grounded_deep_notes"] >= required_deep),
            "blind_replication_ready": expository or not replication_gaps,
            "novelty_boundary_certificate_valid": (
                expository or bool(novelty_boundary_validation.get("valid"))),
            "epistemic_obligations_closed": (
                not getattr(self.cfg, "research_obligation_graph", True)
                or bool(obligation_report.get("ready"))),
            "public_deliberation_chain_valid": bool(thought_chain.get("ok")),
            "model_call_chain_valid": bool(model_call_chain.get("ok")),
        }
        ready = all(checks.values())
        report = {
            "round": self.state.round,
            "question": self.state.question,
            "ready": ready,
            "checks": checks,
            "required_claim_count": len(required),
            "relevant_finding_count": len(relevant),
            "qualifying_finding_count": len(qualifying),
            "missing_required": missing_required,
            "weak_required": weak_required,
            "claim_contract_gaps": claim_contract_gaps,
            "qualifying_strengths": sorted(qualifying_strengths),
            "reading": reading,
            "required_grounded_paper_notes": required_notes,
            "required_grounded_deep_notes": required_deep,
            "novelty_report": self._last_novelty_report,
            "replication_gaps": replication_gaps,
            "novelty_boundary_validation": novelty_boundary_validation,
            "obligation_report": obligation_report,
            "thought_chain": thought_chain,
            "model_call_chain": model_call_chain,
        }
        path = self.dir / f"completion-gate-round-{self.state.round}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.state.completion = report
        self._log_thought(
            "completion-gate",
            f"paper readiness {'passed' if ready else 'blocked'}",
            report=report,
        )
        self.obligations.save()
        return report

    # -- the loop ------------------------------------------------------------
    def run(self, max_rounds: int | None = None, token_budget: int | None = None) -> ResearchState:
        if (self.state.status in {"solved", "new_question"}
                and self.state.completion.get("ready")):
            living_manifest = self.dir / "living-paper.json"
            if living_manifest.is_file():
                living = self._living_status()
                if living.get("current"):
                    self._say("resume · living paper and evidence bundle are current")
                    return self.state
                self._say("refresh · living paper is stale · reopening novelty obligations")
                self.state.status = "open"
                self.state.completion["ready"] = False
                self.state.coverage["novelty_ready"] = False
                self._last_novelty_report = {}
                ids = self.state.active_proposal.get("_obligations") or {}
                for key in ("question", "novelty"):
                    node_id = str(ids.get(key) or "")
                    if node_id:
                        self.obligations.set_status(
                            node_id, "in_progress" if key == "question" else "open",
                            reason="living-paper refresh reopened the literature horizon")
            else:
                self._say("resume · completion gate already green · retrying retained paper")
                return self.state
        budget = token_budget or getattr(self.cfg, "run_token_budget", 500_000)
        explicit_token_budget = token_budget is not None
        metered_models = any(
            spec.name in getattr(self.ol, "providers", {})
            for spec in (self.cfg.planner, self.cfg.worker, self.cfg.critic, self.cfg.escalation)
        )
        cats, queries = self.search_plan()           # the right categories + focused queries
        if getattr(self.cfg, "research_information_scheduler", True):
            ranked = self.scheduler.rank_queries(
                queries, research_map=self.map, coverage=self.state.coverage,
                corpus=self.corpus)
            queries = [row["query"] for row in ranked]
        self._say(f"search plan · {('cat: ' + ', '.join(cats) + ' · ') if cats else ''}"
                  + " | ".join(queries))
        # Empirical topics get a separate catalog frontier. Literature coverage and
        # dataset availability are different observables; neither substitutes for the
        # other when deciding whether a candidate question is feasible.
        self.discover_data_resources()
        dry_rounds = 0
        plateau_rounds = 0
        previous_signature = (
            bool(self.state.coverage.get("discovery_ready")),
            bool(self.state.coverage.get("novelty_ready")),
            int(self.state.coverage.get("relevant_paper_count") or 0),
            int((self.state.coverage.get("search") or {}).get("healthy_unique_queries") or 0),
            int((self.state.coverage.get("graph") or {}).get("unresolved_cocitation_holes") or 0),
            int((self.state.coverage.get("graph") or {}).get("closed_current_seed_count") or 0),
        )
        while True:
            if max_rounds is not None and self.state.round >= max_rounds:
                self.state.status = "exhausted"; break
            budget_used = self.state.tokens if explicit_token_budget else self.state.api_tokens
            if (explicit_token_budget or metered_models) and budget_used >= budget:
                self.state.status = "exhausted"; break
            self.state.round += 1
            self._say(f"── round {self.state.round} ──")
            before_corpus = len(self.corpus.papers)
            before_qualifying = {
                f.get("claim_id") for f in self.state.findings
                if f.get("ok") and f.get("strength") in {"formal", "exact", "computational"}
            }

            per = max(3, int(getattr(
                self.cfg, "research_search_results_per_query", 8)))
            if getattr(self.cfg, "research_information_scheduler", True):
                ranked = self.scheduler.rank_queries(
                    queries, research_map=self.map, coverage=self.state.coverage,
                    corpus=self.corpus)
                queries = [row["query"] for row in ranked]
                if ranked:
                    self._say(
                        f"information gain · {ranked[0]['score']:.2f} · {ranked[0]['query'][:60]}")
            # Liveness discipline for the keyword sweep: never re-issue a paraphrase of
            # a tried query (same arXiv answer, one more rate-limit hit), and when the
            # recent record shows arXiv itself failing with nothing retrieved, cool the
            # sweep for a round instead of hammering a dead instrument — the citation
            # graph still runs below.
            health = self._instrument_health()
            cooldown = (
                plateau_rounds >= 1
                and health["recent_searches"] >= 4
                and health["recent_search_results"] == 0
                and health["recent_search_failures"] >= max(
                    2, health["recent_searches"] // 2))
            if cooldown:
                self._say("search cooldown · arXiv failing with zero yield — "
                          "skipping keyword sweep this round")
            else:
                sweep = [q for q in queries if self.state.round == 1
                         or self._query_is_novel(q)]
                skipped = len(queries) - len(sweep)
                if skipped:
                    self._say(f"  skip · {skipped} paraphrase(s) of tried searches")
                for q in sweep:
                    self.gather(q, k=per, categories=cats)
            # DEPTH: snowball the citation graph — pulls in the foundational works many
            # corpus papers cite but keyword search can't reach, until it saturates. This
            # is coverage as an observable, not the model guessing what it's missing.
            graph_seeds = self._graph_seed_batch(limit=30)
            rep = self.corpus.graph_deepen(
                rounds=1, seed_ids=graph_seeds, on=lambda m: self._say(m))
            self._record_graph(rep)
            self._say(f"  corpus · {len(self.corpus.papers)} papers"
                      + (f" · +{rep['added']} via citation graph" if rep["added"] else "")
                      + (" · saturated" if rep["saturated"] else ""))
            conceptual = self.assess_corpus()             # model names conceptual gaps
            coverage = self.evaluate_corpus_quality(      # tools decide retrieval readiness
                stalled_rounds=plateau_rounds)
            self._checkpoint(
                f"round {self.state.round} corpus frontier",
                phase="corpus", papers=len(self.corpus.papers),
                discovery_ready=coverage.get("discovery_ready"),
                novelty_ready=coverage.get("novelty_ready"),
            )

            if self._task_mode() != "expository" and not coverage.get("discovery_ready"):
                next_query = next((str(q).strip() for q in (conceptual.get("searches") or [])
                                   if isinstance(q, str) and q.strip()
                                   and self._query_is_novel(str(q).strip())), "")
                proposal = {
                    "_no_proposal": True,
                    "question": self.state.question,
                    "claims": [],
                    "reasoning": (
                        "Deterministic corpus coverage gate has not passed; angle discovery "
                        "would be premature."
                    ),
                    "missing": coverage.get("blocking_reasons") or [],
                    "next_query": next_query,
                    "coverage": coverage,
                }
                self._say("proposal · waiting for corpus coverage gate")
            else:
                proposal = self.propose()
            if not proposal.get("question") and not proposal.get("claims"):
                proposal["_no_proposal"] = True
                proposal.setdefault("reasoning", "proposal stage returned no grounded question")
            if proposal.get("question"):
                self.state.question = proposal.get("question")
            self.state.active_proposal = proposal
            if proposal.get("question") and proposal.get("claims"):
                self._checkpoint(
                    f"round {self.state.round} proposal",
                    phase="proposal", question=proposal.get("question"),
                    claims=len(proposal.get("claims") or []),
                    vetted=bool(proposal.get("_vetted")),
                )
            findings: list[Finding] = []
            priors = []
            if proposal.get("_no_proposal") or proposal.get("no_proposal"):
                reason = proposal.get("reasoning", "no grounded checkable proposal yet")
                decision = {
                    "assessment": reason,
                    "novel": False,
                    "action": "continue",
                    "next_query": proposal.get("next_query", ""),
                    "reason": reason,
                }
                self._say("reflect · deepen corpus before verification")
            else:
                findings = self.verify_claims(proposal.get("claims", []))
                self.state.findings.extend(asdict(f) for f in findings)
                if (getattr(self.cfg, "research_taste_model", True)
                        and proposal.get("_angle")
                        and any(f.ok and f.strength in {"formal", "exact", "computational"}
                                for f in findings)):
                    self.taste.observe(proposal["_angle"], "verified")
                if self._task_mode() == "expository":
                    self._say("novelty · skipped for expository note")
                else:
                    priors = self.novelty(self.state.question)
                self._say("reflect · supervisor decision")
                decision = self.reflect(findings, priors)
            gate = self.completion_gate(proposal, decision)
            if decision.get("action") in {"solved", "new_question"} and not gate.get("ready"):
                requested = decision.get("action")
                failed = [name for name, ok in gate.get("checks", {}).items() if not ok]
                decision = {
                    **decision,
                    "supervisor_action": requested,
                    "action": "continue",
                    "reason": (
                        f"Supervisor requested {requested}, but deterministic completion gate "
                        f"blocked it: {', '.join(failed)}."
                    ),
                }
                self._say(f"completion · blocked · {', '.join(failed[:3])}")
            self.state.history.append({
                "round": self.state.round, "action": decision.get("action", "continue"),
                "reason": decision.get("reason", ""), "assessment": decision.get("assessment", ""),
            })
            self._log_thought(
                "supervisor-decision",
                f"{decision.get('action','continue')}: {decision.get('reason') or decision.get('assessment','')}",
                decision=decision,
            )
            self._journal(proposal, findings, decision)
            self._save()
            self._save_map()
            self._checkpoint(
                f"round {self.state.round} decision {decision.get('action', 'continue')}",
                phase="round", action=decision.get("action", "continue"),
                qualifying_findings=sum(
                    1 for f in findings
                    if f.ok and f.strength in {"formal", "exact", "computational"}),
                completion_ready=bool(self.state.completion.get("ready")),
            )

            n_ok_round = sum(1 for f in findings if f.ok)
            action = decision.get("action", "continue")
            dry_candidate = bool(
                not proposal.get("claims") and not findings
                and action in {"continue", "pivot"})
            after_qualifying = {
                f.get("claim_id") for f in self.state.findings
                if f.get("ok") and f.get("strength") in {"formal", "exact", "computational"}
            }
            signature = (
                bool(self.state.coverage.get("discovery_ready")),
                bool(self.state.coverage.get("novelty_ready")),
                int(self.state.coverage.get("relevant_paper_count") or 0),
                int((self.state.coverage.get("search") or {}).get("healthy_unique_queries") or 0),
                int((self.state.coverage.get("graph") or {}).get("unresolved_cocitation_holes") or 0),
                int((self.state.coverage.get("graph") or {}).get("closed_current_seed_count") or 0),
            )
            made_progress = (
                len(self.corpus.papers) > before_corpus
                or bool(after_qualifying - before_qualifying)
                or signature != previous_signature
            )
            dry_rounds = dry_rounds + 1 if dry_candidate and not made_progress else 0
            plateau_rounds = 0 if made_progress else plateau_rounds + 1
            previous_signature = signature

            # Terminate ONLY on a genuine, machine-verified result — never on "no question"
            # or an empty round. Everything else means: keep working.
            if action == "solved" and self.state.completion.get("ready"):
                self.state.status = "solved"; break
            if (action == "new_question" and self.state.completion.get("ready")
                    and self.state.question.strip()
                    and self.state.question.strip() != self.state.topic.strip()):
                self.state.status = "new_question"; break

            # Keep going. A dry round changes the angle rather than giving up: re-plan the
            # search and re-derive from what the referee said was missing. A next query
            # that merely paraphrases a tried search is dropped — it would re-ask arXiv
            # the same question and call the identical answer progress.
            nq = (decision.get("next_query") or "").strip()
            if nq and not self._query_is_novel(nq):
                self._say(f"  drop next query · paraphrase of a tried search · {nq[:60]}")
                nq = ""
            if nq:
                queries = [nq]
            elif n_ok_round == 0:
                self._say("  no verified progress this round — re-planning the search angle")
                cats, queries = self.search_plan()
            patience = max(4, int(getattr(self.cfg, "research_plateau_patience", 8)))
            info_plateau = self.scheduler.plateau_report(
                patience=patience,
                floor=float(getattr(self.cfg, "research_information_gain_floor", 0.04)),
                coverage_ready=bool(self.state.coverage.get("discovery_ready")),
            )
            if dry_rounds >= patience and info_plateau.get("exhausted"):
                self.state.status = "exhausted"
                self.state.history.append({
                    "round": self.state.round,
                    "action": "exhausted",
                    "reason": (
                        f"{patience} dry rounds and measured information gain fell below "
                        f"{info_plateau.get('floor')}"),
                    "assessment": "measured dry research plateau",
                })
                self._say("  exhausted · dry rounds and measured search frontier plateau")
                break
            if plateau_rounds >= patience and info_plateau.get("exhausted"):
                self.state.status = "exhausted"
                reason = (
                    f"{patience} consecutive rounds added no papers, no new qualifying findings, "
                    "and no coverage-gate progress; healthy search yield was below the configured floor")
                self.state.history.append({
                    "round": self.state.round,
                    "action": "exhausted",
                    "reason": reason,
                    "assessment": "observable research plateau",
                })
                self._say("  exhausted · observable research plateau")
                break
            if plateau_rounds >= patience * 2:
                self.state.status = "exhausted"
                reason = (
                    f"{patience * 2} rounds made no observable progress; retrieval or verification "
                    "remained blocked, so absence was not inferred")
                self.state.history.append({
                    "round": self.state.round, "action": "exhausted",
                    "reason": reason, "assessment": "external or verifier plateau",
                })
                self._say("  exhausted · prolonged blocked frontier without observable progress")
                break
            self._save()

        self._save()
        return self.state

    def write(self, out_dir: str | None = None, *, author: str = "", association: str = "") -> dict:
        """Compose the paper as a REAL iterative process, not one dump: outline → draft
        each section with the verified findings as backbone → revise for coherence → write
        the abstract LAST (as humans do) → then a compile GATE (a paper that doesn't compile
        isn't done, so LaTeX errors are fed back for repair). The exact machine-checked
        certificates go in verbatim, and a reproducibility bundle (claims.json) is emitted."""
        import json as _json

        from spiral.research_writer import (
            audit_body,
            audit_pdf_layout,
            blueprint_markdown,
            build_document,
            certificate_appendix,
            citation_evidence_packet,
            claim_scope_packet,
            compile_pdf,
            corpus_style_guide,
            corpus_writing_blueprint,
            notation_consistency_report,
            normalise_outline,
            normalise_section_fragment,
            remove_unsupported_claim_sentences,
            remove_suspicious_overlap_sentences,
            reconcile_referee_audit,
            strip_latex_wrappers,
            validate_citation_audit,
            validate_claim_scope_audit,
            validate_model_blueprint,
        )
        if not self.state.completion.get("ready"):
            failed = [name for name, ok in (self.state.completion.get("checks") or {}).items() if not ok]
            raise RuntimeError(
                "paper writing blocked by completion gate"
                + (f": {', '.join(failed)}" if failed else ""))
        expository_paper = self._task_mode() == "expository"
        out = Path(out_dir or (self.dir / "writeup"))
        from spiral.research_quality import rank_papers_for_topic

        papers = rank_papers_for_topic(self.state.question or self.state.topic, self.corpus.papers.values())
        qkey = self._question_key(self.state.question)
        confirmed = [
            f for f in self.state.findings
            if f.get("ok") and self._question_key(str(f.get("question") or "")) == qkey
        ]
        complex_kinds = {"workbench", "groebner", "ideal_membership", "numeric"}
        substantial_research = bool(
            not expository_paper
            and (
                len(self.state.topic.split()) >= 60
                or len(confirmed) >= 3
                or any((f.get("claim") or {}).get("kind") in complex_kinds for f in confirmed)
            )
        )
        min_paper_words = 250 if expository_paper else (1800 if substantial_research else 900)
        target_paper_words = 500 if expository_paper else (3600 if substantial_research else 1800)
        revision_tokens = 3072 if expository_paper else (8192 if substantial_research else 6144)
        body_prompt_chars = 24_000 if expository_paper else 42_000
        revision_think = False
        corpus_digest = self.corpus.summaries(limit=20, chars=600)
        blueprint = corpus_writing_blueprint(papers)
        blueprint_md = blueprint_markdown(blueprint)
        style_guide = corpus_style_guide(papers)
        out.mkdir(parents=True, exist_ok=True)
        paper_obligation_id = self.obligations.ensure(
            "artifact", "Publication-grade paper and reproducibility bundle",
            node_id="artifact:paper", stage="publication", required=True,
            status="in_progress", verifier="paper audits, LaTeX compiler, and proof bundle",
            metadata={
                "writeup": str(out),
                "proof_manifest": str(out / "proof-carrying-manifest.json"),
            },
        )
        self.obligations.link(paper_obligation_id, self.obligations.objective_id, "produces")
        self.obligations.set_status(
            self.obligations.objective_id, "in_progress", reason="publication pass started")
        self.obligations.save()
        (out / "style-guide.md").write_text(style_guide, encoding="utf-8")
        (out / "writing-blueprint.json").write_text(_json.dumps(blueprint, indent=2), encoding="utf-8")
        (out / "writing-blueprint.md").write_text(blueprint_md, encoding="utf-8")
        self._say("write · style guide from corpus")
        findings_txt = "\n".join(
            f"- [{f.get('strength','unverified')} / {f['backend']}] "
            f"{f['claim'].get('note','')}: {f['detail']}"
                                 for f in confirmed) or "(none machine-verified)"

        bp_sys = (
            "Synthesize a strict writing blueprint for this paper from the deterministic "
            "CORPUS BLUEPRINT. Reply ONLY JSON "
            '{"section_arc":["..."],"notation_plan":[{"concept":"...",'
            '"chosen_symbol":"...","definition":"..."}],"vocabulary":["..."],'
            '"equation_conventions":[{"concept":"...","chosen_form":"...",'
            '"source_equation":"...","paper":"arXiv id","required":true}],'
            '"citation_plan":[{"paper":"arXiv id","use":"..."}],'
            '"validation_checks":["..."]}. '
            "Do not invent citations; use only listed corpus ids. Resolve conflicting notation "
            "by choosing one convention and requiring it be stated explicitly. Corpus equations "
            "are examples, not text to copy: set required=true only for a chosen form genuinely "
            "needed in this paper."
        )
        model_blueprint = self._think_json(
            bp_sys,
            f"QUESTION: {self.state.question}\n\nVERIFIED:\n{findings_txt}\n\n"
            f"DETERMINISTIC CORPUS BLUEPRINT:\n{blueprint_md[:12000]}",
            role="planner", max_chars=10_000, reasoning=True)
        if not model_blueprint:
            model_blueprint = {
                "_fallback": True,
                "section_arc": [s.get("name") for s in blueprint.get("section_template", [])],
                "notation_plan": [r for r in blueprint.get("audit_requirements", []) if "notation" in r or "convention" in r],
                "vocabulary": blueprint.get("vocabulary", [])[:18],
                "equation_conventions": [
                    {
                        "concept": "corpus equation idiom",
                        "chosen_form": e.get("equation"),
                        "source_equation": e.get("equation"),
                        "paper": e.get("paper"),
                        "required": False,
                    }
                    for e in blueprint.get("equation_map", [])[:6]
                ],
                "citation_plan": [{"paper": c.get("id"), "use": c.get("use_for", "")}
                                  for c in blueprint.get("citation_anchors", [])[:8]],
                "validation_checks": blueprint.get("audit_requirements", []),
            }
        model_blueprint, blueprint_issues = validate_model_blueprint(model_blueprint, blueprint)
        blueprint["selected_notation"] = model_blueprint.get("notation_plan") or []
        blueprint["selected_equation_conventions"] = (
            model_blueprint.get("equation_conventions") or [])
        blueprint["selected_vocabulary"] = (
            model_blueprint.get("vocabulary") or [])
        blueprint["writing_contract_enforced"] = bool(
            model_blueprint.get("writing_contract_enforced", True))
        (out / "model-writing-blueprint.json").write_text(
            _json.dumps(model_blueprint, indent=2), encoding="utf-8")
        (out / "model-writing-blueprint-audit.json").write_text(
            _json.dumps({"issues": blueprint_issues}, indent=2), encoding="utf-8")
        model_blueprint_txt = _json.dumps(model_blueprint, indent=2)[:9000]
        self._say("write · notation/template blueprint")

        # 1. OUTLINE — the arc of the paper, title chosen (not the prompt)
        o_sys = (("Plan a concise expository mathematical note. " if expository_paper
                  else "Plan a focused arXiv-style original research paper. ") + "Reply ONLY JSON "
                 '{"title":"<concise specific title>","sections":[{"name":"...",'
                 '"rhetorical_role":"introduction|setup|methods|results|proof|discussion|'
                 'conclusion|other","intent":"..."}]}. '
                 "Use an actual section arc observed in the CORPUS STYLE GUIDE as the genre "
                 "template, adapting subject-specific names to this result. Do not flatten every "
                 "paper into generic Introduction/Setup/Results/Discussion/Conclusion headings. "
                 "The complete arc must still provide context, foundations, the VERIFIED "
                 "contribution, and an honest scope/limitations synthesis. The title is plain "
                 "text, concise, specific, and never the raw prompt.")
        outline = self._think_json(
            o_sys, f"QUESTION: {self.state.question}\n\nVERIFIED:\n{findings_txt}\n\n"
            f"CORPUS STYLE GUIDE:\n{style_guide}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\n"
            f"CORPUS:\n{corpus_digest}",
            role="planner", max_chars=12_000, reasoning=True)
        outline = normalise_outline(outline, blueprint)
        title = " ".join((outline.get("title") or self.state.question or self.state.topic).split())[:180]
        sections = outline["sections"]
        blueprint["selected_section_contract"] = sections
        blueprint["selected_rhetorical_contract"] = outline.get(
            "rhetorical_contract") or {}
        model_blueprint["selected_section_contract"] = sections
        model_blueprint_txt = _json.dumps(model_blueprint, indent=2)[:12_000]
        (out / "writing-blueprint.json").write_text(
            _json.dumps(blueprint, indent=2), encoding="utf-8")
        (out / "model-writing-blueprint.json").write_text(
            _json.dumps(model_blueprint, indent=2), encoding="utf-8")
        self._say(f"write · outline · {len(sections)} sections")

        retained_body = out / "body-latest.tex"
        if self.resume_requested and retained_body.is_file():
            body = strip_latex_wrappers(retained_body.read_text(encoding="utf-8"))
            self._say("write · resume retained draft")
        else:
            # 2. DRAFT each section (focused, verified findings as the factual backbone)
            parts = []
            section_target = max(120, target_paper_words // max(1, len(sections)))
            for s in sections:
                s_sys = ("Write ONE LaTeX section (\\section{...} + content; NO preamble/title/abstract) "
                         "for the paper in the OUTLINE. Follow the CORPUS STYLE GUIDE and WRITING "
                         "BLUEPRINT for structure, vocabulary, notation, equation conventions, citation "
                         "anchors, and theorem/proof rhythm, but do not copy corpus sentences. Ground "
                         "the section in the VERIFIED findings; cite corpus papers as \\cite{arXiv:ID}; "
                         "state conventions. Do not add conjectures or claims about removed assumptions, "
                         "failure cases, uniqueness, or completeness unless the VERIFIED/source evidence "
                         "explicitly establishes them. Do not repeat another section's derivation. "
                         f"Aim for roughly {section_target} substantive words, but never pad with unsupported "
                         "claims. LaTeX only.")
                s_usr = (f"OUTLINE title: {title}\nSECTION: {s.get('name')} — {s.get('intent','')}\n\n"
                         f"QUESTION: {self.state.question}\nVERIFIED:\n{findings_txt}\n\n"
                         f"CORPUS STYLE GUIDE:\n{style_guide}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\n"
                         f"CORPUS:\n{corpus_digest}")
                part, _ = self._think(s_sys, s_usr, think=False, role="worker")
                parts.append(normalise_section_fragment(part, s.get("name", "Section")))
                self._say(f"  draft · {s.get('name','section')}")
            body = "\n\n".join(parts)

            # 3. REVISE the assembled draft into one coherent paper
            body, _ = self._think(
                "Revise this assembled LaTeX body into ONE coherent paper: cut cross-section "
                "repetition, fix transitions and notation consistency, keep every \\cite and "
                "equation, follow the CORPUS STYLE GUIDE and WRITING BLUEPRINT, and ensure "
                "the notation plan is explicit before calculations. Preserve every named section "
                "and its order in the selected corpus-shaped outline contract. "
                "Delete duplicated derivations and unsupported extrapolations. Return the full revised LaTeX body only.",
                f"CORPUS STYLE GUIDE:\n{style_guide}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens)
            body = strip_latex_wrappers(body)
            self._say("  revise · coherence pass")

        body, initial_overlap_removals = remove_suspicious_overlap_sentences(
            body, papers)
        if initial_overlap_removals:
            self._say(
                f"  audit · removed {len(initial_overlap_removals)} copied-source sentence(s)")

        det_issues = audit_body(
            body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
        initial_det_issues = list(det_issues)
        deterministic_repair_history = []

        def enforce_deterministic(current_body: str, current_issues: list[str], *,
                                  phase: str, rounds: int = 3):
            history = []
            previous_hash = ""
            for deterministic_round in range(1, rounds + 1):
                if not current_issues:
                    break
                current_hash = hashlib.sha256(current_body.encode("utf-8")).hexdigest()
                use_escalation = (
                    deterministic_round == rounds
                    or (previous_hash and current_hash == previous_hash))
                repair_role = "escalation" if use_escalation else "worker"
                history.append({
                    "phase": phase, "round": deterministic_round,
                    "issues": list(current_issues),
                    "repair_role": repair_role,
                })
                previous_hash = current_hash
                current_body, _ = self._think(
                    "Repair this LaTeX body to satisfy the deterministic paper audit. Return the "
                    "full corrected LaTeX body only; do not add a preamble. If notation/convention "
                    "issues are named, repair the named foundations section before the first "
                    "results/proof section. Preserve every heading, rhetorical role, and order in "
                    "selected_section_contract; do not replace it with a generic five-section arc. Do not "
                    "draft an Appendix; machine certificates are appended later. Use only "
                    "\\cite{arXiv:ID} for citations, never literal [n] markers. Citations support "
                    "background and method lineage; never use them as the proof of a verified result.",
                    "AUDIT ISSUES:\n" + "\n".join(f"- {i}" for i in current_issues)
                    + f"\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\n"
                    + f"VERIFIED:\n{findings_txt}\n\nBODY:\n{current_body[:body_prompt_chars]}",
                    think=False, role=repair_role,
                    context_limit=16_384 if use_escalation else None)
                current_body = strip_latex_wrappers(current_body)
                current_body, overlap_removals = remove_suspicious_overlap_sentences(
                    current_body, papers)
                if overlap_removals:
                    history[-1]["overlap_sentences_removed"] = overlap_removals
                current_issues = audit_body(
                    current_body, papers, confirmed, blueprint=blueprint,
                    min_words=min_paper_words)
                self._say(
                    f"  audit · deterministic repair · {phase} {deterministic_round}")
            return current_body, current_issues, history

        body, det_issues, repair_history = enforce_deterministic(
            body, det_issues, phase="initial")
        deterministic_repair_history.extend(repair_history)
        if det_issues:
            (out / "body-latest.tex").write_text(body, encoding="utf-8")
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "deterministic-structure",
                "initial_overlap_sentences_removed": initial_overlap_removals,
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "final_deterministic_issues": det_issues,
            }, indent=2), encoding="utf-8")
            raise RuntimeError(
                "paper failed deterministic structure gate: " + "; ".join(det_issues))

        mode_rubric = (
            "This is an expository verification note: demand accuracy, clarity, precise assumptions, "
            "and useful background, but do not demand novelty or manufacture broader implications."
            if expository_paper else
            "This is original research: demand a specific contribution, adequate technical depth, "
            "clear comparison with prior art, and publication-grade exposition."
        )
        qa_sys = (
            "You are an arXiv referee for mathematical/theoretical physics writing. "
            "Judge whether the BODY reads like a real paper in this corpus: precise "
            "assumptions, theorem/proof rhythm when appropriate, "
            "citations used for background not decoration, verified claims not overstated, "
            "limitations of the evidence explicit. Do not infer failure outside assumptions or "
            "invent conjectures merely to fill a discussion section. The body must honor the "
            "corpus-shaped selected_section_contract and cover context, foundations, established "
            "contribution, and honest scope; a deterministic checker has already validated its "
            "headings, order, and rhetorical coverage. "
            "Do not flag heading names/order, do not request an Appendix, and do not repeat static "
            "formatting checks; the machine owns those gates and appends certificates separately. "
            "Check mathematical objections against the "
            "supplied evidence before calling a statement false. Do not deliberate inside JSON "
            "fields: return at most five issue strings, each under 300 characters. "
            + mode_rubric + " Reply ONLY JSON "
            '{"verdict":"accept|revise","issues":["..."],"instructions":"..."}.'
        )

        def semantic_referee_result(result: dict, current_body: str) -> dict:
            """Keep deterministic formatting complaints out of the semantic veto lane."""

            current_static = audit_body(
                current_body, papers, confirmed, blueprint=blueprint,
                min_words=min_paper_words)
            return reconcile_referee_audit(
                result, current_static, expository=expository_paper,
                held_citation_ids={paper.bare_id for paper in papers})

        qa_history = []
        qa = {}
        for referee_round in range(1, 3):
            qa = self._think_json(
                qa_sys,
                f"QUESTION: {self.state.question}\n\nCORPUS STYLE GUIDE:\n{style_guide}\n\n"
                f"WRITING BLUEPRINT:\n{model_blueprint_txt}\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                role="critic", max_chars=12_000, max_tokens=1024,
                reasoning=True)
            qa = qa or {
                "verdict": "revise",
                "issues": ["paper referee returned no parseable JSON"],
                "instructions": "tighten structure, claims, and citations conservatively",
            }
            qa = semantic_referee_result(qa, body)
            qa["round"] = referee_round
            qa["stage"] = "fast-exposition-review"
            qa_history.append(qa)
            if qa.get("verdict") == "accept":
                break
            if referee_round == 2:
                break
            body, _ = self._think(
                "Revise the LaTeX body according to the referee issues. Preserve valid citations "
                "and equations; do not invent stronger results than the VERIFIED block supports. "
                f"Maintain at least {min_paper_words} substantive words without filler. "
                "Return the full revised LaTeX body only.",
                f"REFEREE:\n{json.dumps(qa)[:1600]}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens)
            body = strip_latex_wrappers(body)
            post_fast_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            body, post_fast_issues, repair_history = enforce_deterministic(
                body, post_fast_issues, phase=f"post-fast-referee-{referee_round}", rounds=2)
            deterministic_repair_history.extend(repair_history)
            if post_fast_issues:
                qa = {"verdict": "revise", "issues": post_fast_issues,
                      "instructions": "fast-referee repair broke deterministic structure"}
                qa_history.append({**qa, "round": referee_round,
                                   "stage": "post-fast-repair-deterministic-check"})
                break
            self._say(f"  audit · referee revision {referee_round}")

        strong_role = (
            "research_auditor"
            if self.cfg.research_auditor.name != self.cfg.critic.name else "critic")
        for strong_round in range(1, 3):
            qa = self._think_json(
                qa_sys,
                f"AUTHORITATIVE FINAL REVIEW. QUESTION: {self.state.question}\n\n"
                f"CORPUS STYLE GUIDE:\n{style_guide}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\n"
                f"VERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                role=strong_role, max_chars=12_000, max_tokens=2048,
                context_limit=16_384)
            qa = qa or {
                "verdict": "revise", "issues": ["strong referee returned no parseable JSON"],
                "instructions": "paper remains below an auditable publication standard",
            }
            qa = semantic_referee_result(qa, body)
            qa["round"] = strong_round
            qa["stage"] = "strong-publication-review"
            qa_history.append(qa)
            if qa.get("verdict") == "accept":
                break
            if strong_round == 2:
                break
            body, _ = self._think(
                "Repair the full LaTeX body according to the strong referee. Preserve the exact "
                "corpus-shaped section contract, valid \\cite{arXiv:ID} citations, equations, and evidence scope. "
                "Use corpus citations only for background/method lineage, never as the proof of the "
                "paper's result. Confine every theorem/result to the exact VERIFIED setting; remove "
                "broader ring, model, uniqueness, or failure claims unless independently encoded in "
                "VERIFIED. "
                "Do not add an Appendix or literal [n] citations. Return the full body only.",
                f"STRONG REFEREE:\n{json.dumps(qa)[:2200]}\n\nVERIFIED:\n{findings_txt}\n\n"
                f"BODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens)
            body = strip_latex_wrappers(body)
            strong_det_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            body, strong_det_issues, repair_history = enforce_deterministic(
                body, strong_det_issues, phase=f"post-strong-referee-{strong_round}", rounds=2)
            deterministic_repair_history.extend(repair_history)
            if strong_det_issues:
                qa = {"verdict": "revise", "issues": strong_det_issues,
                      "instructions": "strong-referee repair broke deterministic structure"}
                qa_history.append({**qa, "round": strong_round,
                                   "stage": "post-strong-repair-deterministic-check"})
                break

        (out / "body-latest.tex").write_text(body, encoding="utf-8")
        (out / "paper-audit.json").write_text(_json.dumps({
            "stage": "pre-evidence-referee",
            "deterministic_issues": initial_det_issues,
            "deterministic_repair_history": deterministic_repair_history,
            "referee_history": qa_history,
            "referee": qa,
        }, indent=2), encoding="utf-8")
        if qa.get("verdict") != "accept":
            self._say("  audit · pre-evidence referee requested further revision")

        final_issues = audit_body(
            body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
        if final_issues:
            body, _ = self._think(
                "Perform the final deterministic-audit repair on this LaTeX body. Return the "
                "full body only, preserve all valid citations/equations, and do not strengthen claims.",
                "AUDIT ISSUES:\n" + "\n".join(f"- {i}" for i in final_issues)
                + f"\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens)
            body = strip_latex_wrappers(body)
            final_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
        if final_issues:
            (out / "body-latest.tex").write_text(body, encoding="utf-8")
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "post-referee-deterministic",
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "final_deterministic_issues": final_issues,
                "referee": qa,
                "referee_history": qa_history,
            }, indent=2), encoding="utf-8")
            raise RuntimeError("paper failed deterministic content audit: " + "; ".join(final_issues))

        # A real citation key is necessary but not sufficient. Audit each local claim
        # around a citation against exact anchors from the held source and grounded notes.
        citation_system = (
            "You are the citation-support referee. For EVERY citation context, decide whether "
            "the nearby claim is actually supported by one of that paper's supplied exact "
            "source anchors. If evidence is indirect, missing, or merely topically similar, "
            "mark supported false. Copy one supplied source_anchor exactly when supported. "
            "Reply ONLY JSON "
            '{"verdict":"accept|revise","citations":[{"context_id":"...",'
            '"paper":"arXiv id","supported":true,"source_anchor":"exact supplied anchor",'
            '"reason":"..."}]}. '
            "This is an entailment review; do not reward citations merely for existing."
        )

        def citation_review(current_body: str):
            packet = citation_evidence_packet(
                current_body, papers, notes_root=self.dir / "notes")
            if not packet.get("contexts"):
                return packet, {"verdict": "not-applicable", "citations": []}, []
            audit = self._think_json(
                citation_system,
                "CITATION EVIDENCE PACKET:\n" + _json.dumps(packet, ensure_ascii=False)[:22_000],
                role="critic", max_chars=24_000)
            issues = validate_citation_audit(packet, audit, papers)
            retry_history = []
            response_failure_cues = (
                "was not audited",
                "was assigned to the wrong paper",
                "has an invented or unverified source anchor",
            )
            for audit_retry in range(1, 3):
                retry_ids = {
                    match for issue in issues
                    if any(cue in issue for cue in response_failure_cues)
                    for match in re.findall(r"\b[0-9a-f]{20}\b", issue)
                }
                if not retry_ids:
                    break
                retry_contexts = [
                    row for row in packet.get("contexts") or []
                    if row.get("context_id") in retry_ids
                ]
                wanted_papers = {row.get("paper") for row in retry_contexts}
                retry_packet = {
                    **packet,
                    "contexts": retry_contexts,
                    "sources": [
                        row for row in packet.get("sources") or []
                        if row.get("paper") in wanted_papers
                    ],
                }
                retry_result = self._think_json(
                    citation_system,
                    "The prior audit response was structurally incomplete or used an anchor "
                    "outside the supplied packet. Audit EVERY context below and copy an exact "
                    "supplied anchor when supported.\n\nCITATION EVIDENCE PACKET:\n"
                    + _json.dumps(retry_packet, ensure_ascii=False)[:22_000],
                    role="critic", max_chars=24_000, max_tokens=1536)
                rows = {
                    row.get("context_id"): row
                    for row in (audit.get("citations") or [])
                    if isinstance(row, dict) and row.get("context_id")
                }
                rows.update({
                    row.get("context_id"): row
                    for row in (retry_result.get("citations") or [])
                    if isinstance(row, dict) and row.get("context_id")
                })
                audit = {
                    **audit,
                    "citations": list(rows.values()),
                    "verdict": retry_result.get("verdict", audit.get("verdict", "revise")),
                }
                issues = validate_citation_audit(packet, audit, papers)
                retry_history.append({
                    "round": audit_retry,
                    "context_ids": sorted(retry_ids),
                    "remaining_issues": list(issues),
                })
            if retry_history:
                audit["response_retry_history"] = retry_history
            return packet, audit, issues

        citation_packet, citation_audit, citation_issues = citation_review(body)
        citation_repair_history = []
        for citation_round in range(1, 4):
            if not citation_issues and not final_issues:
                break
            citation_repair_history.append({
                "round": citation_round,
                "citation_issues": list(citation_issues),
                "deterministic_issues": list(final_issues),
            })
            body, _ = self._think(
                "Jointly repair this paper's citation and deterministic issues. Retain at least one "
                "held-corpus citation for a genuine background or method-lineage statement, because "
                "the corpus is part of this note. Every retained citation context must be directly "
                "supported by one supplied exact anchor. Never cite the background paper as proof of "
                "the current machine-verified result. Qualify or remove unsupported nearby claims, "
                "but do not delete every citation. Do not invent citations or quotations. Return the "
                "full LaTeX body only.",
                "CITATION ISSUES:\n" + "\n".join(f"- {i}" for i in citation_issues)
                + "\n\nDETERMINISTIC ISSUES:\n"
                + "\n".join(f"- {i}" for i in final_issues)
                + "\n\nEVIDENCE PACKET:\n" + _json.dumps(citation_packet, ensure_ascii=False)[:14_000]
                + f"\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens)
            body = strip_latex_wrappers(body)
            final_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            citation_packet, citation_audit, citation_issues = citation_review(body)
        (out / "body-latest.tex").write_text(body, encoding="utf-8")
        (out / "citation-evidence.json").write_text(
            _json.dumps(citation_packet, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "citation-audit.json").write_text(
            _json.dumps({"audit": citation_audit, "issues": citation_issues,
                         "repair_history": citation_repair_history}, indent=2,
                        ensure_ascii=False), encoding="utf-8")
        if final_issues:
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "citation-repair-deterministic",
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "final_deterministic_issues": final_issues,
                "referee": qa, "referee_history": qa_history,
                "citation_support_issues": citation_issues,
                "citation_repair_history": citation_repair_history,
            }, indent=2), encoding="utf-8")
            raise RuntimeError(
                "citation repair broke deterministic content audit: " + "; ".join(final_issues))
        if citation_issues:
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "citation-support",
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "final_deterministic_issues": final_issues,
                "referee": qa, "referee_history": qa_history,
                "citation_support_issues": citation_issues,
                "citation_repair_history": citation_repair_history,
            }, indent=2), encoding="utf-8")
            raise RuntimeError("paper failed citation-support audit: " + "; ".join(citation_issues))

        novelty_protocol = (
            self._last_novelty_report
            or (self.state.completion.get("novelty_report") or {}))
        protocol_evidence = []
        if novelty_protocol.get("ready"):
            protocol_evidence.append({
                "evidence_id": "protocol:prior-art",
                "kind": "documented bounded prior-art search",
                "scope": (
                    "Supports only a statement that no matching result was located under "
                    "the recorded queries and healthy databases; it does not prove novelty."),
                "queries": novelty_protocol.get("queries") or [],
                "sources": novelty_protocol.get("sources_ok") or [],
                "result_count": novelty_protocol.get("result_count") or 0,
            })
        if self.state.coverage.get("novelty_ready"):
            protocol_evidence.append({
                "evidence_id": "protocol:corpus-coverage",
                "kind": "deterministic corpus coverage audit",
                "scope": (
                    "Supports the recorded corpus counts, retrieval health, query-route "
                    "coverage, and current citation-frontier closure only."),
                "queries": (self.state.coverage.get("search") or {}).get("queries") or [],
                "sources": ["arxiv", "semantic_scholar_citation_graph"],
                "result_count": self.state.coverage.get("paper_count") or 0,
            })

        scope_system = (
            "You are the adversarial claim-scope referee. Audit EVERY enumerated sentence. "
            "Reply ONLY JSON "
            '{"claims":[{"claim_id":"...","status":"nonclaim|verified|source_supported|'
            'protocol_supported|qualified|unsupported|contradicted",'
            '"evidence_id":"finding:...|citation:...|protocol:...|"}]}. '
            "Return exactly these three fields per item and no explanations. "
            "Use verified only when the sentence is a logical consequence of that exact encoded "
            "finding. Use source_supported only for a listed citation context. "
            "Use protocol_supported only for an explicitly bounded sentence saying what the "
            "documented searches did or did not locate; never use it for absolute novelty. "
            "A purely performative sentence that defines notation or declares the paper's scope "
            "may be nonclaim only when packet.performative is true. "
            "A result proved "
            "under assumptions never establishes necessity of those assumptions or failure when "
            "one is removed. A CAS check does not establish an unstated no-go, uniqueness, "
            "classification, novelty, or physical interpretation. Mark those unsupported or "
            "contradicted. Do not omit any claim_id."
        )

        def scope_review(text: str, current_citations: dict):
            from spiral.research_writer import (
                claims_requiring_escalation,
                merge_claim_scope_audits,
            )

            packet = claim_scope_packet(
                text, confirmed, current_citations, protocol_evidence)
            claims = packet.get("claims") or []

            def audit_batches(rows: list[dict], *, role: str, batch_size: int,
                              max_tokens: int) -> dict:
                audit = {"claims": []}
                for start in range(0, len(rows), batch_size):
                    part = {**packet, "claims": rows[start:start + batch_size]}
                    result = self._think_json(
                        scope_system,
                        "CLAIM-SCOPE PACKET:\n"
                        + _json.dumps(part, ensure_ascii=False)[:22_000],
                        role=role, max_chars=24_000, max_tokens=max_tokens,
                        context_limit=(
                            16_384 if role in {"escalation", "research_auditor"}
                            else None))
                    audit["claims"].extend(result.get("claims") or [])
                return audit

            fast = audit_batches(claims, role="critic", batch_size=12, max_tokens=2048)
            escalated = claims_requiring_escalation(packet, fast)
            different_model = self.cfg.research_auditor.name != self.cfg.critic.name
            if escalated and different_model:
                strong = audit_batches(
                    escalated, role="research_auditor", batch_size=8, max_tokens=1536)
                escalated_ids = {row["claim_id"] for row in escalated}
                merged = merge_claim_scope_audits(fast, strong, escalated_ids)
            else:
                strong = {"claims": []}
                escalated_ids = set()
                merged = fast
            merged["review_protocol"] = {
                "fast_model": self.cfg.critic.name,
                "strong_model": (
                    self.cfg.research_auditor.name if different_model else "same-as-fast"),
                "strong_review_claim_ids": sorted(escalated_ids),
                "strong_review_count": len(escalated_ids),
            }
            issues = validate_claim_scope_audit(packet, merged)
            retry_history = []
            response_failure_cues = (
                "was not audited",
                "cites invalid verified evidence",
                "cites invalid source evidence",
                "cites invalid protocol evidence",
                "has invalid status",
                "was labelled nonclaim",
            )
            retry_role = "research_auditor" if different_model else "critic"
            for audit_retry in range(1, 3):
                retry_ids = {
                    match for issue in issues
                    if any(cue in issue for cue in response_failure_cues)
                    for match in re.findall(r"\b[0-9a-f]{20}\b", issue)
                }
                if not retry_ids:
                    break
                retry_rows = [
                    row for row in claims if row.get("claim_id") in retry_ids
                ]
                retry_result = audit_batches(
                    retry_rows, role=retry_role, batch_size=6, max_tokens=1536)
                rows = {
                    row.get("claim_id"): row
                    for row in (merged.get("claims") or [])
                    if isinstance(row, dict) and row.get("claim_id") not in retry_ids
                }
                rows.update({
                    row.get("claim_id"): row
                    for row in (retry_result.get("claims") or [])
                    if isinstance(row, dict) and row.get("claim_id")
                })
                merged["claims"] = list(rows.values())
                issues = validate_claim_scope_audit(packet, merged)
                retry_history.append({
                    "round": audit_retry,
                    "claim_ids": sorted(retry_ids),
                    "remaining_issues": list(issues),
                })
            if retry_history:
                merged["review_protocol"]["response_retry_history"] = retry_history
            return packet, merged, issues

        def repair_evidence_regressions(
            current_body: str,
            current_det_issues: list[str],
            current_citation_packet: dict,
            current_citation_audit: dict,
            current_citation_issues: list[str],
            current_scope_packet: dict,
            current_scope_audit: dict,
            current_scope_issues: list[str],
            *,
            phase: str,
            rounds: int = 3,
        ):
            """Repair a prose mutation, then rerun every objective content gate.

            Referee and compile repairs are untrusted model output. A single recheck can
            detect a regression but cannot recover from it; this bounded loop gives the
            writer a chance to delete the exact disputed prose and converge safely.
            """
            seen_states: set[tuple] = set()
            for evidence_round in range(1, rounds + 1):
                if not (current_det_issues or current_citation_issues
                        or current_scope_issues):
                    break
                state_key = (
                    hashlib.sha256(current_body.encode("utf-8")).hexdigest(),
                    tuple(sorted(current_det_issues)),
                    tuple(sorted(current_citation_issues)),
                    tuple(sorted(current_scope_issues)),
                )
                if state_key in seen_states:
                    self._say(f"  audit · no repair progress · {phase}")
                    break
                seen_states.add(state_key)
                issue_ids = {
                    match for issue in current_scope_issues
                    for match in re.findall(r"\b[0-9a-f]{20}\b", issue)
                }
                disputed = [
                    row for row in (current_scope_packet.get("claims") or [])
                    if row.get("claim_id") in issue_ids
                ]
                citation_repair_history.append({
                    "phase": phase,
                    "round": evidence_round,
                    "citation_issues": list(current_citation_issues),
                    "claim_scope_issues": list(current_scope_issues),
                    "deterministic_issues": list(current_det_issues),
                })
                current_body, _ = self._think(
                    "Repair the exact evidence regressions introduced by a later prose edit. "
                    "Delete unsupported statements instead of generalizing. Retain at least one "
                    "held-corpus citation only for a background or method statement directly "
                    "supported by a supplied exact anchor; never cite it as proof of the current "
                    "result. Do not follow a referee request to add a theorem, counterexample, "
                    "necessity claim, textbook citation, or broader setting absent from VERIFIED. "
                    "Preserve the exact corpus-shaped section contract and verified equations. Return the full LaTeX "
                    "body only.",
                    "DETERMINISTIC ISSUES:\n"
                    + "\n".join(f"- {i}" for i in current_det_issues)
                    + "\n\nCITATION ISSUES:\n"
                    + "\n".join(f"- {i}" for i in current_citation_issues)
                    + "\n\nCLAIM-SCOPE ISSUES:\n"
                    + "\n".join(f"- {i}" for i in current_scope_issues)
                    + "\n\nEXACT DISPUTED SENTENCES:\n"
                    + _json.dumps(disputed, ensure_ascii=False, indent=2)[:10_000]
                    + "\n\nCITATION EVIDENCE PACKET:\n"
                    + _json.dumps(current_citation_packet, ensure_ascii=False)[:14_000]
                    + f"\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{current_body[:body_prompt_chars]}",
                    think=revision_think, role="planner", num_predict=revision_tokens,
                    max_chars=36_000)
                current_body = strip_latex_wrappers(current_body)
                current_det_issues = audit_body(
                    current_body, papers, confirmed, blueprint=blueprint,
                    min_words=min_paper_words)
                if current_det_issues:
                    current_body, current_det_issues, history = enforce_deterministic(
                        current_body, current_det_issues,
                        phase=f"{phase}-evidence-{evidence_round}", rounds=2)
                    deterministic_repair_history.extend(history)
                (current_citation_packet, current_citation_audit,
                 current_citation_issues) = citation_review(current_body)
                (current_scope_packet, current_scope_audit,
                 current_scope_issues) = scope_review(
                    current_body, current_citation_packet)
                self._say(
                    f"  audit · evidence regression repair · {phase} {evidence_round}")
                (out / "body-latest.tex").write_text(current_body, encoding="utf-8")
            return (
                current_body, current_det_issues,
                current_citation_packet, current_citation_audit,
                current_citation_issues, current_scope_packet,
                current_scope_audit, current_scope_issues,
            )

        scope_history = []
        scope_packet = {}
        scope_audit = {}
        scope_issues = []
        for scope_round in range(1, 6):
            scope_packet, scope_audit, scope_issues = scope_review(body, citation_packet)
            scope_history.append({
                "round": scope_round,
                "packet": scope_packet,
                "audit": scope_audit,
                "issues": scope_issues,
            })
            (out / f"claim-scope-audit-round-{scope_round}.json").write_text(
                _json.dumps(scope_history[-1], indent=2, ensure_ascii=False), encoding="utf-8")
            if not scope_issues:
                break

            cleaned_body, scope_deletions = remove_unsupported_claim_sentences(
                body, scope_packet, scope_audit)
            if scope_deletions:
                cleaned_det_issues = audit_body(
                    cleaned_body, papers, confirmed, blueprint=blueprint,
                    min_words=min_paper_words)
                (cleaned_citation_packet, cleaned_citation_audit,
                 cleaned_citation_issues) = citation_review(cleaned_body)
                if not cleaned_det_issues and not cleaned_citation_issues:
                    body = cleaned_body
                    final_issues = cleaned_det_issues
                    citation_packet = cleaned_citation_packet
                    citation_audit = cleaned_citation_audit
                    citation_issues = cleaned_citation_issues
                    scope_packet, scope_audit, scope_issues = scope_review(
                        body, citation_packet)
                    scope_history[-1]["deterministic_deletions"] = scope_deletions
                    scope_history[-1]["post_deletion_issues"] = list(scope_issues)
                    (out / "body-latest.tex").write_text(body, encoding="utf-8")
                    (out / f"claim-scope-audit-round-{scope_round}.json").write_text(
                        _json.dumps(scope_history[-1], indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    self._say(
                        f"  audit · removed {len(scope_deletions)} unsupported sentence(s)")
                    if not scope_issues:
                        break
            if scope_round == 5:
                break
            issue_ids = {
                match for issue in scope_issues
                for match in re.findall(r"\b[0-9a-f]{20}\b", issue)
            }
            disputed = [
                row for row in (scope_packet.get("claims") or [])
                if row.get("claim_id") in issue_ids
            ]
            body, _ = self._think(
                "Repair this paper so every substantive assertion stays within the verified "
                "findings or exact source-supported citation contexts. Remove false extrapolations, "
                "especially claims that an identity fails when an unneeded assumption is removed. "
                "Do not replace unsupported claims with new speculation. Delete a disputed sentence "
                "when it is not needed. When a replacement is necessary, use only a direct restatement "
                "of the encoded finding or a performative setup sentence. Never add guarantees about "
                "universality, precision, reproducibility, traceability, or independent implementations. "
                "Preserve the exact corpus-shaped section contract, valid citations, and equations. If the paper is already "
                f"above {min_paper_words} substantive words, prefer deletion and do not pad it. Return "
                "the full LaTeX body only.",
                "CLAIM-SCOPE ISSUES:\n" + "\n".join(f"- {i}" for i in scope_issues)
                + "\n\nEXACT DISPUTED SENTENCES (the ids above map to these):\n"
                + _json.dumps(disputed, ensure_ascii=False, indent=2)[:12_000]
                + "\n\nAUDIT:\n" + _json.dumps(scope_audit, ensure_ascii=False)[:10_000]
                + f"\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens,
                max_chars=32_000)
            body = strip_latex_wrappers(body)
            final_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            if final_issues:
                body, final_issues, repair_history = enforce_deterministic(
                    body, final_issues, phase=f"post-claim-scope-{scope_round}", rounds=2)
                deterministic_repair_history.extend(repair_history)
            citation_packet, citation_audit, citation_issues = citation_review(body)
            if final_issues or citation_issues:
                for post_scope_citation_round in range(1, 4):
                    citation_repair_history.append({
                        "phase": f"post-claim-scope-{scope_round}",
                        "round": post_scope_citation_round,
                        "citation_issues": list(citation_issues),
                        "deterministic_issues": list(final_issues),
                    })
                    body, _ = self._think(
                        "Repair the citation and deterministic regressions introduced by the "
                        "claim-scope rewrite. Keep at least one exact-anchor-supported citation "
                        "for background or method lineage; never cite it as proof of the current "
                        "result. Preserve the narrower evidence scope. Return the full LaTeX body only.",
                        "CITATION ISSUES:\n" + "\n".join(f"- {i}" for i in citation_issues)
                        + "\n\nDETERMINISTIC ISSUES:\n"
                        + "\n".join(f"- {i}" for i in final_issues)
                        + "\n\nEVIDENCE PACKET:\n"
                        + _json.dumps(citation_packet, ensure_ascii=False)[:14_000]
                        + f"\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                        think=revision_think, role="planner", num_predict=revision_tokens,
                        max_chars=32_000)
                    body = strip_latex_wrappers(body)
                    final_issues = audit_body(
                        body, papers, confirmed, blueprint=blueprint,
                        min_words=min_paper_words)
                    citation_packet, citation_audit, citation_issues = citation_review(body)
                    if not final_issues and not citation_issues:
                        break
                if final_issues or citation_issues:
                    break

        (out / "body-latest.tex").write_text(body, encoding="utf-8")
        (out / "citation-evidence.json").write_text(
            _json.dumps(citation_packet, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "citation-audit.json").write_text(
            _json.dumps({"audit": citation_audit, "issues": citation_issues}, indent=2,
                        ensure_ascii=False), encoding="utf-8")
        (out / "claim-scope-audit.json").write_text(
            _json.dumps({"packet": scope_packet, "audit": scope_audit,
                         "issues": scope_issues, "history": scope_history},
                        indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "paper-audit.json").write_text(_json.dumps({
            "initial_deterministic_issues": initial_det_issues,
            "deterministic_repair_history": deterministic_repair_history,
            "final_deterministic_issues": final_issues,
            "referee": qa,
            "referee_history": qa_history,
            "citation_support_issues": citation_issues,
            "claim_scope_issues": scope_issues,
            "claim_scope_history": scope_history,
        }, indent=2), encoding="utf-8")

        if final_issues:
            raise RuntimeError(
                "claim-scope repair broke deterministic content audit: " + "; ".join(final_issues))
        if citation_issues:
            raise RuntimeError(
                "claim-scope repair broke citation-support audit: " + "; ".join(citation_issues))
        if scope_issues:
            raise RuntimeError("paper failed claim-scope audit: " + "; ".join(scope_issues))

        final_qa = {}
        rejected_final_candidates = []
        last_revalidation_issues = []
        # Three bounded semantic repairs plus one final review-only assessment.
        for final_review_round in range(1, 5):
            final_qa = self._think_json(
                qa_sys,
                f"QUESTION: {self.state.question}\n\nCORPUS STYLE GUIDE:\n{style_guide}\n\n"
                f"WRITING BLUEPRINT:\n{model_blueprint_txt}\n\nVERIFIED:\n{findings_txt}\n\n"
                f"BODY AFTER CITATION/CLAIM REPAIRS:\n{body[:body_prompt_chars]}",
                role=strong_role, max_chars=12_000, max_tokens=1536,
                context_limit=16_384)
            final_qa = final_qa or {
                "verdict": "revise", "issues": ["final referee returned no parseable JSON"],
                "instructions": "paper remains below an auditable publication standard",
            }
            final_qa = semantic_referee_result(final_qa, body)
            final_qa["round"] = final_review_round
            final_qa["stage"] = "post-evidence-authoritative-review"
            qa_history.append(final_qa)
            qa = final_qa
            if final_qa.get("verdict") == "accept" or final_review_round == 4:
                break

            good_body = body
            good_final_issues = list(final_issues)
            good_citation_packet = citation_packet
            good_citation_audit = citation_audit
            good_citation_issues = list(citation_issues)
            good_scope_packet = scope_packet
            good_scope_audit = scope_audit
            good_scope_issues = list(scope_issues)

            candidate_body, _ = self._think(
                "Apply the authoritative referee's semantic revisions to this full LaTeX body. "
                "The deterministic layout and exact evidence audits were already green. Preserve "
                "the core roles and valid background citations. The VERIFIED block is the hard "
                "scope boundary: remove every broader theorem, generalization, novelty claim, or "
                "interpretive consequence that it does not encode. Keep every theorem/result domain "
                "no broader than the exact encoded finding. A prior failed candidate's validation "
                "errors are constraints to avoid, not claims to repeat. Return the full body only.",
                "FINAL REFEREE:\n" + _json.dumps(final_qa, ensure_ascii=False)[:5000]
                + "\n\nPRIOR REJECTED-CANDIDATE ERRORS:\n"
                + "\n".join(f"- {issue}" for issue in last_revalidation_issues[:12])
                + f"\n\nVERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                think=revision_think, role="planner", num_predict=revision_tokens,
                max_chars=30_000)
            body = strip_latex_wrappers(candidate_body)
            final_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            body, final_issues, repair_history = enforce_deterministic(
                body, final_issues, phase=f"post-final-referee-{final_review_round}", rounds=2)
            deterministic_repair_history.extend(repair_history)
            citation_packet, citation_audit, citation_issues = citation_review(body)
            scope_packet, scope_audit, scope_issues = scope_review(body, citation_packet)
            scope_history.append({
                "round": f"post-final-referee-{final_review_round}",
                "packet": scope_packet, "audit": scope_audit, "issues": scope_issues,
            })
            if final_issues or citation_issues or scope_issues:
                (
                    body, final_issues, citation_packet, citation_audit,
                    citation_issues, scope_packet, scope_audit, scope_issues,
                ) = repair_evidence_regressions(
                    body, final_issues, citation_packet, citation_audit,
                    citation_issues, scope_packet, scope_audit, scope_issues,
                    phase=f"post-final-referee-{final_review_round}", rounds=3)
                scope_history.append({
                    "round": f"post-final-referee-{final_review_round}-repaired",
                    "packet": scope_packet, "audit": scope_audit,
                    "issues": scope_issues,
                })
            (out / "body-latest.tex").write_text(body, encoding="utf-8")
            (out / "citation-evidence.json").write_text(
                _json.dumps(citation_packet, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "citation-audit.json").write_text(
                _json.dumps({"audit": citation_audit, "issues": citation_issues}, indent=2,
                            ensure_ascii=False), encoding="utf-8")
            (out / "claim-scope-audit.json").write_text(
                _json.dumps({"packet": scope_packet, "audit": scope_audit,
                             "issues": scope_issues, "history": scope_history},
                            indent=2, ensure_ascii=False), encoding="utf-8")
            if final_issues or citation_issues or scope_issues:
                last_revalidation_issues = list(
                    (final_issues + citation_issues + scope_issues)[:20])
                rejected_final_candidates.append({
                    "round": final_review_round,
                    "referee": final_qa,
                    "revalidation_issues": last_revalidation_issues,
                    "candidate_sha256": hashlib.sha256(
                        body.encode("utf-8")).hexdigest(),
                })
                body = good_body
                final_issues = good_final_issues
                citation_packet = good_citation_packet
                citation_audit = good_citation_audit
                citation_issues = good_citation_issues
                scope_packet = good_scope_packet
                scope_audit = good_scope_audit
                scope_issues = good_scope_issues
                self._say("  audit · rejected final-referee candidate; restored green draft")
            else:
                last_revalidation_issues = []
            (out / "body-latest.tex").write_text(body, encoding="utf-8")
            (out / "citation-evidence.json").write_text(
                _json.dumps(citation_packet, indent=2, ensure_ascii=False),
                encoding="utf-8")
            (out / "citation-audit.json").write_text(
                _json.dumps({"audit": citation_audit, "issues": citation_issues},
                            indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "claim-scope-audit.json").write_text(
                _json.dumps({"packet": scope_packet, "audit": scope_audit,
                             "issues": scope_issues, "history": scope_history},
                            indent=2, ensure_ascii=False), encoding="utf-8")

        (out / "paper-audit.json").write_text(_json.dumps({
            "stage": "post-evidence-authoritative-review",
            "initial_deterministic_issues": initial_det_issues,
            "deterministic_repair_history": deterministic_repair_history,
            "final_deterministic_issues": final_issues,
            "referee": qa,
            "referee_history": qa_history,
            "citation_support_issues": citation_issues,
            "claim_scope_issues": scope_issues,
            "claim_scope_history": scope_history,
            "rejected_final_candidates": rejected_final_candidates,
        }, indent=2), encoding="utf-8")
        if final_issues or citation_issues or scope_issues:
            raise RuntimeError(
                "final referee repair failed evidence revalidation: "
                + "; ".join((final_issues + citation_issues + scope_issues)[:8]))
        if final_qa.get("verdict") != "accept":
            raise RuntimeError("paper failed final referee gate after evidence repairs")

        # 4. ABSTRACT last, from the finished body
        abstract, _ = self._think(
            "Write a 4-7 sentence abstract for this finished paper: the question, what was "
            "established and its exact evidence scope, why it matters. Do not mention failure "
            "outside assumptions, novelty, completeness, or conjectures unless VERIFIED. Match the mathematical "
            "register in the CORPUS STYLE GUIDE. Plain text, no \\begin{abstract}.",
            f"TITLE: {title}\n\nCORPUS STYLE GUIDE:\n{style_guide}\n\nWRITING BLUEPRINT:\n{model_blueprint_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
            think=False, role="planner")

        abstract_history = []
        abstract_packet = {}
        abstract_audit = {}
        abstract_issues = []
        for abstract_round in range(1, 4):
            abstract = strip_latex_wrappers(abstract).strip()
            abstract_packet, abstract_audit, abstract_issues = scope_review(abstract, {})
            abstract_history.append({
                "round": abstract_round, "packet": abstract_packet,
                "audit": abstract_audit, "issues": abstract_issues,
            })
            (out / "abstract-latest.txt").write_text(abstract + "\n", encoding="utf-8")
            (out / "abstract-claim-scope-audit.json").write_text(
                _json.dumps({"packet": abstract_packet, "audit": abstract_audit,
                             "issues": abstract_issues, "history": abstract_history},
                            indent=2, ensure_ascii=False), encoding="utf-8")
            if not abstract_issues or abstract_round == 3:
                break
            issue_ids = {
                match for issue in abstract_issues
                for match in re.findall(r"\b[0-9a-f]{20}\b", issue)
            }
            disputed = [
                row for row in (abstract_packet.get("claims") or [])
                if row.get("claim_id") in issue_ids
            ]
            abstract, _ = self._think(
                "Rewrite this abstract so every result claim is exactly supported by the VERIFIED "
                "findings. Remove claims about failure outside assumptions, completeness, novelty, "
                "uniqueness, or future consequences unless independently established. A documented "
                "search may be described only as a bounded search result. Plain text only.",
                "ABSTRACT CLAIM-SCOPE ISSUES:\n"
                + "\n".join(f"- {i}" for i in abstract_issues)
                + "\n\nEXACT DISPUTED SENTENCES:\n"
                + _json.dumps(disputed, ensure_ascii=False, indent=2)[:6000]
                + f"\n\nVERIFIED:\n{findings_txt}\n\nABSTRACT:\n{abstract}",
                think=False, role="worker", num_predict=2048)
        if abstract_issues:
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "abstract-claim-scope",
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "referee": qa, "referee_history": qa_history,
                "citation_support_issues": citation_issues,
                "claim_scope_issues": scope_issues,
                "claim_scope_history": scope_history,
                "abstract_claim_scope_history": abstract_history,
            }, indent=2), encoding="utf-8")
            raise RuntimeError("abstract failed claim-scope audit: " + "; ".join(abstract_issues))

        # 5. BUILD + COMPILE GATE — a paper that doesn't compile isn't done
        appendix = certificate_appendix(confirmed)
        tex = build_document(title, abstract, body, papers, out,
                             author=author, association=association, appendix=appendix)
        toolchain_attempt = {}
        if not any(shutil.which(name) for name in ("latexmk", "tectonic", "pdflatex")):
            toolchain_attempt = {
                "attempted": False,
                "request": "brew tectonic",
                "result": "automatic research tooling is disabled",
            }
            if getattr(self.cfg, "research_tool_auto", True) and shutil.which("brew"):
                from spiral.command_broker import CommandBroker

                self._say("  compile · provisioning isolated publication toolchain")
                broker = CommandBroker(self.dir, self.cfg)
                detail = broker.provision(
                    "brew tectonic",
                    timeout=max(900, int(getattr(self.cfg, "verify_timeout", 900))),
                )
                toolchain_attempt = {
                    "attempted": True,
                    "request": "brew tectonic",
                    "result": detail,
                }
        (out / "publication-toolchain.json").write_text(
            _json.dumps(toolchain_attempt, indent=2), encoding="utf-8")
        pdf, errs = compile_pdf(tex)
        compile_history = [{
            "round": 0, "compiled": bool(pdf), "error": errs[:1200],
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }]
        for compile_round in range(1, 3):
            if pdf or not errs:
                break
            self._say(f"  compile · repairing LaTeX ({errs.splitlines()[0][:48] if errs else ''})")
            body, _ = self._think(
                "This LaTeX fails to compile with the ERRORS shown. Return the corrected full "
                "LaTeX body (sections only, no preamble); fix ONLY what breaks compilation.",
                f"ERRORS:\n{errs}\n\nBODY:\n{body[:body_prompt_chars]}", think=False, role="worker")
            body = strip_latex_wrappers(body)
            final_issues = audit_body(
                body, papers, confirmed, blueprint=blueprint, min_words=min_paper_words)
            citation_packet, citation_audit, citation_issues = citation_review(body)
            scope_packet, scope_audit, scope_issues = scope_review(body, citation_packet)
            scope_history.append({
                "round": f"compile-repair-{compile_round}",
                "packet": scope_packet, "audit": scope_audit, "issues": scope_issues,
            })
            repair_qa = self._think_json(
                qa_sys,
                f"AUTHORITATIVE REVIEW AFTER LATEX REPAIR. QUESTION: {self.state.question}\n\n"
                f"VERIFIED:\n{findings_txt}\n\nBODY:\n{body[:body_prompt_chars]}",
                role=strong_role, max_chars=12_000, max_tokens=2048,
                context_limit=16_384)
            repair_qa = repair_qa or {
                "verdict": "revise", "issues": ["no parseable post-compile-repair review"],
                "instructions": "do not accept an unaudited content mutation",
            }
            repair_qa = semantic_referee_result(repair_qa, body)
            repair_qa.update({
                "round": f"compile-repair-{compile_round}",
                "stage": "post-compile-repair-authoritative-review",
            })
            qa_history.append(repair_qa)
            qa = repair_qa
            compile_history.append({
                "round": compile_round, "compiled": False, "error_before": errs[:1200],
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "deterministic_issues": final_issues,
                "citation_issues": citation_issues,
                "claim_scope_issues": scope_issues,
                "referee": repair_qa,
            })
            (out / "body-latest.tex").write_text(body, encoding="utf-8")
            (out / "citation-evidence.json").write_text(
                _json.dumps(citation_packet, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "citation-audit.json").write_text(
                _json.dumps({"audit": citation_audit, "issues": citation_issues}, indent=2,
                            ensure_ascii=False), encoding="utf-8")
            (out / "claim-scope-audit.json").write_text(
                _json.dumps({"packet": scope_packet, "audit": scope_audit,
                             "issues": scope_issues, "history": scope_history},
                            indent=2, ensure_ascii=False), encoding="utf-8")
            if (final_issues or citation_issues or scope_issues
                    or repair_qa.get("verdict") != "accept"):
                (out / "paper-audit.json").write_text(_json.dumps({
                    "stage": "post-compile-repair-content-gates",
                    "initial_deterministic_issues": initial_det_issues,
                    "deterministic_repair_history": deterministic_repair_history,
                    "final_deterministic_issues": final_issues,
                    "referee": qa, "referee_history": qa_history,
                    "citation_support_issues": citation_issues,
                    "claim_scope_issues": scope_issues,
                    "claim_scope_history": scope_history,
                    "abstract_claim_scope_history": abstract_history,
                    "compile_history": compile_history,
                }, indent=2), encoding="utf-8")
                raise RuntimeError(
                    "LaTeX repair failed content revalidation: "
                    + "; ".join((final_issues + citation_issues + scope_issues)[:8]
                                or repair_qa.get("issues") or ["authoritative referee rejected repair"]))
            tex = build_document(title, abstract, body, papers, out,
                                 author=author, association=association, appendix=appendix)
            pdf, errs = compile_pdf(tex)
            compile_history[-1]["compiled"] = bool(pdf)
            compile_history[-1]["error_after"] = errs[:1200]

        if not pdf:
            (out / "paper-audit.json").write_text(_json.dumps({
                "stage": "latex-compilation",
                "initial_deterministic_issues": initial_det_issues,
                "deterministic_repair_history": deterministic_repair_history,
                "final_deterministic_issues": final_issues,
                "referee": qa, "referee_history": qa_history,
                "citation_support_issues": citation_issues,
                "claim_scope_issues": scope_issues,
                "claim_scope_history": scope_history,
                "abstract_claim_scope_history": abstract_history,
                "compile_history": compile_history,
                "toolchain_attempt": toolchain_attempt,
            }, indent=2), encoding="utf-8")
            detail = errs[:500] if errs else (
                "no LaTeX engine is available after the recorded provisioning attempt")
            raise RuntimeError(f"paper failed LaTeX compilation: {detail}")

        notation_audit = notation_consistency_report(body, blueprint)
        (out / "notation-audit.json").write_text(
            _json.dumps(notation_audit, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if not notation_audit.get("ready"):
            raise RuntimeError(
                "paper failed notation consistency gate: "
                + "; ".join(notation_audit.get("issues") or []))

        layout_audit = audit_pdf_layout(
            pdf, render_dir=out / "rendered-pages")
        (out / "publication-layout.json").write_text(
            _json.dumps(layout_audit, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if not layout_audit.get("ready"):
            raise RuntimeError(
                "paper failed publication layout gate: "
                + "; ".join(layout_audit.get("issues") or []))

        visual_review = self._local_vision_json(
            (
                "You are a mathematical journal production editor. Inspect rendered paper "
                "pages for observable publication defects only: clipped or overlapping text, "
                "illegible equations, broken figures/tables, poor hierarchy, inconsistent "
                "margins, stranded headings, visibly excessive whitespace, or unprofessional "
                "typesetting. Do not judge mathematical correctness from images. Return JSON "
                'only: {"verdict":"accept|revise","summary":"...",'
                '"issues":[{"severity":"major|minor","page":1,"evidence":"...",'
                '"fix":"..."}]}.'
            ),
            (
                f"TITLE: {title}\nPAGES: {layout_audit.get('pages')}\n"
                "Review only what is visibly supported by the supplied page images. This is "
                "an advisory aesthetic layer; deterministic geometry and evidence gates are "
                "reported separately."
            ),
            layout_audit.get("rendered_pages") or [],
        ) if getattr(self.cfg, "visual_review", True) else {}
        if not visual_review:
            visual_review = {
                "verdict": "unavailable",
                "summary": "no installed local vision model produced a review",
                "issues": [],
                "_scope": "advisory and local-only",
            }
        (out / "publication-visual-review.json").write_text(
            _json.dumps(visual_review, indent=2, ensure_ascii=False),
            encoding="utf-8")

        (out / "paper-audit.json").write_text(_json.dumps({
            "stage": "complete",
            "initial_deterministic_issues": initial_det_issues,
            "deterministic_repair_history": deterministic_repair_history,
            "final_deterministic_issues": final_issues,
            "referee": qa, "referee_history": qa_history,
            "citation_support_issues": citation_issues,
            "claim_scope_issues": scope_issues,
            "claim_scope_history": scope_history,
            "abstract_claim_scope_history": abstract_history,
            "compile_history": compile_history,
            "notation_audit": notation_audit,
            "layout_audit": layout_audit,
            "visual_review": visual_review,
            "fresh_pdf": True,
        }, indent=2), encoding="utf-8")

        # 6. reproducibility bundle
        (out / "claims.json").write_text(_json.dumps(confirmed, indent=2), encoding="utf-8")
        (out / "completion-gate.json").write_text(
            _json.dumps(self.state.completion, indent=2), encoding="utf-8")
        (out / "corpus-coverage.json").write_text(
            _json.dumps(self.state.coverage, indent=2), encoding="utf-8")
        (out / "novelty-protocol.json").write_text(
            _json.dumps(novelty_protocol, indent=2), encoding="utf-8")
        # 7. Publication obligations, immutable evidence checkpoint, and proof envelope.
        paper_evidence_path = str(pdf or tex)
        self.obligations.add_evidence(
            paper_obligation_id,
            "Compiled paper and all deterministic/semantic audits passed",
            evidence_kind="publication_evidence", artifact=paper_evidence_path,
            verifier="LaTeX compiler and paper audit pipeline",
            relation="supports", status="supported",
            metadata={
                "tex": str(tex), "pdf": str(pdf) if pdf else "",
                "paper_audit": str(out / "paper-audit.json"),
                "citation_audit": str(out / "citation-audit.json"),
                "claim_scope_audit": str(out / "claim-scope-audit.json"),
                "notation_audit": str(out / "notation-audit.json"),
                "layout_audit": str(out / "publication-layout.json"),
                "visual_review": str(out / "publication-visual-review.json"),
            },
            node_id="publication-evidence:paper",
        )
        self.obligations.set_status(
            paper_obligation_id, "supported",
            reason="paper compiled and every writing/evidence gate passed",
            verifier="paper pipeline",
        )
        self.obligations.set_status(
            self.obligations.objective_id, "supported",
            reason="verified result was released as a proof-carrying paper",
            verifier="publication obligation graph",
        )
        self.obligations.save()
        evidence_commit = self._checkpoint(
            "audited paper candidate",
            phase="publication", title=title, verified=len(confirmed),
            publication_ready=self.obligations.report("publication").get("ready"),
        )

        from spiral.research_provenance import LivingPaper, ProofCarryingPaper
        from spiral.research_quality import verify_jsonl_hash_chain

        audit_files = [
            out / "paper-audit.json", out / "citation-evidence.json",
            out / "citation-audit.json", out / "claim-scope-audit.json",
            out / "abstract-claim-scope-audit.json", out / "claims.json",
            out / "completion-gate.json", out / "corpus-coverage.json",
            out / "novelty-protocol.json", out / "writing-blueprint.json",
            out / "model-writing-blueprint.json",
            out / "notation-audit.json",
            out / "publication-layout.json",
            out / "publication-visual-review.json",
            out / "publication-toolchain.json",
            self.dir / "thoughts.jsonl", self.dir / "model-calls.jsonl",
            self.obligations.events_path,
        ]
        proof = ProofCarryingPaper.build(
            out,
            findings=confirmed,
            paper_files=[tex, *([pdf] if pdf else [])],
            audit_files=[path for path in audit_files if Path(path).is_file()],
            novelty_certificate=(
                self.state.novelty_boundary.get("path", "") if not expository_paper else ""),
            obligation_graph=self.obligations.path,
            obligation_report=self.obligations.report("result"),
            citation_packet=citation_packet,
            scope_packet={
                **scope_packet,
                "claims": [
                    *(scope_packet.get("claims") or []),
                    *[
                        {**row, "location": "abstract"}
                        for row in (abstract_packet.get("claims") or [])
                    ],
                ],
            },
            scope_audit={
                "claims": [
                    *(scope_audit.get("claims") or []),
                    *(abstract_audit.get("claims") or []),
                ],
            },
            research_commit=evidence_commit,
            require_novelty=not expository_paper,
            require_replication=not expository_paper,
            integrity_chains={
                "thoughts": verify_jsonl_hash_chain(self.dir / "thoughts.jsonl"),
                "model_calls": verify_jsonl_hash_chain(self.dir / "model-calls.jsonl"),
                "obligations": self.obligations.verify_event_chain(),
            },
        )
        proof_validation = ProofCarryingPaper.validate(proof["path"])
        if not proof.get("valid") or not proof_validation.get("valid"):
            raise RuntimeError(
                "proof-carrying paper bundle failed: "
                + "; ".join(proof_validation.get("issues") or [
                    name for name, ok in proof.get("checks", {}).items() if not ok]))

        living = {}
        if getattr(self.cfg, "research_living_papers", True):
            living = LivingPaper.create(
                self.dir, out,
                topic=self.state.topic,
                question=self.state.question,
                proof_manifest=proof["path"],
                novelty_certificate=(
                    self.state.novelty_boundary.get("path", "")
                    if not expository_paper else proof["path"]),
                obligation_graph=self.obligations.path,
                corpus_manifest=self.corpus._manifest(),
                research_commit=evidence_commit,
                recheck_days=int(getattr(self.cfg, "research_living_recheck_days", 30)),
            )
            self.state.living_paper = living
        self.state.obligation_report = self.obligations.report("publication")
        self._save()
        self._save_map()
        release_commit = self._checkpoint(
            "proof-carrying living paper release",
            phase="release", title=title,
            proof_manifest=proof.get("manifest_sha256"),
        )
        if release_commit:
            self.state.research_commit = release_commit
        self._save()
        self._save_map()
        self._say("write · bundle complete")
        return {"tex": str(tex), "pdf": str(pdf) if pdf else None,
                "title": title, "verified": len(confirmed),
                "style": str(out / "style-guide.md"),
                "blueprint": str(out / "writing-blueprint.md"),
                "audit": str(out / "paper-audit.json"),
                "citation_audit": str(out / "citation-audit.json"),
                "claim_scope_audit": str(out / "claim-scope-audit.json"),
                "notation_audit": str(out / "notation-audit.json"),
                "proof_manifest": proof["path"],
                "living_paper": living.get("path") if living else None,
                "research_commit": self.state.research_commit}
