"""spiralʳᵉˢᵉᵃʳᶜʰ — the iterative research conductor.

The research analogue of spiral's build Conductor. A round is:

  gather  → read primary sources into the corpus
  propose → a local reasoning model states a research question and a set of *checkable
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

import json
import re
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


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (they wrap it in prose/fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        try:                              # tolerate trailing commas / single quotes
            return json.loads(re.sub(r",\s*([}\]])", r"\1", frag.replace("'", '"')))
        except Exception:
            return {}


class ResearchLoop:
    def __init__(self, topic: str, workdir="./spiral-research", cfg=None, ol=None, ui=None):
        from spiral.config import Config
        from spiral.llm import Ollama
        from spiral.research_corpus import Corpus
        self.cfg = cfg or Config.load()
        self.ol = ol or Ollama(self.cfg.base_url, providers=getattr(self.cfg, "providers", None))
        self.ui = ui
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.corpus = Corpus(self.dir / "corpus")
        self.state = self._load() or ResearchState(topic=topic)

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
            return st
        return None

    def _save(self):
        self._statefile().write_text(json.dumps(asdict(self.state), indent=2))

    def _say(self, msg: str):
        if self.ui:
            self.ui(msg)

    # -- llm -----------------------------------------------------------------
    def _think(self, system: str, user: str, think: bool = True) -> tuple[str, int]:
        res = self.ol.chat(
            self.cfg.planner.name,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            think=think, num_predict=self.cfg.planner_max_tokens,
            num_ctx=self.cfg.planner.num_ctx, keep_alive=self.cfg.keep_alive, temperature=0.4,
        )
        self.state.tokens += getattr(res, "completion_tokens", 0) or 0
        return res.text, getattr(res, "completion_tokens", 0) or 0

    # -- phases --------------------------------------------------------------
    def gather(self, query: str, k: int = 8) -> int:
        self._say(f"gather · arXiv: {query[:60]}")
        added = self.corpus.build(query, k=k, on=lambda a: self._say(f"  + {a}"))
        for p in added:
            if p.bare_id not in self.state.corpus_ids:
                self.state.corpus_ids.append(p.bare_id)
        return len(added)

    _CLAIM_SPEC = (
        'Claims must be machine-verifiable, one of: '
        '{"kind":"identity","lhs":"<sympy>","rhs":"<sympy>","note":"..."}, '
        '{"kind":"solution","equation":"<expr=expr>","var":"x","value":"<sympy>","note":"..."}, '
        '{"kind":"numeric","code":"<python printing True/False last line>","note":"..."}, '
        '{"kind":"theorem","statement":"<Lean thm sig>","proof":"<Lean tactics, e.g. by decide>","note":"..."}. '
        "Use sympy syntax (** powers, * products, pi, I, exp, sin). A theorem claim is the "
        "strongest — prefer it when the statement is a clean formal proposition."
    )

    def _draft_proposal(self) -> dict:
        system = (
            "You are a theoretical-research engine. Read the CORPUS and TOPIC and propose "
            "ONE concrete, tractable research question not obviously already solved, plus "
            "CHECKABLE claims that would answer it. Reply ONLY JSON: "
            '{"question":"...","reasoning":"...","claims":[...]}. ' + self._CLAIM_SPEC
        )
        user = (f"TOPIC: {self.state.topic}\nOPEN QUESTION: {self.state.question or '(none yet)'}"
                f"\n\nCORPUS:\n{self.corpus.summaries()}")
        return _extract_json(self._think(system, user)[0])

    def _critique_proposal(self, proposal: dict, priors: list) -> dict:
        """Vet the *proposal* against prior art + rigor BEFORE spending verification on
        it — the research analogue of a referee. Steers away from re-deriving 1962."""
        from spiral.citations import Prior, novelty_digest
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
                f"{novelty_digest([Prior(**p) for p in priors])}")
        return _extract_json(self._think(system, user)[0])

    def _refine_proposal(self, proposal: dict, critique: dict, priors: list) -> dict:
        system = (
            "Revise the PROPOSAL to fix the referee's ISSUES: make it genuinely novel "
            "(distinct from the prior art), sharper, and more clearly checkable. Keep what "
            "worked. Reply ONLY JSON {\"question\":\"...\",\"reasoning\":\"...\",\"claims\":[...]}. "
            + self._CLAIM_SPEC
        )
        user = (f"PROPOSAL: {json.dumps(proposal)[:1500]}\n\nREFEREE: {json.dumps(critique)[:800]}\n\n"
                f"STEER AWAY FROM: {', '.join(p.get('title','') for p in priors[:6])}")
        return _extract_json(self._think(system, user)[0])

    def propose(self, refine_rounds: int = 2) -> dict:
        """Draft a proposal, then iterate it against prior art + a referee critique until
        it is accepted or the refinement budget runs out — so what reaches verification is
        already vetted for novelty and rigor, not the model's first guess."""
        from spiral.citations import prior_art
        proposal = self._draft_proposal()
        for _ in range(max(0, refine_rounds)):
            q = proposal.get("question", "")
            if not q:
                break
            priors = prior_art(q, k=6, physics=True)
            critique = self._critique_proposal(proposal, priors)
            self._say(f"  refine · {critique.get('verdict','?')} · {critique.get('novelty','')[:40]}")
            if critique.get("verdict") == "accept":
                proposal["_vetted"] = True
                break
            refined = self._refine_proposal(proposal, critique, priors)
            if refined.get("question"):
                proposal = refined
        return proposal

    def verify_claims(self, claims: list) -> list[Finding]:
        from spiral.numeric_lab import check_numeric_claim
        from spiral.verify_math import verify
        out = []
        for c in claims or []:
            kind = str(c.get("kind", "")).lower()
            if kind == "numeric":
                r = check_numeric_claim(c.get("code", ""))
                fnd = Finding(c, r.ok, "numeric", (r.error or r.stdout)[:200], self.state.round)
            else:
                v = verify(c)
                fnd = Finding(c, v.ok, v.backend, v.detail, self.state.round)
            self._say(f"  {'✓' if fnd.ok else '✗'} [{fnd.backend}] {c.get('note', kind)[:50]}")
            out.append(fnd)
        return out

    def novelty(self, question: str) -> list:
        from spiral.citations import prior_art
        self._say("novelty · searching prior art")
        priors = prior_art(question, k=8, physics=True)
        return [asdict(p) for p in priors]

    def reflect(self, verified: list[Finding], priors: list) -> dict:
        from spiral.citations import Prior, novelty_digest
        confirmed = [f for f in verified if f.ok]
        system = (
            "You are the research supervisor. Given the QUESTION, the VERIFIED claims "
            "(machine-checked — trust these), the REFUTED claims, and PRIOR ART, decide "
            "the next action. Reply with ONLY JSON: "
            '{"assessment":"...","novel":true|false,'
            '"action":"continue|solved|new_question|pivot",'
            '"next_query":"<arxiv search to deepen the corpus, if continuing>",'
            '"reason":"..."}. '
            "'solved' only if the confirmed claims actually answer the question AND prior "
            "art does not already contain it. 'new_question' if the work instead surfaced a "
            "verified-open question worth pursuing. Be honest: unverified is not solved."
        )
        digest = novelty_digest([Prior(**p) for p in priors])
        user = (f"QUESTION: {self.state.question}\n\n"
                f"VERIFIED:\n" + "\n".join(f"- [{f.backend}] {f.claim.get('note','')}: {f.detail}" for f in confirmed) +
                f"\n\nREFUTED:\n" + "\n".join(f"- {f.claim.get('note','')}: {f.detail}" for f in verified if not f.ok) +
                f"\n\n{digest}")
        text, _ = self._think(system, user)
        return _extract_json(text)

    # -- the loop ------------------------------------------------------------
    def run(self, max_rounds: int | None = None, token_budget: int | None = None) -> ResearchState:
        budget = token_budget or getattr(self.cfg, "run_token_budget", 500_000)
        query = self.state.topic
        while True:
            if max_rounds is not None and self.state.round >= max_rounds:
                self.state.status = "exhausted"; break
            if self.state.tokens >= budget:
                self.state.status = "exhausted"; break
            self.state.round += 1
            self._say(f"── round {self.state.round} ──")

            self.gather(query, k=8)
            proposal = self.propose()
            if proposal.get("question"):
                self.state.question = proposal["question"]
            findings = self.verify_claims(proposal.get("claims", []))
            self.state.findings.extend(asdict(f) for f in findings)
            priors = self.novelty(self.state.question)
            decision = self.reflect(findings, priors)
            self.state.history.append({
                "round": self.state.round, "action": decision.get("action", "continue"),
                "reason": decision.get("reason", ""), "assessment": decision.get("assessment", ""),
            })
            self._save()

            action = decision.get("action", "continue")
            if action in ("solved", "new_question"):
                self.state.status = action; break
            query = decision.get("next_query") or self.state.question or self.state.topic
            if not any(f.ok for f in findings) and self.state.round >= 3:
                # three rounds, nothing survives verification → stop honestly
                self.state.status = "exhausted"; break

        self._save()
        return self.state

    def write(self, out_dir: str | None = None) -> dict:
        """Compose + compile the cited LaTeX write-up of the current state."""
        from spiral.research_writer import build_document, compile_pdf
        out = Path(out_dir or (self.dir / "writeup"))
        confirmed = [f for f in self.state.findings if f.get("ok")]
        system = (
            "Write the body of a short arXiv-style LaTeX paper (sections + equations, NO "
            "preamble, NO \\begin{document}) reporting the QUESTION and the machine-VERIFIED "
            "findings. Cite corpus papers as \\cite{arXiv:ID}. State honestly what is proven "
            "vs conjectured. Return LaTeX only."
        )
        user = (f"QUESTION: {self.state.question}\nSTATUS: {self.state.status}\n\n"
                f"VERIFIED FINDINGS:\n" + "\n".join(f"- {f['detail']}" for f in confirmed) +
                f"\n\nCORPUS:\n{self.corpus.summaries(limit=15, chars=500)}")
        body, _ = self._think(system, user, think=False)
        title = self.state.question or self.state.topic
        abstract = (self.state.history[-1]["assessment"] if self.state.history else "")
        papers = list(self.corpus.papers.values())
        tex = build_document(title, abstract, body, papers, out)
        pdf = compile_pdf(tex)
        return {"tex": str(tex), "pdf": str(pdf) if pdf else None}
