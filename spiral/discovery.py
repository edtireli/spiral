"""Novelty as a structural gap between two literatures, found by counting.

The observation this implements is not mine and not the model's. Swanson, in 1986,
noticed that one literature reported that fish oil lowers blood viscosity, another
reported that Raynaud's patients suffer from high blood viscosity, and *no paper stated
the connection* — the implication was already public, and unread, because nobody read
both literatures. He called it undiscovered public knowledge, and the pattern is
mechanical: A relates to B, B relates to C, and A–C appears nowhere.

That is exactly the move a research loop cannot make by asking a model for "a novel
angle". A model asked to free-associate inside one corpus returns that corpus's own
consensus, which is why five rounds of searching a well-worked field returned five
verdicts of "known". The gap is a property of the *co-occurrence structure* of two
document sets, so it can be computed, and the computation is the proposal.

The contract is spiral's own, with the roles the useful way round: **the code finds the
structurally missing link, and the model is asked only to say whether it means
anything.** A bridge nobody has crossed is a fact about the corpora. Whether crossing it
is interesting is a judgment, and stays one.

Association is measured by **conditional probability** — of the documents mentioning A,
what fraction also mention B — and generic vocabulary is excluded by a document-frequency
band rather than by a correlation threshold.

That combination is deliberate, and the first version got it wrong. Pointwise mutual
information is the obvious choice and it fails here for a structural reason: the A-term
is ubiquitous in its own corpus *by construction*, because the corpus was fetched about
it. With A in 5 of 6 documents, PMI against A is negative for every B in the corpus,
independence being all a saturated term can manage. Checked against Swanson's own
example, the measure scored the known-correct answer below zero and returned nothing.
Conditional probability has no such blind spot, and the terms PMI was there to suppress
are removed instead by dropping anything that appears in more than half the documents —
a word in every paper distinguishes nothing, whatever its correlation.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Bridge:
    """One B-term linking a source concept to a distant one."""
    term: str
    a_strength: float          # P(b | a) in the source corpus
    c_strength: float          # P(c | b) in the distant corpus
    support: int               # documents backing the weaker of the two links

    @property
    def strength(self) -> float:
        """The chain is as strong as its weaker link — a bridge held up at one end
        is not a bridge."""
        return min(self.a_strength, self.c_strength)


@dataclass
class Connection:
    """A candidate A–C link that no single document in either corpus states."""
    source: str                # A — a concept of the problem at hand
    distant: str               # C — a concept from the other literature
    bridges: list[Bridge] = field(default_factory=list)
    score: float = 0.0
    a_documents: int = 0
    c_documents: int = 0

    def sentence(self) -> str:
        vias = ", ".join(b.term for b in self.bridges[:3])
        return (f"{self.source} ←→ {self.distant} (via {vias}; score {self.score:.3f}, "
                f"no document states it)")

    def as_dict(self) -> dict:
        return {
            "source": self.source, "distant": self.distant,
            "score": round(self.score, 4),
            "bridges": [{"term": b.term, "a": round(b.a_strength, 3),
                         "c": round(b.c_strength, 3), "support": b.support}
                        for b in self.bridges[:5]],
            "a_documents": self.a_documents, "c_documents": self.c_documents,
        }


class Associations:
    """Term co-occurrence over a document set.

    ``min_df`` drops terms seen in a single document — one appearance is an anecdote,
    and its conditional probabilities are all 0 or 1. ``max_df_ratio`` drops terms seen
    in more than half of them, which is what removes "model", "theory" and the rest of
    the vocabulary that appears everywhere and therefore distinguishes nothing. The
    band is doing the job PMI was meant to do, without PMI's blind spot for the very
    term the corpus was collected about.
    """

    def __init__(self, documents: list[list[str]], *, min_df: int = 2,
                 max_df_ratio: float = 0.5):
        self.total = len(documents or [])
        self.df: Counter = Counter()
        self._pairs: dict[frozenset, int] = defaultdict(int)
        self._with: dict[str, set] = defaultdict(set)
        for doc in documents or []:
            terms = sorted(set(doc or []))
            self.df.update(terms)
            for i, a in enumerate(terms):
                for b in terms[i + 1:]:
                    self._pairs[frozenset((a, b))] += 1
        ceiling = max(min_df, math.floor(max_df_ratio * self.total)) if self.total else 0
        # Two different sets, and conflating them broke this twice. `vocabulary` is the
        # BRIDGE set — terms specific enough to carry a connection, so the upper bound
        # applies. `attested` is every term with real support, and endpoints are drawn
        # from it: each corpus's own subject saturates that corpus by construction, so
        # a band applied to endpoints throws away both the concept being asked about
        # and the one worth discovering.
        self.vocabulary = {t for t, n in self.df.items() if min_df <= n <= ceiling}
        self.attested = {t for t, n in self.df.items() if n >= min_df}
        for key in self._pairs:
            a, b = tuple(key)
            if a in self.attested and b in self.attested:
                self._with[a].add(b)
                self._with[b].add(a)

    def cooccurrence(self, a: str, b: str) -> int:
        return self._pairs.get(frozenset((a, b)), 0)

    def confidence(self, given: str, then: str) -> float:
        """P(then | given) — of the documents mentioning ``given``, the fraction that
        also mention ``then``. Asymmetric on purpose: the question is what accompanies
        this concept, not how surprising the pair is."""
        base = self.df.get(given, 0)
        return (self.cooccurrence(given, then) / base) if base else 0.0

    def lift(self, a: str, b: str) -> float:
        """P(a,b) / P(a)P(b) — reported as a diagnostic, not used as a gate. Above 1 is
        positive association; the frequency band already removes the terms that would
        otherwise sit near 1 with huge support."""
        da, db = self.df.get(a, 0), self.df.get(b, 0)
        if not (da and db and self.total):
            return 0.0
        return (self.cooccurrence(a, b) / self.total) / ((da / self.total) * (db / self.total))

    def partners(self, term: str) -> set:
        return self._with.get(term, set())


def find_connections(
    source_documents: list[list[str]],
    distant_documents: list[list[str]],
    *,
    seeds: list[str] | None = None,
    distant_endpoints: set[str] | None = None,
    min_strength: float = 0.20,
    min_bridges: int = 2,
    limit: int = 12,
) -> list[Connection]:
    """Swanson-style open discovery across two literatures.

    ``source_documents`` is the problem's own corpus (A-side), ``distant_documents`` the
    other field being brought to bear (C-side). ``seeds`` restricts the A-terms to the
    concepts the run actually cares about; without it every term in the corpus is a
    candidate and the result is dominated by generic vocabulary.

    ``seeds`` and ``distant_endpoints`` should be the terms each side is *distinctive*
    for, measured against a background — but the documents passed in must NOT be
    pre-filtered to those terms, and the distinction is the whole architecture. An
    endpoint earns its place by being specific to its corpus; a bridge earns its place
    by being shared. Filtering both sides down to their distinctive vocabularies first
    left {coset, gauged} facing {anti-de sitter, holography} with an empty intersection
    and no bridge could exist. Specificity picks the endpoints, the frequency band picks
    the bridges, and each measure does only the job it is good at.

    A connection survives only if it is genuinely absent: A and C must never co-occur in
    ANY document of EITHER corpus. A pair that already appears together is not a
    discovery, it is a citation — and this is the whole test, so it is applied to both
    sides rather than only to the corpus that happened to be searched first.
    """
    a_side = Associations(source_documents)
    c_side = Associations(distant_documents)

    # endpoints come from `attested`, bridges from the banded `vocabulary`.
    # `seeds if seeds is not None` rather than `seeds or …`: an EMPTY seed list means
    # the caller found no distinctive source concepts, and falling back to the whole
    # vocabulary there turns "nothing to ask about" into "ask about everything" —
    # which is how `acad cienc` and `algorithms` became research concepts.
    chosen = seeds if seeds is not None else sorted(a_side.attested)
    a_terms = [t for t in chosen if t in a_side.attested]
    shared = a_side.vocabulary & c_side.vocabulary
    if not a_terms or not shared:
        return []

    # anything already said, anywhere, is not a discovery
    def stated_together(a: str, c: str) -> bool:
        return (a_side.cooccurrence(a, c) > 0) or (c_side.cooccurrence(a, c) > 0)

    found: list[Connection] = []
    for a in a_terms:
        # B must bridge: characteristic of A's context here, and present in the other
        # literature. `a_side.partners(a)` is empty when A is above the frequency band,
        # which a seed usually is, so the neighbourhood is taken from the shared
        # vocabulary directly.
        bridges_of_a = {b: a_side.confidence(a, b) for b in shared if b != a}
        bridges_of_a = {b: s for b, s in bridges_of_a.items() if s >= min_strength}
        if not bridges_of_a:
            continue
        # C is reached through those bridges, in the distant corpus only
        reachable: dict[str, list[Bridge]] = defaultdict(list)
        for b, a_strength in bridges_of_a.items():
            for c in c_side.partners(b):
                if c == a or c == b or c in shared:
                    continue
                if distant_endpoints is not None and c not in distant_endpoints:
                    continue
                c_strength = c_side.confidence(b, c)
                if c_strength < min_strength:
                    continue
                reachable[c].append(Bridge(
                    term=b, a_strength=a_strength, c_strength=c_strength,
                    support=min(a_side.cooccurrence(a, b), c_side.cooccurrence(b, c))))
        for c, bridges in reachable.items():
            if len(bridges) < min_bridges or stated_together(a, c):
                continue
            bridges.sort(key=lambda x: x.strength, reverse=True)
            # independent bridges are the evidence, so the score adds them — one very
            # strong bridge is a coincidence, four moderate ones are a pattern
            score = sum(b.strength for b in bridges[:5])
            found.append(Connection(
                source=a, distant=c, bridges=bridges, score=score,
                a_documents=a_side.df.get(a, 0), c_documents=c_side.df.get(c, 0)))

    found.sort(key=lambda x: (-x.score, x.source, x.distant))
    return found[:limit]


def brief(connections: list[Connection], *, source_field: str = "this problem",
          distant_field: str = "the adjacent literature") -> str:
    """The candidate links, as prompt text for the angle proposer.

    Handed over as *structure*, not as claims: every line says what co-occurs with what
    and what does not, and the model is asked to judge whether the gap means anything.
    Stating them as findings would be the fabrication this whole pipeline refuses.
    """
    if not connections:
        return ""
    lines = [
        f"STRUCTURAL GAPS between {source_field} and {distant_field}. Each pair below "
        "is connected through shared intermediate concepts, yet NO paper in either "
        "corpus mentions both endpoints together. This is a fact about the "
        "literatures, not a claim that the connection is real or useful:",
    ]
    for c in connections:
        vias = ", ".join(f"{b.term} ({b.strength:+.2f})" for b in c.bridges[:4])
        lines.append(
            f"- {c.source}  ←→  {c.distant}\n"
            f"    bridged by: {vias}\n"
            f"    {c.source} appears in {c.a_documents} source docs, {c.distant} in "
            f"{c.c_documents} distant docs, together in none")
    lines.append(
        "Judge each one. Most will be coincidences of vocabulary — say so and discard "
        "them. For any that is a real implication, state the specific checkable claim "
        "it implies and how it would be falsified. Do NOT treat a gap as evidence.")
    return "\n".join(lines)
