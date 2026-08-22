"""Writing style as a measurement, not an opinion.

Two jobs, one instrument:

1. **Mine a template from a corpus.** Given the papers a research question actually
   lives among, extract *quantitative* targets — section order, words per section,
   sentence length, hedging density, passive rate, citations and equations per 1000
   words, field vocabulary. The existing style mining in ``research_writer`` extracts
   only categorical things (which sections exist, which macros are defined); nothing
   there tells a writer they are hedging three times as often as the field does.

2. **Detect AI tells.** Wikipedia's "Signs of AI writing" catalogues the specific
   phrases and structures that mark machine prose: significance inflation
   ("stands as a testament to"), copula avoidance ("serves as" for "is"), negative
   parallelism ("not only X but Y"), the rule of three, promotional adjectives,
   vague attribution, and the "Challenges and Future Prospects" ending.

Both are deterministic: same text in, same numbers out. A model may *rewrite*, but only
these counters decide whether the rewrite moved toward the target. That keeps the
project's contract — the model proposes, the tool decides — in a domain where the
temptation to let a model grade its own prose is strongest.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

# ── AI tells, taken from Wikipedia: Signs of AI writing ──────────────────────
# Each entry: (id, human explanation, compiled pattern). Kept as data so the list can
# grow without touching the scorer.
_TELLS: list[tuple[str, str, re.Pattern]] = [
    ("significance-inflation",
     "inflates importance instead of stating facts",
     re.compile(r"\b(?:stands?|serves?)\s+as\b|\bis\s+a\s+testament\b|"
                r"\b(?:crucial|pivotal|vital)\s+role\b|\bunderscor\w+\s+the\s+importance\b|"
                r"\breflects?\s+(?:a\s+)?broader\b|\bmark(?:s|ing)?\s+a\s+(?:pivotal|key|significant)\b|"
                r"\bmarks?\s+a\s+(?:(?:clear|major|important|key|significant)\s+)?milestone\b|"
                r"\bprovides?\s+(?:a\s+)?(?:clear|major|important|key|significant)\s+milestone\b|"
                r"\b(?:provides?|establishes?|defines?)\s+(?:a\s+)?"
                r"(?:reference\s+point|foundation|basis|baseline)\b|"
                r"\boutlines?\s+(?:the\s+)?next\s+steps?\b|"
                r"\bkey\s+turning\s+point\b|\bevolving\s+landscape\b|\bsetting\s+the\s+stage\b", re.I)),
    ("ai-vocabulary",
     "words that cluster heavily in machine prose",
     re.compile(r"\b(?:deep\s+dive|delve|delves|delving|tapestry|testament|pivotal|meticulous(?:ly)?|"
                r"robust(?:ly|ness)?|vibrant|showcas(?:e|es|ing)|underscor(?:e|es|ing)|"
                r"foster(?:s|ing)?|garner(?:s|ed|ing)?|bolster(?:s|ed|ing)?|"
                r"intricate(?:ly)?|interplay|realm|landscape\s+of|myriad|"
                r"crucial(?:ly)?|enduring|holistic|nuanced|leverag(?:e|es|ing))\b", re.I)),
    ("copula-avoidance",
     "replaces plain 'is/are' with inflated verbs",
     re.compile(r"\b(?:serves?\s+as|stands?\s+as|functions?\s+as|represents?|boasts?|"
                r"embodies|encapsulates)\b", re.I)),
    # Wikipedia catalogues THREE distinct negative parallelisms. An earlier version of
    # this file implemented only the first and silently missed the other two, which are
    # the ones that show up most in polished machine prose.
    ("negative-parallelism",
     "'not only X but also Y' — the additive negative parallelism",
     re.compile(r"\bnot\s+(?:only|just|merely|simply)\b[^.!?]{0,80}?\b(?:but|yet)\b", re.I)),
    ("not-x-but-y",
     "'not X, but Y' framing — asserts by contrast instead of stating",
     re.compile(
         # "not a mirror but a portal", "not a representation, but an act"
         r"\bnot\s+(?:a|an|the|his|her|its|their)\s+[\w-]+(?:\s+[\w-]+){0,4}?\s*,?\s*"
         r"\bbut\s+(?:a|an|the|rather|instead)\b|"
         # "It's not X, it's Y"
         r"\bit(?:'s|’s|\s+is|\s+was)\s+not\s+[^.!?]{0,60}?,\s*it(?:'s|’s|\s+is|\s+was)\b|"
         # "no X, no Y, just Z"
         r"\bno\s+[\w-]+\s*,\s*no\s+[\w-]+\s*,\s*(?:just|only|simply)\b", re.I)),
    ("x-rather-than-y",
     "'X rather than Y' framing, typically after a gerund",
     re.compile(r"\b\w+ing\b[^.!?]{0,60}?\brather\s+than\b|"
                r"\brather\s+than\s+(?:a|an|the)?\s*\w+(?:ing|ity|ism|ness|tion)\b", re.I)),
    ("superficial-analysis",
     "trailing '-ing' clause asserting unattributed significance",
     re.compile(r",\s+(?:highlighting|underscoring|emphasizing|reflecting|symbolizing|"
                r"showcasing|demonstrating|ensuring|contributing\s+to|cultivating|"
                r"fostering|solidifying|cementing)\b", re.I)),
    ("promotional",
     "marketing adjectives in place of description",
     re.compile(r"\b(?:nestled|in\s+the\s+heart\s+of|groundbreaking|renowned|"
                r"diverse\s+array|rich\s+(?:history|tapestry|tradition|array)|"
                r"natural\s+beauty|state[- ]of[- ]the[- ]art|cutting[- ]edge)\b", re.I)),
    ("vague-attribution",
     "attributes claims to unnamed authorities",
     re.compile(r"\b(?:industry\s+reports?|observers?\s+have|experts?\s+(?:argue|say|agree|note)|"
                r"some\s+critics?\s+argue|it\s+is\s+widely\s+(?:believed|regarded|considered)|"
                r"studies\s+have\s+shown)\b", re.I)),
    ("additionally-opener",
     "sentences opened with 'Additionally/Moreover/Furthermore'",
     re.compile(r"(?:^|(?<=[.!?]\s))\s*(?:Additionally|Moreover|Furthermore|"
                r"In\s+conclusion|Overall)\s*,", re.M)),
    ("challenges-formula",
     "the canned 'Despite challenges … future prospects' ending",
     re.compile(r"\bdespite\s+(?:these\s+|its\s+|the\s+)?challenges\b|"
                r"\bchallenges\s+and\s+future\s+(?:prospects|directions)\b|"
                r"\bfuture\s+outlook\b|\bopens?\s+(?:up\s+)?promising\s+avenues\b|"
                r"\bsuggests?\s+(?:clear\s+)?directions?\s+for\s+(?:future|subsequent)\s+"
                r"(?:work|research)\b|\boutlines?\s+(?:specific\s+)?(?:avenues|directions)\s+"
                r"for\s+(?:future|subsequent|further)\s+(?:analysis|work|research|study)\b", re.I)),
    ("formulaic-metadiscourse",
     "announces that a point matters instead of stating it directly",
     re.compile(r"\bit\s+is\s+(?:important|worthwhile|essential|crucial)\s+to\s+"
                r"(?:note|remember|consider|recognize)\b|\bit\s+should\s+be\s+noted\b", re.I)),
    ("formulaic-need-claim",
     "canned statement that more work is needed",
     re.compile(r"\b(?:highlights?|emphasizes?|underscores?)\s+the\s+"
                r"(?:need|importance)\s+(?:for|of)\b|\bneed\s+for\s+continued\s+"
                r"(?:investigation|research|study)\b|\b(?:further|additional|continued)\s+"
                r"(?:work|research|investigation|study)\s+is\s+(?:needed|required)\b|"
                r"\b(?:requiring|warranting)\s+(?:further|additional|continued)\s+"
                r"(?:work|research|investigation|study)\b", re.I)),
    ("formulaic-findings-claim",
     "generic findings sentence claims contribution or reliability without analysis",
     re.compile(r"\b(?:these|the)\s+(?:findings|results)\s+"
                r"(?:contribute\s+to\s+(?:the\s+)?field|(?:confirm|demonstrate|establish)\s+"
                r"(?:the\s+)?(?:[\w’'-]+\s+){0,5}(?:reliability|robustness|importance)|"
                r"(?:indicate|show|suggest)\s+[^.!?]{0,70}?\bfunctions?\s+reliably)\b|"
                r"\b(?:the\s+)?(?:experimental\s+)?paradigm\s+"
                r"(?:demonstrates?\s+reliability|yields?\s+reliable\s+data)\b", re.I)),
    ("rule-of-three",
     "three-item adjective runs used for false comprehensiveness",
     re.compile(r"\b(\w+ly|\w+ive|\w+ic|\w+al|\w+ous)\s*,\s*(\w+ly|\w+ive|\w+ic|\w+al|\w+ous)\s*,"
                r"\s*and\s+(\w+ly|\w+ive|\w+ic|\w+al|\w+ous)\b", re.I)),
    ("emoji", "emoji used as formatting",
     re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")),
    ("curly-quotes", "curly quotation marks/apostrophes",
     re.compile("[\u2018\u2019\u201c\u201d]")),
]

# Formatting tells must see the RAW text \u2014 markdown, headings and emphasis are exactly
# what they are about, and prose-stripping destroys them. Each carries a density floor
# so a single bold word or one em dash is not an accusation.
_FORMAT_TELLS: list[tuple[str, str, re.Pattern, float]] = [
    ("title-case-headings",
     "headings in Title Case rather than sentence case",
     # Title case keeps minor words lowercase ("Impact of Technology and Digitalization"),
     # so requiring every word capitalised finds nothing. Instead: a heading line with
     # three or more capitalised content words and no terminal punctuation.
     re.compile(r"^\s{0,3}(?:#{1,6}\s+|\*\*)"
                r"(?=(?:[^\n]*?\b[A-Z][a-z]{2,}\b){3,})"
                r"[A-Z][^\n.!?]{4,90}$", re.M), 0.0),
    ("boldface-overuse",
     "mechanical boldface on terms mid-sentence",
     re.compile(r"\*\*[^*\n]{2,60}\*\*"), 8.0),
    ("inline-header-list",
     "bullet + bold header + colon + description, the LLM list shape",
     re.compile(r"^\s{0,4}(?:[-*+]|\d+\.)\s+\*\*[^*\n]{2,60}\*\*\s*[:\u2014-]", re.M), 0.0),
    ("em-dash-overuse",
     "em dashes standing in for other punctuation",
     re.compile(r"\u2014"), 6.0),
    ("title-as-proper-noun",
     "opening sentence treating the title as a standalone entity",
     re.compile(r"^\s{0,3}\*\*[^*\n]{3,80}\*\*\s+(?:refers?\s+to|is\s+a\s+term|"
                r"describes|denotes)\b", re.M), 0.0),
    ("skipped-heading-level",
     "heading hierarchy jumps a level (## then ####)",
     re.compile(r"^(#{2})\s+[^\n]+\n(?:[^\n#][^\n]*\n|\n)*?(#{4,6})\s", re.M), 0.0),
    ("thematic-break-before-heading",
     "horizontal rule inserted before a heading",
     re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*\n+\s*#{1,6}\s", re.M), 0.0),
]

_HEDGES = re.compile(
    r"\b(?:may|might|could|possibly|potentially|perhaps|arguably|likely|"
    r"suggests?|appears?\s+to|seems?\s+to|relatively|somewhat|generally|"
    r"tends?\s+to|to\s+some\s+extent)\b", re.I)
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(\w+ed|shown|given|taken|"
    r"seen|found|made|done|known|used|held|built|drawn|written)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(?:we|our|us)\b", re.I)
_CITATION = re.compile(r"\\cite[tp]?\{|\\citep?\[|\[\d{1,3}(?:[,–-]\s*\d{1,3})*\]|"
                       r"\([A-Z][A-Za-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}\)")
_EQUATION = re.compile(r"\\begin\{(?:equation|align|gather|multline|eqnarray)\*?\}|"
                       r"(?<!\$)\$(?!\$)[^$\n]{2,}\$(?!\$)|\\\[")
_SECTION = re.compile(r"\\(?:sub)?section\*?\{([^}]{1,80})\}")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$])")

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "which",
    "have", "has", "had", "not", "but", "can", "its", "it's", "our", "we", "they",
    "these", "those", "such", "than", "then", "there", "their", "been", "also", "all",
    "one", "two", "into", "when", "where", "what", "how", "each", "more", "most",
    "other", "some", "only", "over", "any", "may", "will", "would", "should", "must",
}


def _strip_tex(text: str) -> str:
    """Prose only: drop math, commands and comments so counts describe writing."""
    # Only a line whose first non-space character is ``%`` is unambiguously a LaTeX
    # comment here.  Treating every percent sign as a comment silently erased the rest
    # of ordinary Markdown sentences after values such as ``18.4%``.
    t = re.sub(r"(?m)^[ \t]*%(?!%).*$", "", text or "")
    t = re.sub(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z@]+\s*(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
    return re.sub(r"\s+", " ", t)


def _sentences(prose: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(prose or "") if len(s.strip()) > 15]


@dataclass
class StyleMetrics:
    """Deterministic measurements of one text."""
    words: int = 0
    sentences: int = 0
    mean_sentence_words: float = 0.0
    sd_sentence_words: float = 0.0
    hedges_per_1k: float = 0.0
    passive_per_1k: float = 0.0
    first_person_per_1k: float = 0.0
    citations_per_1k: float = 0.0
    equations_per_1k: float = 0.0
    mean_paragraph_sentences: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def measure(text: str) -> StyleMetrics:
    """Measure one document. All rates are per 1000 prose words so documents of
    different lengths compare directly."""
    raw = text or ""
    prose = _strip_tex(raw)
    words = _WORD.findall(prose)
    n = len(words)
    sents = _sentences(prose)
    lens = [len(_WORD.findall(s)) for s in sents] or [0]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    paras = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    per_para = [len(_sentences(_strip_tex(p))) for p in paras] or [0]
    k = (n / 1000.0) or 1e-9
    return StyleMetrics(
        words=n, sentences=len(sents),
        mean_sentence_words=round(mean, 2), sd_sentence_words=round(var ** 0.5, 2),
        hedges_per_1k=round(len(_HEDGES.findall(prose)) / k, 2),
        passive_per_1k=round(len(_PASSIVE.findall(prose)) / k, 2),
        first_person_per_1k=round(len(_FIRST_PERSON.findall(prose)) / k, 2),
        citations_per_1k=round(len(_CITATION.findall(raw)) / k, 2),
        equations_per_1k=round(len(_EQUATION.findall(raw)) / k, 2),
        mean_paragraph_sentences=round(sum(per_para) / len(per_para), 2),
    )


@dataclass
class StyleTemplate:
    """What 'writing like this field' means, in numbers. Mined from real papers."""
    sample_size: int = 0
    section_order: list = field(default_factory=list)
    vocabulary: list = field(default_factory=list)
    targets: dict = field(default_factory=dict)     # metric -> (low, median, high)

    def as_dict(self) -> dict:
        return {"sample_size": self.sample_size, "section_order": self.section_order,
                "vocabulary": self.vocabulary, "targets": self.targets}

    def markdown(self) -> str:
        rows = "\n".join(
            f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in sorted(self.targets.items()))
        secs = " → ".join(self.section_order) or "(no consistent order)"
        return (f"# Writing template ({self.sample_size} papers)\n\n"
                f"**Section order:** {secs}\n\n"
                f"**Field vocabulary:** {', '.join(self.vocabulary[:25])}\n\n"
                f"| metric | low | median | high |\n|---|---|---|---|\n{rows}\n")


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mine_template(texts: list[str], *, vocab_size: int = 40) -> StyleTemplate:
    """Derive a measurable template from a corpus. Targets are (25th, median, 75th)
    percentiles — a band, not a point, because fields vary and a writer who lands
    inside the band writes like the field."""
    usable = [t for t in (texts or []) if t and len(t) > 400]
    if not usable:
        return StyleTemplate()
    per_doc = [measure(t) for t in usable]
    targets: dict = {}
    for name in ("mean_sentence_words", "hedges_per_1k", "passive_per_1k",
                 "first_person_per_1k", "citations_per_1k", "equations_per_1k",
                 "mean_paragraph_sentences"):
        xs = sorted(getattr(m, name) for m in per_doc)
        lo = xs[max(0, int(len(xs) * 0.25) - 1)] if len(xs) > 3 else xs[0]
        hi = xs[min(len(xs) - 1, int(len(xs) * 0.75))] if len(xs) > 3 else xs[-1]
        targets[name] = (round(lo, 2), round(_median(xs), 2), round(hi, 2))

    order: Counter = Counter()
    for t in usable:
        names = [re.sub(r"\s+", " ", s).strip().title() for s in _SECTION.findall(t)]
        for i, nm in enumerate(names[:12]):
            order[(i, nm)] += 1
    by_slot: dict[int, str] = {}
    for (slot, nm), _ in order.most_common():
        by_slot.setdefault(slot, nm)
    # a template listing "Introduction, Results, Introduction, Results" describes
    # nothing — keep first occurrence of each section, in slot order
    section_order: list[str] = []
    for i in sorted(by_slot):
        if by_slot[i] not in section_order:
            section_order.append(by_slot[i])
    section_order = section_order[:10]

    vocab: Counter = Counter()
    for t in usable:
        for w in _WORD.findall(_strip_tex(t).lower()):
            if len(w) > 4 and w not in _STOP:
                vocab[w] += 1
    common = [w for w, c in vocab.most_common(vocab_size * 3) if c >= max(2, len(usable) // 2)]
    return StyleTemplate(sample_size=len(usable), section_order=section_order,
                         vocabulary=common[:vocab_size], targets=targets)


@dataclass
class Tell:
    id: str
    explanation: str
    count: int
    examples: list


# Mined section slug → the hand-written tell that already covers it. Without this the
# same habit is reported twice under two names and the score is inflated.
_MINED_ALIASES = {
    "undue-emphasis-on-significance-legacy-and-broade": "significance-inflation",
    "superficial-analyses": "superficial-analysis",
    "outline-like-conclusions-about-challenges-and-fu": "challenges-formula",
    "promotional-and-advertisement-like-language": "promotional",
    "high-density-of-ai-vocabulary-words": "ai-vocabulary",
    "avoidance-of-basic-copulatives-is-are-phrases": "copula-avoidance",
    "vague-attributions-and-overgeneralization-of-opi": "vague-attribution",
}

_MINED_CACHE: list | None = None


def _mined_tells() -> list:
    """Phrase tells mined from Wikipedia's own 'Words to watch' boxes.

    Loaded once and cached. Absent cache → empty list, so the hand-written structural
    patterns still work; detection degrades, it never breaks."""
    global _MINED_CACHE
    if _MINED_CACHE is None:
        try:
            from spiral.ai_tells import compile_tells, load

            _MINED_CACHE = compile_tells(load())
        except Exception:
            _MINED_CACHE = []
    return _MINED_CACHE


def _collect(rx, haystack: str, words: float, per_1k: bool,
             floor: float = 0.0) -> tuple[float, list] | None:
    hits = [m.group(0).strip() for m in rx.finditer(haystack)]
    if not hits:
        return None
    count = round(len(hits) / (words / 1000.0), 2) if per_1k else float(len(hits))
    if floor and count < floor:
        return None                      # below the density floor: not a signal
    seen, examples = set(), []
    for h in hits:
        key = " ".join(h.lower().split())
        if key not in seen:
            seen.add(key)
            examples.append(" ".join(h.split())[:60])
        if len(examples) == 4:
            break
    return count, examples


def ai_tells(text: str, *, per_1k: bool = True) -> list[Tell]:
    """Find the machine-prose markers Wikipedia catalogues.

    Three sources, all deterministic: phrase lists mined straight from the page's
    'Words to watch' boxes, hand-written patterns for the *structural* tells that are
    described rather than listed (the negative parallelisms, rule of three), and
    formatting tells that must see the raw text (title case, boldface, em dashes,
    inline-header lists). Every hit carries the text that triggered it, so a rewrite can
    be checked rather than trusted."""
    raw = text or ""
    prose = _strip_tex(raw)
    words = max(1.0, float(len(_WORD.findall(prose))))
    out: list[Tell] = []
    seen_ids: set[str] = set()

    for tid, why, rx in _TELLS:
        got = _collect(rx, prose, words, per_1k)
        if got:
            out.append(Tell(tid, why, got[0], got[1]))
            seen_ids.add(tid)

    for slug, name, rx in _mined_tells():
        # A mined section and a hand-written pattern often name the SAME tell
        # ("Superficial analyses" vs superficial-analysis). Counting both double-counts
        # the score and clutters the report, so the mined one only speaks when the
        # hand-written pattern it duplicates found nothing.
        if slug in seen_ids or _MINED_ALIASES.get(slug) in seen_ids:
            continue
        # Wikipedia calls this a HIGH-DENSITY signal and explicitly says one or two
        # such words may be coincidental. Generic entries such as "key", "valuable",
        # and "landscape" must not become a one-word blacklist. The hand-written
        # high-signal vocabulary pattern above still catches a lone "delve", "deep
        # dive", or "stands as a testament".
        if slug == "high-density-of-ai-vocabulary-words" and len(list(rx.finditer(prose))) < 3:
            continue
        got = _collect(rx, prose, words, per_1k)
        if got:
            out.append(Tell(slug, name.lower(), got[0], got[1]))

    for tid, why, rx, floor in _FORMAT_TELLS:
        got = _collect(rx, raw, words, per_1k, floor=floor)
        if got:
            out.append(Tell(tid, why, got[0], got[1]))

    return sorted(out, key=lambda t: -t.count)


def ai_score(text: str) -> float:
    """One number: total AI tells per 1000 words. Lower is more human. Useful as a
    before/after measurement on a rewrite, never as a verdict about authorship —
    human writing contains these too, just less densely."""
    return round(sum(t.count for t in ai_tells(text)), 2)


def ai_raw_score(text: str) -> int:
    """Count matched tell occurrences without length normalisation.

    Density is useful for reporting documents of different sizes, but it is unsafe as
    a rewrite objective: a model can lower a per-1000-word score simply by padding the
    paragraph.  Candidate selection therefore uses this raw count and reports density
    only after the edit.
    """

    return int(round(sum(t.count for t in ai_tells(text, per_1k=False))))


@dataclass
class Gap:
    metric: str
    value: float
    low: float
    high: float
    direction: str        # "raise" | "lower"

    def sentence(self) -> str:
        verb = "raise" if self.direction == "raise" else "lower"
        return (f"{self.metric}: {self.value} — {verb} toward the field band "
                f"[{self.low}, {self.high}]")


def template_distance(text: str, template: StyleTemplate | None) -> float:
    """Normalised distance outside a corpus style band (zero means in-band).

    Each metric contributes its distance to the nearest quartile boundary, scaled by
    the observed band width.  Averaging prevents a template with more populated metrics
    from receiving a larger score merely because it has more columns.
    """

    if template is None or not template.sample_size:
        return 0.0
    metrics = measure(text)
    metric_distances: list[float] = []
    for name, bounds in template.targets.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
            continue
        lo, median, hi = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        value = getattr(metrics, name, None)
        if value is None:
            continue
        # Degenerate corpora are common in tests and small manual samples.  Scale by
        # the median (then one) rather than turning a zero-width band into infinity.
        scale = max(abs(hi - lo), abs(median) * 0.25, 1.0)
        if value < lo:
            metric_distances.append((lo - value) / scale)
        elif value > hi:
            metric_distances.append((value - hi) / scale)
        else:
            metric_distances.append(0.0)

    components: list[float] = []
    if metric_distances:
        components.append(sum(metric_distances) / len(metric_distances))

    vocab = [str(word).lower() for word in (template.vocabulary or [])
             if str(word).strip()][:24]
    if vocab:
        held = set(_WORD.findall(_strip_tex(text or "").lower()))
        word_count = max(1, len(_WORD.findall(_strip_tex(text or ""))))
        # A paragraph need not recite a field glossary; a full paper should use several
        # of the corpus's highest-consensus terms.  Cap the target to prevent stuffing.
        expected = min(len(vocab), min(4, max(1, int(math.log2(word_count / 80 + 1)) + 1)))
        hits = sum(1 for word in vocab if word in held)
        components.append(max(0.0, expected - hits) / expected)

    expected_sections = [re.sub(r"\s+", " ", str(section)).strip().casefold()
                         for section in (template.section_order or []) if str(section).strip()]
    if expected_sections:
        actual_sections = [
            re.sub(r"\s+", " ", name).strip().casefold()
            for name in (
                _SECTION.findall(text or "")
                + re.findall(r"(?m)^\s{0,3}#{1,6}\s+([^\n#].*?)\s*$", text or "")
            )
        ]
        # Longest common subsequence measures both missing headings and the wrong arc.
        row = [0] * (len(actual_sections) + 1)
        for expected in expected_sections:
            previous = row[:]
            for j, actual in enumerate(actual_sections, 1):
                row[j] = (previous[j - 1] + 1 if expected == actual
                          else max(previous[j], row[j - 1]))
        components.append(1.0 - row[-1] / len(expected_sections))

    return round(sum(components) / max(1, len(components)), 6)


def score_against(text: str, template: StyleTemplate) -> dict:
    """Measure a draft against a mined template. Returns the gaps as instructions a
    writer (human or model) can act on, plus the AI tells found. No opinion."""
    m = measure(text)
    gaps: list[Gap] = []
    for name, (lo, _med, hi) in (template.targets or {}).items():
        val = getattr(m, name, None)
        if val is None:
            continue
        if val < lo:
            gaps.append(Gap(name, val, lo, hi, "raise"))
        elif val > hi:
            gaps.append(Gap(name, val, lo, hi, "lower"))
    tells = ai_tells(text)
    section_names = (_SECTION.findall(text or "")
                     + re.findall(r"(?m)^\s{0,3}#{1,6}\s+([^\n#].*?)\s*$", text or ""))
    have = {re.sub(r"\s+", " ", s).strip().title() for s in section_names}
    missing = [s for s in (template.section_order or []) if s not in have]
    vocab = [str(word).lower() for word in (template.vocabulary or [])
             if str(word).strip()][:24]
    held = set(_WORD.findall(_strip_tex(text or "").lower()))
    word_count = max(1, len(_WORD.findall(_strip_tex(text or ""))))
    expected_vocab = min(len(vocab), min(4, max(1, int(math.log2(word_count / 80 + 1)) + 1)))
    vocab_hits = [word for word in vocab if word in held]
    gap_sentences = [g.sentence() for g in gaps]
    if vocab and len(vocab_hits) < expected_vocab:
        gap_sentences.append(
            "field vocabulary: uses " + str(len(vocab_hits)) + " of the expected "
            + str(expected_vocab) + " high-consensus terms — raise coverage with relevant "
            "corpus terms only where the claims warrant them"
        )
    return {
        "metrics": m.as_dict(),
        "gaps": gap_sentences,
        "missing_sections": missing,
        "field_vocabulary_hits": vocab_hits,
        "ai_tells": [{"id": t.id, "per_1k": t.count, "why": t.explanation,
                      "examples": t.examples} for t in tells],
        "ai_score": round(sum(t.count for t in tells), 2),
        "in_band": not gap_sentences and not missing,
    }
