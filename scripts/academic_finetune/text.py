"""Conservative prose extraction and deterministic plan construction."""

from __future__ import annotations

import io
import re
import tarfile
import xml.etree.ElementTree as ET
from typing import Iterable

_SPACE = re.compile(r"[ \t\f\v]+")
_CITATION = re.compile(
    r"(?:"
    r"\\cite[A-Za-z*]*\s*(?:\[[^\]\r\n]*\]\s*){0,2}\{[^{}\r\n]+\}"
    r"|\[[0-9,;\-– ]+\]"
    r"|\([A-Z][A-Za-z'’\-]+(?: et al\.)?,? \d{4}[a-z]?\)"
    r")"
)
_NOTICE_TITLE = re.compile(
    r"(?i)^\s*(?:withdrawn|retracted(?:\s+article)?|retraction|corrigendum|erratum|"
    r"expression\s+of\s+concern)\b"
)
_NOTICE_PROSE = re.compile(
    r"(?i)\b(?:article\s+has\s+been\s+withdrawn|publisher\s+has\s+retracted\s+this\s+article|"
    r"publisher\s+apologizes|policy\s+on\s+article\s+withdrawal|compromised\s+peer\s+review\s+process|"
    r"full\s+text\s+of\s+the\s+retracted\s+article)\b"
)
_UNIT_BOILERPLATE = re.compile(
    r"(?i)(?:https?://|\bwww\.|\bonline\s+access\b|\bcme[- ]accredited\b|"
    r"\baccredited\s+educational\s+programs?\b|\bgoogle\s+play\b|\bapp\s*store\b|"
    r"\bavailable\s+to\s+download\b|\bdownload\s+from\b|\bfull\s+elsevier\s+policy\b|"
    r"\bthe\s+faculty\s+will\s+review\b)"
)
_EQUATION_BLOCK = re.compile(
    r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}.*?"
    r"\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}",
    re.DOTALL,
)
_SECTION_REFERENCES = re.compile(
    r"(?ims)^\s*(?:\\(?:section|chapter)\*?\{\s*)?(?:references|bibliography)\s*\}?\s*$.*\Z"
)


def _normalise_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = _SPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_prose(text: str) -> str:
    """Remove non-prose sections without paraphrasing the human text."""

    text = _EQUATION_BLOCK.sub(" [equation] ", text)
    text = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]", " [equation] ", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$", " [equation] ", text)
    text = _SECTION_REFERENCES.sub("", text)
    text = re.sub(r"(?im)^\s*(?:received|accepted|revised)\s*:?.*$", "", text)
    text = re.sub(r"(?im)^\s*(?:arxiv|preprint)\s*:?.*$", "", text)
    text = re.sub(r"(?im)^\s*(?:copyright|©).*?$", "", text)
    return _normalise_lines(text)


def canonicalize_citations(value: str) -> str:
    """Map paper-specific citation occurrences to deterministic local slots."""

    index = 0

    def replace(_match: re.Match[str]) -> str:
        nonlocal index
        index += 1
        return f"[{index}]"

    return _CITATION.sub(replace, value)


def document_rejection_reason(title: str, prose: str) -> str | None:
    """Return a stable source-hygiene reason for non-paper notice records."""

    if _NOTICE_TITLE.search(title):
        return "notice_title"
    if _NOTICE_PROSE.search(prose):
        return "notice_prose"
    return None


def academic_unit_rejection_reason(value: str) -> str | None:
    """Reject publisher/promotional copy and visibly malformed prose units."""

    if _NOTICE_PROSE.search(value):
        return "notice_prose"
    if _UNIT_BOILERPLATE.search(value):
        return "promotional_or_web_boilerplate"
    if "\ufffd" in value or any(ord(character) < 32 and character not in "\n\t" for character in value):
        return "invalid_character"
    if value.count("(") != value.count(")") or value.count("[") != value.count("]"):
        return "unbalanced_delimiter"
    return None


def _strip_tex(tex: str) -> str:
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    begin = tex.find("\\begin{document}")
    if begin >= 0:
        tex = tex[begin + len("\\begin{document}") :]
    end = tex.find("\\end{document}")
    if end >= 0:
        tex = tex[:end]
    tex = _EQUATION_BLOCK.sub(" [equation] ", tex)
    bibliography_begin = re.search(r"\\begin\{thebibliography\}", tex)
    if bibliography_begin:
        tex = tex[: bibliography_begin.start()]
    tex = re.sub(r"\\(?:bibliography|printbibliography)\b.*", "", tex, flags=re.DOTALL)
    tex = re.sub(
        r"\\begin\{(figure\*?|table\*?|tikzpicture)\}.*?\\end\{\1\}",
        "\n",
        tex,
        flags=re.DOTALL,
    )
    tex = re.sub(r"\\(?:includegraphics|label|ref|eqref|cite\w*)\s*(?:\[[^]]*\])?\{[^{}]*\}", " ", tex)
    tex = re.sub(r"\\(?:section|subsection|subsubsection|paragraph|chapter)\*?\{([^{}]*)\}", r"\n\n\1\n\n", tex)
    # Repeatedly unwrap simple formatting commands while retaining their prose.
    for _ in range(4):
        tex = re.sub(r"\\(?:textbf|textit|emph|mathrm|textrm|mbox|author|title)\{([^{}]*)\}", r"\1", tex)
    tex = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", tex)
    tex = tex.replace("~", " ").replace("\\&", "&").replace("``", '"').replace("''", '"')
    tex = re.sub(r"[{}]", "", tex)
    return clean_prose(tex)


def clean_tex_archive(payload: bytes, *, max_members: int = 256, max_unpacked: int = 32 * 1024 * 1024) -> str:
    """Extract TeX in-memory with archive-bomb and path limits."""

    candidates: list[tuple[str, str]] = []
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.TarError:
        if payload.startswith(b"\x1f\x8b"):
            import gzip

            payload = gzip.decompress(payload)
            if len(payload) > max_unpacked:
                raise ValueError("compressed arXiv TeX exceeds the unpacked size limit")
        return _strip_tex(payload.decode("utf-8", errors="replace"))
    total = 0
    with archive:
        members = archive.getmembers()
        if len(members) > max_members:
            raise ValueError("arXiv source archive has too many members")
        for member in members:
            if not member.isfile() or not member.name.lower().endswith(('.tex', '.ltx')):
                continue
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ValueError("unsafe path in arXiv source archive")
            total += member.size
            if total > max_unpacked:
                raise ValueError("arXiv source archive exceeds unpacked size limit")
            handle = archive.extractfile(member)
            if handle is None:
                continue
            text = handle.read().decode("utf-8", errors="replace")
            candidates.append((member.name, text))
    if not candidates:
        return ""
    # Prefer the main document; ties are stable by path.
    candidates.sort(key=lambda item: ("\\begin{document}" not in item[1], -len(item[1]), item[0]))
    return _strip_tex(candidates[0][1])


def clean_pdf_bytes(payload: bytes, *, max_pages: int = 300) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(payload))
    if len(reader.pages) > max_pages:
        raise ValueError(f"PDF exceeds the {max_pages}-page extraction limit")
    return clean_prose("\n\n".join(page.extract_text() or "" for page in reader.pages))


def clean_pmc_xml(payload: bytes) -> str:
    root = ET.fromstring(payload)
    article = root.find(".//article") if root.tag != "article" else root
    if article is None:
        return ""
    body = article.find("body")
    if body is None:
        return ""
    paragraphs: list[str] = []
    for paragraph in body.findall(".//p"):
        if any(ancestor in {"ref-list", "ack", "fn-group"} for ancestor in _ancestor_tags(body, paragraph)):
            continue
        text = "".join(paragraph.itertext())
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
    return clean_prose("\n\n".join(paragraphs))


def _ancestor_tags(root: ET.Element, target: ET.Element) -> set[str]:
    # ElementTree has no parent pointer; bodies are small enough for this bounded walk.
    result: set[str] = set()

    def visit(node: ET.Element, ancestors: tuple[str, ...]) -> bool:
        if node is target:
            result.update(ancestors)
            return True
        return any(visit(child, ancestors + (node.tag.rsplit("}", 1)[-1],)) for child in node)

    visit(root, ())
    return result


def paragraphs(text: str) -> list[str]:
    units = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n\s*\n", text)]
    return [item for item in units if len(item) >= 60 and _prose_ratio(item) >= 0.65]


def _prose_ratio(value: str) -> float:
    if not value:
        return 0.0
    letters_and_spaces = sum(character.isalpha() or character.isspace() for character in value)
    return letters_and_spaces / len(value)


_ABBREVIATIONS = {"e.g.", "i.e.", "et al.", "fig.", "eq.", "dr.", "prof.", "vs."}


def sentences(paragraph: str) -> list[str]:
    result: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?](?:[\"'’”)]*)\s+(?=[A-Z0-9])", paragraph):
        prefix = paragraph[max(start, match.start() - 8) : match.end()].lower().strip()
        if any(prefix.endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
            continue
        candidate = paragraph[start : match.end()].strip()
        if candidate:
            result.append(candidate)
        start = match.end()
        while start < len(paragraph) and paragraph[start].isspace():
            start += 1
    tail = paragraph[start:].strip()
    if tail:
        result.append(tail)
    return result


_RELATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("limitation", ("limitation", "caveat", "however", "although", "despite", "nevertheless")),
    ("contrast", ("whereas", "in contrast", "on the other hand", "unlike", "but")),
    ("causal", ("because", "owing to", "due to", "thereby", "causes")),
    ("implication", ("therefore", "thus", "hence", "consequently", "implies", "suggests")),
    ("comparison", ("compared with", "similar to", "greater than", "less than")),
    ("evidence", ("we find", "we show", "our results", "the data", "demonstrate")),
    ("definition", ("we define", "is defined", "refers to", "denotes")),
)


def rhetorical_relation(target: str) -> str:
    lowered = target.lower()
    for relation, markers in _RELATIONS:
        if any(marker in lowered for marker in markers):
            return relation
    return "elaboration"


def certainty(target: str) -> str:
    lowered = target.lower()
    if re.search(r"\b(?:may|might|could|appears?|seems?|suggests?|possibly|likely|unlikely)\b", lowered):
        return "tentative"
    if re.search(r"\b(?:establish(?:es|ed)?|prove[sd]?|must|cannot|always|demonstrate[sd]?)\b", lowered):
        return "strong"
    return "calibrated"


def citation_count(target: str) -> int:
    return len(_CITATION.findall(target))


def citation_markers(target: str) -> list[str]:
    """Return exact allowed markers so training never guesses paper numbering."""

    return _CITATION.findall(target)


MINIMUM_PLAN_CONTENT_RECALL = 0.55


_PLAN_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "because", "been", "before",
    "being", "between", "both", "could", "does", "during", "each", "from", "further",
    "have", "having", "however", "into", "itself", "might", "more", "most", "must",
    "other", "should", "since", "some", "such", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "under", "using", "very", "were",
    "where", "which", "while", "with", "would",
}

_FUNCTION_WORDS = _PLAN_STOPWORDS | {
    "a", "an", "and", "as", "at", "be", "by", "for", "if", "in", "is", "it", "its",
    "no", "not", "of", "on", "or", "the", "to", "was", "we", "our", "only", "when",
    "whereas", "although", "but", "yet", "several", "many", "any", "all",
}
_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "show", "shows", "showed",
    "find", "finds", "found", "observe", "observes", "observed", "demonstrate",
    "demonstrates", "demonstrated", "argue", "argues", "propose", "proposes",
    "preserve", "preserves", "avoid", "avoids", "remain", "remains", "yield", "yields",
    "provide", "provides", "retain", "retains", "retained", "constrain", "constrains",
    "constrained", "support", "supports", "limit", "limits", "limited", "reduce", "reduces",
    "increase", "increases", "require", "requires", "determine", "determines", "leave",
    "leaves", "defer", "defers", "deferred", "indicate", "indicates", "suggest",
    "suggests", "reproduce", "reproduces", "establish", "establishes", "become", "becomes",
    "improve", "improves", "improved", "predict", "predicts", "allow", "allows",
    "affect", "affects", "arise", "arises", "account", "accounts", "calculate",
    "calculates", "cause", "causes", "characterize", "characterizes", "constitute",
    "constitutes", "control", "controls", "depend", "depends", "derive", "derives",
    "describe", "describes", "enhance", "enhances", "exhibit", "exhibits", "follow",
    "follows", "generate", "generates", "govern", "governs", "induce", "induces",
    "lead", "leads", "measure", "measures", "obtain", "obtains", "produce",
    "produces", "represent", "represents", "result", "results", "reveal", "reveals",
    "suppress", "suppresses", "quantify", "quantifies",
}
_NOMINAL = {
    "energetically": "energetic",
    "stable": "stability",
    "accurate": "accuracy",
    "reliable": "reliability",
    "compatible": "compatibility",
    "uncertain": "uncertainty",
    "significant": "significance",
    "improved": "improvement",
    "improves": "improvement",
    "constrains": "constraint",
    "constrain": "constraint",
    "limits": "limitation",
    "limited": "limitation",
    "reduces": "reduction",
    "increases": "increase",
}


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'’\-]*|\d+(?:\.\d+)?", value.casefold())


def plan_target_overlap(claims: Iterable[str], target: str) -> dict[str, float | int]:
    """Measure accidental answer leakage from proposition slots.

    Context is deliberately excluded: it is genuine preceding paper prose, not
    a derived representation of the held-out target.
    """

    plan_tokens = _word_tokens(" ".join(claims))
    target_tokens = _word_tokens(target)
    longest = 0
    target_positions: dict[str, list[int]] = {}
    for index, token in enumerate(target_tokens):
        target_positions.setdefault(token, []).append(index)
    for plan_index, token in enumerate(plan_tokens):
        for target_index in target_positions.get(token, ()):
            length = 0
            while (
                plan_index + length < len(plan_tokens)
                and target_index + length < len(target_tokens)
                and plan_tokens[plan_index + length] == target_tokens[target_index + length]
            ):
                length += 1
            longest = max(longest, length)
    plan_content = {token for token in plan_tokens if token not in _PLAN_STOPWORDS and len(token) >= 4}
    target_content = {token for token in target_tokens if token not in _PLAN_STOPWORDS and len(token) >= 4}
    jaccard = (
        len(plan_content & target_content) / len(plan_content | target_content)
        if plan_content or target_content
        else 0.0
    )
    return {"longest_common_ngram": longest, "content_jaccard": round(jaccard, 6)}


def context_target_overlap(context: str, target: str) -> dict[str, float | int | bool]:
    """Detect a held-out completion accidentally copied into its prompt context.

    Academic neighbours should share terminology, so ordinary topical overlap is
    allowed.  Exact containment, near-duplicate content, and long verbatim token
    runs are not: any of those would turn the realization task into copying.
    """

    context_tokens = _word_tokens(context)
    target_tokens = _word_tokens(target)
    normalized_context = " ".join(context_tokens)
    normalized_target = " ".join(target_tokens)
    exact_target_in_context = bool(normalized_target and normalized_target in normalized_context)
    overlap = plan_target_overlap([context], target)
    return {
        **overlap,
        "exact_target_in_context": exact_target_in_context,
    }


def context_is_safe(context: str, target: str) -> bool:
    overlap = context_target_overlap(context, target)
    return (
        not bool(overlap["exact_target_in_context"])
        and int(overlap["longest_common_ngram"]) < 12
        and float(overlap["content_jaccard"]) <= 0.85
    )


def claims_are_safe(claims: Iterable[str], target: str) -> bool:
    overlap = plan_target_overlap(claims, target)
    # Semantic plans must retain the scientific entities and result; prevent
    # surface copying with the stricter contiguous-run gate while allowing a
    # compact slot list to cover most of a short sentence's content words.
    return int(overlap["longest_common_ngram"]) < 5 and float(overlap["content_jaccard"]) <= 0.75


def _content_tokens(value: str) -> list[str]:
    return [token for token in _word_tokens(value) if token not in _FUNCTION_WORDS]


def _claim_payload(claim: str) -> str:
    payload = claim.partition("—")[2].strip()
    return payload or claim


def _unique_content_tokens(value: str) -> list[str]:
    result: list[str] = []
    for token in _content_tokens(_CITATION.sub("", value)):
        if token not in result:
            result.append(token)
    return result


def plan_content_recall(claims: Iterable[str], target: str) -> float:
    """Return the fraction of unique target facts/entities retained by the plan."""

    target_content = set(_unique_content_tokens(target))
    if not target_content:
        return 1.0
    payload_content = set(
        _content_tokens(" ".join(_claim_payload(claim) for claim in claims))
    )
    return len(target_content & payload_content) / len(target_content)


def claims_are_feasible(
    claims: Iterable[str],
    target: str,
    *,
    minimum_content_recall: float = MINIMUM_PLAN_CONTENT_RECALL,
) -> bool:
    claims = list(claims)
    if not claims or not claims_are_safe(claims, target):
        return False
    if any("withheld" in claim.casefold() or "unsupported" in claim.casefold() for claim in claims):
        return False
    if plan_content_recall(claims, target) + 1e-12 < minimum_content_recall:
        return False
    target_numbers = {
        token for token in _word_tokens(_CITATION.sub("", target)) if token[:1].isdigit()
    }
    payload_numbers = {
        token
        for token in _word_tokens(" ".join(_claim_payload(claim) for claim in claims))
        if token[:1].isdigit()
    }
    return target_numbers <= payload_numbers


def _nominalise(tokens: list[str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        replacement = _NOMINAL.get(token, token)
        if replacement not in result:
            result.append(replacement)
    return result


def _clauses(value: str) -> list[tuple[str, str]]:
    marker_pattern = re.compile(
        r"\s*(?:;|,)?\s*\b(but|although|however|whereas|while|because|therefore|thus|hence|consequently|nevertheless)\b\s*",
        re.IGNORECASE,
    )
    result: list[tuple[str, str]] = []
    start = 0
    marker = ""
    for match in marker_pattern.finditer(value):
        clause = value[start : match.start()].strip(" ,;")
        if clause:
            result.append((marker, clause))
        marker = match.group(1).casefold()
        start = match.end()
    tail = value[start:].strip(" ,;")
    if tail:
        result.append((marker, tail))
    return result


def _semantic_slots(value: str, *, maximum: int = 6) -> list[tuple[str, list[str]]]:
    value = _CITATION.sub("", value)
    value = re.sub(
        r"(?i)^\s*we\s+(?:show|find|observe|demonstrate|argue|propose)\s+that\s+",
        "",
        value,
    )
    slots: list[tuple[str, list[str]]] = []
    for clause_index, (marker, clause) in enumerate(_clauses(value)):
        tokens = _word_tokens(clause)
        verb_index = next((index for index, token in enumerate(tokens) if token in _VERBS), -1)
        before = tokens[:verb_index] if verb_index >= 0 else []
        after = tokens[verb_index + 1 :] if verb_index >= 0 else tokens
        subject = _content_tokens(" ".join(before))[-3:]
        predicate = _nominalise(_content_tokens(" ".join(after)))

        # When the deliberately tiny verb lexicon cannot classify a clause,
        # retain both ends of its proposition instead of truncating away the
        # scientific result/object at the end of the sentence.
        if verb_index < 0 and len(predicate) > 4:
            predicate = predicate[:2] + predicate[-2:]

        if clause_index == 0 and subject:
            slots.append(("subject", subject))

        lowered = clause.casefold()
        if re.search(r"\b(?:leave|leaves|defer|defers|deferred|later|future work)\b", lowered):
            role = "scope"
            predicate = [
                token
                for token in predicate
                if token not in {"leave", "leaves", "defer", "defers", "deferred", "later", "work"}
            ][:2]
            predicate.append("deferred")
        elif marker in {"but", "although", "however", "nevertheless"}:
            role = "limitation"
        elif marker in {"whereas", "while"}:
            role = "contrast"
        elif marker == "because":
            role = "cause"
        elif marker in {"therefore", "thus", "hence", "consequently"}:
            role = "implication"
        elif re.search(r"\b(?:limit|limits|limited|uncertain|uncertainty|caveat)\b", lowered):
            role = "limitation"
        elif verb_index >= 0 and tokens[verb_index] in {"show", "shows", "find", "finds", "found", "demonstrate", "demonstrates"}:
            role = "finding"
        else:
            role = "result" if verb_index >= 0 else "proposition"

        # Keep the clause's most informative predicate phrase, not arbitrary
        # words sampled across the target. Scope phrases prefer their object;
        # other predicates preserve the first local 2–4-token unit.
        if clause_index > 0 and subject and role != "scope":
            predicate = subject[-2:] + predicate[:2]
        predicate = predicate[:4]
        if predicate:
            slots.append((role, predicate))

        quantity = re.search(
            r"(?i)\b((?:approximately|roughly|nearly|about|at least|at most)\s+)?"
            r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
            r"(percent|%|fold|times?)\b",
            clause,
        )
        if quantity:
            quantity_tokens = _word_tokens(quantity.group(0))[:4]
            if quantity_tokens and not any(set(quantity_tokens) <= set(payload) for _role, payload in slots):
                slots.append(("quantity", quantity_tokens))
        if len(slots) >= maximum:
            break

    if re.search(r"\b(?:no|not|never|neither|without|cannot)\b", value, re.IGNORECASE):
        slots.append(("polarity", ["negative"]))
    return slots[:maximum]


def _coverage_priority(value: str) -> list[str]:
    """Prefer quantities and both ends of a proposition without copying runs."""

    tokens = _unique_content_tokens(value)
    quantities = [token for token in tokens if token[:1].isdigit()]
    remainder = [token for token in tokens if token not in quantities]
    interleaved: list[str] = []
    left, right = 0, len(remainder) - 1
    while left <= right:
        interleaved.append(remainder[left])
        left += 1
        if left <= right:
            interleaved.append(remainder[right])
            right -= 1
    return quantities + interleaved


def _add_coverage_slots(
    slots: list[tuple[str, list[str]]],
    value: str,
    *,
    role_prefix: str = "detail",
) -> list[tuple[str, list[str]]]:
    target_tokens = _unique_content_tokens(value)
    if not target_tokens:
        return slots
    required = min(
        len(target_tokens),
        max(1, int(len(target_tokens) * MINIMUM_PLAN_CONTENT_RECALL + 0.999999)),
    )
    covered = {
        token
        for _role, payload in slots
        for token in _content_tokens(" ".join(payload))
        if token in target_tokens
    }
    missing = [token for token in _coverage_priority(value) if token not in covered]
    required_numbers = {
        token for token in target_tokens if token[:1].isdigit()
    } - covered
    selected: list[str] = []
    for token in missing:
        if token in required_numbers or len(selected) < max(0, required - len(covered)):
            selected.append(token)
    for offset in range(0, len(selected), 4):
        chunk = selected[offset : offset + 4]
        if chunk:
            slots.append((f"{role_prefix}-{offset // 4 + 1}", chunk))
    return slots


def _format_safe_slots(slots: list[tuple[str, list[str]]], target: str) -> list[str]:
    working = [(role, payload[:]) for role, payload in slots if payload]
    while working:
        claims = [f"{role} — {' '.join(payload)}." for role, payload in working]
        if claims_are_safe(claims, target):
            return claims
        reducible = next((index for index in range(len(working) - 1, -1, -1) if len(working[index][1]) > 1), None)
        if reducible is not None:
            working[reducible][1].pop()
        else:
            working.pop()
    return []


def neutral_claims(target: str, *, maximum: int = 6, paragraph: bool = False) -> list[str]:
    """Build shallow semantic slots without copying a target five-gram.

    Sentence tasks expose subject/result/limitation/scope/quantity roles. For a
    paragraph, one compact proposition is emitted per target sentence (2–8), so
    a long target is never represented by only three arbitrary keywords.
    """

    if paragraph:
        target_sentences = sentences(target)
        if not 2 <= len(target_sentences) <= 8:
            return ["paragraph — unsupported sentence count."]
        paragraph_slots: list[tuple[str, list[str]]] = []
        for index, sentence in enumerate(target_sentences, start=1):
            slots = _semantic_slots(sentence, maximum=max(4, maximum))
            subject = next((payload for role, payload in slots if role == "subject"), [])
            outcome = next(
                (
                    (role, payload)
                    for role, payload in reversed(slots)
                    if role not in {"subject", "polarity", "quantity"}
                ),
                None,
            )
            role = outcome[0] if outcome else "proposition"
            payload = (subject[:2] + (outcome[1][:2] if outcome else []))[:4]
            paragraph_slots.append((f"sentence-{index}-{role}", payload or ["content", "withheld"]))
            paragraph_slots = _add_coverage_slots(
                paragraph_slots,
                sentence,
                role_prefix=f"sentence-{index}-detail",
            )
            if any(slot_role == "polarity" for slot_role, _payload in slots):
                paragraph_slots.append((f"sentence-{index}-polarity", ["negative"]))
        claims = _format_safe_slots(paragraph_slots, target)
        return claims if claims_are_feasible(claims, target) else []

    slots = _semantic_slots(target, maximum=maximum)
    if not slots:
        return []
    slots = _add_coverage_slots(slots, target)
    claims = _format_safe_slots(slots, target)
    return claims if claims_are_feasible(claims, target) else []


def usable_sentence(value: str) -> bool:
    return (
        60 <= len(value) <= 650
        and _prose_ratio(value) >= 0.7
        and len(value.split()) >= 10
        and academic_unit_rejection_reason(value) is None
    )


def usable_paragraph(value: str) -> bool:
    sentence_count = len(sentences(value))
    return (
        180 <= len(value) <= 1800
        and _prose_ratio(value) >= 0.7
        and 2 <= sentence_count <= 8
        and academic_unit_rejection_reason(value) is None
    )
