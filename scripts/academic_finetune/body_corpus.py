"""Offline main-body prose extraction and corpus compilation.

The structure cache already contains official arXiv TeX and PMC/JATS artifacts.
This module reuses those bytes without network access, extracts only structurally
identified main-body paragraphs, emits body-only :class:`SourceDocument` records,
and compiles the existing plan-to-prose contract with section-level provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from scripts.academic_finetune import MANIFEST_SCHEMA
from scripts.academic_finetune import structure_extract as sx
from scripts.academic_finetune.corpus import compile_corpus
from scripts.academic_finetune.sources import (
    SourceDocument,
    canonical_json,
    load_source_documents,
    raw_record_sha256,
    sha256_text,
)
from scripts.academic_finetune.structure_corpus import (
    load_metadata_cache,
    load_structure_cache,
)
from scripts.academic_finetune.text import canonicalize_citations, paragraphs, sentences


BODY_EXTRACTION_SCHEMA = "spiral.academic-main-body-extraction.v1"
BODY_CACHE_MANIFEST_SCHEMA = "spiral.academic-main-body-cache-manifest.v1"
BODY_ATTESTATION_SCHEMA = "spiral.academic-main-body-attestation.v1"
BODY_TEMPORAL_ATTESTATION_SCHEMA = "spiral.academic-jats-temporal-attestation.v1"
BODY_CUTOFF = date(2021, 12, 31)
DEFAULT_MAX_SEQUENCE_LENGTH = 448
EXCLUDED_REGIONS = ("abstract", "references", "acknowledgments", "appendix")
_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
_LOCATOR = re.compile(r"^paragraph:(\d+)(?:;sentence:(\d+))?$")
_DISPLAY_MATH_SENTINEL = "SPIRAL_BODY_REJECT_DISPLAY_MATH_4F8792D1"
_REFERENCE_COMMANDS = {
    "autoref", "cref", "crefrange", "cpageref", "eqref", "figref", "nameref",
    "pageref", "ref", "tabref", "vpageref", "vref",
}
_SAFE_TEX_COMMANDS = {
    "bf", "bfseries", "em", "emph", "it", "itshape", "label", "linebreak",
    "medskip", "newline", "noindent", "par", "protect", "rm", "rmfamily",
    "scshape", "sffamily", "slshape", "smallskip", "textbf", "textit", "textnormal",
    "textrm", "textsc", "textsf", "textsl", "texttt", "ttfamily", "underline",
}
_MATH_COMMANDS = {
    "boxed", "dfrac", "displaystyle", "ensuremath", "frac", "left", "mathbb",
    "mathbf", "mathcal", "mathit", "mathrm", "mathsf", "mathtt", "operatorname",
    "overline", "right", "sqrt", "tfrac", "textstyle",
}
_JATS_MATH_TAGS = {
    "disp-formula", "disp-formula-group", "inline-formula", "math", "tex-math",
}
_JATS_OBJECT_REFERENCE_TYPES = {
    "app", "boxed-text", "disp-formula", "fig", "sec", "supplementary-material",
    "table", "table-fn", "table-wrap",
}
_RESIDUAL_TEX = re.compile(r"\\[A-Za-z@]+|[{}]")
_EMPTY_OBJECT = re.compile(r"(?:\(\s*\)|\[\s*\]|\b(?:the|a|an)\s*[,.):;])", re.IGNORECASE)
_DANGLING_LINK = re.compile(
    r"\b(?:as|at|by|for|from|in|of|over|than|to|under|with)\s*(?:[,.):;]|$)",
    re.IGNORECASE,
)
_ACKNOWLEDGMENT_PROSE = re.compile(
    r"\b(?:"
    r"(?:we|the authors?)\s+(?:acknowledg(?:e|es|ed)|thank|are grateful|would like to thank)"
    r"|(?:is|are|was|were)\s+grateful\s+to"
    r"|thanks?\s+(?:are\s+due\s+)?to"
    r"|grateful\s+for\s+(?:the\s+)?(?:invitation|hospitality|support)"
    r")\b",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = re.compile(
    r"(?:[.?!][\"'’”)]*|[.?!][\"'’”)]*\s*(?:\[\d+\](?:\s*[,;–-]\s*\[\d+\])*)?)$"
)
_KNOWN_SOURCE_TEXT_DEFECT = re.compile(
    r"\b(?:"
    r"annhilation|back\s+hole|back\s+the\s+nearly|ben\s+considered|canstant|"
    r"copies\s+of\s+same|hyperex\s+tended|in\s+details|in\s+par\s+with|"
    r"numbers?\s+of\s+girl\b|order\s+of\s+the\s+order|oscilations|"
    r"to\s+see,\s+if|which\s+the\s+discussion\s+can"
    r")\b",
    re.IGNORECASE,
)
_QUERY_DATE_RANGE = re.compile(
    r"(\d{4})/\d{2}/\d{2}\s*:\s*(\d{4})/\d{2}/\d{2}\[Date\s*-\s*Publication\]",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


class BodyCorpusError(ValueError):
    """The offline artifact/cache lineage or body provenance is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BodyCorpusError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, payload)


@dataclass(frozen=True)
class BodyParagraph:
    section_id: str
    section_title: str
    section_path: tuple[int, ...]
    section_order: int
    paragraph_index_in_section: int
    document_paragraph_index: int
    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("body paragraph text cannot be empty")
        if self.section_order < 0 or self.paragraph_index_in_section < 1:
            raise ValueError("body paragraph section positions are invalid")
        if self.document_paragraph_index < 1 or any(value < 1 for value in self.section_path):
            raise ValueError("body paragraph document/path positions are invalid")

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def word_count(self) -> int:
        return len(_WORD.findall(self.text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "section_path": list(self.section_path),
            "section_order": self.section_order,
            "paragraph_index_in_section": self.paragraph_index_in_section,
            "document_paragraph_index": self.document_paragraph_index,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "word_count": self.word_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodyParagraph":
        paragraph = cls(
            section_id=str(value["section_id"]),
            section_title=str(value["section_title"]),
            section_path=tuple(int(item) for item in value.get("section_path", ())),
            section_order=int(value["section_order"]),
            paragraph_index_in_section=int(value["paragraph_index_in_section"]),
            document_paragraph_index=int(value["document_paragraph_index"]),
            text=str(value["text"]),
        )
        if value.get("text_sha256", paragraph.text_sha256) != paragraph.text_sha256:
            raise BodyCorpusError("cached body paragraph hash mismatch")
        if int(value.get("word_count", paragraph.word_count)) != paragraph.word_count:
            raise BodyCorpusError("cached body paragraph word count mismatch")
        return paragraph


@dataclass(frozen=True)
class BodyExtraction:
    source_format: str
    title: str
    paragraphs: tuple[BodyParagraph, ...]
    source_sha256: str
    provenance: Mapping[str, Any]
    excluded_regions: tuple[str, ...] = EXCLUDED_REGIONS

    def __post_init__(self) -> None:
        if self.source_format not in {"arxiv_tex", "pmc_jats"}:
            raise ValueError("unsupported main-body source format")
        if self.excluded_regions != EXCLUDED_REGIONS:
            raise ValueError("main-body exclusion policy must be exact")
        if tuple(item.document_paragraph_index for item in self.paragraphs) != tuple(
            range(1, len(self.paragraphs) + 1)
        ):
            raise ValueError("body paragraph indices must be contiguous")

    @property
    def body_text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    @property
    def body_sha256(self) -> str:
        return sha256_text(self.body_text)

    @property
    def word_count(self) -> int:
        return sum(paragraph.word_count for paragraph in self.paragraphs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BODY_EXTRACTION_SCHEMA,
            "source_format": self.source_format,
            "title": self.title,
            "source_sha256": self.source_sha256,
            "body_sha256": self.body_sha256,
            "paragraph_count": len(self.paragraphs),
            "word_count": self.word_count,
            "excluded_regions": list(self.excluded_regions),
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodyExtraction":
        if value.get("schema_version") != BODY_EXTRACTION_SCHEMA:
            raise BodyCorpusError("cached body extraction has the wrong schema")
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            raise BodyCorpusError("cached body extraction provenance must be an object")
        raw_paragraphs = value.get("paragraphs")
        if not isinstance(raw_paragraphs, list) or not all(
            isinstance(item, Mapping) for item in raw_paragraphs
        ):
            raise BodyCorpusError("cached body extraction paragraphs must be objects")
        extraction = cls(
            source_format=str(value["source_format"]),
            title=str(value.get("title", "")),
            paragraphs=tuple(BodyParagraph.from_dict(item) for item in raw_paragraphs),
            source_sha256=str(value["source_sha256"]),
            provenance=dict(provenance),
            excluded_regions=tuple(str(item) for item in value.get("excluded_regions", ())),
        )
        for key, observed in (
            ("body_sha256", extraction.body_sha256),
            ("paragraph_count", len(extraction.paragraphs)),
            ("word_count", extraction.word_count),
        ):
            if key in value and value[key] != observed:
                raise BodyCorpusError(f"cached body extraction {key} mismatch")
        return extraction


@dataclass(frozen=True)
class BodySource:
    document: SourceDocument
    extraction: BodyExtraction
    artifact_sha256: str
    structure_sha256: str

    def __post_init__(self) -> None:
        if self.document.abstract:
            raise ValueError("body-only SourceDocument abstract must be empty")
        if self.document.body != self.extraction.body_text:
            raise ValueError("body-only SourceDocument does not match its extraction")
        if not self.document.body:
            raise ValueError("body-only SourceDocument must contain usable body prose")


@dataclass
class _SectionBuffer:
    title: str
    path: tuple[int, ...]
    order: int
    level: int
    included: bool
    exclusion_reason: str
    raw_parts: list[str] = field(default_factory=list)
    clean_paragraphs: list[str] = field(default_factory=list)

    @property
    def section_id(self) -> str:
        return "s" + ".".join(str(value) for value in self.path) if self.path else "body"


_ABSTRACT_TITLE = re.compile(r"^(?:abstract|summary)$", re.IGNORECASE)
_REFERENCE_TITLE = re.compile(
    r"^(?:references?|bibliography|works cited|literature cited|references? and notes?)$",
    re.IGNORECASE,
)
_ACK_TITLE = re.compile(
    r"^(?:acknowledg(?:e)?ments?(?: and funding)?|funding and acknowledg(?:e)?ments?)$",
    re.IGNORECASE,
)


def _body_classification(
    title: str,
    *,
    appendix_mode: bool = False,
    explicit: str = "",
) -> tuple[bool, str]:
    if explicit:
        return False, explicit
    normal = " ".join(title.strip(" .:").casefold().split())
    if _ABSTRACT_TITLE.fullmatch(normal):
        return False, "abstract"
    if _REFERENCE_TITLE.fullmatch(normal):
        return False, "references"
    if _ACK_TITLE.fullmatch(normal):
        return False, "acknowledgments"
    included, reason = sx._classify_title(title, appendix_mode=appendix_mode)
    return included, reason


_DROP_TEX_COMMANDS = {
    "address", "affiliation", "author", "bibliography", "cite", "citep", "citet",
    "date", "email", "eqref", "footnotemark", "href", "includegraphics", "index",
    "institute", "keywords", "keyword", "label", "pageref", "ref", "title", "url",
}
_LINE_TEX_COMMANDS = {
    "bigskip", "item", "linebreak", "medskip", "newline", "noindent", "par", "smallskip",
}


def _skip_balanced(value: str, start: int, opener: str, closer: str) -> int:
    cursor = start
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if cursor >= len(value) or value[cursor] != opener:
        return start
    depth = 0
    index = cursor
    while index < len(value):
        if value[index] == "\\":
            # An escaped delimiter is content, not TeX grouping. Advancing two
            # positions also keeps the scan strictly linear for long runs of
            # escaped braces.
            index += 2
            continue
        if value[index] == opener:
            depth += 1
        elif value[index] == closer:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return start


def _linear_plain_tex_without_dollar_math(value: str) -> str:
    """Strip TeX controls in one pass; never apply a cross-document math regex."""

    output: list[str] = []
    citation_index = 0
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character == "%" and (cursor == 0 or value[cursor - 1] != "\\"):
            newline = value.find("\n", cursor + 1)
            cursor = len(value) if newline < 0 else newline
            continue
        if character != "\\":
            if character in "{}":
                cursor += 1
                continue
            if character == "~":
                output.append(" ")
            else:
                output.append(character)
            cursor += 1
            continue
        if cursor + 1 >= len(value):
            output.append(" ")
            break
        escaped = value[cursor + 1]
        if escaped in "&%_#{}$":
            output.append(escaped)
            cursor += 2
            continue
        if escaped in "[(":
            closer = "]" if escaped == "[" else ")"
            end_token = "\\" + closer
            end = value.find(end_token, cursor + 2)
            cursor = len(value) if end < 0 else end + 2
            output.append(" ")
            continue
        if not (escaped.isalpha() or escaped == "@"):
            output.append(" ")
            cursor += 2
            continue
        end = cursor + 2
        while end < len(value) and (value[end].isalpha() or value[end] == "@"):
            end += 1
        command = value[cursor + 1 : end].casefold()
        if end < len(value) and value[end] == "*":
            end += 1
        if command in _LINE_TEX_COMMANDS:
            output.append("\n")
            cursor = end
            continue
        if command in {"begin", "end"}:
            skipped = _skip_balanced(value, end, "{", "}")
            cursor = skipped if skipped != end else end
            output.append(" ")
            continue
        if command in _DROP_TEX_COMMANDS or command.startswith("cite"):
            optional = _skip_balanced(value, end, "[", "]")
            while optional != end:
                end = optional
                optional = _skip_balanced(value, end, "[", "]")
            first = _skip_balanced(value, end, "{", "}")
            if first != end:
                end = first
                # href has a second argument containing visible anchor prose;
                # preserving it is preferable to counting its URL.
                if command == "href":
                    second = sx._balanced_argument(value, end, "{", "}")
                    if second is not None:
                        output.append(second[0])
                        end = second[1]
            if command.startswith("cite"):
                citation_index += 1
                output.append(f" [{citation_index}] ")
            output.append(" ")
            cursor = end
            continue
        # Formatting and unknown control words are removed while their braced
        # textual arguments remain in the stream and are handled normally.
        output.append(" ")
        cursor = end
    return "".join(output)


def _linear_plain_tex(value: str) -> str:
    """Linear-time TeX prose cleanup with recoverable unbalanced ``$`` input."""

    output: list[str] = []
    pending: list[str] = []
    math_delimiter = ""
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "$" and (cursor == 0 or value[cursor - 1] != "\\"):
            delimiter = "$$" if value.startswith("$$", cursor) else "$"
            if not math_delimiter:
                math_delimiter = delimiter
                pending.clear()
                cursor += len(delimiter)
                continue
            if delimiter == math_delimiter:
                math_delimiter = ""
                pending.clear()
                output.append(" ")
                cursor += len(delimiter)
                continue
        if math_delimiter:
            pending.append(value[cursor])
        else:
            output.append(value[cursor])
        cursor += 1
    # An unmatched dollar is treated as punctuation rather than causing the
    # rest of a paper to vanish. The pending suffix is cleaned once, linearly.
    if math_delimiter and pending:
        output.append(" ")
        output.extend(pending)
    plain = _linear_plain_tex_without_dollar_math("".join(output))
    plain = plain.replace("``", '"').replace("''", '"')
    plain = re.sub(r"[ \t\f\v]+", " ", plain)
    plain = re.sub(r" *\n *", "\n", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return _normalize_citation_spacing(plain.strip())


def _normalize_citation_spacing(value: str) -> str:
    previous = ""
    while previous != value:
        previous = value
        value = re.sub(r"\[\s*\[(\d+)\]\s*\]", r"[\1]", value)
        value = re.sub(
            r"\[\s*((?:\[\d+\](?:\s*[,;–-]\s*)?)+)\s*\]",
            lambda match: match.group(1).strip(),
            value,
        )
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([([])\s+", r"\1", value)
    value = re.sub(r"\s+([)\]])", r"\1", value)
    return value


def _linear_tex_title(value: str) -> str:
    if _tex_raw_rejection_reason(value):
        return "Untitled section"
    title = " ".join(_linear_plain_tex(value).split()).strip(" .")
    return (
        title
        if not _residual_artifact_reason(title, require_terminal=False)
        else "Untitled section"
    )


def _raw_tex_units(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return [
        unit.strip()
        for unit in re.split(r"(?:\n[ \t]*\n|\\par\b)", normalized)
        if unit.strip()
    ]


def _tex_raw_rejection_reason(value: str) -> str:
    """Classify unsafe source constructs with one bounded left-to-right scan."""

    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character == "%":
            newline = value.find("\n", cursor + 1)
            cursor = len(value) if newline < 0 else newline + 1
            continue
        if character == "$":
            return "inline_math"
        if character != "\\":
            cursor += 1
            continue
        if cursor + 1 >= len(value):
            return "residual_tex_control"
        escaped = value[cursor + 1]
        if escaped in "([":
            return "inline_math"
        if not (escaped.isalpha() or escaped == "@"):
            # Escaped punctuation/currency is literal text. A doubled slash is
            # consumed here, so a following dollar remains visible as math.
            cursor += 2
            continue
        end = cursor + 2
        while end < len(value) and (value[end].isalpha() or value[end] == "@"):
            end += 1
        command = value[cursor + 1 : end].casefold()
        if command in _REFERENCE_COMMANDS or command.endswith("ref"):
            return "object_cross_reference"
        if command in _MATH_COMMANDS:
            return "inline_math_command"
        if command.startswith("cite") or command in _SAFE_TEX_COMMANDS:
            cursor = end
            continue
        return "unsupported_tex_command"
    return ""


def _residual_artifact_reason(value: str, *, require_terminal: bool = True) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return "empty_after_cleaning"
    if _DISPLAY_MATH_SENTINEL in normalized:
        return "display_math"
    if _RESIDUAL_TEX.search(normalized):
        return "residual_tex"
    if _EMPTY_OBJECT.search(normalized):
        return "missing_object"
    if _DANGLING_LINK.search(normalized):
        return "dangling_function_word"
    if _ACKNOWLEDGMENT_PROSE.search(normalized):
        return "acknowledgment_prose"
    if _KNOWN_SOURCE_TEXT_DEFECT.search(normalized):
        return "known_source_text_defect"
    if require_terminal and not _TERMINAL_PUNCTUATION.search(normalized):
        return "missing_terminal_punctuation"
    if re.search(r"(?:^|\s)[-–—]\s*(?:dimensional|dim\b)", normalized, re.IGNORECASE):
        return "missing_dimension"
    return ""


def _tex_document_title(value: str) -> str:
    match = re.search(r"\\title\*?\s*\{", value, flags=re.IGNORECASE)
    if match is None:
        return ""
    argument = sx._balanced_argument(value, match.end() - 1, "{", "}")
    return _linear_tex_title(argument[0]) if argument is not None else ""


class _TexBodyParser:
    def __init__(self, *, limits: sx.StructureLimits) -> None:
        self.limits = limits
        self.sections: list[_SectionBuffer] = []
        self.current: _SectionBuffer | None = None
        self.unsectioned = _SectionBuffer(
            "Unsectioned front matter", (), 0, 1, False, "front_matter"
        )
        self.normal_stack: list[_SectionBuffer] = []
        self.appendix_stack: list[_SectionBuffer] = []
        self.normal_roots = 0
        self.appendix_roots = 0
        self.appendix_mode = False
        self.order = 0
        self.candidate_paragraphs = 0
        self.rejections: Counter[str] = Counter()

    def start_appendix(self) -> None:
        self.appendix_mode = True
        self.current = None

    def heading(self, title: str, level: int, *, explicit: str = "") -> _SectionBuffer:
        self.order += 1
        if self.order > self.limits.max_sections:
            raise BodyCorpusError("paper exceeds the body section-count limit")
        included, reason = _body_classification(
            title,
            appendix_mode=self.appendix_mode,
            explicit=explicit,
        )
        if reason == "appendix":
            self.appendix_mode = True
        stack = self.appendix_stack if reason == "appendix" else self.normal_stack
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            parent = stack[-1]
            sibling_count = sum(
                section.path[:-1] == parent.path and len(section.path) == len(parent.path) + 1
                for section in self.sections
            )
            path = parent.path + (sibling_count + 1,)
            if not parent.included:
                included, reason = False, parent.exclusion_reason
        elif reason == "appendix":
            self.appendix_roots += 1
            path = (self.appendix_roots,)
        else:
            self.normal_roots += 1
            path = (self.normal_roots,)
        section = _SectionBuffer(title or "Untitled section", path, self.order, level, included, reason)
        self.sections.append(section)
        stack.append(section)
        self.current = section
        return section

    def content(self, value: str) -> None:
        if not value:
            return
        if self.current is not None:
            self.current.raw_parts.append(value)
        elif not self.appendix_mode:
            self.unsectioned.raw_parts.append(value)

    def reject_current_environment(self) -> None:
        if self.current is not None and self.current.included:
            self.current.raw_parts.append(f" {_DISPLAY_MATH_SENTINEL} ")

    def paragraphs(self) -> tuple[BodyParagraph, ...]:
        buffers = self.sections
        result: list[BodyParagraph] = []
        for section in buffers:
            if not section.included:
                continue
            raw_units = _raw_tex_units("".join(section.raw_parts))
            sentinel_units = {
                index
                for index, unit in enumerate(raw_units)
                if _DISPLAY_MATH_SENTINEL in unit
            }
            for raw_index, raw_unit in enumerate(raw_units):
                self.candidate_paragraphs += 1
                if raw_index in sentinel_units:
                    self.rejections["display_math"] += 1
                    continue
                if raw_index - 1 in sentinel_units or raw_index + 1 in sentinel_units:
                    self.rejections["display_math_context"] += 1
                    continue
                rejection = _tex_raw_rejection_reason(raw_unit)
                if rejection:
                    self.rejections[rejection] += 1
                    continue
                cleaned_units = paragraphs(_linear_plain_tex(raw_unit))
                if len(cleaned_units) != 1:
                    self.rejections["not_one_usable_paragraph"] += 1
                    continue
                cleaned = cleaned_units[0]
                rejection = _residual_artifact_reason(cleaned)
                if rejection:
                    self.rejections[rejection] += 1
                    continue
                section.clean_paragraphs.append(cleaned)
            for section_index, text in enumerate(section.clean_paragraphs, start=1):
                result.append(BodyParagraph(
                    section_id=section.section_id,
                    section_title=section.title,
                    section_path=section.path,
                    section_order=section.order,
                    paragraph_index_in_section=section_index,
                    document_paragraph_index=len(result) + 1,
                    text=text,
                ))
        return tuple(result)


_BODY_SKIP_ENVIRONMENTS = (
    sx._MATH_ENVIRONMENTS
    | sx._FIGURE_ENVIRONMENTS
    | sx._TABLE_ENVIRONMENTS
    | sx._ACK_ENVIRONMENTS
    | {
        "abstract",
        "keywords",
        "keyword",
        "thebibliography",
        "titlepage",
        "frontmatter",
        "tabular",
        "tabular*",
    }
)


def extract_tex_body(
    payload: bytes,
    *,
    limits: sx.StructureLimits = sx.DEFAULT_LIMITS,
) -> BodyExtraction:
    """Extract structurally main-body TeX prose through the bounded source tree."""

    tree = sx._read_tex_source_tree(payload, limits)
    expanded, included_members, include_warnings = sx._expand_tex_tree(tree, limits)
    title = _tex_document_title(expanded)
    document_class = sx._tex_document_class(expanded)
    has_chapters = bool(re.search(r"\\chapter\*?\s*\{", expanded, flags=re.IGNORECASE)) or any(
        value in document_class for value in ("book", "report", "memoir")
    )
    levels = {
        "chapter": 1,
        "section": 2 if has_chapters else 1,
        "subsection": 3 if has_chapters else 2,
        "subsubsection": 4 if has_chapters else 3,
        "paragraph": 5 if has_chapters else 4,
    }
    body = sx._document_body(expanded)
    parser = _TexBodyParser(limits=limits)
    warnings = list(include_warnings)
    cursor = 0
    while True:
        match = sx._TEX_SIGNAL.search(body, cursor)
        if match is None:
            parser.content(body[cursor:])
            break
        parser.content(body[cursor : match.start()])
        command = match.group("command").casefold()
        if command in levels:
            argument = sx._heading_argument(body, match.end())
            if argument is None:
                warnings.append(f"malformed_heading:{command}:{match.start()}")
                parser.content(body[match.start() : match.end()])
                cursor = match.end()
                continue
            parser.heading(_linear_tex_title(argument[0]) or "Untitled section", levels[command])
            cursor = argument[1]
            continue
        if command == "appendix":
            parser.start_appendix()
            cursor = match.end()
            continue
        if command in {"bibliography", "printbibliography"}:
            parser.heading("References", 1, explicit="references")
            argument = sx._heading_argument(body, match.end()) if command == "bibliography" else None
            cursor = argument[1] if argument else match.end()
            continue
        environment_argument = sx._environment_argument(body, match.end())
        if environment_argument is None:
            parser.content(body[match.start() : match.end()])
            cursor = match.end()
            continue
        environment = environment_argument[0].strip().casefold()
        content_start = environment_argument[1]
        if environment in {"appendix", "appendices"}:
            parser.start_appendix()
            cursor = content_start
            continue
        if environment not in _BODY_SKIP_ENVIRONMENTS:
            parser.content(body[match.start() : content_start])
            cursor = content_start
            continue
        environment_end = sx._environment_end(body, environment, content_start)
        if environment_end is None:
            warnings.append(f"unterminated_environment:{environment}:{match.start()}")
            content_end, after_end = len(body), len(body)
        else:
            content_end, after_end = environment_end
        if environment in sx._ACK_ENVIRONMENTS:
            parser.heading("Acknowledgments", 1, explicit="acknowledgments")
        elif environment == "thebibliography":
            parser.heading("References", 1, explicit="references")
        elif environment in sx._MATH_ENVIRONMENTS:
            parser.reject_current_environment()
        # Every skipped environment, including abstract/captions/tables/math, is
        # discarded rather than represented as prose.
        _ = content_end
        cursor = after_end
    body_paragraphs = parser.paragraphs()
    extracted = BodyExtraction(
        source_format="arxiv_tex",
        title=title,
        paragraphs=body_paragraphs,
        source_sha256=_sha256_bytes(payload),
        provenance={
            "parser": "spiral_tex_main_body",
            "parser_version": "1",
            "root_member": tree.root,
            "included_members": list(included_members),
            "expanded_source_sha256": sha256_text(expanded),
            "warnings": warnings,
            "candidate_body_paragraphs": parser.candidate_paragraphs,
            "accepted_body_paragraphs": len(body_paragraphs),
            "rejected_body_paragraphs": sum(parser.rejections.values()),
            "paragraph_rejection_counts": dict(sorted(parser.rejections.items())),
            "excluded_regions": list(EXCLUDED_REGIONS),
            "network_requests": 0,
        },
    )
    return extracted


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _first_direct(node: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in node if _local_name(child.tag) == name), None)


def _jats_section_classification(
    section: ET.Element,
    title: str,
    *,
    parent: _SectionBuffer | None,
) -> tuple[bool, str]:
    sec_type = (section.attrib.get("sec-type") or "").strip().casefold()
    explicit = ""
    if sec_type in {"abstract", "summary"}:
        explicit = "abstract"
    elif sec_type in {"ack", "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}:
        explicit = "acknowledgments"
    elif sec_type in {"ref", "refs", "references", "bibliography"}:
        explicit = "references"
    elif sec_type in {"app", "appendix", "supplement", "supplementary-material"}:
        explicit = "appendix"
    included, reason = _body_classification(title, explicit=explicit)
    if parent is not None and not parent.included:
        return False, parent.exclusion_reason
    return included, reason


def _jats_paragraph_text(element: ET.Element) -> tuple[str, str]:
    for node in element.iter():
        tag = _local_name(node.tag)
        if tag in _JATS_MATH_TAGS:
            return "", "inline_or_display_math"
        if tag in {"graphic", "inline-graphic", "media"}:
            return "", "inline_media"
        if tag == "xref":
            ref_type = (node.attrib.get("ref-type") or "").strip().casefold()
            if ref_type != "bibr":
                reason = (
                    "object_cross_reference"
                    if not ref_type or ref_type in _JATS_OBJECT_REFERENCE_TYPES
                    else "unsupported_cross_reference"
                )
                return "", reason

    citation_index = 0

    def visit(node: ET.Element) -> str:
        nonlocal citation_index
        pieces = [node.text or ""]
        for child in node:
            if _local_name(child.tag) == "xref":
                citation_index += 1
                pieces.append(f" [{citation_index}] ")
            else:
                pieces.append(visit(child))
            pieces.append(child.tail or "")
        return "".join(pieces)

    text = _normalize_citation_spacing(re.sub(r"\s+", " ", visit(element)).strip())
    return text, _residual_artifact_reason(text)


class _JatsBodyParser:
    def __init__(self, *, limits: sx.StructureLimits) -> None:
        self.limits = limits
        self.sections: list[_SectionBuffer] = []
        self.unsectioned = _SectionBuffer(
            "Unsectioned front matter", (), 0, 1, False, "front_matter"
        )
        self.order = 0
        self.root_count = 0
        self.child_counts: Counter[tuple[int, ...]] = Counter()
        self.candidate_paragraphs = 0
        self.rejections: Counter[str] = Counter()

    def _new_section(
        self,
        element: ET.Element,
        *,
        path: tuple[int, ...],
        parent: _SectionBuffer | None,
    ) -> _SectionBuffer:
        self.order += 1
        if self.order > self.limits.max_sections:
            raise BodyCorpusError("paper exceeds the body section-count limit")
        title_node = _first_direct(element, "title")
        title = sx._jats_text(title_node) if title_node is not None else "Untitled section"
        included, reason = _jats_section_classification(element, title, parent=parent)
        section = _SectionBuffer(title, path, self.order, len(path), included, reason)
        self.sections.append(section)
        return section

    def _paragraph(
        self,
        element: ET.Element,
        section: _SectionBuffer,
        *,
        forced_rejection: str = "",
    ) -> None:
        if not section.included:
            return
        self.candidate_paragraphs += 1
        if forced_rejection:
            self.rejections[forced_rejection] += 1
            return
        text, rejection = _jats_paragraph_text(element)
        if rejection:
            self.rejections[rejection] += 1
            return
        clean_units = paragraphs(text)
        if len(clean_units) != 1:
            self.rejections["not_one_usable_paragraph"] += 1
            return
        section.clean_paragraphs.append(clean_units[0])

    def _walk_content(
        self,
        container: ET.Element,
        section: _SectionBuffer,
    ) -> None:
        children = list(container)
        formula_positions = {
            index
            for index, child in enumerate(children)
            if any(_local_name(node.tag) in _JATS_MATH_TAGS for node in child.iter())
        }
        for index, child in enumerate(children):
            tag = _local_name(child.tag)
            if tag in {"title", "label"}:
                continue
            if tag == "sec":
                self.child_counts[section.path] += 1
                nested = self._new_section(
                    child,
                    path=section.path + (self.child_counts[section.path],),
                    parent=section,
                )
                self._walk_content(child, nested)
            elif tag == "p":
                adjacent_formula = index - 1 in formula_positions or index + 1 in formula_positions
                self._paragraph(
                    child,
                    section,
                    forced_rejection="display_math_context" if adjacent_formula else "",
                )
            elif tag in {
                "abstract",
                "ack",
                "ref-list",
                "app",
                "app-group",
                "supplementary-material",
                "fig",
                "fig-group",
                "table",
                "table-wrap",
                "table-wrap-group",
                "boxed-text",
            }:
                continue
            else:
                self._walk_content(child, section)

    def _walk_body_root(self, container: ET.Element) -> None:
        """Discover section roots through publisher-specific body wrappers."""

        for child in container:
            tag = _local_name(child.tag)
            if tag == "sec":
                self.root_count += 1
                section = self._new_section(child, path=(self.root_count,), parent=None)
                self._walk_content(child, section)
            elif tag == "p":
                # Direct pre-section body material is commonly publisher front
                # matter; only explicitly sectioned main-body prose is admitted.
                continue
            elif tag in {
                "abstract",
                "ack",
                "ref-list",
                "app",
                "app-group",
                "supplementary-material",
                "fig",
                "fig-group",
                "table",
                "table-wrap",
                "table-wrap-group",
            }:
                continue
            else:
                # Some publishers wrap body sections in custom grouping tags.
                self._walk_body_root(child)

    def body(self, body: ET.Element) -> None:
        self._walk_body_root(body)

    def paragraphs(self) -> tuple[BodyParagraph, ...]:
        result: list[BodyParagraph] = []
        for section in self.sections:
            if not section.included:
                continue
            for section_index, text in enumerate(section.clean_paragraphs, start=1):
                result.append(BodyParagraph(
                    section_id=section.section_id,
                    section_title=section.title,
                    section_path=section.path,
                    section_order=section.order,
                    paragraph_index_in_section=section_index,
                    document_paragraph_index=len(result) + 1,
                    text=text,
                ))
        return tuple(result)


def _jats_publication_attestation(article: ET.Element) -> dict[str, Any]:
    article_meta = next(
        (node for node in article.iter() if _local_name(node.tag) == "article-meta"),
        None,
    )
    if article_meta is None:
        return {
            "schema_version": BODY_TEMPORAL_ATTESTATION_SCHEMA,
            "article_ids": {},
            "publication_dates": [],
            "copyright_years": [],
            "attested_years": [],
        }

    article_ids: dict[str, list[str]] = defaultdict(list)
    for node in article_meta.iter():
        if _local_name(node.tag) != "article-id":
            continue
        kind = (node.attrib.get("pub-id-type") or "").strip().casefold()
        identifier = sx._jats_text(node).strip()
        if kind and identifier and identifier not in article_ids[kind]:
            article_ids[kind].append(identifier)

    publication_dates: list[dict[str, Any]] = []
    copyright_years: set[int] = set()

    def direct_text(node: ET.Element, name: str) -> str:
        child = _first_direct(node, name)
        return sx._jats_text(child).strip() if child is not None else ""

    def visit(node: ET.Element, *, in_history: bool = False) -> None:
        tag = _local_name(node.tag)
        history = in_history or tag == "history"
        if tag in {"pub-date", "article-date"} and not history:
            raw_year = direct_text(node, "year")
            iso = (node.attrib.get("iso-8601-date") or "").strip()
            if not raw_year and re.match(r"^\d{4}", iso):
                raw_year = iso[:4]
            if raw_year.isdigit() and 1000 <= int(raw_year) <= 9999:
                raw_month = direct_text(node, "month")
                month = (
                    int(raw_month)
                    if raw_month.isdigit() and 1 <= int(raw_month) <= 12
                    else _MONTHS.get(raw_month.casefold(), 0)
                )
                raw_day = direct_text(node, "day")
                day = int(raw_day) if raw_day.isdigit() and 1 <= int(raw_day) <= 31 else 0
                publication_dates.append({
                    "element": tag,
                    "date_type": str(
                        node.attrib.get("pub-type")
                        or node.attrib.get("date-type")
                        or node.attrib.get("publication-format")
                        or ""
                    ).casefold(),
                    "year": int(raw_year),
                    "month": month,
                    "day": day,
                    "iso_8601": iso,
                })
        elif tag == "copyright-year" and not history:
            raw_year = sx._jats_text(node).strip()
            if raw_year.isdigit() and 1000 <= int(raw_year) <= 9999:
                copyright_years.add(int(raw_year))
        for child in node:
            visit(child, in_history=history)

    visit(article_meta)
    publication_dates.sort(
        key=lambda item: (
            item["year"], item["month"], item["day"], item["element"], item["date_type"]
        )
    )
    years = sorted(
        {int(item["year"]) for item in publication_dates} | copyright_years
    )
    return {
        "schema_version": BODY_TEMPORAL_ATTESTATION_SCHEMA,
        "article_ids": {
            key: sorted(values) for key, values in sorted(article_ids.items())
        },
        "publication_dates": publication_dates,
        "copyright_years": sorted(copyright_years),
        "attested_years": years,
    }


def extract_jats_body(
    payload: bytes,
    *,
    limits: sx.StructureLimits = sx.DEFAULT_LIMITS,
) -> BodyExtraction:
    """Extract only paragraphs inside included PMC/JATS body sections."""

    structure = sx.parse_jats_structure(payload, limits=limits)
    xml_payload, doctypes_removed = sx._strip_external_doctypes(payload)
    root = ET.fromstring(xml_payload)
    article = root if _local_name(root.tag) == "article" else next(
        (node for node in root.iter() if _local_name(node.tag) == "article"),
        None,
    )
    if article is None:
        raise BodyCorpusError("JATS payload contains no article")
    temporal_attestation = _jats_publication_attestation(article)
    body = _first_direct(article, "body")
    parser = _JatsBodyParser(limits=limits)
    if body is not None:
        parser.body(body)
    warnings = list(structure.provenance.warnings)
    if doctypes_removed and "external_doctype_ignored" not in warnings:
        warnings.append("external_doctype_ignored")
    body_paragraphs = parser.paragraphs()
    return BodyExtraction(
        source_format="pmc_jats",
        title=structure.title,
        paragraphs=body_paragraphs,
        source_sha256=_sha256_bytes(payload),
        provenance={
            "parser": "spiral_jats_main_body",
            "parser_version": "1",
            "structure_parser": structure.provenance.to_dict(),
            "warnings": warnings,
            "jats_temporal_attestation": temporal_attestation,
            "candidate_body_paragraphs": parser.candidate_paragraphs,
            "accepted_body_paragraphs": len(body_paragraphs),
            "rejected_body_paragraphs": sum(parser.rejections.values()),
            "paragraph_rejection_counts": dict(sorted(parser.rejections.items())),
            "excluded_regions": list(EXCLUDED_REGIONS),
            "network_requests": 0,
        },
    )


def extract_body(
    payload: bytes,
    source_format: str,
    *,
    limits: sx.StructureLimits = sx.DEFAULT_LIMITS,
) -> BodyExtraction:
    if source_format == "arxiv_tex":
        return extract_tex_body(payload, limits=limits)
    if source_format == "pmc_jats":
        return extract_jats_body(payload, limits=limits)
    raise BodyCorpusError(f"unsupported cached source format: {source_format}")


def _record_relative_paths(document: SourceDocument) -> tuple[Path, Path]:
    stratum = re.sub(r"[^A-Za-z0-9_.-]+", "_", document.stratum)
    identity = sha256_text(document.document_id)
    return Path("raw") / stratum / f"{identity}.json", Path("provenance") / stratum / f"{identity}.json"


def _crosslist_identity(document: SourceDocument) -> dict[str, Any]:
    """Return content identity after removing collection-only cross-list fields."""

    value = document.to_dict()
    for key in ("stratum", "query", "content_sha256", "raw_record_sha256"):
        value.pop(key, None)
    return value


def _deduplicate_crosslists(
    documents: Iterable[SourceDocument],
) -> tuple[list[SourceDocument], list[dict[str, Any]]]:
    """Collapse byte-identical arXiv cross-lists, failing closed on disagreement."""

    grouped: dict[str, list[SourceDocument]] = defaultdict(list)
    for document in documents:
        grouped[document.document_id].append(document)
    selected: list[SourceDocument] = []
    duplicates: list[dict[str, Any]] = []
    for document_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: (item.stratum, item.document_id))
        winner = ordered[0]
        if len(ordered) > 1:
            strata = [item.stratum for item in ordered]
            if (
                winner.provider != "arxiv"
                or len(set(strata)) != len(strata)
                or any(
                    _crosslist_identity(item) != _crosslist_identity(winner)
                    for item in ordered[1:]
                )
            ):
                raise BodyCorpusError(
                    f"duplicate source documents disagree for {document_id}: "
                    + ", ".join(strata)
                )
            duplicates.append({
                "document_id": document_id,
                "observed_strata": strata,
                "selected_stratum": winner.stratum,
                "duplicates_removed": len(ordered) - 1,
            })
        selected.append(winner)
    return selected, duplicates


def _jats_temporal_receipt(
    document: SourceDocument,
    extraction: BodyExtraction,
) -> dict[str, Any]:
    raw = extraction.provenance.get("jats_temporal_attestation")
    if not isinstance(raw, Mapping):
        raise BodyCorpusError("PMC/JATS extraction lacks temporal attestation")
    article_ids = raw.get("article_ids")
    years_value = raw.get("attested_years")
    if not isinstance(article_ids, Mapping) or not isinstance(years_value, list):
        raise BodyCorpusError("PMC/JATS temporal attestation is malformed")
    years = sorted({int(value) for value in years_value})
    pmids = [str(value) for value in article_ids.get("pmid", ())]
    pmcids = [str(value).upper() for value in article_ids.get("pmcid", ())]
    expected_pmcid_match = re.search(r"/articles/(PMC\d+)/?", document.artifact_url, re.I)
    expected_pmcid = expected_pmcid_match.group(1).upper() if expected_pmcid_match else ""
    metadata_year = date.fromisoformat(document.published).year
    query_match = _QUERY_DATE_RANGE.search(document.query)
    query_years = (
        [int(query_match.group(1)), int(query_match.group(2))]
        if query_match is not None
        else []
    )
    flags: list[str] = []
    if pmids != [document.source_id]:
        flags.append("pmid_identity_mismatch")
    if expected_pmcid and expected_pmcid not in pmcids:
        flags.append("pmcid_identity_mismatch")
    if not years:
        flags.append("publication_year_unattested")
    if any(year > BODY_CUTOFF.year for year in years):
        flags.append("post_cutoff_publication_or_copyright_year")
    if metadata_year not in years:
        flags.append("metadata_publication_year_mismatch")
    if not query_years or not any(query_years[0] <= year <= query_years[1] for year in years):
        flags.append("query_publication_range_mismatch")
    return {
        "document_id": document.document_id,
        "metadata_published": document.published,
        "metadata_latest_version": document.latest_version,
        "metadata_query_year_range": query_years,
        "jats_attested_years": years,
        "jats_publication_dates": list(raw.get("publication_dates", ())),
        "jats_copyright_years": list(raw.get("copyright_years", ())),
        "jats_pmids": pmids,
        "jats_pmcids": pmcids,
        "expected_pmcid": expected_pmcid,
        "eligible": not flags,
        "rejection_flags": flags,
    }


def build_body_source_cache(
    metadata_documents: Iterable[SourceDocument],
    *,
    structure_cache: Path,
    body_cache: Path,
    limits: sx.StructureLimits = sx.DEFAULT_LIMITS,
) -> tuple[list[BodySource], dict[str, Any]]:
    """Build body-only SourceDocuments from ready, hash-checked local artifacts."""

    input_documents = sorted(
        metadata_documents, key=lambda item: (item.stratum, item.document_id)
    )
    documents, crosslist_duplicates = _deduplicate_crosslists(input_documents)
    documents.sort(key=lambda item: (item.stratum, item.document_id))
    records = load_structure_cache(structure_cache)
    artifacts = structure_cache / "artifacts"
    sources: list[BodySource] = []
    inventory: list[dict[str, Any]] = []
    skipped = Counter()
    paragraph_quality: dict[str, Counter[str]] = defaultdict(Counter)
    paragraph_rejections: dict[str, Counter[str]] = defaultdict(Counter)
    temporal_receipts: list[dict[str, Any]] = []
    temporal_rejections: Counter[str] = Counter()
    for document in documents:
        record = records.get(document.document_id)
        if record is None:
            raise BodyCorpusError(f"no ready structure artifact for {document.document_id}")
        metadata_hash = str(record.get("metadata_raw_record_sha256", ""))
        if metadata_hash != raw_record_sha256(document):
            raise BodyCorpusError(f"metadata lineage mismatch for {document.document_id}")
        artifact_filename = str(record.get("artifact_filename", ""))
        if (
            not artifact_filename
            or Path(artifact_filename).name != artifact_filename
            or artifact_filename.startswith("._")
        ):
            raise BodyCorpusError(f"unsafe artifact filename for {document.document_id}")
        artifact_path = artifacts / artifact_filename
        payload = artifact_path.read_bytes()
        artifact_hash = _sha256_bytes(payload)
        if artifact_hash != record.get("artifact_sha256"):
            raise BodyCorpusError(f"artifact hash mismatch for {document.document_id}")
        structure = record.get("structure")
        if not isinstance(structure, Mapping):
            raise BodyCorpusError(f"structure record is malformed for {document.document_id}")
        source_format = str(structure.get("source_format", ""))
        extraction = extract_body(payload, source_format, limits=limits)
        if source_format == "pmc_jats":
            receipt = _jats_temporal_receipt(document, extraction)
            temporal_receipts.append(receipt)
            temporal_rejections.update(receipt["rejection_flags"])
            if not receipt["eligible"]:
                primary = str(receipt["rejection_flags"][0])
                skipped[f"temporal:{primary}:{document.stratum}"] += 1
                continue
        quality = paragraph_quality[document.stratum]
        quality["candidates"] += int(extraction.provenance["candidate_body_paragraphs"])
        quality["accepted"] += int(extraction.provenance["accepted_body_paragraphs"])
        quality["rejected"] += int(extraction.provenance["rejected_body_paragraphs"])
        rejection_counts = extraction.provenance.get("paragraph_rejection_counts", {})
        if not isinstance(rejection_counts, Mapping):
            raise BodyCorpusError("body extraction rejection counts must be an object")
        paragraph_rejections[document.stratum].update(
            {str(key): int(value) for key, value in rejection_counts.items()}
        )
        if not extraction.paragraphs:
            skipped[f"no_usable_body:{document.stratum}"] += 1
            continue
        body_document = replace(
            document,
            abstract="",
            body=extraction.body_text,
            content_endpoint=str(record.get("content_endpoint", document.content_endpoint)),
            extraction=f"official_{source_format}_main_body_only_v1",
        )
        source = BodySource(
            document=body_document,
            extraction=extraction,
            artifact_sha256=artifact_hash,
            structure_sha256=str(record.get("structure_sha256", "")),
        )
        source_relative, provenance_relative = _record_relative_paths(body_document)
        source_value = body_document.to_dict()
        provenance_value = {
            "schema_version": BODY_EXTRACTION_SCHEMA,
            "document_id": document.document_id,
            "metadata_raw_record_sha256": metadata_hash,
            "body_source_raw_record_sha256": raw_record_sha256(body_document),
            "artifact_sha256": artifact_hash,
            "structure_sha256": source.structure_sha256,
            "extraction": extraction.to_dict(),
        }
        _atomic_json(body_cache / source_relative, source_value)
        _atomic_json(body_cache / provenance_relative, provenance_value)
        inventory.append({
            "document_id": document.document_id,
            "stratum": document.stratum,
            "source_format": source_format,
            "source_record": source_relative.as_posix(),
            "source_record_sha256": _sha256_file(body_cache / source_relative),
            "provenance_record": provenance_relative.as_posix(),
            "provenance_record_sha256": _sha256_file(body_cache / provenance_relative),
            "artifact_sha256": artifact_hash,
            "structure_sha256": source.structure_sha256,
            "paragraphs": len(extraction.paragraphs),
            "words": extraction.word_count,
        })
        sources.append(source)

    strata = Counter(source.document.stratum for source in sources)
    formats = Counter(source.extraction.source_format for source in sources)
    quality_by_stratum = {
        stratum: {
            "candidate_paragraphs": paragraph_quality[stratum]["candidates"],
            "accepted_paragraphs": paragraph_quality[stratum]["accepted"],
            "rejected_paragraphs": paragraph_quality[stratum]["rejected"],
            "acceptance_ratio": round(
                paragraph_quality[stratum]["accepted"]
                / max(1, paragraph_quality[stratum]["candidates"]),
                6,
            ),
            "rejection_counts": dict(sorted(paragraph_rejections[stratum].items())),
        }
        for stratum in sorted(paragraph_quality)
    }
    manifest = {
        "schema_version": BODY_CACHE_MANIFEST_SCHEMA,
        "counts": {
            "metadata_documents": len(input_documents),
            "unique_metadata_documents": len(documents),
            "cross_list_document_ids": len(crosslist_duplicates),
            "cross_list_duplicates_removed": sum(
                int(item["duplicates_removed"]) for item in crosslist_duplicates
            ),
            "body_source_documents": len(sources),
            "body_paragraphs": sum(len(source.extraction.paragraphs) for source in sources),
            "body_words": sum(source.extraction.word_count for source in sources),
            "by_stratum": dict(sorted(strata.items())),
            "by_source_format": dict(sorted(formats.items())),
            "skipped": dict(sorted(skipped.items())),
        },
        "body_only_attestation": {
            "abstract_fields_empty": all(not source.document.abstract for source in sources),
            "body_only_source_document_ratio": 1.0 if sources else 0.0,
            "excluded_regions": list(EXCLUDED_REGIONS),
            "official_cached_artifacts_only": True,
            "network_requests": 0,
            "paragraph_section_provenance": True,
            "raw_paragraph_quality_gate": True,
        },
        "body_paragraph_quality": {
            "policy": (
                "reject raw paragraphs containing math, object cross-references, "
                "unsupported TeX controls, inline media, or residual missing-object artifacts"
            ),
            "by_stratum": quality_by_stratum,
        },
        "jats_temporal_attestation": {
            "schema_version": BODY_TEMPORAL_ATTESTATION_SCHEMA,
            "cutoff": BODY_CUTOFF.isoformat(),
            "policy": (
                "fail closed unless PMID/PMCID, metadata year, and collection-query "
                "range match the JATS article; reject if any top-level publication, "
                "article, or copyright year is after the cutoff"
            ),
            "artifacts_checked": len(temporal_receipts),
            "eligible_artifacts": sum(
                bool(receipt["eligible"]) for receipt in temporal_receipts
            ),
            "rejected_artifacts": sum(
                not bool(receipt["eligible"]) for receipt in temporal_receipts
            ),
            "rejection_flag_counts": dict(sorted(temporal_rejections.items())),
            "all_included_artifacts_identity_and_date_attested": all(
                receipt["eligible"]
                for receipt in temporal_receipts
                if receipt["eligible"]
            ),
            "receipts": temporal_receipts,
        },
        "input_lineage": {
            "input_metadata_raw_records_sha256": sha256_text("\n".join(sorted(
                raw_record_sha256(document) for document in input_documents
            ))),
            "selected_metadata_raw_records_sha256": sha256_text("\n".join(sorted(
                raw_record_sha256(document) for document in documents
            ))),
            "artifact_sha256": sha256_text("\n".join(sorted(
                source.artifact_sha256 for source in sources
            ))),
            "structure_sha256": sha256_text("\n".join(sorted(
                source.structure_sha256 for source in sources
            ))),
        },
        "cross_list_deduplication": crosslist_duplicates,
        "inventory": inventory,
    }
    _atomic_json(body_cache / "body-cache.manifest.json", manifest)
    return sources, manifest


def load_body_source_cache(body_cache: Path) -> tuple[list[BodySource], dict[str, Any]]:
    manifest_path = body_cache / "body-cache.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BodyCorpusError(f"cannot read body cache manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != BODY_CACHE_MANIFEST_SCHEMA:
        raise BodyCorpusError("body cache manifest has the wrong schema")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise BodyCorpusError("body cache manifest inventory must be an array")
    sources: list[BodySource] = []
    for entry in inventory:
        if not isinstance(entry, Mapping):
            raise BodyCorpusError("body cache inventory entry must be an object")
        source_relative = Path(str(entry.get("source_record", "")))
        provenance_relative = Path(str(entry.get("provenance_record", "")))
        for relative in (source_relative, provenance_relative):
            if relative.is_absolute() or ".." in relative.parts:
                raise BodyCorpusError("body cache inventory contains an unsafe path")
        source_path = body_cache / source_relative
        provenance_path = body_cache / provenance_relative
        if _sha256_file(source_path) != entry.get("source_record_sha256"):
            raise BodyCorpusError("body source record hash mismatch")
        if _sha256_file(provenance_path) != entry.get("provenance_record_sha256"):
            raise BodyCorpusError("body provenance record hash mismatch")
        document = load_source_documents([source_path])[0]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("document_id") != document.document_id:
            raise BodyCorpusError("body cache document identity mismatch")
        extraction_value = provenance.get("extraction")
        if not isinstance(extraction_value, Mapping):
            raise BodyCorpusError("body cache extraction must be an object")
        extraction = BodyExtraction.from_dict(extraction_value)
        sources.append(BodySource(
            document=document,
            extraction=extraction,
            artifact_sha256=str(provenance["artifact_sha256"]),
            structure_sha256=str(provenance["structure_sha256"]),
        ))
    expected = int(manifest.get("counts", {}).get("body_source_documents", -1))
    if expected != len(sources):
        raise BodyCorpusError("body cache manifest document count mismatch")
    return sources, manifest


def _target_at_locator(paragraph: BodyParagraph, locator: str) -> str:
    match = _LOCATOR.fullmatch(locator)
    if not match or int(match.group(1)) != paragraph.document_paragraph_index:
        raise BodyCorpusError(f"invalid compiled body locator: {locator}")
    sentence_index = match.group(2)
    if sentence_index is None:
        return canonicalize_citations(paragraph.text)
    units = sentences(paragraph.text)
    index = int(sentence_index)
    if not 1 <= index <= len(units):
        raise BodyCorpusError(f"compiled sentence locator is out of range: {locator}")
    return canonicalize_citations(units[index - 1])


def _write_quality_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
) -> dict[str, Any]:
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source = row.get("source")
        if not isinstance(source, Mapping):
            raise BodyCorpusError("quality sample row source must be an object")
        by_stratum[str(source.get("stratum", ""))].append(row)
    sample: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda row: sha256_text(
                f"{row.get('example_id', '')}\0body-quality-sample-v1"
            ),
        )[:10]
        for row in ordered:
            provenance = row.get("provenance")
            document = row.get("document")
            if not isinstance(provenance, Mapping) or not isinstance(document, Mapping):
                raise BodyCorpusError("quality sample provenance must be an object")
            target = str(row.get("target", ""))
            reason = _residual_artifact_reason(target)
            if reason:
                raise BodyCorpusError(
                    f"compiled quality gate rejected {row.get('example_id')}: {reason}"
                )
            sample.append({
                "schema_version": "spiral.academic-main-body-quality-sample.v1",
                "example_id": str(row.get("example_id", "")),
                "stratum": stratum,
                "document_id": str(document.get("document_id", "")),
                "task_type": str(row.get("task_type", "")),
                "section_id": str(provenance.get("body_section_id", "")),
                "section_title": str(provenance.get("body_section_title", "")),
                "section_path": list(provenance.get("body_section_path", ())),
                "target": target,
                "automated_artifact_gate": "pass",
            })
    sample.sort(key=lambda item: (item["stratum"], item["example_id"]))
    sample_path = output_path.with_name(f"{output_path.name}.quality-sample.jsonl")
    payload = "".join(canonical_json(item) + "\n" for item in sample).encode("utf-8")
    _atomic_bytes(sample_path, payload)
    return {
        "schema_version": "spiral.academic-main-body-quality-audit.v1",
        "filename": sample_path.name,
        "sha256": _sha256_bytes(payload),
        "sample_examples": len(sample),
        "sample_by_stratum": dict(sorted(Counter(
            item["stratum"] for item in sample
        ).items())),
        "selection": "ten lowest sha256(example_id + body-quality-sample-v1) per stratum",
        "automated_artifact_gate_passed": len(sample),
    }


def _tokenizer_receipt(value: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """Validate the stable receipt required with an injected test tokenizer."""

    if isinstance(value, str):
        if not value.strip():
            raise BodyCorpusError("injected tokenizer_identity must not be empty")
        return {"identity": value.strip(), "loader": "injected"}
    if not isinstance(value, Mapping):
        raise BodyCorpusError(
            "an injected tokenizer requires an explicit tokenizer_identity receipt"
        )
    receipt = dict(value)
    identity = receipt.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise BodyCorpusError(
            "injected tokenizer_identity receipt has no stable identity"
        )
    try:
        canonical_json(receipt)
    except (TypeError, ValueError) as exc:
        raise BodyCorpusError(
            "injected tokenizer_identity receipt is not stable JSON"
        ) from exc
    return receipt


def _training_measurement_row(example: Mapping[str, Any]) -> dict[str, Any]:
    """Render the exact completion row later consumed by MLX-LM."""

    from scripts.academic_finetune.training_support import format_academic_prompt

    return {
        "prompt": format_academic_prompt(example),
        "completion": str(example["target"]).strip(),
        "example_id": str(example["example_id"]),
    }


def _exact_token_filter(
    examples: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    max_sequence_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject whole oversized body candidates before balancing and splitting.

    Main-body supervision is a frozen quotation from a verified paragraph
    locator.  Shortening, truncating, or partitioning it here would break that
    one-to-one provenance, so an over-budget row is excluded in its entirety.
    """

    from scripts.academic_finetune.bound_training_data import (
        BoundTrainingDataError,
        measure_completion_sequence,
    )

    accepted: list[dict[str, Any]] = []
    measured_by_stratum: Counter[str] = Counter()
    measured_by_task: Counter[str] = Counter()
    rejected_by_stratum: Counter[str] = Counter()
    rejected_by_task: Counter[str] = Counter()
    largest_candidate = 0
    largest_accepted = 0
    for example in examples:
        source = example.get("source")
        if not isinstance(source, Mapping):
            raise BodyCorpusError("exact token candidate source must be an object")
        stratum = str(source.get("stratum", ""))
        task_type = str(example.get("task_type", ""))
        try:
            measurement = measure_completion_sequence(
                tokenizer, _training_measurement_row(example)
            )
        except BoundTrainingDataError as exc:
            raise BodyCorpusError(
                f"exact token measurement failed for {example.get('example_id')}: {exc}"
            ) from exc
        measured_by_stratum[stratum] += 1
        measured_by_task[task_type] += 1
        largest_candidate = max(largest_candidate, measurement.total_tokens)
        if measurement.total_tokens > max_sequence_length:
            rejected_by_stratum[stratum] += 1
            rejected_by_task[task_type] += 1
            continue
        largest_accepted = max(largest_accepted, measurement.total_tokens)
        bounded = dict(example)
        bounded["provenance"] = {
            **dict(example.get("provenance", {})),
            "exact_training_tokens": measurement.total_tokens,
            "exact_prompt_offset": measurement.prompt_offset,
            "exact_completion_tokens": measurement.completion_tokens,
        }
        accepted.append(bounded)
    return accepted, {
        "candidates_measured": len(examples),
        "candidates_measured_by_stratum": dict(sorted(measured_by_stratum.items())),
        "candidates_measured_by_task_type": dict(sorted(measured_by_task.items())),
        "candidates_rejected": sum(rejected_by_task.values()),
        "candidates_rejected_by_stratum": dict(sorted(rejected_by_stratum.items())),
        "candidates_rejected_by_task_type": dict(sorted(rejected_by_task.items())),
        "largest_candidate_tokens": largest_candidate,
        "largest_accepted_tokens": largest_accepted,
    }


def compile_body_corpus(
    body_sources: Sequence[BodySource],
    *,
    output_path: Path,
    body_cache_manifest: Mapping[str, Any] | None = None,
    balance_sources: bool = True,
    require_trainable_splits: bool = True,
    minimum_documents_per_split_stratum: int = 8,
    minimum_examples_per_split_stratum: int = 8,
    minimum_components_per_split_stratum: int = 8,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    tokenizer_model_path: Path | None = None,
    tokenizer: Any | None = None,
    tokenizer_identity: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Compile exactly token-gated body rows with structural provenance."""

    if not body_sources:
        raise BodyCorpusError("no body sources were supplied")
    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or not 128 <= max_sequence_length <= DEFAULT_MAX_SEQUENCE_LENGTH
    ):
        raise BodyCorpusError("max_sequence_length must be an integer in [128, 448]")
    if tokenizer is None:
        if tokenizer_model_path is None:
            raise BodyCorpusError(
                "exact body compilation requires tokenizer_model_path or an "
                "injected tokenizer"
            )
        if tokenizer_identity is not None:
            raise BodyCorpusError(
                "tokenizer_identity cannot override the pinned production tokenizer receipt"
            )
        from scripts.academic_finetune.bound_training_data import (
            BoundTrainingDataError,
            load_tokenizer_only,
        )

        try:
            tokenizer, exact_tokenizer_receipt = load_tokenizer_only(
                tokenizer_model_path
            )
        except BoundTrainingDataError as exc:
            raise BodyCorpusError(str(exc)) from exc
    else:
        if tokenizer_model_path is not None:
            raise BodyCorpusError(
                "tokenizer_model_path and an injected tokenizer are mutually exclusive"
            )
        exact_tokenizer_receipt = _tokenizer_receipt(tokenizer_identity)
    by_document: dict[str, BodySource] = {}
    for source in body_sources:
        identity = source.document.document_id
        if identity in by_document:
            raise BodyCorpusError(f"duplicate body source identity: {identity}")
        if source.document.abstract or not source.document.body:
            raise BodyCorpusError(f"non-body-only SourceDocument supplied: {identity}")
        by_document[identity] = source
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exact_gate: dict[str, Any] = {}

    def exact_candidate_filter(
        candidates: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        accepted, receipt = _exact_token_filter(
            candidates,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )
        exact_gate.update(receipt)
        return accepted

    with tempfile.TemporaryDirectory(prefix=".body-corpus-", dir=output_path.parent) as temporary:
        temporary_path = Path(temporary) / "base.jsonl"
        manifest = compile_corpus(
            [source.document for source in body_sources],
            output_path=temporary_path,
            balance_sources=balance_sources,
            require_trainable_splits=require_trainable_splits,
            minimum_documents_per_split_stratum=minimum_documents_per_split_stratum,
            minimum_examples_per_split_stratum=minimum_examples_per_split_stratum,
            minimum_components_per_split_stratum=minimum_components_per_split_stratum,
            candidate_filter=exact_candidate_filter,
        )
        rows = [
            json.loads(line)
            for line in temporary_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    section_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    for row in rows:
        document = row.get("document")
        provenance = row.get("provenance")
        if not isinstance(document, Mapping) or not isinstance(provenance, dict):
            raise BodyCorpusError("compiled corpus row is malformed")
        source = by_document.get(str(document.get("document_id", "")))
        if source is None:
            raise BodyCorpusError("compiled row references an unknown body source")
        locator = str(provenance.get("locator", ""))
        match = _LOCATOR.fullmatch(locator)
        if not match:
            raise BodyCorpusError(f"compiled row has an invalid locator: {locator}")
        paragraph_index = int(match.group(1))
        if not 1 <= paragraph_index <= len(source.extraction.paragraphs):
            raise BodyCorpusError(f"compiled paragraph locator is out of range: {locator}")
        paragraph = source.extraction.paragraphs[paragraph_index - 1]
        expected_target = _target_at_locator(paragraph, locator)
        if row.get("target") != expected_target:
            raise BodyCorpusError(
                f"compiled target cannot be verified against main body: {row.get('example_id')}"
            )
        provenance.update({
            "body_extraction_schema": BODY_EXTRACTION_SCHEMA,
            "body_source_region": "main_body",
            "body_only": True,
            "body_source_format": source.extraction.source_format,
            "body_artifact_sha256": source.artifact_sha256,
            "body_structure_sha256": source.structure_sha256,
            "body_document_sha256": source.extraction.body_sha256,
            "body_paragraph_sha256": paragraph.text_sha256,
            "body_document_paragraph_index": paragraph.document_paragraph_index,
            "body_section_paragraph_index": paragraph.paragraph_index_in_section,
            "body_section_id": paragraph.section_id,
            "body_section_title": paragraph.section_title,
            "body_section_path": list(paragraph.section_path),
            "body_section_order": paragraph.section_order,
            "excluded_source_regions": list(EXCLUDED_REGIONS),
        })
        quality_reason = _residual_artifact_reason(str(row.get("target", "")))
        if quality_reason:
            raise BodyCorpusError(
                f"compiled target failed the body quality gate: "
                f"{row.get('example_id')} ({quality_reason})"
            )
        section_counts[f"{source.document.document_id}\0{paragraph.section_id}"] += 1
        format_counts[source.extraction.source_format] += 1

    jsonl = "".join(canonical_json(row) + "\n" for row in rows)
    _atomic_bytes(output_path, jsonl.encode("utf-8"))
    quality_audit = _write_quality_sample(rows, output_path=output_path)
    manifest["corpus_sha256"] = _sha256_bytes(jsonl.encode("utf-8"))
    manifest["output_filename"] = output_path.name
    manifest_counts = manifest.setdefault("counts", {})
    manifest_counts.update({
        "body_only_examples": len(rows),
        "abstract_examples": 0,
        "body_only_example_ratio": 1.0,
        "abstract_example_ratio": 0.0,
        "input_body_source_documents": len(body_sources),
        "input_body_paragraphs": sum(
            len(source.extraction.paragraphs) for source in body_sources
        ),
        "body_examples_by_source_format": dict(sorted(format_counts.items())),
        "exact_token_candidates_measured": int(
            exact_gate.get("candidates_measured", 0)
        ),
        "exact_token_candidates_rejected": int(
            exact_gate.get("candidates_rejected", 0)
        ),
    })
    manifest.setdefault("gates", {})["exact_training_token_gate"] = {
        "method": "mlx_lm.CompletionsDataset.apply_chat_template parity",
        "stage": "after_candidate_dedup_before_author_split_and_source_example_balance",
        "max_sequence_length": max_sequence_length,
        "overflow_policy": "reject_candidate_never_truncate_or_partition",
        "derived_rows": 0,
        "tokenizer": exact_tokenizer_receipt,
        **exact_gate,
    }
    manifest.setdefault("rejections", {})["examples"] = {
        f"over_exact_token_budget:{task_type}": count
        for task_type, count in exact_gate.get(
            "candidates_rejected_by_task_type", {}
        ).items()
    }
    cache_manifest_sha256 = ""
    temporal_summary: dict[str, Any] = {}
    if body_cache_manifest is not None:
        cache_manifest_sha256 = sha256_text(canonical_json(body_cache_manifest))
        temporal_value = body_cache_manifest.get("jats_temporal_attestation", {})
        if isinstance(temporal_value, Mapping):
            temporal_summary = {
                key: value
                for key, value in temporal_value.items()
                if key != "receipts"
            }
            temporal_summary["receipts_canonical_sha256"] = sha256_text(
                canonical_json(temporal_value.get("receipts", []))
            )
    manifest["body_only_attestation"] = {
        "schema_version": BODY_ATTESTATION_SCHEMA,
        "body_only_example_ratio": 1.0,
        "body_only_examples": len(rows),
        "abstract_examples": 0,
        "source_document_abstract_fields_empty": all(
            not source.document.abstract for source in body_sources
        ),
        "every_target_verified_at_body_locator": True,
        "every_example_has_section_paragraph_provenance": True,
        "exact_token_gate_before_split_and_source_example_balance": True,
        "targets_truncated_or_partitioned": 0,
        "excluded_regions": list(EXCLUDED_REGIONS),
        "official_cached_artifacts_only": True,
        "network_requests": 0,
        "body_cache_manifest_canonical_sha256": cache_manifest_sha256,
        "document_section_pairs_with_examples": len(section_counts),
        "deterministic_quality_audit": quality_audit,
        "jats_temporal_attestation": temporal_summary,
    }
    manifest.setdefault("source_hygiene", {})["body_region_policy"] = (
        "only included main-body TeX/JATS paragraphs; abstracts, references, "
        "acknowledgments, appendices, captions, tables, and display math excluded"
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise BodyCorpusError("base corpus compiler emitted an incompatible manifest")
    _atomic_json(output_path.with_name(f"{output_path.name}.manifest.json"), manifest)
    return manifest


def build_offline_body_corpus(
    *,
    metadata_cache: Path,
    structure_cache: Path,
    body_cache: Path,
    output_path: Path,
    balance_sources: bool = True,
    require_trainable_splits: bool = True,
    minimum_documents_per_split_stratum: int = 8,
    minimum_examples_per_split_stratum: int = 8,
    minimum_components_per_split_stratum: int = 8,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    tokenizer_model_path: Path | None = None,
    tokenizer: Any | None = None,
    tokenizer_identity: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    documents = load_metadata_cache(metadata_cache)
    sources, cache_manifest = build_body_source_cache(
        documents,
        structure_cache=structure_cache,
        body_cache=body_cache,
    )
    return compile_body_corpus(
        sources,
        output_path=output_path,
        body_cache_manifest=cache_manifest,
        balance_sources=balance_sources,
        require_trainable_splits=require_trainable_splits,
        minimum_documents_per_split_stratum=minimum_documents_per_split_stratum,
        minimum_examples_per_split_stratum=minimum_examples_per_split_stratum,
        minimum_components_per_split_stratum=minimum_components_per_split_stratum,
        max_sequence_length=max_sequence_length,
        tokenizer_model_path=tokenizer_model_path,
        tokenizer=tokenizer,
        tokenizer_identity=tokenizer_identity,
    )


__all__ = [
    "BODY_ATTESTATION_SCHEMA",
    "BODY_CACHE_MANIFEST_SCHEMA",
    "BODY_CUTOFF",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "BODY_EXTRACTION_SCHEMA",
    "BODY_TEMPORAL_ATTESTATION_SCHEMA",
    "BodyCorpusError",
    "BodyExtraction",
    "BodyParagraph",
    "BodySource",
    "build_body_source_cache",
    "build_offline_body_corpus",
    "compile_body_corpus",
    "extract_body",
    "extract_jats_body",
    "extract_tex_body",
    "load_body_source_cache",
]
