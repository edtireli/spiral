"""Bounded, deterministic extraction of academic-paper structure.

This module intentionally extracts *architecture*, not full-text training prose.
It preserves heading hierarchy and coarse layout statistics while avoiding TeX
execution and external XML resolution.  All archive paths and recursive includes
are resolved inside an in-memory virtual source tree under explicit limits.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import posixpath
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


STRUCTURE_SCHEMA = "spiral.paper-structure.v1"
PARSER_VERSION = "1"


@dataclass(frozen=True)
class StructureLimits:
    """Resource and recursion budgets for untrusted academic artifacts."""

    max_payload_bytes: int = 64 * 1024 * 1024
    max_members: int = 512
    max_unpacked_bytes: int = 32 * 1024 * 1024
    max_member_bytes: int = 8 * 1024 * 1024
    max_include_depth: int = 16
    max_expanded_bytes: int = 32 * 1024 * 1024
    max_xml_depth: int = 64
    max_nodes: int = 100_000
    max_sections: int = 4_096

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


DEFAULT_LIMITS = StructureLimits()


@dataclass(frozen=True)
class AssetPlacement:
    """A figure or table located relative to direct paragraphs in a section."""

    kind: str
    order: int
    after_paragraph: int
    identifier: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"figure", "table"}:
            raise ValueError("asset kind must be figure or table")
        if self.order < 1 or self.after_paragraph < 0:
            raise ValueError("asset positions must be non-negative and one-indexed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "order": self.order,
            "after_paragraph": self.after_paragraph,
            "identifier": self.identifier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AssetPlacement":
        return cls(
            kind=str(value["kind"]),
            order=int(value["order"]),
            after_paragraph=int(value["after_paragraph"]),
            identifier=str(value.get("identifier", "")),
        )


@dataclass(frozen=True)
class SectionNode:
    """One observed heading and the content owned directly by that heading.

    ``direct_*`` fields exclude child sections.  The aggregate properties include
    descendants, which prevents hierarchical corpora from double-counting prose.
    """

    title: str
    level: int
    order: int
    path: tuple[int, ...]
    included_in_main: bool
    exclusion_reason: str = ""
    direct_word_count: int = 0
    direct_paragraph_count: int = 0
    placements: tuple[AssetPlacement, ...] = ()
    children: tuple["SectionNode", ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("section title cannot be empty")
        if self.level < 1 or self.order < 1 or not self.path:
            raise ValueError("section level, order, and path must be positive")
        if any(part < 1 for part in self.path):
            raise ValueError("section path components must be positive")
        if self.direct_word_count < 0 or self.direct_paragraph_count < 0:
            raise ValueError("section counts cannot be negative")
        if self.included_in_main and self.exclusion_reason:
            raise ValueError("a main-text section cannot have an exclusion reason")
        if not self.included_in_main and not self.exclusion_reason:
            raise ValueError("an excluded section must state an exclusion reason")

    @property
    def word_count(self) -> int:
        return self.direct_word_count + sum(child.word_count for child in self.children)

    @property
    def paragraph_count(self) -> int:
        return self.direct_paragraph_count + sum(child.paragraph_count for child in self.children)

    @property
    def direct_figure_count(self) -> int:
        return sum(placement.kind == "figure" for placement in self.placements)

    @property
    def direct_table_count(self) -> int:
        return sum(placement.kind == "table" for placement in self.placements)

    @property
    def figure_count(self) -> int:
        return self.direct_figure_count + sum(child.figure_count for child in self.children)

    @property
    def table_count(self) -> int:
        return self.direct_table_count + sum(child.table_count for child in self.children)

    @property
    def main_word_count(self) -> int:
        if not self.included_in_main:
            return 0
        return self.direct_word_count + sum(child.main_word_count for child in self.children)

    @property
    def main_paragraph_count(self) -> int:
        if not self.included_in_main:
            return 0
        return self.direct_paragraph_count + sum(child.main_paragraph_count for child in self.children)

    @property
    def main_figure_count(self) -> int:
        if not self.included_in_main:
            return 0
        return self.direct_figure_count + sum(child.main_figure_count for child in self.children)

    @property
    def main_table_count(self) -> int:
        if not self.included_in_main:
            return 0
        return self.direct_table_count + sum(child.main_table_count for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "order": self.order,
            "path": list(self.path),
            "included_in_main": self.included_in_main,
            "exclusion_reason": self.exclusion_reason,
            "direct_word_count": self.direct_word_count,
            "direct_paragraph_count": self.direct_paragraph_count,
            "word_count": self.word_count,
            "paragraph_count": self.paragraph_count,
            "direct_figure_count": self.direct_figure_count,
            "direct_table_count": self.direct_table_count,
            "figure_count": self.figure_count,
            "table_count": self.table_count,
            "placements": [placement.to_dict() for placement in self.placements],
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SectionNode":
        node = cls(
            title=str(value["title"]),
            level=int(value["level"]),
            order=int(value["order"]),
            path=tuple(int(part) for part in value["path"]),
            included_in_main=_strict_bool(value["included_in_main"], "included_in_main"),
            exclusion_reason=str(value.get("exclusion_reason", "")),
            direct_word_count=int(value.get("direct_word_count", 0)),
            direct_paragraph_count=int(value.get("direct_paragraph_count", 0)),
            placements=tuple(
                AssetPlacement.from_dict(item)
                for item in _mapping_sequence(value.get("placements", ()), "placements")
            ),
            children=tuple(
                cls.from_dict(item)
                for item in _mapping_sequence(value.get("children", ()), "children")
            ),
        )
        for key, observed in (
            ("word_count", node.word_count),
            ("paragraph_count", node.paragraph_count),
            ("direct_figure_count", node.direct_figure_count),
            ("direct_table_count", node.direct_table_count),
            ("figure_count", node.figure_count),
            ("table_count", node.table_count),
        ):
            if key in value and int(value[key]) != observed:
                raise ValueError(f"section aggregate {key} does not match its content")
        return node


@dataclass(frozen=True)
class StructureProvenance:
    parser: str
    parser_version: str
    source_sha256: str
    source_bytes: int
    root_member: str = ""
    included_members: tuple[str, ...] = ()
    expanded_source_sha256: str = ""
    warnings: tuple[str, ...] = ()
    identifiers: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "parser_version": self.parser_version,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "root_member": self.root_member,
            "included_members": list(self.included_members),
            "expanded_source_sha256": self.expanded_source_sha256,
            "warnings": list(self.warnings),
            "identifiers": [
                {"type": identifier_type, "value": identifier}
                for identifier_type, identifier in self.identifiers
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructureProvenance":
        identifiers: list[tuple[str, str]] = []
        for item in _mapping_sequence(value.get("identifiers", ()), "identifiers"):
            identifiers.append((str(item["type"]), str(item["value"])))
        return cls(
            parser=str(value["parser"]),
            parser_version=str(value["parser_version"]),
            source_sha256=str(value["source_sha256"]),
            source_bytes=int(value["source_bytes"]),
            root_member=str(value.get("root_member", "")),
            included_members=tuple(str(item) for item in value.get("included_members", ())),
            expanded_source_sha256=str(value.get("expanded_source_sha256", "")),
            warnings=tuple(str(item) for item in value.get("warnings", ())),
            identifiers=tuple(identifiers),
        )


@dataclass(frozen=True)
class PaperStructure:
    source_format: str
    title: str
    sections: tuple[SectionNode, ...]
    appendices: tuple[SectionNode, ...]
    abstract_word_count: int
    abstract_paragraph_count: int
    unsectioned_word_count: int
    unsectioned_paragraph_count: int
    unsectioned_placements: tuple[AssetPlacement, ...]
    provenance: StructureProvenance

    def __post_init__(self) -> None:
        if self.source_format not in {"arxiv_tex", "pmc_jats"}:
            raise ValueError("unsupported paper structure source format")
        for value in (
            self.abstract_word_count,
            self.abstract_paragraph_count,
            self.unsectioned_word_count,
            self.unsectioned_paragraph_count,
        ):
            if value < 0:
                raise ValueError("paper structure counts cannot be negative")
        if any(node.included_in_main for node in self.appendices):
            raise ValueError("appendix nodes cannot be included in main text")

    @property
    def main_word_count(self) -> int:
        return self.unsectioned_word_count + sum(node.main_word_count for node in self.sections)

    @property
    def main_paragraph_count(self) -> int:
        return self.unsectioned_paragraph_count + sum(
            node.main_paragraph_count for node in self.sections
        )

    @property
    def main_figure_count(self) -> int:
        return sum(placement.kind == "figure" for placement in self.unsectioned_placements) + sum(
            node.main_figure_count for node in self.sections
        )

    @property
    def main_table_count(self) -> int:
        return sum(placement.kind == "table" for placement in self.unsectioned_placements) + sum(
            node.main_table_count for node in self.sections
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STRUCTURE_SCHEMA,
            "source_format": self.source_format,
            "title": self.title,
            "abstract_word_count": self.abstract_word_count,
            "abstract_paragraph_count": self.abstract_paragraph_count,
            "unsectioned_word_count": self.unsectioned_word_count,
            "unsectioned_paragraph_count": self.unsectioned_paragraph_count,
            "unsectioned_placements": [
                placement.to_dict() for placement in self.unsectioned_placements
            ],
            "counts": {
                "main_words": self.main_word_count,
                "main_paragraphs": self.main_paragraph_count,
                "main_figures": self.main_figure_count,
                "main_tables": self.main_table_count,
                "abstract_words": self.abstract_word_count,
                "abstract_paragraphs": self.abstract_paragraph_count,
                "unsectioned_words": self.unsectioned_word_count,
                "unsectioned_paragraphs": self.unsectioned_paragraph_count,
            },
            "sections": [node.to_dict() for node in self.sections],
            "appendices": [node.to_dict() for node in self.appendices],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PaperStructure":
        if value.get("schema_version", STRUCTURE_SCHEMA) != STRUCTURE_SCHEMA:
            raise ValueError("unsupported paper structure schema")
        paper = cls(
            source_format=str(value["source_format"]),
            title=str(value.get("title", "")),
            sections=tuple(
                SectionNode.from_dict(item)
                for item in _mapping_sequence(value.get("sections", ()), "sections")
            ),
            appendices=tuple(
                SectionNode.from_dict(item)
                for item in _mapping_sequence(value.get("appendices", ()), "appendices")
            ),
            abstract_word_count=int(value.get("abstract_word_count", 0)),
            abstract_paragraph_count=int(value.get("abstract_paragraph_count", 0)),
            unsectioned_word_count=int(value.get("unsectioned_word_count", 0)),
            unsectioned_paragraph_count=int(value.get("unsectioned_paragraph_count", 0)),
            unsectioned_placements=tuple(
                AssetPlacement.from_dict(item)
                for item in _mapping_sequence(
                    value.get("unsectioned_placements", ()), "unsectioned_placements"
                )
            ),
            provenance=StructureProvenance.from_dict(_mapping(value["provenance"], "provenance")),
        )
        counts = value.get("counts")
        if counts is not None:
            count_map = _mapping(counts, "counts")
            for key, observed in (
                ("main_words", paper.main_word_count),
                ("main_paragraphs", paper.main_paragraph_count),
                ("main_figures", paper.main_figure_count),
                ("main_tables", paper.main_table_count),
            ):
                if key in count_map and int(count_map[key]) != observed:
                    raise ValueError(f"paper aggregate {key} does not match its sections")
        return paper


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _mapping_sequence(value: Any, name: str) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    result: list[Mapping[str, Any]] = []
    for item in value:
        result.append(_mapping(item, name))
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
_COMMENT = re.compile(r"(?m)(?<!\\)%.*$")
_EQUATION_ENVIRONMENT = re.compile(
    r"\\begin\{(?:equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|"
    r"eqnarray\*?|displaymath|math)\}.*?"
    r"\\end\{(?:equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|"
    r"eqnarray\*?|displaymath|math)\}",
    re.DOTALL | re.IGNORECASE,
)
_TEXT_COMMAND = re.compile(
    r"\\(?:textbf|textit|emph|texttt|textrm|textsf|textsc|mathrm|mathbf|mathit|"
    r"mbox|underline|foreignlanguage)\*?(?:\[[^\]]*\])?\{([^{}]*)\}"
)
_DROP_COMMAND = re.compile(
    r"\\(?:label|ref|pageref|eqref|autoref|cref|Cref|cite\w*|includegraphics|url|"
    r"href|footnotemark|index|glossary)\*?(?:\[[^\]]*\]\s*)*\{[^{}]*\}"
    r"(?:\{[^{}]*\})?",
    re.IGNORECASE,
)


def _strip_comments(value: str) -> str:
    return _COMMENT.sub("", value)


def _plain_tex(value: str) -> str:
    """Conservatively turn a TeX prose fragment into countable plain text."""

    value = _strip_comments(value).replace("\r\n", "\n").replace("\r", "\n")
    value = _EQUATION_ENVIRONMENT.sub(" ", value)
    value = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)", " ", value, flags=re.DOTALL)
    value = re.sub(r"(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$", " ", value, flags=re.DOTALL)
    value = _DROP_COMMAND.sub(" ", value)
    value = re.sub(
        r"\\(?:newcommand|renewcommand|providecommand|def)\b[^\n]*",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    for _ in range(6):
        updated = _TEXT_COMMAND.sub(r"\1", value)
        if updated == value:
            break
        value = updated
    value = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\\(?:item|noindent|smallskip|medskip|bigskip|newline|linebreak)\b", "\n", value)
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", value)
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\{": "{",
        r"\}": "}",
        "~": " ",
        "``": '"',
        "''": '"',
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _metrics(value: str) -> tuple[int, int]:
    plain = _plain_tex(value)
    if not plain:
        return 0, 0
    paragraphs = [part for part in re.split(r"\n\s*\n", plain) if _WORD.search(part)]
    return sum(len(_WORD.findall(part)) for part in paragraphs), len(paragraphs)


def _title_tex(value: str) -> str:
    return re.sub(r"\s+", " ", _plain_tex(value)).strip(" .")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class _MutableSection:
    title: str
    level: int
    order: int
    path: tuple[int, ...]
    included_in_main: bool
    exclusion_reason: str
    raw_parts: list[str] = field(default_factory=list)
    placements: list[AssetPlacement] = field(default_factory=list)
    children: list["_MutableSection"] = field(default_factory=list)

    def paragraph_count_so_far(self) -> int:
        return _metrics("".join(self.raw_parts))[1]

    def freeze(self) -> SectionNode:
        words, paragraphs = _metrics("".join(self.raw_parts))
        return SectionNode(
            title=self.title,
            level=self.level,
            order=self.order,
            path=self.path,
            included_in_main=self.included_in_main,
            exclusion_reason=self.exclusion_reason,
            direct_word_count=words,
            direct_paragraph_count=paragraphs,
            placements=tuple(self.placements),
            children=tuple(child.freeze() for child in self.children),
        )


_REFERENCE_TITLE = re.compile(
    r"^(?:references?|bibliography|works cited|literature cited)$", re.IGNORECASE
)
_ACK_TITLE = re.compile(
    r"^(?:acknowledg(?:e)?ments?(?:\s+and\s+funding)?|funding\s+and\s+acknowledg(?:e)?ments?)$",
    re.IGNORECASE,
)
_APPENDIX_TITLE = re.compile(
    r"^(?:appendix|appendices|appendix\s+[a-z0-9]+\b.*|supplementary\s+material\b.*|"
    r"supporting\s+information\b.*)$",
    re.IGNORECASE,
)


def _classify_title(title: str, *, appendix_mode: bool = False) -> tuple[bool, str]:
    normalised = re.sub(r"\s+", " ", title).strip(" .:")
    if _REFERENCE_TITLE.fullmatch(normalised):
        return False, "references"
    if _ACK_TITLE.fullmatch(normalised):
        return False, "acknowledgments"
    if appendix_mode or _APPENDIX_TITLE.fullmatch(normalised):
        return False, "appendix"
    return True, ""


class _StructureBuilder:
    def __init__(self, *, limits: StructureLimits) -> None:
        self.limits = limits
        self.sections: list[_MutableSection] = []
        self.appendices: list[_MutableSection] = []
        self._section_stack: list[_MutableSection] = []
        self._appendix_stack: list[_MutableSection] = []
        self.current: _MutableSection | None = None
        self.appendix_mode = False
        self.section_order = 0
        self.asset_order = 0
        self.unsectioned_raw: list[str] = []
        self.unsectioned_placements: list[AssetPlacement] = []
        self.abstract_raw: list[str] = []

    def start_appendices(self) -> None:
        self.appendix_mode = True
        self.current = None

    def add_heading(
        self,
        title: str,
        level: int,
        *,
        forced_exclusion: str = "",
    ) -> _MutableSection:
        title = title or "Untitled section"
        self.section_order += 1
        if self.section_order > self.limits.max_sections:
            raise ValueError("paper exceeds the section-count limit")
        included, reason = _classify_title(title, appendix_mode=self.appendix_mode)
        if forced_exclusion:
            included, reason = False, forced_exclusion
        if reason == "appendix":
            self.appendix_mode = True
        roots = self.appendices if reason == "appendix" else self.sections
        stack = self._appendix_stack if reason == "appendix" else self._section_stack
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            parent = stack[-1]
            if not parent.included_in_main:
                included = False
                reason = parent.exclusion_reason
            path = parent.path + (len(parent.children) + 1,)
        else:
            parent = None
            path = (len(roots) + 1,)
        node = _MutableSection(title, level, self.section_order, path, included, reason)
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
        stack.append(node)
        self.current = node
        return node

    def add_content(self, value: str) -> None:
        if not value:
            return
        if self.current is not None:
            self.current.raw_parts.append(value)
            return
        if self.appendix_mode and _metrics(value)[0]:
            self.add_heading("Appendix", 1, forced_exclusion="appendix").raw_parts.append(value)
            return
        self.unsectioned_raw.append(value)

    def add_abstract(self, value: str) -> None:
        if value:
            self.abstract_raw.append(value)

    def add_asset(self, kind: str, identifier: str = "") -> None:
        self.asset_order += 1
        if self.current is None:
            after = _metrics("".join(self.unsectioned_raw))[1]
        else:
            after = self.current.paragraph_count_so_far()
        placement = AssetPlacement(kind, self.asset_order, after, identifier)
        if self.current is None:
            self.unsectioned_placements.append(placement)
        else:
            self.current.placements.append(placement)

    def freeze(
        self,
        *,
        source_format: str,
        title: str,
        provenance: StructureProvenance,
    ) -> PaperStructure:
        abstract_words, abstract_paragraphs = _metrics("\n\n".join(self.abstract_raw))
        unsectioned_words, unsectioned_paragraphs = _metrics("".join(self.unsectioned_raw))
        return PaperStructure(
            source_format=source_format,
            title=title,
            sections=tuple(node.freeze() for node in self.sections),
            appendices=tuple(node.freeze() for node in self.appendices),
            abstract_word_count=abstract_words,
            abstract_paragraph_count=abstract_paragraphs,
            unsectioned_word_count=unsectioned_words,
            unsectioned_paragraph_count=unsectioned_paragraphs,
            unsectioned_placements=tuple(self.unsectioned_placements),
            provenance=provenance,
        )


_TEXT_MEMBER_SUFFIXES = {"", ".tex", ".ltx", ".inc", ".sty", ".cls"}
_INCLUDE = re.compile(
    r"\\(?P<command>input|include)\b\s*(?:\{(?P<braced>[^{}]+)\}|(?P<bare>[^\s%{}]+))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _TexSourceTree:
    members: Mapping[str, str]
    root: str
    skipped_members: int


def _normalise_member_path(value: str) -> str:
    if not value or "\\" in value or PurePosixPath(value).is_absolute():
        raise ValueError("unsafe path in arXiv source archive")
    normalised = posixpath.normpath(value)
    if normalised in {"", "."} or normalised == ".." or normalised.startswith("../"):
        raise ValueError("unsafe path in arXiv source archive")
    return normalised


def _bounded_gzip(payload: bytes, maximum: int) -> bytes:
    result = bytearray()
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
            if len(result) > maximum:
                raise ValueError("compressed arXiv TeX exceeds the unpacked size limit")
    return bytes(result)


def _read_tex_source_tree(payload: bytes, limits: StructureLimits) -> _TexSourceTree:
    if len(payload) > limits.max_payload_bytes:
        raise ValueError("arXiv source payload exceeds the compressed size limit")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.TarError:
        raw = _bounded_gzip(payload, limits.max_unpacked_bytes) if payload.startswith(b"\x1f\x8b") else payload
        if len(raw) > limits.max_unpacked_bytes:
            raise ValueError("arXiv TeX exceeds the unpacked size limit")
        return _TexSourceTree({"<single-source>": raw.decode("utf-8", errors="replace")}, "<single-source>", 0)

    decoded: dict[str, str] = {}
    skipped = 0
    total = 0
    with archive:
        for member_index, member in enumerate(archive, start=1):
            if member_index > limits.max_members:
                raise ValueError("arXiv source archive has too many members")
            if member.isdir() and member.name.rstrip("/") in {"", "."}:
                skipped += 1
                continue
            name = _normalise_member_path(member.name)
            if not member.isfile():
                skipped += 1
                continue
            total += member.size
            if total > limits.max_unpacked_bytes:
                raise ValueError("arXiv source archive exceeds the unpacked size limit")
            if member.size > limits.max_member_bytes:
                raise ValueError("arXiv source archive member exceeds the per-file size limit")
            if PurePosixPath(name).suffix.lower() not in _TEXT_MEMBER_SUFFIXES:
                skipped += 1
                continue
            if name in decoded:
                raise ValueError("duplicate path in arXiv source archive")
            handle = archive.extractfile(member)
            if handle is None:
                skipped += 1
                continue
            data = handle.read(limits.max_member_bytes + 1)
            if len(data) != member.size or len(data) > limits.max_member_bytes:
                raise ValueError("arXiv source archive member size is inconsistent")
            decoded[name] = data.decode("utf-8", errors="replace")
    if not decoded:
        raise ValueError("arXiv source archive contains no TeX members")
    candidates = [
        (name, value)
        for name, value in decoded.items()
        if PurePosixPath(name).suffix.lower() in {"", ".tex", ".ltx"}
    ]
    if not candidates:
        raise ValueError("arXiv source archive contains no candidate main TeX document")
    candidates.sort(
        key=lambda item: (
            "\\begin{document}" not in item[1],
            "\\documentclass" not in item[1],
            -len(item[1]),
            item[0],
        )
    )
    return _TexSourceTree(decoded, candidates[0][0], skipped)


def _resolve_include(current: str, requested: str, members: Mapping[str, str]) -> str | None:
    requested = requested.strip().strip('"\'')
    if (
        not requested
        or "\\" in requested
        or ":" in requested
        or requested.startswith("/")
        or any(character in requested for character in "#$%{}")
    ):
        raise ValueError(f"unsafe TeX include path: {requested or '<empty>'}")
    parent = "" if current == "<single-source>" else posixpath.dirname(current)
    candidate = posixpath.normpath(posixpath.join(parent, requested))
    if candidate == ".." or candidate.startswith("../") or candidate.startswith("/"):
        raise ValueError(f"TeX include escapes the source archive: {requested}")
    alternatives = [candidate]
    if not PurePosixPath(candidate).suffix:
        alternatives.extend((f"{candidate}.tex", f"{candidate}.ltx"))
    return next((path for path in alternatives if path in members), None)


def _expand_tex_tree(
    tree: _TexSourceTree,
    limits: StructureLimits,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    included: list[str] = []
    included_set: set[str] = set()
    warnings: list[str] = []

    def expand(path: str, stack: tuple[str, ...]) -> str:
        if len(stack) > limits.max_include_depth:
            raise ValueError("TeX include depth exceeds the configured limit")
        if path in stack:
            cycle = " -> ".join(stack + (path,))
            raise ValueError(f"TeX include cycle detected: {cycle}")
        if path not in included_set:
            included.append(path)
            included_set.add(path)
        source = _strip_comments(tree.members[path])
        pieces: list[str] = []
        size = 0
        cursor = 0
        for match in _INCLUDE.finditer(source):
            prefix = source[cursor : match.start()]
            pieces.append(prefix)
            size += len(prefix.encode("utf-8"))
            requested = match.group("braced") or match.group("bare") or ""
            resolved = _resolve_include(path, requested, tree.members)
            if resolved is None:
                warnings.append(f"unresolved_include:{path}:{requested.strip()}")
                replacement = "\n"
            else:
                replacement = expand(resolved, stack + (path,))
            size += len(replacement.encode("utf-8"))
            if size > limits.max_expanded_bytes:
                raise ValueError("expanded TeX exceeds the configured size limit")
            pieces.append(replacement)
            cursor = match.end()
        suffix = source[cursor:]
        pieces.append(suffix)
        size += len(suffix.encode("utf-8"))
        if size > limits.max_expanded_bytes:
            raise ValueError("expanded TeX exceeds the configured size limit")
        return "".join(pieces)

    return expand(tree.root, ()), tuple(included), tuple(warnings)


def _balanced_argument(value: str, start: int, opener: str, closer: str) -> tuple[str, int] | None:
    cursor = start
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if cursor >= len(value) or value[cursor] != opener:
        return None
    depth = 0
    content_start = cursor + 1
    for index in range(cursor, len(value)):
        character = value[index]
        if index > cursor and value[index - 1] == "\\":
            continue
        if character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return value[content_start:index], index + 1
    return None


def _heading_argument(value: str, start: int) -> tuple[str, int] | None:
    cursor = start
    optional = _balanced_argument(value, cursor, "[", "]")
    if optional is not None:
        cursor = optional[1]
    return _balanced_argument(value, cursor, "{", "}")


def _environment_argument(value: str, start: int) -> tuple[str, int] | None:
    return _balanced_argument(value, start, "{", "}")


def _environment_end(value: str, environment: str, start: int) -> tuple[int, int] | None:
    token = re.compile(
        rf"\\(?P<kind>begin|end)\s*\{{\s*{re.escape(environment)}\s*\}}",
        re.IGNORECASE,
    )
    depth = 1
    for match in token.finditer(value, start):
        depth += 1 if match.group("kind").casefold() == "begin" else -1
        if depth == 0:
            return match.start(), match.end()
    return None


_TEX_SIGNAL = re.compile(
    r"\\(?P<command>chapter|section|subsection|subsubsection|paragraph|appendix|begin|"
    r"bibliography|printbibliography)\b\*?",
    re.IGNORECASE,
)
_MATH_ENVIRONMENTS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "math",
}
_FIGURE_ENVIRONMENTS = {"figure", "figure*", "marginfigure", "sidewaysfigure", "wrapfigure"}
_TABLE_ENVIRONMENTS = {"table", "table*", "longtable", "sidewaystable", "wraptable"}
_ACK_ENVIRONMENTS = {"ack", "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}


def _document_body(value: str) -> str:
    begin = re.search(r"\\begin\s*\{\s*document\s*\}", value, flags=re.IGNORECASE)
    if begin:
        value = value[begin.end() :]
    end = re.search(r"\\end\s*\{\s*document\s*\}", value, flags=re.IGNORECASE)
    if end:
        value = value[: end.start()]
    return value


def _tex_document_class(value: str) -> str:
    match = re.search(
        r"\\documentclass(?:\[[^\]]*\])?\s*\{\s*([^{}]+)\s*\}", value, flags=re.IGNORECASE
    )
    return match.group(1).strip().casefold() if match else ""


def _tex_title(value: str) -> str:
    match = re.search(r"\\title\*?\s*\{", value, flags=re.IGNORECASE)
    if not match:
        return ""
    argument = _balanced_argument(value, match.end() - 1, "{", "}")
    return _title_tex(argument[0]) if argument else ""


def _asset_identifier(value: str) -> str:
    match = re.search(r"\\label\s*\{\s*([^{}]+?)\s*\}", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _parse_tex_document(value: str, builder: _StructureBuilder, warnings: list[str]) -> str:
    title = _tex_title(value)
    document_class = _tex_document_class(value)
    has_chapters = bool(re.search(r"\\chapter\*?\s*\{", value, flags=re.IGNORECASE)) or any(
        token in document_class for token in ("book", "report", "memoir")
    )
    levels = {
        "chapter": 1,
        "section": 2 if has_chapters else 1,
        "subsection": 3 if has_chapters else 2,
        "subsubsection": 4 if has_chapters else 3,
        "paragraph": 5 if has_chapters else 4,
    }
    body = _document_body(value)
    cursor = 0
    while True:
        match = _TEX_SIGNAL.search(body, cursor)
        if match is None:
            builder.add_content(body[cursor:])
            break
        builder.add_content(body[cursor : match.start()])
        command = match.group("command").casefold()
        if command in levels:
            argument = _heading_argument(body, match.end())
            if argument is None:
                warnings.append(f"malformed_heading:{command}:{match.start()}")
                builder.add_content(body[match.start() : match.end()])
                cursor = match.end()
                continue
            heading = _title_tex(argument[0]) or "Untitled section"
            builder.add_heading(heading, levels[command])
            cursor = argument[1]
            continue
        if command == "appendix":
            builder.start_appendices()
            cursor = match.end()
            continue
        if command in {"bibliography", "printbibliography"}:
            if builder.current is None or builder.current.exclusion_reason != "references":
                builder.add_heading("References", 1, forced_exclusion="references")
            argument = _heading_argument(body, match.end()) if command == "bibliography" else None
            cursor = argument[1] if argument else match.end()
            continue

        environment_argument = _environment_argument(body, match.end())
        if environment_argument is None:
            builder.add_content(body[match.start() : match.end()])
            cursor = match.end()
            continue
        environment = environment_argument[0].strip().casefold()
        content_start = environment_argument[1]
        environment_end = _environment_end(body, environment, content_start)
        if environment in {"appendices", "appendix"}:
            builder.start_appendices()
            cursor = content_start
            continue
        if environment not in (
            _MATH_ENVIRONMENTS
            | _FIGURE_ENVIRONMENTS
            | _TABLE_ENVIRONMENTS
            | _ACK_ENVIRONMENTS
            | {"abstract", "thebibliography"}
        ):
            builder.add_content(body[match.start() : content_start])
            cursor = content_start
            continue
        if environment_end is None:
            warnings.append(f"unterminated_environment:{environment}:{match.start()}")
            content_end, after_end = len(body), len(body)
        else:
            content_end, after_end = environment_end
        environment_content = body[content_start:content_end]
        if environment == "abstract":
            builder.add_abstract(environment_content)
        elif environment in _FIGURE_ENVIRONMENTS:
            builder.add_asset("figure", _asset_identifier(environment_content))
        elif environment in _TABLE_ENVIRONMENTS:
            builder.add_asset("table", _asset_identifier(environment_content))
        elif environment in _ACK_ENVIRONMENTS:
            node = builder.add_heading("Acknowledgments", 1, forced_exclusion="acknowledgments")
            node.raw_parts.append(environment_content)
        elif environment == "thebibliography":
            builder.add_heading("References", 1, forced_exclusion="references")
        # Display mathematics is intentionally excluded from prose word counts.
        cursor = after_end
    return title


def parse_tex_structure_archive(
    payload: bytes,
    *,
    limits: StructureLimits = DEFAULT_LIMITS,
) -> PaperStructure:
    """Parse an arXiv TeX payload without extracting or executing its files."""

    tree = _read_tex_source_tree(payload, limits)
    expanded, included, include_warnings = _expand_tex_tree(tree, limits)
    warnings = list(include_warnings)
    builder = _StructureBuilder(limits=limits)
    title = _parse_tex_document(expanded, builder, warnings)
    if not builder.sections and not builder.appendices:
        warnings.append("no_section_headings")
    provenance = StructureProvenance(
        parser="spiral_tex_structure",
        parser_version=PARSER_VERSION,
        source_sha256=_sha256(payload),
        source_bytes=len(payload),
        root_member=tree.root,
        included_members=included,
        expanded_source_sha256=_sha256(expanded.encode("utf-8")),
        warnings=tuple(warnings + ([f"skipped_archive_members:{tree.skipped_members}"] if tree.skipped_members else [])),
    )
    return builder.freeze(
        source_format="arxiv_tex",
        title=title,
        provenance=provenance,
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


_JATS_SKIP_TEXT = {
    "xref",
    "disp-formula",
    "inline-formula",
    "tex-math",
    "math",
    "graphic",
    "inline-graphic",
    "media",
    "supplementary-material",
}


def _jats_text(node: ET.Element) -> str:
    pieces: list[str] = [node.text or ""]
    for child in node:
        if _local_name(child.tag) not in _JATS_SKIP_TEXT:
            pieces.append(_jats_text(child))
        pieces.append(child.tail or "")
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _first_direct(node: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in node if _local_name(child.tag) == name), None)


def _first_descendant(node: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in node.iter() if _local_name(child.tag) == name), None)


def _validate_xml_bounds(root: ET.Element, limits: StructureLimits) -> None:
    count = 0

    def visit(node: ET.Element, depth: int) -> None:
        nonlocal count
        count += 1
        if count > limits.max_nodes:
            raise ValueError("JATS document exceeds the XML-node limit")
        if depth > limits.max_xml_depth:
            raise ValueError("JATS document exceeds the XML-depth limit")
        for child in node:
            visit(child, depth + 1)

    visit(root, 1)


class _JatsParser:
    def __init__(self, builder: _StructureBuilder) -> None:
        self.builder = builder

    def _add_plain_paragraph(self, node: _MutableSection | None, paragraph: ET.Element) -> None:
        text = _jats_text(paragraph)
        if not text:
            return
        target = self.builder.unsectioned_raw if node is None else node.raw_parts
        if target and not target[-1].endswith("\n\n"):
            target.append("\n\n")
        target.append(text)

    def _add_asset(self, node: _MutableSection | None, element: ET.Element, kind: str) -> None:
        identifier = (element.attrib.get("id") or "").strip()
        if node is None:
            previous = self.builder.current
            self.builder.current = None
            self.builder.add_asset(kind, identifier)
            self.builder.current = previous
            return
        self.builder.asset_order += 1
        node.placements.append(
            AssetPlacement(kind, self.builder.asset_order, node.paragraph_count_so_far(), identifier)
        )

    def _content(
        self,
        container: ET.Element,
        node: _MutableSection | None,
        level: int,
    ) -> None:
        for child in container:
            tag = _local_name(child.tag)
            if tag in {"title", "label"}:
                continue
            if tag == "sec":
                self._section(child, level + 1, parent=node)
            elif tag == "p":
                self._add_plain_paragraph(node, child)
            elif tag == "fig":
                self._add_asset(node, child, "figure")
            elif tag == "table-wrap":
                self._add_asset(node, child, "table")
            elif tag == "table-wrap-group":
                self._content(child, node, level)
            elif tag == "table":
                self._add_asset(node, child, "table")
            elif tag in {"ref-list", "ack", "app", "app-group", "supplementary-material"}:
                continue
            else:
                self._content(child, node, level)

    def _section(
        self,
        element: ET.Element,
        level: int,
        *,
        parent: _MutableSection | None = None,
        forced_exclusion: str = "",
        forced_title: str = "",
    ) -> _MutableSection:
        title_node = _first_direct(element, "title")
        direct_tags = {_local_name(child.tag) for child in element}
        if title_node is None and "ref-list" in direct_tags:
            forced_exclusion = forced_exclusion or "references"
            forced_title = forced_title or "References"
        elif title_node is None and "ack" in direct_tags:
            forced_exclusion = forced_exclusion or "acknowledgments"
            forced_title = forced_title or "Acknowledgments"
        title = forced_title or (_jats_text(title_node) if title_node is not None else "Untitled section")
        sec_type = (element.attrib.get("sec-type") or "").strip().casefold()
        if sec_type in {"ack", "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}:
            forced_exclusion = "acknowledgments"
        elif sec_type in {"ref", "refs", "references", "bibliography"}:
            forced_exclusion = "references"
        elif sec_type in {"app", "appendix", "supplementary-material", "supplement"}:
            forced_exclusion = "appendix"
        included, reason = _classify_title(title, appendix_mode=forced_exclusion == "appendix")
        if forced_exclusion:
            included, reason = False, forced_exclusion

        self.builder.section_order += 1
        if self.builder.section_order > self.builder.limits.max_sections:
            raise ValueError("paper exceeds the section-count limit")
        roots = self.builder.appendices if reason == "appendix" and parent is None else self.builder.sections
        if parent is None:
            path = (len(roots) + 1,)
        else:
            if not parent.included_in_main:
                included, reason = False, parent.exclusion_reason
            path = parent.path + (len(parent.children) + 1,)
        node = _MutableSection(
            title or "Untitled section",
            level,
            self.builder.section_order,
            path,
            included,
            reason,
        )
        if parent is None:
            roots.append(node)
        else:
            parent.children.append(node)
        self._content(element, node, level)
        return node

    def body(self, body: ET.Element) -> None:
        for child in body:
            tag = _local_name(child.tag)
            if tag == "sec":
                self._section(child, 1)
            elif tag == "p":
                self._add_plain_paragraph(None, child)
            elif tag == "fig":
                self._add_asset(None, child, "figure")
            elif tag in {"table-wrap", "table"}:
                self._add_asset(None, child, "table")
            elif tag == "table-wrap-group":
                self._content(child, None, 0)
            elif tag == "ack":
                title_node = _first_direct(child, "title")
                title = _jats_text(title_node) if title_node is not None else "Acknowledgments"
                self._section(
                    child,
                    1,
                    forced_exclusion="acknowledgments",
                    forced_title=title,
                )
            elif tag == "ref-list":
                title_node = _first_direct(child, "title")
                title = _jats_text(title_node) if title_node is not None else "References"
                self._section(child, 1, forced_exclusion="references", forced_title=title)
            elif tag == "app-group":
                for app in child:
                    if _local_name(app.tag) == "app":
                        self._section(app, 1, forced_exclusion="appendix")
            elif tag in {"app", "supplementary-material"}:
                title = "Supplementary material" if tag == "supplementary-material" else ""
                self._section(child, 1, forced_exclusion="appendix", forced_title=title)
            else:
                # Body wrappers may contain top-level sections in some JATS producers.
                self._content(child, None, 0)

    def back(self, back: ET.Element) -> None:
        for child in back:
            tag = _local_name(child.tag)
            if tag == "ack":
                title_node = _first_direct(child, "title")
                title = _jats_text(title_node) if title_node is not None else "Acknowledgments"
                self._section(
                    child,
                    1,
                    forced_exclusion="acknowledgments",
                    forced_title=title,
                )
            elif tag == "ref-list":
                title_node = _first_direct(child, "title")
                title = _jats_text(title_node) if title_node is not None else "References"
                self._section(child, 1, forced_exclusion="references", forced_title=title)
            elif tag == "app-group":
                for app in child:
                    if _local_name(app.tag) == "app":
                        self._section(app, 1, forced_exclusion="appendix")
            elif tag == "app":
                self._section(child, 1, forced_exclusion="appendix")
            elif tag == "sec":
                self._section(child, 1)
            elif tag not in {"fn-group", "notes", "glossary"}:
                self.back(child)


def _jats_identifiers(article: ET.Element) -> tuple[tuple[str, str], ...]:
    identifiers: set[tuple[str, str]] = set()
    for node in article.iter():
        if _local_name(node.tag) != "article-id":
            continue
        value = _jats_text(node)
        if value:
            identifiers.add(((node.attrib.get("pub-id-type") or "unknown").strip().casefold(), value))
    return tuple(sorted(identifiers))


def _strip_external_doctypes(payload: bytes) -> tuple[bytes, int]:
    """Remove non-resolving external doctypes; reject any internal subset."""

    result = payload
    removed = 0
    while True:
        match = re.search(br"<!\s*DOCTYPE\b", result, flags=re.IGNORECASE)
        if match is None:
            return result, removed
        quote = 0
        internal_subset = False
        end = -1
        for index in range(match.end(), len(result)):
            byte = result[index]
            if quote:
                if byte == quote:
                    quote = 0
                continue
            if byte in {ord('"'), ord("'")}:
                quote = byte
            elif byte == ord("["):
                internal_subset = True
            elif byte == ord(">"):
                end = index + 1
                break
        if end < 0 or internal_subset:
            raise ValueError("JATS internal declarations and entities are not accepted")
        result = result[: match.start()] + result[end:]
        removed += 1


def parse_jats_structure(
    payload: bytes,
    *,
    limits: StructureLimits = DEFAULT_LIMITS,
) -> PaperStructure:
    """Parse one PMC/JATS article with hierarchy and layout counts intact."""

    if len(payload) > limits.max_payload_bytes:
        raise ValueError("JATS payload exceeds the configured size limit")
    if re.search(br"<!\s*ENTITY\b", payload, flags=re.IGNORECASE):
        raise ValueError("JATS internal declarations and entities are not accepted")
    xml_payload, doctypes_removed = _strip_external_doctypes(payload)
    root = ET.fromstring(xml_payload)
    _validate_xml_bounds(root, limits)
    article = root if _local_name(root.tag) == "article" else next(
        (node for node in root.iter() if _local_name(node.tag) == "article"),
        None,
    )
    if article is None:
        raise ValueError("JATS payload contains no article")
    title_node = _first_descendant(article, "article-title")
    title = _jats_text(title_node) if title_node is not None else ""
    builder = _StructureBuilder(limits=limits)
    for abstract in (node for node in article.iter() if _local_name(node.tag) == "abstract"):
        paragraphs = [_jats_text(node) for node in abstract.iter() if _local_name(node.tag) == "p"]
        builder.add_abstract("\n\n".join(part for part in paragraphs if part))
    parser = _JatsParser(builder)
    body = _first_direct(article, "body")
    warnings: list[str] = ["external_doctype_ignored"] if doctypes_removed else []
    if body is None:
        warnings.append("no_body")
    else:
        parser.body(body)
    back = _first_direct(article, "back")
    if back is not None:
        parser.back(back)
    if not builder.sections and not builder.appendices:
        warnings.append("no_section_headings")
    provenance = StructureProvenance(
        parser="spiral_jats_structure",
        parser_version=PARSER_VERSION,
        source_sha256=_sha256(payload),
        source_bytes=len(payload),
        warnings=tuple(warnings),
        identifiers=_jats_identifiers(article),
    )
    return builder.freeze(source_format="pmc_jats", title=title, provenance=provenance)


__all__ = [
    "AssetPlacement",
    "DEFAULT_LIMITS",
    "PARSER_VERSION",
    "PaperStructure",
    "STRUCTURE_SCHEMA",
    "SectionNode",
    "StructureLimits",
    "StructureProvenance",
    "parse_jats_structure",
    "parse_tex_structure_archive",
]
