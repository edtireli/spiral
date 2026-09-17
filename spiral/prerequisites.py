"""Pure, bounded prerequisite declarations; no task inference or acquisition.

Runtime constraints and registry distributions are different types. Legacy
``python:`` names remain readable only when they are not also version literals.
The parser validates an entire list before callers may mutate project manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


MAX_FAMILY_BYTES = 160
MAX_FAMILIES = 64
NODE_REQUIREMENT = re.compile(
    r"^(?P<name>(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+)"
    r"(?:@(?P<version>[A-Za-z0-9*^~<>=_.+-]+))?$"
)
_BINARY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,100}")
_MODEL = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?"
)


class PrerequisiteError(ValueError):
    """Invalid declaration, not a missing package or permission to acquire it."""


@dataclass(frozen=True)
class Prerequisite:
    kind: str
    name: str
    value: str
    source: str


def python_requirement(value: str) -> Requirement:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise PrerequisiteError("Python distribution requirement must be single-line text")
    try:
        parsed = Requirement(value)
    except InvalidRequirement as exc:
        raise PrerequisiteError("invalid Python distribution requirement") from exc
    if parsed.url:
        raise PrerequisiteError("Python prerequisites require public registry names, not URLs or paths")
    return parsed


def parse_family(raw: str) -> Prerequisite:
    if not isinstance(raw, str):
        raise PrerequisiteError("prerequisite must be a typed string")
    try:
        size = len(raw.encode("utf-8"))
    except UnicodeError as exc:
        raise PrerequisiteError("prerequisite must be valid UTF-8") from exc
    if not raw.strip() or size > MAX_FAMILY_BYTES or any(ord(c) < 32 for c in raw):
        raise PrerequisiteError(f"prerequisite must be single-line text of at most {MAX_FAMILY_BYTES} bytes")
    source = raw.strip()
    namespace, separator, value = source.partition(":")
    namespace, value = namespace.lower(), value.strip()
    if not separator or not value:
        raise PrerequisiteError("use an explicit typed prerequisite, not prose")
    if namespace in {"python", "python-package"}:
        if namespace == "python":
            try:
                Version(value)
            except InvalidVersion:
                pass
            else:
                raise PrerequisiteError(
                    "ambiguous legacy python: version literal; declare python-runtime:SPECIFIER "
                    "for an interpreter constraint or python-package:REQUIREMENT for a distribution"
                )
        requirement = python_requirement(value)
        return Prerequisite("python", canonicalize_name(requirement.name), value, source)
    if namespace == "python-runtime":
        try:
            specifier = SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise PrerequisiteError(
                "python-runtime requires an explicit version comparison (for example >=3.11); "
                "it checks the selected interpreter and never installs a package"
            ) from exc
        return Prerequisite("runtime", "python", str(specifier), source)
    if namespace == "node":
        match = NODE_REQUIREMENT.fullmatch(value)
        if not match:
            raise PrerequisiteError("node prerequisite must name a public npm package, not a URL or path")
        return Prerequisite("node", match.group("name"), value, source)
    if namespace in {"brew", "binary"}:
        if not _BINARY.fullmatch(value):
            raise PrerequisiteError("binary/formula prerequisite must be a bounded name, not a command, tap or cask")
        return Prerequisite(namespace, value, value, source)
    if namespace in {"ollama", "local-model", "model"}:
        if not _MODEL.fullmatch(value):
            raise PrerequisiteError("local model prerequisite must be an explicit bounded model identifier")
        return Prerequisite("model", value, value, source)
    raise PrerequisiteError("unknown prerequisite type; runtime constraints are not installable packages")


def parse_families(values: list[str] | None) -> list[Prerequisite]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > MAX_FAMILIES:
        raise PrerequisiteError(f"prerequisites must be an array of at most {MAX_FAMILIES} typed strings")
    # Parse everything first. A valid first item cannot trigger an install or
    # manifest write before an invalid later item has been examined.
    parsed = [parse_family(value) for value in values]
    selected: dict[tuple[str, str], Prerequisite] = {}
    for item in parsed:
        key = (item.kind, item.name)
        previous = selected.get(key)
        if previous is not None and previous.value != item.value:
            raise PrerequisiteError(
                f"conflicting prerequisite declarations for {item.kind}:{item.name}; "
                "supply one explicit combined requirement"
            )
        selected.setdefault(key, item)
    return list(selected.values())
