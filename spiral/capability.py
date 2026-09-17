"""What this build needs, what this machine has, and how to close the difference.

Spiral already owns every primitive for acquiring capability — dependency
provisioning per ecosystem, credential-free reference clones, a GET-only research
door, an empirical recipe registry, and an ``ASK: install`` protocol. What it never
had was the question. Nothing compared *what the work requires* against *what is
here*, so a worker facing a missing package spent ninety thousand tokens editing
source instead, and never once asked to install anything.

This module consumes explicit validated analyst prerequisites. Goal words, site
names and legacy prose do not select packages, binaries or models. Runtime
constraints are observations of the selected interpreter, never packages to install.

Resolution deliberately does NOT install anything itself. It DECLARES the
dependency in the manifest the project already uses (``requirements.txt``,
``package.json``), and the existing provisioning installs it — sandboxed, budgeted,
and recorded. One acquisition path, not two, and the declaration is a durable fact
in the repo rather than a mutation of someone's machine. Things that genuinely
cannot be declared this way are separated out:

* a **model** to pull (gigabytes, so it is opt-in and the size is printed),
* a **reference repository** to clone (already gated behind ``--auto-repos``),
* a **system binary** the user must install, reported with the exact command
  rather than silently ``brew install``-ed.

Every capability carries a *certificate*: a command that exits 0 only when the
capability actually works. "pip said ok" is not evidence; importing the module and
using it is.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from spiral.prerequisites import (
    NODE_REQUIREMENT as _NODE_REQUIREMENT,
    PrerequisiteError, parse_families, python_requirement,
)

@dataclass
class Need:
    """One capability the build requires, and how to prove it is present."""

    id: str
    kind: str                       # "python" | "node" | "binary" | "model" | "runtime"
    packages: tuple[str, ...] = ()
    certificate: str = ""           # python expression or shell command
    why: str = ""
    binary: str = ""
    install_hint: str = ""
    setup_request: str = ""          # typed broker request, never an arbitrary shell
    access: str = "workspace"         # "workspace" | "full-access"
    runtime_specifier: str = ""
    runtime_observation: dict = field(default_factory=dict)
    registry_requirement: str = ""

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Resolution:
    """The outcome of asking for a capability."""

    present: list[Need] = field(default_factory=list)
    declared: list[Need] = field(default_factory=list)     # written to a manifest
    acquired: list[Need] = field(default_factory=list)     # installed + re-certified
    blocked: list[Need] = field(default_factory=list)      # user must act
    setup_reports: list[dict] = field(default_factory=list)
    inspection: dict = field(default_factory=dict)

    def brief(self) -> str:
        """A few lines for the planner: what is here, what was added, what is not."""
        lines: list[str] = []
        if self.inspection:
            roots = ", ".join(self.inspection.get("project_roots") or ["."])
            manifests = ", ".join(
                self.inspection.get("dependency_manifests") or []) or "none"
            design = ", ".join(
                self.inspection.get("design_inputs") or []) or "none"
            lines.append(
                f"Inspected existing workspace before planning: roots {roots}; "
                f"dependency manifests {manifests}; design inputs {design}.")
        if self.present:
            lines.append("Already available: " + ", ".join(
                sorted({p for need in self.present for p in
                        (need.packages or (need.binary or need.id,))})))
        for need in [*self.present, *self.blocked]:
            if need.runtime_observation:
                observation = need.runtime_observation
                lines.append(
                    f"Selected Python runtime: {observation['executable']} "
                    f"version {observation['version']}; constraint {need.runtime_specifier} "
                    f"satisfied={observation['satisfied']}. This does not certify a different project venv.")
        if self.declared:
            lines.append("Added to this project's dependency manifest (the harness "
                         "installs them before the first gate run): " + ", ".join(
                             sorted({p for need in self.declared
                                     for p in need.packages})))
        if self.acquired:
            lines.append("Acquired and certified during preflight: " + ", ".join(
                sorted({
                    need.binary or (need.packages[0] if need.packages else need.id)
                    for need in self.acquired
                })))
        for need in self.blocked:
            lines.append(
                f"NOT available and cannot be installed automatically: "
                f"{need.binary or need.id} — needed for {need.why}. "
                f"Plan around its absence, or the user must run: {need.install_hint}")
        return "\n".join(lines)


def detect_needs(goal: str, tool_families: list[str] | None = None) -> list[Need]:
    """Resolve explicit declarations only; goal prose never selects dependencies."""
    needs = []
    for declaration in parse_families(tool_families):
        kind, name, value = declaration.kind, declaration.name, declaration.value
        if kind == "python":
            needs.append(Need(
                id=f"python:{name}", kind="python", packages=(value,),
                registry_requirement=value,
                certificate=("import importlib.metadata as m; "
                             f"m.version({json.dumps(name)})"),
                why=f"deliverable analyst requested Python distribution {name}",
                setup_request=f"python {value}", access="workspace",
            ))
        elif kind == "runtime":
            needs.append(Need(
                id=f"python-runtime:{value}", kind="runtime",
                runtime_specifier=value,
                why="explicit constraint on the selected Python interpreter",
                install_hint="use an explicitly approved compatible runtime or revise the declared constraint",
            ))
        elif kind == "node":
            needs.append(Need(
                id=f"node:{name.lower()}", kind="node", packages=(value,),
                certificate=f"npm list {name}",
                why=f"deliverable analyst requested Node package {name}",
                setup_request=f"node {value}", access="workspace",
            ))
        elif kind in {"brew", "binary"}:
            formula = kind == "brew"
            needs.append(Need(
                id=f"binary:{name}", kind="binary", binary=name,
                certificate=f"command -v {name}",
                why=f"deliverable analyst requires {'Homebrew core formula' if formula else 'existing binary'} {name}",
                install_hint=f"brew install {name}" if formula else "supply this binary through the approved host profile",
                setup_request=f"brew {name}" if formula else "",
                access="full-access" if formula else "workspace",
            ))
        elif kind == "model":
            needs.append(Need(
                id=f"model:{name}", kind="model", binary=name,
                certificate=f"ollama show {name}",
                why="explicitly declared local model runtime",
                install_hint=f"ollama pull {name}", setup_request=f"ollama {name}",
                access="full-access",
            ))
    return sorted(needs, key=lambda need: need.id)


def _venv_python(root: Path) -> Path | None:
    candidate = (root / ".spiral" / "dependency-cache" / "python" / "venv"
                 / "bin" / "python")
    return candidate if candidate.is_file() else None


def is_present(root: Path, need: Need) -> bool:
    """Prove it, do not assume it — run the certificate."""
    if need.kind == "runtime":
        # Observe this already-selected trusted interpreter, never invoke a
        # project-controlled Python or turn a version into a pip requirement.
        version = platform.python_version()
        satisfied = SpecifierSet(need.runtime_specifier).contains(Version(version))
        need.runtime_observation = {
            "executable": sys.executable, "version": version,
            "satisfied": satisfied, "scope": "selected_engine_interpreter_only",
        }
        return satisfied
    if need.kind == "binary":
        if shutil.which(need.binary) is not None:
            return True
        if need.setup_request.startswith("brew "):
            brew = shutil.which("brew")
            if not brew:
                return False
            formula = need.setup_request.split(" ", 1)[1]
            try:
                done = subprocess.run(
                    [brew, "list", "--formula", formula], capture_output=True,
                    text=True, timeout=60, stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            return done.returncode == 0
        return False
    if need.kind == "python":
        interpreter = _venv_python(root) or Path(sys.executable)
        if need.registry_requirement:
            requirement = python_requirement(need.registry_requirement)
            # Metadata version checks do not certify optional dependency graphs
            # or another interpreter's marker environment. Preserve these in the
            # manifest for its resolver, without falsely calling them present.
            if requirement.extras or requirement.marker is not None:
                return False
            try:
                done = subprocess.run(
                    [str(interpreter), "-I", "-c",
                     "import importlib.metadata as m; "
                     f"print(m.version({json.dumps(requirement.name)}))"],
                    capture_output=True, text=True, timeout=180, cwd=root,
                    stdin=subprocess.DEVNULL,
                )
                version = done.stdout.strip()
                return (done.returncode == 0 and len(version) <= 128
                        and requirement.specifier.contains(Version(version)))
            except (OSError, subprocess.SubprocessError, ValueError):
                return False
        try:
            done = subprocess.run(
                [str(interpreter), "-c", need.certificate],
                capture_output=True, text=True, timeout=180, cwd=root)
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0
    if need.kind == "node":
        package = (need.packages[0] if need.packages else "")
        if package.startswith("@"):
            at = package.rfind("@")
            if at > package.find("/"):
                package = package[:at]
        else:
            package = package.split("@", 1)[0]
        parts = package.split("/")
        for node_modules in (
            root / "node_modules",
            root / ".spiral" / "tooling" / "node" / "node_modules",
        ):
            if node_modules.joinpath(*parts).is_dir():
                return True
        return False
    if need.kind == "model":
        try:
            done = subprocess.run(
                ["ollama", "show", need.binary or need.id.split(":", 1)[-1]],
                capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0
    return False


def _requirements_file(root: Path) -> Path:
    for name in ("requirements.txt", "requirements/base.txt"):
        path = root / name
        if path.is_file():
            return path
    return root / "requirements.txt"


def declare_python(root: Path, packages: tuple[str, ...], *, dry_run: bool = False) -> list[str]:
    """Add packages to requirements.txt, leaving existing pins alone.

    Declaring rather than installing keeps one acquisition path: the provisioning
    that already runs before every gate picks these up, inside the sandbox and
    against the install budget, and the repo records what the project depends on.
    """
    parse_families([f"python-package:{package}" for package in packages])
    parsed_packages = [python_requirement(package) for package in packages]
    target = _requirements_file(root)
    existing_text = target.read_text() if target.is_file() else ""
    existing = set()
    existing_requirements = {}
    for line in existing_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            parsed = python_requirement(line.strip())
            name = canonicalize_name(parsed.name)
            existing.add(name)
            existing_requirements.setdefault(name, []).append(parsed)
        except PrerequisiteError:
            # Preserve existing pip options, includes and invalid/user-authored
            # entries verbatim. This path never silently repairs an old manifest.
            existing.add(canonicalize_name(re.split(r"[<>=!\[ ]", line.strip())[0]))
    added = []
    for package, parsed in zip(packages, parsed_packages):
        name = canonicalize_name(parsed.name)
        for previous in existing_requirements.get(name, []):
            if ((parsed.specifier and str(parsed.specifier) != str(previous.specifier))
                    or not parsed.extras.issubset(previous.extras)
                    or str(parsed.marker or "") != str(previous.marker or "")):
                raise PrerequisiteError(
                    f"existing Python declaration for {name} does not encode the new explicit "
                    "constraint/extras/marker; reconcile the manifest explicitly; existing content was not changed"
                )
        if name not in existing:
            added.append(package)
            existing.add(name)
    if not added:
        return []
    lines = existing_text.splitlines()
    if lines and lines[-1].strip() == "":
        lines = lines[:-1]
    lines += added
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n")
    return added


def declare_node(root: Path, requirements: tuple[str, ...]) -> list[str]:
    """Record typed npm dependencies without accepting URLs, files, taps, or hooks."""
    parse_families([f"node:{requirement}" for requirement in requirements])
    target = Path(root) / "package.json"
    if target.is_file():
        try:
            data = json.loads(target.read_text())
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
    else:
        safe_name = re.sub(r"[^a-z0-9._-]+", "-", Path(root).name.lower()).strip("-")
        data = {
            "name": safe_name or "spiral-project",
            "version": "0.0.0",
            "private": True,
        }
    existing = set()
    for section in (
            "dependencies", "devDependencies", "optionalDependencies",
            "peerDependencies"):
        values = data.get(section) or {}
        if isinstance(values, dict):
            existing.update(str(name) for name in values)
    dependencies = data.get("dependencies")
    if dependencies is None:
        dependencies = {}
        data["dependencies"] = dependencies
    if not isinstance(dependencies, dict):
        return []
    added: list[str] = []
    for requirement in requirements:
        parsed = _NODE_REQUIREMENT.fullmatch(requirement)
        if not parsed:
            continue
        name = parsed.group("name")
        if name in existing:
            continue
        dependencies[name] = parsed.group("version") or "*"
        existing.add(name)
        added.append(requirement)
    if not added:
        return []
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(target)
    return added


def resolve(workspace: str | Path, goal: str,
            tool_families: list[str] | None = None,
            *, declare: bool = True) -> Resolution:
    """Work out the gap and close what can be closed by declaring it."""
    root = Path(workspace).resolve()
    outcome = Resolution()
    needs = detect_needs(goal, tool_families)
    if declare:
        # Existing constraint conflicts must also be found before any earlier
        # valid declaration can modify requirements.txt or package.json.
        declare_python(root, tuple(package for need in needs if need.kind == "python"
                                   for package in need.packages), dry_run=True)
    for need in needs:
        # A project dependency is a durable product fact, not merely a property
        # of the current host. Declare it even when Spiral's own interpreter can
        # import it; a clean checkout must acquire the same dependency later.
        if need.kind == "python" and declare:
            added = declare_python(root, need.packages)
            if added:
                outcome.declared.append(Need(
                    id=need.id, kind=need.kind, packages=tuple(added),
                    certificate=need.certificate, why=need.why,
                    registry_requirement=need.registry_requirement,
                    setup_request=need.setup_request, access=need.access))
            elif is_present(root, need):
                outcome.present.append(need)
            else:
                # It was already in the manifest but its project environment is
                # not synchronized yet. Keeping it in ``declared`` triggers the
                # preflight dependency lane without rewriting the manifest.
                outcome.declared.append(need)
            continue
        if need.kind == "node" and declare:
            added = declare_node(root, need.packages)
            if added:
                outcome.declared.append(Need(
                    id=need.id, kind=need.kind, packages=tuple(added),
                    certificate=need.certificate, why=need.why,
                    setup_request=need.setup_request, access=need.access))
            elif is_present(root, need):
                outcome.present.append(need)
            else:
                # If a valid package.json already declares it, synchronization
                # is still the next setup action. An invalid manifest remains a
                # blocked, inspectable project fault rather than being rewritten.
                target = root / "package.json"
                try:
                    declared = target.is_file() and need.id.split(":", 1)[1] in {
                        str(name).lower()
                        for section in (
                            "dependencies", "devDependencies", "optionalDependencies",
                            "peerDependencies")
                        for name in (json.loads(target.read_text()).get(section) or {})
                    }
                except Exception:
                    declared = False
                if declared:
                    outcome.declared.append(need)
                else:
                    outcome.blocked.append(need)
            continue
        if is_present(root, need):
            outcome.present.append(need)
            continue
        outcome.blocked.append(need)
    return outcome


def manifest_tool_families(manifest: dict | None) -> list[str]:
    """Flatten the analyst's typed tool-family evidence without trusting prose."""

    families: list[str] = []
    for deliverable in (manifest or {}).get("deliverables") or []:
        if not isinstance(deliverable, dict):
            continue
        values = deliverable.get("tool_families", [])
        if not isinstance(values, list) or len(values) > 24:
            raise PrerequisiteError("each deliverable permits at most 24 typed prerequisites")
        parse_families(values)
        for raw in values:
            if not isinstance(raw, str):
                raise PrerequisiteError("prerequisite must be a typed string")
            value = raw.strip()
            if value and value not in families:
                families.append(value)
    parse_families(families)
    return families


def inspect_workspace(workspace: str | Path) -> dict:
    """Record the deterministic design/tool surface before a model plans edits."""

    root = Path(workspace).resolve()
    try:
        from spiral.builder_tools import discover_project_roots

        roots = discover_project_roots(root)
    except Exception:
        roots = [root]
    manifest_names = (
        "pyproject.toml", "requirements.txt", "package.json", "Cargo.toml",
        "go.mod", "Makefile", "CMakeLists.txt", "build.gradle",
        "build.gradle.kts", "pom.xml", "Package.swift",
    )
    design_names = (
        ".spiral/design.md", ".spiral/design_tokens.json", "tokens.css",
        "README.md", "DESIGN.md", "design.md", "figma.json",
    )
    manifests: list[str] = []
    for project_root in roots:
        for name in manifest_names:
            path = project_root / name
            if path.is_file():
                manifests.append(str(path.relative_to(root)))
    design = [name for name in design_names if (root / name).is_file()]
    file_count = 0
    ignored = {
        ".git", ".spiral", "node_modules", ".venv", "venv", "target",
        "build", "dist", "__pycache__", ".pytest_cache",
    }
    for _current, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in ignored]
        file_count += len(files)
        if file_count >= 200_000:
            file_count = 200_000
            break
    return {
        "schema_version": 1,
        "project_roots": [str(path.relative_to(root) or Path(".")) for path in roots],
        "dependency_manifests": sorted(set(manifests)),
        "design_inputs": design,
        "workspace_files": file_count,
    }


def setup_capabilities(
    workspace: str | Path,
    goal: str,
    tool_families: list[str] | None = None,
    *,
    declare: bool = True,
    synchronize_projects: bool | str = True,
    tool_auto: bool = True,
    full_access: bool = False,
    timeout: int = 900,
    allow_scripts: bool = False,
    broker=None,
) -> Resolution:
    """Inspect, declare, acquire, and certify capabilities through typed brokers.

    Project dependencies remain inside the workspace cache and keep lifecycle
    scripts off by default.  Host-changing Homebrew installs and multi-gigabyte
    Ollama pulls are considered only under the run's immutable full-access grant.
    """

    root = Path(workspace).resolve()
    outcome = resolve(root, goal, tool_families, declare=declare)
    outcome.inspection = inspect_workspace(root)

    should_synchronize = bool(synchronize_projects) and (
        synchronize_projects != "if-declared" or bool(outcome.declared)
    )
    if should_synchronize and tool_auto:
        try:
            from spiral.builder_tools import ensure_builder_dependencies

            dependency_report = ensure_builder_dependencies(
                root, timeout=timeout, allow_scripts=allow_scripts,
            )
        except Exception as exc:
            dependency_report = {
                "applicable": True, "ok": False, "failure_kind": "transient",
                "detail": f"dependency preflight unavailable: {type(exc).__name__}: {exc}",
            }
        if dependency_report.get("applicable"):
            outcome.setup_reports.append({
                "kind": "project-dependencies",
                "ok": bool(dependency_report.get("ok")),
                "changed": bool(dependency_report.get("changed")),
                "failure_kind": str(dependency_report.get("failure_kind") or ""),
                "detail": str(dependency_report.get("detail") or "")[:2000],
                "reports": dependency_report.get("reports") or [],
            })

    if not tool_auto:
        return outcome
    if broker is None:
        try:
            from spiral.command_broker import CommandBroker

            broker = CommandBroker(root)
        except Exception:
            broker = None
    if broker is None:
        return outcome

    still_blocked: list[Need] = []
    for need in outcome.blocked:
        if not need.setup_request:
            still_blocked.append(need)
            continue
        if need.access == "full-access" and not full_access:
            still_blocked.append(need)
            continue
        try:
            typed = getattr(broker, "provision_typed", None)
            if callable(typed):
                result = typed(
                    need.setup_request, timeout=timeout, full_access=full_access)
                ok = bool(result.ok)
                detail = str(result.message)
                failure_kind = str(result.failure_kind)
            else:
                detail = str(broker.provision(
                    need.setup_request, timeout=timeout, full_access=full_access))
                ok = detail.startswith("tool installed:")
                failure_kind = "" if ok else "setup"
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
            failure_kind = "transient"
        certified = ok and is_present(root, need)
        outcome.setup_reports.append({
            "kind": need.kind, "id": need.id, "request": need.setup_request,
            "ok": certified, "failure_kind": "" if certified else failure_kind,
            "detail": detail[:2000],
        })
        if certified:
            outcome.acquired.append(need)
        else:
            still_blocked.append(need)
    outcome.blocked = still_blocked
    return outcome


def write_capabilities(root: Path, outcome: Resolution) -> Path:
    path = Path(root) / ".spiral" / "capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    phase = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inspection": outcome.inspection,
        "present": [n.to_json() for n in outcome.present],
        "declared": [n.to_json() for n in outcome.declared],
        "acquired": [n.to_json() for n in outcome.acquired],
        "blocked": [n.to_json() for n in outcome.blocked],
        "setup": outcome.setup_reports,
    }
    phases: list[dict] = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text())
            phases = [
                row for row in previous.get("phases") or []
                if isinstance(row, dict)
            ][-15:]
        except Exception:
            phases = []
    payload = {
        "schema_version": 2,
        **phase,
        # Existing-manifest inspection/setup and the analyst-family pass happen
        # at different points before editing. Preserve both receipts even though
        # the top-level fields intentionally expose the latest effective state.
        "phases": [*phases, phase],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
    return path


__all__ = [
    "Need", "Resolution", "detect_needs", "is_present", "declare_python",
    "declare_node",
    "resolve", "setup_capabilities", "inspect_workspace",
    "manifest_tool_families", "write_capabilities",
]
