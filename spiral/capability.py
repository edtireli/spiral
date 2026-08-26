"""What this build needs, what this machine has, and how to close the difference.

Spiral already owns every primitive for acquiring capability — dependency
provisioning per ecosystem, credential-free reference clones, a GET-only research
door, an empirical recipe registry, and an ``ASK: install`` protocol. What it never
had was the question. Nothing compared *what the work requires* against *what is
here*, so a worker facing a missing package spent ninety thousand tokens editing
source instead, and never once asked to install anything.

This module asks it. Needs come from two places that do not require a model to be
imaginative: the deliverable analyst's own ``tool_families``, and a table of
domain words whose implementation route is not really in doubt — a diffusion model
needs torch, a Reddit bot needs an API client, an Android app needs a JDK.

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
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# domain word -> (python packages, certificate expression, why)
# Only entries where the route is genuinely uncontroversial. A wrong guess here
# costs a wasted install, so the bar is "any practitioner would reach for this".
DOMAIN_PACKAGES: dict[str, tuple[tuple[str, ...], str, str]] = {
    "diffusion": (("torch", "diffusers", "transformers"),
                  "import torch, diffusers", "image generation with diffusers"),
    "stable diffusion": (("torch", "diffusers", "transformers"),
                         "import torch, diffusers", "image generation"),
    "neural network": (("torch",), "import torch; torch.zeros(1)",
                       "tensor maths and autograd"),
    "deep learning": (("torch",), "import torch; torch.zeros(1)", "training"),
    "train a model": (("torch",), "import torch; torch.zeros(1)", "training"),
    "transformer": (("torch", "transformers"), "import transformers", "LLM work"),
    "embedding": (("sentence-transformers",), "import sentence_transformers",
                  "text embeddings"),
    "computer vision": (("opencv-python", "numpy"), "import cv2, numpy",
                        "image processing"),
    "image processing": (("pillow",), "import PIL.Image", "raster image work"),
    "reddit": (("praw",), "import praw", "the Reddit API"),
    "discord": (("discord.py",), "import discord", "the Discord API"),
    "telegram": (("python-telegram-bot",), "import telegram", "the Telegram API"),
    "scrape": (("requests", "beautifulsoup4"), "import requests, bs4",
               "fetching and parsing pages"),
    # spelled out because matching is word-bounded and agent nouns are not
    # tolerated as an inflection: "scraper" is the work, but "gamer" is a person.
    "scraper": (("requests", "beautifulsoup4"), "import requests, bs4",
                "fetching and parsing pages"),
    "dataframe": (("pandas",), "import pandas", "tabular data"),
    "csv analysis": (("pandas",), "import pandas", "tabular data"),
    "plot": (("matplotlib",), "import matplotlib", "charts"),
    "chart": (("matplotlib",), "import matplotlib", "charts"),
    "pdf": (("pypdf",), "import pypdf", "reading and writing PDFs"),
    "spreadsheet": (("openpyxl",), "import openpyxl", "xlsx files"),
    "fastapi": (("fastapi", "uvicorn"), "import fastapi", "the web framework"),
    "flask": (("flask",), "import flask", "the web framework"),
    "sqlalchemy": (("sqlalchemy",), "import sqlalchemy", "the ORM"),
    "websocket": (("websockets",), "import websockets", "socket transport"),
    "audio": (("soundfile", "numpy"), "import soundfile, numpy", "audio io"),
    "speech": (("openai-whisper",), "import whisper", "transcription"),
    "game": (("pygame",), "import pygame", "the game loop and rendering"),
    "classifier": (("scikit-learn", "numpy"), "import sklearn, numpy",
                   "fitting and evaluating a model"),
    "scikit": (("scikit-learn", "numpy"), "import sklearn, numpy", "modelling"),
    "random forest": (("scikit-learn",), "import sklearn.ensemble", "the ensemble"),
    # qualified, never bare: in the goals this CLI is handed "regression" is
    # overwhelmingly a regression *test*, so the bare word bought scikit-learn
    # and numpy for "add regression tests for the parser".
    "linear regression": (("scikit-learn", "numpy"), "import sklearn, numpy",
                          "fitting"),
    "logistic regression": (("scikit-learn", "numpy"), "import sklearn, numpy",
                            "fitting"),
    "regression model": (("scikit-learn", "numpy"), "import sklearn, numpy",
                         "fitting"),
    "generate a pdf": (("reportlab",), "import reportlab", "laying out a PDF"),
    "pdf report": (("reportlab",), "import reportlab", "laying out a PDF"),
}

# domain word -> (binary, install hint, why). Reported, never installed silently.
DOMAIN_BINARIES: dict[str, tuple[str, str, str]] = {
    "android": ("java", "brew install --cask temurin", "the JDK gradle needs"),
    "ios": ("xcodebuild", "install Xcode from the App Store", "building for iOS"),
    "video": ("ffmpeg", "brew install ffmpeg", "encoding and decoding video"),
    "ffmpeg": ("ffmpeg", "brew install ffmpeg", "media processing"),
    "docker": ("docker", "install Docker Desktop", "container builds"),
    # its own entry because matching is word-bounded and this is how the word
    # actually shows up in a goal ("add a Dockerfile").
    "dockerfile": ("docker", "install Docker Desktop", "container builds"),
    "latex": ("pdflatex", "brew install --cask mactex", "typesetting"),
}


@dataclass
class Need:
    """One capability the build requires, and how to prove it is present."""

    id: str
    kind: str                       # "python" | "node" | "binary" | "model"
    packages: tuple[str, ...] = ()
    certificate: str = ""           # python expression or shell command
    why: str = ""
    binary: str = ""
    install_hint: str = ""
    setup_request: str = ""          # typed broker request, never an arbitrary shell
    access: str = "workspace"         # "workspace" | "full-access"

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
                        (need.packages or (need.binary,))})))
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


# A table word counts only as a whole word, plus the plain inflections it really
# takes ("charts", "games"). Agent nouns are excluded on purpose — see _mentions.
_INFLECTIONS = r"(?:s|es|d|ed|ing)?"
_OLLAMA_MODEL = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$"
)
_NODE_REQUIREMENT = re.compile(
    r"^(?P<name>(?:@[A-Za-z0-9_.-]+/)?[A-Za-z0-9_.-]+)"
    r"(?:@(?P<version>[A-Za-z0-9*^~<>=_.+-]+))?$"
)
_HUGGING_FACE = re.compile(r"\b(?:hugging[\s-]+face|huggingface)\b", re.I)
_MODEL_RUNTIME_ARTIFACT = re.compile(
    r"\b(?:checkpoint|pipeline|pretrained|weights?)\b",
    re.I,
)
_LOCAL_EXECUTION = re.compile(r"\b(?:local|locally|offline|on[ -]device)\b", re.I)
_MODEL_EXECUTION_ACTION = re.compile(
    r"\b(?:infer(?:s|red|ring)?|inference|load(?:s|ed|ing)?)\b", re.I,
)
_LOCAL_PYTHON_MODEL_RUNTIME = (
    re.compile(r"\bpython\b", re.I),
    _LOCAL_EXECUTION,
    re.compile(
        r"\b(?:pretrained\s+(?:language[- ]?)?model|model\s+(?:checkpoint|weights?))\b",
        re.I,
    ),
    re.compile(r"\bload(?:s|ed|ing)?\b", re.I),
    re.compile(r"\b(?:infer(?:s|red|ring)?|inference)\b", re.I),
)


def _mentions(text: str, word: str) -> bool:
    """True when the table's word is used, not merely spelled somewhere inside.

    The bare ``word in text`` test read "gamers" as pygame work, "flowchart" as
    matplotlib work and every "ratios" as an iOS build — and a detected need is
    not a suggestion: it is written into requirements.txt, committed, and then
    really installed against the run's install budget.
    """
    return re.search(rf"\b{re.escape(word)}{_INFLECTIONS}\b", text) is not None


def _requires_transformers_runtime(text: str) -> bool:
    """Recognise an actual in-process model runtime, not an API client.

    ``LLM`` describes what many products talk *to*; it does not identify their
    Python implementation.  Named Hugging Face model work is strong evidence.
    Without a named framework, require the much narrower combination of Python,
    local execution, loading model weights/checkpoints, and inference.
    """

    if _HUGGING_FACE.search(text):
        if _MODEL_RUNTIME_ARTIFACT.search(text):
            return True
        if _LOCAL_EXECUTION.search(text) and _MODEL_EXECUTION_ACTION.search(text):
            return True
    return all(pattern.search(text) for pattern in _LOCAL_PYTHON_MODEL_RUNTIME)


def detect_needs(goal: str, tool_families: list[str] | None = None) -> list[Need]:
    """Capabilities implied by the goal and by the analyst's declared tool families."""
    typed_prefix = re.compile(
        r"\s*(?:python|node|brew|binary|ollama|local-model|model)\s*[:=]",
        re.I,
    )
    legacy_families = [
        str(value) for value in (tool_families or [])
        if not typed_prefix.match(str(value))
    ]
    text = " ".join([goal or "", " ".join(legacy_families)]).lower()
    needs: dict[str, Need] = {}
    # The analyst's generic acquisition vocabulary. Values still cross the same
    # registry/formula/model validators as ASK: install; this is a typed hint, not
    # permission to execute prose or a generated shell command.
    for raw in tool_families or []:
        family = str(raw).strip()
        typed = re.fullmatch(
            r"(python|node|brew|binary)\s*:\s*(\S+)", family, re.I)
        if not typed:
            continue
        ecosystem, value = typed.group(1).lower(), typed.group(2)
        if ecosystem == "python":
            try:
                from packaging.requirements import Requirement
            except Exception:
                from pip._vendor.packaging.requirements import Requirement  # type: ignore
            try:
                parsed = Requirement(value)
            except Exception:
                continue
            if parsed.url:
                continue
            distribution = parsed.name
            needs.setdefault(f"python:{distribution.lower()}", Need(
                id=f"python:{distribution.lower()}", kind="python",
                packages=(value,),
                certificate=(
                    "import importlib.metadata as m; "
                    f"m.version({json.dumps(distribution)})"
                ),
                why=f"deliverable analyst requested Python distribution {distribution}",
                setup_request=f"python {value}", access="workspace",
            ))
        elif ecosystem == "node" and _NODE_REQUIREMENT.fullmatch(value):
            package_name = _NODE_REQUIREMENT.fullmatch(value).group("name")  # type: ignore[union-attr]
            needs.setdefault(f"node:{package_name.lower()}", Need(
                id=f"node:{package_name.lower()}", kind="node",
                packages=(value,), certificate=f"npm list {package_name}",
                why=f"deliverable analyst requested Node package {package_name}",
                setup_request=f"node {value}", access="workspace",
            ))
        elif ecosystem == "brew" and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,100}", value):
            needs.setdefault(f"binary:{value}", Need(
                id=f"binary:{value}", kind="binary", binary=value,
                certificate=f"command -v {value}",
                why=f"deliverable analyst requested Homebrew core formula {value}",
                install_hint=f"brew install {value}", setup_request=f"brew {value}",
                access="full-access",
            ))
        elif ecosystem == "binary" and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,100}", value):
            needs.setdefault(f"binary:{value}", Need(
                id=f"binary:{value}", kind="binary", binary=value,
                certificate=f"command -v {value}",
                why=f"deliverable analyst requires the existing binary {value}",
                install_hint="supply this binary through the approved host profile",
            ))
    for word, (packages, certificate, why) in DOMAIN_PACKAGES.items():
        if _mentions(text, word):
            needs.setdefault(f"python:{packages[0]}", Need(
                id=f"python:{packages[0]}", kind="python", packages=packages,
                certificate=certificate, why=why))
    if _requires_transformers_runtime(text):
        needs.setdefault("python:transformers", Need(
            id="python:transformers", kind="python", packages=("transformers",),
            certificate="import transformers",
            why="explicit local Hugging Face or Python model inference work",
        ))
    for word, (binary, hint, why) in DOMAIN_BINARIES.items():
        if _mentions(text, word):
            formula = ""
            match = re.fullmatch(r"brew install ([A-Za-z0-9][A-Za-z0-9_.+-]{0,100})", hint)
            if match:
                formula = match.group(1)
            needs.setdefault(f"binary:{binary}", Need(
                id=f"binary:{binary}", kind="binary", binary=binary,
                certificate=f"command -v {binary}", why=why, install_hint=hint,
                setup_request=f"brew {formula}" if formula else "",
                access="full-access" if formula else "workspace"))

    # A project-local model is too large to infer from a vague word such as "AI".
    # Admit only an explicit typed family from the deliverable manifest, or an
    # explicit ``ollama model NAME`` / ``ollama:NAME`` phrase in the user's goal.
    model_names: list[str] = []
    for family in tool_families or []:
        match = re.fullmatch(
            r"\s*(?:ollama|local-model|model)\s*[:=]\s*(\S+)\s*",
            str(family), re.I,
        )
        if match:
            model_names.append(match.group(1))
    for pattern in (
        r"\bollama\s+model\s+([A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?)",
        r"\bollama\s*:\s*([A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?)",
    ):
        model_names.extend(re.findall(pattern, goal or "", re.I))
    for name in model_names:
        if not _OLLAMA_MODEL.fullmatch(name):
            continue
        needs.setdefault(f"model:{name}", Need(
            id=f"model:{name}", kind="model", binary=name,
            certificate=f"ollama show {name}",
            why="the generated product's local model runtime",
            install_hint=f"ollama pull {name}", setup_request=f"ollama {name}",
            access="full-access",
        ))
    return sorted(needs.values(), key=lambda n: n.id)


def _venv_python(root: Path) -> Path | None:
    candidate = (root / ".spiral" / "dependency-cache" / "python" / "venv"
                 / "bin" / "python")
    return candidate if candidate.is_file() else None


def is_present(root: Path, need: Need) -> bool:
    """Prove it, do not assume it — run the certificate."""
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


def declare_python(root: Path, packages: tuple[str, ...]) -> list[str]:
    """Add packages to requirements.txt, leaving existing pins alone.

    Declaring rather than installing keeps one acquisition path: the provisioning
    that already runs before every gate picks these up, inside the sandbox and
    against the install budget, and the repo records what the project depends on.
    """
    target = _requirements_file(root)
    existing_text = target.read_text() if target.is_file() else ""
    existing = {
        re.split(r"[<>=!\[ ]", line.strip())[0].lower()
        for line in existing_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    added = [p for p in packages if p.split("[")[0].lower() not in existing]
    if not added:
        return []
    lines = existing_text.splitlines()
    if lines and lines[-1].strip() == "":
        lines = lines[:-1]
    lines += added
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    return added


def declare_node(root: Path, requirements: tuple[str, ...]) -> list[str]:
    """Record typed npm dependencies without accepting URLs, files, taps, or hooks."""

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
    for need in detect_needs(goal, tool_families):
        # A project dependency is a durable product fact, not merely a property
        # of the current host. Declare it even when Spiral's own interpreter can
        # import it; a clean checkout must acquire the same dependency later.
        if need.kind == "python" and declare:
            added = declare_python(root, need.packages)
            if added:
                outcome.declared.append(Need(
                    id=need.id, kind=need.kind, packages=tuple(added),
                    certificate=need.certificate, why=need.why,
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
        for raw in deliverable.get("tool_families") or []:
            value = str(raw).strip()
            if value and value not in families:
                families.append(value[:160])
    return families[:64]


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
        # Goal-only inspection/setup and the stronger analyst-family pass happen
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
    "manifest_tool_families", "write_capabilities", "DOMAIN_PACKAGES",
    "DOMAIN_BINARIES",
]
