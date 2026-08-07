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
import re
import shutil
import subprocess
import sys
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
    "llm": (("transformers",), "import transformers", "language model work"),
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

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


@dataclass
class Resolution:
    """The outcome of asking for a capability."""

    present: list[Need] = field(default_factory=list)
    declared: list[Need] = field(default_factory=list)     # written to a manifest
    blocked: list[Need] = field(default_factory=list)      # user must act

    def brief(self) -> str:
        """A few lines for the planner: what is here, what was added, what is not."""
        lines: list[str] = []
        if self.present:
            lines.append("Already available: " + ", ".join(
                sorted({p for need in self.present for p in
                        (need.packages or (need.binary,))})))
        if self.declared:
            lines.append("Added to this project's dependency manifest (the harness "
                         "installs them before the first gate run): " + ", ".join(
                             sorted({p for need in self.declared
                                     for p in need.packages})))
        for need in self.blocked:
            lines.append(
                f"NOT available and cannot be installed automatically: "
                f"{need.binary or need.id} — needed for {need.why}. "
                f"Plan around its absence, or the user must run: {need.install_hint}")
        return "\n".join(lines)


# A table word counts only as a whole word, plus the plain inflections it really
# takes ("charts", "games"). Agent nouns are excluded on purpose — see _mentions.
_INFLECTIONS = r"(?:s|es|d|ed|ing)?"


def _mentions(text: str, word: str) -> bool:
    """True when the table's word is used, not merely spelled somewhere inside.

    The bare ``word in text`` test read "gamers" as pygame work, "flowchart" as
    matplotlib work and every "ratios" as an iOS build — and a detected need is
    not a suggestion: it is written into requirements.txt, committed, and then
    really installed against the run's install budget.
    """
    return re.search(rf"\b{re.escape(word)}{_INFLECTIONS}\b", text) is not None


def detect_needs(goal: str, tool_families: list[str] | None = None) -> list[Need]:
    """Capabilities implied by the goal and by the analyst's declared tool families."""
    text = " ".join([goal or "", " ".join(tool_families or [])]).lower()
    needs: dict[str, Need] = {}
    for word, (packages, certificate, why) in DOMAIN_PACKAGES.items():
        if _mentions(text, word):
            needs.setdefault(f"python:{packages[0]}", Need(
                id=f"python:{packages[0]}", kind="python", packages=packages,
                certificate=certificate, why=why))
    for word, (binary, hint, why) in DOMAIN_BINARIES.items():
        if _mentions(text, word):
            needs.setdefault(f"binary:{binary}", Need(
                id=f"binary:{binary}", kind="binary", binary=binary,
                certificate=f"command -v {binary}", why=why, install_hint=hint))
    return sorted(needs.values(), key=lambda n: n.id)


def _venv_python(root: Path) -> Path | None:
    candidate = (root / ".spiral" / "dependency-cache" / "python" / "venv"
                 / "bin" / "python")
    return candidate if candidate.is_file() else None


def is_present(root: Path, need: Need) -> bool:
    """Prove it, do not assume it — run the certificate."""
    if need.kind == "binary":
        return shutil.which(need.binary) is not None
    if need.kind == "python":
        interpreter = _venv_python(root) or Path(sys.executable)
        try:
            done = subprocess.run(
                [str(interpreter), "-c", need.certificate],
                capture_output=True, text=True, timeout=180, cwd=root)
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0
    if need.kind == "model":
        try:
            done = subprocess.run(["ollama", "list"], capture_output=True,
                                  text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return False
        return need.id.split(":", 1)[-1] in done.stdout
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


def resolve(workspace: str | Path, goal: str,
            tool_families: list[str] | None = None,
            *, declare: bool = True) -> Resolution:
    """Work out the gap and close what can be closed by declaring it."""
    root = Path(workspace).resolve()
    outcome = Resolution()
    for need in detect_needs(goal, tool_families):
        if is_present(root, need):
            outcome.present.append(need)
            continue
        if need.kind == "python" and declare:
            added = declare_python(root, need.packages)
            if added:
                outcome.declared.append(Need(
                    id=need.id, kind=need.kind, packages=tuple(added),
                    certificate=need.certificate, why=need.why))
                continue
            # already declared but the certificate fails — provisioning will run,
            # so this is not yet a blockage
            outcome.present.append(need)
            continue
        outcome.blocked.append(need)
    return outcome


def write_capabilities(root: Path, outcome: Resolution) -> Path:
    path = Path(root) / ".spiral" / "capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "present": [n.to_json() for n in outcome.present],
        "declared": [n.to_json() for n in outcome.declared],
        "blocked": [n.to_json() for n in outcome.blocked],
    }, indent=2))
    return path


__all__ = [
    "Need", "Resolution", "detect_needs", "is_present", "declare_python",
    "resolve", "write_capabilities", "DOMAIN_PACKAGES", "DOMAIN_BINARIES",
]
