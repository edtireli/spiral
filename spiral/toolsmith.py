"""Local empirical capability registry for Spiral's build and research tools.

The registry learns from executed gates and certificates, not model confidence.
It records success rates, latency, failure classes, available binaries, and compact
recipes from successful runs.  Future planners can prefer tools that have actually
worked on this machine while retaining exploration for untried methods.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path


KNOWN_TOOLS = {
    "python": ["python3", "python"],
    "lean": ["lean"],
    "singular": ["Singular", "singular"],
    "c++": ["clang++", "c++", "g++"],
    "c": ["clang", "cc", "gcc"],
    "node": ["node"],
    "npm": ["npm"],
    "rust": ["cargo"],
    "go": ["go"],
    "java": ["java"],
    "julia": ["julia"],
    "r": ["Rscript"],
    "swift": ["swiftc", "swift"],
    "sage": ["sage"],
    "gradle": ["gradle"],
    "maven": ["mvn"],
    "latex": ["latexmk", "pdflatex", "xelatex"],
    "git": ["git"],
    "browser": ["Google Chrome", "chromium", "chromium-browser"],
}


def _failure_class(detail: str) -> str:
    text = str(detail or "").lower()
    patterns = (
        ("timeout", ("timed out", "timeout")),
        ("missing_tool", ("not found", "not installed", "no such file")),
        ("dependency", ("module not found", "no module named", "dependency", "package")),
        ("compile", ("compiler error", "compilation", "syntax error", "failed to compile")),
        ("assertion", ("assertion", "residual", "criterion", "expected output")),
        ("sandbox", ("sandbox", "blocked", "permission")),
        ("network", ("network", "connection", "dns", "http")),
    )
    for label, needles in patterns:
        if any(needle in text for needle in needles):
            return label
    return "unknown"


def _tool_from_command(command: str) -> str:
    command = str(command or "").strip()
    first = re.split(r"\s+", command, maxsplit=1)[0] if command else "unknown"
    first = Path(first).name.lower()
    aliases = {
        "python3": "python", "pytest": "python", "clang++": "c++", "g++": "c++",
        "clang": "c", "gcc": "c", "cc": "c", "singular": "singular",
        "lake": "lean", "latexmk": "latex", "pdflatex": "latex", "xelatex": "latex",
        "cargo": "rust", "mvn": "maven",
        "rscript": "r", "swiftc": "swift",
    }
    return aliases.get(first, first or "unknown")


class Toolsmith:
    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace).resolve() if workspace else None
        self.path = Path.home() / ".local" / "share" / "spiral" / "toolsmith.json"
        try:
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.state = {}
        if not isinstance(self.state, dict):
            self.state = {}
        self.state.setdefault("schema_version", 1)
        self.state.setdefault("tools", {})
        self.state.setdefault("recipes", [])
        self.state.setdefault("capabilities", {})

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.state, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def scan(self, *, force: bool = False) -> dict:
        now = time.time()
        last = float(self.state.get("last_scan_epoch") or 0)
        if not force and now - last < 3600 and self.state.get("capabilities"):
            return self.state["capabilities"]
        capabilities = {}
        for name, candidates in KNOWN_TOOLS.items():
            executable = next((shutil.which(candidate) for candidate in candidates if shutil.which(candidate)), "")
            capabilities[name] = {"available": bool(executable), "path": executable, "version": ""}
            if executable:
                version_args = ["--version"]
                if name == "java":
                    version_args = ["-version"]
                try:
                    result = subprocess.run(
                        [executable, *version_args], capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=3,
                    )
                    line = (result.stdout or result.stderr).splitlines()
                    capabilities[name]["version"] = line[0][:180] if line else ""
                except Exception:
                    pass
        self.state["capabilities"] = capabilities
        self.state["last_scan_epoch"] = now
        self.state["last_scan_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._save()
        return capabilities

    def record(self, *, context: str, command: str, ok: bool, duration: float,
               detail: str = "", metadata=None, recipe=None) -> None:
        tool = _tool_from_command(command)
        row = self.state["tools"].setdefault(tool, {
            "attempts": 0, "successes": 0, "failures": 0,
            "ema_seconds": None, "failure_classes": {}, "contexts": {},
        })
        row["attempts"] += 1
        row["successes" if ok else "failures"] += 1
        old_ema = row.get("ema_seconds")
        row["ema_seconds"] = round(
            float(duration) if old_ema is None else 0.75 * float(old_ema) + 0.25 * float(duration),
            4,
        )
        row["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        row["last_ok"] = bool(ok)
        row["last_command"] = str(command)[:500]
        context_row = row["contexts"].setdefault(context, {"attempts": 0, "successes": 0})
        context_row["attempts"] += 1
        context_row["successes"] += int(bool(ok))
        if not ok:
            category = _failure_class(detail)
            row["failure_classes"][category] = int(row["failure_classes"].get(category, 0)) + 1
            row["last_failure"] = str(detail)[:1000]
        if ok and recipe:
            compact = {
                "ts": row["last_run_at"],
                "tool": tool,
                "context": context,
                "summary": str(recipe.get("summary") or "")[:800],
                "method_family": str(recipe.get("method_family") or tool)[:120],
                "command_shape": str(recipe.get("command_shape") or command)[:500],
                "requirements": [str(x) for x in (recipe.get("requirements") or [])][:20],
                "tags": [str(x) for x in (recipe.get("tags") or [])][:20],
                "artifact": str(recipe.get("artifact") or "")[:1000],
            }
            signature = json.dumps(compact, sort_keys=True)
            if not any(json.dumps(existing, sort_keys=True) == signature
                       for existing in self.state["recipes"][-100:]):
                self.state["recipes"].append(compact)
                self.state["recipes"] = self.state["recipes"][-300:]
        self._save()

    def observe_workbench(self, claim: dict, result, duration: float) -> None:
        steps = claim.get("steps") or [claim.get("cmd") or claim.get("command") or "workbench"]
        command = str(steps[-1] if steps else "workbench")
        note = str(claim.get("statement") or claim.get("note") or "research certificate")
        self.record(
            context="research_workbench",
            command=command,
            ok=bool(getattr(result, "ok", False)),
            duration=duration,
            detail=str(getattr(result, "detail", "")),
            metadata={"manifest": getattr(result, "manifest", "")},
            recipe={
                "summary": note,
                "method_family": claim.get("method_family") or _tool_from_command(command),
                "command_shape": command,
                "requirements": claim.get("requirements") or [],
                "tags": claim.get("tags") or [],
                "artifact": getattr(result, "manifest", ""),
            } if getattr(result, "ok", False) else None,
        )

    def rank(self, candidates: list[str], *, context: str = "") -> list[dict]:
        capabilities = self.scan()
        rows = []
        total = sum(int(row.get("attempts", 0)) for row in self.state["tools"].values())
        for candidate in candidates:
            tool = _tool_from_command(candidate)
            stats = self.state["tools"].get(tool, {})
            attempts = int(stats.get("attempts", 0))
            successes = int(stats.get("successes", 0))
            posterior = (successes + 1.0) / (attempts + 2.0)
            exploration = math.sqrt(math.log(total + 2.0) / (attempts + 1.0))
            available = capabilities.get(tool, {}).get("available", bool(shutil.which(tool)))
            context_stats = (stats.get("contexts") or {}).get(context, {})
            context_rate = (
                (context_stats.get("successes", 0) + 1.0) /
                (context_stats.get("attempts", 0) + 2.0)
            )
            score = (0.55 * posterior + 0.20 * context_rate + 0.15 * min(1.0, exploration)
                     + 0.10 * float(bool(available)))
            if not available:
                score *= 0.35
            rows.append({
                "candidate": candidate,
                "tool": tool,
                "available": bool(available),
                "score": round(score, 6),
                "attempts": attempts,
                "success_rate": round(successes / attempts, 4) if attempts else None,
                "ema_seconds": stats.get("ema_seconds"),
            })
        return sorted(rows, key=lambda row: (-row["score"], row["candidate"]))

    def recipe_brief(self, *, limit: int = 6, tags: set[str] | None = None) -> str:
        recipes = list(reversed(self.state.get("recipes", [])))
        if tags:
            recipes = [
                recipe for recipe in recipes
                if tags & {str(tag).lower() for tag in recipe.get("tags", [])}
            ]
        lines = []
        for recipe in recipes[:max(0, limit)]:
            lines.append(
                f"- {recipe.get('method_family')} via `{recipe.get('command_shape')}`: "
                f"{recipe.get('summary')}"
            )
        return "\n".join(lines)

    def capability_brief(self) -> str:
        capabilities = self.scan()
        available = [
            f"{name} ({row.get('version') or Path(row.get('path', '')).name})"
            for name, row in capabilities.items() if row.get("available")
        ]
        unavailable = [name for name, row in capabilities.items() if not row.get("available")]
        lines = ["Available local toolchains: " + ", ".join(available)]
        if unavailable:
            lines.append("Currently unavailable: " + ", ".join(unavailable))
        recipes = self.recipe_brief(limit=4)
        if recipes:
            lines.extend(["Previously successful local recipes:", recipes])
        return "\n".join(lines)[:3000]
