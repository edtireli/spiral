"""Installed-model picker. Inventory reads and config writes never run a model."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import select
import sys
import tempfile

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spiral.theme import CLAY

ROLES = ("worker", "planner", "escalation", "critic", "research_auditor", "janitor")


class PickerError(Exception):
    pass


@dataclass(frozen=True)
class InstalledModel:
    name: str
    size: int = 0
    parameters: str = ""
    quantization: str = ""
    embedding_only: bool = False

    @property
    def detail(self) -> str:
        parts = [self.parameters, self.quantization]
        if self.size:
            parts.append(f"{self.size / 1_000_000_000:.1f} GB")
        if self.embedding_only:
            parts.append("embedding only")
        return " · ".join(part for part in parts if part)


def installed_models(base_url: str, *, transport=None) -> list[InstalledModel]:
    if os.environ.get("SPIRAL_OFFLINE_TESTS") and transport is None:
        raise PickerError("Model inventory is disabled during offline tests.")
    try:
        with httpx.Client(timeout=5.0, transport=transport) as client:
            response = client.get(base_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("invalid inventory")
        models = {}
        for item in payload["models"]:
            if not isinstance(item, dict):
                raise ValueError("invalid inventory entry")
            name = item.get("name") or item.get("model")
            if not isinstance(name, str) or not name or not name.isprintable():
                raise ValueError("invalid model name")
            details = item.get("details") or {}
            if not isinstance(details, dict):
                raise ValueError("invalid model details")
            capabilities = item.get("capabilities") or []
            family = str(details.get("family") or "").lower()
            embedding = ("embedding" in capabilities and "completion" not in capabilities
                         or "embed" in name.lower() or family in {"bert", "nomic-bert"})
            models[name] = InstalledModel(
                name=name, size=max(0, int(item.get("size") or 0)),
                parameters=str(details.get("parameter_size") or ""),
                quantization=str(details.get("quantization_level") or ""),
                embedding_only=embedding,
            )
        return sorted(models.values(), key=lambda m: (m.embedding_only, m.name.casefold()))
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise PickerError(
            "Could not read the installed models from Ollama. "
            "Check that your configured server is reachable, then run spiral models again."
        ) from exc


def save_selection(model: InstalledModel, *, path: Path | None = None) -> None:
    if model.embedding_only:
        raise PickerError("An embedding model cannot be used for builds.")
    path = path or Path.home() / ".config" / "spiral" / "config.json"
    try:
        original = path.read_bytes() if path.exists() else None
        config = json.loads(original) if original is not None else {}
        if not isinstance(config, dict) or not isinstance(config.get("models", {}), dict):
            raise ValueError("invalid config")
    except (OSError, ValueError) as exc:
        raise PickerError(f"Could not read {path}; the configuration was not changed.") from exc
    config.setdefault("models", {}).update({role: model.name for role in ROLES})
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".model-", delete=False) as f:
            temporary = Path(f.name)
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if (path.read_bytes() if path.exists() else None) != original:
            raise PickerError("Configuration changed while selecting; run spiral models again.")
        os.replace(temporary, path)
    except OSError as exc:
        raise PickerError(f"Could not save the model choice to {path}.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class Picker:
    models: list[InstalledModel]
    current: str
    query: str = ""
    index: int = 0
    note: str = ""
    visible: list[InstalledModel] = field(init=False)

    def __post_init__(self):
        self.visible = self.models[:]
        self.index = next((i for i, m in enumerate(self.visible) if m.name == self.current), 0)

    def key(self, key: str, page_size: int = 10) -> tuple[bool, InstalledModel | None]:
        self.note = ""
        if key in ("ESC", "CTRL_C", "EOF"):
            return True, None
        if key == "ENTER":
            if self.visible:
                model = self.visible[self.index]
                if not model.embedding_only:
                    return True, model
                self.note = "Embedding models cannot run builds. Choose a chat model."
        elif key in ("UP", "DOWN", "PAGE_UP", "PAGE_DOWN", "HOME", "END"):
            if self.visible:
                delta = {"UP": -1, "DOWN": 1, "PAGE_UP": -page_size,
                         "PAGE_DOWN": page_size, "HOME": -len(self.visible),
                         "END": len(self.visible)}[key]
                self.index = max(0, min(len(self.visible) - 1, self.index + delta))
        elif key in ("BACKSPACE", "CLEAR") or (len(key) == 1 and key.isprintable()):
            self.query = (self.query[:-1] if key == "BACKSPACE" else "" if key == "CLEAR"
                          else (self.query + key)[:100])
            self.visible = [m for m in self.models if self.query.casefold() in m.name.casefold()]
            self.index = 0
        return False, None

    def render(self, page_size: int, *, listing: bool = False) -> Panel:
        start = max(0, min(self.index - page_size // 2, len(self.visible) - page_size))
        rows = self.visible[start:start + page_size]
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(width=4)
        table.add_column(ratio=1, no_wrap=not listing, overflow="fold" if listing else "ellipsis")
        table.add_column(no_wrap=True, overflow="ellipsis", justify="right")
        for i, model in enumerate(rows, start):
            highlighted = not listing and i == self.index
            style = f"bold black on {CLAY}" if highlighted else "dim" if model.embedding_only else ""
            marker = "›" if highlighted else " "
            name = model.name + ("  ✓ current" if model.name == self.current else "")
            table.add_row(Text(f"{marker}{i + 1:2}", style=style),
                          Text(name, style=style), Text(model.detail, style=style))
        contents = []
        if not listing:
            contents.extend([Text("Filter: " + (self.query or "type to search…"), style=CLAY), Text("")])
        contents.append(table if rows else Text("No matching models. Backspace or Ctrl-U to clear.", style="dim"))
        contents.append(Text(f"\n{len(self.models)} installed · showing {start + 1 if rows else 0}–{start + len(rows)} of {len(self.visible)}", style="dim"))
        if not listing:
            if self.visible:
                contents.append(Text(self.visible[self.index].name, style=CLAY))
            contents.append(Text(self.note or "↑ ↓ move · PgUp/PgDn scroll · Enter save · Esc cancel", style="yellow" if self.note else "dim"))
        return Panel(Group(*contents), title="⠿  installed models", title_align="left", border_style=CLAY)


def _read_key(fd: int) -> str:
    first = os.read(fd, 1)
    if not first:
        return "EOF"
    simple = {b"\r": "ENTER", b"\n": "ENTER", b"\x03": "CTRL_C", b"\x04": "EOF",
              b"\x7f": "BACKSPACE", b"\x08": "BACKSPACE", b"\x15": "CLEAR"}
    if first in simple:
        return simple[first]
    if first == b"\x1b":
        sequence = b""
        while select.select([fd], [], [], 0.06)[0]:
            part = os.read(fd, 1)
            if not part:
                break
            sequence += part
            if len(sequence) >= 2 and 0x40 <= sequence[-1] <= 0x7e:
                break
            if len(sequence) >= 16:
                break
        if not sequence:
            return "ESC"
        return {b"[A": "UP", b"OA": "UP", b"[B": "DOWN", b"OB": "DOWN",
                b"[5~": "PAGE_UP", b"[6~": "PAGE_DOWN", b"[H": "HOME",
                b"OH": "HOME", b"[F": "END", b"OF": "END"}.get(sequence, "IGNORE")
    count = 1 if first[0] < 0x80 else 2 if first[0] < 0xe0 else 3 if first[0] < 0xf0 else 4
    data = first
    for _ in range(count - 1):
        if not select.select([fd], [], [], 0.06)[0]:
            return "IGNORE"
        data += os.read(fd, 1)
    return data.decode("utf-8", errors="replace")


def choose(console: Console, picker: Picker) -> InstalledModel | None:
    if console.is_dumb_terminal:
        console.print(picker.render(len(picker.models), listing=True))
        while True:
            try:
                value = console.input("Model number (Enter to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not value or value == "\x1b":
                return None
            if value.isascii() and value.isdigit() and 1 <= int(value) <= len(picker.models):
                model = picker.models[int(value) - 1]
                if not model.embedding_only:
                    return model
            console.print("Choose the number of an installed chat model.", style="yellow")

    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with Live(console=console, auto_refresh=False, transient=True) as live:
            while True:
                page_size = max(1, min(12, console.size.height - 13))
                live.update(picker.render(page_size), refresh=True)
                done, selected = picker.key(_read_key(fd), page_size)
                if done:
                    return selected
    except KeyboardInterrupt:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def run(console: Console, cfg, *, list_only: bool = False) -> int:
    try:
        models = installed_models(cfg.base_url)
        if not models:
            console.print("No installed models were found on the configured Ollama server.")
            return 0
        picker = Picker(models, cfg.worker.name)
        if list_only or not (sys.stdin.isatty() and console.is_terminal):
            console.print(picker.render(len(models), listing=True))
            if not list_only:
                console.print("Run spiral models in a terminal to choose a model.", style="dim")
            return 0
        console.print("Choose the model for planning, coding and review. Nothing loads until you start work.", style="dim")
        selected = choose(console, picker)
        if selected is None:
            console.print("Model choice unchanged.", style="dim")
            return 0
        save_selection(selected)
        console.print(Text.assemble(("⠿ Saved: ", CLAY), (selected.name, "bold")))
        overrides = ["SPIRAL_" + role.upper() for role in ROLES
                     if os.environ.get("SPIRAL_" + role.upper())]
        if overrides:
            console.print("Shell overrides still take priority. To use the saved choice here:", style="yellow")
            console.print("unset " + " ".join(overrides), markup=False)
        console.print('Next: spiral build "Describe what you want built"', style="dim")
        return 0
    except PickerError as exc:
        console.print(str(exc), style="red", markup=False)
        return 1
