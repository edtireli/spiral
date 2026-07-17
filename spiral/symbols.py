"""Static symbol index — a cheap semantic map so the worker reads what exists
instead of guessing.

Every recurring failure class (referencing a view id through the wrong
viewBinding, inventing a data type that rivals an existing one, calling a member
that isn't there) is the model hallucinating a symbol. The gradle gate reports
*that* something is wrong, slowly; this reports *what actually exists*, up front.

Regex extraction, no language server, offline. Android-aware: layout ids are
mapped to their binding class, because an id in scan_overlay.xml lives on
ScanOverlayBinding — not ActivityMainBinding — which is the single most common
mistake a local model makes on Android.
"""
from __future__ import annotations

import re
from pathlib import Path

from spiral import tools

KT = {".kt"}
LAYOUT_DIRS = ("/res/layout/",)


def _camel(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _binding_name(filename: str) -> str:
    base = filename[:-4] if filename.endswith(".xml") else filename
    return "".join(p[:1].upper() + p[1:] for p in base.split("_")) + "Binding"


def _kotlin_symbols(text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\bdata class (\w+)\s*\((.*?)\)", text, re.S):
        fields = re.findall(r"\b(?:val|var)\s+(\w+)\s*:", m.group(2))
        out.append(f"data class {m.group(1)}({', '.join(fields)})")
    data_names = {re.match(r"data class (\w+)", s).group(1) for s in out}
    for name in re.findall(r"\bobject (\w+)", text):
        out.append(f"object {name}")
    for name in re.findall(r"\b(?:class|interface) (\w+)", text):
        if name not in data_names:
            out.append(f"class {name}")
    funs = re.findall(r"\bfun (\w+)\s*\(", text)
    if funs:
        seen = list(dict.fromkeys(funs))
        out.append("fun " + ", ".join(seen[:20]))
    return out


def _layout_ids(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'android:id="@\+id/(\w+)"', text)))


def build_symbol_index(root: str | Path, cap: int = 4500) -> str:
    root = Path(root)
    kt_lines: list[str] = []
    layout_lines: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in tools._SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        if p.suffix in KT:
            syms = _kotlin_symbols(text)
            if syms:
                kt_lines.append(f"  {p.name}: " + " · ".join(syms))
        elif p.suffix == ".xml" and any(d in str(rel) for d in LAYOUT_DIRS):
            ids = _layout_ids(text)
            if ids:
                binding = _binding_name(p.name)
                camel = [f"{_camel(i)}" for i in ids]
                layout_lines.append(f"  {p.name} → {binding}: " + ", ".join(camel))

    if not kt_lines and not layout_lines:
        return ""
    parts = ["REPO SYMBOLS — use these EXACT names; do not invent types, members, or ids."]
    if kt_lines:
        parts.append("Kotlin (by file):")
        parts += kt_lines
    if layout_lines:
        parts.append("Android layout ids belong to that layout's OWN *Binding class "
                     "(an id in one layout is NOT reachable from another layout's binding):")
        parts += layout_lines
    text = "\n".join(parts)
    return text[:cap] + ("\n…(truncated)" if len(text) > cap else "")
