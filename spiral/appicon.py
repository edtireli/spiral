"""Deterministic Android launcher icon — harness ground truth, not model output.

A small local model reliably fumbles adaptive-icon XML and density folders, so
the default app ships with the stock robot ("no icon"). Instead the harness
draws a clean vector mark from the design tokens and wires the manifest. One
combined vector drawable referenced as android:icon works on every minSdk (21+),
needs no PNG rasterization, and always builds.

The glyph set is intentionally small and geometric — a design director picks one
that fits the product (an eye for surveillance satire, a lock for a vault app).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

GLYPHS = ("spiral", "eye", "shield", "lock", "bubble", "bolt")


def _spiral_path(cx: float = 54, cy: float = 54, r0: float = 5, r1: float = 30,
                 turns: float = 2.6) -> str:
    """An Archimedean spiral as a polyline — the same mark spiral wears itself."""
    tmax = turns * 2 * math.pi
    pts: list[str] = []
    t = 0.0
    while t <= tmax:
        r = r0 + (r1 - r0) * (t / tmax)
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t)
        pts.append(f"{x:.1f},{y:.1f}")
        t += 0.18
    return "M" + " L".join(pts)


def _glyph_paths(glyph: str, accent: str) -> list[str]:
    """Return <path> element strings for the chosen mark, drawn in the accent.

    Coordinates live in a 108×108 viewport, mark centered near (54,54) inside the
    safe zone. Strokes use round caps so the marks read cleanly at small sizes.
    """
    stroke = (f'<path android:pathData="{{d}}" android:strokeColor="{accent}" '
              f'android:strokeWidth="{{w}}" android:strokeLineCap="round" '
              f'android:strokeLineJoin="round"/>')
    fill = f'<path android:pathData="{{d}}" android:fillColor="{accent}"/>'
    if glyph == "spiral":
        return [stroke.format(d=_spiral_path(), w=7)]
    if glyph == "eye":
        return [
            stroke.format(d="M24,54 Q54,30 84,54 Q54,78 24,54 Z", w=6),
            fill.format(d="M54,43 A11,11 0 1 0 54.1,43 Z"),
        ]
    if glyph == "shield":
        return [
            fill.format(d="M54,25 L81,35 V57 Q81,81 54,91 Q27,81 27,57 V35 Z"),
        ]
    if glyph == "lock":
        return [
            stroke.format(d="M42,52 v-9 a12,12 0 0 1 24,0 v9", w=6),
            fill.format(d="M35,52 h38 a5,5 0 0 1 5,5 v21 a5,5 0 0 1 -5,5 "
                          "h-38 a5,5 0 0 1 -5,-5 v-21 a5,5 0 0 1 5,-5 Z"),
        ]
    if glyph == "bubble":
        return [
            fill.format(d="M30,33 h48 a9,9 0 0 1 9,9 v20 a9,9 0 0 1 -9,9 h-28 "
                          "l-13,11 v-11 h-7 a9,9 0 0 1 -9,-9 v-20 a9,9 0 0 1 9,-9 Z"),
        ]
    if glyph == "bolt":
        return [fill.format(d="M59,25 L37,59 H52 L49,85 L75,49 H59 Z")]
    return [stroke.format(d=_spiral_path(), w=7)]


def _circle(radius: float, color: str, cx: float = 54, cy: float = 54) -> str:
    return (f'<path android:pathData="M{cx - radius},{cy} '
            f'A{radius},{radius} 0 1 0 {cx + radius},{cy} '
            f'A{radius},{radius} 0 1 0 {cx - radius},{cy} Z" '
            f'android:fillColor="{color}"/>')


_VEC_HEAD = (
    '<vector xmlns:android="http://schemas.android.com/apk/res/android"\n'
    '    android:width="108dp" android:height="108dp"\n'
    '    android:viewportWidth="108" android:viewportHeight="108">\n'
)


def icon_vector(accent: str, background: str, glyph: str = "spiral") -> str:
    """A self-contained 108×108 vector: filled disc + centered mark.
    Used as the pre-26 fallback (a plain vector launcher icon, minSdk 21+)."""
    inner = "\n    ".join([_circle(50, background)] + _glyph_paths(glyph, accent))
    return _VEC_HEAD + f"    {inner}\n</vector>\n"


def foreground_vector(accent: str, glyph: str = "spiral") -> str:
    """Adaptive-icon FOREGROUND: the mark alone on a transparent 108 canvas.
    The mark sits inside the center safe zone so launcher masks never clip it."""
    return _VEC_HEAD + "    " + "\n    ".join(_glyph_paths(glyph, accent)) + "\n</vector>\n"


def background_vector(background: str) -> str:
    """Adaptive-icon BACKGROUND: a full-bleed fill the launcher mask shapes."""
    rect = f'<path android:pathData="M0,0h108v108h-108z" android:fillColor="{background}"/>'
    return _VEC_HEAD + f"    {rect}\n</vector>\n"


_ADAPTIVE = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
    '    <background android:drawable="@drawable/ic_launcher_background"/>\n'
    '    <foreground android:drawable="@drawable/ic_launcher_foreground"/>\n'
    '</adaptive-icon>\n'
)


def _norm_hex(c: str, fallback: str) -> str:
    c = (c or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", c) or re.fullmatch(r"#[0-9a-fA-F]{8}", c):
        return c.upper()
    return fallback


def _find_res(ws: Path) -> Path | None:
    """Locate the main res/ dir (module layout varies: app/, mobile/, plain)."""
    manifest = _find_manifest(ws)
    if manifest:
        res = manifest.parent / "res"
        if res.is_dir() or not res.exists():
            return res
    hits = [p for p in ws.rglob("src/main/res") if "build" not in p.parts]
    return hits[0] if hits else None


def _find_manifest(ws: Path) -> Path | None:
    hits = [p for p in ws.rglob("src/main/AndroidManifest.xml") if "build" not in p.parts]
    return hits[0] if hits else None


def _wire_manifest(manifest: Path) -> bool:
    """Ensure <application> carries android:icon → @mipmap/ic_launcher and
    android:roundIcon → @mipmap/ic_launcher_round. Returns True if changed."""
    text = manifest.read_text()
    changed = False

    def set_attr(t: str, attr: str, ref: str) -> str:
        nonlocal changed
        pat = re.compile(rf'({attr}\s*=\s*")[^"]*(")')
        if pat.search(t):
            new = pat.sub(rf'\g<1>{ref}\g<2>', t, count=1)
        else:
            # insert right after the opening <application tag name
            new = re.sub(r'(<application\b)', rf'\g<1>\n        {attr}="{ref}"', t, count=1)
        if new != t:
            changed = True
        return new

    text = set_attr(text, "android:icon", "@mipmap/ic_launcher")
    text = set_attr(text, "android:roundIcon", "@mipmap/ic_launcher_round")
    if changed:
        manifest.write_text(text)
    return changed


TOKEN_COLORS = ("token_accent", "token_background", "token_surface", "token_on_dark")


def write_android_tokens(ws: str | Path, tokens: dict) -> list[str]:
    """Materialize the palette as a canonical color resource the harness owns.

    Written to res/values/spiral_tokens.xml — a NEW file, so existing colors.xml
    and every reference to it are untouched (additive, zero risk). Gives the app
    one authoritative palette instead of the rival, per-task color sets a model
    tends to invent. Returns the repo-relative path if it changed, else []."""
    ws = Path(ws)
    res = _find_res(ws)
    if res is None or not isinstance(tokens, dict):
        return []
    bg = _norm_hex(tokens.get("background"), "#0A0A0A")
    pairs = [
        ("token_accent", _norm_hex(tokens.get("accent"), "#D97757")),
        ("token_background", bg),
        ("token_surface", _norm_hex(tokens.get("surface"), "#141414")),
        ("token_on_dark", _norm_hex(tokens.get("on_dark"), "#F2F2F2")),
    ]
    body = "\n".join(f'    <color name="{n}">{v}</color>' for n, v in pairs)
    content = f'<?xml version="1.0" encoding="utf-8"?>\n<resources>\n{body}\n</resources>\n'
    vals = res / "values"
    vals.mkdir(parents=True, exist_ok=True)
    f = vals / "spiral_tokens.xml"
    if not f.is_file() or f.read_text() != content:
        f.write_text(content)
        return [str(f.relative_to(ws))]
    return []


def write_android_icon(ws: str | Path, accent: str, background: str,
                       glyph: str = "spiral") -> list[str]:
    """Draw an adaptive launcher icon from tokens and wire the manifest.

    Emits (all vector, no raster — builds on any minSdk):
      drawable/ic_launcher_foreground.xml + _background.xml  — adaptive layers
      mipmap-anydpi-v26/ic_launcher{,_round}.xml             — adaptive (API 26+)
      mipmap-anydpi/ic_launcher{,_round}.xml                 — plain-vector fallback (24-25)
    then points android:icon/roundIcon at @mipmap/ic_launcher{,_round}.

    Returns the repo-relative paths written/changed (empty if not an Android app
    or nothing changed). Idempotent: unchanged files are left untouched.
    """
    ws = Path(ws)
    res = _find_res(ws)
    manifest = _find_manifest(ws)
    if res is None or manifest is None:
        return []
    accent = _norm_hex(accent, "#D97757")
    background = _norm_hex(background, "#0A0A0A")
    glyph = glyph if glyph in GLYPHS else "spiral"

    fallback = icon_vector(accent, background, glyph)
    files = {
        "drawable/ic_launcher_foreground.xml": foreground_vector(accent, glyph),
        "drawable/ic_launcher_background.xml": background_vector(background),
        "mipmap-anydpi-v26/ic_launcher.xml": _ADAPTIVE,
        "mipmap-anydpi-v26/ic_launcher_round.xml": _ADAPTIVE,
        "mipmap-anydpi/ic_launcher.xml": fallback,
        "mipmap-anydpi/ic_launcher_round.xml": fallback,
    }
    written: list[str] = []
    for rel, content in files.items():
        f = res / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.is_file() or f.read_text() != content:
            f.write_text(content)
            written.append(str(f.relative_to(ws)))
    if _wire_manifest(manifest):
        written.append(str(manifest.relative_to(ws)))
    return written
