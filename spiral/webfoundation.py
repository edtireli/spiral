"""Deterministic design ground truth for anything with a web page.

An Android build starts with a generated launcher icon and a canonical palette
wired before feature work — the fiddly, always-the-same plumbing a small model
reliably gets wrong — and that is a large part of why the Android output looks like
an app while the web output looks like a scaffold. Web had no equivalent: whether a
token system existed at all depended on the model deciding to invent one, and
whether the text was readable depended on it guessing hex values.

So the harness writes it. ``tokens.css`` is generated from the design tokens with
the text colours **chosen by contrast arithmetic**, not by taste: a small ladder of
candidates is measured against the actual surface colour and the best one that
clears WCAG AA is used, for the light theme and the dark one. The result is that
contrast, focus visibility, a colour-scheme override, and reduced-motion support
are true by construction on the first commit rather than remediated later — and the
model gets a palette to reference instead of a blank page to invent one on.

Nothing here is clever. That is the point: it is the part of design that is
mechanical, done mechanically, so the model's attention goes to the part that is not.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from spiral.appicon import _spiral_path
from spiral.quality import contrast_ratio

TOKENS_FILE = "tokens.css"
FAVICON_FILE = "favicon.svg"

DEFAULT_ACCENT = "#0D9488"
_DEFAULT_RGB = (0x0D, 0x94, 0x88)  # DEFAULT_ACCENT, as channels

# candidates for text on a surface, darkest and lightest first — near-black and
# near-white read better than pure black and white, so they are preferred when
# they clear the threshold
_INK = ("#111827", "#1F2937", "#000000")
_PAPER = ("#F9FAFB", "#E5E7EB", "#FFFFFF")

# The colour names a model reaches for when it ignores the "#RRGGBB" it was asked
# for. Not the full CSS list — anything outside this falls back visibly, which is
# safe; the point is to stop the common cases costing a build.
_NAMED = {
    "black": "#000000", "white": "#FFFFFF", "silver": "#C0C0C0",
    "gray": "#808080", "grey": "#808080", "lightgray": "#D3D3D3",
    "lightgrey": "#D3D3D3", "darkgray": "#A9A9A9", "darkgrey": "#A9A9A9",
    "slategray": "#708090", "slategrey": "#708090", "whitesmoke": "#F5F5F5",
    "red": "#FF0000", "darkred": "#8B0000", "crimson": "#DC143C",
    "firebrick": "#B22222", "maroon": "#800000", "tomato": "#FF6347",
    "coral": "#FF7F50", "salmon": "#FA8072", "orange": "#FFA500",
    "darkorange": "#FF8C00", "gold": "#FFD700", "yellow": "#FFFF00",
    "khaki": "#F0E68C", "olive": "#808000", "green": "#008000",
    "darkgreen": "#006400", "forestgreen": "#228B22", "seagreen": "#2E8B57",
    "limegreen": "#32CD32", "lime": "#00FF00", "springgreen": "#00FF7F",
    "teal": "#008080", "darkcyan": "#008B8B", "turquoise": "#40E0D0",
    "aqua": "#00FFFF", "cyan": "#00FFFF", "skyblue": "#87CEEB",
    "steelblue": "#4682B4", "dodgerblue": "#1E90FF", "royalblue": "#4169E1",
    "blue": "#0000FF", "darkblue": "#00008B", "navy": "#000080",
    "midnightblue": "#191970", "slateblue": "#6A5ACD", "indigo": "#4B0082",
    "purple": "#800080", "darkviolet": "#9400D3", "violet": "#EE82EE",
    "orchid": "#DA70D6", "plum": "#DDA0DD", "magenta": "#FF00FF",
    "fuchsia": "#FF00FF", "hotpink": "#FF69B4", "pink": "#FFC0CB",
    "lavender": "#E6E6FA", "brown": "#A52A2A", "sienna": "#A0522D",
    "chocolate": "#D2691E", "peru": "#CD853F", "tan": "#D2B48C",
    "wheat": "#F5DEB3", "beige": "#F5F5DC", "ivory": "#FFFFF0",
}

_RGB_FUNC = re.compile(r"^rgba?\(([^)]*)\)$", re.I)
_HEX_ONLY = re.compile(r"^[0-9A-Fa-f]+$")


def _clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _channel(part: str) -> int:
    """One rgb() argument, as 0-255. Raises ValueError on anything non-numeric,
    which parse_color turns into a clean None."""
    if part.endswith("%"):
        return _clamp(float(part[:-1]) * 255 / 100)
    return _clamp(float(part))


def parse_color(color: object) -> tuple[int, int, int] | None:
    """The channels behind a CSS colour, or None if it cannot be read.

    Deliberately tolerant on the way in, because the token JSON is model output
    with nothing enforcing it: planner's TOKENS_SCHEMA types the colour fields as
    bare strings, and llm.py asks for response_format json_object, so "#RRGGBB" is
    a request rather than a constraint. "teal", "rgb(13, 148, 136)" and "#0d9" all
    turn up in real runs.

    Equally strict about length: "#12345" used to read channels "12", "34", "5"
    and hand back a colour nobody asked for. A refusal the caller can see beats a
    wrong answer it cannot.
    """
    if not isinstance(color, str):
        return None
    text = _NAMED.get(color.strip().lower(), color.strip())
    match = _RGB_FUNC.match(text)
    if match:
        parts = [p for p in re.split(r"[\s,/]+", match.group(1)) if p]
        if len(parts) < 3:
            return None
        try:
            return (_channel(parts[0]), _channel(parts[1]), _channel(parts[2]))
        except ValueError:
            return None
    value = text[1:] if text.startswith("#") else text
    if not _HEX_ONLY.match(value):
        return None
    if len(value) in (3, 4):                    # #RGB and #RGBA; alpha dropped
        value = "".join(c * 2 for c in value[:3])
    elif len(value) in (6, 8):                  # #RRGGBB and #RRGGBBAA
        value = value[:6]
    else:
        return None
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _report_fallback(color: object, default: str, slot: str) -> None:
    """Say what was substituted. A bad colour used to kill the run from inside
    Conductor._foundation; the cure for that must not be a silent swap, which is
    the same defect with the evidence removed."""
    print(f"spiral: unreadable {slot} {color!r} — falling back to {default}",
          file=sys.stderr)


def _rgb(color: str) -> tuple[int, int, int]:
    """Channels for a colour, falling back to the default accent rather than
    aborting the build. Callers that want the colour normalised for output — and
    so want the fallback to reach the stylesheet too — use normalize_color."""
    rgb = parse_color(color)
    if rgb is None:
        _report_fallback(color, DEFAULT_ACCENT, "colour")
        return _DEFAULT_RGB
    return rgb


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def normalize_color(color: object, default: str = DEFAULT_ACCENT, *,
                    slot: str = "colour") -> str:
    """Canonical #RRGGBB for a model-supplied colour, or ``default`` — out loud.

    Normalising on the way in matters as much as not crashing: "var(--brand)"
    copied straight into --accent leaves the page with no accent at all, and into
    an SVG attribute it leaves a broken favicon.
    """
    rgb = parse_color(color)
    if rgb is None:
        _report_fallback(color, default, slot)
        return default
    return _hex(rgb)


def mix(a: str, b: str, weight: float) -> str:
    """Blend two colours; weight 0 gives a, 1 gives b."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(  # type: ignore[arg-type]
        _clamp(ra[i] + (rb[i] - ra[i]) * weight) for i in range(3)))


def readable_on(surface: str, *, minimum: float = 4.5,
                prefer: float = 7.0) -> str:
    """The best text colour for this surface, decided by measurement.

    Tries near-black then near-white (whichever direction the surface calls for),
    returns the first candidate clearing ``prefer``, else the first clearing
    ``minimum``, else the highest-contrast candidate available. A colour chosen
    this way cannot fail the contrast gate.
    """
    light_surface = (sum(_rgb(surface)) / 3) > 127
    ladder = _INK + _PAPER if light_surface else _PAPER + _INK
    best, best_ratio, acceptable = ladder[0], 0.0, ""
    for candidate in ladder:
        ratio = contrast_ratio(candidate, surface) or 0.0
        if ratio >= prefer:
            return candidate
        # the ladder is ordered by preference, so a candidate that is merely good
        # enough beats a later one that only measures higher — near-black over
        # pure black. Taking the maximum instead is what left `minimum` inert.
        if not acceptable and ratio >= minimum:
            acceptable = candidate
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return acceptable or best


def _secondary_on(surface: str, primary: str) -> str:
    """A muted text colour that still clears AA — stepped toward the surface until
    it would stop being readable, then stepped back."""
    chosen = primary
    for weight in (0.25, 0.35, 0.45, 0.55):
        candidate = mix(primary, surface, weight)
        if (contrast_ratio(candidate, surface) or 0.0) >= 4.5:
            chosen = candidate
        else:
            break
    return chosen


def build_tokens_css(tokens: dict | None = None) -> str:
    """Generate the stylesheet. Deterministic for a given token set."""
    tokens = tokens if isinstance(tokens, dict) else {}
    accent = normalize_color(tokens.get("accent") or DEFAULT_ACCENT,
                             DEFAULT_ACCENT, slot="accent")
    light_bg = normalize_color(tokens.get("background") or "#F2F4F6",
                               "#F2F4F6", slot="background")
    light_surface = normalize_color(tokens.get("surface") or "#FFFFFF",
                                    "#FFFFFF", slot="surface")
    dark_bg = "#0F1115"
    dark_surface = "#181B20"

    light_text = readable_on(light_surface)
    dark_text = readable_on(dark_surface)
    rows_light = {
        "--bg-app": light_bg,
        "--bg-surface": light_surface,
        "--bg-raised": mix(light_surface, light_text, 0.04),
        "--text-primary": light_text,
        "--text-secondary": _secondary_on(light_surface, light_text),
        "--border": mix(light_surface, light_text, 0.14),
        "--accent": accent,
        "--accent-hover": mix(accent, light_text, 0.12),
        "--accent-on": readable_on(accent, prefer=4.5),
        "--focus-ring": accent,
        "--danger": "#DC2626",
        "--success": "#15803D",
    }
    rows_dark = {
        "--bg-app": dark_bg,
        "--bg-surface": dark_surface,
        "--bg-raised": mix(dark_surface, dark_text, 0.06),
        "--text-primary": dark_text,
        "--text-secondary": _secondary_on(dark_surface, dark_text),
        "--border": mix(dark_surface, dark_text, 0.18),
        "--accent": mix(accent, "#FFFFFF", 0.18),
        "--accent-hover": mix(accent, "#FFFFFF", 0.30),
        "--accent-on": readable_on(mix(accent, "#FFFFFF", 0.18), prefer=4.5),
        "--focus-ring": mix(accent, "#FFFFFF", 0.28),
        "--danger": "#F87171",
        "--success": "#4ADE80",
    }
    scale = {
        "--space-1": "4px", "--space-2": "8px", "--space-3": "12px",
        "--space-4": "16px", "--space-5": "24px", "--space-6": "32px",
        "--space-7": "48px",
        "--radius-sm": "6px", "--radius-md": "10px", "--radius-lg": "16px",
        "--radius-pill": "999px",
        "--text-sm": "0.875rem", "--text-base": "1rem", "--text-lg": "1.25rem",
        "--text-xl": "1.75rem", "--text-2xl": "2.5rem",
        "--font-sans": ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
                        "'Helvetica Neue', Arial, sans-serif"),
        "--font-mono": ("ui-monospace, SFMono-Regular, Menlo, Consolas, "
                        "'Liberation Mono', monospace"),
        "--shadow-1": "0 1px 2px rgba(0, 0, 0, 0.08)",
        "--shadow-2": "0 8px 24px rgba(0, 0, 0, 0.12)",
        "--motion-fast": "120ms", "--motion-base": "200ms",
        "--motion-slow": "320ms",
        "--ease": "cubic-bezier(0.2, 0, 0.2, 1)",
        "--tap-target": "44px",
    }

    def block(rows: dict[str, str], indent: str = "  ") -> str:
        return "\n".join(f"{indent}{name}: {value};"
                         for name, value in rows.items())

    return f"""/* Generated by spiral — the project's canonical design tokens.
 *
 * Text colours are not taste: each was measured against its own surface and the
 * best candidate clearing WCAG AA was used, so contrast holds in both themes.
 * Reference these with var(); do not introduce colour literals in rules.
 */
:root {{
  color-scheme: light dark;
{block(rows_light)}

{block(scale)}
}}

@media (prefers-color-scheme: dark) {{
  :root {{
{block(rows_dark, "    ")}
  }}
}}

*, *::before, *::after {{ box-sizing: border-box; }}

body {{
  margin: 0;
  background: var(--bg-app);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}

/* every interactive control is reachable and visibly focused by default */
:focus-visible {{
  outline: 2px solid var(--focus-ring);
  outline-offset: 2px;
}}

button, [role="button"], input, select, textarea {{
  font: inherit;
  color: inherit;
}}

button, [role="button"] {{
  min-height: var(--tap-target);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-raised);
  cursor: pointer;
  transition: background var(--motion-fast) var(--ease),
              transform var(--motion-fast) var(--ease),
              border-color var(--motion-fast) var(--ease);
}}

button:hover, [role="button"]:hover {{ background: var(--bg-surface); }}
button:active, [role="button"]:active {{ transform: scale(0.97); }}
button:disabled, [role="button"][aria-disabled="true"] {{
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }}
}}
"""


def _svg_glyph(glyph: str, accent: str) -> str:
    """The mark, in SVG. Falls back to the spiral for anything unrecognised."""
    stroke = (f'stroke="{accent}" fill="none" stroke-width="7" '
              'stroke-linecap="round" stroke-linejoin="round"')
    if glyph == "eye":
        return (f'<ellipse cx="54" cy="54" rx="34" ry="21" {stroke}/>'
                f'<circle cx="54" cy="54" r="9" fill="{accent}"/>')
    if glyph == "shield":
        return (f'<path d="M54 24 L78 34 V56 C78 70 68 80 54 86 '
                f'C40 80 30 70 30 56 V34 Z" {stroke}/>')
    if glyph == "lock":
        return (f'<rect x="34" y="52" width="40" height="30" rx="6" {stroke}/>'
                f'<path d="M42 52 V44 A12 12 0 0 1 66 44 V52" {stroke}/>')
    if glyph == "bubble":
        return (f'<path d="M28 40 A10 10 0 0 1 38 30 H70 A10 10 0 0 1 80 40 '
                f'V60 A10 10 0 0 1 70 70 H50 L36 82 V70 A10 10 0 0 1 28 60 Z" '
                f'{stroke}/>')
    if glyph == "bolt":
        return f'<path d="M58 22 L36 60 H52 L46 86 L74 46 H56 Z" fill="{accent}"/>'
    return f'<path d="{_spiral_path()}" {stroke}/>'


def build_favicon(accent: str, background: str, glyph: str = "spiral") -> str:
    """A self-contained SVG favicon — the web counterpart of the launcher icon,
    so the page never ships the browser's blank-document default."""
    # normalised here too: these go straight into SVG attributes, where a
    # "var(--brand)" or a stray quote is a broken icon rather than a bad shade
    accent = normalize_color(accent, DEFAULT_ACCENT, slot="icon foreground")
    background = normalize_color(background, "#0F1115", slot="icon background")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108 108" '
        'width="108" height="108" role="img" aria-label="app icon">\n'
        f'  <rect width="108" height="108" rx="24" fill="{background}"/>\n'
        f'  {_svg_glyph(glyph, accent)}\n'
        '</svg>\n'
    )


def write_web_foundation(workspace: str | Path, tokens: dict | None = None,
                         *, subdir: str = "") -> list[str]:
    """Write tokens.css and favicon.svg. Returns the paths written, relative to
    the workspace; existing files are never overwritten."""
    root = Path(workspace).resolve()
    target = root / subdir if subdir else root
    tokens = tokens if isinstance(tokens, dict) else {}
    icon = tokens.get("icon") if isinstance(tokens.get("icon"), dict) else {}
    accent = tokens.get("accent") or DEFAULT_ACCENT
    background = tokens.get("background") or "#0F1115"
    glyph = (icon or {}).get("glyph") or "spiral"

    written: list[str] = []
    for name, body in ((TOKENS_FILE, build_tokens_css(tokens)),
                       (FAVICON_FILE, build_favicon(accent, background, glyph))):
        path = target / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        written.append(str(path.relative_to(root)))
    return written


def tokens_brief(paths: list[str]) -> str:
    """What to tell the planner and every worker about what already exists."""
    if not paths:
        return ""
    css = next((p for p in paths if p.endswith(TOKENS_FILE)), TOKENS_FILE)
    icon = next((p for p in paths if p.endswith(FAVICON_FILE)), FAVICON_FILE)
    return (
        f"CANONICAL DESIGN TOKENS — the palette, type scale, spacing, radii, "
        f"shadows, motion timings, focus ring, reduced-motion handling and default "
        f"button states already exist in {css}, generated with contrast measured "
        f"against each surface. Link it first in every page "
        f'(<link rel="stylesheet" href="{css}">) and reference the tokens with '
        f"var(--name). Do not write colour literals into rules and do not invent a "
        f"second palette. A favicon is already at {icon}; link it with "
        f'<link rel="icon" href="{icon}">.'
    )


__all__ = [
    "build_tokens_css", "build_favicon", "write_web_foundation", "tokens_brief",
    "readable_on", "mix", "parse_color", "normalize_color",
    "TOKENS_FILE", "FAVICON_FILE", "DEFAULT_ACCENT",
]
