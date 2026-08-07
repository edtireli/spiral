"""Finish quality as arithmetic, so "make it look finished" can fail a gate.

Everything spiral can verify, it optimises. Everything it cannot, it skips — not
out of laziness but because nothing ever objects. That is the whole explanation for
barebones output: the build gate proves behaviour, the product audit proves an
absence of TODOs, and the vision reviewer is advisory, so "a coherent visual
system" is a sentence a small model grades itself against and passes.

These checks replace that opinion with a computation. Each one is a property of
the source a person would notice immediately and a compiler never would:

    colour tokens        colours declared once as custom properties
    token adherence      rules reference the tokens instead of literals
    contrast             WCAG AA is division, not taste
    focus                a visible focus ring exists
    interaction states   hover and pressed feedback on controls
    motion               something transitions
    empty state          a collection renders something when it is empty
    colour scheme        the OS light/dark preference is honoured
    labels               controls are named for assistive tech
    viewport             a phone gets a layout, not a scaled-down desktop

They are deliberately cheap and syntactic. A passing set does not prove the thing
is beautiful; it proves the floor was not skipped, which is the difference between
a scaffold and a product.

Results are ``product_audit``-shaped issues, so the existing finish loop turns
them into remediation tasks with no new machinery.
"""
from __future__ import annotations

import re
from pathlib import Path

from spiral.ladder import SKIP_DIRS

WEB_SUFFIXES = {".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue",
                ".svelte", ".scss", ".sass", ".less"}
STYLE_SUFFIXES = {".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue",
                  ".svelte", ".jsx", ".tsx"}
UI_KINDS = {"web", "gui", "desktop", "game", "android", "ios"}

_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_FUNC_COLOR = re.compile(r"\b(?:rgb|rgba|hsl|hsla)\s*\(", re.I)
_PROP_DECL = re.compile(r"^\s*(?:--|\$|@)[A-Za-z0-9_-]+\s*:")
_PROP_VALUE = re.compile(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;}\n]+)")
_FOCUS = re.compile(r":focus(?:-visible|-within)?\s*[,{)\s]")
_HOVER = re.compile(r":hover\s*[,{)\s]")
_ACTIVE = re.compile(r":active\s*[,{)\s]|:checked\s*[,{)\s]|\[aria-pressed")
_MOTION = re.compile(r"transition[a-z-]*\s*:|@keyframes|animation[a-z-]*\s*:|"
                     r"animate\(|framer-motion|\.transition\b")
_SCHEME = re.compile(r"prefers-color-scheme|data-theme|\.dark\b|classList\.toggle\(['\"]dark")
_LABEL = re.compile(r"aria-label|aria-labelledby|aria-live|role\s*=|<label[\s>]|"
                     r"contentDescription")
_VIEWPORT = re.compile(r"name=[\"']viewport[\"']")
_MEDIA = re.compile(r"@media|useMediaQuery|matchMedia")
_EMPTY = re.compile(
    r"\.length\s*(?:===?|<|<=)\s*0|\.length\s*\?|!\w+\.length|\blen\(\w+\)\s*==\s*0|"
    r"\bis_?empty\b|no results|nothing (?:yet|here)|empty[-_]?state|"
    r"class=[\"'][^\"']*\bempty\b", re.I)
_INTERACTIVE = re.compile(r"<button|<a\s|<input|<select|<textarea|role=[\"']button|"
                          r"onclick|addEventListener\(\s*[\"']click")


def _issue(identifier: str, severity: str, evidence: str, fix: str,
           files: list[str] | None = None) -> dict:
    return {"id": identifier, "severity": severity, "evidence": evidence,
            "fix": fix, "files": files or []}


def _sources(root: Path, suffixes: set[str]) -> list[Path]:
    out = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if any(part.lower() in {"tests", "test", "fixtures", "examples", "vendor"}
               for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def _read(paths: list[Path]) -> str:
    chunks = []
    for path in paths:
        try:
            chunks.append(path.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _luminance(color: str) -> float | None:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) < 6:
        return None
    try:
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(fg: str, bg: str) -> float | None:
    a, b = _luminance(fg), _luminance(bg)
    if a is None or b is None:
        return None
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def best_contrast(text: str) -> tuple[float, str, str]:
    """Best text-on-background ratio among the declared tokens."""
    props: dict[str, str] = {}
    for name, raw in _PROP_VALUE.findall(text):
        found = _HEX.search(raw)
        if found:
            props.setdefault(name.lower(), found.group(0))
    fg = {k: v for k, v in props.items()
          if any(w in k for w in ("text", "fg", "foreground", "ink", "on-"))}
    bg = {k: v for k, v in props.items()
          if any(w in k for w in ("bg", "background", "surface", "panel", "base",
                                  "card"))}
    best, pair = 0.0, ("", "")
    for fname, fval in fg.items():
        for bname, bval in bg.items():
            got = contrast_ratio(fval, bval)
            if got and got > best:
                best, pair = got, (fname, bname)
    return best, pair[0], pair[1]


def loose_colours(paths: list[Path]) -> tuple[int, int, list[str]]:
    """(on token declarations, loose in rules, examples).

    Counted per line rather than per block: a ``:root`` nested inside an
    ``@media (prefers-color-scheme: dark)`` is the ordinary way to write a dark
    theme, and any block matcher that stops at the first ``}`` scores every one of
    those tokens as loose.
    """
    inside = outside = 0
    examples: list[str] = []
    for path in paths:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            count = len(_HEX.findall(line)) + len(_FUNC_COLOR.findall(line))
            if not count:
                continue
            if _PROP_DECL.match(line):
                inside += count
            else:
                outside += count
                if len(examples) < 3:
                    examples.append(f"{path.name}: {line.strip()[:60]}")
    return inside, outside, examples


def audit_ui(workspace: str | Path, project_kind: str = "other") -> list[dict]:
    """Deterministic finish-quality gaps for anything with an interface.

    Returns an empty list when the project has no UI to hold to this standard —
    a CLI is not deficient for lacking a focus ring.
    """
    root = Path(workspace).resolve()
    web = _sources(root, WEB_SUFFIXES)
    if not web and project_kind not in UI_KINDS:
        return []
    if not web:
        return []                     # android/ios handled by their own pack
    markup = _read([p for p in web if p.suffix.lower() in {".html", ".htm", ".vue",
                                                           ".svelte", ".jsx", ".tsx"}])
    styles = _read([p for p in web if p.suffix.lower() in STYLE_SUFFIXES])
    scripts = _read(web)
    everything = styles + "\n" + scripts

    issues: list[dict] = []
    style_paths = [p for p in web if p.suffix.lower() in STYLE_SUFFIXES]
    inside, outside, examples = loose_colours(style_paths)

    if inside == 0:
        issues.append(_issue(
            "quality-colour-tokens", "major",
            "no colours are declared as reusable custom properties",
            "Define the palette once as CSS custom properties on :root (and an "
            "@media (prefers-color-scheme: dark) override), then reference them "
            "with var() everywhere."))
    elif outside > 3:
        issues.append(_issue(
            "quality-token-adherence", "major",
            f"{outside} colour literal(s) sit in rules instead of referencing a "
            f"token — e.g. {'; '.join(examples)}",
            "Move every colour into a custom property and reference it with "
            "var(); literals scattered through the rules are not a colour system."))

    ratio, fg_name, bg_name = best_contrast(styles)
    if inside and ratio and ratio < 4.5:
        issues.append(_issue(
            "quality-contrast", "major",
            f"best text/background token pair is {fg_name}/{bg_name} at "
            f"{ratio:.2f}:1; WCAG AA requires 4.50:1",
            "Darken the text token or lighten the surface token until the ratio "
            "reaches 4.5:1 for body text."))

    if _INTERACTIVE.search(markup or scripts):
        if not _FOCUS.search(everything):
            issues.append(_issue(
                "quality-focus-visible", "major",
                "no :focus or :focus-visible rule, so keyboard users cannot see "
                "where they are",
                "Give every interactive control a visible focus ring via "
                ":focus-visible; never remove the outline without replacing it."))
        if not (_HOVER.search(everything) and _ACTIVE.search(everything)):
            missing = [name for name, found in (
                ("hover", _HOVER.search(everything)),
                ("pressed/active", _ACTIVE.search(everything))) if not found]
            issues.append(_issue(
                "quality-interaction-states", "minor",
                f"controls have no {' or '.join(missing)} state, so they do not "
                "respond to being used",
                "Add :hover and :active styles to buttons and other controls."))
        if not _LABEL.search(markup):
            issues.append(_issue(
                "quality-labels", "major",
                "no aria-label, role, or <label> anywhere — controls are unnamed "
                "for assistive tech",
                "Label every control: <label for> on inputs, aria-label on "
                "icon-only buttons, aria-live on regions that update."))

    if not _MOTION.search(everything):
        issues.append(_issue(
            "quality-motion", "minor",
            "nothing transitions or animates; state changes snap",
            "Add transitions on interactive state changes and on content "
            "appearing, respecting prefers-reduced-motion."))

    if not _SCHEME.search(everything):
        issues.append(_issue(
            "quality-colour-scheme", "minor",
            "the operating system's light/dark preference is ignored",
            "Add an @media (prefers-color-scheme: dark) block overriding the "
            "colour tokens."))

    if not (_VIEWPORT.search(markup) and _MEDIA.search(everything)):
        missing = []
        if not _VIEWPORT.search(markup):
            missing.append("a viewport meta tag")
        if not _MEDIA.search(everything):
            missing.append("any responsive rule")
        issues.append(_issue(
            "quality-responsive", "major",
            f"the page lacks {' and '.join(missing)}, so a phone gets a "
            "scaled-down desktop layout",
            'Add <meta name="viewport" content="width=device-width, '
            'initial-scale=1"> and @media rules for narrow widths.'))

    renders_collection = re.search(
        r"\.map\(|\.forEach\(|for\s*\(|v-for|\{#each|\.innerHTML\s*\+?=", scripts)
    if renders_collection and not _EMPTY.search(scripts + markup):
        issues.append(_issue(
            "quality-empty-state", "minor",
            "a list is rendered but nothing handles it being empty, so first run "
            "shows a blank area",
            "Branch on zero length and render a short line explaining what will "
            "appear there."))
    return issues


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    kind = args[0] if args else "web"
    root = Path(args[1] if len(args) > 1 else ".").resolve()
    issues = audit_ui(root, kind)
    for issue in issues:
        print(f"{issue['severity']:5} {issue['id']}: {issue['evidence']}")
    print(f"{len(issues)} finish-quality gap(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
