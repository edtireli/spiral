"""The generated foundation must be readable in both themes, by measurement.

The point of generating tokens.css rather than asking for one is that contrast
stops being a thing to hope for. So the test that matters is not "does it contain
tokens" but "is every text-on-surface pair it declares actually AA, in the light
theme AND the dark one" — including the muted secondary text and text on the accent,
which are the pairs a person gets wrong by eye.

Runs standalone (`python tests/test_webfoundation.py`) or under pytest.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.quality import audit_ui, contrast_ratio  # noqa: E402
from spiral.webfoundation import (  # noqa: E402
    build_favicon, build_tokens_css, mix, readable_on, tokens_brief,
    write_web_foundation,
)

_DECL = re.compile(r"(--[a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{3,8})\s*;")


def _themes(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """(light, dark) token maps. Dark is the @media override applied over light."""
    dark_start = css.index("@media (prefers-color-scheme: dark)")
    dark_end = css.index("*, *::before", dark_start)
    light = dict(_DECL.findall(css[:dark_start]))
    dark = dict(light)
    dark.update(dict(_DECL.findall(css[dark_start:dark_end])))
    return light, dark


def _assert_aa(tokens: dict[str, str], theme: str) -> None:
    pairs = [
        ("--text-primary", "--bg-surface", 4.5),
        ("--text-primary", "--bg-app", 4.5),
        ("--text-secondary", "--bg-surface", 4.5),
        ("--accent-on", "--accent", 4.5),
    ]
    for fg, bg, minimum in pairs:
        assert fg in tokens and bg in tokens, f"{theme}: {fg}/{bg} not declared"
        ratio = contrast_ratio(tokens[fg], tokens[bg])
        assert ratio is not None, f"{theme}: {fg}/{bg} unparseable"
        assert ratio >= minimum, (
            f"{theme}: {fg} ({tokens[fg]}) on {bg} ({tokens[bg]}) is "
            f"{ratio:.2f}:1, below {minimum}")


def test_both_themes_clear_aa_on_every_text_pair():
    light, dark = _themes(build_tokens_css({
        "accent": "#0D9488", "background": "#F2F4F6", "surface": "#FFFFFF"}))
    _assert_aa(light, "light")
    _assert_aa(dark, "dark")


def test_aa_holds_for_awkward_accents():
    """A pale yellow and a near-black accent both have to end up readable — this is
    exactly where picking colours by eye fails."""
    for accent in ("#FFE066", "#111111", "#FF00FF", "#7C3AED", "#00FFFF"):
        light, dark = _themes(build_tokens_css({"accent": accent}))
        _assert_aa(light, f"light/{accent}")
        _assert_aa(dark, f"dark/{accent}")


def test_readable_on_measures_rather_than_guesses():
    assert contrast_ratio(readable_on("#FFFFFF"), "#FFFFFF") >= 7
    assert contrast_ratio(readable_on("#000000"), "#000000") >= 7
    assert contrast_ratio(readable_on("#808080"), "#808080") >= 4.5


def test_readable_on_honours_the_minimum_it_documents():
    """`minimum` was documented and never read, so the docstring's middle clause
    described behaviour the body could not produce. With `prefer` out of reach the
    ladder's own preference order decides, and raising `minimum` has to move the
    answer — on #9A9A9A the preferred near-black is 6.30:1 and pure black 7.46:1."""
    grey = "#9A9A9A"
    assert readable_on(grey, minimum=4.5, prefer=99) == "#111827"
    assert readable_on(grey, minimum=7.0, prefer=99) == "#000000"
    assert readable_on(grey, minimum=99, prefer=99) == "#000000"  # nothing clears it


def test_a_colour_a_model_actually_emits_does_not_abort_the_build():
    """The token JSON is model output with nothing enforcing #RRGGBB: planner's
    TOKENS_SCHEMA types the colour fields as bare strings, and llm.py downgrades the
    schema to response_format json_object, so "teal" and "rgb(13, 148, 136)" arrive
    in real runs. Every one of these used to reach int(value, 16) and kill the run
    from inside Conductor._foundation."""
    for accent in ("teal", "TEAL", "rgb(13, 148, 136)", "rgba(13,148,136,0.6)",
                   "var(--brand)", "hsl(174, 80%, 30%)", "#0d9", "#12345"):
        light, dark = _themes(build_tokens_css({"accent": accent}))
        _assert_aa(light, f"light/{accent}")
        _assert_aa(dark, f"dark/{accent}")


def test_colour_values_are_read_rather_than_guessed():
    from spiral.webfoundation import parse_color

    assert parse_color("teal") == (0, 128, 128)
    assert parse_color("  Teal ") == (0, 128, 128)
    assert parse_color("rgb(13, 148, 136)") == (13, 148, 136)
    assert parse_color("rgb(13 148 136)") == (13, 148, 136)
    assert parse_color("rgba(13,148,136,0.6)") == (13, 148, 136)  # tokens are opaque
    assert parse_color("rgb(50%, 0%, 100%)") == (128, 0, 255)
    assert parse_color("#0D9488") == (13, 148, 136)
    assert parse_color("0D9488") == (13, 148, 136)  # bare hex still parsed
    assert parse_color("#0d9") == (0, 221, 153)
    assert parse_color("#0D9488FF") == (13, 148, 136)
    for junk in ("var(--brand)", "hsl(174, 80%, 30%)", "rgb(a, b, c)", "rgb(1, 2)",
                 "#12345", "#1234567", "#123456789", "#", "", "   ", None, 12345):
        assert parse_color(junk) is None, f"{junk!r} must not parse"


def test_a_wrong_length_hex_is_refused_not_silently_miscoloured():
    """"#12345" never raised — it read channels "12", "34", "5" and produced
    (18, 52, 5), a plausible-looking colour nobody asked for. Silent corruption is
    worse than the crash it was hiding behind."""
    from spiral.webfoundation import DEFAULT_ACCENT, parse_color

    assert parse_color("#12345") is None
    assert mix("#12345", "#000000", 0.0) == DEFAULT_ACCENT


def test_an_unreadable_colour_is_announced_not_swallowed():
    """A fallback the run cannot see is just a quieter version of the same bug."""
    import contextlib
    import io

    noise = io.StringIO()
    with contextlib.redirect_stderr(noise):
        css = build_tokens_css({"accent": "var(--brand)"})
    said = noise.getvalue()
    assert "var(--brand)" in said, "the rejected value has to be named"
    assert "#0D9488" in said, "the substitute has to be named"
    assert "var(--brand)" not in css


def test_the_stylesheet_a_named_colour_produces_is_still_css():
    """Normalising on the way in matters as much as not crashing: a raw
    "var(--brand)" written into --accent would leave the page with no accent at
    all, and the contrast measured against it would be measured against nothing."""
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        css = build_tokens_css({"accent": "teal", "background": "rgb(242,244,246)",
                                "surface": "white"})
    assert "--accent: #008080;" in css
    assert "--bg-app: #F2F4F6;" in css
    assert "--bg-surface: #FFFFFF;" in css


def test_mix_is_a_real_blend():
    assert mix("#000000", "#FFFFFF", 0.5) == "#808080"
    assert mix("#000000", "#FFFFFF", 0.0) == "#000000"
    assert mix("#000000", "#FFFFFF", 1.0) == "#FFFFFF"


def test_generated_css_has_no_loose_colour_literals(tmp_path):
    from spiral.quality import loose_colours

    write_web_foundation(tmp_path, {"accent": "#0D9488"})
    inside, outside, examples = loose_colours([tmp_path / "tokens.css"])
    assert inside > 15, "the palette should be substantial"
    assert outside == 0, f"the harness must not ship loose literals: {examples}"


def test_a_page_that_links_the_foundation_starts_at_the_quality_floor(tmp_path):
    write_web_foundation(tmp_path, {"accent": "#0D9488"})
    (tmp_path / "index.html").write_text(
        '<!doctype html><meta name="viewport" content="width=device-width, '
        'initial-scale=1"><link rel="stylesheet" href="tokens.css">'
        '<link rel="icon" href="favicon.svg">'
        '<main><button aria-label="Go">Go</button><ul id="l"></ul></main>'
        '<script src="app.js"></script>')
    (tmp_path / "app.js").write_text(
        "const rows=[];function r(){const l=document.getElementById('l');"
        "if(rows.length===0){l.innerHTML='<li>Nothing yet</li>';return;}"
        "l.innerHTML=rows.map(x=>`<li>${x}</li>`).join('');}"
        "document.querySelector('button').addEventListener('click',r);")
    (tmp_path / "style.css").write_text(
        "@media (max-width: 480px) { button { width: 100%; } }\n")
    gaps = [i["id"] for i in audit_ui(tmp_path, "web")]
    assert gaps == [], f"the generated foundation should clear the floor: {gaps}"


def test_nothing_is_overwritten(tmp_path):
    (tmp_path / "tokens.css").write_text("/* mine */\n")
    written = write_web_foundation(tmp_path, {})
    assert "tokens.css" not in written
    assert (tmp_path / "tokens.css").read_text() == "/* mine */\n"
    assert "favicon.svg" in written


def test_the_foundation_survives_the_tokens_a_run_actually_writes(tmp_path):
    """Exactly what Conductor._foundation does: the JSON round-trip through
    .spiral/design_tokens.json, straight into write_web_foundation. A named accent
    used to raise ValueError out of here, and _foundation re-raises after rolling
    back — the whole build died on a colour."""
    import contextlib
    import io
    import json

    tokens = json.loads(json.dumps({
        "accent": "teal", "background": "rgb(15, 17, 21)", "surface": "white",
        "icon": {"glyph": "eye", "background": "var(--surface)"}}))
    with contextlib.redirect_stderr(io.StringIO()):
        written = write_web_foundation(tmp_path, tokens)
    assert sorted(written) == ["favicon.svg", "tokens.css"]
    svg = (tmp_path / "favicon.svg").read_text()
    assert 'fill="#0F1115"' in svg and 'stroke="#008080"' in svg
    assert "teal" not in svg and "var(--" not in svg


def test_favicon_is_self_contained_svg():
    svg = build_favicon("#0D9488", "#0F1115", "eye")
    assert svg.startswith("<svg") and "</svg>" in svg
    assert "http://www.w3.org/2000/svg" in svg
    assert "<image" not in svg and "url(" not in svg
    for glyph in ("spiral", "eye", "shield", "lock", "bubble", "bolt", "nonsense"):
        assert build_favicon("#fff", "#000", glyph).count("<svg") == 1


def test_brief_tells_workers_what_already_exists():
    brief = tokens_brief(["tokens.css", "favicon.svg"])
    assert "tokens.css" in brief and "var(--" in brief
    assert "do not invent a second palette" in brief.lower()


if __name__ == "__main__":
    import tempfile

    failures = 0
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    total, ran = len(cases), 0
    for name, fn in cases:
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            ran += 1
            print(f"  ok   {name}")
        except AssertionError as exc:
            ran += 1
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:                    # an error is a failure, not a skip
            ran += 1
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    # report the COUNT, not just the failures: "0 failure(s)" over a subset this
    # runner quietly declined to call reads exactly like a clean suite.
    print(f"\n{ran}/{total} ran · {failures} failure(s)")
    sys.exit(1 if failures or ran != total else 0)
