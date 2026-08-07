"""The finish-quality floor has to fail on a scaffold and pass on real work.

Two properties matter more than the individual checks. It must not fire on a
project with no interface — a CLI is not deficient for lacking a focus ring — and
it must not mistake a dark theme written the ordinary way (a `:root` nested inside
`@media (prefers-color-scheme: dark)`) for colours scattered through the rules.

Runs standalone (`python tests/test_quality.py`) or under pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.quality import (  # noqa: E402
    audit_ui, best_contrast, contrast_ratio, loose_colours,
)

POLISHED_CSS = """
:root {
  --bg-surface: #FFFFFF;
  --text-primary: #111827;
  --accent: #0F766E;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-surface: #0F1115;
    --text-primary: #F9FAFB;
    --accent: #2DD4BF;
  }
}
button { background: var(--accent); transition: transform 120ms ease; }
button:hover { filter: brightness(1.1); }
button:active { transform: scale(0.97); }
button:focus-visible { outline: 2px solid var(--accent); }
@media (max-width: 480px) { button { width: 100%; } }
"""

POLISHED_HTML = """
<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="style.css">
<main><button aria-label="Clear">&#x232B;</button>
<ul id="list" aria-live="polite"></ul></main>
<script src="app.js"></script>
"""

POLISHED_JS = """
const rows = [];
function render() {
  const list = document.getElementById('list');
  if (rows.length === 0) { list.innerHTML = '<li>No calculations yet</li>'; return; }
  list.innerHTML = rows.map(r => `<li>${r}</li>`).join('');
}
document.querySelector('button').addEventListener('click', render);
"""


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


def _ids(issues: list[dict]) -> set[str]:
    return {i["id"] for i in issues}


def test_a_polished_page_has_no_gaps(tmp_path):
    root = _write(tmp_path, {"index.html": POLISHED_HTML, "style.css": POLISHED_CSS,
                             "app.js": POLISHED_JS})
    assert audit_ui(root, "web") == [], (
        f"unexpected gaps: {[i['id'] for i in audit_ui(root, 'web')]}")


def test_nested_dark_theme_tokens_are_not_counted_as_loose(tmp_path):
    """A `:root` inside an @media block is how dark mode is written; a matcher
    that stops at the first `}` calls all of it loose."""
    root = _write(tmp_path, {"style.css": POLISHED_CSS})
    inside, outside, _ = loose_colours([root / "style.css"])
    assert inside == 6 and outside == 0


def test_a_scaffold_fails_on_the_things_a_person_notices(tmp_path):
    root = _write(tmp_path, {
        "index.html": '<button onclick="go()">Go</button><ul id="l"></ul>',
        "style.css": "button { background: #cccccc; color: #999999; }",
        "app.js": "function go(){document.getElementById('l').innerHTML = "
                  "items.map(i => i).join('');}",
    })
    found = _ids(audit_ui(root, "web"))
    for expected in ("quality-colour-tokens", "quality-focus-visible",
                     "quality-interaction-states", "quality-labels",
                     "quality-motion", "quality-colour-scheme",
                     "quality-responsive", "quality-empty-state"):
        assert expected in found, f"{expected} should have been reported: {found}"


def test_contrast_is_computed_not_asserted():
    assert round(contrast_ratio("#000000", "#ffffff"), 1) == 21.0
    assert contrast_ratio("#777777", "#808080") < 4.5
    best, fg, bg = best_contrast(
        ":root{--text-primary:#111827;--bg-surface:#ffffff;}")
    assert best > 15 and fg == "--text-primary" and bg == "--bg-surface"


def test_failing_contrast_is_reported_with_the_ratio(tmp_path):
    css = POLISHED_CSS.replace("--text-primary: #111827;", "--text-primary: #9CA3AF;")
    css = css.replace("--text-primary: #F9FAFB;", "--text-primary: #6B7280;")
    root = _write(tmp_path, {"index.html": POLISHED_HTML, "style.css": css,
                             "app.js": POLISHED_JS})
    issues = [i for i in audit_ui(root, "web") if i["id"] == "quality-contrast"]
    assert issues, "grey on white must fail AA"
    assert ":1" in issues[0]["evidence"] and "4.50" in issues[0]["evidence"]


def test_a_cli_is_not_held_to_ui_standards(tmp_path):
    root = _write(tmp_path, {
        "cli.py": "def main():\n    print('hello')\n",
        "pyproject.toml": "[project]\nname='x'\n",
    })
    assert audit_ui(root, "cli") == []
    assert audit_ui(root, "other") == []


def test_issues_carry_a_fix_a_worker_can_act_on(tmp_path):
    root = _write(tmp_path, {"index.html": "<button>Go</button>", "style.css": ""})
    for issue in audit_ui(root, "web"):
        assert issue["severity"] in {"major", "minor"}
        assert len(issue["fix"]) > 30, f"{issue['id']} has no actionable fix"
        assert issue["evidence"], f"{issue['id']} has no evidence"


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
            print(f"  ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
