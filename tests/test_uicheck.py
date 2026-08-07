"""Pressing the buttons has to catch what reading the source cannot.

The regression this exists for: a task titled "handle division by zero by showing
an inline error state" stopped clearing the input buffer (correct) while the divide
path still returned null (not correct), so the calculator displayed the word "null".
Every static check passed. The code even cited the requirement in a comment.

A plain sweep of every control in DOM order does NOT find that — the failing
sequence is a digit, then divide, then zero, then equals, which DOM order never
produces. Hence hazard sequences, which fire only when a page has the controls for
them.

Runs standalone (`python tests/test_uicheck.py`) or under pytest.
"""
# Run with pytest. There was once a hand-rolled runner below this point, which
# collected globals() from where it sat mid-file, called each test with no
# arguments, and caught only AssertionError. So it silently skipped every test
# defined after it and every test taking a fixture, then printed "N/N passed".
# A runner that reports a pass count over a subset it chose is the vacuous green
# this suite exists to catch.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spiral.uicheck import FORBIDDEN, find_page, probe  # noqa: E402


def _playwright() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


CALC = """<!doctype html><meta charset="utf-8">
<div id="display">0</div>
<button data-d="7">7</button><button data-d="0">0</button>
<button id="div">/</button><button id="eq">=</button>
<script>
let buf = '', op = null, prev = null;
// String(v) as real formatting code does — assigning a bare null to
// textContent renders empty, so it is the stringification that makes the
// escaped value visible to the user
function show(v) {{ document.getElementById('display').textContent = String(v); }}
for (const b of document.querySelectorAll('[data-d]'))
  b.addEventListener('click', () => {{ buf += b.dataset.d; show(buf); }});
document.getElementById('div').addEventListener('click', () => {{
  prev = parseFloat(buf); op = '/'; buf = ''; }});
document.getElementById('eq').addEventListener('click', () => {{
  const second = parseFloat(buf);
  {equals}
}});
</script>"""

# Both variants guard an empty entry the same way — the probe rightly flagged an
# earlier fixture that rendered NaN for `7 0 / =`, which is a real defect and not
# the one under test. The ONLY difference between these two is the zero branch.
_GUARD = """
  if (!isFinite(second) || !isFinite(prev)) { show('Enter a number'); return; }
"""
# the shipped bug: the guard returns a bare null and the caller renders it
BROKEN_EQUALS = _GUARD + """
  let result;
  if (second === 0) { result = null; } else { result = prev / second; }
  show(result);
"""
FIXED_EQUALS = _GUARD + """
  if (second === 0) { show('Cannot divide by zero'); return; }
  show(prev / second);
"""


def _page(root: Path, equals: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(CALC.format(equals=equals))
    return root


def test_forbidden_tokens_are_the_ones_that_are_never_designed():
    assert set(FORBIDDEN) == {"undefined", "null", "NaN", "[object Object]"}


def test_find_page_prefers_the_conventional_locations(tmp_path):
    assert find_page(tmp_path) is None
    (tmp_path / "app" / "static").mkdir(parents=True)
    (tmp_path / "app" / "static" / "index.html").write_text("<p>hi</p>")
    assert find_page(tmp_path).name == "index.html"
    (tmp_path / "index.html").write_text("<p>root</p>")
    assert find_page(tmp_path) == tmp_path / "index.html"


def test_a_missing_page_is_a_note_not_a_phantom_issue(tmp_path):
    issues, note = probe(tmp_path)
    assert issues == [], "an unrunnable probe must not invent a remediation task"
    assert "no index.html" in note


def test_divide_by_zero_rendering_null_is_caught(tmp_path):
    """The exact regression, in a page small enough to read."""
    if not _playwright():
        return
    root = _page(tmp_path / "broken", BROKEN_EQUALS)
    issues, note = probe(root)
    ids = {i["id"] for i in issues}
    assert "runtime-renders-broken-value" in ids, f"missed it (note: {note})"
    finding = next(i for i in issues if i["id"] == "runtime-renders-broken-value")
    assert "divide by zero" in finding["evidence"]
    assert finding["severity"] == "major"
    assert "message" in finding["fix"].lower()


def test_a_real_error_message_passes(tmp_path):
    if not _playwright():
        return
    root = _page(tmp_path / "fixed", FIXED_EQUALS)
    issues, _note = probe(root)
    assert [i["id"] for i in issues] == [], (
        "showing readable text for the error is the correct behaviour")


def test_a_page_that_throws_is_reported(tmp_path):
    if not _playwright():
        return
    root = tmp_path / "throws"
    root.mkdir()
    (root / "index.html").write_text(
        '<button id="b">Go</button><script>'
        'document.getElementById("b").addEventListener("click", () => '
        '{ throw new Error("boom"); });</script>')
    issues, _note = probe(root)
    assert "runtime-throws" in {i["id"] for i in issues}


def test_a_clean_page_reports_nothing(tmp_path):
    if not _playwright():
        return
    root = tmp_path / "clean"
    root.mkdir()
    (root / "index.html").write_text(
        '<p id="out">ready</p><button>Tap</button><script>'
        'document.querySelector("button").addEventListener("click", () => '
        '{ document.getElementById("out").textContent = "tapped"; });</script>')
    issues, note = probe(root)
    assert issues == [], f"false positive: {[i['evidence'] for i in issues]}"
    assert "pressed" in note




def test_a_page_whose_subject_is_data_is_not_punished(tmp_path):
    """A viewer whose job is to show values may legitimately render the word null.
    A token present before any interaction is content, so it is excluded from the
    rest of the run — and the exclusion is stated, never silent."""
    if not _playwright():
        return
    root = tmp_path / "viewer"
    root.mkdir()
    (root / "index.html").write_text(
        '<pre id="out">null</pre><button>Next</button><script>'
        'document.querySelector("button").addEventListener("click", () => '
        '{ document.getElementById("out").textContent = "null"; });'
        '</script>')
    issues, note = probe(root)
    assert issues == [], f"false positive on legitimate content: {issues}"
    assert "intentional" in note and "null" in note


def test_delimited_matching_does_not_fire_inside_json_text(tmp_path):
    """`{"a": null, "b": 1}` is not a bare rendered null; the matcher is
    deliberately conservative about what counts as a value escaping its guard."""
    if not _playwright():
        return
    root = tmp_path / "json"
    root.mkdir()
    (root / "index.html").write_text('<pre>{"a": null, "b": 1}</pre>')
    issues, _note = probe(root)
    assert issues == []


def test_the_probe_shares_the_visual_review_browser_runtime():
    """Launching chromium bare demands the exact build the installed playwright
    expects. When package and browser drift apart — package 1228, installed 1208 —
    a bare launch silently no-ops while visual review keeps working off its own
    cache. The probe must go through the same resolver, or it reports nothing on
    precisely the machines where it is needed."""
    source = Path(__file__).resolve().parent.parent / "spiral" / "uicheck.py"
    body = source.read_text()
    assert "ensure_playwright_chromium" in body
    assert "PLAYWRIGHT_BROWSERS_PATH" not in body, (
        "take the path from the resolver rather than hard-coding a cache location")
    assert "env=env" in body, "the resolved environment must reach the subprocess"
