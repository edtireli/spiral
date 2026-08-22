"""Click the thing before believing it works.

A remediation task can satisfy a requirement's wording and break its behaviour at
the same time, and nothing downstream notices. The case that produced this module:
a task titled "handle division by zero by showing an inline error state" correctly
stopped clearing the input buffer, but the divide path still returned ``null`` and
that null reached the display. Afterwards the code applied an `.error` class, cited
the requirement in a comment, and passed every gate — while the calculator now
showed the word "null" to the user.

No amount of reading the source catches that. Only pressing the buttons does.

The invariant checked here is deliberately domain-agnostic, because a check tied to
one product's semantics has to be written per product, and one that is not can run
on every web build spiral ever does:

    a user interface must never render "null", "undefined" or "NaN",
    and must not throw while being used.

Those strings are never a designed outcome; they are always a value escaping its
guard. Combined with clicking every visible control, that single rule catches a
broad class of half-finished logic — including the case above, which it finds in
three clicks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("undefined", "null", "NaN", "[object Object]")
PAGE_NAMES = ("index.html", "public/index.html", "src/index.html",
              "app/static/index.html", "static/index.html", "dist/index.html")

DRIVER = '''\
"""Drive a page and report anything a user would call broken."""
import json, pathlib, re, sys
from playwright.sync_api import sync_playwright

page_path = sys.argv[1]
forbidden = json.loads(sys.argv[2])
max_clicks = int(sys.argv[3])

# roles are decided from what a control is LABELLED, so a hazard only runs when the
# page actually has the pieces for it. A page with a divide key, a zero key and an
# equals key has a divide-by-zero path whether or not anyone tested it.
ROLES = {
    "digit": r"^[1-9]$",
    "zero": r"^0$",
    "divide": r"^[/\u00f7]$",
    "operator": r"^[-+\u2212*\u00d7/\u00f7]$",
    "decimal": r"^[.,]$",
    "equals": r"^=$|^enter$|^equals$",
    "clear": r"^(ac|c|clear|reset)$",
}
HAZARDS = [
    ("divide by zero", ["digit", "divide", "zero", "equals"]),
    ("equals with nothing entered", ["equals"]),
    ("two operators in a row", ["digit", "operator", "operator", "equals"]),
    ("two decimal points", ["digit", "decimal", "decimal", "equals"]),
]

console, crashes, findings, intentional = [], [], [], []


def rendered_forbidden(page):
    try:
        text = page.inner_text("body")
    except Exception:
        return None
    padded = f" {text} ".lower()
    for token in forbidden:
        low = token.lower()
        for candidate in (f" {low} ", f"\\n{low}\\n", f">{low}<", f" {low}\\n",
                          f"\\n{low} "):
            if candidate in padded:
                return token
    return next((t for t in forbidden if text.strip().lower() == t.lower()), None)


def controls(page):
    locator = page.locator(
        "button:visible, [role=button]:visible, input[type=button]:visible")
    out = []
    try:
        total = locator.count()
    except Exception:
        return out
    for index in range(min(total, 64)):
        control = locator.nth(index)
        try:
            label = (control.inner_text() or "").strip()
            if not label:
                label = (control.get_attribute("aria-label") or "").strip()
        except Exception:
            label = ""
        out.append((label, control))
    return out


def role_of(label):
    low = label.strip().lower()
    for role, pattern in ROLES.items():
        if re.match(pattern, low, re.I):
            return role
    return ""


unreachable = []
blocked = []
clicked = []


def obstruction(message):
    """The element playwright names in its call log as swallowing the pointer."""
    found = re.search(r"(<[^\\n]*?>) intercepts pointer events", message)
    return found.group(1) if found else ""


def press(control, label, trail):
    try:
        control.click(timeout=2000)
    except Exception as exc:
        # a control that exists, is visible and enabled, and still cannot be
        # clicked is BROKEN, not skippable — a mobile-layout remediation once
        # pushed the whole keypad outside the viewport and the probe read the
        # page as clean because every click "failed quietly"
        message = str(exc)
        if "outside of the viewport" in message:
            unreachable.append(label or "?")
        else:
            # the other half of that same principle, and the half that was
            # missing: playwright refuses a click for two reasons and raises the
            # SAME TimeoutError for both. Recording only the viewport one let a
            # calculator rendering "null" behind an un-torn-down backdrop finish
            # with zero findings — pass 1 skipped every button and pass 2 broke
            # the hazard sequence before divide-by-zero could reach the display.
            over = obstruction(message)
            blocked.append(f"{label or '?'} ({over})" if over else (label or "?"))
        return False
    trail.append(label or "?")
    # trail is rebuilt for each hazard sequence, so it can only ever describe the
    # last one. How much of the page the probe actually pressed — the only signal
    # a human gets when nothing is found — needs a list that spans the run.
    clicked.append(label or "?")
    return True


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.on("console",
            lambda m: console.append(f"{m.type}: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: crashes.append(str(e)))
    url = pathlib.Path(page_path).resolve().as_uri()
    page.goto(url)
    page.wait_for_timeout(300)

    # A token already present before anyone touches anything is CONTENT, not an
    # escaped value — a JSON viewer or a data table may legitimately display the
    # word null. Those tokens are excluded from the rest of the run rather than
    # reported, so the check stays usable on tools whose subject matter is data.
    baseline = rendered_forbidden(page)
    if baseline:
        forbidden = [t for t in forbidden if t.lower() != baseline.lower()]
        intentional.append(baseline)
        while True:
            again = rendered_forbidden(page)
            if not again:
                break
            forbidden = [t for t in forbidden if t.lower() != again.lower()]
            intentional.append(again)

    # pass 1: press everything once — catches state that degrades as it accumulates
    trail = []
    for label, control in controls(page)[:max_clicks]:
        if not press(control, label, trail):
            continue
        page.wait_for_timeout(50)
        hit = rendered_forbidden(page)
        if hit:
            findings.append({"where": " ".join(trail[-6:]), "token": hit})
            break

    # pass 2: hazard sequences, each from a freshly loaded page
    for name, roles in HAZARDS:
        page.goto(url)
        page.wait_for_timeout(200)
        available = {}
        for label, control in controls(page):
            role = role_of(label)
            if role and role not in available:
                available[role] = (label, control)
        if not all(role in available for role in roles):
            continue
        trail = []
        for role in roles:
            label, control = available[role]
            if not press(control, label, trail):
                break
            page.wait_for_timeout(60)
        page.wait_for_timeout(120)
        hit = rendered_forbidden(page)
        if hit:
            findings.append({"where": f"{name} ({' '.join(trail)})", "token": hit})

    print(json.dumps({
        "findings": findings,
        "console_errors": console[:6],
        "crashes": crashes[:6],
        "clicked": clicked,
        "hazards": [name for name, _ in HAZARDS],
        "intentional": intentional,
        "unreachable": unreachable[:12],
        "blocked": blocked[:12],
    }))
    browser.close()
'''


def find_page(root: str | Path) -> Path | None:
    root = Path(root)
    for name in PAGE_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(root.rglob("index.html")):
        if not any(part in {".git", ".spiral", "node_modules"}
                   for part in candidate.relative_to(root).parts):
            return candidate
    return None


def _issue(identifier: str, severity: str, evidence: str, fix: str) -> dict:
    return {"id": identifier, "severity": severity, "evidence": evidence,
            "fix": fix, "files": []}


def probe(root: str | Path, *, max_clicks: int = 24,
          timeout: int = 120) -> tuple[list[dict], str]:
    """Returns (issues, note). ``note`` explains a probe that could not run —
    surfaced in the report rather than turned into an unfixable remediation task."""
    root = Path(root).resolve()
    page = find_page(root)
    if page is None:
        return [], "no index.html to open"
    try:
        import playwright  # noqa: F401
    except ImportError:
        return [], "playwright is not installed, so runtime behaviour was not checked"

    # Use the SAME runtime resolution as visual review. Launching chromium bare
    # demands the exact browser build the installed playwright expects, and when
    # those drift apart (package 1228, installed browser 1208) the probe silently
    # does nothing while visual review keeps working off its own cache. Sharing the
    # resolver means the probe runs wherever visual review runs.
    from spiral.builder_tools import ensure_playwright_chromium

    runtime = ensure_playwright_chromium(root, timeout=timeout)
    if not runtime.get("ok"):
        return [], str(runtime.get("detail") or "no browser runtime available")
    env = {**os.environ, **(runtime.get("environment") or {})}

    driver = root / ".spiral" / "rungs" / "uiprobe.py"
    driver.parent.mkdir(parents=True, exist_ok=True)
    if not driver.is_file() or driver.read_text() != DRIVER:
        driver.write_text(DRIVER)
    try:
        done = subprocess.run(
            [sys.executable, str(driver), str(page), json.dumps(list(FORBIDDEN)),
             str(max_clicks)],
            capture_output=True, text=True, timeout=timeout, cwd=root, env=env)
    except subprocess.TimeoutExpired:
        return [], f"the page did not settle within {timeout}s"
    line = next((row for row in reversed(done.stdout.splitlines())
                 if row.startswith("{")), "")
    if not line:
        detail = (done.stderr or done.stdout).strip().splitlines()[-1:] or [""]
        if "Executable doesn't exist" in done.stderr:
            return [], "chromium is not installed (playwright install chromium)"
        return [], f"the runtime probe produced no verdict: {detail[0][:160]}"

    report = json.loads(line)
    issues: list[dict] = []
    for finding in report.get("findings", []):
        issues.append(_issue(
            "runtime-renders-broken-value", "major",
            f'the interface displayed "{finding["token"]}" to the user after '
            f'{finding["where"]} — a value escaped its guard',
            f'Never render {finding["token"]}: return a real value or an explicit '
            "message from that code path, and render the message. An error state "
            "must show text a person can read."))
    # One fault, one issue. A syntax error fires on every click, and five identical
    # entries inflate the remediation batch into five tasks chasing one bug.
    unreachable = report.get("unreachable") or []
    if unreachable:
        issues.append(_issue(
            "runtime-unreachable-controls", "major",
            f"{len(unreachable)} visible control(s) cannot be clicked — outside "
            f"the viewport (e.g. {', '.join(unreachable[:5])}); the layout has "
            "pushed them off-screen",
            "Fix the layout so every control is inside the viewport at the "
            "default window size; check position/overflow/height rules added "
            "recently."))
    # the same control is retried by pass 1 and again by each hazard sequence, so
    # the raw list counts attempts; one covered control is one fault
    blocked = list(dict.fromkeys(report.get("blocked") or []))
    if blocked:
        issues.append(_issue(
            "runtime-blocked-controls", "major",
            f"{len(blocked)} visible control(s) cannot be clicked — something "
            f"stacked over them intercepts the pointer (e.g. "
            f"{', '.join(blocked[:5])}); an overlay is covering the interface",
            "Remove or dismiss whatever sits over the controls — a modal whose "
            "close handler was never wired, a splash screen never torn down, a "
            "full-bleed backdrop, an oversized sticky bar. If the element must "
            "stay, give it pointer-events: none."))
    for crash in dict.fromkeys(str(c)[:180] for c in report.get("crashes", [])):
        issues.append(_issue(
            "runtime-throws", "major",
            f"the page threw while being used: {crash}",
            "Fix the exception; a control that throws is not implemented. If it is "
            "a SyntaxError the script never ran at all."))
    for error in dict.fromkeys(str(e)[:180] for e in report.get("console_errors", [])):
        issues.append(_issue(
            "runtime-console-error", "minor",
            f"the console logged an error during use: {error}",
            "Resolve the console error — a missing asset, a failed request, or a "
            "caught-and-logged bug."))
    clicked = list(dict.fromkeys(report.get("clicked") or []))
    note = f"pressed {len(clicked)} control(s): {', '.join(clicked[:12])}"
    intentional = report.get("intentional") or []
    if intentional:
        note += (f"; treated {', '.join(sorted(set(intentional)))} as intentional "
                 "content because it was already rendered before any interaction")
    return issues, note


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = Path(args[0] if args else ".").resolve()
    issues, note = probe(root)
    if note:
        print(note)
    for issue in issues:
        print(f"{issue['severity']:5} {issue['id']}: {issue['evidence']}")
    print(f"{len(issues)} runtime issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
