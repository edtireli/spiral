"""Score what spiral actually built, not what it claims.

A probe is a goal plus a rubric of machine-checkable facts. Nothing here asks a
model whether the work is good; every check is a shell exit code, an arithmetic
computation over the source, or a real click in a real browser. That is the only
way "is it up to scratch" becomes a number that can be driven down instead of an
opinion that can be argued with.

    python bench/run.py --list
    python bench/run.py web-calculator                # build, then score
    python bench/run.py web-calculator --score ~/dir  # score an existing tree

A check that cannot run (no browser driver, say) is reported as EXCLUDED and
removed from the denominator, and the reduced coverage is printed. A silent skip
would read as a pass.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROBES = ROOT / "probes"

_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_FUNC_COLOR = re.compile(r"\b(?:rgb|rgba|hsl|hsla)\s*\(", re.I)
_CUSTOM_PROP = re.compile(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;}\n]+)")
_CUSTOM_PROP_DECL = re.compile(r"^\s*--[A-Za-z0-9_-]+\s*:")


@dataclass
class Result:
    check: str
    weight: int
    why: str
    status: str          # "pass" | "fail" | "excluded"
    detail: str = ""


@dataclass
class Scorecard:
    probe: str
    target: Path
    results: list[Result] = field(default_factory=list)

    @property
    def scored(self) -> list[Result]:
        return [r for r in self.results if r.status != "excluded"]

    @property
    def earned(self) -> int:
        return sum(r.weight for r in self.scored if r.status == "pass")

    @property
    def possible(self) -> int:
        return sum(r.weight for r in self.scored)

    @property
    def percent(self) -> float:
        return 100.0 * self.earned / self.possible if self.possible else 0.0


# ---- source-level computations ------------------------------------------------

def _web_sources(root: Path) -> list[Path]:
    out = []
    for suffix in ("*.html", "*.css", "*.js"):
        for path in root.rglob(suffix):
            parts = path.relative_to(root).parts
            if any(p in {".git", ".spiral", "node_modules", "dist", "build"}
                   for p in parts):
                continue
            out.append(path)
    return out


def _luminance(hex_color: str) -> float | None:
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) not in (6, 8):
        return None
    try:
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _ratio(a: str, b: str) -> float | None:
    la, lb = _luminance(a), _luminance(b)
    if la is None or lb is None:
        return None
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def check_contrast(root: Path) -> tuple[bool, str]:
    """WCAG AA is arithmetic over the declared tokens, not a matter of taste."""
    props: dict[str, str] = {}
    for path in _web_sources(root):
        for name, raw in _CUSTOM_PROP.findall(path.read_text(errors="replace")):
            found = _HEX.search(raw)
            if found:
                props.setdefault(name.lower(), found.group(0))
    if not props:
        return False, "no colour tokens declared, so nothing to check"
    fg = {k: v for k, v in props.items()
          if any(w in k for w in ("text", "fg", "foreground", "ink", "on-"))}
    bg = {k: v for k, v in props.items()
          if any(w in k for w in ("bg", "background", "surface", "panel", "base"))}
    if not fg or not bg:
        return False, (
            f"could not identify a text/background token pair among {sorted(props)[:6]}")
    best, pair = 0.0, ("", "")
    for fname, fval in fg.items():
        for bname, bval in bg.items():
            got = _ratio(fval, bval)
            if got and got > best:
                best, pair = got, (fname, bname)
    ok = best >= 4.5
    return ok, f"best pair {pair[0]}/{pair[1]} = {best:.2f}:1 (AA needs 4.50)"


def check_token_adherence(root: Path) -> tuple[bool, str]:
    """Colours belong on custom-property declarations; literals scattered through
    the rules are how a 'coherent colour system' turns out to be a claim.

    Decided per line, not per block: a `:root` nested inside an
    `@media (prefers-color-scheme: dark)` is the normal way to write a dark theme,
    and a block regex that stops at the first `}` miscounts every one of those
    tokens as loose. Asking "is this literal on a `--name:` declaration?" is both
    simpler and correct under nesting.
    """
    inside = outside = 0
    loose_examples: list[str] = []
    for path in _web_sources(root):
        if path.suffix not in {".css", ".html"}:
            continue
        for line in path.read_text(errors="replace").splitlines():
            count = len(_HEX.findall(line)) + len(_FUNC_COLOR.findall(line))
            if not count:
                continue
            if _CUSTOM_PROP_DECL.match(line):
                inside += count
            else:
                outside += count
                if len(loose_examples) < 3:
                    loose_examples.append(f"{path.name}: {line.strip()[:60]}")
    if inside == 0:
        return False, "no colours declared as custom properties"
    ok = outside <= 3
    detail = (f"{inside} colour literal(s) on token declarations, {outside} loose "
              "in rules (3 allowed for shadows/gradients)")
    if loose_examples and not ok:
        detail += " — e.g. " + "; ".join(loose_examples)
    return ok, detail


PYTHON_CHECKS = {"contrast": check_contrast, "token_adherence": check_token_adherence}


# ---- behaviour in a real browser ---------------------------------------------

BROWSER_DRIVER = '''\
import json, pathlib, sys
from playwright.sync_api import sync_playwright

page_path, spec_json = sys.argv[1], sys.argv[2]
spec = json.loads(spec_json)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(pathlib.Path(page_path).resolve().as_uri())
    page.wait_for_timeout(250)
    clicked = []
    for label in spec["click"]:
        target = None
        for selector in (
            f'button:text-is("{label}")', f'[role=button]:text-is("{label}")',
            f'button:has-text("{label}")', f'[data-key="{label}"]',
            f'[data-value="{label}"]',
        ):
            found = page.locator(selector).first
            try:
                if found.count() and found.is_visible():
                    target = found
                    break
            except Exception:
                continue
        if target is None:
            print(json.dumps({"ok": False,
                              "detail": f"no clickable control labelled {label!r}"
                                        f" (clicked {clicked} first)"}))
            browser.close()
            sys.exit(0)
        target.click()
        clicked.append(label)
        page.wait_for_timeout(80)
    page.wait_for_timeout(200)
    where = "body"
    for selector in spec.get("read_any", []):
        try:
            if page.locator(selector).first.count():
                where = selector
                break
        except Exception:
            continue
    text = page.inner_text(where)
    ok = any(str(want).lower() in text.lower() for want in spec["expect_any"])
    for forbidden in spec.get("reject_any", []):
        if str(forbidden).lower() in page.inner_text("body").lower():
            ok = False
            text += f"  [rendered {forbidden!r} somewhere on the page]"
            break
    print(json.dumps({
        "ok": ok,
        "detail": f'clicked {"".join(clicked)} -> {where} shows {text.strip()[:90]!r}',
    }))
    browser.close()
'''


def _find_page(root: Path) -> Path | None:
    for candidate in ("index.html", "public/index.html", "src/index.html",
                      "app/static/index.html", "static/index.html"):
        path = root / candidate
        if path.is_file():
            return path
    return next(iter(sorted(root.rglob("index.html"))), None)


def run_browser_check(root: Path, spec: dict) -> tuple[str, str]:
    """Returns (status, detail). 'excluded' when there is no driver to run it."""
    page = _find_page(root)
    if page is None:
        return "fail", "no index.html to open"
    try:
        import playwright  # noqa: F401
    except ImportError:
        return "excluded", "playwright is not installed in this interpreter"
    driver = Path(sys.argv[0]).resolve().parent / ".driver.py"
    driver.write_text(BROWSER_DRIVER)
    try:
        proc = subprocess.run(
            [sys.executable, str(driver), str(page), json.dumps(spec)],
            capture_output=True, text=True, timeout=120, cwd=root)
    except subprocess.TimeoutExpired:
        return "fail", "the page did not settle within 120s"
    finally:
        driver.unlink(missing_ok=True)
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), "")
    if not line:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
        if "Executable doesn't exist" in proc.stderr or "playwright install" in proc.stderr:
            return "excluded", "chromium is not installed (playwright install chromium)"
        return "fail", f"driver produced no verdict: {tail[0][:160]}"
    verdict = json.loads(line)
    return ("pass" if verdict["ok"] else "fail"), verdict.get("detail", "")


# ---- the runner --------------------------------------------------------------

def score(probe: dict, target: Path) -> Scorecard:
    card = Scorecard(probe["id"], target)
    for check in probe["checks"]:
        weight, why = int(check.get("weight", 1)), check.get("why", "")
        if "shell" in check:
            done = subprocess.run(check["shell"], shell=True, cwd=target,
                                  capture_output=True, text=True)
            lines = (done.stdout or done.stderr).strip().splitlines()
            card.results.append(Result(
                check["id"], weight, why,
                "pass" if done.returncode == 0 else "fail",
                lines[0][:120] if done.returncode and lines else ""))
        elif "python" in check:
            ok, detail = PYTHON_CHECKS[check["python"]](target)
            card.results.append(Result(
                check["id"], weight, why, "pass" if ok else "fail", detail))
        elif "browser" in check:
            status, detail = run_browser_check(target, check["browser"])
            card.results.append(Result(check["id"], weight, why, status, detail))
        else:
            card.results.append(Result(
                check["id"], weight, why, "excluded", "no runner for this check"))
    return card


def report(card: Scorecard) -> None:
    width = max(len(r.check) for r in card.results)
    glyph = {"pass": "PASS", "fail": "FAIL", "excluded": "----"}
    print(f"\n  {card.probe}  ({card.target})\n")
    for r in card.results:
        print(f"  {glyph[r.status]}  {r.check.ljust(width)}  {str(r.weight).rjust(2)}  "
              f"{r.why}")
        if r.detail:
            print(f"        {' ' * width}      {r.detail[:150]}")
    excluded = [r for r in card.results if r.status == "excluded"]
    print(f"\n  score {card.earned}/{card.possible}  ({card.percent:.0f}%)")
    if excluded:
        print(f"  coverage reduced — {len(excluded)} check(s) could not run: "
              + ", ".join(f"{r.check} ({r.detail})" for r in excluded))
    print()


def build(probe: dict, workdir: Path, python: str) -> float:
    """Run spiral build on a fresh repo seeded with the probe's goal."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "GOAL.md").write_text(probe["goal"] + "\n")
    subprocess.run("git init -q .", shell=True, cwd=workdir, check=True)
    subprocess.run(
        'git -c user.name=bench -c user.email=bench@local add -A && '
        'git -c user.name=bench -c user.email=bench@local commit -qm seed',
        shell=True, cwd=workdir, check=True)
    log = workdir.parent / f"{probe['id']}.log"
    started = time.time()
    budget = probe.get("budget_minutes", 90) * 60
    # A timeout must not lose the run. Whatever was committed before the clock ran
    # out is still evidence, and scoring a partial build is far more useful than a
    # traceback — the first time this fired it discarded a 119-minute build.
    with log.open("w") as handle:
        try:
            subprocess.run(
                [python, "-c", "from spiral.cli import entry; entry()",
                 "build", "--goal-file", "GOAL.md"],
                cwd=workdir, stdout=handle, stderr=subprocess.STDOUT, timeout=budget)
        except subprocess.TimeoutExpired:
            handle.write(f"\n[bench] budget of {budget // 60} min expired; "
                         "scoring whatever was committed\n")
            print(f"  budget of {budget // 60} min expired — scoring the partial build")
    return (time.time() - started) / 60


def score_all(available: list[str], workdir: Path) -> int:
    """Score every probe with a run directory and print one table.

    Probes with no run are listed as such rather than omitted — a missing row
    reads as a pass otherwise.
    """
    rows, missing = [], []
    for name in available:
        target = workdir / name
        if not target.is_dir():
            missing.append(name)
            continue
        probe = json.loads((PROBES / f"{name}.json").read_text())
        card = score(probe, target)
        rows.append(card)
        out = workdir / f"{name}-scorecard.json"
        out.write_text(json.dumps({
            "probe": card.probe, "target": str(card.target),
            "earned": card.earned, "possible": card.possible,
            "percent": round(card.percent, 1),
            "results": [r.__dict__ for r in card.results],
        }, indent=2))
    if not rows:
        print(f"\n  nothing built yet — run a probe first ({', '.join(available)})\n")
        return 2
    width = max(len(card.probe) for card in rows)
    print()
    for card in rows:
        failed = [r.check for r in card.scored if r.status == "fail"]
        print(f"  {card.probe.ljust(width)}  {card.earned:3}/{card.possible:3}  "
              f"{card.percent:5.0f}%   " + (", ".join(failed[:5]) if failed else "all pass"))
    earned = sum(card.earned for card in rows)
    possible = sum(card.possible for card in rows)
    print(f"\n  {'total'.ljust(width)}  {earned:3}/{possible:3}  "
          f"{100.0 * earned / possible:5.0f}%")
    if missing:
        print(f"  not built: {', '.join(missing)}")
    print()
    return 0 if earned == possible else 1


def record_history(card: Scorecard) -> None:
    """Append (spiral sha, probe, score) to bench/history.jsonl.

    "Did spiral get better this month" must be answerable with a plot, not a
    recollection. Timestamp plus the harness's own commit make each row
    attributable; failures are named so a regression says WHICH check fell.
    """
    sha = subprocess.run(
        "git rev-parse --short HEAD", shell=True, cwd=REPO,
        capture_output=True, text=True).stdout.strip() or "uncommitted"
    dirty = bool(subprocess.run(
        "git status --porcelain", shell=True, cwd=REPO,
        capture_output=True, text=True).stdout.strip())
    row = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spiral": sha + ("+dirty" if dirty else ""),
        "probe": card.probe,
        "earned": card.earned, "possible": card.possible,
        "percent": round(card.percent, 1),
        "failed": [r.check for r in card.scored if r.status == "fail"],
        "excluded": [r.check for r in card.results if r.status == "excluded"],
    }
    history = ROOT / "history.jsonl"
    with history.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="score every probe that already has a run directory")
    parser.add_argument("--score", metavar="DIR",
                        help="score an existing tree instead of building")
    parser.add_argument("--workdir", default=str(ROOT / "runs"))
    parser.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    args = parser.parse_args()

    available = sorted(p.stem for p in PROBES.glob("*.json"))
    if args.all:
        return score_all(available, Path(args.workdir).expanduser().resolve())
    if args.list or not args.probe:
        print("probes: " + ", ".join(available) or "none")
        return 0
    if args.probe not in available:
        print(f"unknown probe {args.probe!r}; have {available}")
        return 2
    probe = json.loads((PROBES / f"{args.probe}.json").read_text())

    if args.score:
        target = Path(args.score).expanduser().resolve()
    else:
        target = Path(args.workdir).expanduser().resolve() / args.probe
        if target.exists():
            shutil.rmtree(target)
        minutes = build(probe, target, args.python)
        print(f"  built in {minutes:.0f} min")
    card = score(probe, target)
    report(card)
    out = Path(args.workdir).expanduser().resolve() / f"{args.probe}-scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": card.probe, "target": str(card.target),
        "earned": card.earned, "possible": card.possible,
        "percent": round(card.percent, 1),
        "results": [r.__dict__ for r in card.results],
    }, indent=2))
    record_history(card)
    print(f"  scorecard → {out}\n")
    return 0 if card.percent == 100 else 1


if __name__ == "__main__":
    sys.exit(main())
