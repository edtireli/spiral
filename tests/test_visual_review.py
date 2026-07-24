from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_discover_static_html_target():
    from spiral.config import Config
    from spiral.visual_review import discover_visual_target

    d = Path(tempfile.mkdtemp())
    (d / "index.html").write_text("<h1>Hello</h1>")
    target = discover_visual_target(d, Config(), goal="make a website", kind="web")

    assert target and target.url.startswith("file://") and target.label == "index.html"


def test_discover_static_html_in_nested_product():
    from spiral.config import Config
    from spiral.visual_review import discover_visual_target

    d = Path(tempfile.mkdtemp())
    product = d / "project"
    product.mkdir()
    (product / "Makefile").write_text("test:\n\t@echo ok\n")
    (product / "index.html").write_text("<h1>Hello</h1>")

    target = discover_visual_target(d, Config(), goal="make an advertisement", kind="web")

    assert target and target.label == "project/index.html"
    assert target.workdir == str(product.resolve())


def test_discover_configured_visual_url_wins():
    from spiral.config import Config
    from spiral.visual_review import discover_visual_target

    d = Path(tempfile.mkdtemp())
    cfg = Config()
    cfg.visual_review_url = "http://127.0.0.1:9999"

    target = discover_visual_target(d, cfg, goal="desktop app", kind="gui")

    assert target and target.url == "http://127.0.0.1:9999"


def test_js_dev_server_wins_over_source_index(monkeypatch):
    from spiral.config import Config
    from spiral import visual_review

    d = Path(tempfile.mkdtemp())
    (d / "index.html").write_text("<script type='module' src='/src.ts'></script>")
    (d / "package.json").write_text(json.dumps({
        "scripts": {"dev": "vite"}, "dependencies": {"vite": "latest"},
    }))
    monkeypatch.setattr(visual_review, "_pm", lambda ws: "npm")

    target = visual_review.discover_visual_target(d, Config(), "build a web app", "web")

    assert target and target.managed and target.command[:3] == ["npm", "run", "dev"]


def test_issues_to_verdicts_are_remediation_tasks():
    from spiral.visual_review import VisualReviewResult, issues_to_verdicts

    result = VisualReviewResult(
        status="revise",
        detail="bad",
        report="/tmp/report.md",
        issues=[{
            "severity": "major",
            "screen": "mobile",
            "evidence": "button text clips",
            "fix": "increase button min-width or wrap label",
            "selector_or_file_hint": "src/App.css",
        }],
    )

    verdicts = issues_to_verdicts(result)

    assert verdicts[0]["id"] == "visual-1"
    assert verdicts[0]["status"] == "missing"
    assert "src/App.css" in verdicts[0]["fix"]["description"]


def test_choose_vision_model_uses_capabilities():
    from spiral.config import Config
    from spiral.visual_review import choose_vision_model

    class Client:
        def post(self, url, json):
            caps = ["completion"]
            if json["model"] == "vision-ok":
                caps.append("vision")
            return SimpleNamespace(status_code=200, json=lambda: {"capabilities": caps})

    cfg = Config()
    cfg.vision_model = "vision-ok"
    ol = SimpleNamespace(base_url="http://ollama.test", _client=Client())

    assert choose_vision_model(cfg, ol) == "vision-ok"


def test_dash_renders_pinned_idea_box():
    from rich.console import Console

    from spiral.dash import Dash

    console = Console(record=True, force_terminal=False, width=100)
    dash = Dash(console=console, plan=None)
    dash.idea("Working angle: inspect mobile clipping before changing CSS.")

    console.print(dash._render())
    rendered = console.export_text()

    assert "thoughts" in rendered and "mobile clipping" in rendered


def test_dash_expands_and_logs_visible_thoughts(tmp_path):
    from rich.console import Console

    from spiral.dash import Dash

    log = tmp_path / "thoughts.jsonl"
    console = Console(record=True, force_terminal=False, width=110)
    dash = Dash(console=console, plan=None, thought_log=log)
    dash.phase("verifying claims")
    dash.idea("Checking whether the proposed theorem survived the symbolic gate.")
    dash.idea("Novelty pass is comparing the surviving claim against corpus anchors.")
    dash.toggle_thoughts()

    console.print(dash._render())
    rendered = console.export_text()

    assert "expanded" in rendered
    assert "symbolic gate" in rendered and "Novelty pass" in rendered
    assert log.is_file()
    assert "corpus anchors" in log.read_text()


def test_watcher_hotkey_callback_is_consumed():
    from spiral.keys import Watcher

    watcher = Watcher()
    seen = []
    watcher.on_key("t", lambda: seen.append("toggle"))

    watcher.feed(b"ta")

    assert seen == ["toggle"]
    assert watcher.ask(timeout=0) == "a"


def test_review_project_visuals_saves_structured_report(monkeypatch):
    from spiral.config import Config
    from spiral import builder_tools, visual_review

    d = Path(tempfile.mkdtemp())
    (d / ".spiral").mkdir()
    (d / ".spiral" / "design.md").write_text("Use clear hierarchy.")
    shot = d / "shot.png"
    shot.write_bytes(b"png bytes are enough for base64 in this mocked test")

    cfg = Config()
    cfg.visual_review_url = "http://127.0.0.1:1234"
    cfg.vision_model = "vision-ok"

    monkeypatch.setattr(visual_review, "choose_vision_model", lambda cfg, ol: "vision-ok")
    monkeypatch.setattr(
        visual_review, "_capture",
        lambda url, out_dir, timeout_ms=30000, executable_path="": [shot],
    )
    monkeypatch.setattr(
        builder_tools, "ensure_playwright_chromium",
        lambda ws, timeout=900: {"ok": True, "environment": {}},
    )

    class FakeOl:
        def chat(self, *args, **kwargs):
            return SimpleNamespace(text=json.dumps({
                "verdict": "revise",
                "summary": "one issue",
                "issues": [{"severity": "major", "screen": "desktop",
                            "evidence": "low contrast", "fix": "increase contrast"}],
                "positives": [],
            }))

    res = visual_review.review_project_visuals(d, cfg, FakeOl(), "make a web UI", "web")

    assert res.status == "revise" and len(res.issues) == 1
    assert Path(res.report).is_file()
    assert json.loads(Path(res.manifest).read_text())["model"] == "vision-ok"
