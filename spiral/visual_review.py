"""Vision-model UI review for ``spiral build``.

This module is deliberately split into three pieces:

1. find a visual target (configured URL, static HTML, or a known JS dev server),
2. capture screenshots with Playwright,
3. ask an Ollama vision model for a structured defect report.

The reviewer is advisory but actionable: serious issues are converted into normal
validation-style remediation tasks, so the existing gated build loop fixes them.
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VisualTarget:
    url: str
    label: str = "ui"
    command: list[str] | None = None
    managed: bool = False
    workdir: str = ""


@dataclass
class VisualReviewResult:
    status: str                       # skipped | pass | revise
    detail: str
    model: str = ""
    target: str = ""
    screenshots: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    report: str = ""
    manifest: str = ""
    raw: str = ""
    deterministic_checks: dict = field(default_factory=dict)


def _json_object(text: str) -> dict:
    text = text or ""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return {}
    return {}


def _pm(ws: Path) -> str:
    if (ws / "pnpm-lock.yaml").is_file() and shutil.which("pnpm"):
        return "pnpm"
    if (ws / "yarn.lock").is_file() and shutil.which("yarn"):
        return "yarn"
    if ((ws / "bun.lockb").is_file() or (ws / "bun.lock").is_file()) and shutil.which("bun"):
        return "bun"
    return "npm"


def _package_json(ws: Path) -> dict:
    f = ws / "package.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_js_web(pkg: dict, goal: str) -> bool:
    blob = json.dumps(pkg).lower()
    return any(k in blob for k in (
        "vite", "next", "react", "vue", "svelte", "astro", "solid-js", "angular",
    )) or any(k in goal.lower() for k in ("website", "web app", "frontend", "dashboard", "page"))


def _visible_artifacts(ws: Path, extensions: set[str]) -> list[Path]:
    rows = []
    for path in ws.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        rel = path.relative_to(ws)
        if any(part.startswith(".") or part.lower() in {
                "node_modules", "target", "coverage",
                "tests", "test", "fixtures", "examples",
        } for part in rel.parts):
            continue
        rows.append(path)
    return rows


def _declared_visual_artifacts(
    ws: Path, kind: str, extensions: set[str],
) -> tuple[list[Path], bool]:
    """Prefer the deliverable analyst's exact outputs over arbitrary source assets."""

    manifest = ws / ".spiral" / "artifacts.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return [], False
    rows = []
    authoritative = False
    for deliverable in data.get("deliverables") or []:
        if not isinstance(deliverable, dict) or str(
                deliverable.get("kind") or "") != kind:
            continue
        patterns = list(deliverable.get("output_globs") or [])
        if not patterns and kind in {
                "plot", "image", "video", "audio", "document", "presentation",
                "dataset", "notebook", "3d", "formal-proof"}:
            patterns = ["output/*"]
        authoritative = authoritative or bool(patterns)
        for raw in patterns:
            pattern = str(raw).strip().removeprefix("./")
            if (
                not pattern or pattern.startswith(("/", "~"))
                or ".." in Path(pattern).parts
            ):
                continue
            for path in ws.glob(pattern):
                try:
                    path.resolve().relative_to(ws)
                except (OSError, ValueError):
                    continue
                if (
                    path.is_file() and path.suffix.lower() in extensions
                    and path not in rows
                ):
                    rows.append(path)
    return rows, authoritative


def _visual_artifacts(ws: Path, kind: str, extensions: set[str]) -> list[Path]:
    declared, authoritative = _declared_visual_artifacts(
        ws, kind, extensions)
    return declared if authoritative else _visible_artifacts(ws, extensions)


def _artifact_score(path: Path) -> tuple[int, int, float]:
    name = path.stem.lower()
    intent = sum(
        20 for token in (
            "final", "output", "render", "poster", "advert", "figure", "plot",
            "report", "paper", "presentation", "result",
        ) if token in name
    )
    try:
        size = path.stat().st_size
        modified = path.stat().st_mtime
    except OSError:
        size = 0
        modified = 0.0
    return intent, size, modified


def _write_media_gallery(ws: Path, paths: list[Path], kind: str) -> Path:
    target = ws / ".spiral" / "visual-review" / f"{kind}-target.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for path in paths:
        source = html.escape(path.resolve().as_uri(), quote=True)
        label = html.escape(str(path.relative_to(ws)))
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            media = (
                f'<video src="{source}" controls muted autoplay playsinline '
                'style="max-width:100%;max-height:82vh"></video>'
            )
        else:
            media = (
                f'<img src="{source}" alt="{label}" '
                'style="max-width:100%;height:auto;display:block;margin:auto">'
            )
        body.append(
            f'<figure><figcaption>{label}</figcaption>{media}</figure>')
    target.write_text(
        "<!doctype html><meta charset=utf-8><title>Artifact review</title>"
        "<style>html{background:#111;color:#eee;font:14px system-ui}"
        "body{margin:0;padding:24px;display:grid;gap:32px}"
        "figure{margin:0;min-width:0}figcaption{margin:0 0 10px;color:#bbb}"
        "img,video{background:#fff;box-shadow:0 1px 12px #000}</style>"
        "<body>" + "".join(body) + "</body>",
        encoding="utf-8",
    )
    return target


def _render_pdf_gallery(ws: Path, pdf: Path) -> Path | None:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return None
    rendered = ws / ".spiral" / "visual-review" / "pdf-pages"
    shutil.rmtree(rendered, ignore_errors=True)
    rendered.mkdir(parents=True, exist_ok=True)
    prefix = rendered / "page"
    result = subprocess.run(
        [pdftoppm, "-png", "-r", "120", "-f", "1", "-l", "12",
         str(pdf), str(prefix)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180,
    )
    pages = sorted(rendered.glob("page-*.png"))
    return _write_media_gallery(ws, pages, "document") if result.returncode == 0 and pages else None


def _office_to_pdf(ws: Path, path: Path) -> Path | None:
    office = shutil.which("libreoffice") or shutil.which("soffice")
    if not office:
        return None
    rendered = ws / ".spiral" / "visual-review" / "office"
    rendered.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [office, "--headless", "--convert-to", "pdf", "--outdir",
         str(rendered), str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180,
        env={
            key: value for key, value in os.environ.items()
            if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME"}
        },
    )
    candidate = rendered / f"{path.stem}.pdf"
    return candidate if result.returncode == 0 and candidate.is_file() else None


def _render_3d_preview(ws: Path, source: Path, cfg) -> Path | None:
    """Render a deterministic inspection frame when Blender is available."""

    blender = shutil.which("blender")
    if not blender:
        return None
    rendered = ws / ".spiral" / "visual-review" / "3d"
    rendered.mkdir(parents=True, exist_ok=True)
    output = rendered / "preview.png"
    script = rendered / "render_preview.py"
    script.write_text(
        """import bpy, math, sys
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
source, output = args[0], args[1]
if not source.lower().endswith(".blend"):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if source.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=source)
    elif source.lower().endswith(".obj"):
        (bpy.ops.wm.obj_import if hasattr(bpy.ops.wm, "obj_import")
         else bpy.ops.import_scene.obj)(filepath=source)
    elif source.lower().endswith(".stl"):
        (bpy.ops.wm.stl_import if hasattr(bpy.ops.wm, "stl_import")
         else bpy.ops.import_mesh.stl)(filepath=source)
    elif source.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=source)
    elif source.lower().endswith(".ply"):
        (bpy.ops.wm.ply_import if hasattr(bpy.ops.wm, "ply_import")
         else bpy.ops.import_mesh.ply)(filepath=source)

objects = [o for o in bpy.context.scene.objects
           if o.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}]
if not objects:
    raise RuntimeError("scene contains no renderable objects")
points = [o.matrix_world @ Vector(corner) for o in objects for corner in o.bound_box]
mins = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
maxs = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
center = (mins + maxs) * 0.5
span = max((maxs - mins).length, 0.1)

camera_data = bpy.data.cameras.new("SpiralPreviewCamera")
camera = bpy.data.objects.new("SpiralPreviewCamera", camera_data)
bpy.context.collection.objects.link(camera)
camera.location = center + Vector((span * 0.9, -span * 1.25, span * 0.75))
camera.rotation_euler = ((center - camera.location).to_track_quat("-Z", "Y").to_euler())
camera_data.lens = 52
bpy.context.scene.camera = camera

for name, location, energy, size in (
    ("Key", center + Vector((span, -span, span * 1.6)), 1100, span),
    ("Fill", center + Vector((-span, -span * 0.3, span * 0.7)), 500, span * 1.4),
):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.shape, data.size = energy, "DISK", max(size, 0.5)
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = ((center - location).to_track_quat("-Z", "Y").to_euler())

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 1200, 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = output
scene.world.color = (0.035, 0.04, 0.045)
scene.render.film_transparent = False
bpy.ops.render.render(write_still=True)
""",
        encoding="utf-8",
    )
    from spiral.command_broker import CommandBroker

    loaded = (
        f"{shlex.quote(blender)} --background "
        + (
            f"{shlex.quote(str(source))} "
            if source.suffix.lower() == ".blend" else ""
        )
        + "--disable-autoexec "
        f"--python {shlex.quote(str(script))} -- "
        f"{shlex.quote(str(source))} {shlex.quote(str(output))}"
    )
    result = CommandBroker(ws, cfg).run(
        loaded, timeout=max(300, int(getattr(cfg, "verify_timeout", 900))),
        purpose="3d-preview", allow_network=False,
        require_sandbox=bool(getattr(cfg, "builder_require_sandbox", True)),
    ).result
    return output if result.ok and output.is_file() else None


def _android_package(ws: Path) -> str:
    manifests = [
        path for path in ws.rglob("AndroidManifest.xml")
        if not any(part in {"build", ".gradle"} for part in path.parts)
    ]
    for manifest in manifests:
        try:
            package = str(ET.parse(manifest).getroot().attrib.get("package") or "")
            if package:
                return package
        except Exception:
            continue
    for build in [*ws.rglob("build.gradle.kts"), *ws.rglob("build.gradle")]:
        if any(part in {"build", ".gradle"} for part in build.parts):
            continue
        try:
            match = re.search(
                r"\b(?:namespace|applicationId)\s*(?:=|\s)\s*[\"']([^\"']+)",
                build.read_text(errors="replace"),
            )
            if match:
                return match.group(1)
        except OSError:
            continue
    return ""


def _android_preview(ws: Path, cfg) -> Path | None:
    """Capture the launcher screen from an Android emulator, never a physical device."""

    adb = (
        shutil.which("adb")
        or next((
            str(path) for path in (
                Path.home() / "Library/Android/sdk/platform-tools/adb",
                Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools/adb",
            ) if path.is_file()
        ), "")
    )
    if not adb:
        return None

    def run(*args: str, timeout: int = 90, text: bool = True):
        return subprocess.run(
            [adb, *args], capture_output=True, text=text,
            stdin=subprocess.DEVNULL, timeout=timeout,
            env={
                key: value for key, value in os.environ.items()
                if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "ANDROID_HOME"}
            },
        )

    try:
        devices = run("devices").stdout.splitlines()[1:]
    except Exception:
        return None
    serial = next((
        line.split()[0] for line in devices
        if line.startswith("emulator-") and "\tdevice" in line
    ), "")
    emulator_process = None
    if not serial:
        emulator = (
            shutil.which("emulator")
            or str(Path.home() / "Library/Android/sdk/emulator/emulator")
        )
        if not Path(emulator).is_file():
            return None
        listed = subprocess.run(
            [emulator, "-list-avds"], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=30,
        )
        avd = next((line.strip() for line in listed.stdout.splitlines()
                    if line.strip()), "")
        if not avd:
            return None
        emulator_process = subprocess.Popen(
            [emulator, "-avd", avd, "-no-snapshot-save", "-no-audio",
             "-no-boot-anim", "-gpu", "swiftshader_indirect", "-netfast"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                devices = run("devices", timeout=10).stdout.splitlines()[1:]
                serial = next((
                    line.split()[0] for line in devices
                    if line.startswith("emulator-") and "\tdevice" in line
                ), "")
                if serial and run(
                        "-s", serial, "shell", "getprop", "sys.boot_completed",
                        timeout=10).stdout.strip() == "1":
                    break
            except Exception:
                pass
            time.sleep(2)
        if not serial:
            if emulator_process.poll() is None:
                emulator_process.terminate()
            return None

    apks = [
        path for path in ws.rglob("*.apk")
        if "/build/outputs/apk/" in str(path).replace("\\", "/")
        and "debug" in path.name.lower()
    ]
    package = _android_package(ws)
    if not apks or not package:
        if emulator_process is not None and emulator_process.poll() is None:
            run("-s", serial, "emu", "kill", timeout=15)
        return None
    apk = max(apks, key=lambda path: path.stat().st_mtime)
    target = ws / ".spiral" / "visual-review" / "android-launcher.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    installed = False
    wifi_was_on = run(
        "-s", serial, "shell", "settings", "get", "global", "wifi_on",
        timeout=15).stdout.strip() == "1"
    data_was_on = run(
        "-s", serial, "shell", "settings", "get", "global", "mobile_data",
        timeout=15).stdout.strip() == "1"
    try:
        run("-s", serial, "shell", "svc", "wifi", "disable", timeout=15)
        run("-s", serial, "shell", "svc", "data", "disable", timeout=15)
        install = run("-s", serial, "install", "-r", "-t", str(apk), timeout=180)
        if install.returncode != 0:
            return None
        installed = True
        run("-s", serial, "shell", "am", "force-stop", package, timeout=20)
        launch = run(
            "-s", serial, "shell", "monkey", "-p", package,
            "-c", "android.intent.category.LAUNCHER", "1", timeout=30)
        if launch.returncode != 0:
            return None
        time.sleep(2)
        shot = run(
            "-s", serial, "exec-out", "screencap", "-p",
            timeout=30, text=False)
        if shot.returncode != 0 or not shot.stdout:
            return None
        target.write_bytes(shot.stdout)
        return target
    finally:
        if installed:
            run("-s", serial, "uninstall", package, timeout=60)
        if emulator_process is not None and emulator_process.poll() is None:
            run("-s", serial, "emu", "kill", timeout=15)
        elif emulator_process is None:
            if wifi_was_on:
                run("-s", serial, "shell", "svc", "wifi", "enable", timeout=15)
            if data_was_on:
                run("-s", serial, "shell", "svc", "data", "enable", timeout=15)


def _sandboxed_server_command(ws: Path, command: list[str]) -> tuple[list[str], bool]:
    """Allow a local preview server to listen while denying outbound transmission."""

    sandbox = shutil.which("sandbox-exec")
    if sandbox and sys.platform == "darwin":
        def quoted(value: str | Path) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"')

        temp = os.environ.get("TMPDIR") or "/tmp"
        profile = " ".join([
            "(version 1)",
            "(allow default)",
            "(deny network-outbound)",
            "(deny file-write*)",
            f'(allow file-write* (subpath "{quoted(ws)}"))',
            f'(allow file-write* (subpath "{quoted(temp)}"))',
            '(allow file-write* (subpath "/tmp"))',
            '(allow file-write* (subpath "/private/tmp"))',
            '(allow file-write* (literal "/dev/null"))',
            f'(deny file-write* (subpath "{quoted(ws / ".git")}"))',
            f'(deny file-write* (subpath "{quoted(ws / ".spiral" / "tools")}"))',
        ])
        return [sandbox, "-p", profile, *command], True
    return command, False


def discover_visual_target(ws: str | Path, cfg, goal: str = "", kind: str = "") -> VisualTarget | None:
    """Find something Playwright can open without inventing project-specific commands."""
    ws = Path(ws).resolve()
    from spiral.builder_tools import discover_project_roots

    project_roots = discover_project_roots(ws)
    configured = (
        os.environ.get("SPIRAL_VISUAL_URL")
        or getattr(cfg, "visual_review_url", "")
        or ((ws / ".spiral" / "visual_url").read_text().strip()
            if (ws / ".spiral" / "visual_url").is_file() else "")
    )
    if configured:
        return VisualTarget(configured, "configured URL")

    if kind == "android":
        preview = _android_preview(ws, cfg)
        if preview:
            gallery = _write_media_gallery(ws, [preview], "android")
            return VisualTarget(
                gallery.resolve().as_uri(),
                "Android emulator launcher screen", workdir=str(ws))

    # Source index.html files in Vite and similar projects are module entry points,
    # not standalone pages. Start the actual app before considering static files.
    web_root = next((
        root for root in project_roots
        if _package_json(root) and _is_js_web(_package_json(root), goal)
    ), None)
    pkg = _package_json(web_root) if web_root else {}
    scripts = pkg.get("scripts", {}) if isinstance(pkg.get("scripts"), dict) else {}
    if web_root and "dev" in scripts:
        project_root = web_root
        pm = _pm(project_root)
        blob = json.dumps(pkg).lower()
        if "next" in blob:
            cmd = [pm, "run", "dev", "--", "-H", "127.0.0.1"]
            label = "Next dev server"
        else:
            cmd = [pm, "run", "dev", "--", "--host", "127.0.0.1"]
            label = "JS dev server"
        return VisualTarget(
            "", label, command=cmd, managed=True, workdir=str(project_root))

    for project_root in project_roots:
        for rel in ("dist/index.html", "build/index.html", "index.html", "public/index.html"):
            f = project_root / rel
            if f.is_file():
                return VisualTarget(
                    f.resolve().as_uri(), str(f.relative_to(ws)),
                    workdir=str(project_root))

    if kind in {"visualization", "plot", "image"}:
        candidates = _visual_artifacts(
            ws, kind, {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"})
        if candidates:
            chosen = sorted(candidates, key=_artifact_score, reverse=True)[:6]
            gallery = _write_media_gallery(ws, chosen, kind)
            return VisualTarget(
                gallery.resolve().as_uri(),
                ", ".join(str(path.relative_to(ws)) for path in chosen),
                workdir=str(ws))

    if kind == "video":
        candidates = _visual_artifacts(
            ws, kind, {".mp4", ".mov", ".mkv", ".webm"})
        if candidates:
            chosen = sorted(candidates, key=_artifact_score, reverse=True)[:3]
            gallery = _write_media_gallery(ws, chosen, kind)
            return VisualTarget(
                gallery.resolve().as_uri(),
                ", ".join(str(path.relative_to(ws)) for path in chosen),
                workdir=str(ws))

    if kind in {"document", "presentation"}:
        pdfs = _visual_artifacts(ws, kind, {".pdf"})
        source = max(pdfs, key=_artifact_score) if pdfs else None
        if source is None:
            offices = _visual_artifacts(
                ws, kind, {".docx", ".odt", ".pptx", ".odp"})
            if offices:
                source = _office_to_pdf(
                    ws, max(offices, key=_artifact_score))
        if source:
            gallery = _render_pdf_gallery(ws, source)
            return VisualTarget(
                (gallery or source).resolve().as_uri(),
                str(source.relative_to(ws)) if source.is_relative_to(ws) else source.name,
                workdir=str(ws))

    if kind == "notebook":
        notebooks = _visual_artifacts(ws, kind, {".ipynb"})
        jupyter = shutil.which("jupyter")
        if notebooks and jupyter:
            notebook = max(notebooks, key=_artifact_score)
            out = ws / ".spiral" / "visual-review" / "notebook"
            out.mkdir(parents=True, exist_ok=True)
            converted = subprocess.run(
                [jupyter, "nbconvert", "--to", "html", "--output-dir", str(out),
                 "--output", "notebook.html", str(notebook)],
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=180,
            )
            html_target = out / "notebook.html"
            if converted.returncode == 0 and html_target.is_file():
                return VisualTarget(
                    html_target.resolve().as_uri(),
                    str(notebook.relative_to(ws)), workdir=str(ws))

    if kind == "3d":
        models = _visual_artifacts(
            ws, kind, {".blend", ".glb", ".gltf", ".obj", ".stl", ".fbx", ".ply"})
        if models:
            model = max(models, key=_artifact_score)
            preview = _render_3d_preview(ws, model, cfg)
            if preview:
                gallery = _write_media_gallery(ws, [preview], "3d")
                return VisualTarget(
                    gallery.resolve().as_uri(),
                    str(model.relative_to(ws)), workdir=str(ws))

    return None


def _wait_for_url(proc: subprocess.Popen, *, timeout: float = 45.0) -> str:
    common = [5173, 3000, 4173, 4321, 8080, 8000]
    deadline = time.time() + timeout
    seen = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            if proc.stdout and select.select([proc.stdout], [], [], 0.1)[0]:
                line = proc.stdout.readline()
                if line:
                    seen += line
                    m = re.search(r"https?://(?:localhost|127\.0\.0\.1):\d+", line)
                    if m:
                        return m.group(0).replace("localhost", "127.0.0.1")
        except Exception:
            pass
        for port in common:
            url = f"http://127.0.0.1:{port}"
            try:
                with urllib.request.urlopen(url, timeout=0.6) as r:
                    if r.status < 500:
                        return url
            except Exception:
                pass
        time.sleep(0.5)
    tail = " ".join(seen.splitlines()[-4:])
    raise RuntimeError(f"dev server did not become reachable{': ' + tail if tail else ''}")


def _capture(url: str, out_dir: Path, *, timeout_ms: int = 30_000,
             executable_path: str = "") -> tuple[list[Path], dict]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(f"playwright is not installed: {exc}") from exc

    shots: list[Path] = []
    checks: dict[str, Any] = {"viewports": {}, "issues": []}
    target_url = urllib.parse.urlparse(url)
    discovered_routes: list[str] = []

    def route_request(route) -> None:
        parsed = urllib.parse.urlparse(route.request.url)
        local = parsed.scheme in {"file", "data", "blob", "about"}
        same_remote = (
            target_url.scheme in {"http", "https"}
            and parsed.scheme in {"http", "https"}
            and parsed.hostname == target_url.hostname
            and (
                parsed.port == target_url.port
                or parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            )
        )
        if local or same_remote:
            route.continue_()
        else:
            route.abort("blockedbyclient")

    viewports = [("desktop", 1440, 900), ("mobile", 390, 844), ("wide", 1920, 1080)]
    with sync_playwright() as p:
        launch = {"headless": True}
        if executable_path:
            launch["executable_path"] = executable_path
        browser = p.chromium.launch(
            **launch, timeout=max(5_000, min(timeout_ms, 30_000)))
        try:
            for name, width, height in viewports:
                page = browser.new_page(viewport={"width": width, "height": height},
                                        device_scale_factor=1)
                page.route("**/*", route_request)
                console_errors: list[str] = []
                page_errors: list[str] = []
                request_failures: list[str] = []
                page.on("console", lambda msg: console_errors.append(msg.text)
                        if msg.type == "error" else None)
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on("requestfailed", lambda request: request_failures.append(
                    f"{request.method} {request.url}: {request.failure}"))
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
                except Exception:
                    pass
                page.wait_for_timeout(700)
                audit = page.evaluate("""() => {
                  const visible = (el) => {
                    const s = getComputedStyle(el), r = el.getBoundingClientRect();
                    return s.display !== 'none' && s.visibility !== 'hidden' &&
                      Number(s.opacity || 1) > 0 && r.width > 0 && r.height > 0 &&
                      el.getAttribute('aria-hidden') !== 'true';
                  };
                  const label = (el) => (el.getAttribute('aria-label') ||
                    el.getAttribute('title') || el.innerText || el.value || '').trim();
                  const all = [...document.querySelectorAll('body *')].filter(visible);
                  const clipped = all.filter((el) => {
                    const s = getComputedStyle(el);
                    const hides = ['hidden', 'clip'].includes(s.overflowX) ||
                      ['hidden', 'clip'].includes(s.overflowY);
                    const intentional = s.textOverflow === 'ellipsis' ||
                      s.webkitLineClamp !== 'none';
                    return hides && !intentional && (el.innerText || '').trim() &&
                      (el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2);
                  }).slice(0, 12).map(el => ({tag: el.tagName, id: el.id,
                    text: (el.innerText || '').trim().slice(0, 80)}));
                  const controls = [...document.querySelectorAll(
                    'button,input,select,textarea,a[href],[role="button"],[role="link"]'
                  )].filter(visible);
                  const unlabeled = controls.filter(el => !label(el)).slice(0, 12)
                    .map(el => ({tag: el.tagName, id: el.id}));
                  const touch = controls.filter(el => !['A'].includes(el.tagName)).filter(el => {
                    const r = el.getBoundingClientRect(); return r.width < 44 || r.height < 44;
                  }).slice(0, 12).map(el => { const r = el.getBoundingClientRect(); return {
                    tag: el.tagName, id: el.id, text: label(el).slice(0, 50),
                    width: Math.round(r.width), height: Math.round(r.height)}; });
                  const ids = [...document.querySelectorAll('[id]')].map(el => el.id).filter(Boolean);
                  const duplicates = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))].slice(0, 12);
                  const missingAlt = [...document.querySelectorAll('img')].filter(visible)
                    .filter(img => !img.hasAttribute('alt')).slice(0, 12)
                    .map(img => img.currentSrc || img.src || '(image)');
                  const zeroCanvas = [...document.querySelectorAll('canvas')].filter(visible)
                    .filter(c => c.width === 0 || c.height === 0).length;
                  const canvasRects = [...document.querySelectorAll('canvas')].filter(visible)
                    .map(c => { const r = c.getBoundingClientRect(); return {
                      left: Math.round(r.left + scrollX), top: Math.round(r.top + scrollY),
                      width: Math.round(r.width), height: Math.round(r.height),
                      backingWidth: c.width, backingHeight: c.height}; });
                  const internalRoutes = [...document.querySelectorAll('a[href]')]
                    .filter(visible).map(a => {
                      try { return new URL(a.href, location.href); } catch (_) { return null; }
                    }).filter(u => u && u.origin === location.origin &&
                      !u.pathname.match(/\.(?:png|jpe?g|gif|svg|pdf|zip|json)$/i))
                    .map(u => `${u.origin}${u.pathname}${u.search}`);
                  const deadAnchors = [...document.querySelectorAll('a[href]')].filter(visible)
                    .filter(a => {
                      const h = (a.getAttribute('href') || '').trim().toLowerCase();
                      return !h || h === '#' || h.startsWith('javascript:');
                    }).slice(0, 12).map(a => ({text: label(a).slice(0, 60),
                      href: a.getAttribute('href') || ''}));
                  return {
                    viewport: {width: innerWidth, height: innerHeight},
                    document: {width: document.documentElement.scrollWidth,
                      height: document.documentElement.scrollHeight},
                    horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 2,
                    clipped, unlabeled, smallTargets: touch, duplicateIds: duplicates,
                    missingImageAlt: missingAlt, zeroSizedCanvases: zeroCanvas,
                    canvasRects, interactiveCount: controls.length,
                    internalRoutes: [...new Set(internalRoutes)].slice(0, 8),
                    deadAnchors,
                    visibleTextLength: (document.body.innerText || '').trim().length,
                  };
                }""")
                focus_seen = set()
                for _ in range(min(24, int(audit.get("interactiveCount") or 0))):
                    page.keyboard.press("Tab")
                    focused = page.evaluate("""() => {
                      const e = document.activeElement;
                      return e ? `${e.tagName}:${e.id || e.getAttribute('aria-label') || ''}` : '';
                    }""")
                    if focused and not str(focused).startswith("BODY:"):
                        focus_seen.add(str(focused))
                page.evaluate("""() => {
                  const e = document.activeElement; if (e && e.blur) e.blur();
                }""")
                audit["keyboardReachableCount"] = len(focus_seen)
                audit["consoleErrors"] = console_errors[:12]
                audit["pageErrors"] = page_errors[:12]
                audit["requestFailures"] = request_failures[:12]
                checks["viewports"][name] = audit
                if name == "desktop":
                    discovered_routes = [
                        route for route in (audit.get("internalRoutes") or [])
                        if route.rstrip("/") != url.rstrip("/")
                    ][:4]
                if audit.get("horizontalOverflow"):
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": "document is wider than the viewport",
                        "fix": "remove horizontal overflow and constrain responsive content",
                        "selector_or_file_hint": "documentElement.scrollWidth",
                    })
                for key, evidence, fix, severity in (
                    ("clipped", "visible text is clipped", "remove unintended fixed-height/overflow clipping", "major"),
                    ("unlabeled", "interactive controls lack accessible names", "add semantic labels or aria-labels", "major"),
                    ("smallTargets", "interactive targets are smaller than 40px", "increase control hit areas to at least 44px", "minor"),
                    ("duplicateIds", "duplicate DOM ids were found", "make every DOM id unique", "major"),
                    ("missingImageAlt", "visible images lack alt attributes", "add meaningful alt text or explicit empty alt for decoration", "major"),
                    ("deadAnchors", "visible links have inert destinations", "wire links to real destinations or use buttons for commands", "minor"),
                ):
                    rows = audit.get(key) or []
                    if rows:
                        checks["issues"].append({
                            "severity": severity, "screen": name,
                            "evidence": f"{evidence}: {json.dumps(rows[:5])[:500]}",
                            "fix": fix, "selector_or_file_hint": key,
                        })
                if audit.get("zeroSizedCanvases"):
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": f"{audit['zeroSizedCanvases']} visible canvas element(s) have zero backing dimensions",
                        "fix": "give each canvas stable CSS and backing-store dimensions",
                        "selector_or_file_hint": "canvas",
                    })
                if audit.get("interactiveCount") and not audit.get("keyboardReachableCount"):
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": "interactive controls exist but none were reachable by Tab",
                        "fix": "use native focusable controls or add correct keyboard semantics and focus order",
                        "selector_or_file_hint": "focus navigation",
                    })
                if audit.get("pageErrors") or audit.get("consoleErrors"):
                    errors = [*(audit.get("pageErrors") or []), *(audit.get("consoleErrors") or [])]
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": f"browser runtime errors: {json.dumps(errors[:5])[:600]}",
                        "fix": "repair runtime exceptions and console errors before visual sign-off",
                        "selector_or_file_hint": "browser console",
                    })
                if audit.get("requestFailures"):
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": f"resources or requests failed: {json.dumps(audit['requestFailures'][:5])[:600]}",
                        "fix": "make required assets/data routes load reliably and render a deliberate error state",
                        "selector_or_file_hint": "network requests",
                    })
                if (audit.get("visibleTextLength", 0) < 8
                        and not audit.get("canvasRects")):
                    checks["issues"].append({
                        "severity": "major", "screen": name,
                        "evidence": "page has almost no visible content and no visible canvas",
                        "fix": "render the actual primary product view and a deliberate loading/error state",
                        "selector_or_file_hint": "body",
                    })
                path = out_dir / f"{name}.png"
                figures = page.locator("figure")
                figure_count = figures.count()
                long_gallery = (
                    figure_count > 1
                    and int((audit.get("document") or {}).get("height") or 0) > 6000
                )
                page.screenshot(path=str(path), full_page=not long_gallery)
                try:
                    from PIL import Image, ImageStat

                    with Image.open(path).convert("RGB") as screenshot:
                        for index, rect in enumerate(audit.get("canvasRects") or [], 1):
                            box = (
                                max(0, rect["left"]), max(0, rect["top"]),
                                min(screenshot.width, rect["left"] + rect["width"]),
                                min(screenshot.height, rect["top"] + rect["height"]),
                            )
                            if box[2] <= box[0] or box[3] <= box[1]:
                                continue
                            stat = ImageStat.Stat(screenshot.crop(box).resize((64, 64)))
                            spread = max(hi - lo for lo, hi in stat.extrema)
                            if spread <= 2:
                                checks["issues"].append({
                                    "severity": "major", "screen": name,
                                    "evidence": f"canvas {index} is pixel-uniform in the captured frame",
                                    "fix": "initialize and frame the real canvas scene/data before sign-off",
                                    "selector_or_file_hint": f"canvas:nth-of-type({index})",
                                })
                except ImportError:
                    audit["canvasPixelAudit"] = "Pillow unavailable"
                shots.append(path)
                if name == "desktop" and figure_count > 1:
                    for figure_index in range(min(8, figure_count)):
                        figure_path = out_dir / f"artifact-{figure_index + 1}.png"
                        figures.nth(figure_index).screenshot(path=str(figure_path))
                        shots.append(figure_path)
                page.close()
            for index, route_url in enumerate(discovered_routes, 1):
                page = browser.new_page(
                    viewport={"width": 1440, "height": 900},
                    device_scale_factor=1,
                )
                page.route("**/*", route_request)
                route_errors: list[str] = []
                page.on("pageerror", lambda exc: route_errors.append(str(exc)))
                page.on(
                    "console",
                    lambda msg: route_errors.append(msg.text)
                    if msg.type == "error" else None,
                )
                try:
                    response = page.goto(
                        route_url, wait_until="domcontentloaded",
                        timeout=timeout_ms)
                    page.wait_for_timeout(500)
                    route_audit = page.evaluate("""() => ({
                      text: (document.body.innerText || '').trim().length,
                      width: document.documentElement.scrollWidth,
                      viewport: innerWidth,
                      title: document.title
                    })""")
                    checks["viewports"][f"route-{index}"] = {
                        "url": route_url, **route_audit,
                        "status": response.status if response else 0,
                        "errors": route_errors[:8],
                    }
                    if (
                        not response or response.status >= 400
                        or route_audit.get("text", 0) < 8 or route_errors
                    ):
                        checks["issues"].append({
                            "severity": "major",
                            "screen": f"route-{index}",
                            "evidence": (
                                f"linked route failed its smoke check: status "
                                f"{response.status if response else 'none'}, "
                                f"text {route_audit.get('text', 0)}, "
                                f"errors {route_errors[:3]}"
                            ),
                            "fix": (
                                "make every visible navigation destination render "
                                "a complete error-free product view"
                            ),
                            "selector_or_file_hint": route_url,
                        })
                    if route_audit.get("width", 0) > route_audit.get(
                            "viewport", 0) + 2:
                        checks["issues"].append({
                            "severity": "major",
                            "screen": f"route-{index}",
                            "evidence": "linked route overflows the desktop viewport",
                            "fix": "constrain the linked view to the responsive layout",
                            "selector_or_file_hint": route_url,
                        })
                    route_path = out_dir / f"route-{index}.png"
                    page.screenshot(path=str(route_path), full_page=True)
                    shots.append(route_path)
                except Exception as exc:
                    checks["issues"].append({
                        "severity": "major",
                        "screen": f"route-{index}",
                        "evidence": f"linked route could not be opened: {exc}",
                        "fix": "repair or remove the broken navigation destination",
                        "selector_or_file_hint": route_url,
                    })
                finally:
                    page.close()
        finally:
            browser.close()
    return shots, checks


def _model_supports_vision(ol, model: str) -> bool:
    try:
        r = ol._client.post(f"{ol.base_url}/api/show", json={"model": model})
        if r.status_code != 200:
            return False
        caps = r.json().get("capabilities") or []
        return "vision" in caps
    except Exception:
        return False


def choose_vision_model(cfg, ol) -> str:
    candidates = [
        getattr(cfg, "vision_model", "") or "",
        getattr(cfg.planner, "name", ""),
        getattr(cfg.worker, "name", ""),
        "qwen3.6:35b-a3b",
        "qwen3.6:latest",
    ]
    seen = set()
    for model in candidates:
        if not model or model in seen:
            continue
        seen.add(model)
        if _model_supports_vision(ol, model):
            return model
    return ""


def _review_prompt(goal: str, design: str, target: VisualTarget) -> tuple[str, str]:
    system = (
        "You are Spiral's visual QA reviewer. Inspect screenshots like a product "
        "designer and frontend QA engineer. Return JSON only. Be strict about visual "
        "overlap, clipped text, unreadable contrast, mobile responsiveness, broken images, "
        "blank canvases, awkward hierarchy, inconsistent spacing, generic card-grid/landing-page "
        "treatment, poor domain fit, and obvious missing states. Verify that the actual product "
        "workflow dominates the screen, controls use familiar forms, real content/assets are "
        "inspectable, plots are legible, and mobile/desktop/wide compositions are intentional. "
        "Do not complain about subjective style if the screen is coherent."
    )
    user = (
        "Return exactly this JSON shape:\n"
        '{"verdict":"pass|revise","summary":"...","issues":[{"severity":"critical|major|minor",'
        '"screen":"desktop|mobile|wide|multiple","evidence":"what is visibly wrong",'
        '"fix":"specific implementation fix","selector_or_file_hint":"optional"}],"positives":["..."]}\n\n'
        f"GOAL:\n{goal[:4000]}\n\nDESIGN SPEC:\n{design[:6000] if design else '(none)'}\n\n"
        f"TARGET: {target.label} {target.url}\n"
    )
    return system, user


def _report_md(result: VisualReviewResult, data: dict) -> str:
    lines = ["# Visual review", "", f"Status: {result.status}", f"Model: {result.model}",
             f"Target: {result.target}", ""]
    if data.get("summary"):
        lines += ["## Summary", str(data.get("summary")), ""]
    lines += ["## Issues"]
    if not result.issues:
        lines.append("(none)")
    for i, issue in enumerate(result.issues, 1):
        lines.append(
            f"{i}. [{issue.get('severity', 'major')}] {issue.get('screen', 'both')}: "
            f"{issue.get('evidence', '')} Fix: {issue.get('fix', '')}"
        )
    lines += ["", "## Screenshots"]
    for shot in result.screenshots:
        lines.append(f"- {shot}")
    return "\n".join(lines) + "\n"


def review_project_visuals(ws: str | Path, cfg, ol, goal: str, kind: str = "",
                           *, on=None, on_thought=None, round_no: int = 1) -> VisualReviewResult:
    ws = Path(ws)
    kind_slug = re.sub(r"[^a-z0-9]+", "-", kind.lower()).strip("-") or "visual"
    out_dir = (
        ws / ".spiral" / "visual-review"
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind_slug}-r{round_no}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(cfg, "visual_review", True):
        return VisualReviewResult("skipped", "visual review disabled")

    target = discover_visual_target(ws, cfg, goal, kind)
    if target is None:
        return VisualReviewResult("skipped", "no visual target found; set SPIRAL_VISUAL_URL or .spiral/visual_url")

    model = choose_vision_model(cfg, ol)
    if not model:
        return VisualReviewResult("skipped", "no installed Ollama model reports vision capability")

    from spiral.builder_tools import ensure_playwright_chromium

    browser_runtime = ensure_playwright_chromium(
        ws, timeout=max(300, int(getattr(cfg, "verify_timeout", 900))))
    if not browser_runtime.get("ok"):
        return VisualReviewResult("skipped", str(browser_runtime.get("detail") or
                                                  "Chromium runtime unavailable"))

    proc = None
    url = target.url
    browser_env = browser_runtime.get("environment") or {}
    old_browser_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    try:
        os.environ.update(browser_env)
        if target.managed and target.command:
            if on:
                on(f"starting {target.label}")
            server_home = ws / ".spiral" / "visual-review" / "server-home"
            server_home.mkdir(parents=True, exist_ok=True)
            from spiral.command_broker import scrubbed_environment

            server_env = scrubbed_environment(
                ws, {"HOME": str(server_home), "CI": "1"})
            server_command, sandboxed = _sandboxed_server_command(
                ws.resolve(), target.command)
            if (getattr(cfg, "builder_require_sandbox", True)
                    and not sandboxed):
                raise RuntimeError(
                    "preview server refused because no outbound-network sandbox is available")
            proc = subprocess.Popen(
                server_command,
                cwd=Path(target.workdir) if target.workdir else ws,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=server_env,
            )
            url = _wait_for_url(proc)
            target.url = url
        if on:
            on(f"capturing {url}")
        captured = _capture(
            url, out_dir,
            timeout_ms=int(getattr(cfg, "visual_review_timeout", 45)) * 1000,
            executable_path=str(browser_runtime.get("executable") or ""),
        )
        if isinstance(captured, tuple):
            shots, deterministic_checks = captured
        else:  # compatibility for third-party/test capture providers
            shots, deterministic_checks = captured, {"viewports": {}, "issues": []}
        design_f = ws / ".spiral" / "design.md"
        design = design_f.read_text(encoding="utf-8") if design_f.is_file() else ""
        system, user = _review_prompt(goal, design, target)
        images = [base64.b64encode(Path(s).read_bytes()).decode("ascii") for s in shots]

        def _delta(kind: str, piece: str) -> None:
            if kind == "think" and on_thought:
                on_thought(piece)
            elif kind == "text" and on:
                on("writing visual verdict")

        res = ol.chat(
            model,
            [{"role": "system", "content": system},
             {"role": "user", "content": user, "images": images}],
            think=True,
            num_predict=max(1536, int(getattr(cfg, "visual_review_max_tokens", 2048))),
            num_ctx=getattr(cfg.planner, "num_ctx", None),
            keep_alive=getattr(cfg, "keep_alive", None),
            temperature=0.1,
            on_delta=_delta,
        )
        text = (res.text or "").strip()
        if not text:
            # Thinking models can spend the entire cap in reasoning. Retry without
            # thinking so the build loop still gets a usable verdict.
            res = ol.chat(
                model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": user, "images": images}],
                think=False,
                num_predict=max(1024, int(getattr(cfg, "visual_review_max_tokens", 2048))),
                num_ctx=getattr(cfg.planner, "num_ctx", None),
                keep_alive=getattr(cfg, "keep_alive", None),
                temperature=0.1,
            )
            text = (res.text or "").strip()
        data = _json_object(text)
        model_issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
        deterministic_issues = [
            i for i in (deterministic_checks.get("issues") or []) if isinstance(i, dict)
        ]
        issues = [
            *deterministic_issues,
            *model_issues,
        ]
        verdict = str(data.get("verdict") or ("revise" if issues else "pass")).lower()
        status = "revise" if issues and (deterministic_issues or verdict == "revise") else "pass"
        result = VisualReviewResult(
            status=status,
            detail=str(data.get("summary") or ("visual issues found" if issues else "visual review passed")),
            model=model,
            target=url,
            screenshots=[str(s) for s in shots],
            issues=issues,
            raw=text,
            deterministic_checks=deterministic_checks,
        )
        report = _report_md(result, data)
        report_f = out_dir / "report.md"
        manifest_f = out_dir / "manifest.json"
        result.report = str(report_f)
        result.manifest = str(manifest_f)
        report_f.write_text(report, encoding="utf-8")
        manifest_f.write_text(json.dumps({**asdict(result), "round": round_no}, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        manifest_f = out_dir / "manifest.json"
        result = VisualReviewResult("skipped", f"{type(exc).__name__}: {exc}",
                                    model=model, target=url, manifest=str(manifest_f))
        manifest_f.write_text(json.dumps({**asdict(result), "round": round_no}, indent=2), encoding="utf-8")
        return result
    finally:
        if old_browser_path is None:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old_browser_path
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def issues_to_verdicts(result: VisualReviewResult) -> list[dict]:
    verdicts = []
    for i, issue in enumerate(result.issues, 1):
        sev = str(issue.get("severity") or "major").lower()
        if sev not in {"critical", "major", "minor"}:
            sev = "major"
        status = "missing" if sev in {"critical", "major"} else "partial"
        evidence = f"{issue.get('screen', 'both')}: {issue.get('evidence', '')}".strip()
        fix = str(issue.get("fix") or "repair the visual defect")
        hint = issue.get("selector_or_file_hint") or ""
        if hint:
            fix += f" Hint: {hint}"
        verdicts.append({
            "id": f"visual-{i}",
            "status": status,
            "evidence": evidence,
            "fix": {
                "title": f"fix visual issue {i}: {sev}",
                "description": (
                    f"Vision review found a {sev} UI issue. Evidence: {evidence}. "
                    f"Required fix: {fix}. Review report: {result.report}"
                ),
                "files": [],
            },
        })
    return verdicts
