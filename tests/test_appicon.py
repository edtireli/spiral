"""The launcher-icon generator is deterministic harness ground truth — so it is
unit-testable without a model. Runs standalone (`python tests/test_appicon.py`)
or under pytest.
"""
from __future__ import annotations

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spiral.appicon import (  # noqa: E402
    GLYPHS, icon_vector, write_android_icon, write_android_tokens, _norm_hex,
)

_MANIFEST = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<manifest xmlns:android="http://schemas.android.com/apk/res/android">\n'
    '    <application\n'
    '        android:label="@string/app_name"\n'
    '        android:theme="@style/Theme.App">\n'
    '        <activity android:name=".MainActivity" android:exported="true"/>\n'
    '    </application>\n'
    '</manifest>\n'
)


def _android_project(d: str) -> Path:
    root = Path(d)
    main = root / "app" / "src" / "main"
    (main / "res" / "values").mkdir(parents=True)
    (main / "AndroidManifest.xml").write_text(_MANIFEST)
    return root


def test_every_glyph_is_wellformed_vector():
    for g in GLYPHS:
        ET.fromstring(icon_vector("#D97757", "#0A0A0A", g))  # raises if malformed


def test_unknown_glyph_falls_back_to_spiral():
    # an out-of-set glyph must not crash — it degrades to the default mark
    ET.fromstring(icon_vector("#FFFFFF", "#000000", "definitely-not-a-glyph"))


def test_write_creates_adaptive_icon_and_wires_manifest():
    with tempfile.TemporaryDirectory() as d:
        ws = _android_project(d)
        written = write_android_icon(ws, "#FF1744", "#0A0A0A", "eye")
        res = ws / "app/src/main/res"
        expected = [
            "drawable/ic_launcher_foreground.xml",
            "drawable/ic_launcher_background.xml",
            "mipmap-anydpi-v26/ic_launcher.xml",
            "mipmap-anydpi-v26/ic_launcher_round.xml",
            "mipmap-anydpi/ic_launcher.xml",
            "mipmap-anydpi/ic_launcher_round.xml",
        ]
        for rel in expected:
            f = res / rel
            assert f.is_file(), f"{rel} not written"
            ET.fromstring(f.read_text())  # every emitted file is well-formed
            assert str(f.relative_to(ws)) in written
        manifest = (ws / "app/src/main/AndroidManifest.xml").read_text()
        ET.fromstring(manifest)  # still valid after patch
        assert 'android:icon="@mipmap/ic_launcher"' in manifest
        assert 'android:roundIcon="@mipmap/ic_launcher_round"' in manifest


def test_idempotent_second_run_reports_no_change():
    with tempfile.TemporaryDirectory() as d:
        ws = _android_project(d)
        write_android_icon(ws, "#FF1744", "#0A0A0A", "eye")
        again = write_android_icon(ws, "#FF1744", "#0A0A0A", "eye")
        assert again == [], f"expected no changes on re-run, got {again}"


def test_non_android_dir_is_noop():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "main.py").write_text("print('hi')\n")
        assert write_android_icon(d, "#FF1744", "#0A0A0A", "spiral") == []
        assert write_android_tokens(d, {"accent": "#FF1744"}) == []


def test_tokens_are_additive_and_wellformed():
    with tempfile.TemporaryDirectory() as d:
        ws = _android_project(d)
        (ws / "app/src/main/res/values/colors.xml").write_text(
            '<resources><color name="existing">#123456</color></resources>\n')
        written = write_android_tokens(ws, {"accent": "#B71C1C", "background": "#0A0A0A",
                                            "surface": "#141414", "on_dark": "#F2F2F2"})
        tok = ws / "app/src/main/res/values/spiral_tokens.xml"
        assert tok.is_file() and str(tok.relative_to(ws)) in written
        root = ET.fromstring(tok.read_text())
        names = {c.get("name") for c in root.findall("color")}
        assert names == {"token_accent", "token_background", "token_surface", "token_on_dark"}, names
        # existing colors.xml is untouched (additive)
        assert 'name="existing"' in (ws / "app/src/main/res/values/colors.xml").read_text()
        # idempotent
        assert write_android_tokens(ws, {"accent": "#B71C1C", "background": "#0A0A0A",
                                         "surface": "#141414", "on_dark": "#F2F2F2"}) == []


def test_bad_hex_is_normalized():
    assert _norm_hex("not-a-color", "#0A0A0A") == "#0A0A0A"
    assert _norm_hex("#abcdef", "#000000") == "#ABCDEF"
    assert _norm_hex("#FF112233", "#000000") == "#FF112233"  # 8-digit ARGB allowed


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  \033[32mPASS\033[0m {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  \033[31mFAIL\033[0m {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
