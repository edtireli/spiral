import io
import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

from spiral import cli, model_picker as mp
from spiral.config import Config


def test_inventory_only_reads_tags_and_keeps_installed_names():
    requests = []

    def tags(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"models": [
            {"name": "qwen:small", "size": 2_000_000_000, "details": {
                "parameter_size": "4B", "quantization_level": "Q4_K_M"}},
            {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
            {"name": "qwen:small"},
            {"name": "gemma:tiny", "size": 1_000_000_000},
        ]})

    models = mp.installed_models("http://inventory.test/prefix", transport=httpx.MockTransport(tags))
    assert requests == [("GET", "/prefix/api/tags")]
    assert [m.name for m in models] == ["gemma:tiny", "qwen:small", "nomic-embed-text:latest"]
    assert models[-1].embedding_only


@pytest.mark.parametrize("payload", [{}, {"models": [None]}, {"models": [{"name": "x\x1b[2J"}]}])
def test_malformed_inventory_is_an_error_not_an_empty_list(payload):
    with pytest.raises(mp.PickerError, match="Could not read"):
        mp.installed_models("http://inventory.test", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=payload)))


def test_offline_tests_cannot_accidentally_contact_real_server():
    with pytest.raises(mp.PickerError, match="disabled"):
        mp.installed_models("http://127.0.0.1:11434")


def test_save_updates_all_six_roles_and_preserves_other_config(tmp_path):
    path = tmp_path / "config.json"
    previous = {"models": {"critic": "old", "custom": "keep"}, "num_ctx": {"new": 8192},
                "providers": {"remote": {"base_url": "https://example.test"}},
                "style": "uzumaki", "visual_review": False}
    path.write_text(json.dumps(previous))
    mp.save_selection(mp.InstalledModel("new"), path=path)
    saved = json.loads(path.read_text())
    assert saved["models"] == {"custom": "keep", **{r: "new" for r in mp.ROLES}}
    assert {k: v for k, v in saved.items() if k != "models"} == {
        k: v for k, v in previous.items() if k != "models"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".model-*"))


def test_saved_choice_is_used_by_the_build_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for role in mp.ROLES:
        monkeypatch.delenv("SPIRAL_" + role.upper(), raising=False)
    mp.save_selection(mp.InstalledModel("already-installed:small"))
    cfg = Config.load()
    assert {getattr(cfg, role).name for role in mp.ROLES} == {"already-installed:small"}


@pytest.mark.parametrize("content", ["broken json", "[]", '{"models": []}'])
def test_invalid_config_is_never_overwritten(tmp_path, content):
    path = tmp_path / "config.json"
    path.write_text(content)
    with pytest.raises(mp.PickerError, match="not changed"):
        mp.save_selection(mp.InstalledModel("new"), path=path)
    assert path.read_text() == content


def test_config_write_failure_keeps_original(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"models": {"worker": "old"}}')
    original = path.read_bytes()
    monkeypatch.setattr(mp.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(mp.PickerError, match="Could not save"):
        mp.save_selection(mp.InstalledModel("new"), path=path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".model-*"))


def test_arrow_scrolling_filtering_and_selection():
    picker = mp.Picker([mp.InstalledModel(f"model-{i:02}") for i in range(30)], "model-15")
    assert picker.index == 15
    picker.key("PAGE_DOWN", page_size=7)
    assert picker.index == 22
    picker.key("UP")
    assert picker.index == 21
    for char in "model-29":
        picker.key(char)
    assert picker.key("ENTER") == (True, mp.InstalledModel("model-29"))
    picker.key("CLEAR")
    assert len(picker.visible) == 30
    picker.key("x")
    assert picker.key("ENTER") == (False, None)
    assert picker.key("ESC") == (True, None)


def test_embedding_models_cannot_be_selected_or_saved(tmp_path):
    model = mp.InstalledModel("embedding", embedding_only=True)
    picker = mp.Picker([model], "")
    assert picker.key("ENTER") == (False, None)
    assert "cannot run builds" in picker.note
    with pytest.raises(mp.PickerError):
        mp.save_selection(model, path=tmp_path / "config.json")
    assert not (tmp_path / "config.json").exists()


def test_render_scrolls_selected_model_into_view_and_escapes_markup():
    picker = mp.Picker([mp.InstalledModel(f"model-{i:02}") for i in range(30)], "model-29")
    output = io.StringIO()
    console = Console(file=output, width=90)
    console.print(picker.render(7))
    assert "model-29" in output.getvalue()
    assert "model-00" not in output.getvalue()
    assert "24–30 of 30" in output.getvalue()
    output.truncate(0)
    output.seek(0)
    console.print(mp.Picker([mp.InstalledModel("model[red]")], "").render(7))
    assert "model[red]" in output.getvalue()


@pytest.mark.parametrize("keys, expected", [(b"\x1b[A", "UP"), (b"\x1b[B", "DOWN"),
    (b"\x1b[6~", "PAGE_DOWN"), (b"\x1b", "ESC"), (b"\n", "ENTER"),
    ("ø".encode(), "ø"), (b"\x15", "CLEAR")])
def test_terminal_key_sequences(keys, expected):
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, keys)
        assert mp._read_key(read_fd) == expected
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("argv, tty, expected", [(["spiral"], True, False),
    (["spiral", "models"], True, False), (["spiral", "models", "--list"], False, True)])
def test_cli_picker_bypasses_model_admission(monkeypatch, argv, tty, expected):
    calls = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr(cli, "make_console", lambda: Console(file=io.StringIO(), force_terminal=tty))
    monkeypatch.setattr(cli, "print_banner", lambda *_a, **_k: None)
    monkeypatch.setattr(cli.Config, "load", classmethod(lambda cls: Config()))
    monkeypatch.setattr(cli, "_admit_model_command", lambda *_: pytest.fail("model admission"))
    monkeypatch.setattr(mp, "run", lambda c, cfg, **kw: calls.append(kw) or 0)
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 0
    assert calls == [{"list_only": expected}]


def test_bare_piped_cli_still_only_prints_status(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["spiral"])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(cli, "make_console", lambda: Console(file=io.StringIO()))
    monkeypatch.setattr(cli, "print_banner", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_health", lambda *_: calls.append("health"))
    monkeypatch.setattr(mp, "run", lambda *_a, **_k: pytest.fail("picker"))
    cli.main()
    assert calls == ["health"]


@pytest.mark.parametrize("list_only,interactive", [(True, True), (False, False)])
def test_listing_does_not_change_selection(monkeypatch, list_only, interactive):
    monkeypatch.setattr(mp, "installed_models", lambda _: [mp.InstalledModel("local")])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: interactive))
    monkeypatch.setattr(mp, "save_selection", lambda *_: pytest.fail("selection changed"))
    monkeypatch.setattr(mp, "choose", lambda *_: pytest.fail("prompted"))
    assert mp.run(Console(file=io.StringIO(), force_terminal=interactive), Config(), list_only=list_only) == 0


def test_cancel_does_not_change_selection(monkeypatch):
    monkeypatch.setattr(mp, "installed_models", lambda _: [mp.InstalledModel("local")])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(mp, "choose", lambda *_: None)
    monkeypatch.setattr(mp, "save_selection", lambda *_: pytest.fail("selection changed"))
    assert mp.run(Console(file=io.StringIO(), force_terminal=True), Config()) == 0


@pytest.mark.parametrize("answers,expected", [([""], None), (["bad", "99", "1"], "chat"),
    (["2", "1"], "chat")])
def test_basic_terminals_have_a_numbered_fallback(answers, expected):
    answers = iter(answers)
    console = SimpleNamespace(is_dumb_terminal=True, print=lambda *_a, **_k: None,
                              input=lambda *_: next(answers))
    picker = mp.Picker([mp.InstalledModel("chat"),
                       mp.InstalledModel("embed", embedding_only=True)], "chat")
    selected = mp.choose(console, picker)
    assert (selected.name if selected else None) == expected
