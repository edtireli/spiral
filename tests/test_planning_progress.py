import json

from spiral.banner import Spinner
from spiral.dash import Dash
from spiral.ui_progress import UI_EVENT_PREFIX


def test_authenticated_heartbeat_keeps_live_frames_without_duplicate_log_lines(monkeypatch, capfd):
    from spiral.banner import Spinner
    from spiral.dash import Dash
    from rich.console import Console
    import json

    class Once:
        def __init__(self): self.calls = 0
        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1

    for token in ("ef" * 16, "invalid", ""):
        monkeypatch.setenv("SPIRAL_UI_EVENT_TOKEN", token)
        dash = Dash(console=Console(force_terminal=False))
        spinner = Spinner("planning"); spinner._tty = False
        for display, loop in ((dash, dash._hb_loop), (spinner, spinner._loop)):
            display._stop = Once(); display.tick(17); loop()
            output = capfd.readouterr().out
            if token == "ef" * 16:
                lines = output.strip().splitlines()
                assert len(lines) == 1 and lines[0].startswith(UI_EVENT_PREFIX)
                frame = json.loads(lines[0].split(" ", 2)[2])
                assert frame["tokens"] == 17 and frame["final"] is False
            else:
                assert UI_EVENT_PREFIX not in output and "⠿ [" in output


def test_preplan_phases_and_worker_share_one_public_sequence(monkeypatch, capsys):
    token = "bc" * 16
    monkeypatch.setenv("SPIRAL_UI_EVENT_TOKEN", token)
    with Spinner("extracting requirements"):
        pass
    with Spinner("mapping deliverables"):
        pass
    dash = Dash()
    dash.phase("building", model="chosen:exact")
    frames = [json.loads(line.split(" ", 2)[2]) for line in capsys.readouterr().out.splitlines()
              if line.startswith(UI_EVENT_PREFIX)]
    assert [f["phase"] for f in frames] == ["extracting requirements", "mapping deliverables", "building"]
    assert [f["sequence"] for f in frames] == [1, 2, 3]
    assert all(f["final"] is False for f in frames)
    assert all(f["idea"] == "" for f in frames)


def test_interrupted_phase_never_claims_completed_work(monkeypatch, capsys):
    monkeypatch.setenv("SPIRAL_UI_EVENT_TOKEN", "cd" * 16)
    try:
        with Spinner("planning"):
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass
    frames = [json.loads(line.split(" ", 2)[2]) for line in capsys.readouterr().out.splitlines()
              if line.startswith(UI_EVENT_PREFIX)]
    assert len(frames) == 1 and frames[0]["final"] is False and frames[0]["done"] == 0


def test_standalone_spinner_does_not_emit_machine_frames(monkeypatch, capsys):
    monkeypatch.delenv("SPIRAL_UI_EVENT_TOKEN", raising=False)
    with Spinner("planning"):
        pass
    assert UI_EVENT_PREFIX not in capsys.readouterr().out


def test_final_phases_keep_project_progress_without_inventing_completion(monkeypatch, capfd):
    from spiral.planner import Plan, Milestone, Task
    from spiral.ui_progress import emit_progress

    monkeypatch.setenv("SPIRAL_UI_EVENT_TOKEN", "ab" * 16)
    plan = Plan("project", [Milestone("implementation", [Task("write", "write code"),
                                                         Task("verify", "run checks")])])
    dash = Dash(plan=plan, plan_scope="project")
    dash.task(1, 1, "done")
    dash.task(1, 2, "blocked")
    with Spinner("final review"):
        pass
    frames = [json.loads(line.split(" ", 2)[2]) for line in capfd.readouterr().out.splitlines()
              if line.startswith(UI_EVENT_PREFIX)]
    final_phase = frames[-1]
    assert final_phase["phase"] == "final review" and final_phase["final"] is False
    assert [t["status"] for t in final_phase["milestones"][0]["tasks"]] == ["done", "blocked"]
    assert final_phase["done"] == 1 and final_phase["blocked"] == 1

    subplan = Plan("repair", [Milestone("repair", [Task("repair", "fix failed check")])])
    repair = Dash(plan=subplan)
    repair.task(1, 1, "run")
    with Spinner("check after repair"):
        pass
    frames = [json.loads(line.split(" ", 2)[2]) for line in capfd.readouterr().out.splitlines()
              if line.startswith(UI_EVENT_PREFIX)]
    assert [m["index"] for m in frames[0]["milestones"]] == [1, 2]
    assert frames[0]["milestones"][1]["tasks"][0]["status"] == "running"
    assert len(frames[-1]["milestones"]) == 1
    assert frames[-1]["blocked"] == 1  # no verified repair result has updated it
    assert frames[-1]["elapsed_seconds"] >= frames[0]["elapsed_seconds"]

    monkeypatch.setenv("SPIRAL_UI_EVENT_TOKEN", "ac" * 16)
    emit_progress({"phase": "new independent run", "milestones": []})
    fresh = json.loads(capfd.readouterr().out.split(" ", 2)[2])
    assert fresh["milestones"] == [] and fresh["sequence"] == 1
