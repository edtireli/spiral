"""Authenticated public progress frames shared by pre-plan and worker displays."""
import json
import os
import re
import sys
import threading
import time
from copy import deepcopy

UI_EVENT_SCHEMA = "spiral.ui.progress.v1"
UI_EVENT_PREFIX = "SPIRAL_UI_EVENT_V1"
_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_lock = threading.Lock()
_token = ""
_sequence = 0
_project_milestones = None
_started = 0.0


def emit_progress(payload):
    """Return whether a frame was emitted on the authenticated UI channel."""
    global _token, _sequence, _project_milestones, _started
    token = os.environ.get("SPIRAL_UI_EVENT_TOKEN", "").strip().lower()
    if not _TOKEN.fullmatch(token):
        return False
    with _lock:
        if token != _token:
            _token, _sequence = token, 0
            _project_milestones, _started = None, time.monotonic()
        _sequence += 1
        frame = {**payload, "schema_version": UI_EVENT_SCHEMA, "sequence": _sequence}
        # A final-check spinner is a phase within the same project, not a new
        # empty project. Only explicitly typed project frames establish that
        # durable view; never infer ownership from titles or terminal text.
        scope = frame.pop("plan_scope", "phase")
        current = deepcopy(frame.get("milestones") or [])
        if scope == "project":
            _project_milestones = current
        elif _project_milestones is not None:
            current = deepcopy(_project_milestones) + current
        if _project_milestones is not None:
            for index, milestone in enumerate(current, 1):
                milestone["index"] = index
            frame["milestones"] = current
            frame["done"] = sum(task.get("status") == "done"
                                for m in current for task in m.get("tasks", []))
            frame["blocked"] = sum(task.get("status") == "blocked"
                                   for m in current for task in m.get("tasks", []))
        frame["elapsed_seconds"] = max(float(frame.get("elapsed_seconds", 0)),
                                       round(time.monotonic() - _started, 3))
        line = (f"{UI_EVENT_PREFIX} {token} "
                + json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
        try:
            os.write(sys.stdout.fileno(), line.encode("utf-8"))
        except (AttributeError, OSError, ValueError):
            sys.stdout.write(line)
            sys.stdout.flush()
        return True
