"""Engine producer for the host's versioned, crash-persistent inference fence.

The caller must own the machine flock throughout this context. A terminal model
record plus a successfully closed reader establishes drainage. Timeouts, HTTP
errors and process death do not. This records no prompts or model output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import uuid


class InferenceRequestBoundary:
    def __init__(self, root: Path | None, model: str, operation: str):
        self.root = root
        self.record = {
            "schema": "spiral.compute.inference.v1", "attempt_id": "engine-" + uuid.uuid4().hex,
            "model": model, "operation": operation, "state": "dispatching", "started_at": time.time(),
        }
        self.terminal = False
        self.reader_closed = False
        self.rejected = False

    def _write(self):
        if self.root is None:
            return
        raw = json.dumps(self.record, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 8192:
            raise ValueError("inference boundary exceeds wire limit")
        fd, temporary = tempfile.mkstemp(prefix=".inference-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.root / "spiral-compute.inference.json")
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def __enter__(self):
        self._write()  # Persist before POST; crashes remain conservatively uncertain.
        return self

    def __exit__(self, *_):
        if self.root is not None:
            with (self.root / "spiral-compute.inference.json").open("rb") as source:
                raw = source.read(8193)
            current = json.loads(raw)
            if len(raw) > 8192 or current.get("attempt_id") != self.record["attempt_id"]:
                raise RuntimeError("stale engine inference settlement")
        self.record.update(
            state=("rejected" if self.rejected and self.reader_closed else
                   "drained" if self.terminal and self.reader_closed else "uncertain"),
            finished_at=time.time(),
        )
        self._write()


def thinking_rejection(client, base_url: str, model: str, response) -> bool:
    """Only the pinned pre-scheduler rejection permits a compatibility replay."""
    if response.status_code != 400:
        return False
    try:
        # This is an error body, not a model stream; cap before decoding it.
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=1024):
            body.extend(chunk)
            if len(body) > 4096:
                return False
        if json.loads(body).get("error") != json.dumps(model, ensure_ascii=False) + " does not support thinking":
            return False
        with client.stream("GET", base_url.rstrip("/") + "/api/version", timeout=2) as version:
            version.raise_for_status()
            raw = bytearray()
            for chunk in version.iter_bytes(chunk_size=1024):
                raw.extend(chunk)
                if len(raw) > 1024:
                    return False
        return json.loads(raw) == {"version": "0.32.14"}
    except Exception:
        return False
