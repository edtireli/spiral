"""Authenticated cooperative pause control for controller-managed Spiral runs.

The host cannot safely freeze the Spiral process group: an inference lease may be
held by the frozen process and brokered commands run in their own process groups.
Instead, the host publishes an authenticated desired state.  Spiral acknowledges a
pause only at a safe checkpoint after all registered activity has drained.

The protocol is deliberately file based.  It works while stdout is congested, does
not expose a listening socket, and lets a restarted host recover the exact process'
last authenticated acknowledgement.
"""
from __future__ import annotations

import contextlib
import hmac
import json
import os
import stat
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator


SCHEMA = "spiral.runtime.control.v1"
CAPABILITY = "cooperative_pause_v1"
COMMAND_PATH_ENV = "SPIRAL_RUNTIME_CONTROL_PATH"
ACK_PATH_ENV = "SPIRAL_RUNTIME_CONTROL_ACK_PATH"
TOKEN_ENV = "SPIRAL_RUNTIME_CONTROL_TOKEN"
RUN_ID_ENV = "SPIRALCHAT_RUN_ID"
ATTEMPT_ENV = "SPIRALCHAT_ATTEMPT"
_MAX_CONTROL_BYTES = 16 * 1024


def _message_mac(token: str, value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(token.encode("utf-8"), payload, "sha256").hexdigest()


class RuntimeControlConfigurationError(RuntimeError):
    """A managed run supplied an incomplete or insecure control channel."""


class RuntimeControl:
    """One process-wide cooperative pause coordinator.

    The monitor is allowed to acknowledge ``pausing`` while work is active and to
    acknowledge ``running`` for a resume.  Only :meth:`checkpoint`, executing on an
    application thread with no registered activity, may acknowledge ``paused``.
    """

    def __init__(
        self,
        command_path: str | os.PathLike[str] | None,
        ack_path: str | os.PathLike[str] | None,
        token: str | None,
        run_id: str | None,
        attempt: int | str | None,
        *,
        poll_seconds: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
    ):
        supplied = [command_path, ack_path, token, run_id, attempt]
        self.enabled = any(value not in (None, "") for value in supplied)
        self.command_path = Path(command_path) if command_path else None
        self.ack_path = Path(ack_path) if ack_path else None
        self.token = str(token or "")
        self.run_id = str(run_id or "")
        try:
            self.attempt = int(attempt or 0)
        except (TypeError, ValueError):
            self.attempt = 0
        self.poll_seconds = max(0.01, float(poll_seconds))
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._activities: Counter[str] = Counter()
        self._local = threading.local()
        self._generation = 0
        self._request_id = ""
        self._desired = "running"
        self._state = "running"
        self._paused_started: float | None = None
        self._paused_total = 0.0
        self._started = False
        self._cancelled = False
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None

    @classmethod
    def from_env(cls) -> "RuntimeControl":
        return cls(
            os.environ.get(COMMAND_PATH_ENV),
            os.environ.get(ACK_PATH_ENV),
            os.environ.get(TOKEN_ENV),
            os.environ.get(RUN_ID_ENV),
            os.environ.get(ATTEMPT_ENV),
        )

    def _validate_private_parent(self, path: Path) -> None:
        if not path.is_absolute():
            raise RuntimeControlConfigurationError(
                "runtime control paths must be absolute")
        try:
            info = path.parent.stat()
        except OSError as exc:
            raise RuntimeControlConfigurationError(
                f"runtime control directory is unavailable: {exc}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeControlConfigurationError(
                "runtime control parent is not a directory")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise RuntimeControlConfigurationError(
                "runtime control directory is not owned by this user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeControlConfigurationError(
                "runtime control directory must be private (mode 0700)")

    def _validate_configuration(self) -> None:
        if not self.enabled:
            return
        if not all((self.command_path, self.ack_path, self.token, self.run_id)):
            raise RuntimeControlConfigurationError(
                "runtime control requires command path, ack path, token, run id, and attempt")
        if self.attempt <= 0:
            raise RuntimeControlConfigurationError(
                "runtime control attempt must be a positive integer")
        if len(self.token) < 16 or len(self.token) > 4096:
            raise RuntimeControlConfigurationError(
                "runtime control token length is invalid")
        if len(self.run_id) > 512 or "\x00" in self.run_id:
            raise RuntimeControlConfigurationError("runtime control run id is invalid")
        assert self.command_path is not None and self.ack_path is not None
        if self.command_path == self.ack_path:
            raise RuntimeControlConfigurationError(
                "runtime command and acknowledgement paths must differ")
        self._validate_private_parent(self.command_path)
        self._validate_private_parent(self.ack_path)

    def start(self) -> "RuntimeControl":
        with self._condition:
            if self._started or not self.enabled:
                return self
            self._validate_configuration()
            self._started = True
            self._write_ack_locked()
            self._monitor = threading.Thread(
                target=self._monitor_loop,
                name="spiral-runtime-control",
                daemon=True,
            )
            self._monitor.start()
        return self

    def _activity_payload_locked(self) -> dict:
        kinds = sorted(
            kind for kind, count in self._activities.items() if count > 0)
        return {"count": sum(self._activities.values()), "kinds": kinds}

    def _ack_payload_locked(self) -> dict:
        payload = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "generation": self._generation,
            "request_id": self._request_id,
            "desired": self._desired,
            "state": self._state,
            "activity": self._activity_payload_locked(),
            "pid": os.getpid(),
            "capability": CAPABILITY,
            "acknowledged_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        payload["mac"] = _message_mac(self.token, payload)
        return payload

    def _write_ack_locked(self) -> None:
        if not self.enabled or self.ack_path is None:
            return
        payload = json.dumps(
            self._ack_payload_locked(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(payload) > _MAX_CONTROL_BYTES:
            raise RuntimeControlConfigurationError(
                "runtime acknowledgement exceeds its bounded protocol size")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.ack_path.name}.",
            suffix=".tmp",
            dir=self.ack_path.parent,
        )
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("could not write runtime acknowledgement")
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, self.ack_path)
            try:
                directory_fd = os.open(self.ack_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # The file replacement is already atomic. Some filesystems do not
                # permit directory fsync; that does not weaken process-level recovery.
                pass
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _read_command(self) -> dict | None:
        if not self.enabled or self.command_path is None:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.command_path, flags)
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return None
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                return None
            if stat.S_IMODE(info.st_mode) & 0o077:
                return None
            raw = os.read(fd, _MAX_CONTROL_BYTES + 1)
            if len(raw) > _MAX_CONTROL_BYTES:
                return None
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        finally:
            os.close(fd)

    def _valid_command(self, command: dict) -> tuple[int, str, str] | None:
        if command.get("schema") != SCHEMA:
            return None
        supplied_mac = command.get("mac")
        signed = dict(command)
        signed.pop("mac", None)
        if (
            not isinstance(supplied_mac, str)
            or not hmac.compare_digest(
                supplied_mac, _message_mac(self.token, signed)
            )
        ):
            return None
        if command.get("run_id") != self.run_id:
            return None
        attempt = command.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            return None
        if attempt != self.attempt:
            return None
        generation = command.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            return None
        request_id = command.get("request_id")
        desired = command.get("desired")
        if generation <= 0 or not isinstance(request_id, str) or not request_id:
            return None
        if len(request_id) > 512 or desired not in {"paused", "running"}:
            return None
        return generation, request_id, desired

    def _apply_command(self, command: dict) -> None:
        parsed = self._valid_command(command)
        if parsed is None:
            return
        generation, request_id, desired = parsed
        with self._condition:
            if generation < self._generation:
                return
            if generation == self._generation:
                if (request_id, desired) != (self._request_id, self._desired):
                    return
                if self.ack_path is not None and not self.ack_path.exists():
                    self._write_ack_locked()
                return
            was_safely_paused = (
                self._desired == "paused"
                and self._state == "paused"
                and self._paused_started is not None
                and not self._activities
            )
            self._generation = generation
            self._request_id = request_id
            self._desired = desired
            if desired == "paused":
                # The monitor never claims the safe point. It only exposes that the
                # request is observed and the current activity is draining.
                self._state = "paused" if was_safely_paused else "pausing"
            else:
                if self._paused_started is not None:
                    self._paused_total += max(
                        0.0, self._clock() - self._paused_started)
                    self._paused_started = None
                self._state = "running"
            self._write_ack_locked()
            self._condition.notify_all()

    def poll(self) -> None:
        """Synchronously observe one command without ever entering a pause wait."""
        if not self.enabled or self._cancelled:
            return
        self.start()
        command = self._read_command()
        if command is not None:
            self._apply_command(command)

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                command = self._read_command()
                if command is not None:
                    self._apply_command(command)
            except Exception:
                # Invalid/transient files are ignored. A valid later atomic command
                # remains observable; the monitor must not silently die.
                continue

    def checkpoint(self) -> None:
        """Pause at this safe point if the authenticated desired state requires it."""
        if not self.enabled or self._cancelled:
            return
        self.poll()
        with self._condition:
            while not self._cancelled and self._desired == "paused":
                if sum(self._activities.values()) > 0:
                    if self._state != "pausing":
                        self._state = "pausing"
                        self._write_ack_locked()
                    # A checkpoint reached *inside* an admitted activity must let
                    # that activity drain. An outside coordinator thread waits for
                    # all activity, so it cannot race ahead while another thread is
                    # still finishing inference or a command.
                    if int(getattr(self._local, "depth", 0)) > 0:
                        return
                    self._condition.wait(timeout=self.poll_seconds * 2)
                    continue
                if self._state != "paused":
                    self._state = "paused"
                    self._paused_started = self._clock()
                    self._write_ack_locked()
                self._condition.wait(timeout=self.poll_seconds * 2)

    @contextlib.contextmanager
    def activity(self, kind: str) -> Iterator[None]:
        """Register work that must finish before a pause can be acknowledged."""
        label = str(kind or "work")[:80]
        while True:
            self.checkpoint()
            with self._condition:
                depth = int(getattr(self._local, "depth", 0))
                if self._cancelled or self._desired != "paused" or depth > 0:
                    self._activities[label] += 1
                    self._local.depth = depth + 1
                    break
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            with self._condition:
                self._local.depth = max(
                    0, int(getattr(self._local, "depth", 1)) - 1)
                self._activities[label] -= 1
                if self._activities[label] <= 0:
                    self._activities.pop(label, None)
                if self._desired == "paused" and not self._cancelled:
                    self._state = "pausing"
                    self._write_ack_locked()
                self._condition.notify_all()
            # Any exceptional exit may still have caller-owned cleanup outside this
            # context (notably CommandBroker's process-tree termination). Never
            # interpose a pause wait before that cleanup. The caller's next explicit
            # safe checkpoint can acknowledge paused afterward.
            if failed:
                self.poll()
            else:
                self.checkpoint()

    def paused_seconds(self) -> float:
        if not self.enabled:
            return 0.0
        self.start()
        with self._condition:
            current = (
                max(0.0, self._clock() - self._paused_started)
                if self._paused_started is not None else 0.0
            )
            return self._paused_total + current

    def cancel(self) -> None:
        """Disable pause waits immediately so process cleanup can unwind."""
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def close(self) -> None:
        self.cancel()
        self._stop.set()
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=max(0.1, self.poll_seconds * 4))


_singleton_lock = threading.Lock()
_singleton: RuntimeControl | None = None


def get_runtime_control() -> RuntimeControl:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = RuntimeControl.from_env()
        return _singleton.start()


def checkpoint() -> None:
    get_runtime_control().checkpoint()


def activity(kind: str):
    return get_runtime_control().activity(kind)


def paused_seconds() -> float:
    return get_runtime_control().paused_seconds()


def cancel() -> None:
    get_runtime_control().cancel()


def close() -> None:
    get_runtime_control().close()


def _replace_runtime_control_for_tests(control: RuntimeControl | None) -> None:
    """Install a test-local singleton without mutating process environment races."""
    global _singleton
    with _singleton_lock:
        previous = _singleton
        _singleton = control
    if previous is not None and previous is not control:
        previous.close()
