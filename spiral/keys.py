"""Live keyboard control during a run — Shift+Tab cycles autonomy modes.

    auto  → spiral runs unattended (default)
    step  → confirm at every task boundary (enter run · s skip · a auto · q quit)

The watcher reads raw keys on a background thread (cbreak-style: ICANON+ECHO
off, ISIG kept so Ctrl-C still works). Single-key prompts are answered through
the same watcher, so no terminal-mode juggling with input(). No-ops when stdin
isn't a TTY.
"""
from __future__ import annotations

import os
import queue
import select
import sys
import threading
import atexit
from collections.abc import Callable

SHIFT_TAB = "\x1b[Z"
_ACTIVE: set["Watcher"] = set()


def _restore_terminals() -> None:
    for watcher in list(_ACTIVE):
        watcher.stop()


atexit.register(_restore_terminals)


class Watcher:
    def __init__(self) -> None:
        self.mode = "auto"
        self.keys: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_attrs = None
        self._buf = b""
        self.enabled = sys.stdin.isatty()
        self._handlers: dict[str, list[Callable[[], None]]] = {}

    # -- byte stream → events (separated for testability) ----------------------
    def feed(self, data: bytes) -> None:
        self._buf += data
        while self._buf:
            if self._buf.startswith(b"\x1b"):
                if self._buf.startswith(b"\x1b[Z"):
                    self.mode = "step" if self.mode == "auto" else "auto"
                    self.keys.put("mode")
                    self._buf = self._buf[3:]
                    continue
                if len(self._buf) < 3:
                    return  # possibly incomplete escape sequence — wait for more
                # unknown escape sequence — drop the ESC and move on
                self._buf = self._buf[1:]
                continue
            ch = chr(self._buf[0])
            self._buf = self._buf[1:]
            if ch in self._handlers:
                for fn in list(self._handlers[ch]):
                    try:
                        fn()
                    except Exception:
                        pass
                continue
            self.keys.put(ch)

    def on_key(self, ch: str, fn: Callable[[], None]) -> None:
        """Register a live hotkey consumed before prompt handling."""
        if ch:
            self._handlers.setdefault(ch[0], []).append(fn)

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "Watcher":
        if not self.enabled:
            return self
        import termios

        fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)  # raw-ish; ISIG stays on
        termios.tcsetattr(fd, termios.TCSANOW, new)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        _ACTIVE.add(self)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._old_attrs is not None:
            import termios

            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old_attrs)
            except (OSError, termios.error):
                pass
            self._old_attrs = None
        _ACTIVE.discard(self)

    def _loop(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            r, _, _ = select.select([fd], [], [], 0.15)
            if r:
                try:
                    data = os.read(fd, 16)
                except OSError:
                    break
                if data:
                    self.feed(data)

    # -- single-key prompt (used at step-mode task boundaries) --------------------
    def ask(self, timeout: float | None = None) -> str:
        """Next real keypress ('mode' events are transparent — they just retoggle)."""
        while True:
            try:
                k = self.keys.get(timeout=timeout)
            except queue.Empty:
                return ""
            if k != "mode":
                return k

    def drain(self) -> None:
        while not self.keys.empty():
            try:
                self.keys.get_nowait()
            except queue.Empty:
                break
