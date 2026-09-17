"""Optional local vocabulary-only source counts for production planning.

These counts do not measure the chat template, backend allocation or model quality.
Generation still needs its provider's exact context admission. No model weights
are loaded and no context/model setting is enlarged by this helper.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from spiral.runtime_control import checkpoint as runtime_checkpoint
from spiral.runtime_control import activity


class ContextMeasurementUnavailable(RuntimeError):
    pass


def planning_inventory_view(inventory: dict, reference: str, *, max_bytes: int = 12000) -> dict:
    """Keep a large measured inventory navigable without flooding the planner."""
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
    view = {**inventory, "full_inventory": reference}
    if size(view) <= max_bytes:
        return view
    files = inventory["files"]
    view = {key: value for key, value in inventory.items() if key != "files"}
    view.update(full_inventory=reference, files=[], directories=[],
                view="bounded directory summary and selected file costs; remaining measured records are in full_inventory",
                indexed_file_count=len(files), total_source_tokens=sum(row["tokens"] for row in files),
                omitted_view_files=len(files), omitted_view_directories=0)
    groups = {}
    for row in files:
        directory = row["path"].split("/", 1)[0] if "/" in row["path"] else "."
        count, tokens = groups.get(directory, (0, 0))
        groups[directory] = (count + 1, tokens + row["tokens"])
    view["omitted_view_directories"] = len(groups)
    for path, (count, tokens) in sorted(groups.items()):
        row = {"path": path, "files": count, "source_tokens": tokens}
        candidate = {**view, "directories": view["directories"] + [row]}
        if size(candidate) > max_bytes * 2 // 3:
            break
        view["directories"].append(row)
        view["omitted_view_directories"] -= 1
    for row in sorted(files, key=lambda item: (-item["tokens"], item["path"])):
        candidate = {**view, "files": view["files"] + [row]}
        if size(candidate) > max_bytes:
            break
        view["files"].append(row)
        view["omitted_view_files"] -= 1
    if size(view) > max_bytes:
        raise ContextMeasurementUnavailable("source inventory metadata exceeds planning view budget")
    return view


def _identity(path: Path):
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class SourceTokenizer:
    def __init__(self, binary: Path, model_path: Path, model: str, *,
                 timeout: float = 30.0, checkpoint=runtime_checkpoint):
        self.binary = binary.resolve(strict=True)
        self.model_path = model_path.resolve(strict=True)
        self.model = model
        if not model or not self.binary.is_file() or not os.access(self.binary, os.X_OK):
            raise ContextMeasurementUnavailable("local tokenizer is not executable or has no selected model")
        with self.model_path.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ContextMeasurementUnavailable("selected tokenizer source is not GGUF")
        self._binding = (_identity(self.binary), _identity(self.model_path))
        self.identity = hashlib.sha256(json.dumps(self._binding).encode()).hexdigest()
        self.timeout = min(120.0, max(0.1, timeout))
        self.checkpoint = checkpoint
        self._cache: dict[str, int] = {}

    def _check(self):
        self.checkpoint()
        if self._binding != (_identity(self.binary), _identity(self.model_path)):
            raise ContextMeasurementUnavailable("selected tokenizer source or executable changed")

    def count(self, texts: list[str]) -> list[int]:
        # A pause is acknowledged only after the owned vocabulary process has
        # exited; never leave it running behind an acknowledged idle state.
        with activity("source-tokenization"):
            return self._count(texts)

    def _count(self, texts: list[str]) -> list[int]:
        self._check()
        encoded = [text.encode("utf-8") for text in texts]
        if (len(encoded) > 4096 or sum(map(len, encoded)) > 32 * 1024 * 1024
                or any(len(raw) > 2 * 1024 * 1024 for raw in encoded)):
            raise ContextMeasurementUnavailable("source token measurement exceeds its bounded inventory")
        keys = [hashlib.sha256(raw).hexdigest() for raw in encoded]
        pending = dict((key, raw) for key, raw in zip(keys, encoded) if key not in self._cache)
        if not pending:
            return [self._cache[key] for key in keys]
        payload = b"".join(str(len(raw)).encode() + b"\n" + raw for raw in pending.values())
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("DYLD_", "LD_"))}
        output_file = tempfile.TemporaryFile()
        error_file = tempfile.TemporaryFile()
        input_file = tempfile.TemporaryFile()
        try:
            # A vocabulary load can exceed the poll interval before reading
            # stdin. Retrying communicate(input=None) on Python 3.11 can stop
            # writing a partially sent pipe forever. A bounded private file
            # supplies the complete batch independently of polling/cancellation.
            input_file.write(payload)
            input_file.seek(0)
            process = subprocess.Popen([str(self.binary), str(self.model_path)],
                stdin=input_file, stdout=output_file, stderr=error_file,
                env=environment, start_new_session=True)
        except BaseException:
            input_file.close()
            output_file.close()
            error_file.close()
            raise
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                self._check()
                if os.fstat(output_file.fileno()).st_size > 512_000 or os.fstat(error_file.fileno()).st_size > 65536:
                    raise ContextMeasurementUnavailable("local tokenizer exceeded its output bound")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ContextMeasurementUnavailable("source token measurement exceeded its time bound")
                try:
                    process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            self._check()
            output_file.seek(0)
            error_file.seek(0)
            output, error = output_file.read(512_001), error_file.read(65537)
            if process.returncode or len(output) > 512_000 or len(error) > 65536:
                raise ContextMeasurementUnavailable("local tokenizer failed or exceeded its output bound")
            try:
                def pairs(items):
                    value = {}
                    for key, item in items:
                        if key in value:
                            raise ValueError("duplicate measurement field")
                        value[key] = item
                    return value
                rows = [json.loads(line, object_pairs_hook=pairs) for line in output.splitlines()]
                terminal = rows.pop()
                if (not isinstance(terminal, dict) or type(terminal.get("count")) is not int
                        or terminal != {"schema": "spiral.tokenizer.v1", "scope": "raw_text_no_bos_no_special", "count": len(pending)}):
                    raise ValueError("invalid terminal receipt")
                if len(rows) != len(pending):
                    raise ValueError("missing source count")
                measured = {}
                for index, (key, row) in enumerate(zip(pending, rows)):
                    if (not isinstance(row, dict) or set(row) != {"index", "tokens"} or row.get("index") != index
                            or type(row.get("index")) is not int
                            or type(row.get("tokens")) is not int or not 0 <= row["tokens"] <= 8_388_608):
                        raise ValueError("invalid source count")
                    measured[key] = row["tokens"]
            except (ValueError, TypeError, IndexError, AttributeError) as exc:
                raise ContextMeasurementUnavailable("invalid source-token measurement receipt") from exc
            self._cache.update(measured)  # no partial batch is cached
            result = [self._cache[key] for key in keys]
            while len(self._cache) > 8192:
                self._cache.pop(next(iter(self._cache)))
            return result
        finally:
            if process.poll() is None:
                process.kill()  # only this vocabulary helper, never a model service
            process.wait()
            input_file.close()
            output_file.close()
            error_file.close()

    def inventory(self, sources: dict, dependencies: dict | None = None) -> dict:
        paths = sorted(sources)
        counts = self.count([sources[path].text for path in paths])
        return {"schema": "spiral.measured-sources.v1", "model": self.model,
                "tokenizer_identity": self.identity,
                "count_scope": "raw source text without BOS or special-token parsing; not the rendered chat template",
                "execution_admitted": False, "model_quality_verified": False,
                "files": [{"path": path, "tokens": count, "sha256": sources[path].sha256,
                           "dependencies": (dependencies or {}).get(path, [])}
                          for path, count in zip(paths, counts)]}
