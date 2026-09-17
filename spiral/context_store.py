"""Workspace-confined, content-addressed originals and pageable map records."""
import hashlib
import json
from pathlib import Path
import re


class ContextStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def save(self, text: str, *, kind: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", kind):
            raise ValueError("invalid context record kind")
        raw = text.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.root / ".spiral" / "context"
        directory.resolve().relative_to(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{kind}-{digest}.txt"
        if path.is_symlink():
            raise RuntimeError("saved task context must not be a symlink")
        if not path.exists():
            path.write_bytes(raw)
        elif path.read_bytes() != raw:
            raise RuntimeError("saved task context no longer matches its content identity")
        return path.relative_to(self.root).as_posix()

    def save_map(self, value: dict, *, kind: str = "source-map") -> str:
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= 64_000:
            return self.save(text, kind=kind)
        parts = [{"start": start, "end": min(start + 64_000, len(text)),
                  "path": self.save(text[start:start + 64_000], kind=kind + "-part")}
                 for start in range(0, len(text), 64_000)]
        index = {"schema": "spiral.context-map-pages.v1", "characters": len(text),
                 "sha256": hashlib.sha256(text.encode()).hexdigest(), "parts": parts,
                 "instruction": "Concatenate parts in order for exact JSON; rebuild for current source versions."}
        return self.save(json.dumps(index), kind=kind)
