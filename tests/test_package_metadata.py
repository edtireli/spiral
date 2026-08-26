"""Package metadata exported at runtime must match the release manifest source."""

from __future__ import annotations

import tomllib
from pathlib import Path

import spiral


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_version_matches_authoritative_project_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert spiral.__version__ == project["project"]["version"]
