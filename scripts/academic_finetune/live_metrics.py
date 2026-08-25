#!/usr/bin/env python3
"""Regenerate Spiral academic training's dependency-free local loss view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from .training_support import (
        TRAINING_METRIC_SCHEMA,
        HarnessError,
        render_training_metrics_html,
    )
except ImportError:  # direct script execution
    from training_support import (  # type: ignore
        TRAINING_METRIC_SCHEMA,
        HarnessError,
        render_training_metrics_html,
    )


def load_metric_records(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HarnessError(f"cannot read training metrics {path}: {exc}") from exc
    records = []
    keys = set()
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"metrics line {line_number} is corrupt: {exc}") from exc
        if not isinstance(record, dict) or record.get("schema_version") != TRAINING_METRIC_SCHEMA:
            raise HarnessError(f"metrics line {line_number} has the wrong schema")
        key = (record.get("event"), record.get("iteration"))
        if key in keys:
            raise HarnessError(f"metrics contain duplicate event/iteration {key}")
        keys.add(key)
        records.append(record)
    return records


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Render local Spiral academic loss curves")
    result.add_argument("--metrics", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        records = load_metric_records(arguments.metrics)
        modes = {str(record.get("mode") or "") for record in records}
        if len(modes) != 1 or next(iter(modes)) not in {
            "production", "smoke_only", "feasibility_only",
        }:
            raise HarnessError(
                "metrics must attest exactly one production/smoke_only/feasibility_only mode")
        mode = next(iter(modes))
        render_training_metrics_html(records, arguments.output, mode=mode)
        print(arguments.output.resolve())
        return 0
    except HarnessError as exc:
        print(f"academic metrics: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
