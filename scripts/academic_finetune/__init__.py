"""Deterministic tooling for Spiral's optional academic-writing adapter.

The package deliberately has no import-time network or model side effects.  The
corpus builder and the MLX training harness are separate commands so a cached
corpus can be inspected before any training is started.
"""

CORPUS_SCHEMA = "spiral.academic-plan-prose.v1"
MANIFEST_SCHEMA = "spiral.academic-corpus-manifest.v1"

__all__ = ["CORPUS_SCHEMA", "MANIFEST_SCHEMA"]
