"""Hydrate and compile the pre-2022 academic paper-structure curriculum."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.academic_finetune.structure_corpus import (
    StructureHydrator,
    compile_structure_corpus,
    load_metadata_cache,
    load_replay_corpus,
    load_structure_cache,
    official_fetchers,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--metadata-cache",
        type=Path,
        required=True,
        help="existing CorpusCollector cache (or its raw directory)",
    )
    result.add_argument(
        "--structure-cache",
        type=Path,
        required=True,
        help="separate resumable official-artifact and parsed-structure cache",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help=(
            "pinned local Qwen3.8 model directory; only tokenizer assets are loaded "
            "to reject complete rows over the MLX sequence boundary"
        ),
    )
    result.add_argument(
        "--prose-replay",
        type=Path,
        help="existing v1 plan-to-prose JSONL; required for a trainable 20%% replay mix",
    )
    result.add_argument(
        "--email",
        default="",
        help="contact address sent to official arXiv/NCBI endpoints during hydration",
    )
    result.add_argument(
        "--compile-only",
        action="store_true",
        help="perform no network requests; only compile already-ready structure records",
    )
    result.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry cached unavailable/parse-error records (ready records remain immutable)",
    )
    result.add_argument("--no-balance", action="store_true")
    result.add_argument("--max-sequence-length", type=int, default=448)
    result.add_argument("--minimum-per-split-stratum", type=int, default=8)
    result.add_argument(
        "--allow-nontrainable-pilot",
        action="store_true",
        help="write a smoke corpus even when production coverage/replay gates fail",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    documents = load_metadata_cache(arguments.metadata_cache)
    if not arguments.compile_only:
        if not arguments.email.strip():
            raise SystemExit("--email is required unless --compile-only is used")
        ncbi_api_key = os.environ.get("NCBI_API_KEY", "")
        arxiv_fetcher, pubmed_fetcher = official_fetchers(
            email=arguments.email,
            ncbi_api_key=ncbi_api_key,
        )
        try:
            counts = StructureHydrator(
                arguments.structure_cache,
                arxiv_fetcher=arxiv_fetcher,
                pubmed_fetcher=pubmed_fetcher,
                ncbi_api_key=ncbi_api_key,
                retry_failed=arguments.retry_failed,
            ).hydrate(documents)
        finally:
            arxiv_fetcher.close()
            pubmed_fetcher.close()
        print("structure cache: " + ", ".join(f"{key}={value}" for key, value in counts.items()))

    structures = load_structure_cache(arguments.structure_cache)
    replay = load_replay_corpus(arguments.prose_replay) if arguments.prose_replay else []
    manifest = compile_structure_corpus(
        documents,
        structures,
        output_path=arguments.output,
        replay_records=replay,
        balance_sources=not arguments.no_balance,
        require_trainable_splits=not arguments.allow_nontrainable_pilot,
        minimum_per_split_stratum=arguments.minimum_per_split_stratum,
        max_sequence_length=arguments.max_sequence_length,
        tokenizer_model_path=arguments.tokenizer_model,
    )
    counts = manifest["counts"]
    print(
        f"wrote {counts['examples']} examples from {counts['documents']} documents "
        f"to {arguments.output} ({manifest['corpus_sha256']})"
    )
    if not manifest["trainable"]:
        print("pilot manifest is NOT TRAINABLE: " + "; ".join(manifest["non_trainable_reasons"][:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
