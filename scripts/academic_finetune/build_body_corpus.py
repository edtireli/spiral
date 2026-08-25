"""Build a body-only academic prose corpus from hydrated local artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.academic_finetune.body_corpus import build_offline_body_corpus


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--metadata-cache",
        type=Path,
        required=True,
        help="existing metadata cache paired with the hydrated structure cache",
    )
    result.add_argument(
        "--structure-cache",
        type=Path,
        required=True,
        help="existing ready official TeX/JATS artifact cache",
    )
    result.add_argument(
        "--body-cache",
        type=Path,
        required=True,
        help="destination for body-only SourceDocuments and paragraph provenance",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help=(
            "pinned local Qwen3.8 model directory; only tokenizer assets are "
            "loaded and model weights remain unopened"
        ),
    )
    result.add_argument("--max-sequence-length", type=int, default=448)
    result.add_argument("--no-balance", action="store_true")
    result.add_argument(
        "--allow-nontrainable-pilot",
        action="store_true",
        help="allow a smoke corpus below the production split-coverage gates",
    )
    result.add_argument("--minimum-documents-per-split-stratum", type=int, default=8)
    result.add_argument("--minimum-examples-per-split-stratum", type=int, default=8)
    result.add_argument("--minimum-components-per-split-stratum", type=int, default=8)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    manifest = build_offline_body_corpus(
        metadata_cache=arguments.metadata_cache,
        structure_cache=arguments.structure_cache,
        body_cache=arguments.body_cache,
        output_path=arguments.output,
        balance_sources=not arguments.no_balance,
        require_trainable_splits=not arguments.allow_nontrainable_pilot,
        minimum_documents_per_split_stratum=(
            arguments.minimum_documents_per_split_stratum
        ),
        minimum_examples_per_split_stratum=(
            arguments.minimum_examples_per_split_stratum
        ),
        minimum_components_per_split_stratum=(
            arguments.minimum_components_per_split_stratum
        ),
        max_sequence_length=arguments.max_sequence_length,
        tokenizer_model_path=arguments.tokenizer_model,
    )
    counts = manifest["counts"]
    attestation = manifest["body_only_attestation"]
    print(
        f"wrote {counts['examples']} verified main-body examples to "
        f"{arguments.output} ({manifest['corpus_sha256']})"
    )
    print(
        "body-only attestation: "
        f"ratio={attestation['body_only_example_ratio']:.6f}, "
        f"abstract_examples={attestation['abstract_examples']}, "
        f"section_provenance="
        f"{attestation['every_example_has_section_paragraph_provenance']}"
    )
    temporal = attestation.get("jats_temporal_attestation", {})
    if temporal:
        print(
            "JATS temporal attestation: "
            f"checked={temporal['artifacts_checked']}, "
            f"eligible={temporal['eligible_artifacts']}, "
            f"rejected={temporal['rejected_artifacts']}, "
            f"cutoff={temporal['cutoff']}"
        )
    if not manifest["trainable"]:
        reasons = manifest.get("non_trainable_reasons", ())
        print("pilot manifest is NOT TRAINABLE: " + "; ".join(reasons[:5]))
    token_gate = manifest["gates"]["exact_training_token_gate"]
    print(
        "exact token gate: "
        f"measured={token_gate['candidates_measured']}, "
        f"rejected={token_gate['candidates_rejected']}, "
        f"largest_accepted={token_gate['largest_accepted_tokens']}/"
        f"{token_gate['max_sequence_length']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
