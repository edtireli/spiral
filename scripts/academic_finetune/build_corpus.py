#!/usr/bin/env python3
"""Collect and compile the pre-2022 Spiral academic-writing corpus."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from scripts.academic_finetune.corpus import CollectionConfig, CorpusCollector, compile_corpus
from scripts.academic_finetune.sources import (
    ArxivSource,
    PoliteFetcher,
    PubMedBaselineSource,
    PubMedSource,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("academic_corpus.jsonl"))
    result.add_argument("--cache", type=Path, default=Path(".academic-corpus-cache"))
    result.add_argument("--email", required=True, help="contact address sent to arXiv/NCBI")
    result.add_argument(
        "--pubmed-query",
        default="hasabstract[text] NOT (editorial[pt] OR letter[pt] OR news[pt] OR comment[pt])",
        help="biomedical topic expression; an abstract/article-type filter is applied by default",
    )
    result.add_argument("--pubmed-baseline", type=Path)
    result.add_argument("--arxiv-body", choices=("none", "source", "pdf"), default="none")
    result.add_argument("--pubmed-body", choices=("abstract", "pmc"), default="abstract")
    result.add_argument("--maximum-documents-per-source", type=int, default=100)
    result.add_argument("--maximum-scanned-per-source", type=int, default=500)
    result.add_argument("--page-size", type=int, default=25)
    result.add_argument("--year-start", type=int, default=2012)
    result.add_argument("--year-end", type=int, default=2021)
    result.add_argument("--no-balance", action="store_true")
    result.add_argument(
        "--allow-nontrainable-pilot",
        action="store_true",
        help="write a smoke-test corpus even if an author-safe held-out split is empty",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    cutoff = date(2021, 12, 31)
    if not 1991 <= arguments.year_start <= arguments.year_end <= cutoff.year:
        raise SystemExit("year range must satisfy 1991 <= start <= end <= 2021")
    years = range(arguments.year_start, arguments.year_end + 1)
    user_agent = f"SpiralAcademicCorpus/1.0 (mailto:{arguments.email})"
    # arXiv asks clients to leave three seconds between API/download requests.
    with PoliteFetcher(user_agent=user_agent, min_interval_seconds=3.0) as arxiv_fetcher:
        sources = [
            ArxivSource(
                category,
                cutoff=cutoff,
                fetcher=arxiv_fetcher,
                body_mode=arguments.arxiv_body,
                year=year,
            )
            for category in ("hep-th", "hep-ph")
            for year in years
        ]
        pubmed_fetcher: PoliteFetcher | None = None
        if arguments.pubmed_baseline:
            sources.append(PubMedBaselineSource(arguments.pubmed_baseline, cutoff=cutoff))
        else:
            # NCBI permits at most 3 requests/s without an API key; 0.4s is conservative.
            api_key = os.environ.get("NCBI_API_KEY", "")
            pubmed_fetcher = PoliteFetcher(
                user_agent=user_agent,
                min_interval_seconds=0.12 if api_key else 0.4,
            )
            sources.extend(
                PubMedSource(
                    arguments.pubmed_query,
                    cutoff=cutoff,
                    email=arguments.email,
                    fetcher=pubmed_fetcher,
                    api_key=api_key,
                    body_mode=arguments.pubmed_body,
                    year=year,
                )
                for year in years
            )
        try:
            collector = CorpusCollector(
                arguments.cache,
                CollectionConfig(
                    cutoff=cutoff,
                    maximum_documents_per_source=arguments.maximum_documents_per_source,
                    page_size=arguments.page_size,
                    maximum_scanned_per_source=arguments.maximum_scanned_per_source,
                ),
            )
            documents = collector.collect(sources)
            manifest = compile_corpus(
                documents,
                output_path=arguments.output,
                cutoff=cutoff,
                balance_sources=not arguments.no_balance,
                require_trainable_splits=not arguments.allow_nontrainable_pilot,
            )
        finally:
            if pubmed_fetcher is not None:
                pubmed_fetcher.close()
    counts = manifest["counts"]
    print(
        f"wrote {counts['examples']} examples from {counts['documents']} documents "
        f"to {arguments.output} ({manifest['corpus_sha256']})"
    )
    if not manifest["trainable"]:
        print("pilot manifest is NOT TRAINABLE: " + "; ".join(manifest["non_trainable_reasons"][:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
