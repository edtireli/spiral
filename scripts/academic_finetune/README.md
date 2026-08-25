# Academic plan-to-prose corpus

This builder creates a local, inspectable corpus for the optional Spiral
academic-writing adapter. It does **not** train or run a model. Its three
explicitly balanced source strata are `arxiv:hep-th`, `arxiv:hep-ph`, and
`pubmed`.

Every record asks the model to realise a small content plan—context, one to
four sparse semantic-role proposition slots, rhetorical relation, certainty, and citation
slots—as the exact sentence or paragraph written by the paper's authors. The
proposition builder never copies a target five-gram and enforces a conservative
content-overlap gate; its construction method and measured overlap are recorded
per example. Sentence plans retain roles such as subject, result, limitation,
scope and quantity. Paragraph plans contain one compact proposition per target
sentence and accept only 2–8-sentence paragraphs. This teaches argument realisation rather than target copying or
indiscriminately continuing raw papers.

The fixed cutoff is 2021-12-31. arXiv records whose latest content revision is
newer are rejected even when version 1 is older. For PubMed, the cutoff applies
to the article/electronic publication date: MEDLINE `DateRevised` is indexing
maintenance rather than an author prose version, so it is retained as
provenance but does not exclude the article. Explicit erratum, retraction,
correction/republishing, and expression-of-concern publication types are
rejected. Records contain source URLs, extraction method,
content/unit hashes and exact locators. Document/author connected components
are kept in one 90/5/5 split, and exact normalized duplicate prose is removed.
Examples are capped at eight sentences and four paragraphs per document, so a
long full-text article cannot dominate abstracts. Production compilation fails
unless all three source strata and author-safe train/validation/test components
are nonempty. `--allow-nontrainable-pilot` exists only for smoke tests and marks
the resulting manifest `trainable: false`. A production manifest requires at
least two documents and two examples from **each** of hep-th, hep-ph and PubMed
in **each** split; aggregate train/validation/test counts are not sufficient.

## Collect (bounded and resumable)

```sh
python -m scripts.academic_finetune.build_corpus \
  --email you@example.org \
  --pubmed-query 'your biomedical topic[Title/Abstract] AND hasabstract[text]' \
  --maximum-documents-per-source 100 \
  --output academic_corpus.jsonl
```

The default PubMed query covers abstract-bearing research broadly while
excluding editorials, letters, news and comments; pass a domain query to narrow
the biomedical content, not to search for papers *about writing*. By default,
each source's bounded document allowance is divided deterministically across
2012–2021 year partitions (`--year-start`/`--year-end`), rather than filling the
corpus with the newest search page. The manifest reports document and example
year histograms.

The default uses abstracts, which are much smaller and have clean provenance.
`--arxiv-body source` extracts official version-pinned TeX, while
`--arxiv-body pdf` uses the official PDF. `--pubmed-body pmc` adds full text
only when the PubMed record has a PMC identifier. The arXiv client waits at
least three seconds per request; NCBI is kept below its documented unauthenticated
rate. Retries, response bytes, pages and documents are bounded.

For an already downloaded official NLM baseline shard, use
`--pubmed-baseline pubmed25n0001.xml.gz`. No network request is made for that
stratum. Do not reuse a cache with different queries or settings: the builder
detects the configuration hash and fails closed. Output is written atomically;
recompiling the same raw cache produces byte-identical JSONL and manifest.

The combined JSONL embeds `split`; its manifest is written beside it as
`academic_corpus.jsonl.manifest.json`. Source license metadata is retained when
the upstream record exposes it, but the builder does not claim that every
article grants redistribution or commercial training rights. Keep source
artifacts and provenance when auditing or sharing a derived adapter.
