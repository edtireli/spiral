"""Deterministic quality gates for an autonomous literature corpus.

No finite search can prove that an open-world literature corpus is complete.  What we
*can* verify is that a documented retrieval protocol ran successfully, reached enough
on-topic primary material, and approached a stable citation frontier.  This module
turns those observable facts into explicit discovery/novelty readiness gates.

The scores here are retrieval diagnostics, not scientific judgments.  Models may use
them to decide what to read next, but cannot override a failed gate by assertion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


_STOP = {
    "about", "above", "across", "after", "against", "allow", "also", "among",
    "arbitrary", "because", "before", "being", "below", "between", "both", "build",
    "classify", "complete", "compute", "could", "derive", "determine", "every",
    "exact", "find", "fixed", "from", "general", "give", "have", "identify", "into",
    "known", "make", "model", "most", "only", "other", "paper", "possible", "present",
    "produce", "provide", "question", "recover", "release", "research", "result",
    "results", "search", "select", "should", "state", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "through", "under", "using", "valid",
    "verify", "what", "when", "where", "which", "whose", "will", "with", "within",
    "would", "write",
}


@dataclass(frozen=True)
class CoveragePolicy:
    """Conservative minima for beginning discovery and asserting novelty."""

    min_papers: int = 10
    min_usable_texts: int = 6
    min_relevant_papers: int = 5
    min_relevant_usable_primary_texts: int = 4
    min_unique_queries: int = 3
    min_healthy_searches: int = 2
    min_relevant_query_families: int = 2
    min_topic_term_coverage: float = 0.45
    min_graph_success_rate: float = 0.60


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", folded)]


def topic_terms(topic: str, *, limit: int = 20) -> list[str]:
    """Extract stable domain terms from a prompt without asking a model."""

    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, word in enumerate(_tokens(topic)):
        word = word.strip("-")
        if len(word) < 4 or word in _STOP:
            continue
        counts[word] = counts.get(word, 0) + 1
        first.setdefault(word, i)
    ranked = sorted(
        counts,
        key=lambda w: (-(counts[w] * (1.0 + min(len(w), 14) / 14.0)), first[w]),
    )
    return ranked[:limit]


def _paper_blob(paper) -> str:
    title = getattr(paper, "title", "") or ""
    abstract = getattr(paper, "abstract", "") or ""
    text = getattr(paper, "text", "") or ""
    # Titles and abstracts are high-signal; a bounded body prevents a long review from
    # dominating lexical diagnostics merely by size.
    return " ".join([title, title, title, abstract, abstract, text[:40_000]])


def _relevance(topic: str, papers: list) -> tuple[list[str], list[dict], float]:
    terms = topic_terms(topic)
    if not terms or not papers:
        return terms, [], 0.0
    token_sets = [set(_tokens(_paper_blob(p))) for p in papers]
    n = len(token_sets)
    df = {t: sum(1 for toks in token_sets if t in toks) for t in terms}
    weights = {t: 1.0 + math.log((n + 1.0) / (df[t] + 1.0)) for t in terms}
    denominator = sum(weights.values()) or 1.0
    rows = []
    for paper, toks in zip(papers, token_sets):
        matched = [t for t in terms if t in toks]
        score = sum(weights[t] for t in matched) / denominator
        relevant = len(matched) >= 2 and score >= 0.16
        rows.append({
            "id": getattr(paper, "bare_id", getattr(paper, "arxiv_id", "")),
            "title": getattr(paper, "title", ""),
            "score": round(score, 4),
            "matched_terms": matched,
            "relevant": relevant,
        })
    covered = sum(1 for t in terms if df[t] > 0) / max(1, len(terms))
    rows.sort(key=lambda row: (-row["score"], row["id"]))
    return terms, rows, covered


def _result_ids(retrieval: dict) -> set[str]:
    """Collect identifiers from a direct or category-fallback retrieval report."""

    ids = {
        str(value).replace("arXiv:", "").split("v")[0]
        for value in (retrieval.get("result_ids") or [])
        if str(value).strip()
    }
    for key in ("restricted", "fallback"):
        child = retrieval.get(key)
        if isinstance(child, dict):
            ids.update(_result_ids(child))
    return ids


def _incidence_coverage(query_results: dict[str, set[str]]) -> dict:
    """Estimate unseen records from overlap among independent query result lists.

    This is the incidence analogue of a Chao lower-bound estimator. Search-engine
    rankings violate capture-recapture's equal-catchability assumption, so the result
    is intentionally diagnostic and never used as a pass/fail gate.
    """

    sets = [values for values in query_results.values() if values]
    frequencies: dict[str, int] = {}
    for values in sets:
        for identifier in values:
            frequencies[identifier] = frequencies.get(identifier, 0) + 1
    observed = len(frequencies)
    singletons = sum(1 for count in frequencies.values() if count == 1)
    doubletons = sum(1 for count in frequencies.values() if count == 2)
    if doubletons:
        unseen = (singletons * singletons) / (2.0 * doubletons)
    else:
        unseen = (singletons * max(0, singletons - 1)) / 2.0
    estimate = observed + unseen
    coverage = observed / estimate if estimate else 0.0
    valid = len(sets) >= 3 and observed >= 5
    return {
        "diagnostic_valid": valid,
        "query_lists": len(sets),
        "observed_unique_records": observed,
        "singletons": singletons,
        "doubletons": doubletons,
        "estimated_unseen_lower_bound": round(unseen, 2),
        "estimated_total_lower_bound": round(estimate, 2),
        "estimated_observed_fraction": round(coverage, 4),
        "used_as_gate": False,
        "caveat": (
            "Ranked query results are heterogeneous, non-random samples; this is a "
            "lower-bound diagnostic, not proof of corpus completeness."
        ),
    }


def query_family_count(queries: set[str] | list[str]) -> int:
    """Count meaningfully different query formulations, not string variants."""

    families: list[set[str]] = []
    for query in sorted(queries):
        terms = set(_tokens(query)) or {query}
        duplicate = False
        for held in families:
            similarity = len(terms & held) / max(1, len(terms | held))
            if similarity >= 0.75:
                duplicate = True
                break
        if not duplicate:
            families.append(terms)
    return len(families)


def _search_metrics(research_map: dict, relevant_ids: set[str] | None = None) -> dict:
    searches = research_map.get("searches") or []
    unique = {" ".join(str(s.get("query", "")).lower().split()) for s in searches}
    unique.discard("")
    healthy = 0
    healthy_queries: set[str] = set()
    relevant_healthy_queries: set[str] = set()
    failed = 0
    unknown = 0
    result_count = 0
    query_results: dict[str, set[str]] = {}
    for search in searches:
        retrieval = search.get("retrieval") or {}
        normalised_query = " ".join(str(search.get("query", "")).lower().split())
        ok = retrieval.get("source_ok")
        if ok is True:
            healthy += 1
            healthy_queries.add(normalised_query)
            result_ids = _result_ids(retrieval)
            if normalised_query:
                query_results.setdefault(normalised_query, set()).update(
                    result_ids)
                if relevant_ids and result_ids & relevant_ids:
                    relevant_healthy_queries.add(normalised_query)
        elif ok is False:
            failed += 1
        else:
            unknown += 1
        result_count += int(retrieval.get("result_count") or 0)
    return {
        "attempts": len(searches),
        "queries": sorted(unique),
        "unique_queries": len(unique),
        "healthy_attempts": healthy,
        "healthy_unique_queries": len({q for q in healthy_queries if q}),
        "healthy_query_families": query_family_count(
            {q for q in healthy_queries if q}),
        "relevant_healthy_queries": len(relevant_healthy_queries),
        "relevant_query_families": query_family_count(relevant_healthy_queries),
        "failed_attempts": failed,
        "unknown_legacy_attempts": unknown,
        "results_seen": result_count,
        "incidence_coverage": _incidence_coverage(query_results),
    }


def _graph_metrics(research_map: dict, have_ids: set[str], policy: CoveragePolicy,
                   relevant_ids: set[str] | None = None) -> dict:
    rounds = research_map.get("graph_rounds") or []
    requests = successes = failures = 0
    valid_rounds = 0
    unresolved: list[dict] = []
    successful_current_seeds: set[str] = set()
    closed_current_seeds: set[str] = set()
    delivered_ids: set[str] = set()
    for report in rounds:
        health = report.get("health") or report.get("graph_health") or {}
        requests += int(health.get("requests") or 0)
        successes += int(health.get("successful_requests") or 0)
        failures += int(health.get("failed_requests") or 0)
        if health.get("coverage_valid") is True:
            valid_rounds += 1
        delivered_ids.update(
            str(added).replace("arXiv:", "").split("v")[0]
            for added in (report.get("added") or [])
        )
        successful = {
            str(seed).replace("arXiv:", "").split("v")[0]
            for seed in (health.get("successful_seeds") or [])
        } & have_ids
        successful_current_seeds.update(successful)
        if (health.get("coverage_valid") is True
                and (report.get("batch_frontier_closed") is True
                     or report.get("saturated") is True)
                and not report.get("frontier_truncated")
                and not (report.get("unresolved_holes_after_round") or [])):
            closed_current_seeds.update(successful)
        unresolved = [
            h for h in (report.get("holes") or [])
            if str(h.get("id") or "").split("v")[0] not in have_ids
        ]
    rate = successes / requests if requests else 0.0
    latest = rounds[-1] if rounds else {}
    latest_health = latest.get("health") or latest.get("graph_health") or {}
    latest_requests = int(latest_health.get("requests") or 0)
    latest_successes = int(latest_health.get("successful_requests") or 0)
    latest_rate = latest_successes / latest_requests if latest_requests else 0.0
    latest_size = latest.get("corpus_size")
    latest_matches_corpus = latest_size is None or int(latest_size) == len(have_ids)
    source_healthy = bool(
        latest_matches_corpus
        and latest_health.get("coverage_valid") is True
        and latest_requests
        and latest_rate >= policy.min_graph_success_rate
    )
    seed_count = len(have_ids)
    successful_seed_fraction = (
        len(successful_current_seeds) / seed_count if seed_count else 0.0)
    closed_seed_fraction = len(closed_current_seeds) / seed_count if seed_count else 0.0
    all_current_seeds_observed = bool(seed_count and len(successful_current_seeds) == seed_count)
    all_current_seed_frontiers_closed = bool(
        seed_count and len(closed_current_seeds) == seed_count)
    return {
        "rounds": len(rounds),
        "valid_rounds": valid_rounds,
        "requests": requests,
        "successful_requests": successes,
        "failed_requests": failures,
        "success_rate": round(rate, 4),
        "latest_success_rate": round(latest_rate, 4),
        "latest_matches_corpus": latest_matches_corpus,
        "healthy": source_healthy,
        "current_seed_count": seed_count,
        "successful_current_seed_count": len(successful_current_seeds),
        "successful_seed_fraction": round(successful_seed_fraction, 4),
        "closed_current_seed_count": len(closed_current_seeds),
        "closed_seed_fraction": round(closed_seed_fraction, 4),
        "all_current_seeds_observed": all_current_seeds_observed,
        "all_current_seed_frontiers_closed": all_current_seed_frontiers_closed,
        "saturated": bool(source_healthy and all_current_seed_frontiers_closed),
        "unresolved_cocitation_holes": len(unresolved),
        "top_unresolved_holes": unresolved[:10],
        "delivered_count": len(delivered_ids & have_ids),
        "relevant_delivered_count": len(
            delivered_ids & have_ids & (relevant_ids or set())),
    }


def reading_metrics(notes_root: str | Path | None) -> dict:
    if not notes_root:
        return {"paper_notes": 0, "grounded_paper_notes": 0,
                "deep_notes": 0, "grounded_deep_notes": 0}
    root = Path(notes_root)
    paper_paths = list((root / "papers").glob("*.json")) if (root / "papers").is_dir() else []
    deep_paths = list((root / "deep").glob("*.json")) if (root / "deep").is_dir() else []

    def read_rows(paths: list[Path], *, deep: bool) -> tuple[set[str], set[str]]:
        seen: set[str] = set()
        grounded_ids: set[str] = set()
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pid = str(data.get("arxiv_id") or path.stem).replace("arXiv:", "").split("v")[0]
            seen.add(pid)
            evidence = data.get("grounded_evidence") if deep else data.get("evidence")
            if data.get("grounded") and evidence:
                grounded_ids.add(pid)
        return seen, grounded_ids

    paper_ids, grounded_papers = read_rows(paper_paths, deep=False)
    deep_ids, grounded_deep = read_rows(deep_paths, deep=True)
    return {
        "paper_note_files": len(paper_paths),
        "paper_notes": len(paper_ids),
        "grounded_paper_notes": len(grounded_papers),
        "deep_note_files": len(deep_paths),
        "deep_notes": len(deep_ids),
        "grounded_deep_notes": len(grounded_deep),
    }


def corpus_quality_report(topic: str, papers, research_map: dict, *,
                          notes_root: str | Path | None = None,
                          policy: CoveragePolicy | None = None) -> dict:
    """Compute an auditable readiness report from local files and retrieval telemetry."""

    policy = policy or CoveragePolicy()
    paper_list = list(papers)
    ids = {str(getattr(p, "bare_id", getattr(p, "arxiv_id", ""))) for p in paper_list}
    terms, relevance_rows, term_coverage = _relevance(topic, paper_list)
    relevant = [row for row in relevance_rows if row["relevant"]]
    usable = [p for p in paper_list if len((getattr(p, "text", "") or "").strip()) >= 1_000]
    primary = [
        p for p in paper_list
        if getattr(p, "tex_path", "") or getattr(p, "pdf_path", "")
        or str(getattr(p, "body_source", "")) in {"tex", "pdf"}
    ]
    primary_ids = {id(p) for p in primary}
    usable_primary = [p for p in usable if id(p) in primary_ids]
    relevant_ids = {str(row.get("id") or "") for row in relevant}
    relevant_usable_primary = [
        p for p in usable_primary
        if str(getattr(p, "bare_id", getattr(p, "arxiv_id", ""))) in relevant_ids
    ]
    searches = _search_metrics(research_map, relevant_ids)
    graph = _graph_metrics(research_map, ids, policy, relevant_ids)
    reading = reading_metrics(notes_root)

    # The route check asks whether relevant material reached the corpus through more
    # than one independent retrieval path. Keyword-query families are one kind of
    # route; the citation graph is another, genuinely independent one (snowballed
    # references reach papers no keyword search returned). Counting it prevents a
    # run whose growth came through the graph from being blocked forever by a check
    # that only ever looked at keyword telemetry.
    graph_route = 1 if graph["relevant_delivered_count"] >= 2 else 0
    relevant_route_count = searches["relevant_query_families"] + graph_route

    small_field_exception = bool(
        6 <= len(paper_list) < policy.min_papers
        and graph["saturated"]
        and len(usable_primary) >= max(4, math.ceil(0.70 * len(paper_list)))
        and len(relevant) >= max(4, math.ceil(0.60 * len(paper_list)))
    )

    discovery_checks = {
        "paper_count_or_saturated_small_field": (
            len(paper_list) >= policy.min_papers or small_field_exception),
        "usable_primary_texts": len(usable_primary) >= policy.min_usable_texts,
        "topically_relevant_papers": len(relevant) >= policy.min_relevant_papers,
        "relevant_usable_primary_texts": (
            len(relevant_usable_primary) >= policy.min_relevant_usable_primary_texts),
        "query_diversity": (
            searches["healthy_query_families"] >= policy.min_unique_queries),
        "retrieval_health": searches["healthy_attempts"] >= policy.min_healthy_searches,
        "relevant_query_routes": (
            relevant_route_count >= policy.min_relevant_query_families),
        "topic_term_coverage": term_coverage >= policy.min_topic_term_coverage,
    }
    discovery_ready = all(discovery_checks.values())
    novelty_checks = {
        **discovery_checks,
        "citation_graph_health": graph["healthy"],
        "citation_frontier_saturated": graph["saturated"],
    }
    novelty_ready = all(novelty_checks.values())
    blocking = [name for name, ok in discovery_checks.items() if not ok]
    novelty_blocking = [name for name, ok in novelty_checks.items() if not ok]
    warnings = []
    if searches["unknown_legacy_attempts"]:
        warnings.append("some legacy searches have no source-health telemetry")
    if graph["rounds"] and not graph["healthy"]:
        warnings.append("citation graph is unavailable or too incomplete to establish saturation")
    if len(primary) < len(usable):
        warnings.append("some usable records are abstract-only rather than downloaded primary text")

    return {
        "schema_version": 1,
        "paper_count": len(paper_list),
        "usable_text_count": len(usable),
        "primary_text_count": len(primary),
        "usable_primary_text_count": len(usable_primary),
        "relevant_usable_primary_text_count": len(relevant_usable_primary),
        "relevant_paper_count": len(relevant),
        "relevance_ratio": round(len(relevant) / max(1, len(paper_list)), 4),
        "topic_terms": terms,
        "topic_term_coverage": round(term_coverage, 4),
        "top_relevant_papers": relevance_rows[:12],
        "relevant_route_count": relevant_route_count,
        "search": searches,
        "graph": graph,
        "reading": reading,
        "small_field_exception": small_field_exception,
        "policy": policy.__dict__,
        "discovery_checks": discovery_checks,
        "novelty_checks": novelty_checks,
        "discovery_ready": discovery_ready,
        "novelty_ready": novelty_ready,
        "blocking_reasons": blocking,
        "novelty_blocking_reasons": novelty_blocking,
        "warnings": warnings,
    }


# Discovery checks that measure the health/diversity of retrieval INSTRUMENTS rather
# than the content of the corpus. When the instruments are provably dead (rate-limited
# arXiv, a failing citation-graph API) these checks can never change no matter how many
# rounds run — they are the checks a stall may degrade. Content checks (enough relevant
# usable primary text on topic) are never overridden.
INSTRUMENT_CHECKS = frozenset({
    "query_diversity", "retrieval_health", "relevant_query_routes",
})


def apply_stall_override(report: dict, *, stalled_rounds: int, patience: int = 3,
                         instruments_dead: bool = False,
                         evidence: dict | None = None) -> dict:
    """Degrade instrument-dependent discovery checks after a measured stall.

    A soundness gate must not become a liveness bug. If retrieval instruments are
    demonstrably dead and the corpus has been flat for ``patience`` rounds while ONLY
    instrument checks fail (every content check passes), discovery proceeds with the
    override RECORDED in the report — an explicit, auditable limitation, never a
    silent pass. Novelty gates are deliberately NOT overridden: absence of prior art
    cannot be asserted on the word of a dead instrument, so a paper still waits for a
    live literature check even when discovery/verification is allowed to continue."""

    if report.get("discovery_ready") or stalled_rounds < patience or not instruments_dead:
        return report
    blocking = [str(b) for b in (report.get("blocking_reasons") or [])]
    if not blocking or not set(blocking) <= INSTRUMENT_CHECKS:
        return report
    report["stall_override"] = {
        "stalled_rounds": stalled_rounds,
        "patience": patience,
        "overridden_checks": blocking,
        "instrument_evidence": dict(evidence or {}),
        "note": (
            "Discovery proceeded in degraded mode: retrieval instruments were dead for "
            f"{stalled_rounds} stalled rounds and only instrument checks "
            f"({', '.join(blocking)}) failed while every content check passed. "
            "Novelty readiness was NOT overridden."
        ),
    }
    report["discovery_ready"] = True
    report["blocking_reasons"] = []
    for name in blocking:
        report.setdefault("discovery_checks", {})[name] = True
    report.setdefault("warnings", []).append(
        "discovery opened by stall override with dead retrieval instruments; "
        "see stall_override for the recorded limitation")
    return report


def rank_papers_for_topic(topic: str, papers) -> list:
    """Deterministically put topically central papers first for reading/writing budgets."""

    paper_list = list(papers)
    _, rows, _ = _relevance(topic, paper_list)
    order = {row["id"]: index for index, row in enumerate(rows)}
    return sorted(
        paper_list,
        key=lambda p: order.get(str(getattr(p, "bare_id", getattr(p, "arxiv_id", ""))), len(order)),
    )


def report_markdown(report: dict) -> str:
    """Render the machine report without changing its meaning."""

    status = "ready" if report.get("discovery_ready") else "not ready"
    novelty = "ready" if report.get("novelty_ready") else "not ready"
    lines = [
        "# Corpus coverage report",
        "",
        f"Discovery: **{status}**",
        f"Novelty protocol: **{novelty}**",
        "",
        f"- Papers: {report.get('paper_count', 0)}",
        f"- Usable texts: {report.get('usable_text_count', 0)}",
        f"- Primary texts: {report.get('primary_text_count', 0)}",
        f"- Usable primary texts: {report.get('usable_primary_text_count', 0)}",
        f"- Relevant usable primary texts: "
        f"{report.get('relevant_usable_primary_text_count', 0)}",
        f"- Lexically relevant papers: {report.get('relevant_paper_count', 0)}",
        f"- Topic-term coverage: {report.get('topic_term_coverage', 0):.0%}",
        f"- Healthy searches: {(report.get('search') or {}).get('healthy_attempts', 0)}",
        f"- Search-list observed fraction estimate: "
        f"{((report.get('search') or {}).get('incidence_coverage') or {}).get('estimated_observed_fraction', 0):.0%} "
        "(diagnostic only)",
        f"- Citation graph healthy: {(report.get('graph') or {}).get('healthy', False)}",
        f"- Citation seeds covered: "
        f"{(report.get('graph') or {}).get('successful_seed_fraction', 0):.0%}",
        f"- Citation seed frontiers closed: "
        f"{(report.get('graph') or {}).get('closed_seed_fraction', 0):.0%}",
        f"- Citation frontier saturated: {(report.get('graph') or {}).get('saturated', False)}",
        "",
        "## Blocking checks",
    ]
    blocks = report.get("novelty_blocking_reasons") or []
    lines += [f"- {b}" for b in blocks] if blocks else ["- none"]
    if report.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"


def verify_jsonl_hash_chain(path: str | Path) -> dict:
    """Verify the append-only integrity chain used by ``thoughts.jsonl``."""

    previous = ""
    count = 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return {"ok": False, "entries": 0, "error": f"{type(exc).__name__}: {exc}"}
    for index, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
            claimed = str(entry.pop("entry_hash"))
        except Exception as exc:
            return {"ok": False, "entries": count, "line": index,
                    "error": f"invalid entry: {exc}"}
        if str(entry.get("prev_hash") or "") != previous:
            return {"ok": False, "entries": count, "line": index,
                    "error": "previous hash mismatch"}
        canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual != claimed:
            return {"ok": False, "entries": count, "line": index,
                    "error": "entry hash mismatch"}
        previous = claimed
        count += 1
    return {"ok": True, "entries": count, "head": previous}
