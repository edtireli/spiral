"""Novelty boundaries, proof-carrying papers, and living-paper manifests."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from spiral.epistemic import file_sha256


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8", "ignore")).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_file_record(path: str | Path, base: str | Path | None = None) -> dict:
    value = Path(path)
    record = {"path": str(value)}
    try:
        display = str(value)
        if base:
            try:
                display = str(value.relative_to(base))
            except ValueError:
                display = str(value.resolve())
        record.update({
            "path": display,
            "sha256": file_sha256(value),
            "bytes": value.stat().st_size,
        })
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


class NoveltyBoundaryCertificate:
    """A bounded, signed account of what the novelty search did and did not show."""

    @staticmethod
    def build(research_root: str | Path, *, question: str, proposal: dict,
              findings: list[dict], search_report: dict, priors: list[dict],
              coverage: dict) -> dict:
        root = Path(research_root)
        query_reports = search_report.get("query_reports") or []
        result_fingerprints = []
        for prior in priors:
            if not isinstance(prior, dict):
                continue
            material = {
                "title": " ".join(str(prior.get("title") or "").lower().split()),
                "identifier": str(prior.get("identifier") or ""),
                "year": prior.get("year"),
                "url": str(prior.get("url") or ""),
            }
            result_fingerprints.append({**material, "sha256": _digest(material)})
        claim_scope = []
        for finding in findings:
            claim = finding.get("claim") or {}
            if not finding.get("ok"):
                continue
            claim_scope.append({
                "claim_id": finding.get("claim_id"),
                "statement": claim.get("statement") or claim.get("note") or "",
                "assumptions": claim.get("assumptions") or [],
                "strength": finding.get("strength"),
                "backend": finding.get("backend"),
            })
        checks = {
            "multiple_healthy_queries": search_report.get("healthy_queries", 0) >= 2,
            "independent_query_families": search_report.get("healthy_query_families", 0) >= 2,
            "multiple_healthy_sources": len(search_report.get("sources_ok") or []) >= 2,
            "corpus_novelty_frontier_ready": bool(coverage.get("novelty_ready")),
            "query_results_recorded": bool(query_reports),
            "scoped_verified_claims_recorded": bool(claim_scope),
        }
        certificate = {
            "schema_version": 1,
            "kind": "bounded_novelty_search_certificate",
            "as_of": _now(),
            "question": question,
            "scope_statement": (
                "No matching result was identified within the documented corpus, queries, "
                "sources, and search date. This is a bounded search result, not proof of "
                "global absence or priority."
            ),
            "proposal_scope": {
                "question": proposal.get("question", ""),
                "conventions": proposal.get("conventions", ""),
                "corpus_basis": (proposal.get("_basis_audit") or {}).get("evidence") or [],
                "angle_audit": proposal.get("_angle_audit") or {},
                "nearest_prior_deep_reads": proposal.get("_grounded_prior_report") or {},
            },
            "claim_scope": claim_scope,
            "protocol": {
                "queries": search_report.get("queries") or [],
                "query_reports": query_reports,
                "sources_ok": search_report.get("sources_ok") or [],
                "healthy_queries": int(search_report.get("healthy_queries") or 0),
                "healthy_query_families": int(
                    search_report.get("healthy_query_families") or 0),
                "corpus_coverage_digest": {
                    "paper_count": coverage.get("paper_count"),
                    "relevant_paper_count": coverage.get("relevant_paper_count"),
                    "usable_primary_texts": coverage.get("usable_primary_texts"),
                    "graph": coverage.get("graph"),
                    "search": coverage.get("search"),
                    "topic_term_coverage": coverage.get("topic_term_coverage"),
                },
            },
            "nearest_results": result_fingerprints,
            "checks": checks,
            "valid": all(checks.values()),
        }
        unsigned = dict(certificate)
        certificate["certificate_sha256"] = _digest(unsigned)
        path = root / "novelty-boundary.json"
        _write(path, certificate)
        lines = [
            "# Novelty boundary certificate", "",
            f"As of: {certificate['as_of']}", "",
            f"Question: {question}", "",
            certificate["scope_statement"], "",
            f"Protocol valid: **{str(certificate['valid']).lower()}**", "",
            "## Queries", "",
        ]
        lines.extend(f"- `{query}`" for query in certificate["protocol"]["queries"])
        lines.extend(["", "## Scoped claims", ""])
        for claim in claim_scope:
            lines.append(
                f"- `{claim.get('claim_id')}` [{claim.get('strength')}] "
                f"{claim.get('statement')}")
        lines.extend(["", f"Certificate SHA-256: `{certificate['certificate_sha256']}`", ""])
        (root / "novelty-boundary.md").write_text("\n".join(lines), encoding="utf-8")
        return {**certificate, "path": str(path)}

    @staticmethod
    def validate(value_or_path) -> dict:
        if isinstance(value_or_path, (str, Path)):
            path = Path(value_or_path)
            if not path.is_file():
                return {"valid": False, "issues": ["certificate file is missing"]}
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"valid": False, "issues": [f"invalid JSON: {exc}"]}
        else:
            value = dict(value_or_path or {})
        signature = value.pop("certificate_sha256", "")
        value.pop("path", None)
        issues = []
        if not signature or signature != _digest(value):
            issues.append("certificate digest mismatch")
        if not all((value.get("checks") or {}).values()):
            issues.append("one or more novelty protocol checks failed")
        if "not proof of global absence" not in str(value.get("scope_statement") or ""):
            issues.append("bounded-absence disclaimer is missing")
        if not value.get("claim_scope"):
            issues.append("verified claim scope is empty")
        return {"valid": not issues, "issues": issues, "certificate": value}


class ProofCarryingPaper:
    """Build and verify the machine-readable evidence envelope around a paper."""

    @staticmethod
    def build(out_dir: str | Path, *, findings: list[dict], paper_files: list[str | Path],
              audit_files: list[str | Path], novelty_certificate: str | Path,
              obligation_graph: str | Path, obligation_report: dict,
              citation_packet: dict, scope_packet: dict, scope_audit: dict | None = None,
              research_commit: str = "", require_novelty: bool = True,
              require_replication: bool = True, integrity_chains: dict | None = None) -> dict:
        out = Path(out_dir)
        artifacts = []
        seen = set()
        for path in [*paper_files, *audit_files, novelty_certificate, obligation_graph]:
            if not path:
                continue
            record = _safe_file_record(path, out)
            key = record.get("path")
            if key not in seen:
                seen.add(key)
                artifacts.append(record)
        claims = []
        missing_certificates = []
        failed_replications = []
        for finding in findings:
            claim = finding.get("claim") or {}
            manifest = str(claim.get("manifest") or "")
            manifest_record = _safe_file_record(manifest, out) if manifest else {}
            if manifest and manifest_record.get("error"):
                missing_certificates.append(finding.get("claim_id"))
            replication = finding.get("replication") or {}
            if (require_replication and finding.get("required", True)
                    and not replication.get("passed")):
                failed_replications.append(finding.get("claim_id"))
            claims.append({
                "claim_id": finding.get("claim_id"),
                "statement": claim.get("statement") or claim.get("note") or "",
                "assumptions": claim.get("assumptions") or [],
                "strength": finding.get("strength"),
                "backend": finding.get("backend"),
                "required": bool(finding.get("required", True)),
                "certificate": manifest_record,
                "replication": replication,
                "obligation_id": finding.get("obligation_id", ""),
            })
        novelty_validation = (
            NoveltyBoundaryCertificate.validate(novelty_certificate)
            if require_novelty else {"valid": True, "issues": [], "not_required": True})
        audit_by_claim = {
            str(row.get("claim_id")): row for row in ((scope_audit or {}).get("claims") or [])
            if isinstance(row, dict) and row.get("claim_id")
        }
        paper_claims = []
        for row in scope_packet.get("claims") or []:
            claim_id = str(row.get("claim_id") or "")
            adjudication = audit_by_claim.get(claim_id, {})
            paper_claims.append({
                "claim_id": claim_id,
                "sentence": row.get("sentence", ""),
                "status": adjudication.get("status", ""),
                "evidence_id": adjudication.get("evidence_id", ""),
                "high_risk": bool(row.get("high_risk")),
            })
        bad_paper_claims = [
            row["claim_id"] for row in paper_claims
            if row.get("status") in {"", "unsupported", "contradicted"}
        ]
        checks = {
            "paper_artifacts_hashed": bool(artifacts) and all(
                record.get("sha256") for record in artifacts if not record.get("error")),
            "all_artifacts_readable": not any(record.get("error") for record in artifacts),
            "claim_registry_present": bool(claims),
            "workbench_certificates_present": not missing_certificates,
            "required_claims_blindly_replicated": (
                not require_replication or not failed_replications),
            "novelty_boundary_valid": bool(novelty_validation.get("valid")),
            "obligation_result_gate_ready": bool(obligation_report.get("ready")),
            "citation_packet_present": isinstance(citation_packet, dict),
            "claim_scope_packet_present": bool(scope_packet.get("claims") is not None),
            "every_paper_claim_adjudicated": not bad_paper_claims,
            "integrity_chains_valid": bool(integrity_chains) and all(
                bool(row.get("ok", row.get("valid", False)))
                for row in (integrity_chains or {}).values()),
        }
        manifest = {
            "schema_version": 1,
            "kind": "proof_carrying_paper",
            "created_at": _now(),
            "research_commit": research_commit,
            "claims": claims,
            "paper_claim_registry": paper_claims,
            "artifacts": artifacts,
            "citation_evidence_digest": _digest(citation_packet),
            "claim_scope_digest": _digest(scope_packet),
            "claim_scope_audit_digest": _digest(scope_audit or {}),
            "integrity_chains": integrity_chains or {},
            "obligation_report": obligation_report,
            "novelty_boundary": novelty_validation,
            "checks": checks,
            "valid": all(checks.values()),
        }
        manifest["manifest_sha256"] = _digest(manifest)
        path = out / "proof-carrying-manifest.json"
        _write(path, manifest)
        lines = [
            "# Proof-carrying paper", "",
            f"Bundle valid: **{str(manifest['valid']).lower()}**", "",
            f"Manifest SHA-256: `{manifest['manifest_sha256']}`", "",
            "## Claims", "",
        ]
        for claim in claims:
            marker = "verified" if claim.get("strength") in {
                "formal", "exact", "computational"} else claim.get("strength")
            lines.append(
                f"- `{claim.get('claim_id')}` ({marker}, {claim.get('backend')}): "
                f"{claim.get('statement')}")
            if claim.get("replication"):
                lines.append(
                    f"  blind replication: {claim['replication'].get('status', 'unknown')}")
        lines.extend(["", "## Bundle checks", ""])
        lines.extend(f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items())
        (out / "proof-carrying-paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {**manifest, "path": str(path)}

    @staticmethod
    def validate(path: str | Path) -> dict:
        path = Path(path)
        if not path.is_file():
            return {"valid": False, "issues": ["proof manifest is missing"]}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"valid": False, "issues": [f"invalid proof manifest JSON: {exc}"]}
        signature = value.pop("manifest_sha256", "")
        issues = []
        if signature != _digest(value):
            issues.append("manifest digest mismatch")
        for record in value.get("artifacts") or []:
            artifact = Path(record.get("path") or "")
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            try:
                if file_sha256(artifact) != record.get("sha256"):
                    issues.append(f"artifact changed: {record.get('path')}")
            except Exception:
                issues.append(f"artifact missing: {record.get('path')}")
        if not all((value.get("checks") or {}).values()):
            issues.append("recorded bundle checks are not all green")
        return {"valid": not issues, "issues": issues}


class LivingPaper:
    """Track when a completed paper's local evidence or search horizon goes stale."""

    @staticmethod
    def create(research_root: str | Path, out_dir: str | Path, *, topic: str,
               question: str, proof_manifest: str | Path,
               novelty_certificate: str | Path, obligation_graph: str | Path,
               corpus_manifest: str | Path, research_commit: str,
               recheck_days: int = 30) -> dict:
        root = Path(research_root)
        out = Path(out_dir)
        dependencies = [
            _safe_file_record(proof_manifest, root),
            _safe_file_record(novelty_certificate, root),
            _safe_file_record(obligation_graph, root),
            _safe_file_record(corpus_manifest, root),
        ]
        value = {
            "schema_version": 1,
            "kind": "living_paper",
            "created_at": _now(),
            "last_checked_at": _now(),
            "topic": topic,
            "question": question,
            "status": "current",
            "recheck_days": max(1, int(recheck_days)),
            "research_commit": research_commit,
            "dependencies": dependencies,
            "refresh_protocol": [
                "revalidate all artifact hashes",
                "rerun bounded prior-art searches after the recheck horizon",
                "reopen the obligation graph if a new nearest prior changes scope",
                "rerun affected certificates and blind replications",
                "publish a new research checkpoint rather than overwriting history",
            ],
        }
        value["manifest_sha256"] = _digest(value)
        path = out / "living-paper.json"
        _write(path, value)
        _write(root / "living-paper.json", value)
        return {**value, "path": str(path)}

    @staticmethod
    def inspect(path: str | Path, research_root: str | Path | None = None) -> dict:
        path = Path(path)
        if not path.is_file():
            return {"current": False, "stale": False, "issues": ["living-paper manifest missing"]}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"current": False, "stale": True, "issues": [f"invalid manifest: {exc}"]}
        signature = value.pop("manifest_sha256", "")
        issues = []
        if signature != _digest(value):
            issues.append("living-paper manifest digest mismatch")
        base = Path(research_root) if research_root else path.parent
        for record in value.get("dependencies") or []:
            dep = Path(record.get("path") or "")
            if not dep.is_absolute():
                dep = base / dep
            try:
                if file_sha256(dep) != record.get("sha256"):
                    issues.append(f"dependency changed: {record.get('path')}")
            except Exception:
                issues.append(f"dependency missing: {record.get('path')}")
            if dep.name == "proof-carrying-manifest.json" and dep.is_file():
                proof_status = ProofCarryingPaper.validate(dep)
                if not proof_status.get("valid"):
                    issues.extend(
                        f"proof envelope: {issue}" for issue in proof_status.get("issues") or [])
        try:
            checked = datetime.fromisoformat(str(value.get("last_checked_at") or ""))
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - checked.astimezone(timezone.utc)).total_seconds() / 86400
        except Exception:
            age_days = float("inf")
            issues.append("last-checked timestamp is invalid")
        overdue = age_days >= max(1, int(value.get("recheck_days") or 30))
        if overdue:
            issues.append(f"literature recheck horizon exceeded ({age_days:.1f} days)")
        return {
            "current": not issues,
            "stale": bool(issues),
            "overdue": overdue,
            "age_days": round(age_days, 3) if age_days != float("inf") else None,
            "issues": issues,
            "manifest": value,
        }
