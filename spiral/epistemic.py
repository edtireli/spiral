"""Persistent obligation graph for accountable autonomous work.

The graph is Spiral's shared epistemic kernel.  Models may propose questions,
claims, assumptions, experiments, and interpretations, but a required obligation
is only closed by an explicit evidence path.  Research and writing therefore use
the same durable object instead of passing confidence-flavoured prose between
loosely connected phases.

The canonical files are deliberately plain JSON/JSONL so a run remains
inspectable without Spiral:

``epistemic/obligations.json``
    Current materialized graph.
``epistemic/events.jsonl``
    Hash-chained mutations and decisions.
``epistemic/obligations.md``
    Human-readable open/closed obligation register.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path


SCHEMA_VERSION = 1
STATUSES = {
    "open", "in_progress", "declared", "supported", "refuted",
    "blocked", "superseded", "waived",
}
TERMINAL = {"declared", "supported", "refuted", "superseded", "waived"}
STAGE_ORDER = {"discovery": 0, "result": 1, "publication": 2}
RELATIONS = {
    "depends_on", "supports", "refutes", "tests", "falsifies",
    "derived_from", "replicates", "scopes", "cites", "produces",
    "answers", "supersedes",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "ignore")).hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_obligation_id(kind: str, statement: str, namespace: str = "") -> str:
    material = f"{namespace}\n{kind}\n{' '.join(str(statement or '').split()).lower()}"
    return f"{kind}:{_sha256_text(material)[:20]}"


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


class ObligationGraph:
    """Materialized obligation graph with an append-only mutation recorder."""

    def __init__(self, research_root: str | Path, topic: str):
        self.research_root = Path(research_root)
        self.root = self.research_root / "epistemic"
        self.path = self.root / "obligations.json"
        self.events_path = self.root / "events.jsonl"
        self.markdown_path = self.root / "obligations.md"
        self.root.mkdir(parents=True, exist_ok=True)
        self.data = self._load(topic)
        self._last_event_hash = self._read_last_event_hash()
        self.objective_id = self.ensure(
            "objective",
            topic,
            node_id="objective:root",
            stage="publication",
            required=True,
            metadata={"role": "user_intent"},
            emit=False,
        )
        self.save()

    def _load(self, topic: str) -> dict:
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                value = {}
        else:
            value = {}
        if not isinstance(value, dict):
            value = {}
        value.setdefault("schema_version", SCHEMA_VERSION)
        value.setdefault("topic", topic)
        value.setdefault("created_at", _now())
        value.setdefault("updated_at", _now())
        if not isinstance(value.get("nodes"), dict):
            value["nodes"] = {}
        if not isinstance(value.get("edges"), list):
            value["edges"] = []
        if not isinstance(value.get("metadata"), dict):
            value["metadata"] = {}
        return value

    def _read_last_event_hash(self) -> str:
        if not self.events_path.is_file():
            return ""
        try:
            line = self.events_path.read_text(encoding="utf-8").splitlines()[-1]
            return str(json.loads(line).get("entry_hash") or "")
        except Exception:
            return ""

    def _event(self, action: str, **payload) -> None:
        entry = {
            "ts": _now(),
            "action": action,
            **payload,
            "prev_hash": self._last_event_hash,
        }
        entry["entry_hash"] = _sha256_text(_canonical(entry))
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._last_event_hash = entry["entry_hash"]
        except Exception:
            pass

    def ensure(self, kind: str, statement: str, *, node_id: str = "",
               stage: str = "result", required: bool = False,
               status: str = "open", scope: str = "", assumptions=None,
               verifier: str = "", falsifier: str = "", provenance=None,
               metadata=None, emit: bool = True) -> str:
        kind = re.sub(r"[^a-z0-9_-]+", "_", str(kind or "obligation").lower())
        statement = " ".join(str(statement or "").split())
        node_id = node_id or stable_obligation_id(kind, statement, self.data.get("topic", ""))
        stage = stage if stage in STAGE_ORDER else "result"
        status = status if status in STATUSES else "open"
        existing = self.data["nodes"].get(node_id)
        now = _now()
        if existing:
            changed = False
            updates = {
                "statement": statement or existing.get("statement", ""),
                "stage": stage,
                "required": bool(required or existing.get("required")),
                "scope": scope or existing.get("scope", ""),
                "verifier": verifier or existing.get("verifier", ""),
                "falsifier": falsifier or existing.get("falsifier", ""),
            }
            for key, value in updates.items():
                if value != existing.get(key):
                    existing[key] = value
                    changed = True
            for key, values in (("assumptions", assumptions), ("provenance", provenance)):
                if values:
                    merged = list(dict.fromkeys([
                        *[str(x) for x in existing.get(key, [])],
                        *[str(x) for x in values],
                    ]))
                    if merged != existing.get(key):
                        existing[key] = merged
                        changed = True
            if metadata:
                merged_meta = {**existing.get("metadata", {}), **dict(metadata)}
                if merged_meta != existing.get("metadata"):
                    existing["metadata"] = merged_meta
                    changed = True
            if existing.get("status") == "open" and status != "open":
                existing["status"] = status
                changed = True
            if changed:
                existing["updated_at"] = now
                if emit:
                    self._event("node_update", node_id=node_id, node=existing)
            return node_id

        node = {
            "id": node_id,
            "kind": kind,
            "statement": statement,
            "stage": stage,
            "required": bool(required),
            "status": status,
            "scope": scope,
            "assumptions": [str(x) for x in (assumptions or [])],
            "verifier": verifier,
            "falsifier": falsifier,
            "provenance": [str(x) for x in (provenance or [])],
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
        }
        self.data["nodes"][node_id] = node
        if emit:
            self._event("node_create", node_id=node_id, node=node)
        return node_id

    def node(self, node_id: str) -> dict:
        return self.data["nodes"].get(node_id, {})

    def link(self, source: str, target: str, relation: str, *, metadata=None) -> None:
        if source not in self.data["nodes"] or target not in self.data["nodes"]:
            return
        relation = relation if relation in RELATIONS else "depends_on"
        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "metadata": dict(metadata or {}),
        }
        key = (source, target, relation)
        for existing in self.data["edges"]:
            if (existing.get("source"), existing.get("target"),
                    existing.get("relation")) == key:
                if metadata:
                    existing["metadata"] = {
                        **existing.get("metadata", {}), **dict(metadata),
                    }
                return
        self.data["edges"].append(edge)
        self._event("edge_create", edge=edge)

    def set_status(self, node_id: str, status: str, *, reason: str = "",
                   verifier: str = "", metadata=None) -> bool:
        node = self.node(node_id)
        if not node or status not in STATUSES:
            return False
        old = node.get("status", "open")
        merged_metadata = {**node.get("metadata", {}), **dict(metadata or {})}
        if (old == status
                and (not verifier or verifier == node.get("verifier", ""))
                and merged_metadata == node.get("metadata", {})):
            return True
        node["status"] = status
        node["updated_at"] = _now()
        if verifier:
            node["verifier"] = verifier
        if reason:
            node.setdefault("status_history", []).append({
                "ts": _now(), "from": old, "to": status, "reason": reason[:2000],
            })
        if metadata:
            node["metadata"] = merged_metadata
        self._event(
            "status", node_id=node_id, old=old, new=status,
            reason=reason[:2000], verifier=verifier,
        )
        return True

    def add_evidence(self, target_id: str, statement: str, *,
                     evidence_kind: str = "evidence", artifact: str = "",
                     verifier: str = "", independent: bool = False,
                     relation: str = "supports", status: str = "supported",
                     metadata=None, node_id: str = "") -> str:
        artifact_hash = ""
        if artifact:
            try:
                artifact_hash = file_sha256(artifact)
            except Exception:
                artifact_hash = ""
        evidence_id = self.ensure(
            evidence_kind,
            statement,
            node_id=node_id,
            stage=self.node(target_id).get("stage", "result"),
            required=False,
            status=status,
            verifier=verifier,
            provenance=[artifact] if artifact else [],
            metadata={
                **dict(metadata or {}),
                "artifact": artifact,
                "artifact_sha256": artifact_hash,
                "independent": bool(independent),
            },
        )
        self.link(evidence_id, target_id, relation)
        return evidence_id

    def incoming(self, node_id: str, relation: str = "") -> list[tuple[dict, dict]]:
        rows = []
        for edge in self.data["edges"]:
            if edge.get("target") != node_id:
                continue
            if relation and edge.get("relation") != relation:
                continue
            source = self.node(str(edge.get("source") or ""))
            if source:
                rows.append((source, edge))
        return rows

    def outgoing(self, node_id: str, relation: str = "") -> list[tuple[dict, dict]]:
        rows = []
        for edge in self.data["edges"]:
            if edge.get("source") != node_id:
                continue
            if relation and edge.get("relation") != relation:
                continue
            target = self.node(str(edge.get("target") or ""))
            if target:
                rows.append((target, edge))
        return rows

    def _cycles(self) -> list[list[str]]:
        adjacency: dict[str, list[str]] = {}
        for edge in self.data["edges"]:
            if edge.get("relation") != "depends_on":
                continue
            adjacency.setdefault(str(edge.get("source")), []).append(str(edge.get("target")))
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []
        cycles: list[list[str]] = []

        def visit(node_id: str) -> None:
            if node_id in visiting:
                try:
                    start = stack.index(node_id)
                except ValueError:
                    start = 0
                cycles.append(stack[start:] + [node_id])
                return
            if node_id in visited:
                return
            visiting.add(node_id)
            stack.append(node_id)
            for nxt in adjacency.get(node_id, []):
                visit(nxt)
            stack.pop()
            visiting.discard(node_id)
            visited.add(node_id)

        for node_id in adjacency:
            visit(node_id)
        return cycles

    def verify_event_chain(self) -> dict:
        if not self.events_path.is_file():
            return {"valid": True, "entries": 0, "last_hash": ""}
        previous = ""
        entries = 0
        try:
            for line_no, line in enumerate(
                    self.events_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                signature = str(entry.pop("entry_hash", ""))
                if entry.get("prev_hash", "") != previous:
                    return {"valid": False, "entries": entries,
                            "error": f"previous hash mismatch at line {line_no}"}
                expected = _sha256_text(_canonical(entry))
                if not signature or signature != expected:
                    return {"valid": False, "entries": entries,
                            "error": f"entry hash mismatch at line {line_no}"}
                previous = signature
                entries += 1
        except Exception as exc:
            return {"valid": False, "entries": entries,
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"valid": True, "entries": entries, "last_hash": previous}

    def report(self, stage: str = "result") -> dict:
        """Return a deterministic closure report for obligations up to ``stage``."""

        max_stage = STAGE_ORDER.get(stage, STAGE_ORDER["result"])
        considered = [
            node for node in self.data["nodes"].values()
            if STAGE_ORDER.get(node.get("stage", "result"), 1) <= max_stage
        ]
        required = [node for node in considered if node.get("required")]
        blockers = []
        evidence_gaps = []
        replication_gaps = []
        for node in required:
            status = node.get("status", "open")
            if status not in {"supported", "declared", "waived", "superseded"}:
                blockers.append({
                    "id": node["id"], "kind": node.get("kind"),
                    "status": status, "statement": node.get("statement", ""),
                })
                continue
            if status == "supported" and node.get("kind") in {
                    "claim", "question", "novelty", "artifact", "objective"}:
                support = [
                    source for source, edge in self.incoming(node["id"])
                    if edge.get("relation") in {"supports", "answers", "scopes", "produces"}
                    and source.get("status") == "supported"
                ]
                if not support:
                    evidence_gaps.append(node["id"])
            if node.get("metadata", {}).get("requires_replication"):
                replications = [
                    source for source, edge in self.incoming(node["id"], "replicates")
                    if source.get("status") == "supported"
                    and source.get("metadata", {}).get("independent") is True
                ]
                if not replications:
                    replication_gaps.append(node["id"])

        cycles = self._cycles()
        event_chain = self.verify_event_chain()
        ready = (not blockers and not evidence_gaps and not replication_gaps
                 and not cycles and event_chain.get("valid") is True)
        counts: dict[str, int] = {}
        for node in considered:
            key = str(node.get("status", "open"))
            counts[key] = counts.get(key, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "ready": ready,
            "node_count": len(considered),
            "required_count": len(required),
            "counts": counts,
            "blockers": blockers,
            "evidence_gaps": evidence_gaps,
            "replication_gaps": replication_gaps,
            "dependency_cycles": cycles,
            "event_chain": event_chain,
            "graph_sha256": self.digest(),
            "last_event_hash": self._last_event_hash,
        }

    def digest(self) -> str:
        material = {
            "schema_version": self.data.get("schema_version"),
            "topic": self.data.get("topic"),
            "nodes": self.data.get("nodes"),
            "edges": self.data.get("edges"),
        }
        return _sha256_text(_canonical(material))

    def compact(self) -> dict:
        """Return the graph layer consumed by the browser research map."""

        return {
            "digest": self.digest(),
            "nodes": [
                {
                    "id": node["id"],
                    "type": f"obligation_{node.get('kind', 'item')}",
                    "kind": node.get("kind"),
                    "label": node.get("statement", "")[:160],
                    "title": node.get("statement", ""),
                    "status": node.get("status"),
                    "stage": node.get("stage"),
                    "required": bool(node.get("required")),
                    "scope": node.get("scope", ""),
                    "verifier": node.get("verifier", ""),
                    "metadata": node.get("metadata", {}),
                }
                for node in self.data["nodes"].values()
            ],
            "edges": [dict(edge) for edge in self.data["edges"]],
            "result_report": self.report("result"),
            "publication_report": self.report("publication"),
        }

    def markdown(self) -> str:
        lines = [
            "# Epistemic obligation register", "",
            f"Topic: {self.data.get('topic', '')}",
            f"Graph digest: `{self.digest()}`", "",
        ]
        for stage in ("discovery", "result", "publication"):
            lines.extend([f"## {stage.title()}", ""])
            stage_nodes = sorted(
                (n for n in self.data["nodes"].values() if n.get("stage") == stage),
                key=lambda n: (not n.get("required"), n.get("kind", ""), n.get("id", "")),
            )
            if not stage_nodes:
                lines.extend(["(none)", ""])
                continue
            for node in stage_nodes:
                mark = {
                    "supported": "[x]", "declared": "[x]", "waived": "[-]",
                    "superseded": "[-]", "refuted": "[!]", "blocked": "[!]",
                }.get(node.get("status"), "[ ]")
                req = "required" if node.get("required") else "optional"
                lines.append(
                    f"- {mark} **{node.get('kind')}** ({req}, {node.get('status')}): "
                    f"{node.get('statement', '')}  ")
                lines.append(f"  id: `{node.get('id')}`")
                incoming = self.incoming(node["id"])
                for source, edge in incoming[:8]:
                    lines.append(
                        f"  - {edge.get('relation')}: `{source.get('id')}` "
                        f"({source.get('status')}) {source.get('statement', '')[:180]}")
            lines.append("")
        for stage in ("result", "publication"):
            report = self.report(stage)
            lines.extend([
                f"## {stage.title()} gate", "",
                f"Ready: **{str(report['ready']).lower()}**", "",
            ])
            for blocker in report["blockers"]:
                lines.append(
                    f"- blocked `{blocker['id']}` ({blocker['status']}): "
                    f"{blocker['statement']}")
            for node_id in report["evidence_gaps"]:
                lines.append(f"- missing evidence path: `{node_id}`")
            for node_id in report["replication_gaps"]:
                lines.append(f"- missing blind replication: `{node_id}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def save(self) -> None:
        self.data["schema_version"] = SCHEMA_VERSION
        digest = self.digest()
        if self.data["metadata"].get("graph_sha256") != digest:
            self.data["updated_at"] = _now()
        self.data["metadata"]["graph_sha256"] = digest
        self.data["metadata"]["last_event_hash"] = self._last_event_hash
        _atomic_json(self.path, self.data)
        self.markdown_path.write_text(self.markdown(), encoding="utf-8")
