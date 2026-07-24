"""Browser-ready research-map visualization.

``research-map.json`` is the audit log; this module turns it into an inspectable
field map: keyword searches, papers gathered directly, citation/reference edges,
co-citation holes, and recent citing papers. The output is a self-contained HTML
file so a long run can be opened locally without a server or external assets.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path


def _short(s: str, n: int = 72) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _paper_meta(corpus) -> dict[str, dict]:
    papers = {}
    for pid, p in getattr(corpus, "papers", {}).items():
        papers[pid] = {
            "id": pid,
            "label": _short(getattr(p, "title", "") or pid),
            "title": getattr(p, "title", "") or pid,
            "authors": list(getattr(p, "authors", []) or [])[:8],
            "categories": list(getattr(p, "categories", []) or []),
            "url": getattr(p, "url", "") or f"https://arxiv.org/abs/{pid}",
            "type": "paper",
            "in_corpus": True,
        }
    return papers


def build_graph_data(map_data: dict, corpus=None) -> dict:
    """Normalize the persisted map/corpus into nodes and edges for the viewer."""
    topic = map_data.get("topic") or "research topic"
    nodes: dict[str, dict] = {
        "topic": {"id": "topic", "type": "topic", "layer": "field",
                  "label": _short(topic, 86), "title": topic},
    }
    nodes.update(_paper_meta(corpus))
    for value in nodes.values():
        value.setdefault("layer", "field")
    edges: list[dict] = []

    def node(nid: str, typ: str, label: str, **extra):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": typ, "layer": "field",
                          "label": _short(label), "title": label, **extra}
        else:
            nodes[nid].update({k: v for k, v in extra.items() if v not in ("", None, [], {})})
            if nodes[nid].get("type") in {"external", "recent", "hole"} and typ == "paper":
                nodes[nid]["type"] = "paper"
                nodes[nid]["in_corpus"] = True
        return nodes[nid]

    def paper(nid: str, **extra):
        return node(nid, "paper" if extra.get("in_corpus") else extra.get("type", "external"),
                    extra.get("title") or nid, **extra)

    for i, s in enumerate(map_data.get("searches") or [], 1):
        sid = f"search:{i}"
        q = s.get("query") or f"search {i}"
        node(sid, "search", q, round=s.get("round", 0),
             categories=s.get("categories") or [], corpus_size=s.get("corpus_size"))
        edges.append({"source": "topic", "target": sid, "type": "search", "label": "searched"})
        for aid in s.get("added") or []:
            paper(aid, in_corpus=True)
            edges.append({"source": sid, "target": aid, "type": "found", "label": "found"})

    for gi, g in enumerate(map_data.get("graph_rounds") or [], 1):
        for aid in g.get("seeds") or []:
            paper(aid, in_corpus=True)
        for e in g.get("edges") or []:
            src, tgt = e.get("source"), e.get("target")
            if not src or not tgt:
                continue
            paper(src, in_corpus=True)
            typ = "recent" if e.get("direction") == "citations" else "external"
            paper(tgt, type=typ, title=e.get("title") or tgt,
                  citations=e.get("citations"), year=e.get("year"))
            edges.append({
                "source": src,
                "target": tgt,
                "type": "cited-by" if e.get("direction") == "citations" else "references",
                "label": "cited by" if e.get("direction") == "citations" else "references",
                "round": g.get("research_round", gi),
            })
        for h in g.get("holes") or []:
            hid = h.get("id")
            if not hid:
                continue
            paper(hid, type="hole", title=h.get("title") or hid,
                  cocitations=h.get("count"), citations=h.get("citations"))
        for r in g.get("recent") or []:
            rid = r.get("id")
            if rid:
                paper(rid, type="recent", title=r.get("title") or rid,
                      citations=r.get("citations"))
        for aid in g.get("added") or []:
            paper(aid, in_corpus=True)

    data_catalog = map_data.get("data_catalog")
    if isinstance(data_catalog, dict):
        for index, record in enumerate(data_catalog.get("records") or []):
            if not isinstance(record, dict):
                continue
            source = str(record.get("source") or "data")
            dataset_id = str(record.get("dataset_id") or index)
            did = f"data:{source}:{dataset_id}"
            node(
                did, "dataset", record.get("title") or dataset_id,
                source_name=source, dataset_id=dataset_id,
                description=record.get("description") or "",
                version=record.get("version") or "",
                doi=record.get("doi") or "",
                license=record.get("license") or "",
                species=record.get("species") or "",
                modalities=record.get("modalities") or [],
                url=record.get("url") or "",
            )
            edges.append({
                "source": "topic", "target": did, "type": "data",
                "label": "candidate dataset", "layer": "field",
            })

    epistemic = map_data.get("epistemic") if isinstance(map_data.get("epistemic"), dict) else {}
    kind_types = {
        "objective": "objective", "question": "question", "candidate_question": "question",
        "claim": "claim", "assumption": "assumption", "falsifier": "falsifier",
        "verification": "evidence", "replication": "evidence",
        "novelty_certificate": "evidence", "publication_evidence": "evidence",
        "source": "source", "deep_read": "source", "search": "search_action",
        "artifact": "artifact", "decision": "decision", "novelty": "novelty",
        "idea_family": "idea",
    }
    epistemic_ids = set()
    for raw in epistemic.get("nodes") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        eid = f"ep:{raw['id']}"
        epistemic_ids.add(str(raw["id"]))
        kind = str(raw.get("kind") or "obligation")
        typ = kind_types.get(kind, "evidence")
        nodes[eid] = {
            **raw,
            "id": eid,
            "obligation_id": raw["id"],
            "type": typ,
            "layer": "epistemic",
            "label": _short(raw.get("label") or raw.get("title") or raw["id"]),
            "title": raw.get("title") or raw.get("label") or raw["id"],
        }
    if "objective:root" in epistemic_ids:
        edges.append({
            "source": "topic", "target": "ep:objective:root",
            "type": "epistemic", "label": "research objective", "layer": "epistemic",
        })
    for raw in epistemic.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source, target = str(raw.get("source") or ""), str(raw.get("target") or "")
        if source not in epistemic_ids or target not in epistemic_ids:
            continue
        relation = str(raw.get("relation") or "depends_on")
        edges.append({
            "source": f"ep:{source}", "target": f"ep:{target}",
            "type": relation, "label": relation.replace("_", " "),
            "layer": "epistemic", "metadata": raw.get("metadata") or {},
        })

    # Deduplicate edges while keeping type distinctions.
    seen = set()
    clean_edges = []
    for e in edges:
        key = (e.get("source"), e.get("target"), e.get("type"))
        if key in seen:
            continue
        seen.add(key)
        clean_edges.append(e)

    counts = {}
    for n in nodes.values():
        counts[n["type"]] = counts.get(n["type"], 0) + 1
    return {
        "topic": topic,
        "nodes": list(nodes.values()),
        "edges": clean_edges,
        "counts": counts,
        "searches": len(map_data.get("searches") or []),
        "graph_rounds": len(map_data.get("graph_rounds") or []),
        "epistemic_digest": epistemic.get("digest", ""),
        "obligation_report": epistemic.get("result_report") or {},
    }


def _html_template(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    title = html.escape(_short(data.get("topic", "research graph"), 96))
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Spiral Research Graph - {title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #101114;
    --fg: #f2eee8;
    --muted: #a7a29a;
    --panel: #181a1f;
    --line: #343840;
    --accent: #d97757;
    --paper: #5fb3a6;
    --search: #e7b75f;
    --hole: #e06666;
    --recent: #8ba6ff;
    --question: #d97757;
    --claim: #43c59e;
    --evidence: #74a7ff;
    --assumption: #c8a96b;
    --artifact: #f18f6c;
    --decision: #c58af9;
    --dataset: #d58ab7;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#fbfaf8; --fg:#202124; --muted:#66615b; --panel:#ffffff; --line:#d8d2ca; }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ min-height: 100%; }}
  body {{ margin: 0; height: 100dvh; overflow: hidden; display: flex; flex-direction: column;
    background: var(--bg); color: var(--fg);
    font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  header {{ flex: none; padding: 18px 22px 12px; border-bottom: 1px solid var(--line); }}
  h1 {{ margin: 0 0 8px; font-size: 20px; font-weight: 600; }}
  .meta {{ color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap; }}
  main {{ flex: 1; min-height: 0; display: grid;
    grid-template-columns: minmax(0, 1fr) clamp(300px, 26vw, 380px); }}
  .graph-wrap {{ position: relative; min-width: 0; min-height: 0; overflow: hidden; }}
  #graph {{ width: 100%; height: 100%; display: block; background: var(--bg);
    cursor: grab; touch-action: none; }}
  #graph.dragging {{ cursor: grabbing; }}
  .zoom-bar {{ position: absolute; left: 12px; top: 12px; display: flex; align-items: center; gap: 6px;
    padding: 6px; border: 1px solid var(--line); border-radius: 8px; background: color-mix(in srgb, var(--panel) 92%, transparent);
    box-shadow: 0 10px 30px color-mix(in srgb, var(--bg) 35%, transparent); }}
  .zoom-bar button {{ min-width: 32px; height: 32px; border: 1px solid var(--line); border-radius: 6px;
    background: var(--panel); color: var(--fg); font: inherit; cursor: pointer; }}
  .zoom-bar button:hover {{ border-color: var(--accent); }}
  .zoom-label {{ min-width: 46px; color: var(--muted); text-align: center; font-variant-numeric: tabular-nums; }}
  aside {{ border-left: 1px solid var(--line); padding: 14px; background: color-mix(in srgb, var(--panel) 88%, transparent); overflow: auto; }}
  .controls {{ display: grid; gap: 10px; margin-bottom: 12px; }}
  .layers {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }}
  .layers button {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 8px;
    background: var(--panel); color: var(--muted); cursor: pointer; }}
  .layers button.active {{ color: var(--fg); border-color: var(--accent); }}
  input {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px;
    color: var(--fg); background: var(--panel); }}
  .legend {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
  .pill {{ display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }}
  .sw {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .detail {{ border-top: 1px solid var(--line); padding-top: 12px; margin-top: 12px; }}
  .detail h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .detail p {{ margin: 0 0 8px; }}
  .detail a {{ color: var(--accent); }}
  .node text {{ pointer-events: none; fill: var(--fg); font-size: 11px; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }}
  .edge {{ stroke: var(--line); stroke-opacity: .52; stroke-width: 1.2; }}
  .edge.references {{ stroke-dasharray: 4 4; }}
  .edge.found {{ stroke: var(--accent); stroke-opacity: .75; }}
  .edge.search {{ stroke: var(--search); stroke-opacity: .7; }}
  .edge.cited-by {{ stroke: var(--recent); stroke-opacity: .58; }}
  .node circle {{ stroke: var(--bg); stroke-width: 1.5; cursor: pointer; }}
  .node.topic circle {{ fill: var(--accent); }}
  .node.search circle {{ fill: var(--search); }}
  .node.paper circle {{ fill: var(--paper); }}
  .node.external circle {{ fill: var(--muted); }}
  .node.hole circle {{ fill: var(--hole); }}
  .node.recent circle {{ fill: var(--recent); }}
  .node.objective circle, .node.question circle {{ fill: var(--question); }}
  .node.claim circle {{ fill: var(--claim); }}
  .node.evidence circle, .node.source circle, .node.search_action circle {{ fill: var(--evidence); }}
  .node.assumption circle, .node.falsifier circle {{ fill: var(--assumption); }}
  .node.artifact circle {{ fill: var(--artifact); }}
  .node.dataset circle {{ fill: var(--dataset); }}
  .node.decision circle, .node.novelty circle, .node.idea circle {{ fill: var(--decision); }}
  .node[data-status="refuted"] circle, .node[data-status="blocked"] circle {{ fill: var(--hole); }}
  .node.dim, .edge.dim {{ opacity: .16; }}
  .node.hit circle, .node.selected circle {{ stroke: var(--accent); stroke-width: 4; }}
  @media (max-width: 840px) {{
    body {{ height: auto; min-height: 100dvh; overflow: auto; }}
    main {{ grid-template-columns: 1fr; }}
    .graph-wrap, #graph {{ height: 68svh; min-height: 420px; }}
    aside {{ border-left: 0; border-top: 1px solid var(--line); }}
  }}
</style>
<header>
  <h1>{title}</h1>
  <div class="meta" id="metrics"></div>
</header>
<main>
  <div class="graph-wrap">
    <svg id="graph" role="img" aria-label="Research search and citation graph"></svg>
    <div class="zoom-bar" aria-label="Graph navigation">
      <button id="zoomOut" title="Zoom out">−</button>
      <button id="zoomIn" title="Zoom in">+</button>
      <button id="fitGraph" title="Fit graph">Fit</button>
      <span class="zoom-label" id="zoomLabel">100%</span>
    </div>
  </div>
  <aside>
    <div class="controls">
      <input id="search" placeholder="Filter by paper id, title, query, author">
      <div class="layers" aria-label="Graph layer">
        <button data-layer="all">All</button>
        <button class="active" data-layer="field">Literature</button>
        <button data-layer="epistemic">Reasoning</button>
      </div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="detail" id="detail"></div>
  </aside>
</main>
<script>
const data = {payload};
const color = {{topic:'var(--accent)', search:'var(--search)', paper:'var(--paper)', external:'var(--muted)', hole:'var(--hole)', recent:'var(--recent)', dataset:'var(--dataset)', objective:'var(--question)', question:'var(--question)', claim:'var(--claim)', evidence:'var(--evidence)', source:'var(--evidence)', search_action:'var(--evidence)', assumption:'var(--assumption)', falsifier:'var(--assumption)', artifact:'var(--artifact)', decision:'var(--decision)', novelty:'var(--decision)', idea:'var(--decision)'}};
const svg = document.getElementById('graph');
const detail = document.getElementById('detail');
const metrics = document.getElementById('metrics');
const legend = document.getElementById('legend');
const obligations = data.nodes.filter(n => n.layer === 'epistemic').length;
metrics.textContent = `${{data.nodes.length}} nodes · ${{data.edges.length}} edges · ${{data.searches}} searches · ${{data.graph_rounds}} graph rounds · ${{obligations}} obligations`;
for (const t of Object.keys(data.counts).sort((a,b) => (a === 'topic' ? -1 : b === 'topic' ? 1 : a.localeCompare(b)))) {{
  const n = data.counts[t] || 0;
  if (!n) continue;
  const el = document.createElement('div');
  el.className = 'pill';
  el.innerHTML = `<span class="sw" style="background:${{color[t] || 'var(--muted)'}}"></span>${{t.replace('_',' ')}} ${{n}}`;
  legend.appendChild(el);
}}
function radius(n) {{
  if (n.type === 'topic') return 18;
  if (n.type === 'search') return 10;
  if (n.type === 'hole') return 11;
  if (n.type === 'paper') return 8;
  if (n.type === 'objective') return 16;
  if (n.type === 'question') return 12;
  if (n.type === 'claim') return n.required ? 10 : 8;
  if (n.type === 'artifact') return 11;
  if (n.type === 'dataset') return 10;
  return 6;
}}
function show(n) {{
  const rows = [];
  rows.push(`<h2>${{escapeHtml(n.title || n.label || n.id)}}</h2>`);
  rows.push(`<p><b>${{escapeHtml(n.type)}}</b> · <code>${{escapeHtml(n.id)}}</code></p>`);
  if (n.status) rows.push(`<p><b>${{escapeHtml(n.status)}}</b>${{n.required ? ' · required' : ''}}${{n.stage ? ' · ' + escapeHtml(n.stage) : ''}}</p>`);
  if (n.scope) rows.push(`<p>${{escapeHtml(n.scope)}}</p>`);
  if (n.verifier) rows.push(`<p>verifier: ${{escapeHtml(n.verifier)}}</p>`);
  if (n.authors && n.authors.length) rows.push(`<p>${{escapeHtml(n.authors.join(', '))}}</p>`);
  if (n.categories && n.categories.length) rows.push(`<p>${{escapeHtml(n.categories.join(', '))}}</p>`);
  if (n.cocitations) rows.push(`<p>co-citation hole ×${{n.cocitations}}</p>`);
  if (n.citations) rows.push(`<p>${{n.citations}} citations</p>`);
  if (n.url) rows.push(`<p><a href="${{escapeAttr(n.url)}}" target="_blank" rel="noreferrer">open source</a></p>`);
  const near = data.edges.filter(e => e.source === n.id || e.target === n.id).slice(0, 12)
    .map(e => `${{e.source === n.id ? '→' : '←'}} ${{escapeHtml(e.source === n.id ? e.target : e.source)}} <small>${{escapeHtml(e.label || e.type)}}</small>`);
  if (near.length) rows.push(`<p>${{near.join('<br>')}}</p>`);
  detail.innerHTML = rows.join('');
}}
function escapeHtml(s) {{ return String(s || '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c])); }}
function escapeAttr(s) {{ return escapeHtml(s).replace(/'/g, '&#39;'); }}

let width = 1000, height = 700;
const world = {{width: 1600, height: 1000}};
function configureWorld() {{
  updateSize();
  const aspect = Math.max(.75, Math.min(2.35, width / Math.max(1, height)));
  const base = Math.max(920, Math.sqrt(data.nodes.length || 1) * 90);
  world.width = Math.round(Math.max(1320, base * Math.sqrt(aspect) * 1.35));
  world.height = Math.round(Math.max(860, base / Math.sqrt(aspect) * 1.05));
}}
configureWorld();
const nodes = data.nodes.map((n, i) => Object.assign({{}}, n, {{_i: i, x: 0, y: 0, vx: 0, vy: 0}}));
const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
const edges = data.edges.map(e => Object.assign({{}}, e, {{a:byId[e.source], b:byId[e.target]}})).filter(e => e.a && e.b);
const degree = Object.fromEntries(nodes.map(n => [n.id, 0]));
for (const e of edges) {{ degree[e.source] += 1; degree[e.target] += 1; }}
const searchNodes = nodes.filter(n => n.type === 'search').sort((a, b) => a.id.localeCompare(b.id));
const corpusNodes = nodes.filter(n => n.in_corpus || n.type === 'paper').sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0) || a.id.localeCompare(b.id));
const corpusIndex = new Map(corpusNodes.map((n, i) => [n.id, i]));
const searchIndex = new Map(searchNodes.map((n, i) => [n.id, i]));
const epistemicNodes = nodes.filter(n => n.layer === 'epistemic').sort((a, b) => a.id.localeCompare(b.id));
const epistemicIndex = new Map(epistemicNodes.map((n, i) => [n.id, i]));
const primary = new Map();
for (const e of edges) {{
  for (const [n, other] of [[e.a, e.b], [e.b, e.a]]) {{
    const have = primary.get(n.id);
    const score = (other.type === 'topic' ? 4 : other.type === 'search' ? 3 : other.in_corpus ? 2 : 1);
    if (!have || score > have.score) primary.set(n.id, {{node: other, score}});
  }}
}}
const leafBuckets = new Map();
for (const n of nodes) {{
  const isLeaf = ['external', 'recent', 'hole'].includes(n.type) || (!n.in_corpus && degree[n.id] <= 1);
  if (!isLeaf) continue;
  const p = primary.get(n.id)?.node;
  if (!p) continue;
  if (!leafBuckets.has(p.id)) leafBuckets.set(p.id, []);
  leafBuckets.get(p.id).push(n);
}}
for (const bucket of leafBuckets.values()) {{
  bucket.sort((a, b) => a.id.localeCompare(b.id));
  const total = bucket.length;
  bucket.forEach((n, i) => {{
    const spread = Math.min(Math.PI * 1.65, Math.max(Math.PI * .55, total * .11));
    const start = -spread / 2;
    const base = angleFor(primary.get(n.id)?.node || n);
    n._leafAngle = base + start + spread * ((i + .5) / Math.max(1, total));
    n._leafRadius = 105 + 22 * Math.floor(i / 22) + 4 * (i % 3);
  }});
}}
let view = {{x: 0, y: 0, k: 1}};
let selectedId = 'topic';
let filterQuery = '';
let filterHits = new Set();
let layerMode = 'field';
let userMoved = false;
let frameCount = 0;

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
function hash01(s) {{
  let h = 2166136261;
  for (let i = 0; i < String(s).length; i++) {{
    h ^= String(s).charCodeAt(i);
    h = Math.imul(h, 16777619);
  }}
  return ((h >>> 0) % 10000) / 10000;
}}
function angleFor(n) {{
  if (!n) return 0;
  if (n.type === 'search') {{
    const i = searchIndex.get(n.id) ?? 0;
    return -Math.PI * .82 + ((i + .5) / Math.max(1, searchNodes.length)) * Math.PI * 1.64;
  }}
  if (n.id === 'topic') return -Math.PI / 2;
  if (n.layer === 'epistemic') {{
    const i = epistemicIndex.get(n.id) ?? 0;
    return -Math.PI * .08 + i * 2.399963229728653;
  }}
  const i = corpusIndex.get(n.id) ?? n._i ?? 0;
  return i * 2.399963229728653 + hash01(n.id) * .7;
}}
function anchor(n) {{
  const cx = world.width / 2, cy = world.height / 2;
  if (n.id === 'topic' || n.type === 'topic') return {{x: cx, y: cy * .45}};
  if (n.type === 'objective') return {{x: cx, y: cy * 1.28}};
  if (n.layer === 'epistemic') {{
    const i = epistemicIndex.get(n.id) ?? 0;
    const a = angleFor(n);
    const band = n.type === 'question' ? .14 : n.type === 'claim' ? .23 : .32;
    return {{x: cx + Math.cos(a) * world.width * band,
             y: cy * 1.27 + Math.sin(a) * world.height * band * .54}};
  }}
  if (n.type === 'search') {{
    const a = angleFor(n);
    return {{x: cx + Math.cos(a) * world.width * .26, y: cy * .58 + Math.sin(a) * world.height * .24}};
  }}
  if (['external', 'recent', 'hole'].includes(n.type) && primary.has(n.id)) {{
    const p = primary.get(n.id).node;
    const pa = anchor(p);
    const a = n._leafAngle ?? angleFor(n);
    const r = n._leafRadius ?? 130;
    return {{x: pa.x + Math.cos(a) * r, y: pa.y + Math.sin(a) * r}};
  }}
  const i = corpusIndex.get(n.id) ?? n._i ?? 0;
  const a = angleFor(n);
  const r = 70 + Math.sqrt(i + 1) * 42;
  return {{x: cx + Math.cos(a) * r, y: cy * .68 + Math.sin(a) * r * .62}};
}}
for (const n of nodes) {{
  const a = anchor(n);
  n.x = a.x + (hash01(n.id + ':x') - .5) * 42;
  n.y = a.y + (hash01(n.id + ':y') - .5) * 42;
}}
function updateSize() {{
  width = svg.clientWidth || 1000;
  height = svg.clientHeight || 700;
  svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
}}
function updateZoomLabel() {{
  document.getElementById('zoomLabel').textContent = `${{Math.round(view.k * 100)}}%`;
}}
function graphBounds() {{
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of nodes) {{
    if (!layerVisible(n)) continue;
    minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
    maxX = Math.max(maxX, n.x); maxY = Math.max(maxY, n.y);
  }}
  if (!Number.isFinite(minX)) return {{minX:0,minY:0,maxX:world.width,maxY:world.height}};
  return {{minX, minY, maxX, maxY}};
}}
function fitScale() {{
  const b = graphBounds();
  const pad = 70;
  const spanX = Math.max(100, b.maxX - b.minX + pad * 2);
  const spanY = Math.max(100, b.maxY - b.minY + pad * 2);
  return clamp(Math.min(width / spanX, height / spanY), 0.08, 2.2);
}}
function minimumZoom() {{ return Math.max(0.1, fitScale() * .82); }}
function fitGraph(markUserMoved = true) {{
  updateSize();
  const b = graphBounds();
  view.k = fitScale();
  view.x = (width - (b.minX + b.maxX) * view.k) / 2;
  view.y = (height - (b.minY + b.maxY) * view.k) / 2;
  if (markUserMoved) userMoved = true;
  updateZoomLabel();
  render();
}}
function zoomAt(clientX, clientY, factor) {{
  updateSize();
  const rect = svg.getBoundingClientRect();
  const sx = clientX - rect.left;
  const sy = clientY - rect.top;
  const wx = (sx - view.x) / view.k;
  const wy = (sy - view.y) / view.k;
  view.k = clamp(view.k * factor, minimumZoom(), 7);
  view.x = sx - wx * view.k;
  view.y = sy - wy * view.k;
  userMoved = true;
  updateZoomLabel();
  render();
}}
function zoomCenter(factor) {{ zoomAt(svg.getBoundingClientRect().left + width / 2, svg.getBoundingClientRect().top + height / 2, factor); }}
function focusNode(n, scale = Math.max(view.k, 1.4)) {{
  updateSize();
  selectedId = n.id;
  view.k = clamp(scale, 0.2, 5);
  view.x = width / 2 - n.x * view.k;
  view.y = height / 2 - n.y * view.k;
  userMoved = true;
  updateZoomLabel();
  show(n);
  render();
}}
function nodeMatches(n) {{
  if (!filterQuery) return true;
  return filterHits.has(n.id);
}}
function edgeDim(e) {{
  if (!filterQuery) return false;
  return !(filterHits.has(e.source) || filterHits.has(e.target));
}}
function layerVisible(n) {{
  return layerMode === 'all' || n.layer === layerMode || n.type === 'topic';
}}
function tick() {{
  for (const n of nodes) {{
    if (!layerVisible(n)) continue;
    const a = anchor(n);
    const strength = n.type === 'topic' ? .08 : n.type === 'search' ? .035 : ['external','recent','hole'].includes(n.type) ? .026 : .012;
    n.vx += (a.x - n.x) * strength;
    n.vy += (a.y - n.y) * strength;
  }}
  for (let i=0;i<nodes.length;i++) for (let j=i+1;j<nodes.length;j++) {{
    const a=nodes[i], b=nodes[j], dx=a.x-b.x, dy=a.y-b.y, d2=Math.max(30, dx*dx+dy*dy);
    if (!layerVisible(a) || !layerVisible(b)) continue;
    const min = radius(a) + radius(b) + 10;
    const d = Math.sqrt(d2);
    const charge = (d < min * 2 ? 120 : 34) / d2;
    const f = charge + (d < min ? (min - d) * .025 : 0);
    a.vx += dx*f; a.vy += dy*f; b.vx -= dx*f; b.vy -= dy*f;
  }}
  for (const e of edges) {{
    if (!layerVisible(e.a) || !layerVisible(e.b)) continue;
    const dx=e.b.x-e.a.x, dy=e.b.y-e.a.y, d=Math.max(1, Math.hypot(dx,dy));
    const leafEdge = ['external','recent','hole'].includes(e.a.type) || ['external','recent','hole'].includes(e.b.type);
    const want = e.type === 'search' ? 260 : e.type === 'found' ? 170 : leafEdge ? 115 : 145;
    const f=(d-want)*.006; const ux=dx/d, uy=dy/d;
    e.a.vx += ux*f; e.a.vy += uy*f; e.b.vx -= ux*f; e.b.vy -= uy*f;
  }}
  for (const n of nodes) {{
    if (!layerVisible(n)) continue;
    const cx = world.width / 2, cy = world.height / 2;
    const rx = world.width * .5, ry = world.height * .5;
    const dx = n.x - cx, dy = n.y - cy;
    const q = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
    if (q > 1) {{
      const pull = (Math.sqrt(q) - 1) * .12;
      n.vx -= dx * pull;
      n.vy -= dy * pull;
    }}
    n.vx = clamp(n.vx * .78, -18, 18);
    n.vy = clamp(n.vy * .78, -18, 18);
    n.x += n.vx; n.y += n.vy;
  }}
}}
function render() {{
  updateSize();
  svg.innerHTML = '';
  const gWorld = document.createElementNS('http://www.w3.org/2000/svg','g');
  gWorld.setAttribute('transform', `translate(${{view.x}},${{view.y}}) scale(${{view.k}})`);
  svg.appendChild(gWorld);
  for (const e of edges) {{
    if (!layerVisible(e.a) || !layerVisible(e.b)) continue;
    const line = document.createElementNS('http://www.w3.org/2000/svg','line');
    line.setAttribute('class', `edge ${{e.type}}${{edgeDim(e) ? ' dim' : ''}}`);
    line.setAttribute('x1', e.a.x); line.setAttribute('y1', e.a.y);
    line.setAttribute('x2', e.b.x); line.setAttribute('y2', e.b.y);
    gWorld.appendChild(line);
  }}
  for (const n of nodes) {{
    if (!layerVisible(n)) continue;
    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    const isHit = filterQuery && filterHits.has(n.id);
    const isDim = filterQuery && !isHit;
    const isSelected = selectedId === n.id;
    g.setAttribute('class', `node ${{n.type}}${{isDim ? ' dim' : ''}}${{isHit ? ' hit' : ''}}${{isSelected ? ' selected' : ''}}`);
    g.setAttribute('data-status', n.status || '');
    g.setAttribute('transform', `translate(${{n.x}},${{n.y}})`);
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('r', radius(n));
    c.addEventListener('click', () => {{ selectedId = n.id; show(n); render(); }});
    c.addEventListener('dblclick', ev => {{ ev.stopPropagation(); focusNode(n); }});
    g.appendChild(c);
    const showLabel = isSelected || isHit || n.type === 'topic'
      || (view.k > .55 && (n.type === 'search' || degree[n.id] >= 5))
      || (view.k > .85 && (n.type === 'hole' || n.in_corpus))
      || view.k > 1.35;
    if (showLabel) {{
      const text = document.createElementNS('http://www.w3.org/2000/svg','text');
      text.setAttribute('x', radius(n) + 4); text.setAttribute('y', 4);
      text.textContent = n.label || n.id;
      g.appendChild(text);
    }}
    gWorld.appendChild(g);
  }}
}}
let raf;
function animate() {{
  tick();
  render();
  frameCount += 1;
  if (frameCount === 100 && !userMoved) fitGraph(false);
  raf = requestAnimationFrame(animate);
}}
fitGraph(false);
animate(); setTimeout(() => cancelAnimationFrame(raf), 9000);
document.getElementById('zoomIn').addEventListener('click', () => zoomCenter(1.35));
document.getElementById('zoomOut').addEventListener('click', () => zoomCenter(1 / 1.35));
document.getElementById('fitGraph').addEventListener('click', () => fitGraph(true));
svg.addEventListener('wheel', ev => {{
  ev.preventDefault();
  zoomAt(ev.clientX, ev.clientY, Math.exp(-ev.deltaY * 0.0012));
}}, {{passive:false}});
let drag = null;
svg.addEventListener('pointerdown', ev => {{
  drag = {{x: ev.clientX, y: ev.clientY}};
  svg.classList.add('dragging');
  svg.setPointerCapture(ev.pointerId);
}});
svg.addEventListener('pointermove', ev => {{
  if (!drag) return;
  view.x += ev.clientX - drag.x;
  view.y += ev.clientY - drag.y;
  drag = {{x: ev.clientX, y: ev.clientY}};
  userMoved = true;
  render();
}});
function endDrag(ev) {{
  drag = null;
  svg.classList.remove('dragging');
  try {{ svg.releasePointerCapture(ev.pointerId); }} catch (_e) {{}}
}}
svg.addEventListener('pointerup', endDrag);
svg.addEventListener('pointercancel', endDrag);
svg.addEventListener('dblclick', ev => zoomAt(ev.clientX, ev.clientY, 1.8));
let resizeWidth = width, resizeHeight = height;
function handleResize() {{
  const centerX = (resizeWidth / 2 - view.x) / Math.max(view.k, .001);
  const centerY = (resizeHeight / 2 - view.y) / Math.max(view.k, .001);
  const oldWidth = resizeWidth, oldHeight = resizeHeight;
  updateSize();
  const widthChange = Math.abs(width - oldWidth) / Math.max(1, oldWidth);
  const heightChange = Math.abs(height - oldHeight) / Math.max(1, oldHeight);
  const crossedMobile = (oldWidth <= 840) !== (width <= 840);
  const majorResize = widthChange > .25 || heightChange > .25 || crossedMobile;
  if (!userMoved || majorResize) {{
    fitGraph(false);
  }} else {{
    view.k = Math.max(view.k, minimumZoom());
    view.x = width / 2 - centerX * view.k;
    view.y = height / 2 - centerY * view.k;
    updateZoomLabel();
    render();
  }}
  resizeWidth = width;
  resizeHeight = height;
}}
const resizeObserver = new ResizeObserver(handleResize);
resizeObserver.observe(svg);
document.getElementById('search').addEventListener('input', e => {{
  filterQuery = e.target.value.trim().toLowerCase();
  filterHits = new Set();
  for (const n of nodes) {{
    const hay = [n.id,n.label,n.title,(n.authors||[]).join(' ')].join(' ').toLowerCase();
    if (filterQuery && hay.includes(filterQuery)) filterHits.add(n.id);
  }}
  render();
}});
for (const button of document.querySelectorAll('[data-layer]')) {{
  button.addEventListener('click', () => {{
    layerMode = button.dataset.layer;
    for (const peer of document.querySelectorAll('[data-layer]')) {{
      peer.classList.toggle('active', peer === button);
    }}
    if (!layerVisible(byId[selectedId] || {{type:'topic', layer:'field'}})) selectedId = 'topic';
    fitGraph(true);
    render();
  }});
}}
document.addEventListener('keydown', ev => {{
  if (ev.key === '+' || ev.key === '=') zoomCenter(1.25);
  if (ev.key === '-') zoomCenter(1 / 1.25);
  if (ev.key === '0') fitGraph(true);
}});
show(byId.topic || nodes[0]);
</script>
"""


def write_graph_view(map_data: dict, corpus, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_graph_data(map_data, corpus)
    data_path = out / "research-graph-data.json"
    html_path = out / "research-graph.html"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_html_template(data), encoding="utf-8")
    return {
        "html": str(html_path),
        "data": str(data_path),
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "counts": data["counts"],
    }


def write_graph_view_from_files(research_dir: str | Path) -> dict:
    research_dir = Path(research_dir)
    from spiral.research_corpus import Corpus

    map_data = json.loads((research_dir / "research-map.json").read_text(encoding="utf-8"))
    corpus = Corpus(research_dir / "corpus")
    return write_graph_view(map_data, corpus, research_dir)
