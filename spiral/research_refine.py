"""spiral research --refine — rebuild an existing LaTeX paper in its field's own voice.

The pipeline: survey the project on disk → understand the work → build a corpus of
the field's actual literature → learn its style from primary sources → read the
figures with a *local* vision model → rewrite section by section → enrich with
corpus-verified connections → emit a submittable PDF and a blue-edit diff PDF.

Two hard commitments, in the spirit of everything else here:

1. **The original is never touched.** Every write lands under ``spiral-refined/``.
   The source tree is opened read-only, full stop.
2. **The model proposes, deterministic gates decide.** A rewritten section that
   invents a number, drops an equation, or loses a label is rejected and the
   original text kept. An enrichment sentence enters the paper only when its
   anchor quote is literally present in the cited corpus paper. Ideas that
   cannot be verified are reported in ``suggestions.md`` — never asserted.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from spiral.config import Config
from spiral.llm import Ollama
from spiral.research_corpus import Corpus
from spiral.research_loop import _extract_json
from spiral.research_writer import (
    bibtex_from_corpus,
    compile_pdf,
    corpus_style_guide,
    remove_suspicious_overlap_sentences,
)


class RefineError(RuntimeError):
    """A condition the pipeline cannot proceed past (no project, no model...)."""


# --------------------------------------------------------------------------- survey

_TEX_SKIP_DIRS = {".git", "spiral-refined", "spiral-research", "__pycache__",
                  ".spiral", "node_modules", ".venv", "venv"}


@dataclass
class ProjectSurvey:
    root: Path
    main_tex: Path
    included: list[Path] = field(default_factory=list)
    bib_files: list[Path] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)


def _tex_files(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*.tex")):
        if any(part in _TEX_SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def find_main_tex(root: Path) -> Path | None:
    """The compilable entry point: has ``\\documentclass`` *and* ``\\begin{document}``.
    Several candidates → prefer one named main.tex, then the largest."""
    candidates = []
    for p in _tex_files(root):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\\documentclass" in t and "\\begin{document}" in t:
            candidates.append(p)
    if not candidates:
        return None
    for p in candidates:
        if p.name.lower() == "main.tex":
            return p
    return max(candidates, key=lambda p: p.stat().st_size)


_INPUT_RE = re.compile(r"(?<!\\)%.*$|\\(?:input|include)\{([^}]+)\}", re.MULTILINE)


def flatten_tex(path: Path, *, _seen: set | None = None) -> str:
    """Inline every ``\\input``/``\\include`` (comment-aware, cycle-safe) so the
    whole document is one string — for the model, the gates, and the rebuild."""
    _seen = _seen if _seen is not None else set()
    key = str(path.resolve())
    if key in _seen or not path.is_file():
        return ""
    _seen.add(key)
    text = path.read_text(encoding="utf-8", errors="ignore")

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name is None:            # a comment line matched — keep it verbatim
            return m.group(0)
        child = (path.parent / name)
        if child.suffix != ".tex":
            child = child.with_suffix(".tex")
        return flatten_tex(child, _seen=_seen) if child.is_file() else m.group(0)

    return _INPUT_RE.sub(_sub, text)


_GRAPHIC_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_GPATH_RE = re.compile(r"\\graphicspath\{((?:\{[^}]*\})+)\}")
_IMG_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")


def resolve_figures(tex: str, base: Path) -> list[Path]:
    gpaths = [base]
    for m in _GPATH_RE.finditer(tex):
        gpaths += [base / g.strip("{}") for g in re.findall(r"\{([^}]*)\}", m.group(1))]
    found: list[Path] = []
    for name in _GRAPHIC_RE.findall(tex):
        for d in gpaths:
            cand = d / name.strip()
            trials = [cand] if cand.suffix else [cand.with_suffix(e) for e in _IMG_EXTS]
            hit = next((t for t in trials if t.is_file()), None)
            if hit and hit not in found:
                found.append(hit)
                break
    return found


def survey_project(root: Path, *, out_name: str = "spiral-refined") -> ProjectSurvey:
    root = Path(root).resolve()
    main = find_main_tex(root)
    if main is None:
        raise RefineError(
            f"no LaTeX project found under {root} — --refine needs a .tex file with "
            "\\documentclass and \\begin{document}")
    flat = flatten_tex(main)
    included = []
    for m in _INPUT_RE.finditer(main.read_text(encoding="utf-8", errors="ignore")):
        if m.group(1):
            child = (main.parent / m.group(1))
            if child.suffix != ".tex":
                child = child.with_suffix(".tex")
            if child.is_file():
                included.append(child)
    bibs = [p for p in sorted(root.rglob("*.bib"))
            if not any(part in _TEX_SKIP_DIRS for part in p.relative_to(root).parts)]
    data = [p for p in sorted(root.rglob("*"))
            if p.suffix in {".csv", ".tsv", ".dat", ".json"} and p.is_file()
            and not any(part in _TEX_SKIP_DIRS for part in p.relative_to(root).parts)][:40]
    return ProjectSurvey(root=root, main_tex=main, included=included, bib_files=bibs,
                         figures=resolve_figures(flat, main.parent), data_files=data)


# ------------------------------------------------------------------- document parsing

_SECTION_RE = re.compile(r"(\\section\*?\{[^}]*\})")


def split_document(flat: str) -> dict:
    """``{preamble, opening, sections: [{head, tex}], closing}`` — opening is
    everything from ``\\begin{document}`` to the first ``\\section`` (title block,
    abstract); closing is from the bibliography/appendix boundary to the end."""
    i = flat.find("\\begin{document}")
    if i < 0:
        raise RefineError("main tex has no \\begin{document}")
    preamble, rest = flat[:i], flat[i:]
    end = rest.rfind("\\end{document}")
    body, tail = (rest[:end], rest[end:]) if end >= 0 else (rest, "")
    close_at = len(body)
    for marker in ("\\bibliography{", "\\printbibliography", "\\begin{thebibliography}",
                   "\\appendix"):
        j = body.find(marker)
        if 0 <= j < close_at:
            close_at = j
    closing = body[close_at:] + tail
    body = body[:close_at]
    parts = _SECTION_RE.split(body)
    opening = parts[0]
    sections = [{"head": parts[k], "tex": parts[k + 1] if k + 1 < len(parts) else ""}
                for k in range(1, len(parts), 2)]
    return {"preamble": preamble, "opening": opening, "sections": sections,
            "closing": closing}


# ------------------------------------------------------------------ deterministic gates

_STRIP_ARGS_RE = re.compile(
    r"\\(?:cite\w*|ref|eqref|autoref|cref|label|includegraphics|bibitem|input|include|"
    r"graphicspath|documentclass|usepackage|bibliography(?:style)?)\s*(?:\[[^\]]*\])?\{[^}]*\}")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numeric_tokens(tex: str) -> set[str]:
    """Numbers that carry meaning to a reader — reference/label/graphics arguments
    stripped first so citekeys and file names don't count."""
    return set(_NUM_RE.findall(_STRIP_ARGS_RE.sub(" ", tex)))


_MATH_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}(.*?)"
    r"\\end\{\1\}|\\\[(.*?)\\\]", re.DOTALL)


def display_math(tex: str) -> list[str]:
    out = []
    for m in _MATH_RE.finditer(tex):
        body = m.group(2) if m.group(2) is not None else m.group(3)
        out.append(re.sub(r"\s+", "", body or ""))
    return sorted(out)


def cite_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for m in re.finditer(r"\\cite\w*\s*(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def labels_of(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", tex))


def rebuild_violations(original: str, rewritten: str, *, allowed_numbers: set[str],
                       allowed_cites: set[str]) -> list[str]:
    """Why a rewritten fragment must be rejected. Empty list == acceptable."""
    problems = []
    new_numbers = numeric_tokens(rewritten) - allowed_numbers
    if new_numbers:
        problems.append("invented numbers not present in the original: "
                        + ", ".join(sorted(new_numbers)[:8]))
    if display_math(rewritten) != display_math(original):
        problems.append("display math environments altered — they must be copied verbatim")
    if labels_of(rewritten) != labels_of(original):
        problems.append("\\label set changed")
    stray = cite_keys(rewritten) - allowed_cites
    if stray:
        problems.append("citations of unknown keys: " + ", ".join(sorted(stray)[:6]))
    orig_figs = sorted(_GRAPHIC_RE.findall(original))
    new_figs = sorted(_GRAPHIC_RE.findall(rewritten))
    if orig_figs != new_figs:
        problems.append("\\includegraphics set changed")
    return problems


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def anchored_in(paper, anchor: str) -> bool:
    """An enrichment claim is admissible only if its anchor quote literally appears
    in the cited paper — the same standard as grounded reading notes."""
    anchor = _norm(anchor or "")
    if len(anchor) < 25:
        return False
    return anchor in _norm(f"{paper.title}\n{paper.abstract}\n{paper.text or ''}")


# --------------------------------------------------------------------------- the run


class RefineRun:
    def __init__(self, project_dir: str | Path, *, cfg: Config | None = None,
                 ol: Ollama | None = None, on=None, out_name: str = "spiral-refined"):
        self.cfg = cfg or Config.load()
        self.ol = ol or Ollama(self.cfg.ollama_url, providers=self.cfg.providers)
        self.on = on or (lambda msg: None)
        self.root = Path(project_dir).resolve()
        self.out = self.root / out_name
        self.out.mkdir(parents=True, exist_ok=True)
        self.corpus = Corpus(self.out / "corpus")
        self.tokens = 0
        self.report: dict = {"stages": [], "kept_verbatim": [], "dropped": [],
                             "enriched": [], "suggestions": 0}

    # -- plumbing ---------------------------------------------------------------
    def _say(self, msg: str):
        self.on(msg)

    def _stage(self, name: str, detail: str = ""):
        self.report["stages"].append({"stage": name, "detail": detail})
        self._say(f"{name}{' · ' + detail if detail else ''}")

    def _json(self, system: str, user: str, *, role: str = "critic",
              max_tokens: int = 4096) -> dict:
        spec = getattr(self.cfg, role, None) or self.cfg.critic
        sys_msg = (system + "\nReturn exactly one compact JSON object. No Markdown, "
                   "no prose outside JSON.")
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": user}]
        for attempt in (1, 2):
            try:
                res = self.ol.chat(
                    spec.name, messages, think=getattr(spec, "think", False),
                    num_predict=max_tokens, num_ctx=getattr(spec, "num_ctx", None),
                    keep_alive=self.cfg.keep_alive, temperature=0.3, fmt="json")
            except Exception as e:
                raise RefineError(f"model call failed ({spec.name}): {e}") from e
            self.tokens += (getattr(res, "prompt_tokens", 0) or 0) \
                + (getattr(res, "completion_tokens", 0) or 0)
            err = (res.raw or {}).get("error")
            if err:
                raise RefineError(f"model call failed ({spec.name}): {err}")
            data = _extract_json((res.text or "").strip())
            if data:
                return data
            messages.append({"role": "assistant", "content": (res.text or "")[:2000]})
            messages.append({"role": "user", "content":
                             "That was not a parseable JSON object. Reply with ONLY the "
                             "JSON object, nothing else."})
        return {}

    # -- stages -----------------------------------------------------------------
    def understand(self, flat: str) -> dict:
        body = flat[:18000] + ("\n…\n" + flat[-6000:] if len(flat) > 24000 else "")
        got = self._json(
            "You are reading a LaTeX research manuscript. Extract what it IS — no "
            "judgement, no invention. JSON: {\"title\":str, \"field\":str, "
            "\"summary\":str, \"contributions\":[str], \"key_results\":[str], "
            "\"terminology\":[str], \"arxiv_categories\":[str], \"queries\":[str]} — "
            "queries: 3-5 short arXiv keyword searches (3-6 words) that would surface "
            "the closest related literature.",
            f"MANUSCRIPT:\n{body}", role="critic", max_tokens=4096)
        if not got.get("queries"):
            raise RefineError("could not understand the manuscript (no search queries)")
        return got

    def build_corpus(self, understanding: dict) -> None:
        cats = [c for c in (understanding.get("arxiv_categories") or [])
                if re.fullmatch(r"[a-z-]+(?:\.[A-Za-z-]+)?", str(c))][:4]
        queries = [str(q) for q in understanding.get("queries", [])][:4]
        for i, q in enumerate(queries):
            if i:
                time.sleep(3)                       # arXiv politeness — hard lesson
            added = self.corpus.build(q, k=6, categories=cats or None,
                                      on=lambda m: self._say(f"  {m}"))
            self._say(f"  search “{q[:48]}” → +{len(added)}")
        if self.corpus.papers:
            try:
                self.corpus.graph_deepen(rounds=1, min_cocite=2, cap=16,
                                         on=lambda m: self._say(f"  {m}"))
            except Exception:
                pass                                 # graph is enrichment, not a gate
        self.corpus.save()

    def read_figures(self, figures: list[Path]) -> dict[str, dict]:
        """Local vision only — figures are unpublished work and never leave the
        machine. Missing vision model → captions stay as they are, honestly noted."""
        if not figures:
            self._stage("figures", "none referenced")
            return {}
        from spiral.visual_review import choose_vision_model
        try:
            model = choose_vision_model(self.cfg, self.ol)
        except Exception:
            model = ""
        if not model or model in (getattr(self.ol, "providers", {}) or {}):
            self._stage("figures", "no local vision model — figure pass skipped")
            return {}
        notes: dict[str, dict] = {}
        for fig in figures[:12]:
            img = self._rasterize(fig)
            if img is None:
                continue
            b64 = base64.b64encode(img.read_bytes()).decode("ascii")
            try:
                res = self.ol.chat(
                    model,
                    [{"role": "system", "content":
                      "Describe this scientific figure for the paper's author. JSON: "
                      "{\"description\":str, \"shows\":str, \"suggested_caption\":str}. "
                      "Return exactly one compact JSON object."},
                     {"role": "user", "content": f"FILE: {fig.name}", "images": [b64]}],
                    think=False, num_predict=1536, temperature=0.1, fmt="json",
                    keep_alive=self.cfg.keep_alive)
            except Exception:
                continue
            self.tokens += (getattr(res, "prompt_tokens", 0) or 0) \
                + (getattr(res, "completion_tokens", 0) or 0)
            data = _extract_json(res.text or "")
            if data.get("description"):
                notes[fig.name] = data
        self._stage("figures", f"{len(notes)}/{len(figures)} described (local vision)")
        return notes

    def _rasterize(self, fig: Path) -> Path | None:
        if fig.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return fig
        if fig.suffix.lower() in {".pdf", ".eps"}:
            out = self.out / "figs-png" / (fig.stem + ".png")
            out.parent.mkdir(exist_ok=True)
            if out.is_file():
                return out
            for cmd in (["sips", "-s", "format", "png", str(fig), "--out", str(out)],
                        ["pdftoppm", "-png", "-r", "110", "-singlefile", str(fig),
                         str(out.with_suffix(""))]):
                if shutil.which(cmd[0]):
                    try:
                        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
                        if out.is_file():
                            return out
                    except Exception:
                        continue
        return None

    def rebuild(self, doc: dict, understanding: dict, style: str,
                fig_notes: dict[str, dict]) -> dict:
        """Section-by-section rewrite behind the gates. A section that cannot be
        rewritten acceptably stays verbatim — recorded, never silent."""
        full = doc["opening"] + "".join(s["head"] + s["tex"] for s in doc["sections"])
        full_numbers = numeric_tokens(full)
        all_cites = cite_keys(full + doc["closing"])
        terms = ", ".join(str(t) for t in understanding.get("terminology", [])[:16])
        rules = (
            "Rewrite the LaTeX fragment for clarity and fit with the field's style. "
            "HARD RULES — violations are rejected mechanically:\n"
            "1. Copy every display math environment byte-for-byte.\n"
            "2. Never add, remove, or alter a number.\n"
            "3. Keep every \\cite, \\ref, \\label and \\includegraphics exactly.\n"
            "4. No new claims, no strengthened results, no invented content.\n"
            "5. Prose only: reorganise sentences, tighten wording, match the register.\n"
            f"FIELD STYLE:\n{style[:2600]}\n"
            + (f"TERMINOLOGY IN USE: {terms}\n" if terms else "")
            + "JSON: {\"tex\": \"<the rewritten fragment>\"}")

        def _rewrite(fragment: str, name: str) -> str:
            if len(fragment) > 14000 or len(fragment.strip()) < 40:
                self.report["kept_verbatim"].append({"part": name, "why": "size"})
                return fragment
            fig_ctx = "\n".join(
                f"FIGURE {n}: {v.get('description', '')[:300]}"
                for n, v in fig_notes.items() if n.split(".")[0] in fragment)
            user = (f"FRAGMENT ({name}):\n{fragment}"
                    + (f"\n\nWHAT THE FIGURES SHOW:\n{fig_ctx}" if fig_ctx else ""))
            allowed = numeric_tokens(fragment) | full_numbers
            history = ""
            for attempt in (1, 2):
                got = self._json(rules, user + history, role="critic", max_tokens=8192)
                new = str(got.get("tex") or "")
                if not new.strip():
                    break
                problems = rebuild_violations(fragment, new, allowed_numbers=allowed,
                                              allowed_cites=all_cites)
                if not problems:
                    return new
                history = ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: "
                           + "; ".join(problems) + " — fix exactly that.")
            self.report["kept_verbatim"].append({"part": name, "why": "gates"})
            return fragment

        out = dict(doc)
        out["opening"] = _rewrite(doc["opening"], "opening/abstract")
        out["sections"] = [
            {"head": s["head"],
             "tex": _rewrite(s["tex"], s["head"])}
            for s in doc["sections"]]
        kept = len(self.report["kept_verbatim"])
        self._stage("rebuild", f"{len(doc['sections']) + 1} fragments · "
                               f"{kept} kept verbatim")
        return out

    def enrich(self, doc: dict) -> tuple[dict, list[dict]]:
        """Corpus-verified connections only. Anything the anchors cannot carry goes
        to suggestions.md — reported, never asserted."""
        papers = list(self.corpus.papers.values())
        if not papers:
            self._stage("enrich", "no corpus — skipped")
            return doc, []
        _, keymap = bibtex_from_corpus(papers)
        digest = "\n".join(
            f"[{p.bare_id}] {p.title}\n{(p.abstract or p.text or '')[:500]}"
            for p in papers[:24])
        body = doc["opening"] + "".join(s["head"] + s["tex"] for s in doc["sections"])
        got = self._json(
            "You connect a manuscript to its literature — strictly, no stretching. "
            "For each connection give the corpus paper id and an ANCHOR: a verbatim "
            "quote (>=25 chars) from that paper that supports the sentence. Also list "
            "stronger angles the author could push, each with grounding ids+anchor. "
            "JSON: {\"connections\":[{\"sentence\":str,\"cite_id\":str,\"anchor\":str}],"
            "\"stronger_angles\":[{\"idea\":str,\"cite_id\":str,\"anchor\":str}]}",
            f"MANUSCRIPT (refined draft):\n{body[:16000]}\n\nCORPUS:\n{digest}",
            role="escalation", max_tokens=6144)

        verified: list[dict] = []
        suggestions: list[dict] = []
        base_numbers = numeric_tokens(body)
        for c in (got.get("connections") or []):
            pid = re.sub(r"^arXiv:", "", str(c.get("cite_id", ""))).split("v")[0]
            paper = self.corpus.papers.get(pid)
            sent = str(c.get("sentence") or "").strip()
            anchor = str(c.get("anchor") or "")
            ok = bool(paper and sent) and anchored_in(paper, anchor)
            if ok and not (numeric_tokens(sent) - base_numbers - numeric_tokens(anchor)):
                verified.append({"sentence": sent, "paper": paper,
                                 "key": keymap.get(pid, ""), "anchor": anchor})
            else:
                suggestions.append({"kind": "connection", "text": sent,
                                    "cite_id": str(c.get("cite_id", "")),
                                    "why_dropped": "anchor not found in source"
                                    if not ok else "introduces unanchored numbers"})
        for a in (got.get("stronger_angles") or []):
            pid = re.sub(r"^arXiv:", "", str(a.get("cite_id", ""))).split("v")[0]
            paper = self.corpus.papers.get(pid)
            grounded = bool(paper) and anchored_in(paper, str(a.get("anchor") or ""))
            suggestions.append({"kind": "stronger-angle",
                                "text": str(a.get("idea") or "").strip(),
                                "cite_id": str(a.get("cite_id", "")),
                                "grounded": grounded})
        if verified and doc["sections"]:
            para = ("\n\n" + "\n".join(
                f"{v['sentence'].rstrip('.')}~\\cite{{{v['key']}}}."
                for v in verified if v["key"]) + "\n")
            doc["sections"][-1]["tex"] = doc["sections"][-1]["tex"].rstrip() + para
        self.report["enriched"] = [
            {"sentence": v["sentence"][:160], "cite": v["paper"].bare_id}
            for v in verified]
        self.report["dropped"] += [s for s in suggestions if s["kind"] == "connection"]
        self.report["suggestions"] = len(suggestions)
        self._stage("enrich", f"{len(verified)} verified connections woven · "
                              f"{len(suggestions)} to suggestions.md")
        return doc, suggestions

    # -- emit --------------------------------------------------------------------
    def _assemble(self, doc: dict, survey: ProjectSurvey) -> str:
        closing = doc["closing"]
        papers = list(self.corpus.papers.values())
        if papers:
            bibtex, keymap = bibtex_from_corpus(papers)
            body_cites = cite_keys(doc["opening"]
                                   + "".join(s["head"] + s["tex"] for s in doc["sections"]))
            new_keys = body_cites & set(keymap.values())
            if new_keys:
                merged = ""
                for bib in survey.bib_files:
                    merged += bib.read_text(encoding="utf-8", errors="ignore") + "\n\n"
                merged += "\n\n".join(
                    e for e in bibtex.split("\n\n")
                    if any(f"{{{k}," in e for k in new_keys))
                (self.out / "refined").mkdir(exist_ok=True)
                (self.out / "refined" / "references.bib").write_text(merged)
                closing = re.sub(r"\\bibliography\{[^}]*\}",
                                 r"\\bibliography{references}", closing)
                if "\\bibliography{" not in closing and "thebibliography" not in closing:
                    closing = "\\bibliographystyle{unsrt}\n\\bibliography{references}\n" + closing
        return (doc["preamble"] + doc["opening"]
                + "".join(s["head"] + s["tex"] for s in doc["sections"])
                + closing)

    def emit(self, doc: dict, survey: ProjectSurvey,
             suggestions: list[dict]) -> dict:
        refined_dir = self.out / "refined"
        refined_dir.mkdir(exist_ok=True)
        # figures/data travel with the paper so the folder compiles standalone
        for fig in survey.figures:
            rel = fig.relative_to(survey.root) if survey.root in fig.parents else fig.name
            dst = refined_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig, dst)
        tex = self._assemble(doc, survey)
        papers = list(self.corpus.papers.values())
        if papers:
            tex, overlaps = remove_suspicious_overlap_sentences(tex, papers)
            if overlaps:
                self.report["dropped"] += [
                    {"kind": "overlap", "text": o.get("sentence", "")[:120]}
                    for o in overlaps]
        main = refined_dir / "main.tex"
        main.write_text(tex, encoding="utf-8")

        pdf, log = compile_pdf(main)
        for _ in range(2):
            if pdf is not None or not log:
                break
            got = self._json(
                "You repair LaTeX compile errors without changing content. "
                "JSON: {\"tex\": \"<the corrected full document>\"}",
                f"COMPILE LOG (tail):\n{log[-3000:]}\n\nDOCUMENT:\n{tex[:30000]}",
                role="critic", max_tokens=16384)
            fixed = str(got.get("tex") or "")
            if not fixed.strip():
                break
            if numeric_tokens(fixed) - numeric_tokens(tex):
                break                                    # a "repair" may not invent
            tex = fixed
            main.write_text(tex, encoding="utf-8")
            pdf, log = compile_pdf(main)

        diff_pdf = self._latexdiff(survey.main_tex, main, refined_dir)

        sug = self.out / "suggestions.md"
        lines = ["# Refine suggestions — reported, not asserted\n"]
        for s in suggestions:
            mark = "grounded" if s.get("grounded") else s.get("why_dropped", "ungrounded")
            lines.append(f"- **[{s['kind']} · {mark}]** {s['text']}"
                         + (f"  _({s.get('cite_id')})_" if s.get("cite_id") else ""))
        sug.write_text("\n".join(lines) + "\n")

        rep = self.out / "refine-report.md"
        rep.write_text(self._report_md(pdf, diff_pdf))
        return {"tex": str(main), "pdf": str(pdf) if pdf else "",
                "diff_pdf": str(diff_pdf) if diff_pdf else "",
                "suggestions": str(sug), "report": str(rep),
                "compile_log": log[-1200:] if pdf is None else ""}

    def _latexdiff(self, original_main: Path, refined_main: Path,
                   refined_dir: Path) -> Path | None:
        """The blue-edit PDF: latexdiff (additions in blue, deletions struck red)
        against the untouched original."""
        if not shutil.which("latexdiff"):
            self._stage("diff", "latexdiff not installed — diff PDF skipped")
            return None
        diff_tex = refined_dir / "diff.tex"
        try:
            r = subprocess.run(
                ["latexdiff", "--flatten", "--type=UNDERLINE",
                 str(original_main), str(refined_main)],
                capture_output=True, text=True, timeout=180)
            if r.returncode != 0 or not r.stdout.strip():
                self._stage("diff", "latexdiff failed — diff PDF skipped")
                return None
            diff_tex.write_text(r.stdout, encoding="utf-8")
        except Exception:
            return None
        pdf, _ = compile_pdf(diff_tex)
        if pdf:
            target = refined_dir / "paper-diff.pdf"
            shutil.move(str(pdf), target)
            return target
        return None

    def _report_md(self, pdf, diff_pdf) -> str:
        lines = ["# spiral refine report\n",
                 f"- corpus: {len(self.corpus.papers)} papers",
                 f"- tokens: {self.tokens:,}",
                 f"- submittable pdf: {pdf or 'compile failed — main.tex retained'}",
                 f"- blue-edit diff pdf: {diff_pdf or 'skipped'}",
                 f"- fragments kept verbatim (gates/size): "
                 f"{len(self.report['kept_verbatim'])}",
                 f"- verified enrichments: {len(self.report['enriched'])}",
                 f"- dropped as unverifiable: "
                 f"{len([d for d in self.report['dropped'] if d.get('kind') != 'overlap'])}",
                 "\n## stages"]
        lines += [f"- {s['stage']}: {s['detail']}" for s in self.report["stages"]]
        if self.report["enriched"]:
            lines.append("\n## verified enrichments")
            lines += [f"- {e['sentence']} — [{e['cite']}]" for e in self.report["enriched"]]
        if self.report["kept_verbatim"]:
            lines.append("\n## kept verbatim")
            lines += [f"- {k['part'][:60]} ({k['why']})" for k in self.report["kept_verbatim"]]
        return "\n".join(lines) + "\n"

    # -- orchestration -------------------------------------------------------------
    def run(self) -> dict:
        survey = survey_project(self.root)
        self._stage("survey", f"{survey.main_tex.relative_to(survey.root)} · "
                              f"{len(survey.included)} inputs · "
                              f"{len(survey.figures)} figures · "
                              f"{len(survey.bib_files)} bib")
        flat = flatten_tex(survey.main_tex)
        (self.out / "original-flattened.tex").write_text(flat, encoding="utf-8")

        understanding = self.understand(flat)
        self._stage("understand", str(understanding.get("field", ""))[:60])

        self.build_corpus(understanding)
        self._stage("corpus", f"{len(self.corpus.papers)} papers")

        style = corpus_style_guide(list(self.corpus.papers.values())) \
            if self.corpus.papers else ""
        self._stage("style", "field style learned" if style else "no corpus style")

        fig_notes = self.read_figures(survey.figures)

        doc = split_document(flat)
        doc = self.rebuild(doc, understanding, style, fig_notes)
        doc, suggestions = self.enrich(doc)
        art = self.emit(doc, survey, suggestions)
        self._stage("emit", art["pdf"] or "no PDF (see compile log)")
        return art
