"""`spiral style` — measure writing, and rewrite it only if the measurement improves.

The rewrite loop is the project's contract applied to prose: the model proposes a
rewrite, and deterministic checks decide whether to keep it. A rewrite is accepted only
when it (a) lowers the AI-tell score, and (b) preserves the things a rewrite must never
invent or drop — numbers, citations, equations. Otherwise it is rejected and retried.
That last guard matters: the easiest way to make prose "sound human" is to quietly
delete the specifics, which would be a downgrade disguised as an improvement.
"""
from __future__ import annotations

import re
from pathlib import Path

from spiral.writing_style import (
    ai_score, ai_tells, measure, mine_template, score_against,
)

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_CITE = re.compile(r"\\cite[tp]?\{([^}]*)\}")
_EQ = re.compile(r"\\begin\{(?:equation|align|gather)\*?\}(.*?)\\end\{(?:equation|align|gather)\*?\}",
                 re.S)
_TEXTY = {".txt", ".md", ".tex", ".rst", ".markdown", ""}


def read_document(path: Path) -> str:
    """Plain text of a document, for measurement. Editing goes through
    ``spiral.documents``, which keeps the document's structure instead of flattening it."""
    from spiral.documents import read_document as _read

    return _read(path).text()


def _load_corpus(root: Path) -> list[str]:
    texts: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".tex", ".txt", ".md", ".pdf"}:
            try:
                body = read_document(p)
            except Exception:
                continue
            if body and len(body) > 400:
                texts.append(body)
    return texts[:60]


def content_drift(before: str, after: str) -> list[str]:
    """What a rewrite silently changed that it must not have. Deterministic."""
    problems: list[str] = []
    lost_nums = sorted(set(_NUM.findall(before)) - set(_NUM.findall(after)))
    if lost_nums:
        problems.append(f"dropped numbers: {', '.join(lost_nums[:8])}")
    b_c = {k.strip() for m in _CITE.findall(before) for k in m.split(",") if k.strip()}
    a_c = {k.strip() for m in _CITE.findall(after) for k in m.split(",") if k.strip()}
    if b_c - a_c:
        problems.append(f"dropped citations: {', '.join(sorted(b_c - a_c)[:6])}")
    if a_c - b_c:
        problems.append(f"invented citations: {', '.join(sorted(a_c - b_c)[:6])}")
    b_e = [re.sub(r"\s+", "", e) for e in _EQ.findall(before)]
    a_e = [re.sub(r"\s+", "", e) for e in _EQ.findall(after)]
    if b_e != a_e:
        problems.append(f"display equations changed ({len(b_e)} → {len(a_e)})")
    words_before, words_after = len(before.split()), len(after.split())
    if words_after < words_before * 0.6:
        problems.append(
            f"rewrite is {words_after} words vs {words_before} — content was cut, not rephrased")
    return problems


def _report(console, text: str, template=None) -> dict:
    m = measure(text)
    tells = ai_tells(text)
    console.print(f"  [bold]{m.words}[/] words · {m.sentences} sentences · "
                  f"mean {m.mean_sentence_words} words/sentence")
    console.print(f"  [bold]AI-tell score:[/] {ai_score(text):.1f} per 1000 words"
                  + ("  [dim](lower is more human)[/]" if tells else ""))
    for t in tells[:8]:
        console.print(f"    [yellow]{t.count:6.1f}[/] {t.id:24s} [dim]{t.explanation}[/]")
        console.print(f"            [dim]e.g. {'; '.join(t.examples[:3])}[/]")
    if not tells:
        console.print("    [green]no catalogued AI tells found[/]")
    out = {"metrics": m.as_dict(), "ai_score": ai_score(text)}
    if template is not None and template.sample_size:
        rep = score_against(text, template)
        out.update(rep)
        if rep["gaps"]:
            console.print("  [bold]against the field template:[/]")
            for g in rep["gaps"]:
                console.print(f"    [cyan]·[/] {g}")
        else:
            console.print("  [green]within the field's style band[/]")
        if rep["missing_sections"]:
            console.print(f"    [dim]sections the field uses that this lacks: "
                          f"{', '.join(rep['missing_sections'][:6])}[/]")
    return out


_SYSTEM = (
    "Rewrite the passage so it reads as plain human writing. Remove machine-prose "
    "habits: significance inflation ('stands as a testament'), 'serves as' for 'is', "
    "'not only X but Y', trailing '-ing' clauses asserting importance, promotional "
    "adjectives, vague attribution ('experts argue'), and 'Additionally,' openers. "
    "HARD RULES: keep every number, every \\cite{...} key and every display equation "
    "EXACTLY as they are. Do not add facts. Do not summarise or shorten the content — "
    "rephrase it. Output ONLY the rewritten text, no commentary."
)


def run_style(console, args) -> int:
    """Entry point. Wrapped in the live cockpit so a long rewrite shows its work."""
    from spiral.prose_ui import ProseProgress

    path = Path(args.file)
    if not path.is_file():
        console.print(f"  [red]no such file:[/] {path}")
        return 2
    rewriting = bool(getattr(args, "rewrite", False))
    with_corpus = bool(getattr(args, "corpus", None))
    with ProseProgress(console, with_corpus=with_corpus, rewriting=rewriting,
                       thought_log=path.parent / ".spiral-prose-thoughts.jsonl") as ui:
        try:
            return _run_style(ui, args, path, rewriting=rewriting)
        except KeyboardInterrupt:
            ui.print("  [yellow]interrupted — nothing was written[/]")
            return 130


def _run_style(ui, args, path: Path, *, rewriting: bool) -> int:
    from spiral.documents import (
        default_output_path, read_document as read_doc, write_document,
    )

    # ── M1 read the document ────────────────────────────────────────────────
    ui.stage(0, 0, phase="parsing structure",
             idea="Splitting prose from scaffolding: equations, preamble and headings are "
                  "never handed to a rewriter.",
             detail=path.name)
    doc = read_doc(path)
    text = doc.text()
    if not text.strip():
        ui.print(f"  [red]no readable text in[/] {path}")
        return 2
    editable = doc.editable_segments
    ui.print(f"  [green]●[/] {path.name} · [bold]{doc.kind}[/] · {len(doc.segments)} "
             f"segments, [bold]{len(editable)}[/] editable")
    ui.done(0, 0)

    ui.stage(0, 1, phase="measuring prose",
             idea="Style is a measurement here, not an opinion — sentence length, hedging, "
                  "passive rate, citation and equation density.")
    m = measure(text)
    ui.print(f"  [dim]{m.words} words · {m.sentences} sentences · "
             f"mean {m.mean_sentence_words} words/sentence · "
             f"hedges {m.hedges_per_1k}/1k · passive {m.passive_per_1k}/1k[/]")
    ui.done(0, 1)

    ui.stage(0, 2, phase="detecting AI tells",
             idea="Matching the catalogue mined from Wikipedia's Signs of AI writing, plus "
                  "structural and formatting patterns it describes but does not list.")
    template = None
    before = _report(ui, text, None)
    ui.detail(f"score {before['ai_score']:.1f}/1k")
    ui.done(0, 2)

    mi = 1
    if args.corpus:
        croot = Path(args.corpus)
        if not croot.is_dir():
            ui.print(f"  [red]--corpus is not a directory:[/] {croot}")
            return 2
        ui.stage(mi, 0, phase="mining the field template",
                 idea="Reading real papers from this field to learn what its writing "
                      "actually measures, instead of guessing a house style.",
                 detail=str(croot))
        texts = _load_corpus(croot)
        template = mine_template(texts)
        if not template.sample_size:
            ui.print("  [yellow]no usable exemplars found; scoring without a template[/]")
            ui.blocked(mi, 0)
        else:
            ui.print(f"  [green]●[/] template from [bold]{template.sample_size}[/] documents"
                     + (f" · sections: {' → '.join(template.section_order[:6])}"
                        if template.section_order else ""))
            ui.done(mi, 0)
            ui.stage(mi, 1, phase="scoring against the band",
                     idea="Reporting where this document sits outside the field's measured "
                          "range — the gaps become the rewrite instructions.")
            rep = score_against(text, template)
            for g in rep["gaps"]:
                ui.print(f"    [cyan]·[/] {g}")
            if not rep["gaps"]:
                ui.print("  [green]within the field's style band[/]")
            ui.done(mi, 1)
        mi += 1

    if not rewriting:
        ui.print("\n  [dim]measurement only — pass [bold]--rewrite[/bold] to edit[/]")
        return 0
    return _rewrite(ui, args, path, doc, template, before, mi,
                    default_output_path=default_output_path,
                    write_document=write_document)


def _rewrite(ui, args, path: Path, doc, template, before: dict, mi: int, *,
             default_output_path, write_document) -> int:
    """Segment-by-segment rewriting with the plan advancing as it goes."""
    from spiral.cli import _apply_tier
    from spiral.config import Config
    from spiral.llm import Ollama

    cfg = Config.load()
    if getattr(args, "api", None):
        _apply_tier(cfg, ui.dash.c, "api", api_key=args.api)
    ol = Ollama(cfg.base_url, providers=cfg.providers)
    # rewriting prose is a reasoning job — take the strongest configured model, not the
    # critic slot, which spiral deliberately fills with a small fast one
    model = (getattr(args, "model", "") or cfg.planner.name or cfg.escalation.name
             or cfg.worker.name or cfg.critic.name)

    text = doc.text()
    guide = ""
    if template is not None and template.sample_size:
        gaps = score_against(text, template)["gaps"]
        if gaps:
            guide = "\nMatch this field's measured style:\n- " + "\n- ".join(gaps[:6])

    editable = [s for s in doc.editable_segments if len(s.text.split()) >= 12]
    protected = len(doc.segments) - len(editable)
    ui.stage(mi, 0, phase="rewriting segments", model=model,
             idea=f"One paragraph at a time — the scorer decides each, so a single bad "
                  f"generation cannot take the document with it. {protected} segment(s) "
                  f"are protected and never sent to the model.")
    ui.print(f"\n  [dim]rewriting with [bold]{model}[/bold] · {len(editable)} segment(s) "
             f"· {protected} protected[/]")

    replacements: dict = {}
    kept = rejected = 0
    facts_saved = 0
    tokens = 0
    rounds = max(1, int(getattr(args, "rounds", 3)))
    for n, seg in enumerate(editable, 1):
        ui.detail(f"segment {n}/{len(editable)} · {seg.text.strip()[:48]}")
        seg_before = ai_score(seg.text)
        best_text, best_score = None, seg_before
        for attempt in range(1, rounds + 1):
            try:
                res = ol.chat(model, [
                    {"role": "system", "content": _SYSTEM + guide},
                    {"role": "user", "content": seg.text},
                ], num_predict=max(512, min(4096, len(seg.text))), temperature=0.4,
                    num_ctx=cfg.spec_for(model).num_ctx, keep_alive=cfg.keep_alive)
            except Exception as exc:
                ui.print(f"    [red]segment {n}: model error {exc}[/]")
                break
            tokens += int(getattr(res, "prompt_tokens", 0) or 0) + int(
                getattr(res, "completion_tokens", 0) or 0)
            ui.tokens(tokens)
            candidate = (res.text or "").strip()
            if not candidate:
                continue
            drift = content_drift(seg.text, candidate)
            if drift:
                facts_saved += 1
                ui.print(f"    [yellow]○[/] segment {n} attempt {attempt} rejected — "
                         f"[dim]{drift[0]}[/]")
                continue                        # never trade facts for smoothness
            score = ai_score(candidate)
            if score < best_score:
                best_text, best_score = candidate, score
                ui.print(f"    [green]✔[/] segment {n} · {seg_before:.0f} → "
                         f"[bold]{score:.0f}[/] tells/1k")
                break                           # first improvement wins; move on
            ui.print(f"    [yellow]○[/] segment {n} attempt {attempt} rejected — "
                     f"[dim]score {score:.0f} did not beat {best_score:.0f}[/]")
        if best_text is not None:
            replacements[seg.index] = best_text
            kept += 1
        else:
            rejected += 1
    ui.done(mi, 0)

    ui.stage(mi, 1, phase="guarding the facts",
             idea="Every accepted rewrite kept its numbers, citations and equations — "
                  "the easy way to sound human is to delete the specifics.")
    ui.print(f"  [dim]{kept} improved · {rejected} left as written · "
             f"{facts_saved} rejected for changing facts[/]")
    ui.done(mi, 1)

    if not replacements:
        ui.print("\n  [yellow]no segment rewrite passed the checks; original untouched[/]")
        return 1

    ui.stage(mi + 1, 0, phase="writing the document",
             idea="Written back through the document's own structure, so a .docx keeps its "
                  "styles and a .tex keeps its preamble and math. The original is never "
                  "overwritten.")
    after_text = doc.with_replacements(replacements)
    out = Path(getattr(args, "out", None) or default_output_path(path))
    write_document(doc, replacements, out)
    ui.done(mi + 1, 0)

    ui.stage(mi + 1, 1, phase="re-measuring",
             idea="Reporting the score after the edit, not the score hoped for.")
    ui.print(f"\n  [bold]after rewrite[/] [dim]({kept} of {len(editable)} segments improved)[/]")
    after = _report(ui, after_text, template)
    ui.done(mi + 1, 1)

    delta = before["ai_score"] - after["ai_score"]
    ui.print(f"\n  [bold]AI-tell score[/] {before['ai_score']:.1f} → "
             f"[bold]{after['ai_score']:.1f}[/] per 1000 words "
             f"[green]({delta:+.1f})[/]" if delta > 0 else
             f"\n  [bold]AI-tell score[/] {before['ai_score']:.1f} → {after['ai_score']:.1f}")
    if path.suffix.lower() == ".pdf":
        ui.print("  [dim]a PDF cannot be rewritten in place; the edit is markdown[/]")
    ui.print(f"  [green]●[/] written to [dim]{out}[/]")
    ui.print(f"  [dim]original untouched: {path}[/]")
    return 0
