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
    """Text from a document. PDFs go through the corpus extractor already in spiral."""
    if path.suffix.lower() == ".pdf":
        from spiral.research_corpus import extract_pdf_text

        return extract_pdf_text(path)
    return path.read_text(errors="replace")


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
    path = Path(args.file)
    if not path.is_file():
        console.print(f"  [red]no such file:[/] {path}")
        return 2
    text = read_document(path)
    if not text.strip():
        console.print(f"  [red]no readable text in[/] {path}")
        return 2

    template = None
    if getattr(args, "corpus", None):
        croot = Path(args.corpus)
        if not croot.is_dir():
            console.print(f"  [red]--corpus is not a directory:[/] {croot}")
            return 2
        console.print(f"  [dim]mining a writing template from {croot}…[/]")
        texts = _load_corpus(croot)
        template = mine_template(texts)
        if not template.sample_size:
            console.print("  [yellow]no usable exemplars found; scoring without a template[/]")
        else:
            console.print(f"  [green]●[/] template from {template.sample_size} documents"
                          + (f" · sections: {' → '.join(template.section_order[:6])}"
                             if template.section_order else ""))

    console.print(f"\n  [bold]{path.name}[/]")
    before = _report(console, text, template)

    if not getattr(args, "rewrite", False):
        return 0

    from spiral.config import Config
    from spiral.llm import Ollama
    from spiral.cli import _apply_tier

    cfg = Config.load()
    if getattr(args, "api", None):
        _apply_tier(cfg, console, "api", api_key=args.api)
    ol = Ollama(cfg.base_url, providers=cfg.providers)
    model = cfg.critic.name or cfg.planner.name

    guide = ""
    if template is not None and template.sample_size:
        gaps = score_against(text, template)["gaps"]
        if gaps:
            guide = "\nMatch this field's measured style:\n- " + "\n- ".join(gaps[:6])

    best, best_score = text, before["ai_score"]
    console.print(f"\n  [dim]rewriting with {model} — the scorer decides[/]")
    for attempt in range(1, int(getattr(args, "rounds", 3)) + 1):
        try:
            res = ol.chat(model, [
                {"role": "system", "content": _SYSTEM + guide},
                {"role": "user", "content": text},
            ], num_predict=max(1024, min(8192, len(text) // 2)), temperature=0.4,
                num_ctx=cfg.spec_for(model).num_ctx, keep_alive=cfg.keep_alive)
        except Exception as exc:
            console.print(f"    [red]attempt {attempt}: model error {exc}[/]")
            continue
        candidate = (res.text or "").strip()
        if not candidate:
            console.print(f"    [yellow]attempt {attempt}: empty reply[/]")
            continue
        drift = content_drift(text, candidate)
        score = ai_score(candidate)
        if drift:
            console.print(f"    [yellow]attempt {attempt} REJECTED[/] — {drift[0]}")
            continue
        if score >= best_score:
            console.print(f"    [yellow]attempt {attempt} REJECTED[/] — score "
                          f"{score:.1f} did not beat {best_score:.1f}")
            continue
        console.print(f"    [green]attempt {attempt} accepted[/] — {best_score:.1f} → {score:.1f}")
        best, best_score = candidate, score

    if best is text:
        console.print("\n  [yellow]no rewrite passed the checks; original kept[/]")
        return 1
    out = Path(getattr(args, "out", None) or
               path.with_suffix(f".human{path.suffix or '.txt'}"))
    out.write_text(best)
    console.print(f"\n  [bold]after rewrite[/]")
    _report(console, best, template)
    console.print(f"\n  [green]●[/] written to [dim]{out}[/]")
    return 0
