"""Generate the clay SVG mark set used as README elements — same log-spiral
math as the banner and GIF, so the brand never drifts.

    assets/mark.svg     22px header icon
    assets/dot.svg      13px bullet
    assets/divider.svg  centered section divider (spiral + fading dot runs)

    python scripts/make_marks.py
"""
from __future__ import annotations

import math
from pathlib import Path

CLAY = "#D97757"
OUT = Path(__file__).parent.parent / "assets"


def spiral_points(cx: float, cy: float, r_max: float, turns: float = 2.2, cw: bool = False) -> str:
    tmax = turns * 2 * math.pi
    a = r_max / math.exp(0.28 * tmax) * 3.2  # tuned so the tail reaches r_max
    b = math.log(r_max / a) / tmax
    pts = []
    t = 0.0
    while t <= tmax:
        r = a * math.exp(b * t)
        ang = -t if cw else t
        pts.append(f"{cx + r * math.cos(ang):.2f},{cy + r * math.sin(ang):.2f}")
        t += 0.05
    return " ".join(pts)


def spiral_svg(size: int, stroke: float, turns: float) -> str:
    m = size / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">'
        f'<polyline points="{spiral_points(m, m, m - stroke, turns)}" fill="none" '
        f'stroke="{CLAY}" stroke-width="{stroke}" stroke-linecap="round"/></svg>'
    )


def divider_svg(width: int = 520, height: int = 16) -> str:
    m = height / 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    # center mini-spiral
    parts.append(
        f'<polyline points="{spiral_points(width / 2, m, m - 2.2, 2.0)}" fill="none" '
        f'stroke="{CLAY}" stroke-width="1.8" stroke-linecap="round"/>'
    )
    # dot runs fading outward on both sides
    for i in range(1, 15):
        for sign in (-1, 1):
            x = width / 2 + sign * (14 + i * 15)
            op = max(0.08, 0.55 - i * 0.038)
            parts.append(f'<circle cx="{x:.0f}" cy="{m}" r="1.4" fill="{CLAY}" opacity="{op:.2f}"/>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "mark.svg").write_text(spiral_svg(24, 2.3, 2.2))
    (OUT / "dot.svg").write_text(spiral_svg(16, 2.6, 1.8))
    (OUT / "divider.svg").write_text(divider_svg())
    for f in ("mark.svg", "dot.svg", "divider.svg"):
        print(f"wrote assets/{f} · {(OUT / f).stat().st_size} B")


if __name__ == "__main__":
    main()
