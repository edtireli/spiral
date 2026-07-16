---
name: design-principles
description: Universal product design craft — typography scale, spacing systems, color tokens, hierarchy, component states, motion, empty states, microcopy. Use for any task that creates or styles UI, screens, layouts, themes, or user-facing text.
---
# Design principles (implement, don't decorate)

## Typography
- ONE typeface family per voice. Two voices max (e.g. sans for humans, mono for
  systems). Never three.
- Build a scale, don't pick sizes ad hoc: base 16, then ×1.25 steps → 12 / 16 /
  20 / 25 / 31. Round to whole units. Every text element uses a scale step.
- Two weights only: regular (400) and medium/semibold (500-600). Bold everything
  = bold nothing.
- Line height: 1.4-1.6 body, 1.1-1.2 headings. Long text lines max ~70 chars.

## Spacing — the 4/8 grid
- Every margin/padding/gap is a multiple of 4 (prefer 8): 4, 8, 12, 16, 24, 32, 48.
- Related things sit CLOSER than unrelated things — grouping is spacing, not boxes.
- Generosity reads as quality: when unsure, add 50% more whitespace than feels
  necessary. Cramped = cheap.
- Touch targets ≥ 44-48px. Screen gutters 16-24.

## Color — tokens, not hexes-in-place
- Define tokens once (bg / surface / raised / accent / text-primary / text-secondary
  / success / danger / warning), reference tokens everywhere. A hex in a component
  is a bug.
- ONE accent color. Semantic colors (green/red/amber) are reserved for MEANING —
  never decoration. If everything is colored, nothing is.
- Dark themes: never pure black (#0A0A0A floor) or pure white text (#F5F5F5 cap);
  elevation = lighter surface, not shadows.
- Contrast: body text ≥ 4.5:1 against its surface, large text ≥ 3:1. Check, don't eyeball.

## Hierarchy — one king per screen
- Every screen has exactly ONE primary action, visually loudest. Secondary actions
  are quieter (outline/ghost). Destructive actions are never the loudest.
- The eye path should be F- or Z-shaped: title → key content → action.
- If everything is emphasized, the screen has no design — remove emphasis until
  only the essentials carry it.

## States — every component has five
Design ALL of them, not just the happy one:
1. default · 2. hover/focus (visible focus ring) · 3. active/pressed ·
4. disabled (reduced opacity, no color shift) · 5. loading (skeleton or spinner
IN PLACE — layout never jumps).
Plus screen-level: EMPTY state (explain + point to first action, never a blank
void) and ERROR state (say what happened + what to do, never just "error").

## Motion — physics, not decoration
- Durations: micro-feedback 100-150ms, transitions 200-300ms, attention 400-600ms.
  Nothing over 800ms, ever.
- Ease-out for entrances (fast start, gentle stop), ease-in for exits. Linear only
  for continuous progress.
- Animate at most 2 properties at once. Every animation must answer "what did
  this teach the user?" — if nothing, delete it.
- Loading feels deliberate at 1.5-3s of animated progress; instant feels fake,
  >6s feels broken (show real progress).

## Microcopy
- Buttons say what they DO ("Send message", not "OK"/"Submit").
- The product has ONE voice — pick it (formal, playful, deadpan) and never break it.
- Errors: what happened + what to do next, in the product's voice, no codes-only.

## The restraint law
Great design is deletion. For every element ask: does removing it lose meaning?
If no — remove it. Borders can be spacing. Backgrounds can be nothing. Icons
next to labels are usually noise. Ship the version with less.
