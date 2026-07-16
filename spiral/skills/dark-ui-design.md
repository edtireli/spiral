---
name: dark-ui-design
description: Dark-theme UI design taste — layout, spacing, color, typography, hierarchy, status/feedback states. Use for any task creating or styling screens, layouts, themes, colors, or visual polish.
---
# Dark UI craft

## Surfaces & color
- Never pure black: page #0A0A0A, raised surfaces #141414 → #1E1E1E → #252525.
  Each elevation step is a LIGHTER surface, not a shadow.
- One accent color, used sparingly (this project: blood red #B71C1C family with
  gold #FFD600 for state text). If everything is red, nothing is.
- Text: primary #F5F5F5, secondary #9E9E9E, hints #616161. Never pure white.
- Semantic states: danger/scanning red #FF1744, approved green #00E676,
  warning amber #FFAB00 — reserve these for STATUS, not decoration.
- Contrast: body text ≥ 4.5:1 against its surface. #9E9E9E on #0A0A0A passes;
  #616161 is hint-only.

## Space & type
- 4/8dp rhythm: padding 8/12/16/24, gaps between groups ≥ 16, screen gutters 16.
- Type scale, few sizes: title 20sp bold, body 16sp, meta 12sp, ticker 11-12sp.
- Monospace (`android:fontFamily="monospace"`) for "machine voice" — tickers,
  scan logs, surveillance status. Sans for the human's own messages. The contrast
  between the two IS the satire.
- Touch targets ≥ 48dp; input bars 52-56dp tall.

## Chat-app anatomy
- Sent bubbles: accent-tinted surface, right-aligned, rounded 16dp with one
  4dp corner (bottom-right) for direction. Received: neutral surface, mirrored.
- Bubble max width ~78% of screen; padding 12dp; 4dp between consecutive
  bubbles from the same sender, 12dp between speakers.
- System/state messages (verdicts, warnings): full-width, centered, monospace,
  boxed with a 1dp border in the semantic color — visually THE STATE, not a person.

## Motion & feedback
- Every async state needs visible progress (scan bar, pulsing eye) — 2-4s feels
  deliberate; instant feels fake, >6s feels broken.
- State transitions announce themselves: color shift + one-line status text
  ("ANALYZING… → APPROVED ✓ / FLAGGED ✗"). Don't animate more than 2 things at once.
- Blink/pulse loops: 3-6s randomized intervals read as "alive"; fixed 1s loops
  read as broken GIFs.

## Dystopian voice (this project)
- The state is POLITE and BUREAUCRATIC — menace through cheerful compliance
  language, not shouting: "Thank you for your transparency, Citizen."
- Every message gets a case number. Loyalty ratings tick. Articles cited
  ("per Article 7.3 of the Information Control Act").
- ALL-CAPS only for the state's stamps (APPROVED / FLAGGED), never body text.
