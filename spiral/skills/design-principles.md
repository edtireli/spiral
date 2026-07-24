---
name: design-principles
description: Domain-adaptive product design craft for complete UI, visualization, and interaction work.
---
# Product design craft

## Start with the work
- Build the actual usable product as the first screen, not a landing page, feature tour,
  or decorative dashboard. Identify the repeated primary workflow and make it efficient.
- Match the domain. Operational tools are quiet, dense, predictable, and easy to scan;
  consumer and creative products may be more expressive. Do not impose one house style.
- Complete the surrounding experience: navigation, real data, persistence when relevant,
  settings that matter, recovery, and loading/empty/error/offline states. No dead controls,
  fake results, lorem ipsum, or “coming soon” surfaces.

## Composition
- Use full-width bands or unframed layouts for page structure. Cards are for repeated
  items, modals, or genuinely framed tools; never put cards inside cards.
- Give each view one clear primary action. Use grouping, alignment, and space before
  borders or boxes. Keep headings proportional to their container.
- Define stable grids, min/max widths, aspect ratios, and control dimensions. Test mobile,
  desktop, and wide screens; text, plots, canvases, and controls must not overlap or clip.

## Type, color, and space
- Use one type family per voice, two voices at most, a restrained fixed scale, and normal
  letter spacing. Body line height 1.4-1.6 and long lines about 55-75 characters.
- Use a 4/8 spacing system and minimum 44px interaction targets. Related items are closer
  than unrelated ones; density follows the task rather than a blanket “more whitespace”.
- Define semantic tokens once. Use a balanced neutral foundation, restrained brand colors,
  and success/warning/danger only for meaning. Verify 4.5:1 body contrast.
- Avoid generic purple-blue gradients, beige/brown monotones, one-hue palettes,
  glassmorphism, decorative blobs, and arbitrary shadows.

## Controls and states
- Use familiar icons from the project’s icon library for compact commands, with tooltips
  for unfamiliar meanings. Use tabs for views, segmented controls for modes, toggles for
  binary settings, sliders/steppers for bounded numbers, and menus for option sets.
- Implement default, hover/focus, pressed, disabled, loading, success, empty, error, and
  offline states where relevant. Keep loading feedback in place so layout never jumps.
- Keyboard order, visible focus, labels, reduced motion, and screen-reader names are part
  of the feature, not cleanup.

## Assets and data
- Use real, inspectable product imagery, maps, media, diagrams, or domain objects. Do not
  hide the subject behind dark crops, atmospheric stock, or decorative SVG substitutes.
- Plots need units, uncertainty where applicable, legends or direct labels, distinctions
  beyond color, inspectable values, responsive framing, empty-data behavior, and export of
  both figure and data.

## Motion and language
- Motion must explain a state change: 100-150ms for micro feedback, 200-300ms for
  transitions, reduced-motion fallback, and no animation that blocks work.
- Buttons say what they do. Errors state what happened and the recovery action. Do not add
  visible prose that explains the interface itself.

## Finish gate
Exercise the real primary workflow at mobile, desktop, and wide sizes. Reject overlap,
clipping, blank canvases, broken assets, dead routes, placeholder content, inaccessible
controls, and a visually polished shell around incomplete behavior.
