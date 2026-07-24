---
name: dark-ui-design
description: Accessible dark-surface design for products whose domain genuinely benefits from a dark theme.
---
# Dark-surface craft

- Choose dark mode because the product context supports it, not as a default aesthetic.
  Use near-black only for the base canvas, then clearly separated neutral surfaces.
- Keep the palette multivalent: neutral surfaces, one or two restrained brand colors,
  and semantic status colors. Avoid blue-slate monotony, neon-on-everything, gradients,
  glows, glass panels, and decorative colored blobs.
- Primary text should be softly off-white, secondary text must still pass contrast, and
  disabled text is never the only carrier of meaning. Verify ratios programmatically.
- Elevation comes from surface contrast and spacing before shadow. Cards remain reserved
  for repeated items or framed tools; do not turn every section into a floating panel.
- Charts use luminance, dash/shape, and direct labels as well as hue. Tooltips and focus
  states must remain readable over every series and surface.
- Inputs and controls need visible hover, focus, pressed, disabled, loading, and error
  states. Avoid using the brand accent as the focus, selection, link, and status color all
  at once.
- Inspect screenshots at mobile, desktop, and wide sizes for crushed blacks, clipped text,
  low-contrast metadata, excessive empty space, and bright elements that dominate the task.
