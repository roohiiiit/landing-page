# RFernz Signature Intro

A lightweight, tablet-friendly opening loading screen: a crisp `#ffffff` paper sheet with animated monochrome film grain, a light-grey `#d9d9d9` ghost signature revealed stroke-by-stroke in `#000000` via an animated SVG `<mask>`, and a curved liquid curtain exit that fires `signature:done`.

## Files

- `signature.png` — Source signature image.
- `tools/trace.py` — Generates `signature-fill.svg` (via `potrace`) and `signature-centerline.svg` (via `skimage.morphology.skeletonize`).
- `signature-fill.svg` — Filled vector outline of the signature.
- `signature-centerline.svg` — Ordered, smoothed pen-stroke centerlines used as the `<mask>`.
- `index.html` — Self-contained intro screen using GSAP (`3.12.5` from jsDelivr).

## Rerunning the Trace

If you replace `signature.png`, rerun:

```bash
python3 tools/trace.py
```

(Requires `potrace`, `pillow`, `numpy`, `scipy`, and `scikit-image`.)

Then copy the `<path>` elements from `signature-centerline.svg` into `#mask-strokes` and the `<g>` from `signature-fill.svg` into `.loader__ghost`, the masked `<g>`, and `#sig-solid` in `index.html`.

## Tuning

- **Colors & Sizing:** Edit `--bg` (`#ffffff`), `--ink` (`#000000`), `--ghost` (`#d9d9d9`), `--sig-height` (`82px`), and `--mask-stroke` (`18`) in `:root` inside `index.html`.
- **Stroke Cadence:** Adjust the `timings` array inside `playIntro()` in `index.html`.
- **Chaining your Hero Section:** Listen for the `signature:done` event on `document`:
  ```js
  document.addEventListener('signature:done', () => {
    // Reveal your hero section here
  });
  ```
