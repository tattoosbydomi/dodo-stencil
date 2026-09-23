# Stencil Generator

Turns your finished digital tattoo artwork into a print-ready stencil, entirely in the browser (no server, no account, no per-use cost, nothing uploaded anywhere).

## Running locally

Any static file server works — ES modules and Web Workers require `http://`, not `file://`. Double-click `Start Stencil Generator.cmd` (Windows), or run:

```
npx serve .
# or
python -m http.server 8000
```

Then open the printed URL.

## Deploying to GitHub Pages

1. Push this folder's contents to a repo (or to a `stencil_generator/` path if kept alongside other projects). This includes `models/informative-drawings.onnx` (~17MB) — GitHub Pages serves it like any other static file.
2. In the repo's Settings → Pages, set the source to the branch/folder containing `index.html`.
3. No build step is required — it's plain HTML/CSS/JS.

## How it works

- **Line art comes from a real neural network, not an edge filter.** A classical filter (Sobel/Canny/XDoG) can only ever find "every edge in the pixels" — it has no way to know which lines an artist would actually keep. `models/informative-drawings.onnx` is a network trained specifically to convert a photo/illustration into line-art, so it produces selective, confident linework much closer to a hand-drawn stencil.
- `js/worker.js` — runs entirely off the main thread: loads the ONNX model (via `onnxruntime-web`, WASM backend), runs inference once per uploaded image, and caches the result so slider tweaks never re-run the network.
- `js/pipeline.js` — pure post-processing math (levels cleanup, line thickness via grayscale morphology, posterize, stencil-over-reference colour tinting). Operates on the network's output and on the original artwork; no DOM access, so it's reusable and easy to test outside the browser.
- `js/main.js` — UI wiring: upload, two-phase preview (one-time "analyze" pass + instant slider-driven "finalize" pass), preview modes (Original / Stencil / Stencil + Ref / Split), layer toggle, stencil colour picker, export.

**Two-phase preview:** uploading a photo triggers one network inference pass (a second or two) at a capped working resolution. Every slider after that — Keep Detail, Background Cleanup, Line Thickness, Crisp, Invert, reference tone bands, stencil colour, reference opacity — is cheap post-processing on the cached network output, so the preview stays responsive on mobile despite the heavier underlying model.

**Export resolution:** the network runs at a capped analysis resolution (quality plateaus above this, and it keeps export time reasonable on mobile), then the result is upscaled to your requested print size (width + DPI). This works well because line art is mostly smooth curves, not fine texture.

**Three exports:**
- *Stencil* — binary/high-contrast line art, for a thermal stencil printer.
- *Grayscale Reference* — continuous-tone grayscale copy, for a regular printer with a grey ink tank, to use as a shading reference alongside the stencil.
- *Stencil + Reference (Colour)* — the line art tinted in a chosen colour and layered on top of the grayscale reference, so both are visible in one image.

## Why this beats a generic AI image prompt

ChatGPT/Gemini-style image generation *regenerates* the image through a diffusion model — for realism/portrait tattoo work that means hallucinated details and an inaccurate likeness. This tool instead runs a purpose-built line-extraction network directly on your actual artwork (image-to-image, not generative), so the output stays faithful to your composition, and every stage after that (detail level, cleanup, line thickness) is manually tunable rather than a black box.

## Model credit & license

`models/informative-drawings.onnx` is the ONNX export (by [Joseph Rocca](https://github.com/josephrocca/image-to-line-art-js)) of **Informative Drawings** by Caroline Chan et al. ("Learning to Generate Line Drawings that Convey Geometry and Semantics", CVPR 2022) — [original repo](https://github.com/carolineec/informative-drawings), MIT licensed. `onnxruntime-web` (Microsoft, MIT licensed) is loaded from jsDelivr at runtime to run it.

## Known limitations

- Best results are on clean digital artwork (the intended input); real skin/tattoo photos are noisier and need more slider tuning (Background Cleanup / Keep Detail especially).
- The network won't reproduce hand-drawn conventions like solid-vs-dashed-vs-dotted line semantics — that judgment call still belongs to you. This tool gets you a strong, editable starting point, not a finished stencil.
