// Cache-busting (see index.html): this static import's ?v=, ASSET_VERSION
// below, and index.html's ?v= must all be bumped together on every deploy
// that touches js/ or css/. ASSET_VERSION is threaded through to the
// dynamically-loaded worker.js and pipeline.js further down, since import
// specifiers (static or dynamic) don't inherit this file's own query string.
import { grayToRGBA, tintStencilOverReference, unsharpMaskRGBA } from './pipeline.js?v=3';
const ASSET_VERSION = '3';

const MAX_PREVIEW_DIM = 900;
const MAX_EXPORT_ANALYSIS_DIM = 1400; // cap the network's input size; final print can still be larger (see renderFullResLayers)
const DEBOUNCE_MS = 130;

const el = (id) => document.getElementById(id);

const dom = {
  fileInput: el('file-input'),
  dropzone: el('dropzone'),
  resetBtn: el('reset-btn'),
  uploadView: el('upload-view'),
  editorView: el('editor-view'),
  stage: el('canvas-stage'),
  canvasOriginal: el('canvas-original'),
  canvasStencil: el('canvas-stencil'),
  canvasReference: el('canvas-reference'),
  canvasColour: el('canvas-colour'),
  overlay: el('processing-overlay'),
  overlayLabel: el('processing-label'),
  modeButtons: Array.from(document.querySelectorAll('.mode-btn')),
  quickIconButtons: Array.from(document.querySelectorAll('.quick-icon-btn')),
  quickSliderOverlay: el('quick-slider-overlay'),
  quickSliderInput: el('quick-slider-input'),
  quickSliderDot: el('quick-slider-dot'),
  quickSliderOutput: el('quick-slider-output'),
  quickSliderClose: el('quick-slider-close'),
  quickDescription: el('quick-description'),
  exportStencilBtn: el('export-stencil-btn'),
  exportReferenceBtn: el('export-reference-btn'),
  exportColourBtn: el('export-colour-btn'),
  colourSwatches: Array.from(document.querySelectorAll('.swatch-btn')),
  refOpacity: el('in-ref-opacity'),
  refOpacityOut: el('out-ref-opacity'),
  outputSizeHint: el('output-size-hint'),
  printWidth: el('in-print-width'),
  printDpi: el('in-print-dpi'),
};

const sliderIds = [
  'keep-detail', 'bg-cleanup', 'thickness',
  'ref-levels',
];
// Reference Levels is a literal tone-band count (see percentFromValue's caller
// below for the same rule on the mobile quick-slider) — every other slider here
// shows a normalised 0-100 reading instead of its real underlying range, so the
// same control reads the same whether you're adjusting it here or via the
// mobile quick-adjust overlay.
const RAW_SCALE_IDS = new Set(['ref-levels']);
const sliders = {};
for (const id of sliderIds) {
  sliders[id] = { input: el(`in-${id}`), output: el(`out-${id}`) };
  const updateOutput = () => {
    const { input, output } = sliders[id];
    output.textContent = RAW_SCALE_IDS.has(id)
      ? input.value
      : percentFromValue(Number(input.value), Number(input.min), Number(input.max));
  };
  updateOutput();
  sliders[id].input.addEventListener('input', () => {
    updateOutput();
    schedulePreviewFinalize();
  });
}

const checkboxIds = ['invert', 'ref-posterize'];
for (const id of checkboxIds) {
  el(`in-${id}`).addEventListener('change', schedulePreviewFinalize);
}

el('in-line-style').addEventListener('change', schedulePreviewFinalize);

// Sharpening changes what the *network* sees, not just post-processing, so it
// needs a full re-analysis (re-running the network), not the cheap finalize path
// slider tweaks use.
el('in-presharpen').addEventListener('change', () => {
  if (previewAnalyzed) runPreviewAnalysis();
});

// --- Worker plumbing -------------------------------------------------
const worker = new Worker(new URL(`./worker.js?v=${ASSET_VERSION}`, import.meta.url));
let nextRequestId = 1;
const pending = new Map();

worker.onmessage = (e) => {
  const { requestId, error } = e.data;
  const resolver = pending.get(requestId);
  if (!resolver) return;
  pending.delete(requestId);
  if (error) { resolver.reject(new Error(error)); return; }
  if (e.data.type === 'previewAnalyzeDone') { resolver.resolve({ analyzed: true }); return; }
  const { width, height, lineBuf, referenceBuf } = e.data;
  resolver.resolve({
    width, height,
    lines: new Float32Array(lineBuf),
    reference: new Float32Array(referenceBuf),
  });
};

function send(message, transferList) {
  const requestId = nextRequestId++;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    worker.postMessage({ requestId, ...message }, transferList);
  });
}

// --- State -------------------------------------------------------------
let sourceBitmap = null;      // full-resolution ImageBitmap, kept for export
let previewImageData = null;  // downscaled RGBA used for live preview + "Original" view
let previewW = 0, previewH = 0;
let previewLayers = null;     // last computed { lines, reference, width, height }
let currentMode = 'stencil';
let stencilColour = dom.colourSwatches.find((btn) => btn.classList.contains('active'))?.dataset.colour || '#FF007F';
let debounceTimer = null;
let latestFinalizeToken = 0;
let previewAnalyzed = false;

function readParams() {
  const v = (id) => Number(sliders[id].input.value);
  return {
    blackPoint: v('keep-detail'),
    whitePoint: v('bg-cleanup'),
    lineStyle: el('in-line-style').value,
    lineThickness: v('thickness'),
    invertLines: el('in-invert').checked,
    referencePosterize: el('in-ref-posterize').checked,
    referenceLevels: v('ref-levels'),
  };
}

// --- Image loading -------------------------------------------------------
dom.fileInput.addEventListener('change', () => {
  const file = dom.fileInput.files[0];
  if (file) loadFile(file);
});
dom.dropzone.addEventListener('dragover', (e) => e.preventDefault());
dom.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});
dom.resetBtn.addEventListener('click', () => {
  sourceBitmap = null;
  previewImageData = null;
  previewLayers = null;
  previewAnalyzed = false;
  closeQuickSlider();
  dom.fileInput.value = '';
  dom.editorView.hidden = true;
  dom.uploadView.hidden = false;
  dom.resetBtn.hidden = true;
});

// The canvases you actually see are sized to the real screen (CSS box × devicePixelRatio),
// decoupled from previewW/previewH (the resolution fed to the network). Without this, a
// canvas whose pixel backing store is smaller than its on-screen box gets stretched by the
// browser to fill it — invisible for "Soft" mode since its blur already blends through the
// stretch, but a hard-edged binarized/skeletonized line has nothing to blend through, so the
// same stretch reads as blocky pixelation. Worse again on any HiDPI/retina screen, where the
// physical pixel count is even higher than CSS px. blitToDisplay stages the actual (lower-res)
// processed pixels on an offscreen canvas, then draws that onto the real one with smoothing —
// the browser's own image-scaling filter is what removes the aliasing, matching what "Soft"
// mode gets from the network's own blur.
const displayScratchCanvas = document.createElement('canvas');
const displayScratchCtx = displayScratchCanvas.getContext('2d');

function setupDisplayCanvas(canvas) {
  const rect = dom.stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
}

function blitToDisplay(displayCanvas, rgba, w, h) {
  displayScratchCanvas.width = w;
  displayScratchCanvas.height = h;
  displayScratchCtx.putImageData(new ImageData(rgba, w, h), 0, 0);
  const ctx = displayCanvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  ctx.drawImage(displayScratchCanvas, 0, 0, w, h, 0, 0, displayCanvas.width, displayCanvas.height);
}

function resizeDisplayCanvasesAndRedraw() {
  if (!previewImageData) return;
  for (const canvas of [dom.canvasOriginal, dom.canvasStencil, dom.canvasReference, dom.canvasColour]) {
    setupDisplayCanvas(canvas);
  }
  blitToDisplay(dom.canvasOriginal, previewImageData.data, previewW, previewH);
  renderAllCanvases();
}
window.addEventListener('resize', resizeDisplayCanvasesAndRedraw);

async function loadFile(file) {
  const bitmap = await createImageBitmap(file);
  sourceBitmap = bitmap;
  previewAnalyzed = false;

  const scale = Math.min(1, MAX_PREVIEW_DIM / Math.max(bitmap.width, bitmap.height));
  previewW = Math.max(1, Math.round(bitmap.width * scale));
  previewH = Math.max(1, Math.round(bitmap.height * scale));

  const scratch = document.createElement('canvas');
  scratch.width = previewW; scratch.height = previewH;
  const ctx = scratch.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, previewW, previewH);
  previewImageData = ctx.getImageData(0, 0, previewW, previewH);

  dom.stage.style.aspectRatio = `${previewW} / ${previewH}`;

  dom.uploadView.hidden = true;
  dom.editorView.hidden = false;
  dom.resetBtn.hidden = false;

  for (const canvas of [dom.canvasOriginal, dom.canvasStencil, dom.canvasReference, dom.canvasColour]) {
    setupDisplayCanvas(canvas);
  }
  blitToDisplay(dom.canvasOriginal, previewImageData.data, previewW, previewH);

  updateOutputSizeHint();
  setMode(currentMode);
  await runPreviewAnalysis();
}

// --- Preview pipeline (two-phase: heavy neural analysis once, cheap finalize per slider tick) ---
async function runPreviewAnalysis() {
  if (!previewImageData) return;
  setOverlay(true, 'Analyzing artwork… (first run downloads the AI model, ~17MB)');
  try {
    const sourceRgba = el('in-presharpen').checked
      ? unsharpMaskRGBA(previewImageData.data, previewW, previewH)
      : previewImageData.data.slice();
    const buffer = sourceRgba.buffer;
    await send({ type: 'previewAnalyze', buffer, width: previewW, height: previewH }, [buffer]);
    previewAnalyzed = true;
    await runPreviewFinalize();
  } catch (err) {
    console.error(err);
    alert('Could not analyze the image: ' + err.message);
  } finally {
    setOverlay(false);
  }
}

function schedulePreviewFinalize() {
  if (!previewAnalyzed) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runPreviewFinalize, DEBOUNCE_MS);
}

async function runPreviewFinalize() {
  const token = ++latestFinalizeToken;
  setOverlay(true, 'Updating…');
  try {
    const params = readParams();
    const result = await send({ type: 'previewFinalize', params });
    if (token !== latestFinalizeToken) return; // a newer edit superseded this one
    previewLayers = result;
    renderAllCanvases();
  } finally {
    if (token === latestFinalizeToken) setOverlay(false);
  }
}

function setOverlay(visible, label) {
  dom.overlay.hidden = !visible;
  if (label) dom.overlayLabel.textContent = label;
}

function renderStencilCanvas() {
  if (!previewLayers) return;
  const rgba = grayToRGBA(previewLayers.lines, previewLayers.width, previewLayers.height);
  blitToDisplay(dom.canvasStencil, rgba, previewLayers.width, previewLayers.height);
}

function renderReferenceCanvas() {
  if (!previewLayers) return;
  const rgba = grayToRGBA(previewLayers.reference, previewLayers.width, previewLayers.height);
  blitToDisplay(dom.canvasReference, rgba, previewLayers.width, previewLayers.height);
}

function renderColourCanvas() {
  if (!previewLayers) return;
  const refOpacity = Number(dom.refOpacity.value) / 100;
  const rgba = tintStencilOverReference(previewLayers.lines, previewLayers.reference, previewLayers.width, previewLayers.height, stencilColour, refOpacity);
  blitToDisplay(dom.canvasColour, rgba, previewLayers.width, previewLayers.height);
}

function renderAllCanvases() {
  renderStencilCanvas();
  renderReferenceCanvas();
  renderColourCanvas();
}

for (const btn of dom.colourSwatches) {
  btn.addEventListener('click', () => {
    stencilColour = btn.dataset.colour;
    for (const b of dom.colourSwatches) {
      b.classList.toggle('active', b === btn);
      b.setAttribute('aria-pressed', String(b === btn));
    }
    renderColourCanvas();
  });
}

dom.refOpacityOut.textContent = dom.refOpacity.value;
dom.refOpacity.addEventListener('input', () => {
  dom.refOpacityOut.textContent = dom.refOpacity.value;
  renderColourCanvas();
});

// --- Preview mode (Original / Stencil / Reference / ST + RF) -------------
function setMode(mode) {
  currentMode = mode;
  for (const btn of dom.modeButtons) {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  }
  dom.canvasOriginal.hidden = mode !== 'original';
  dom.canvasStencil.hidden = mode !== 'stencil';
  dom.canvasReference.hidden = mode !== 'reference';
  dom.canvasColour.hidden = mode !== 'colour';
}

for (const btn of dom.modeButtons) {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
}

// --- Mobile quick-adjust icons (vertical slider overlay on the preview) ---
// Most controls show a normalised 0-100 scale on the overlay, regardless of the
// real control's underlying range (e.g. Line Thickness is really -25..25).
// Reference Levels is the exception: it's a literal count of tone bands, so
// normalising it to 0-100 would misrepresent it — it keeps its own raw scale.
const quickSliderTargets = {
  'keep-detail': { input: sliders['keep-detail'].input, label: 'Keep Detail' },
  'bg-cleanup': { input: sliders['bg-cleanup'].input, label: 'Background Cleanup' },
  'thickness': { input: sliders['thickness'].input, label: 'Line Thickness' },
  'ref-opacity': { input: dom.refOpacity, label: 'Reference Opacity' },
  'ref-levels': { input: sliders['ref-levels'].input, label: 'Reference Levels', raw: true },
};

function percentFromValue(value, min, max) {
  if (max === min) return 0;
  return Math.round(((value - min) / (max - min)) * 100);
}

function valueFromPercent(percent, min, max, step) {
  const raw = min + (percent / 100) * (max - min);
  const snapped = Math.round(raw / step) * step;
  return Math.min(max, Math.max(min, snapped));
}

// Positions the custom dot to match the input's current value — see the CSS
// comment on .quick-slider-dot for why this isn't just the native thumb.
function updateQuickSliderDot() {
  const min = Number(dom.quickSliderInput.min);
  const max = Number(dom.quickSliderInput.max);
  const value = Number(dom.quickSliderInput.value);
  const percent = percentFromValue(value, min, max);
  dom.quickSliderDot.style.top = `${100 - percent}%`; // higher value = higher up
}

let activeQuickTarget = null;

function openQuickSlider(key) {
  const target = quickSliderTargets[key];
  if (!target) return;
  activeQuickTarget = key;
  const min = Number(target.input.min);
  const max = Number(target.input.max);
  if (target.raw) {
    dom.quickSliderInput.min = min;
    dom.quickSliderInput.max = max;
    dom.quickSliderInput.step = target.input.step || 1;
    dom.quickSliderInput.value = target.input.value;
    dom.quickSliderOutput.textContent = target.input.value;
  } else {
    dom.quickSliderInput.min = 0;
    dom.quickSliderInput.max = 100;
    dom.quickSliderInput.step = 1;
    const percent = percentFromValue(Number(target.input.value), min, max);
    dom.quickSliderInput.value = percent;
    dom.quickSliderOutput.textContent = percent;
  }
  updateQuickSliderDot();
  dom.quickSliderOverlay.hidden = false;
  dom.quickDescription.textContent = target.label;
  dom.quickDescription.classList.add('visible');
  for (const btn of dom.quickIconButtons) {
    const active = btn.dataset.target === key;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  }
}

function closeQuickSlider() {
  activeQuickTarget = null;
  dom.quickSliderOverlay.hidden = true;
  dom.quickDescription.classList.remove('visible');
  for (const btn of dom.quickIconButtons) {
    btn.classList.remove('active');
    btn.setAttribute('aria-pressed', 'false');
  }
}

for (const btn of dom.quickIconButtons) {
  btn.addEventListener('click', () => {
    const key = btn.dataset.target;
    if (activeQuickTarget === key) closeQuickSlider();
    else openQuickSlider(key);
  });
}

dom.quickSliderClose.addEventListener('click', closeQuickSlider);

dom.quickSliderInput.addEventListener('input', () => {
  if (!activeQuickTarget) return;
  const target = quickSliderTargets[activeQuickTarget];
  const { input } = target;
  if (target.raw) {
    input.value = dom.quickSliderInput.value;
    dom.quickSliderOutput.textContent = dom.quickSliderInput.value;
  } else {
    const min = Number(input.min);
    const max = Number(input.max);
    const step = Number(input.step) || 1;
    const percent = Number(dom.quickSliderInput.value);
    input.value = valueFromPercent(percent, min, max, step);
    dom.quickSliderOutput.textContent = percent;
  }
  updateQuickSliderDot();
  input.dispatchEvent(new Event('input', { bubbles: true }));
});

// --- Export ------------------------------------------------------------
function computeOutputPixelSize() {
  const widthCm = Number(dom.printWidth.value) || 15;
  const dpi = Number(dom.printDpi.value) || 300;
  const widthInches = widthCm / 2.54;
  const widthPx = Math.max(1, Math.round(widthInches * dpi));
  let heightPx = widthPx;
  if (sourceBitmap) heightPx = Math.max(1, Math.round(widthPx * (sourceBitmap.height / sourceBitmap.width)));
  return { widthPx, heightPx, dpi };
}

function updateOutputSizeHint() {
  const { widthPx, heightPx, dpi } = computeOutputPixelSize();
  dom.outputSizeHint.textContent = `Output size: ${widthPx} × ${heightPx} px at ${dpi} DPI`;
}
[dom.printWidth, dom.printDpi].forEach((elm) => elm.addEventListener('input', updateOutputSizeHint));

async function renderFullResLayers() {
  const { widthPx, heightPx } = computeOutputPixelSize();
  // The network runs at a capped resolution (quality plateaus above this, and it
  // keeps export time reasonable); the result is upscaled to the final print size,
  // which works well since line art is mostly smooth curves rather than fine texture.
  const analysisScale = Math.min(1, MAX_EXPORT_ANALYSIS_DIM / Math.max(widthPx, heightPx));
  const analysisW = Math.max(1, Math.round(widthPx * analysisScale));
  const analysisH = Math.max(1, Math.round(heightPx * analysisScale));

  const scratch = document.createElement('canvas');
  scratch.width = analysisW; scratch.height = analysisH;
  const ctx = scratch.getContext('2d');
  ctx.drawImage(sourceBitmap, 0, 0, analysisW, analysisH);
  const imageData = ctx.getImageData(0, 0, analysisW, analysisH);
  const params = readParams();
  const sourceRgba = el('in-presharpen').checked
    ? unsharpMaskRGBA(imageData.data, analysisW, analysisH)
    : imageData.data.slice();
  const buffer = sourceRgba.buffer;
  const result = await send({ type: 'export', buffer, width: analysisW, height: analysisH, params }, [buffer]);

  if (analysisW === widthPx && analysisH === heightPx) return result;
  return upscaleLayers(result, widthPx, heightPx);
}

function upscaleLayer(gray, srcW, srcH, dstW, dstH) {
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = srcW; srcCanvas.height = srcH;
  srcCanvas.getContext('2d').putImageData(new ImageData(grayToRGBA(gray, srcW, srcH), srcW, srcH), 0, 0);

  const dstCanvas = document.createElement('canvas');
  dstCanvas.width = dstW; dstCanvas.height = dstH;
  const dstCtx = dstCanvas.getContext('2d');
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = 'high';
  dstCtx.drawImage(srcCanvas, 0, 0, dstW, dstH);
  const data = dstCtx.getImageData(0, 0, dstW, dstH).data;

  const out = new Float32Array(dstW * dstH);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = data[p];
  return out;
}

function upscaleLayers(result, dstW, dstH) {
  return {
    width: dstW,
    height: dstH,
    lines: upscaleLayer(result.lines, result.width, result.height, dstW, dstH),
    reference: upscaleLayer(result.reference, result.width, result.height, dstW, dstH),
  };
}

function downloadRGBA(rgba, width, height, filename) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

function downloadGrayscale(gray, width, height, filename) {
  downloadRGBA(grayToRGBA(gray, width, height), width, height, filename);
}

async function withButtonBusy(button, label, fn) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try { await fn(); } finally { button.disabled = false; button.textContent = original; }
}

dom.exportStencilBtn.addEventListener('click', () => {
  if (!sourceBitmap) return;
  withButtonBusy(dom.exportStencilBtn, 'Rendering…', async () => {
    const result = await renderFullResLayers();
    downloadGrayscale(result.lines, result.width, result.height, 'stencil.png');
  });
});

dom.exportReferenceBtn.addEventListener('click', () => {
  if (!sourceBitmap) return;
  withButtonBusy(dom.exportReferenceBtn, 'Rendering…', async () => {
    const result = await renderFullResLayers();
    downloadGrayscale(result.reference, result.width, result.height, 'reference.png');
  });
});

dom.exportColourBtn.addEventListener('click', () => {
  if (!sourceBitmap) return;
  withButtonBusy(dom.exportColourBtn, 'Rendering…', async () => {
    const result = await renderFullResLayers();
    const refOpacity = Number(dom.refOpacity.value) / 100;
    const rgba = tintStencilOverReference(result.lines, result.reference, result.width, result.height, stencilColour, refOpacity);
    downloadRGBA(rgba, result.width, result.height, 'stencil-with-reference.png');
  });
});

updateOutputSizeHint();
