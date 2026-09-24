// Cache-busting (see index.html): this static import's ?v=, ASSET_VERSION
// below, and index.html's ?v= must all be bumped together on every deploy
// that touches js/ or css/. ASSET_VERSION is threaded through to the
// dynamically-loaded worker.js and pipeline.js further down, since import
// specifiers (static or dynamic) don't inherit this file's own query string.
import { grayToRGBA, tintStencilOverReference } from './pipeline.js?v=1';
const ASSET_VERSION = '1';

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
  canvasColour: el('canvas-colour'),
  splitHandle: el('split-handle'),
  overlay: el('processing-overlay'),
  overlayLabel: el('processing-label'),
  modeButtons: Array.from(document.querySelectorAll('.mode-btn')),
  layoutButtons: Array.from(document.querySelectorAll('.layout-btn')),
  exportStencilBtn: el('export-stencil-btn'),
  exportReferenceBtn: el('export-reference-btn'),
  exportColourBtn: el('export-colour-btn'),
  colourSwatches: Array.from(document.querySelectorAll('.swatch-btn')),
  refOpacity: el('in-ref-opacity'),
  refOpacityOut: el('out-ref-opacity'),
  hideLinesBtn: el('toggle-hide-lines'),
  outputSizeHint: el('output-size-hint'),
  printWidth: el('in-print-width'),
  printUnit: el('in-print-unit'),
  printDpi: el('in-print-dpi'),
};

const sliderIds = [
  'keep-detail', 'bg-cleanup', 'thickness',
  'ref-levels',
];
const sliders = {};
for (const id of sliderIds) {
  sliders[id] = { input: el(`in-${id}`), output: el(`out-${id}`) };
  sliders[id].output.textContent = sliders[id].input.value;
  sliders[id].input.addEventListener('input', () => {
    sliders[id].output.textContent = sliders[id].input.value;
    schedulePreviewFinalize();
  });
}

const checkboxIds = ['crisp', 'invert', 'ref-posterize'];
for (const id of checkboxIds) {
  el(`in-${id}`).addEventListener('change', schedulePreviewFinalize);
}

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
let linesHidden = false;
let splitPercent = 50;
let debounceTimer = null;
let latestFinalizeToken = 0;
let previewAnalyzed = false;

function readParams() {
  const v = (id) => Number(sliders[id].input.value);
  return {
    blackPoint: v('keep-detail'),
    whitePoint: v('bg-cleanup'),
    crisp: el('in-crisp').checked,
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
  dom.fileInput.value = '';
  dom.editorView.hidden = true;
  dom.uploadView.hidden = false;
  dom.resetBtn.hidden = true;
});

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

  dom.canvasOriginal.width = previewW; dom.canvasOriginal.height = previewH;
  dom.canvasStencil.width = previewW; dom.canvasStencil.height = previewH;
  dom.canvasColour.width = previewW; dom.canvasColour.height = previewH;
  dom.canvasOriginal.getContext('2d').putImageData(previewImageData, 0, 0);
  dom.stage.style.aspectRatio = `${previewW} / ${previewH}`;

  dom.uploadView.hidden = true;
  dom.editorView.hidden = false;
  dom.resetBtn.hidden = false;

  updateOutputSizeHint();
  setMode(currentMode);
  await runPreviewAnalysis();
}

// --- Preview pipeline (two-phase: heavy neural analysis once, cheap finalize per slider tick) ---
async function runPreviewAnalysis() {
  if (!previewImageData) return;
  setOverlay(true, 'Analyzing artwork… (first run downloads the AI model, ~17MB)');
  try {
    const buffer = previewImageData.data.slice().buffer;
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
  const imageData = new ImageData(rgba, previewLayers.width, previewLayers.height);
  dom.canvasStencil.getContext('2d').putImageData(imageData, 0, 0);
}

function renderColourCanvas() {
  if (!previewLayers) return;
  const lines = linesHidden
    ? new Float32Array(previewLayers.lines.length).fill(255) // paper-white = fully transparent overlay, so only the reference shows
    : previewLayers.lines;
  const refOpacity = linesHidden ? 1 : Number(dom.refOpacity.value) / 100;
  const rgba = tintStencilOverReference(lines, previewLayers.reference, previewLayers.width, previewLayers.height, stencilColour, refOpacity);
  const imageData = new ImageData(rgba, previewLayers.width, previewLayers.height);
  dom.canvasColour.getContext('2d').putImageData(imageData, 0, 0);
}

function renderAllCanvases() {
  renderStencilCanvas();
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

dom.hideLinesBtn.addEventListener('click', () => {
  linesHidden = !linesHidden;
  dom.hideLinesBtn.setAttribute('aria-pressed', String(linesHidden));
  dom.hideLinesBtn.textContent = linesHidden ? 'Show Stencil Lines (preview)' : 'Hide Stencil Lines (preview)';
  renderColourCanvas();
});

// --- Preview mode (Original / Stencil / Split) ---------------------------
function setMode(mode) {
  currentMode = mode;
  for (const btn of dom.modeButtons) {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  }
  dom.canvasColour.hidden = mode !== 'colour';
  if (mode === 'original') {
    dom.canvasStencil.style.clipPath = 'inset(0 0 0 100%)';
    dom.splitHandle.hidden = true;
  } else if (mode === 'stencil') {
    dom.canvasStencil.style.clipPath = 'inset(0 0 0 0%)';
    dom.splitHandle.hidden = true;
  } else if (mode === 'colour') {
    dom.splitHandle.hidden = true;
  } else {
    dom.splitHandle.hidden = false;
    applySplitPercent();
  }
}

function applySplitPercent() {
  dom.canvasStencil.style.clipPath = `inset(0 0 0 ${splitPercent}%)`;
  dom.splitHandle.style.left = `${splitPercent}%`;
}

for (const btn of dom.modeButtons) {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
}

// --- Mobile layout (Full Preview / Live Edit) -----------------------------
function setLayout(layout) {
  dom.editorView.classList.toggle('live-layout', layout === 'live');
  for (const btn of dom.layoutButtons) {
    const active = btn.dataset.layout === layout;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  }
}

for (const btn of dom.layoutButtons) {
  btn.addEventListener('click', () => setLayout(btn.dataset.layout));
}

let dragging = false;
function pointerToPercent(clientX) {
  const rect = dom.stage.getBoundingClientRect();
  return Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
}
dom.stage.addEventListener('pointerdown', (e) => {
  if (currentMode !== 'split') return;
  dragging = true;
  splitPercent = pointerToPercent(e.clientX);
  applySplitPercent();
  dom.stage.setPointerCapture(e.pointerId);
});
dom.stage.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  splitPercent = pointerToPercent(e.clientX);
  applySplitPercent();
});
dom.stage.addEventListener('pointerup', () => { dragging = false; });
dom.stage.addEventListener('pointercancel', () => { dragging = false; });

// --- Export ------------------------------------------------------------
function computeOutputPixelSize() {
  const widthVal = Number(dom.printWidth.value) || 6;
  const unit = dom.printUnit.value;
  const dpi = Number(dom.printDpi.value) || 300;
  const widthInches = unit === 'cm' ? widthVal / 2.54 : widthVal;
  const widthPx = Math.max(1, Math.round(widthInches * dpi));
  let heightPx = widthPx;
  if (sourceBitmap) heightPx = Math.max(1, Math.round(widthPx * (sourceBitmap.height / sourceBitmap.width)));
  return { widthPx, heightPx, dpi };
}

function updateOutputSizeHint() {
  const { widthPx, heightPx, dpi } = computeOutputPixelSize();
  dom.outputSizeHint.textContent = `Output size: ${widthPx} × ${heightPx} px at ${dpi} DPI`;
}
[dom.printWidth, dom.printUnit, dom.printDpi].forEach((elm) => elm.addEventListener('input', updateOutputSizeHint));

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
  const buffer = imageData.data.slice().buffer;
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
