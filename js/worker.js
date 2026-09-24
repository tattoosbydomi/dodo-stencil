// Classic (non-module) worker: onnxruntime-web is loaded via importScripts (its
// distributed build is UMD, not an ES module), while our own pipeline.js is
// pulled in via dynamic import() — supported in classic workers in all current
// evergreen browsers, and keeps pipeline.js a normal ES module.
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.10.0/dist/ort.min.js');

// Inside a worker loaded via importScripts, onnxruntime-web can't auto-detect
// where its .wasm binaries live (that detection relies on document/script-tag
// context, which doesn't exist here) — without this it silently falls back to
// a same-origin path, gets our index.html back on a 404, and fails trying to
// parse HTML as WASM. Point it at the CDN explicitly.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.10.0/dist/';

const MODEL_URL = new URL('../models/informative-drawings.onnx', self.location.href).href;

// Cache-busting: carry the ?v= this worker was instantiated with (see
// ASSET_VERSION in main.js) through to pipeline.js, since a dynamic import()
// specifier doesn't inherit it automatically.
const ASSET_VERSION = new URLSearchParams(self.location.search).get('v') || '1';
let pipelinePromise = import(`./pipeline.js?v=${ASSET_VERSION}`);
let sessionPromise = null;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
  }
  return sessionPromise;
}

// Cache of the most recently analyzed preview image, so slider tweaks
// (levels/thickness/crisp/invert/reference) never re-run the network —
// only a fresh photo upload does.
let previewCache = null; // { gray, rawLineMap, width, height }

// The network internally downsamples/upsamples by some fixed stride, so an
// input whose dimensions aren't a multiple of that stride comes back a pixel
// or two larger than requested (e.g. 659 -> 660) — silently reinterpreting
// that buffer at the original size is what caused the sheared/distorted
// output. Padding up to a safe multiple (with edge-replication, so we're not
// inventing new content at the border) and cropping back down afterward
// sidesteps the issue for any input size.
const NETWORK_STRIDE = 32;
function padToMultiple(value, multiple) {
  return Math.ceil(value / multiple) * multiple;
}

function rgbaToPlanarTensor(rgba, width, height, padW, padH) {
  const plane = padW * padH;
  const data = new Float32Array(3 * plane);
  for (let y = 0; y < padH; y++) {
    const sy = Math.min(y, height - 1);
    for (let x = 0; x < padW; x++) {
      const sx = Math.min(x, width - 1);
      const srcIdx = (sy * width + sx) * 4;
      const dstIdx = y * padW + x;
      data[dstIdx] = rgba[srcIdx] / 255;
      data[plane + dstIdx] = rgba[srcIdx + 1] / 255;
      data[2 * plane + dstIdx] = rgba[srcIdx + 2] / 255;
    }
  }
  return new ort.Tensor('float32', data, [1, 3, padH, padW]);
}

async function runNetwork(rgba, width, height) {
  const session = await getSession();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const padW = padToMultiple(width, NETWORK_STRIDE);
  const padH = padToMultiple(height, NETWORK_STRIDE);

  const results = await session.run({ [inputName]: rgbaToPlanarTensor(rgba, width, height, padW, padH) });
  const outputTensor = results[outputName];
  const [, , outH, outW] = outputTensor.dims;
  if (outW !== padW || outH !== padH) {
    throw new Error(`Model output size (${outW}x${outH}) does not match padded input (${padW}x${padH})`);
  }
  const raw = outputTensor.data;

  // Crop back down to the originally requested size (the padding was edge-replicated, not real content).
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = raw[y * padW + x] * 255;
      out[y * width + x] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

function packLayers(pipeline, rawLineMap, gray, width, height, params) {
  const lines = pipeline.finalizeLines(rawLineMap, params);
  const reference = pipeline.buildReferenceLayer(gray, params);
  return { lines, reference, width, height };
}

function respondWithLayers(requestId, result) {
  const lineBuf = Float32Array.from(result.lines).buffer;
  const referenceBuf = Float32Array.from(result.reference).buffer;
  self.postMessage({
    requestId, width: result.width, height: result.height,
    lineBuf, referenceBuf,
  }, [lineBuf, referenceBuf]);
}

self.onmessage = async (e) => {
  const { type, requestId } = e.data;
  try {
    const pipeline = await pipelinePromise;

    if (type === 'previewAnalyze') {
      const { buffer, width, height } = e.data;
      const rgba = new Uint8ClampedArray(buffer);
      const [rawLineMap, gray] = await Promise.all([
        runNetwork(rgba, width, height),
        Promise.resolve(pipeline.toGrayscale(rgba, width, height)),
      ]);
      previewCache = { rawLineMap, gray, width, height };
      self.postMessage({ requestId, type: 'previewAnalyzeDone' });
      return;
    }

    if (type === 'previewFinalize') {
      if (!previewCache) throw new Error('No image analyzed yet');
      const { params } = e.data;
      const result = packLayers(pipeline, previewCache.rawLineMap, previewCache.gray, previewCache.width, previewCache.height, params);
      respondWithLayers(requestId, result);
      return;
    }

    if (type === 'export') {
      const { buffer, width, height, params } = e.data;
      const rgba = new Uint8ClampedArray(buffer);
      const rawLineMap = await runNetwork(rgba, width, height);
      const gray = pipeline.toGrayscale(rgba, width, height);
      const result = packLayers(pipeline, rawLineMap, gray, width, height, params);
      respondWithLayers(requestId, result);
      return;
    }

    throw new Error(`Unknown message type: ${type}`);
  } catch (err) {
    self.postMessage({ requestId, error: err.message });
  }
};
