// Pure image-processing primitives. No DOM access — safe to run in a Web Worker.
// All "channel" values below are single-plane Float32Array grayscale buffers, 0-255.
//
// Line art itself comes from a neural network (see worker.js) rather than a
// classical edge filter — a plain filter can only ever find "every edge in the
// pixels," never "the lines an artist chose to keep." Everything in this file
// operates on top of that network's output (or on the original photo, for the
// reference layer).

export function toGrayscale(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    // Perceptual luminance weighting — holds up better on skin tones than a flat average.
    out[i] = 0.2126 * rgba[p] + 0.7152 * rgba[p + 1] + 0.0722 * rgba[p + 2];
  }
  return out;
}

function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

// Photoshop-style levels remap: push everything at/below blackPoint to 0 (line),
// everything at/above whitePoint to 255 (paper), linear ramp between. This is how
// the "Keep Detail" / "Background Cleanup" sliders clean up the network's raw
// (slightly soft/grayish) output into confident line art.
export function applyLevels(src, blackPoint, whitePoint) {
  const range = Math.max(1, whitePoint - blackPoint);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = clamp255(((src[i] - blackPoint) / range) * 255);
  }
  return out;
}

export function binarize(src, threshold = 128) {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] >= threshold ? 255 : 0;
  return out;
}

export function posterize(src, levels) {
  const step = 255 / (levels - 1);
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = Math.round(Math.round(src[i] / step) * step);
  return out;
}

export function grayToRGBA(gray, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const v = gray[i];
    out[p] = v; out[p + 1] = v; out[p + 2] = v; out[p + 3] = 255;
  }
  return out;
}

function hexToRgb(hex) {
  const num = parseInt(hex.replace('#', ''), 16);
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}

// Layers the stencil (tinted with colourHex) on top of the grayscale reference
// photo: darker stencil pixels (lines) become more opaque colour, while
// paper-white stencil pixels are fully transparent and let the reference show
// through underneath — so the two outputs read as one combined image.
// refOpacity (0-1) fades the reference photo toward white before that blend, so
// the tinted lines pop out more without changing their own colour.
export function tintStencilOverReference(stencilGray, referenceGray, w, h, colourHex, refOpacity = 1) {
  const { r, g, b } = hexToRgb(colourHex);
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0, p = 0; i < stencilGray.length; i++, p += 4) {
    const alpha = (255 - stencilGray[i]) / 255;
    const ref = referenceGray[i] * refOpacity + 255 * (1 - refOpacity);
    out[p] = r * alpha + ref * (1 - alpha);
    out[p + 1] = g * alpha + ref * (1 - alpha);
    out[p + 2] = b * alpha + ref * (1 - alpha);
    out[p + 3] = 255;
  }
  return out;
}

// Post-processes the neural network's raw line-art output (0-255 grayscale,
// network already produced) into the final "lines" layer.
//
// "Line Thickness" is implemented as a shift of the black/white points
// together (same knobs "Keep Detail"/"Background Cleanup" already expose),
// applied before those levels run — not as pixel-radius morphology. Morphology
// reassigns each pixel to the darkest/lightest value within a radius, which for
// soft anti-aliased line art is a flood, not a nudge: even radius 1 could wipe
// thin strokes out entirely (thinner) or blow them into solid blobs (thicker),
// with nothing gradual in between. Shifting the levels window instead moves
// where *within the network's existing soft gradient* the line/paper cutoff
// falls, so it stays continuous and reuses the exact remap that already
// produces clean results for the other two sliders.
export function finalizeLines(rawLineMap, params) {
  const shift = params.lineThickness || 0;
  const blackPoint = clamp255(params.blackPoint + shift);
  const whitePoint = clamp255(params.whitePoint + shift);
  let lines = applyLevels(rawLineMap, blackPoint, whitePoint);
  if (params.crisp) lines = binarize(lines, 128);
  if (params.invertLines) {
    for (let i = 0; i < lines.length; i++) lines[i] = 255 - lines[i];
  }
  return lines;
}

// Grayscale reference photo, derived from the original artwork rather than
// the line-art network.
export function buildReferenceLayer(originalGray, params) {
  if (params.referencePosterize) return posterize(originalGray, params.referenceLevels);
  return originalGray;
}
