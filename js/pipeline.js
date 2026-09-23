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

// Grayscale morphology: negative radius thins lines (max filter / erode-the-dark),
// positive radius thickens lines (min filter / dilate-the-dark). 0 = no-op.
export function adjustLineThickness(src, w, h, radius) {
  if (!radius) return Float32Array.from(src);
  const grow = radius > 0;
  const r = Math.abs(radius);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let best = grow ? 255 : 0;
      for (let dy = -r; dy <= r; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const v = src[yy * w + xx];
          best = grow ? Math.min(best, v) : Math.max(best, v);
        }
      }
      out[y * w + x] = best;
    }
  }
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
export function finalizeLines(rawLineMap, w, h, params) {
  let lines = applyLevels(rawLineMap, params.blackPoint, params.whitePoint);
  if (params.crisp) lines = binarize(lines, 128);
  if (params.lineThickness) lines = adjustLineThickness(lines, w, h, params.lineThickness);
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
