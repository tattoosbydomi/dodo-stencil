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

// Otsu's method: picks the cutoff that best separates a grayscale histogram into
// two classes (line vs. paper) by maximizing between-class variance. Used instead
// of a fixed threshold (128) so "Crisp" lines stay clean across photos where the
// network's soft output happens to sit at a different brightness overall — a
// fixed cutoff either eats thin lines or lets background haze through depending
// on the image, while this adapts per photo.
export function otsuThreshold(src) {
  const hist = new Array(256).fill(0);
  for (let i = 0; i < src.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(src[i])));
    hist[v]++;
  }
  const total = src.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];

  let sumB = 0, weightB = 0, maxVariance = 0, threshold = 128;
  for (let t = 0; t < 256; t++) {
    weightB += hist[t];
    if (weightB === 0) continue;
    const weightF = total - weightB;
    if (weightF === 0) break;
    sumB += t * hist[t];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const variance = weightB * weightF * (meanB - meanF) * (meanB - meanF);
    if (variance > maxVariance) { maxVariance = variance; threshold = t; }
  }
  return threshold;
}

// Zhang-Suen thinning: iteratively erodes a binary foreground (ink) mask down to
// a 1-pixel-wide skeleton while preserving connectivity and branching (Y/T
// junctions survive, unlike a naive contour trace which can only represent
// non-branching paths). This is what turns a thick, unevenly-blurry threshold
// result into a single clean centerline per stroke.
// maxIterations bounds worst-case runtime: each outer iteration only erodes one
// layer off every blob's boundary, so thin stencil lines (the normal case, a
// handful of iterations) finish long before this cap is ever reached — it only
// kicks in for a pathological input (e.g. a large solid dark region the network/
// threshold misclassified as "line"), where full erosion could otherwise take
// tens of seconds. Any such remainder is still usable, just not fully thinned.
export function skeletonize(binaryGray, w, h, maxIterations = 40) {
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = binaryGray[i] < 128 ? 1 : 0;

  const at = (x, y) => (x < 0 || x >= w || y < 0 || y >= h) ? 0 : fg[y * w + x];

  let changed = true;
  let iterations = 0;
  while (changed && iterations++ < maxIterations) {
    changed = false;
    for (let subIter = 0; subIter < 2; subIter++) {
      const toRemove = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          if (!fg[y * w + x]) continue;
          const p2 = at(x, y - 1), p3 = at(x + 1, y - 1), p4 = at(x + 1, y);
          const p5 = at(x + 1, y + 1), p6 = at(x, y + 1), p7 = at(x - 1, y + 1);
          const p8 = at(x - 1, y), p9 = at(x - 1, y - 1);
          const ring = [p2, p3, p4, p5, p6, p7, p8, p9];
          const blackNeighbours = ring.reduce((a, b) => a + b, 0);
          if (blackNeighbours < 2 || blackNeighbours > 6) continue;
          let transitions = 0;
          for (let k = 0; k < 8; k++) {
            if (ring[k] === 0 && ring[(k + 1) % 8] === 1) transitions++;
          }
          if (transitions !== 1) continue;
          if (subIter === 0) {
            if (p2 * p4 * p6 !== 0) continue;
            if (p4 * p6 * p8 !== 0) continue;
          } else {
            if (p2 * p4 * p8 !== 0) continue;
            if (p2 * p6 * p8 !== 0) continue;
          }
          toRemove.push(y * w + x);
        }
      }
      if (toRemove.length) {
        changed = true;
        for (const i of toRemove) fg[i] = 0;
      }
    }
  }

  const out = new Float32Array(w * h).fill(255);
  for (let i = 0; i < out.length; i++) if (fg[i]) out[i] = 0;
  return out;
}

// Re-thickens a (typically skeletonized) binary line to a constant radius, so
// every stroke reads at the same weight regardless of how thick/thin the network's
// soft output happened to make it before thresholding.
export function dilate(binaryGray, w, h, radius) {
  if (radius <= 0) return binaryGray;
  const fg = new Uint8Array(w * h);
  for (let i = 0; i < fg.length; i++) fg[i] = binaryGray[i] < 128 ? 1 : 0;
  const out = new Uint8Array(w * h);
  const r2 = radius * radius;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!fg[y * w + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > r2) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          out[yy * w + xx] = 1;
        }
      }
    }
  }
  const result = new Float32Array(w * h).fill(255);
  for (let i = 0; i < result.length; i++) if (out[i]) result[i] = 0;
  return result;
}

function boxBlur3(src, w, h) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(w - 1, x + 1);
      tmp[row + x] = (src[row + x0] + src[row + x] + src[row + x1]) / 3;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - 1), y1 = Math.min(h - 1, y + 1);
      out[y * w + x] = (tmp[y0 * w + x] + tmp[y * w + x] + tmp[y1 * w + x]) / 3;
    }
  }
  return out;
}

// Cheap unsharp mask on the source photo (before it ever reaches the line-art
// network): blur each channel, then push the original away from that blur.
// Helps when the *photo* itself is soft/low-contrast, which is a case no amount
// of post-processing the network's output can fix — the network never had a
// strong gradient to lock onto in the first place.
export function unsharpMaskRGBA(rgba, w, h, amount = 0.8) {
  const plane = w * h;
  const out = new Uint8ClampedArray(rgba.length);
  for (let c = 0; c < 3; c++) {
    const channel = new Float32Array(plane);
    for (let i = 0, p = c; i < plane; i++, p += 4) channel[i] = rgba[p];
    const blurred = boxBlur3(channel, w, h);
    for (let i = 0, p = c; i < plane; i++, p += 4) {
      out[p] = channel[i] + amount * (channel[i] - blurred[i]);
    }
  }
  for (let i = 0, p = 3; i < plane; i++, p += 4) out[p] = 255;
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
//
// `lineStyle` picks how (or whether) that soft gradient then gets crisped up —
// see index.html's "Line Style" dropdown for the four methodologies being compared:
//   'soft'           — the network's gradient as-is, no thresholding.
//   'threshold'      — flat cutoff at 128 (the original "Crisp lines" behaviour).
//   'auto-threshold' — same cutoff, but picked per-image via Otsu instead of fixed.
//   'clean'          — auto-threshold, then skeletonize + re-thicken to a constant
//                      width, so every stroke reads the same regardless of how
//                      thick/thin/blurry the raw network output made it.
export function finalizeLines(rawLineMap, width, height, params) {
  const shift = params.lineThickness || 0;
  const blackPoint = clamp255(params.blackPoint + shift);
  const whitePoint = clamp255(params.whitePoint + shift);
  let lines = applyLevels(rawLineMap, blackPoint, whitePoint);

  const style = params.lineStyle || 'soft';
  if (style === 'threshold') {
    lines = binarize(lines, 128);
  } else if (style === 'auto-threshold') {
    lines = binarize(lines, otsuThreshold(lines));
  } else if (style === 'clean') {
    lines = binarize(lines, otsuThreshold(lines));
    lines = skeletonize(lines, width, height);
    const radius = Math.max(1, Math.min(5, Math.round(1 + shift / 10)));
    lines = dilate(lines, width, height, radius);
  }

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
