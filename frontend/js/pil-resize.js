/**
 * PIL-exact image resize for the browser.
 *
 * WHY THIS FILE EXISTS — a measured bug, not a precaution.
 *
 * The classifier head is a linear map fitted to DINOv2 features of images
 * preprocessed by torchvision/PIL: Resize(shortest=256, BICUBIC) ->
 * CenterCrop(224) -> /255 -> ImageNet normalise. transformers.js does its
 * in-browser resize through canvas drawImage, and canvas downscaling does NOT
 * apply a proper antialiasing filter — for a 1500px photo going to 256px it
 * behaves closer to point sampling.
 *
 * Measured cost of that, on real dataset images (cosine distance of the final
 * 768-d feature vector against the pinned PIL BICUBIC transform):
 *
 *     BILINEAR            1.4e-3      \  proper antialiased filters all agree
 *     LANCZOS             7.3e-4       |  to ~1e-3, i.e. the KERNEL barely
 *     BOX                 2.4e-3      /  matters
 *     NEAREST (aliased)   4.7e-2 mean, 1.3e-1 max   <-- no antialiasing
 *
 *     transformers.js in-browser, observed:  5.4e-2 to 1.4e-1
 *
 * The browser numbers sit exactly on the aliased row. Separately measured: the
 * dtype is irrelevant (fp32 and q4f16 agreed to four significant figures at
 * 1.025e-1), and CLS-token pooling is correct (mean-pooling gave 0.45-0.55). So
 * resampling was the whole error, and this file removes it.
 *
 * WHAT IS REPLICATED
 *   Pillow's Resample.c, including the fixed-point 8-bit path, because that is
 *   what actually ran on the training images:
 *     * support scales with the downscale factor (this IS the antialiasing)
 *     * bicubic kernel with a = -0.5
 *     * per-output-pixel weight normalisation
 *     * horizontal pass then vertical pass, clipped to uint8 between them
 *     * PRECISION_BITS = 22 fixed-point accumulate with a rounding half-add
 *   Reproducing the intermediate uint8 clipping matters: skipping it is "more
 *   accurate" in the abstract and further from what the head was trained on.
 */

const PRECISION_BITS = 22;
const HALF = 1 << (PRECISION_BITS - 1);

/** Pillow's bicubic_filter, a = -0.5. Support = 2.0. */
function bicubic(x) {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

/**
 * Pillow's precompute_coeffs followed by normalize_coeffs_8bpc.
 * Returns per-output-pixel {min, len} bounds and int32 fixed-point weights.
 */
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(1.0, scale);
  const support = 2.0 * filterscale;              // bicubic support * scale
  const ksize = Math.ceil(support) * 2 + 1;
  const k = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const tmp = new Float64Array(ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;

    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bicubic((x + xmin - center + 0.5) * ss);
      tmp[x] = w;
      ww += w;
    }
    if (ww !== 0) for (let x = 0; x < xmax; x++) tmp[x] /= ww;

    // Pillow quantises the normalised weights to fixed point here. Rounding
    // away from zero matches its (int)(x < 0 ? -0.5 : +0.5) idiom.
    const base = xx * ksize;
    for (let x = 0; x < xmax; x++) {
      const v = tmp[x] * (1 << PRECISION_BITS);
      k[base + x] = v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { k, bounds, ksize };
}

function clip8(v) {
  const s = Math.floor(v / (1 << PRECISION_BITS));
  return s < 0 ? 0 : s > 255 ? 255 : s;
}

/**
 * Resize tightly packed RGB uint8 data exactly as PIL's BICUBIC does:
 * horizontal pass into an (outW x inH) intermediate, then a vertical pass.
 */
export function pilResizeRGB(src, inW, inH, outW, outH) {
  let cur = src, curW = inW, curH = inH;

  if (curW !== outW) {
    const { k, bounds, ksize } = coeffs(curW, outW);
    const dst = new Uint8ClampedArray(outW * curH * 3);
    for (let y = 0; y < curH; y++) {
      const rowIn = y * curW * 3, rowOut = y * outW * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], len = bounds[xx * 2 + 1], kb = xx * ksize;
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < len; i++) {
          const w = k[kb + i], p = rowIn + (xmin + i) * 3;
          r += cur[p] * w; g += cur[p + 1] * w; b += cur[p + 2] * w;
        }
        const o = rowOut + xx * 3;
        dst[o] = clip8(r); dst[o + 1] = clip8(g); dst[o + 2] = clip8(b);
      }
    }
    cur = dst; curW = outW;
  }

  if (curH !== outH) {
    const { k, bounds, ksize } = coeffs(curH, outH);
    const dst = new Uint8ClampedArray(curW * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2], len = bounds[yy * 2 + 1], kb = yy * ksize;
      const rowOut = yy * curW * 3;
      for (let x = 0; x < curW; x++) {
        let r = HALF, g = HALF, b = HALF;
        for (let i = 0; i < len; i++) {
          const w = k[kb + i], p = ((ymin + i) * curW + x) * 3;
          r += cur[p] * w; g += cur[p + 1] * w; b += cur[p + 2] * w;
        }
        const o = rowOut + x * 3;
        dst[o] = clip8(r); dst[o + 1] = clip8(g); dst[o + 2] = clip8(b);
      }
    }
    cur = dst; curH = outH;
  }
  return { data: cur, width: curW, height: curH };
}

/**
 * Python's round(), which is NOT JavaScript's Math.round().
 *
 * torchvision computes the centre-crop offset as
 *
 *     crop_top = int(round((image_height - crop_height) / 2.0))
 *
 * and Python 3's round() is banker's rounding -- ties go to the EVEN integer.
 * Math.round() breaks ties upward. They agree everywhere except on an exact .5,
 * which is exactly what happens whenever (side - crop) is odd:
 *
 *     round(16.5) = 16      Math.round(16.5) = 17
 *     round(17.5) = 18      Math.round(17.5) = 18
 *
 * A non-square photo resized with Resize(256) usually lands on an odd long side,
 * so the odd case is the COMMON one for real camera photos, not an edge case. The
 * browser was cropping one pixel to the right of, and one pixel below, where the
 * server cropped. Every patch token then sees slightly different pixels.
 *
 * That is small, but the whole point of the parity harness is that the browser
 * computes the same features the accuracy figures were measured on. A silent
 * one-pixel shift on most real photographs is not something to leave in because
 * it happens to be small.
 */
function pyRound(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  // Exact tie: choose the even neighbour, as Python does.
  return (floor % 2 === 0) ? floor : floor + 1;
}

/**
 * The full pinned transform, producing a CHW float32 tensor ready for DINOv2.
 *
 * torchvision Resize(256) with an int size scales the SHORTER side to 256 and
 * TRUNCATES the other (int(), not round()) — reproduced exactly below. CenterCrop
 * is reproduced too, but its offset is round-half-to-EVEN (Python's round), not
 * round-half-up: see pyRound().
 */
export function preprocess(rgb, inW, inH, opts) {
  const o = opts || {};
  const shortestEdge = o.shortestEdge || 256;
  const crop = o.crop || 224;
  const mean = o.mean || [0.485, 0.456, 0.406];
  const std = o.std || [0.229, 0.224, 0.225];

  let ow, oh;
  if (inW < inH) { ow = shortestEdge; oh = Math.trunc(shortestEdge * inH / inW); }
  else { oh = shortestEdge; ow = Math.trunc(shortestEdge * inW / inH); }
  // torchvision short-circuits when the shorter side already matches.
  if ((inW <= inH && inW === shortestEdge) || (inH <= inW && inH === shortestEdge)) {
    ow = inW; oh = inH;
  }

  const r = pilResizeRGB(rgb, inW, inH, ow, oh);
  // pyRound, not Math.round: see the note above. torchvision's CenterCrop uses
  // Python's banker's rounding, and the two disagree on every odd difference.
  const top = pyRound((r.height - crop) / 2);
  const left = pyRound((r.width - crop) / 2);

  const out = new Float32Array(3 * crop * crop);
  const plane = crop * crop;
  for (let y = 0; y < crop; y++) {
    for (let x = 0; x < crop; x++) {
      const sy = top + y, sx = left + x;
      // Edge clamp: only reachable if the source was smaller than the crop.
      const cy = sy < 0 ? 0 : sy >= r.height ? r.height - 1 : sy;
      const cx = sx < 0 ? 0 : sx >= r.width ? r.width - 1 : sx;
      const p = (cy * r.width + cx) * 3, q = y * crop + x;
      out[q] = (r.data[p] / 255 - mean[0]) / std[0];
      out[plane + q] = (r.data[p + 1] / 255 - mean[1]) / std[1];
      out[2 * plane + q] = (r.data[p + 2] / 255 - mean[2]) / std[2];
    }
  }
  return { data: out, dims: [1, 3, crop, crop], resized: [ow, oh] };
}

/**
 * The 2x2 tiled transform: four non-overlapping crops of the SAME size the model
 * normally sees, taken from a region twice as large in each dimension.
 *
 * WHY THIS RATHER THAN A BIGGER CROP
 *   A 448px centre crop and a 2x2 grid of 224px tiles from that same 448px region
 *   are pixel-identical in field of view, ground sampling and total pixels
 *   (200,704). Tiling is cheaper (4 x 257 tokens versus one 1025-token pass, and
 *   attention is quadratic) and it keeps the ONNX input shape at exactly 224x224 --
 *   the shape already being run -- so peak memory stays flat on a phone and no
 *   position-embedding interpolation is needed.
 *
 * WHY IT WAS WORTH ADDING
 *   Measured on grouped 5-fold CV: global CLS alone 0.8622 accuracy, global + the
 *   two tile views 0.8816, and 0.8899 with the probe ensemble -- better than
 *   adding a second 609 MB backbone, from the same 173 MB model. Abstention drops
 *   from 10% of photos to 5%.
 *
 * Returns { tiles: [{data, dims}, ...] } in row-major order. Order does not
 * matter for the mean/max pooling the head expects, but is kept deterministic so
 * parity tests are reproducible.
 */
export function preprocessTiles(rgb, inW, inH, opts) {
  const o = opts || {};
  const grid = o.grid || 2;
  const crop = o.crop || 224;
  const shortestEdge = o.shortestEdge || 256;
  const mean = o.mean || [0.485, 0.456, 0.406];
  const std = o.std || [0.229, 0.224, 0.225];

  // Resize and crop grid-times larger, then cut. Same arithmetic as
  // Pestivid/extract_views.py TiledImages, which produced the training features.
  const bigShort = shortestEdge * grid;
  const bigCrop = crop * grid;

  let ow, oh;
  if (inW < inH) { ow = bigShort; oh = Math.trunc(bigShort * inH / inW); }
  else { oh = bigShort; ow = Math.trunc(bigShort * inW / inH); }
  if ((inW <= inH && inW === bigShort) || (inH <= inW && inH === bigShort)) {
    ow = inW; oh = inH;
  }

  const r = pilResizeRGB(rgb, inW, inH, ow, oh);
  // Same rounding rule as the global path -- the tiled features are fitted from
  // a torchvision CenterCrop too, so a one-pixel offset here shifts all four
  // tiles at once.
  const top = pyRound((r.height - bigCrop) / 2);
  const left = pyRound((r.width - bigCrop) / 2);

  const tiles = [];
  const plane = crop * crop;
  for (let ty = 0; ty < grid; ty++) {
    for (let tx = 0; tx < grid; tx++) {
      const out = new Float32Array(3 * plane);
      for (let y = 0; y < crop; y++) {
        for (let x = 0; x < crop; x++) {
          const sy = top + ty * crop + y;
          const sx = left + tx * crop + x;
          const cy = sy < 0 ? 0 : sy >= r.height ? r.height - 1 : sy;
          const cx = sx < 0 ? 0 : sx >= r.width ? r.width - 1 : sx;
          const pi = (cy * r.width + cx) * 3;
          const q = y * crop + x;
          out[q] = (r.data[pi] / 255 - mean[0]) / std[0];
          out[plane + q] = (r.data[pi + 1] / 255 - mean[1]) / std[1];
          out[2 * plane + q] = (r.data[pi + 2] / 255 - mean[2]) / std[2];
        }
      }
      tiles.push({ data: out, dims: [1, 3, crop, crop] });
    }
  }
  return { tiles, resized: [ow, oh], cropSize: bigCrop };
}

/** Decode any drawable / Blob / URL into tightly packed RGB bytes. */
export async function toRGB(input) {
  let bmp = input;
  if (typeof input === 'string') {
    bmp = await createImageBitmap(await (await fetch(input)).blob());
  } else if (typeof Blob !== 'undefined' && input instanceof Blob) {
    bmp = await createImageBitmap(input);
  }
  const w = bmp.naturalWidth || bmp.videoWidth || bmp.width;
  const h = bmp.naturalHeight || bmp.videoHeight || bmp.height;
  if (!w || !h) throw new Error('Cannot determine image dimensions.');

  const cv = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const cx = cv.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  cx.imageSmoothingEnabled = false;        // 1:1 draw; nothing should resample
  cx.drawImage(bmp, 0, 0);
  const d = cx.getImageData(0, 0, w, h, { colorSpace: 'srgb' }).data;   // RGBA

  const rgb = new Uint8ClampedArray(w * h * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) {
    rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2];
  }
  return { rgb, width: w, height: h };
}
