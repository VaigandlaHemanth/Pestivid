# Browser-inference parity harness (development only)

Not part of the shipped app. **Exclude `public/parity/` from any deploy** — see the
`.gitignore` entries. The app itself needs only `public/js/potato-browser.js`,
`public/js/pil-resize.js` and `public/model/`.

## Why this exists

The classifier head is a linear map fitted to DINOv2 features of images
preprocessed by torchvision/PIL. The browser has to reproduce that preprocessing
or the head silently sees shifted features: nothing errors, accuracy just drops.
This harness measures the drift instead of trusting it.

## What was found here

Running it turned up three things that were not guesses:

1. **transformers.js's in-browser resize does not antialias a downscale.** Its
   canvas path cost a cosine distance of 5.4e-2 to 1.4e-1 on the final feature
   vector. For reference, swapping between *proper* filters in PIL (bicubic vs
   bilinear vs Lanczos vs box) costs only 7e-4 to 2.4e-3 — the kernel barely
   matters, the antialiasing is everything. `pil-resize.js` replicates Pillow's
   `Resample.c` including its fixed-point 8-bit path, and now reproduces PIL
   **bit-for-bit**: 0 of 196,608 bytes differ on all 7 probe images.

2. **`dtype: 'q8'` is destroyed, not merely degraded** — 0.878 mean cosine
   distance, i.e. features essentially unrelated to what the head was fitted to,
   while the model still returns confident-looking probabilities. It is the
   obvious size/speed compromise and it is the one that must not be shipped.
   Measured, with preprocessing exact:

   | dtype | download | mean cosine distance |
   |-------|----------|----------------------|
   | fp16  | 173 MB   | 3.15e-4  ← shipped   |
   | fp32  | 347 MB   | 3.17e-4              |
   | q4f16 | 50 MB    | 4.12e-2              |
   | q8    | 87 MB    | 8.78e-1  BROKEN      |

3. **Mean-pooling is wrong for this head** (0.45–0.55); the CLS token is right.
   The ONNX export exposes only `last_hidden_state`, so index 0 of it is what
   corresponds to the `pooler_output` training read.

## Running it

    python ml-training/dump_parity.py <dataset-root> frontend/parity
    python ml-training/dump_pixels.py          # writes public/parity/pixels/ (~50 MB)
    # serve public/ on a static server, then open:
    #   /parity/           end-to-end feature parity, real JPEG decode  (expect ~3e-4)
    #   /parity/pixels.html  resize vs PIL, byte-for-byte               (expect 0 differing)
    #   /parity/dtype.html   per-dtype drift, preprocessing held exact

`/parity/` is the one to run after touching anything in the preprocessing chain.
Threshold: cosine distance < 1e-3. Above that, the server-measured accuracy no
longer describes what the browser does.
