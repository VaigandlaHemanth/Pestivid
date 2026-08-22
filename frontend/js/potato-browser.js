/**
 * potato-browser.js — run the potato leaf classifier entirely in the browser.
 *
 * WHY THIS EXISTS
 *   The backbone is a frozen, unmodified public checkpoint with an ONNX export on
 *   the Hub, and everything after it is a Linear layer, one divide, and a
 *   Mahalanobis distance. So there is no reason to host an ML server at all —
 *   which matters because in 2026 every free 512 MB container is smaller than the
 *   torch wheel alone (526 MB), and Hugging Face now charges for Docker Spaces.
 *
 *   It also makes the app work with no signal, which is the actual deployment
 *   context: a farmer standing in a field.
 *
 * IT RETURNS THE SAME VERDICT SHAPE AS THE SERVER
 *   { status: 'ok' | 'not_a_leaf' | 'uncertain', ... }
 *   Refusals carry no disease name and no treatment text, exactly as
 *   potato_infer.py does. The UI already renders all three (index.html handles
 *   not_a_leaf / uncertain / model_unavailable), so nothing downstream changes.
 *
 * PREPROCESSING PARITY IS THE WHOLE BALLGAME
 *   The head was fitted to L2-normalised pooler_output from a specific transform:
 *   resize shortest edge 256 (BICUBIC) -> center crop 224 -> ImageNet mean/std.
 *   transformers.js reads exactly that from the model's preprocessor_config.json,
 *   and potato_infer.py / train_potato.py are pinned to match. If any of those
 *   three drift apart, features shift, accuracy silently drops, and nothing
 *   errors. verifyParity() below exists to catch that.
 *
 * USAGE
 *   const clf = await PotatoBrowser.load({ modelDir: '/model' });
 *   const verdict = await clf.predict(fileOrBlobOrUrlOrCanvas);
 */
(function (global) {
  'use strict';

  const DEFAULTS = {
    modelDir: '/model',
    // Pinned. transformers.js majors have changed the pipeline API before, and a
    // floating version would silently change preprocessing under us.
    transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5',
    // Sits next to this file. A literal, not import.meta.url: this file is also
    // loaded as a classic <script>, where import.meta is a SyntaxError.
    resizeUrl: '/js/pil-resize.js',
    // MEASURED against the Python reference, with preprocessing bit-exact, on 7
    // real dataset images (mean cosine distance of the 768-d feature vector;
    // parity needs < 1e-3):
    //
    //     fp16    173 MB   3.15e-4   <- default: parity, half the size of fp32
    //     fp32    347 MB   3.17e-4      identical within noise, twice the bytes
    //     q4f16    50 MB   4.12e-2      drifted; cheap, but off-distribution
    //     q8       87 MB   8.78e-1      BROKEN. Do not ship. See below.
    //
    // q8 does not degrade gracefully, it collapses — 0.878 mean cosine distance
    // means the features are essentially unrelated to what the head was fitted
    // to, while the model still returns confident-looking probabilities. It is
    // the obvious size/speed compromise and it is the one that silently destroys
    // the predictions, so it is named here explicitly.
    dtype: 'fp16',
    device: null,        // null = auto (webgpu if available, else wasm)
    onProgress: null,
  };

  function readF32(buffer, layout, name) {
    const e = layout.find((x) => x.name === name);
    if (!e) throw new Error(`missing tensor '${name}' in model blob`);
    return new Float32Array(buffer, e.offset * 4, e.count);
  }

  /** y = W x + b, with W stored row-major as (out, in). */
  function linear(x, W, b, outDim, inDim) {
    const y = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let s = b ? b[o] : 0;
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) s += W[base + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  function gelu(v) {
    // tanh approximation, which is what PyTorch's default nn.GELU uses in practice
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      v[i] = 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    }
    return v;
  }

  function softmax(z, T) {
    const t = T || 1;
    let m = -Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] / t > m) m = z[i] / t;
    let sum = 0;
    const out = new Float32Array(z.length);
    for (let i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] / t - m); sum += out[i]; }
    for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
  }

  function l2normalise(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  /**
   * Mahalanobis distance to the nearest class centroid.
   * The export ships the packed UPPER triangle of Lt where P = L Lᵀ, so
   * dist = ‖Lt·(x−μ)‖² — identical arithmetic to the server, half the bytes.
   */
  function mahalanobis(x, means, LtUpper, dim, nClasses, diagOnly) {
    let best = Infinity;
    const diff = new Float32Array(dim);
    for (let c = 0; c < nClasses; c++) {
      const off = c * dim;
      for (let i = 0; i < dim; i++) diff[i] = x[i] - means[off + i];
      let d = 0;
      if (diagOnly) {
        for (let i = 0; i < dim; i++) d += diff[i] * diff[i] * LtUpper[i];
      } else {
        // Lt is upper triangular: row r has entries for columns r..dim-1.
        let p = 0;
        for (let r = 0; r < dim; r++) {
          let s = 0;
          for (let col = r; col < dim; col++) s += LtUpper[p++] * diff[col];
          d += s * s;
        }
      }
      if (d < best) best = d;
    }
    return best;
  }

  class PotatoBrowser {
    constructor(cfg, manifest, model, head, ood, tjs, px) {
      this.tjs = tjs;
      this.model = model;
      this.px = px;                       // resolved pil-resize module
      // What to build and in what order. An older single-view manifest has no
      // `views`, so default to the global CLS token it was exported with.
      this.views = (manifest.backbone && manifest.backbone.views)
          || ['global_cls'];
      const pp = manifest.preprocessing || {};
      this.pp = { shortestEdge: pp.resize_shortest_edge || 256,
                  crop: pp.center_crop || 224,
                  mean: pp.mean || [0.485, 0.456, 0.406],
                  std: pp.std || [0.229, 0.224, 0.225] };
      this.cfg = cfg;
      this.manifest = manifest;
      this.head = head;
      this.ood = ood;
      this.classes = manifest.classes;
      this.dim = manifest.feat_dim;
      this.hidden = manifest.hidden || 0;
      this.oodThreshold = (manifest.thresholds && manifest.thresholds.ood_mahalanobis_p99) ?? Infinity;
      this.confidenceFloor = (manifest.thresholds && manifest.thresholds.abstain_below_confidence) ?? 0;
    }

    /** One backbone pass. Returns { cls, patchMean }, each L2-normalised. */
    async _pass(pre) {
      const pixel_values = new this.tjs.Tensor('float32', pre.data, pre.dims);
      const out = await this.model({ pixel_values });
      const t = out.pooler_output || out.last_hidden_state;
      if (!t) throw new Error('Backbone returned neither pooler_output nor last_hidden_state.');
      const d = t.dims[t.dims.length - 1];
      const seq = t.dims.length === 3 ? t.dims[1] : 1;

      // CLS is index 0 of last_hidden_state. For Dinov2Model that tensor is
      // already post-layernorm, so this equals the pooler_output training read.
      const cls = new Float32Array(t.data.slice(0, d));

      let patchMean = null;
      if (seq > 1) {
        patchMean = new Float32Array(d);
        for (let tok = 1; tok < seq; tok++) {
          const off = tok * d;
          for (let i = 0; i < d; i++) patchMean[i] += t.data[off + i];
        }
        for (let i = 0; i < d; i++) patchMean[i] /= (seq - 1);
      }
      return { cls, patchMean, dim: d };
    }

    /**
     * Build the feature vector the head was fitted to.
     *
     * PREPROCESSING PARITY IS THE WHOLE BALLGAME
     *   The head is a linear map over a specific concatenation of specific views
     *   of specific pixels. Every part of that has to match training exactly, and
     *   nothing errors when it does not -- accuracy just quietly drops.
     *
     *   We do NOT use transformers.js's image pipeline. Its in-browser resize goes
     *   through canvas drawImage, which does not antialias a downscale; on this
     *   dataset (1500px photos) that shifted the feature vector by 5.4e-2 to
     *   1.4e-1 cosine distance. pil-resize.js reproduces Pillow bit-for-bit
     *   instead (0 of 196,608 bytes differing on the probe set).
     *
     * VIEWS AND ORDER
     *   manifest.backbone.views lists what to build, in the order the head's
     *   columns expect. A permuted vector would not error; it would score
     *   confidently wrong. Each view is L2-normalised SEPARATELY before
     *   concatenation, matching extract_views.py.
     *
     *   Measured on grouped 5-fold CV: global CLS alone 0.8622 accuracy;
     *   global + the two 2x2 tile views 0.8816. Tiling costs 4 extra passes and
     *   beats adding a second 609 MB backbone.
     *
     * Ruled out by measurement rather than argument: dtype (fp32 and q4f16 agreed
     * to four significant figures) and pooling choice (mean-only gave 0.45-0.55
     * cosine distance, so CLS was never the problem).
     */
    async features(input) {
      const px = this.px;
      const { rgb, width, height } = await px.toRGB(input);

      const needGlobal = this.views.some((v) => v.startsWith('global'));
      const needTiles = this.views.some((v) => v.startsWith('tile'));

      let g = null;
      if (needGlobal) {
        g = await this._pass(px.preprocess(rgb, width, height, this.pp));
      }

      let tileClsMean = null;
      let tilePatchMean = null;
      if (needTiles) {
        const grid = (this.manifest.backbone.tiling
          && this.manifest.backbone.tiling.grid) || 2;
        const { tiles } = px.preprocessTiles(rgb, width, height,
          Object.assign({}, this.pp, { grid }));
        let accCls = null;
        let accPatch = null;
        for (const tile of tiles) {
          const r = await this._pass(tile);
          if (!accCls) {
            accCls = new Float32Array(r.dim);
            accPatch = r.patchMean ? new Float32Array(r.dim) : null;
          }
          for (let i = 0; i < r.dim; i++) accCls[i] += r.cls[i];
          if (accPatch && r.patchMean) {
            for (let i = 0; i < r.dim; i++) accPatch[i] += r.patchMean[i];
          }
        }
        const n = tiles.length;
        for (let i = 0; i < accCls.length; i++) accCls[i] /= n;
        tileClsMean = accCls;
        if (accPatch) {
          for (let i = 0; i < accPatch.length; i++) accPatch[i] /= n;
          tilePatchMean = accPatch;
        }
      }

      const blocks = [];
      for (const v of this.views) {
        if (v === 'global_cls') blocks.push(l2normalise(g.cls));
        else if (v === 'global_mean') blocks.push(l2normalise(g.patchMean));
        else if (v === 'tile2x2_mean_cls') blocks.push(l2normalise(tileClsMean));
        else if (v === 'tile2x2_mean_patch') blocks.push(l2normalise(tilePatchMean));
        else throw new Error(`Unknown view "${v}" in the manifest.`);
      }

      let vec;
      if (blocks.length === 1) {
        vec = blocks[0];
      } else {
        const total = blocks.reduce((n, b) => n + b.length, 0);
        vec = new Float32Array(total);
        let o = 0;
        for (const b of blocks) { vec.set(b, o); o += b.length; }
      }

      if (vec.length !== this.dim) {
        throw new Error(
          `feature dim ${vec.length} != head dim ${this.dim}. The manifest views ` +
          `[${this.views.join(', ')}] do not reproduce what the head was fitted to.`);
      }
      return vec;
    }

    async predict(input) {
      const t0 = performance.now();
      const f = await this.features(input);

      // 1. Is this even a potato leaf? Refuse before naming a disease.
      let oodScore = null;
      if (this.ood) {
        oodScore = mahalanobis(f, this.ood.means, this.ood.Lt, this.dim,
                               this.classes.length, this.ood.diagOnly);
        if (oodScore > this.oodThreshold) {
          return {
            status: 'not_a_leaf',
            // NOT "filling the frame" -- see the note in potato_infer.py. Measured
            // on the parity set with aspect ratio held constant: cropping 10% into
            // the leaf takes the mean Mahalanobis score from 2019 to 3976 against a
            // 2713 threshold and rejects 6 of 7 images. The old wording told the
            // farmer to do the exact thing that triggers this refusal.
            message: 'This does not look like a potato leaf, so no diagnosis was '
                   + 'produced.\n\n'
                   + 'Two things cause this most often:\n'
                   + '- Part of the leaf is outside the photo. Step back so the WHOLE '
                   + 'leaf is in view with a little space around it. Do not fill the '
                   + 'frame.\n'
                   + '- It is a close-up of one spot. This model reads whole leaves '
                   + 'and cannot judge a zoomed-in patch.\n\n'
                   + 'Take the photo straight down over a single leaf, in daylight, '
                   + 'with the whole leaf visible.',
            ood_score: oodScore, ood_threshold: this.oodThreshold,
            ms: Math.round(performance.now() - t0), source: 'browser',
          };
        }
      }

      // 2. Average the temperature-scaled folds, exactly as the server does.
      const K = this.classes.length;
      const probs = new Float32Array(K);
      for (const fold of this.head.folds) {
        let z;
        if (this.hidden) {
          const h = gelu(linear(f, fold.w0, fold.b0, this.hidden, this.dim));
          z = linear(h, fold.w1, fold.b1, K, this.hidden);
        } else {
          z = linear(f, fold.w, fold.b, K, this.dim);
        }
        const p = softmax(z, fold.temperature);
        for (let i = 0; i < K; i++) probs[i] += p[i] / this.head.folds.length;
      }

      const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
      const top = order[0];
      const confidence = probs[top];
      const all = {};
      this.classes.forEach((c, i) => { all[c] = probs[i]; });

      // 3. Confident enough to name a disease?
      if (confidence < this.confidenceFloor) {
        return {
          status: 'uncertain',
          // Same correction as the not_a_leaf message above.
          message: 'Not confident enough to give a diagnosis. Retake the photo in '
                 + 'even daylight with the WHOLE leaf in view and a little space '
                 + 'around it -- do not fill the frame or zoom in. If it is still '
                 + 'uncertain, ask an agronomist.',
          confidence, confidence_floor: this.confidenceFloor,
          all_probabilities: all, ood_score: oodScore,
          ms: Math.round(performance.now() - t0), source: 'browser',
        };
      }

      return {
        status: 'ok',
        disease: this.classes[top],
        confidence,
        runner_up: this.classes[order[1]],
        runner_up_confidence: probs[order[1]],
        all_probabilities: all,
        ood_score: oodScore,
        calibrated: true,
        n_folds: this.head.folds.length,
        ms: Math.round(performance.now() - t0),
        source: 'browser',
      };
    }

    /**
     * Compare browser features against features computed by Python for the same
     * image. Feed it the output of a small Python dump; anything above ~1e-3
     * cosine distance means preprocessing has drifted and the accuracy numbers no
     * longer apply.
     */
    async verifyParity(input, pythonFeatures) {
      const mine = await this.features(input);
      const ref = Float32Array.from(pythonFeatures);
      if (ref.length !== mine.length) return { ok: false, reason: 'dim mismatch' };
      let dot = 0, maxAbs = 0;
      for (let i = 0; i < mine.length; i++) {
        dot += mine[i] * ref[i];
        maxAbs = Math.max(maxAbs, Math.abs(mine[i] - ref[i]));
      }
      const cosDist = 1 - dot;
      return {
        ok: cosDist < 1e-3,
        cosine_distance: cosDist,
        max_abs_diff: maxAbs,
        note: cosDist < 1e-3
          ? 'parity good — server metrics apply to the browser'
          : 'PARITY BROKEN: preprocessing differs. Check resize (256), ' +
            'interpolation (bicubic), crop (224), mean/std, and which token is pooled.',
      };
    }

    get info() {
      return {
        classes: this.classes, dim: this.dim, folds: this.head.folds.length,
        backbone: this.manifest.backbone, oodEnabled: Boolean(this.ood),
        serverMetrics: this.manifest.server_metrics,
      };
    }
  }

  async function load(options) {
    const cfg = Object.assign({}, DEFAULTS, options || {});
    const say = (m, pct) => cfg.onProgress && cfg.onProgress(m, pct);

    say('Loading runtime…', 2);
    // Resolve against the document base before importing. A bare specifier like
    // 'js/pil-resize.js' is NOT a relative path to dynamic import() — it is a
    // module name, and it fails with "Failed to resolve module specifier". This
    // accepts 'js/x.js', './js/x.js' and '/js/x.js' alike.
    const resizeHref = new URL(cfg.resizeUrl, document.baseURI).href;
    // Start both imports together; pil-resize is local and tiny. It is AWAITED
    // here rather than lazily inside features(): if it cannot load, load() must
    // fail so the caller falls back to the server, instead of returning a
    // ready-looking classifier that throws on the first photo.
    const pxPromise = import(/* webpackIgnore: true */ resizeHref);
    const tjs = await import(/* webpackIgnore: true */ cfg.transformersUrl);
    const px = await pxPromise;
    if (typeof px.preprocess !== 'function' || typeof px.toRGB !== 'function') {
      throw new Error(`${resizeHref} did not export preprocess/toRGB.`);
    }
    const { pipeline, env } = tjs;

    // Fetch the ONNX weights from the Hub CDN by default. If a strict CSP blocks
    // that (Cloudflare Pages with a tight policy will), copy the model files next
    // to the app and set env.localModelPath instead.
    if (cfg.localModelPath) {
      env.allowRemoteModels = false;
      env.localModelPath = cfg.localModelPath;
    }

    say('Loading model manifest…', 5);
    const manifest = await (await fetch(`${cfg.modelDir}/manifest.json`)).json();

    say('Loading classifier head…', 8);
    const headBuf = await (await fetch(`${cfg.modelDir}/head.bin`)).arrayBuffer();
    const L = manifest.head.layout;
    const folds = manifest.folds.map((fm, i) => {
      const t = fm.temperature;
      return manifest.hidden
        ? { temperature: t,
            w0: readF32(headBuf, L, `f${i}_w0`), b0: readF32(headBuf, L, `f${i}_b0`),
            w1: readF32(headBuf, L, `f${i}_w1`), b1: readF32(headBuf, L, `f${i}_b1`) }
        : { temperature: t,
            w: readF32(headBuf, L, `f${i}_w`), b: readF32(headBuf, L, `f${i}_b`) };
    });

    let ood = null;
    if (manifest.ood) {
      say('Loading out-of-distribution gate…', 12);
      const oodBuf = await (await fetch(`${cfg.modelDir}/${manifest.ood.file}`)).arrayBuffer();
      const OL = manifest.ood.layout;
      const diagOnly = manifest.ood.mode === 'diagonal';
      ood = {
        means: readF32(oodBuf, OL, 'means'),
        Lt: readF32(oodBuf, OL, diagOnly ? 'prec_diag' : 'Lt_upper'),
        diagOnly,
      };
    } else {
      console.warn('[potato] No OOD gate in the manifest — non-leaf images will ' +
                   'be classified into one of the disease classes.');
    }

    const mb = manifest.backbone.approx_download_mb;
    say(`Downloading backbone (~${mb ? Math.round(mb) : '?'} MB, cached after this)…`, 15);
    // AutoModel, not pipeline(): we feed pixel_values we prepared ourselves so
    // the processor's canvas resize never runs. See features() for the numbers.
    const model = await tjs.AutoModel.from_pretrained(manifest.backbone.onnx_model_id, {
      dtype: cfg.dtype || manifest.backbone.dtype,
      device: cfg.device || undefined,
      progress_callback: (p) => {
        if (p && p.status === 'progress' && p.progress != null) {
          say(`Downloading backbone… ${Math.round(p.progress)}%`, 15 + 0.8 * p.progress);
        }
      },
    });

    say('Ready', 100);
    return new PotatoBrowser(cfg, manifest, model, { folds }, ood, tjs, px);
  }

  global.PotatoBrowser = { load, VERSION: 2 };
})(typeof window !== 'undefined' ? window : globalThis);
