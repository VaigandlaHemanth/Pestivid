/**
 * Browser-vs-server FEATURE parity, headless.
 *
 * Why this exists rather than public/parity/tiled.html.
 *
 * tiled.html is the same comparison, but it only runs in a real browser. That
 * turned out to be a hard blocker: in a hidden/non-compositing pane the renderer
 * throttles WASM to a standstill, so the check could not be run on demand and
 * certainly not in CI. A verification step that requires a human to keep a
 * browser window visible is a verification step that stops happening.
 *
 * This runs the SAME ONNX graph and the SAME weights through onnxruntime-node,
 * feeds it tensors from the SAME pil-resize.js the browser uses, and assembles the
 * multi-view vector with the SAME rules as potato-browser.js. What it does not
 * cover is the onnxruntime-WEB kernels specifically -- that is the one thing only
 * a browser can test. Everything else that has actually broken here (the
 * normalisation ORDER, the crop offset, the feature recipe, stale cached features)
 * is covered, and those are the failures that silently changed predictions.
 *
 * The reference is Pestivid/dump_parity_tiled.py's output: per-VIEW blocks from
 * potato_infer.PotatoClassifier._features, i.e. the real server path. Per-view
 * matters -- a wrong 768-d block hides inside the cosine of the concatenated
 * 2304-d vector, which is how the normalisation-order bug survived for a while.
 *
 *     node test_feature_parity.mjs [path/to/parity_tiled.json]
 */

import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP = resolve(HERE, '..');
const PARITY_DIR = join(APP, 'public', 'parity');

const refPath = process.argv[2] || join(PARITY_DIR, 'parity_tiled.json');

if (!existsSync(refPath)) {
    console.log(`  SKIP  no reference at ${refPath}`);
    console.log('        Generate it with: python Pestivid/dump_parity_tiled.py <dataset> '
              + 'public/parity');
    process.exit(0);
}

let tjs;
try {
    tjs = await import('@huggingface/transformers');
} catch (e) {
    console.log(`  SKIP  @huggingface/transformers is not installed (${e.code || e.message})`);
    console.log('        npm install @huggingface/transformers@3.7.5');
    process.exit(0);
}

// pathToFileURL, not a raw path: on Windows the ESM loader parses "C:" as a URL
// scheme and refuses an absolute path outright.
const { preprocess, preprocessTiles } = await import(
    pathToFileURL(join(APP, 'public', 'js', 'pil-resize.js')).href);

const ref = JSON.parse(readFileSync(refPath, 'utf8'));
const manifest = JSON.parse(readFileSync(join(APP, 'public', 'model', 'manifest.json'), 'utf8'));

// The recipe is read from the manifest, not hardcoded, so a recipe change cannot
// pass by comparing against the wrong thing.
const views = manifest.backbone.views;
const VIEW_DIM = manifest.backbone.view_dim;
if (views.join(',') !== ref.recipe.join(',')) {
    console.log(`  FAIL  manifest recipe [${views}] != reference recipe [${ref.recipe}]`);
    process.exit(1);
}

console.log(`  recipe: ${views.join(' + ')}  (${views.length * VIEW_DIM}-d)`);
console.log(`  model:  ${manifest.backbone.onnx_model_id} @ ${manifest.backbone.dtype}`);

tjs.env.allowLocalModels = false;

const model = await tjs.AutoModel.from_pretrained(manifest.backbone.onnx_model_id, {
    dtype: manifest.backbone.dtype,
});
console.log('  backbone loaded\n');

// The OOD statistics, so this test can compare the quantity that actually broke.
//
// Per-view cosine distance alone is NOT sufficient, and that was measured rather
// than assumed: reintroducing the normalisation-order bug (L2 each tile, then
// average, instead of average then one L2) moved the worst per-view cosine only
// to 6.4e-4 -- inside any tolerance loose enough for fp16 -- while the same bug in
// production pushed the Mahalanobis score past the p99 threshold and made the gate
// answer "not a potato leaf" for genuine potato leaves. The decision flipped; the
// cosine barely moved. So the decision is what gets checked.
function loadOod() {
    const mpath = join(APP, 'public', 'model', 'ood.bin');
    if (!existsSync(mpath) || !manifest.ood) return null;
    const buf = readFileSync(mpath);
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const byName = {};
    for (const seg of manifest.ood.layout) {
        byName[seg.name] = f32.subarray(seg.offset, seg.offset + seg.count);
    }
    return {
        means: byName.means,
        Lt: byName.Lt_upper,
        diagOnly: manifest.ood.mode !== 'cholesky_upper',
    };
}

// Byte-for-byte the same algorithm as potato-browser.js's mahalanobis().
function mahalanobis(x, ood, dim, nClasses) {
    let best = Infinity;
    const diff = new Float32Array(dim);
    for (let c = 0; c < nClasses; c++) {
        const off = c * dim;
        for (let i = 0; i < dim; i++) diff[i] = x[i] - ood.means[off + i];
        let d = 0;
        if (ood.diagOnly) {
            for (let i = 0; i < dim; i++) d += diff[i] * diff[i] * ood.Lt[i];
        } else {
            let pp = 0;
            for (let r = 0; r < dim; r++) {
                let sum = 0;
                for (let col = r; col < dim; col++) sum += ood.Lt[pp++] * diff[col];
                d += sum * sum;
            }
        }
        if (d < best) best = d;
    }
    return best;
}

const ood = loadOod();
const N_CLASSES = (manifest.classes || []).length || 7;
const OOD_THRESHOLD = (manifest.thresholds && manifest.thresholds.ood_mahalanobis_p99)
    ?? Infinity;
console.log(ood
    ? `  ood:    loaded, p99 threshold ${OOD_THRESHOLD.toFixed(1)}`
    : '  ood:    NOT loaded -- the decision check is disabled');

const l2 = (v) => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    const n = Math.sqrt(s) || 1;
    const o = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) o[i] = v[i] / n;
    return o;
};

/** RAW cls and patch-mean for a batch. Deliberately NOT normalised here. */
async function passRaw(data, dims) {
    const pixel_values = new tjs.Tensor('float32', data, dims);
    const out = await model({ pixel_values });
    const h = out.last_hidden_state;
    const [b, tokens, dim] = h.dims;
    const flat = h.data;
    const cls = [];
    const patch = [];
    for (let i = 0; i < b; i++) {
        const base = i * tokens * dim;
        const c = new Float64Array(dim);
        for (let d = 0; d < dim; d++) c[d] = flat[base + d];
        cls.push(c);
        const pm = new Float64Array(dim);
        for (let tk = 1; tk < tokens; tk++) {
            const o = base + tk * dim;
            for (let d = 0; d < dim; d++) pm[d] += flat[o + d];
        }
        for (let d = 0; d < dim; d++) pm[d] /= (tokens - 1);
        patch.push(pm);
    }
    return { cls, patch };
}

/**
 * Decode a JPEG/PNG to raw RGB without a browser.
 * transformers.js ships RawImage, which uses sharp in Node.
 */
async function decodeRGB(file) {
    const img = await tjs.RawImage.read(file);
    const rgb = img.channels === 3 ? img : await img.rgb();
    return { data: rgb.data, width: rgb.width, height: rgb.height };
}

let worst = 0;
let worstWhere = null;
let oodWorst = 0;
let oodWorstWhere = null;
let failures = 0;

const header = `${'image'.padEnd(30)} `
    + views.map((v) => v.replace('tile2x2_', 't_').padEnd(11)).join(' ')
    + ' ood mine/server';
console.log(header);

for (const rec of ref.records) {
    const file = join(PARITY_DIR, rec.file);
    if (!existsSync(file)) {
        console.log(`${rec.file.padEnd(30)} MISSING`);
        failures++;
        continue;
    }

    const { data: rgb, width, height } = await decodeRGB(file);

    // ── global view ─────────────────────────────────────────────────────────
    const g = preprocess(rgb, width, height, { shortestEdge: 256, crop: 224 });
    const gp = await passRaw(g.data, g.dims);
    const globalCls = l2(gp.cls[0]);

    // ── 2x2 tiles: RAW mean across tiles, then ONE L2 ───────────────────────
    // The order is load-bearing. Normalising each tile first and then averaging
    // yields a different vector, and getting it wrong made the OOD gate reject
    // genuine training images as "not a potato leaf".
    // preprocessTiles returns { tiles: [{data, dims}, ...] } -- four separate
    // [1,3,224,224] tensors, not one batched [4,3,224,224]. potato-browser.js
    // passes them one at a time, so this does too: batching them here would be a
    // different computation from the one the browser actually performs.
    const { tiles } = preprocessTiles(rgb, width, height, {});
    const mCls = new Float64Array(VIEW_DIM);
    const mPatch = new Float64Array(VIEW_DIM);
    for (const tile of tiles) {
        const r = await passRaw(tile.data, tile.dims);
        for (let d = 0; d < VIEW_DIM; d++) {
            mCls[d] += r.cls[0][d];
            mPatch[d] += r.patch[0][d];
        }
    }
    const nT = tiles.length;
    for (let d = 0; d < VIEW_DIM; d++) { mCls[d] /= nT; mPatch[d] /= nT; }

    const mine = {
        global_cls: globalCls,
        tile2x2_mean_cls: l2(mCls),
        tile2x2_mean_patch: l2(mPatch),
    };

    // The concatenated vector, in recipe order, exactly as the head sees it.
    const full = new Float32Array(views.length * VIEW_DIM);
    views.forEach((v, i) => full.set(mine[v], i * VIEW_DIM));

    let oodCell = '   -   ';
    if (ood && rec.ood_score != null) {
        const score = mahalanobis(full, ood, full.length, N_CLASSES);
        // Relative, because the score is O(1000) and fp16 moves it by a few units.
        const rel = Math.abs(score - rec.ood_score) / Math.max(1, rec.ood_score);
        if (rel > oodWorst) { oodWorst = rel; oodWorstWhere = rec.file; }

        // The verdict must agree, not just the number. A score that crosses the
        // threshold changes the answer from a diagnosis to a refusal.
        const mineRejects = score > OOD_THRESHOLD;
        const refRejects = rec.status === 'not_a_leaf';
        if (mineRejects !== refRejects) {
            console.log(`
  VERDICT MISMATCH on ${rec.file}: browser `
                + `${mineRejects ? 'rejects' : 'accepts'} (${score.toFixed(0)}), server `
                + `${refRejects ? 'rejects' : 'accepts'} (${rec.ood_score.toFixed(0)}), `
                + `threshold ${OOD_THRESHOLD.toFixed(0)}`);
            failures++;
        }
        oodCell = `${score.toFixed(0)}/${rec.ood_score.toFixed(0)}`;
    }

    const cells = [];
    for (const v of views) {
        const a = mine[v];
        const b = rec.views[v];
        if (!a || !b) {
            cells.push('NO DATA'.padEnd(11));
            failures++;
            continue;
        }
        let dot = 0; let na = 0; let nb = 0;
        for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        const dist = 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
        if (dist > worst) { worst = dist; worstWhere = `${rec.file}/${v}`; }
        cells.push(dist.toExponential(2).padEnd(11));
    }
    console.log(`${rec.file.padEnd(30)} ${cells.join(' ')} ${oodCell}`);
}

// 1e-3 per view. Above that the server-measured accuracy no longer describes what
// this code computes. fp16 inference against an fp32 reference sits around 1e-4.
// Per-view cosine tolerance. fp16 ONNX against fp32 PyTorch measures ~3.9e-4 at
// worst on this set, so 1e-3 leaves headroom without being meaningless.
const TOL = 1e-3;
// The OOD score is the check with teeth, and this threshold is set from two
// measurements rather than picked:
//
//   correct code, fp16 vs fp32 : worst drift 2.35%  (parity_Bacteria)
//   normalisation-order bug    : worst drift 7.95%  (parity_Virus)
//
// 4% sits cleanly between them. Note that with the bug present every VERDICT on
// this set still agreed -- all 8 fixtures sit well below the 2712 threshold, so
// nothing crossed it. In production the bug hit an image near the boundary and
// flipped it to "not a potato leaf". That is why the score DRIFT is the
// load-bearing assertion here and the verdict check is the backstop, not the
// other way round.
const OOD_TOL = 0.04;

console.log();
const cosOk = worst < TOL;
const oodOk = oodWorst < OOD_TOL;

if (failures === 0 && cosOk && oodOk) {
    console.log(`PASS  worst per-view cosine ${worst.toExponential(3)} < ${TOL}`);
    if (ood) {
        console.log(`      worst OOD score drift ${(oodWorst * 100).toFixed(2)}% < `
                  + `${OOD_TOL * 100}%, every verdict agrees`);
    }
    console.log(`      (${ref.records.length} images x ${views.length} views, `
              + `fp16 ONNX vs fp32 PyTorch)`);
    process.exit(0);
}

if (!cosOk) {
    console.log(`FAIL  worst per-view cosine ${worst.toExponential(3)} on ${worstWhere}`);
}
if (!oodOk) {
    console.log(`FAIL  OOD score drifted ${(oodWorst * 100).toFixed(2)}% on ${oodWorstWhere}`);
    console.log('      The features are close in cosine terms but land in a different');
    console.log('      place relative to the fitted distribution, which is what changes');
    console.log('      the verdict. Check the normalisation ORDER and the transform.');
}
if (failures) console.log(`FAIL  ${failures} verdict mismatch(es) or unusable record(s)`);
process.exit(1);
