/**
 * Compare the browser's preprocessing against torchvision's, in Node.
 *
 * Why this exists alongside parity/tiled.html.
 *
 * tiled.html compares FEATURES, which needs five ONNX passes per image (~10s
 * each) and only runs in a browser. That makes it slow, and it makes a
 * preprocessing bug hard to localise: a wrong crop offset and a wrong backbone
 * both show up as "the cosine distance is a bit high".
 *
 * This compares the INPUT TENSOR instead -- decode -> Resize -> CenterCrop ->
 * Normalize -- against a reference dumped straight out of torchvision. It is
 * exact, it takes under a second, and it runs in CI without a browser or WASM.
 *
 * It also exists because of a specific miss. Every parity image was 1500x1500, so
 * Resize(256) gave 256x256 and the centre-crop offset was (256-224)/2 = 16 --
 * never a tie, so the rounding RULE was never tested. torchvision's CenterCrop
 * uses Python's round() (ties to even); the browser used Math.round() (ties up).
 * They differ by one pixel whenever (side - crop) is odd, which is what every
 * non-square phone photo produces. The square test set could not see it.
 *
 *     node check_preprocess.mjs <ref-dir>
 *
 * where <ref-dir> is the output of Pestivid/dump_preprocess_ref.py.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { preprocess } from '../js/pil-resize.js';

const refDir = process.argv[2];
if (!refDir) {
    console.error('usage: node check_preprocess.mjs <ref-dir>');
    process.exit(2);
}

const index = JSON.parse(readFileSync(join(refDir, 'index.json'), 'utf8'));

// Float32 tolerance. The transform is deterministic and the resize is bit-exact
// (pil-resize.js reproduces Pillow's bicubic kernel), so anything above float
// noise is a real disagreement, not a rounding artefact.
const TOL = 2e-6;

let worst = 0;
let worstImage = null;
let failures = 0;
let ties = 0;

console.log(`${'image'.padEnd(28)} ${'geometry'.padEnd(20)} ${'max|diff|'.padEnd(11)} tie  result`);

for (const r of index.records) {
    const rgb = new Uint8Array(readFileSync(join(refDir, `${r.stem}.rgb.bin`)));
    const expectBuf = readFileSync(join(refDir, `${r.stem}.ref.bin`));
    const expect = new Float32Array(
        expectBuf.buffer, expectBuf.byteOffset, expectBuf.byteLength / 4);

    const got = preprocess(rgb, r.inW, r.inH, {
        shortestEdge: 256,
        crop: 224,
        mean: index.mean,
        std: index.std,
    });

    if (got.data.length !== expect.length) {
        console.log(`${r.file.padEnd(28)} LENGTH MISMATCH ${got.data.length} vs ${expect.length}`);
        failures++;
        continue;
    }

    let maxDiff = 0;
    for (let i = 0; i < expect.length; i++) {
        const d = Math.abs(got.data[i] - expect[i]);
        if (d > maxDiff) maxDiff = d;
    }
    if (maxDiff > worst) { worst = maxDiff; worstImage = r.file; }
    if (r.tie) ties++;

    // The resized dimensions must match too -- a Resize disagreement would
    // otherwise be masked by a coincidentally similar crop.
    const geomOk = got.resized[0] === r.resizedW && got.resized[1] === r.resizedH;
    const ok = maxDiff <= TOL && geomOk;
    if (!ok) failures++;

    const geom = `${r.inW}x${r.inH}→${r.resizedW}x${r.resizedH}`;
    console.log(`${r.file.padEnd(28)} ${geom.padEnd(20)} `
        + `${maxDiff.toExponential(2).padEnd(11)} ${(r.tie ? 'YES' : ' - ').padEnd(4)} `
        + `${ok ? 'ok' : 'MISMATCH' + (geomOk ? '' : ' (geometry)')}`);
}

console.log();
if (ties === 0) {
    // A green run over squares only would prove nothing about the rounding rule.
    console.log('FAIL: no image in the reference set exercises the rounding tie.');
    console.log('      Add a non-square image (see Pestivid/add_nonsquare_parity.py),');
    console.log('      or this check cannot detect a centre-crop offset bug.');
    process.exit(1);
}

if (failures === 0) {
    console.log(`PASS: ${index.records.length} images match torchvision `
        + `(${ties} exercising the rounding tie), worst |diff| `
        + `${worst.toExponential(2)} on ${worstImage}`);
    process.exit(0);
}

console.log(`FAIL: ${failures} of ${index.records.length} images disagree with `
    + `torchvision; worst |diff| ${worst.toExponential(2)} on ${worstImage}`);
process.exit(1);
