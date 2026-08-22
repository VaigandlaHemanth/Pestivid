/**
 * Tests for video perceptual fingerprinting.
 *
 * The point of this module is to catch a re-encoded copy of an existing video,
 * so the tests generate real videos with ffmpeg and then attack them the way a
 * fraudster would: re-encode at a different bitrate, rescale, change brightness,
 * and trim. A fingerprint that does not survive those is useless, and a
 * fingerprint that matches UNRELATED videos is worse than useless because it
 * would block honest farmers.
 *
 *     node test_fingerprint.js
 */

const assert = require('assert');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const fp = require('./services/videoFingerprint');

let pass = 0;
let fail = 0;

async function t(name, fn) {
    try {
        await fn();
        console.log(`  PASS  ${name}`);
        pass++;
    } catch (e) {
        console.log(`  FAIL  ${name}\n          ${e.message}`);
        fail++;
    }
}

const tmp = path.join(os.tmpdir(), `pv_fp_${Date.now()}`);
fs.mkdirSync(tmp, { recursive: true });
const P = (n) => path.join(tmp, n);

function ff(args) {
    execFileSync(fp.ffmpegPath, ['-v', 'error', '-y', ...args], { stdio: 'pipe' });
}

(async () => {
    // Two visually distinct 6-second clips. testsrc2 and smptebars are structured
    // patterns, so they exercise the gradient hash far better than flat colour.
    ff(['-f', 'lavfi', '-i', 'testsrc2=size=640x480:rate=30:duration=6', '-pix_fmt', 'yuv420p', P('a.mp4')]);
    ff(['-f', 'lavfi', '-i', 'smptebars=size=640x480:rate=30:duration=6', '-pix_fmt', 'yuv420p', P('b.mp4')]);

    // The attacks: a fraudster re-encodes to change every byte.
    ff(['-i', P('a.mp4'), '-b:v', '200k', '-pix_fmt', 'yuv420p', P('a_reencoded.mp4')]);
    ff(['-i', P('a.mp4'), '-vf', 'scale=320:240', '-pix_fmt', 'yuv420p', P('a_small.mp4')]);
    ff(['-i', P('a.mp4'), '-vf', 'eq=brightness=0.12', '-pix_fmt', 'yuv420p', P('a_bright.mp4')]);
    ff(['-i', P('a.mp4'), '-ss', '1', '-t', '4', '-pix_fmt', 'yuv420p', P('a_trimmed.mp4')]);

    const F = {};
    for (const n of ['a', 'b', 'a_reencoded', 'a_small', 'a_bright', 'a_trimmed']) {
        F[n] = await fp.fingerprint(P(`${n}.mp4`));
    }

    await t('fingerprint returns frame hashes of the right shape', async () => {
        assert.ok(F.a.nFrames >= 6, `expected >=6 frames, got ${F.a.nFrames}`);
        for (const h of F.a.frameHashes) {
            assert.strictEqual(h.length, 16, 'a 64-bit dHash is 16 hex chars');
            assert.ok(/^[0-9a-f]{16}$/.test(h), `not hex: ${h}`);
        }
    });

    await t('duration is probed', async () => {
        assert.ok(F.a.durationSeconds > 4 && F.a.durationSeconds < 8,
            `duration looked wrong: ${F.a.durationSeconds}`);
    });

    await t('fingerprinting is deterministic', async () => {
        const again = await fp.fingerprint(P('a.mp4'));
        assert.deepStrictEqual(again.frameHashes, F.a.frameHashes);
    });

    // ── the attacks it must survive ───────────────────────────────────────────
    const attacks = [
        ['re-encoded at a lower bitrate', 'a_reencoded'],
        ['rescaled to half resolution', 'a_small'],
        ['brightness raised', 'a_bright'],
        ['trimmed by 1s at the start and 1s at the end', 'a_trimmed'],
    ];
    for (const [label, key] of attacks) {
        await t(`DETECTS a copy ${label}`, async () => {
            const c = fp.compare(F.a, F[key]);
            assert.ok(c.matchedFraction >= 0.5,
                `only ${(c.matchedFraction * 100).toFixed(0)}% of frames matched ` +
                `(best distance ${c.bestDistance}) — a re-encode would slip through`);
        });
    }

    // ── it must NOT flag unrelated content ───────────────────────────────────
    await t('does NOT match two unrelated videos (false-positive guard)', async () => {
        const c = fp.compare(F.a, F.b);
        assert.ok(c.matchedFraction < 0.5,
            `unrelated videos matched at ${(c.matchedFraction * 100).toFixed(0)}% ` +
            `— this would block honest farmers`);
    });

    await t('separation between a copy and an unrelated video is large', async () => {
        const same = fp.compare(F.a, F.a_reencoded).matchedFraction;
        const diff = fp.compare(F.a, F.b).matchedFraction;
        assert.ok(same - diff >= 0.4,
            `separation too small: copy ${same.toFixed(2)} vs unrelated ${diff.toFixed(2)}`);
        console.log(`          copy ${same.toFixed(2)} vs unrelated ${diff.toFixed(2)}`);
    });

    // ── hamming sanity ───────────────────────────────────────────────────────
    await t('hamming: identical is 0, inverted is 64', async () => {
        assert.strictEqual(fp.hamming('0000000000000000', '0000000000000000'), 0);
        assert.strictEqual(fp.hamming('ffffffffffffffff', '0000000000000000'), 64);
        assert.strictEqual(fp.hamming('ffffffffffffffff', 'fffffffffffffffe'), 1);
    });

    await t('hamming refuses to compare mismatched lengths', async () => {
        assert.strictEqual(fp.hamming('abc', 'abcdef0123456789'), Infinity);
    });

    await t('compare() on an empty fingerprint is not comparable, not a match', async () => {
        const c = fp.compare({ frameHashes: [] }, F.a);
        assert.strictEqual(c.comparable, false);
        assert.strictEqual(c.matchedFraction, 0);
    });

    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch((e) => {
    console.error('harness error:', e.message);
    process.exit(1);
});
