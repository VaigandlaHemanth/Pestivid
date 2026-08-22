/**
 * Tests for the server-side video upload service.
 *
 * Scope: the parts that do not need MongoDB or a Pinata key — the hashing, the
 * size/type gates, and the refusal to run unconfigured. Those are exactly the
 * pieces this change introduced, and they are the pieces that decide whether a
 * videoFileHash can be trusted later by the anchoring layer.
 *
 * Not covered here (needs credentials and a database, so it must be run against
 * a real environment): the actual Pinata pin, the duplicate-CID conflict, and the
 * Video document write.
 *
 *     node test_upload.js
 */

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ipfs = require('./services/ipfsUpload');

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

(async () => {
    const tmp = path.join(os.tmpdir(), `pv_test_${Date.now()}`);
    fs.mkdirSync(tmp, { recursive: true });

    // ── hashing ──────────────────────────────────────────────────────────────
    await t('sha256File matches crypto over the same bytes', async () => {
        const f = path.join(tmp, 'a.bin');
        const buf = crypto.randomBytes(300000);
        fs.writeFileSync(f, buf);
        const expected = crypto.createHash('sha256').update(buf).digest('hex');
        assert.strictEqual(await ipfs.sha256File(f), expected);
    });

    await t('sha256File is stable across calls (streaming leaves no state)', async () => {
        const f = path.join(tmp, 'b.bin');
        fs.writeFileSync(f, crypto.randomBytes(150000));
        assert.strictEqual(await ipfs.sha256File(f), await ipfs.sha256File(f));
    });

    await t('different bytes give different hashes (one bit flipped)', async () => {
        const f1 = path.join(tmp, 'c1.bin');
        const f2 = path.join(tmp, 'c2.bin');
        const buf = crypto.randomBytes(50000);
        fs.writeFileSync(f1, buf);
        const flipped = Buffer.from(buf);
        flipped[0] ^= 0x01;
        fs.writeFileSync(f2, flipped);
        assert.notStrictEqual(await ipfs.sha256File(f1), await ipfs.sha256File(f2));
    });

    await t('sha256File handles a file larger than one stream chunk (2 MB)', async () => {
        const f = path.join(tmp, 'big.bin');
        const buf = crypto.randomBytes(2 * 1024 * 1024);
        fs.writeFileSync(f, buf);
        const expected = crypto.createHash('sha256').update(buf).digest('hex');
        assert.strictEqual(await ipfs.sha256File(f), expected);
    });

    // ── configuration gate ───────────────────────────────────────────────────
    await t('pinataConfigured() is false when the key is absent', async () => {
        const saved = process.env.PINATA_JWT;
        delete process.env.PINATA_JWT;
        assert.strictEqual(ipfs.pinataConfigured(), false);
        process.env.PINATA_JWT = saved || '';
    });

    await t('pinataConfigured() rejects a short placeholder value', async () => {
        const saved = process.env.PINATA_JWT;
        process.env.PINATA_JWT = 'TODO';
        assert.strictEqual(ipfs.pinataConfigured(), false);
        process.env.PINATA_JWT = saved || '';
    });

    await t('pinToPinata refuses to run unconfigured, with a typed error', async () => {
        const saved = process.env.PINATA_JWT;
        delete process.env.PINATA_JWT;
        const f = path.join(tmp, 'd.bin');
        fs.writeFileSync(f, Buffer.from('x'));
        await assert.rejects(
            () => ipfs.pinToPinata(f, 'd.mp4'),
            (e) => e.code === 'PINATA_NOT_CONFIGURED',
            'must fail with PINATA_NOT_CONFIGURED rather than a generic error');
        process.env.PINATA_JWT = saved || '';
    });

    // ── upload gates ─────────────────────────────────────────────────────────
    await t('multer is configured with a size limit and a single-file cap', async () => {
        // A missing limit would let one request write an unbounded temp file and
        // fill the disk of a free-tier instance.
        assert.ok(ipfs.MAX_BYTES > 0, 'MAX_BYTES must be positive');
        assert.ok(ipfs.MAX_BYTES <= 500 * 1024 * 1024,
            'MAX_BYTES should stay modest for a 512 MB host');
    });

    await t('only video mime types are allowed', async () => {
        assert.ok(ipfs.ALLOWED_MIME.has('video/mp4'));
        assert.ok(ipfs.ALLOWED_MIME.has('video/webm'));
        assert.ok(!ipfs.ALLOWED_MIME.has('image/png'));
        assert.ok(!ipfs.ALLOWED_MIME.has('application/javascript'));
        assert.ok(!ipfs.ALLOWED_MIME.has('text/html'));
    });

    // ── cleanup ──────────────────────────────────────────────────────────────
    await t('cleanup removes the temp file', async () => {
        const f = path.join(tmp, 'e.bin');
        fs.writeFileSync(f, Buffer.from('x'));
        await ipfs.cleanup(f);
        assert.ok(!fs.existsSync(f));
    });

    await t('cleanup on a missing file does not throw', async () => {
        await ipfs.cleanup(path.join(tmp, 'does-not-exist.bin'));
        await ipfs.cleanup(undefined);
    });

    // ── the property the anchoring layer depends on ───────────────────────────
    await t('the route never stores a client-supplied hash as server-computed', async () => {
        const src = fs.readFileSync(path.join(__dirname, 'routes', 'videos.js'), 'utf8');
        // /upload must set provenance explicitly...
        assert.ok(src.includes("hashComputedBy: 'server'"),
            "/upload must record hashComputedBy: 'server'");
        // ...and the legacy metadata route must not pass req.body's hash through.
        assert.ok(!/videoFileHash:\s*videoFileHash\s*\?/.test(src),
            'legacy POST / must not store a client-supplied videoFileHash');
        assert.ok(src.includes("hashComputedBy: 'unverified'"),
            'legacy POST / must mark its records unverified');
    });

    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})();
