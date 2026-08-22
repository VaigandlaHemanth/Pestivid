/**
 * GET /api/videos must not leak one farmer's private evidence to another.
 *
 * Before this, ?farmerId=<victim> returned every one of another farmer's videos
 * -- including CIDs and SHA-256 file hashes -- to any authenticated account. A
 * competitor could enumerate a farmer's funding evidence and pull the files
 * straight off a public IPFS gateway.
 *
 *     node test_video_scope.js
 */

const assert = require('assert');
const http = require('http');

const express = require('express');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

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
    const mem = await MongoMemoryServer.create();
    await mongoose.connect(mem.getUri(), { dbName: 'scope_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) {}
    });

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');

    const alice = await User.create({
        name: 'Alice Farmer', email: 'a@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const bob = await User.create({
        name: 'Bob Farmer', email: 'b@t.local', password: 'x'.repeat(20), role: 'farmer',
    });

    const mk = (owner, cid, purpose) => Video.create({
        cid, storageType: 'ipfs',
        videoFileHash: 'f'.repeat(64), hashComputedBy: 'server',
        farmerWallet: owner._id, crop: 'Potato', location: 'L', purpose,
        fingerprint: { frameHashes: ['a'.repeat(16), 'b'.repeat(16), 'c'.repeat(16), 'd'.repeat(16)], nFrames: 4 },
    });

    await mk(alice, 'bafy-alice-funding', 'funding');
    await mk(alice, 'bafy-alice-sell', 'sell');
    await mk(alice, 'bafy-alice-public', 'agristream');
    await mk(bob, 'bafy-bob-public', 'agristream');

    // Mount the real router with a stubbed auth middleware so we exercise the
    // actual handler rather than a reimplementation of it.
    let actingAs = bob;
    const authStub = (req, res, next) => { req.user = actingAs; next(); };
    const authModule = require('./routes/auth');
    const realAuth = authModule.authenticateToken;
    authModule.authenticateToken = authStub;
    delete require.cache[require.resolve('./routes/videos')];
    const videosRouter = require('./routes/videos');

    const app = express();
    app.use(express.json());
    app.use('/api/videos', videosRouter);
    const srv = app.listen(4821);
    const get = async (q) => {
        const r = await fetch(`http://127.0.0.1:4821/api/videos${q}`);
        return { status: r.status, body: await r.json().catch(() => null) };
    };

    await t('THE LEAK: Bob asking for Alice\'s videos gets no file hashes', async () => {
        actingAs = bob;
        const { status, body } = await get(`?farmerId=${alice._id}`);
        assert.strictEqual(status, 200);
        for (const v of body) {
            assert.ok(!v.videoFileHash,
                `leaked videoFileHash for ${v.cid} to another farmer`);
        }
    });

    await t('Bob only sees Alice\'s PUBLIC showcase videos, not her funding evidence', async () => {
        actingAs = bob;
        const { body } = await get(`?farmerId=${alice._id}`);
        const cids = body.map((v) => v.cid);
        assert.ok(!cids.includes('bafy-alice-funding'),
            'exposed a private funding video to another farmer');
        assert.ok(!cids.includes('bafy-alice-sell'),
            'exposed a private sale video to another farmer');
        assert.ok(cids.includes('bafy-alice-public'),
            'public showcase browsing broke');
    });

    await t('Alice asking for her OWN videos gets everything she needs', async () => {
        actingAs = alice;
        const { body } = await get(`?farmerId=${alice._id}`);
        const cids = body.map((v) => v.cid);
        assert.strictEqual(cids.length, 3, `own view returned ${cids.length} of 3`);
        assert.ok(cids.includes('bafy-alice-funding'), 'own funding video missing');
        const withHash = body.filter((v) => v.videoFileHash).length;
        assert.strictEqual(withHash, 3, 'own videos must carry the hash for the evidence dropdown');
        assert.ok(body.every((v) => v.fingerprinted === true),
            'own videos should report their fingerprint state');
    });

    await t('an unscoped browse is public-only', async () => {
        actingAs = bob;
        const { body } = await get('');
        assert.ok(body.every((v) => v.purpose === 'agristream'),
            'unscoped browse exposed non-showcase videos');
        assert.ok(body.every((v) => !v.videoFileHash),
            'unscoped browse leaked file hashes');
    });

    await t('a malformed farmerId is rejected, not ignored', async () => {
        actingAs = bob;
        const { status } = await get('?farmerId=not-an-objectid');
        assert.strictEqual(status, 400);
    });

    srv.close();
    authModule.authenticateToken = realAuth;
    await mongoose.disconnect();
    await mem.stop();
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch(async (e) => {
    console.error('harness error:', e.message);
    try { await mongoose.disconnect(); } catch (_) {}
    process.exit(1);
});
