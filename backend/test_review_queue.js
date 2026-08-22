/**
 * Provenance review: the queue, the decision, and the privilege boundary.
 *
 * services/provenance.js has always set reviewState='flagged' when a new upload's
 * perceptual fingerprint matches one already on the platform. Nothing listed
 * those videos, nothing could record a decision, and the 'cleared'/'rejected'
 * values sitting in the schema enum were unreachable by any code path. Detection
 * ran on every upload and the result was discarded.
 *
 * Adding a reviewer means adding a privileged role, and that is the dangerous
 * part: putting 'admin' into a public enum without blocking self-assignment at
 * registration would hand reviewer powers to anyone who can POST a signup. That
 * case is tested first and deliberately.
 *
 *     node test_review_queue.js
 */

const assert = require('assert');
const express = require('express');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-secret-for-review-suite';

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
    await mongoose.connect(mem.getUri(), { dbName: 'review_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');

    const farmer = await User.create({
        name: 'Review Farmer', email: 'rf@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const admin = await User.create({
        name: 'Platform Reviewer', email: 'admin@t.local', password: 'x'.repeat(20), role: 'admin',
    });
    const investor = await User.create({
        name: 'Some Investor', email: 'si@t.local', password: 'x'.repeat(20), role: 'investor',
    });

    let actingAs = admin;
    const authModule = require('./routes/auth');
    const realAuth = authModule.authenticateToken;
    authModule.authenticateToken = (req, res, next) => { req.user = actingAs; next(); };
    delete require.cache[require.resolve('./routes/videos')];
    const videosRouter = require('./routes/videos');

    const app = express();
    app.use(express.json());
    app.use('/api/videos', videosRouter);
    // The real auth router, for the registration-escalation test.
    authModule.authenticateToken = realAuth;
    app.use('/api/auth', authModule.router);
    authModule.authenticateToken = (req, res, next) => { req.user = actingAs; next(); };
    const srv = app.listen(4883);

    const call = async (method, path, body) => {
        const r = await fetch(`http://127.0.0.1:4883${path}`, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        let parsed = null;
        try { parsed = await r.json(); } catch (_) { /* empty body */ }
        return { status: r.status, body: parsed };
    };

    let n = 0;
    const mkVideo = (overrides = {}) => Video.create({
        cid: `bafy-review-${n++}`,
        farmerWallet: farmer._id,
        crop: 'Potato', location: 'Test Field', purpose: 'funding',
        hashComputedBy: 'server',
        videoFileHash: 'a'.repeat(64),
        fingerprint: {
            frameHashes: ['a'.repeat(16), 'b'.repeat(16), 'c'.repeat(16), 'd'.repeat(16)],
            nFrames: 4,
        },
        provenance: { flags: [], reviewState: 'none' },
        ...overrides,
    });

    // ── the privilege boundary comes first ──────────────────────────────────
    await t('THE ESCALATION: registration refuses a self-assigned admin role', async () => {
        const r = await call('POST', '/api/auth/register', {
            name: 'Sneaky', email: 'sneaky@t.local', password: 'password123', role: 'admin',
        });
        assert.strictEqual(r.status, 403,
            `admin was self-assignable (${r.status}) -- the reviewer role is public`);
        assert.strictEqual(r.body.code, 'role_not_assignable');
        const u = await User.findOne({ email: 'sneaky@t.local' }).lean();
        assert.strictEqual(u, null, 'the account was created anyway');
    });

    await t('a farmer cannot read the review queue', async () => {
        actingAs = farmer;
        const r = await call('GET', '/api/videos/review-queue');
        assert.strictEqual(r.status, 403, `expected 403, got ${r.status}`);
        assert.strictEqual(r.body.code, 'admin_only');
    });

    await t('an investor cannot read the review queue', async () => {
        actingAs = investor;
        const r = await call('GET', '/api/videos/review-queue');
        assert.strictEqual(r.status, 403, `expected 403, got ${r.status}`);
    });

    await t('a farmer cannot clear the flag on their own video', async () => {
        const v = await mkVideo({
            provenance: { flags: ['duplicate_of_other_farmer'], reviewState: 'flagged' },
        });
        actingAs = farmer;
        const r = await call('POST', `/api/videos/${v.cid}/review`, { decision: 'cleared' });
        assert.strictEqual(r.status, 403, `a farmer self-cleared a flag (${r.status})`);
        const after = await Video.findById(v._id).lean();
        assert.strictEqual(after.provenance.reviewState, 'flagged', 'the flag was cleared anyway');
    });

    // ── the queue ───────────────────────────────────────────────────────────
    await t('THE DEAD END: flagged videos are now listed', async () => {
        const flagged = await mkVideo({
            provenance: {
                flags: ['duplicate_of_other_farmer'], reviewState: 'flagged',
                similarityFraction: 0.92, similarityKind: 'other_farmer',
            },
        });
        await mkVideo();   // an unflagged one, which must NOT appear
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue');
        assert.strictEqual(r.status, 200, `expected 200, got ${r.status}`);
        const cids = r.body.items.map((i) => i.cid);
        assert.ok(cids.includes(flagged.cid), 'the flagged video is not in the queue');
        assert.strictEqual(r.body.items.every((i) => i.reviewState === 'flagged'), true,
            'the queue contains videos that are not flagged');
    });

    await t('the queue carries the context a decision needs', async () => {
        const original = await mkVideo({ crop: 'Potato' });
        const dupe = await mkVideo({
            provenance: {
                flags: ['duplicate_of_other_farmer'], reviewState: 'flagged',
                similarTo: original._id, similarToFarmer: farmer._id,
                similarityFraction: 0.88, similarityKind: 'other_farmer',
            },
        });
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue');
        const row = r.body.items.find((i) => i.cid === dupe.cid);
        assert.ok(row, 'the flagged video is missing');
        // These are the fields that were being read under the wrong names and
        // would have serialised as null on every single row.
        assert.ok(row.similarTo, 'similarTo is missing -- no way to compare the two videos');
        assert.strictEqual(row.similarTo.cid, original.cid,
            `similarTo.cid was ${row.similarTo && row.similarTo.cid}`);
        assert.strictEqual(row.similarityFraction, 0.88,
            `similarityFraction was ${row.similarityFraction}`);
        assert.strictEqual(row.similarityKind, 'other_farmer',
            `similarityKind was ${row.similarityKind}`);
        assert.ok(row.farmer && row.farmer.name, 'the farmer is not identified');
        assert.ok(row.flags.length > 0, 'the flags are missing');
    });

    await t('the queue is oldest-first, so nobody is overtaken', async () => {
        await Video.deleteMany({});
        const old = await mkVideo({
            uploadTimestamp: new Date('2026-01-01'),
            provenance: { flags: ['x'], reviewState: 'flagged' },
        });
        const recent = await mkVideo({
            uploadTimestamp: new Date('2026-08-01'),
            provenance: { flags: ['x'], reviewState: 'flagged' },
        });
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue');
        assert.strictEqual(r.body.items[0].cid, old.cid,
            'the newest upload was listed first, so an older one waits longer');
        assert.strictEqual(r.body.items[1].cid, recent.cid);
    });

    await t('a truncated queue says so', async () => {
        await Video.deleteMany({});
        for (let i = 0; i < 3; i++) {
            await mkVideo({ provenance: { flags: ['x'], reviewState: 'flagged' } });
        }
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue?limit=2');
        assert.strictEqual(r.body.items.length, 2);
        assert.strictEqual(r.body.truncated, true,
            'a truncated list did not admit it, so a reviewer would think they were done');
    });

    // ── the decision ────────────────────────────────────────────────────────
    await t('THE UNREACHABLE STATE: a flag can be cleared', async () => {
        const v = await mkVideo({
            provenance: { flags: ['duplicate_same_farmer'], reviewState: 'flagged' },
        });
        actingAs = admin;
        const r = await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'cleared', note: 'Re-filmed the same field the same morning.' });
        assert.strictEqual(r.status, 200, `expected 200, got ${r.status} ${JSON.stringify(r.body)}`);
        const after = await Video.findById(v._id).lean();
        assert.strictEqual(after.provenance.reviewState, 'cleared');
        assert.strictEqual(String(after.provenance.reviewedBy), String(admin._id),
            'the decision was not attributed to anyone');
        assert.ok(after.provenance.reviewedAt, 'the decision has no timestamp');
    });

    await t('a rejection must be explained', async () => {
        const v = await mkVideo({
            provenance: { flags: ['duplicate_of_other_farmer'], reviewState: 'flagged' },
        });
        actingAs = admin;
        const bare = await call('POST', `/api/videos/${v.cid}/review`, { decision: 'rejected' });
        assert.strictEqual(bare.status, 400,
            'a video was rejected with no reason given, so it could not be appealed');
        assert.strictEqual(bare.body.code, 'note_required');

        const tooShort = await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'rejected', note: 'nope' });
        assert.strictEqual(tooShort.status, 400);

        const ok = await call('POST', `/api/videos/${v.cid}/review`, {
            decision: 'rejected',
            note: 'Identical footage to bafy-review-0, uploaded by a different farmer.',
        });
        assert.strictEqual(ok.status, 200, `a properly explained rejection failed: ${ok.status}`);
        const after = await Video.findById(v._id).lean();
        assert.strictEqual(after.provenance.reviewState, 'rejected');
        assert.ok(after.provenance.reviewNote.length >= 10, 'the note was not stored');
    });

    await t('a decision cannot be made twice', async () => {
        const v = await mkVideo({
            provenance: { flags: ['x'], reviewState: 'flagged' },
        });
        actingAs = admin;
        const first = await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'cleared', note: 'fine' });
        assert.strictEqual(first.status, 200);
        const second = await call('POST', `/api/videos/${v.cid}/review`, {
            decision: 'rejected', note: 'changed my mind about this one entirely',
        });
        assert.strictEqual(second.status, 409, `a second decision was accepted (${second.status})`);
        assert.strictEqual(second.body.code, 'already_reviewed');
        const after = await Video.findById(v._id).lean();
        assert.strictEqual(after.provenance.reviewState, 'cleared', 'the decision was overwritten');
    });

    await t('an unflagged video cannot be decided', async () => {
        const v = await mkVideo();
        actingAs = admin;
        const r = await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'rejected', note: 'no reason at all, this is a clean video' });
        assert.strictEqual(r.status, 409, `expected 409, got ${r.status}`);
        assert.strictEqual(r.body.code, 'not_flagged');
    });

    await t('an invalid decision value is refused', async () => {
        const v = await mkVideo({ provenance: { flags: ['x'], reviewState: 'flagged' } });
        actingAs = admin;
        for (const decision of ['banned', 'flagged', 'none', '', null]) {
            const r = await call('POST', `/api/videos/${v.cid}/review`, { decision });
            assert.strictEqual(r.status, 400, `decision=${decision} gave ${r.status}`);
        }
    });

    await t('a decision does NOT delete the video or change eligibility', async () => {
        const v = await mkVideo({
            purpose: 'funding',
            provenance: { flags: ['duplicate_of_other_farmer'], reviewState: 'flagged' },
        });
        actingAs = admin;
        await call('POST', `/api/videos/${v.cid}/review`, {
            decision: 'rejected', note: 'Clearly the same footage as an earlier upload.',
        });
        const after = await Video.findById(v._id).lean();
        assert.ok(after, 'the video was deleted -- review must record, not enforce');
        assert.strictEqual(after.purpose, 'funding', 'the purpose was changed');
        assert.strictEqual(after.hashComputedBy, 'server', 'the integrity fields were touched');
    });

    await t('an unknown CID is a 404', async () => {
        actingAs = admin;
        const r = await call('POST', '/api/videos/bafy-does-not-exist/review',
            { decision: 'cleared', note: 'n/a' });
        assert.strictEqual(r.status, 404, `expected 404, got ${r.status}`);
    });

    await t('reviewed videos can be listed back for audit', async () => {
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue?state=rejected');
        assert.strictEqual(r.status, 200);
        assert.strictEqual(r.body.state, 'rejected');
        assert.ok(r.body.items.every((i) => i.reviewState === 'rejected'),
            'the rejected list contains other states');
        assert.ok(r.body.items.every((i) => i.reviewedBy),
            'a reviewed item has no reviewer recorded');
    });

    srv.close();
    await mongoose.disconnect();
    await mem.stop();

    console.log(`\n  ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})();
