/**
 * The farmer's side of a provenance flag: the notice, and the appeal.
 *
 * The reviewer queue closed one gap (nothing could act on a flag) and left
 * another: the flag was still invisible to the person it was about. A video could
 * be flagged, and later rejected, and the farmer would never be told it happened,
 * why, or that they could say anything about it. A check only the platform can see
 * is not a safeguard.
 *
 * It also left review one-sided by construction. The reviewer has a dHash
 * similarity score; only the farmer knows that the first take was too dark and
 * they filmed the row again. That explanation cannot appear in a frame
 * comparison, and it is the common case.
 *
 *     node test_appeal.js
 */

const assert = require('assert');
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
    await mongoose.connect(mem.getUri(), { dbName: 'appeal_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');
    const Notification = mongoose.model('Notification');

    const farmer = await User.create({
        name: 'Appeal Farmer', email: 'af@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const other = await User.create({
        name: 'Other Farmer', email: 'of@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const admin = await User.create({
        name: 'Reviewer', email: 'rev@t.local', password: 'x'.repeat(20), role: 'admin',
    });

    let actingAs = farmer;
    const authModule = require('./routes/auth');
    authModule.authenticateToken = (req, res, next) => { req.user = actingAs; next(); };
    delete require.cache[require.resolve('./routes/videos')];
    const app = express();
    app.use(express.json());
    app.use('/api/videos', require('./routes/videos'));
    const srv = app.listen(4885);

    const call = async (method, path, body) => {
        const r = await fetch(`http://127.0.0.1:4885${path}`, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        let parsed = null;
        try { parsed = await r.json(); } catch (_) { /* empty */ }
        return { status: r.status, body: parsed };
    };

    let n = 0;
    const mkVideo = (prov, owner = farmer) => Video.create({
        cid: `bafy-appeal-${n++}`,
        farmerWallet: owner._id,
        crop: 'Potato', location: 'Field', purpose: 'funding',
        hashComputedBy: 'server', videoFileHash: 'a'.repeat(64),
        fingerprint: { frameHashes: ['a'.repeat(16), 'b'.repeat(16),
                                     'c'.repeat(16), 'd'.repeat(16)], nFrames: 4 },
        provenance: prov,
    });

    const flagged = () => ({
        flags: ['matches_another_upload'], reviewState: 'flagged',
        similarityFraction: 0.91, similarityKind: 'other_farmer',
    });

    // ── the appeal ──────────────────────────────────────────────────────────
    await t('THE SILENCE: a flagged farmer can state their side', async () => {
        const v = await mkVideo(flagged());
        actingAs = farmer;
        const r = await call('POST', `/api/videos/${v.cid}/appeal`, {
            statement: 'The first take was too dark so I filmed the same row again.',
        });
        assert.strictEqual(r.status, 200, `expected 200, got ${r.status} ${JSON.stringify(r.body)}`);
        const after = await Video.findById(v._id).lean();
        assert.ok(after.provenance.appeal.statement.includes('too dark'),
            'the statement was not stored');
        assert.ok(after.provenance.appeal.submittedAt, 'no submission time recorded');
    });

    await t('an appeal does NOT change the review state', async () => {
        const v = await mkVideo(flagged());
        actingAs = farmer;
        await call('POST', `/api/videos/${v.cid}/appeal`,
            { statement: 'This is my own field, filmed twice.' });
        const after = await Video.findById(v._id).lean();
        assert.strictEqual(after.provenance.reviewState, 'flagged',
            'appealing cleared the flag -- a farmer must not be able to self-clear');
    });

    await t('a REJECTED farmer can still respond', async () => {
        const v = await mkVideo({
            flags: ['matches_another_upload'], reviewState: 'rejected',
            reviewNote: 'Looks like an earlier upload.',
        });
        actingAs = farmer;
        const r = await call('POST', `/api/videos/${v.cid}/appeal`, {
            statement: 'That earlier upload is also mine, from the same plot last week.',
        });
        assert.strictEqual(r.status, 200,
            `a rejected farmer could not respond (${r.status}) -- they need it most`);
        assert.ok(/looked at again/i.test(r.body.message),
            `message did not promise re-review: ${r.body.message}`);
    });

    await t('nobody can appeal about someone else\'s video', async () => {
        const v = await mkVideo(flagged());
        actingAs = other;
        const r = await call('POST', `/api/videos/${v.cid}/appeal`,
            { statement: 'Putting words in another farmer\'s mouth.' });
        assert.strictEqual(r.status, 403, `expected 403, got ${r.status}`);
        const after = await Video.findById(v._id).lean();
        assert.ok(!after.provenance.appeal || !after.provenance.appeal.statement,
            'a stranger\'s statement was attached to this farmer\'s video');
    });

    await t('an unflagged or cleared video has nothing to appeal', async () => {
        actingAs = farmer;
        const clean = await mkVideo({ flags: [], reviewState: 'none' });
        const r1 = await call('POST', `/api/videos/${clean.cid}/appeal`,
            { statement: 'Nothing is wrong with this one.' });
        assert.strictEqual(r1.status, 409);
        assert.strictEqual(r1.body.code, 'not_flagged');

        const ok = await mkVideo({ flags: ['x'], reviewState: 'cleared' });
        const r2 = await call('POST', `/api/videos/${ok.cid}/appeal`,
            { statement: 'Already fine, no need.' });
        assert.strictEqual(r2.status, 409);
        assert.strictEqual(r2.body.code, 'already_cleared');
    });

    await t('an empty or trivial statement is refused', async () => {
        const v = await mkVideo(flagged());
        actingAs = farmer;
        for (const statement of [undefined, '', '   ', 'no']) {
            const r = await call('POST', `/api/videos/${v.cid}/appeal`, { statement });
            assert.strictEqual(r.status, 400, `statement=${JSON.stringify(statement)} gave ${r.status}`);
        }
    });

    await t('a revised appeal replaces the statement and counts the revision', async () => {
        const v = await mkVideo(flagged());
        actingAs = farmer;
        await call('POST', `/api/videos/${v.cid}/appeal`, { statement: 'First explanation here.' });
        await call('POST', `/api/videos/${v.cid}/appeal`, { statement: 'Better explanation here.' });
        const after = await Video.findById(v._id).lean();
        assert.ok(after.provenance.appeal.statement.includes('Better'),
            'the revised statement did not replace the first');
        assert.strictEqual(after.provenance.appeal.revisions, 1,
            `revisions was ${after.provenance.appeal.revisions}, expected 1`);
    });

    // ── the reviewer must see it ─────────────────────────────────────────────
    await t('THE ONE-SIDED REVIEW: the queue shows the farmer\'s statement', async () => {
        await Video.deleteMany({});
        const v = await mkVideo(flagged());
        actingAs = farmer;
        await call('POST', `/api/videos/${v.cid}/appeal`,
            { statement: 'Re-filmed because the light was bad.' });

        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue');
        const row = r.body.items.find((i) => i.cid === v.cid);
        assert.ok(row, 'the flagged video left the queue');
        assert.ok(row.appeal, 'the queue does not carry the appeal -- review stays one-sided');
        assert.ok(row.appeal.statement.includes('light was bad'),
            `statement not surfaced: ${JSON.stringify(row.appeal)}`);
    });

    await t('the queue shows whether the farmer was ever told', async () => {
        await Video.deleteMany({});
        const notified = await mkVideo({ ...flagged(), flagNotifiedAt: new Date() });
        const silent = await mkVideo(flagged());
        actingAs = admin;
        const r = await call('GET', '/api/videos/review-queue');
        const a = r.body.items.find((i) => i.cid === notified.cid);
        const b = r.body.items.find((i) => i.cid === silent.cid);
        assert.ok(a.farmerNotifiedAt, 'a notified farmer shows as un-notified');
        assert.strictEqual(b.farmerNotifiedAt, null,
            'an un-notified flag is indistinguishable from a notified one');
    });

    // ── the outcome reaches the farmer ──────────────────────────────────────
    await t('THE SECRET DECISION: a cleared farmer is told', async () => {
        await Notification.deleteMany({});
        const v = await mkVideo(flagged());
        actingAs = admin;
        const r = await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'cleared', note: 'Same farmer, legitimate re-film.' });
        assert.strictEqual(r.status, 200);
        const notes = await Notification.find({ recipient: farmer._id }).lean();
        assert.strictEqual(notes.length, 1, `farmer got ${notes.length} notices, expected 1`);
        assert.strictEqual(notes[0].type, 'success');
        assert.ok(/cleared/i.test(notes[0].message), `notice text: ${notes[0].message}`);
    });

    await t('a rejected farmer is told, WITH the reason', async () => {
        await Notification.deleteMany({});
        const v = await mkVideo(flagged());
        actingAs = admin;
        const reason = 'Identical footage to an earlier upload by another farmer.';
        await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'rejected', note: reason });
        const notes = await Notification.find({ recipient: farmer._id }).lean();
        assert.strictEqual(notes.length, 1, `farmer got ${notes.length} notices`);
        assert.ok(notes[0].message.includes(reason),
            `the reason was withheld from the farmer: ${notes[0].message}`);
        assert.ok(/explanation/i.test(notes[0].message),
            'the notice does not tell them they can respond');
    });

    await t('the notice does not accuse -- it describes', async () => {
        await Notification.deleteMany({});
        const v = await mkVideo(flagged());
        actingAs = admin;
        await call('POST', `/api/videos/${v.cid}/review`,
            { decision: 'rejected', note: 'Matches an earlier upload.' });
        const note = (await Notification.findOne({ recipient: farmer._id }).lean()).message;
        // A similarity heuristic must not be reported as proven misconduct.
        for (const word of ['fraud', 'fraudulent', 'stole', 'stolen', 'cheat', 'lying', 'fake']) {
            assert.ok(!new RegExp(word, 'i').test(note),
                `the notice calls the farmer a ${word}: "${note}"`);
        }
    });

    await t('a notice failure does not lose the review decision', async () => {
        const v = await mkVideo(flagged());
        actingAs = admin;
        // Break notification saving, then confirm the decision still persists.
        const proto = Notification.prototype.save;
        Notification.prototype.save = () => Promise.reject(new Error('simulated outage'));
        try {
            const r = await call('POST', `/api/videos/${v.cid}/review`,
                { decision: 'cleared', note: 'fine' });
            assert.strictEqual(r.status, 200, `the decision failed with notices down (${r.status})`);
            const after = await Video.findById(v._id).lean();
            assert.strictEqual(after.provenance.reviewState, 'cleared',
                'the decision was rolled back because a notice failed');
        } finally {
            Notification.prototype.save = proto;
        }
    });

    srv.close();
    await mongoose.disconnect();
    await mem.stop();

    console.log(`\n  ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})();
