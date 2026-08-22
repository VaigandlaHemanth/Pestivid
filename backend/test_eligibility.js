/**
 * Proves the provenance pipeline cannot be bypassed by using a different URL.
 *
 * THE ATTACK
 *   All the provenance work happens in POST /api/videos/upload. The legacy
 *   POST /api/videos accepts metadata only and creates a Video with no
 *   fingerprint and hashComputedBy 'unverified'. Before this gate, both the
 *   funding-request and listing routes accepted any CID the caller owned, so:
 *
 *     POST /api/videos            -> unverified record, zero checks
 *     POST /api/funding-requests  -> raise money against it
 *
 *   gave a money claim backed by a "video" we never hashed, never fingerprinted,
 *   and which need not exist on IPFS at all.
 *
 *     node test_eligibility.js
 */

const assert = require('assert');
const crypto = require('crypto');

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
    await mongoose.connect(mem.getUri(), { dbName: 'elig_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'AnchorBatch', 'RetiredFingerprint']
        .forEach((m) => require(`./models/${m}`));

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');
    const { requireFundableVideo } = require('./services/videoEligibility');

    const farmer = await User.create({
        name: 'F', email: 'f@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const other = await User.create({
        name: 'O', email: 'o@t.local', password: 'x'.repeat(20), role: 'farmer',
    });

    const fakeHashes = () => Array.from({ length: 12 },
        (_, i) => crypto.createHash('sha256').update(`f${i}`).digest('hex').slice(0, 16));

    // A properly uploaded video: server hash + fingerprint.
    const good = await Video.create({
        cid: 'bafygood', storageType: 'ipfs',
        videoFileHash: 'a'.repeat(64), hashComputedBy: 'server',
        farmerWallet: farmer._id, crop: 'Potato', location: 'L', purpose: 'funding',
        fingerprint: { frameHashes: fakeHashes(), nFrames: 12, algorithm: 'dhash64/9x8' },
    });

    // What the legacy metadata-only route produces.
    await Video.create({
        cid: 'bafylegacy', storageType: 'ipfs',
        hashComputedBy: 'unverified',
        farmerWallet: farmer._id, crop: 'Potato', location: 'L', purpose: 'funding',
    });

    // Server-hashed but undecodable, so the reuse check never ran.
    await Video.create({
        cid: 'bafynofp', storageType: 'ipfs',
        videoFileHash: 'b'.repeat(64), hashComputedBy: 'server',
        farmerWallet: farmer._id, crop: 'Potato', location: 'L', purpose: 'funding',
    });

    // Uploaded for a different declared purpose.
    await Video.create({
        cid: 'bafyshowcase', storageType: 'ipfs',
        videoFileHash: 'c'.repeat(64), hashComputedBy: 'server',
        farmerWallet: farmer._id, crop: 'Potato', location: 'L', purpose: 'agristream',
        fingerprint: { frameHashes: fakeHashes(), nFrames: 12, algorithm: 'dhash64/9x8' },
    });

    await t('a properly uploaded video IS fundable', async () => {
        const r = await requireFundableVideo('bafygood', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, true, r.message);
        assert.strictEqual(String(r.video._id), String(good._id));
    });

    await t('THE BYPASS: a metadata-only record is REFUSED', async () => {
        const r = await requireFundableVideo('bafylegacy', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false, 'an unverified video was accepted as funding evidence');
        assert.strictEqual(r.code, 'video_not_verified');
        assert.strictEqual(r.status, 409);
    });

    await t('a server-hashed video with no fingerprint is REFUSED', async () => {
        const r = await requireFundableVideo('bafynofp', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false);
        assert.strictEqual(r.code, 'video_not_analysed');
    });

    await t('a video declared for a different purpose is REFUSED', async () => {
        const r = await requireFundableVideo('bafyshowcase', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false, 'a showcase video was accepted as loan evidence');
        assert.strictEqual(r.code, 'video_wrong_purpose');
    });

    await t('another farmer cannot use this video', async () => {
        const r = await requireFundableVideo('bafygood', other._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false);
        assert.strictEqual(r.code, 'video_not_found');
    });

    await t('a non-existent CID is refused', async () => {
        const r = await requireFundableVideo('bafynope', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false);
        assert.strictEqual(r.code, 'video_not_found');
    });

    await t('empty / missing / non-string CIDs are refused, not crashed on', async () => {
        for (const bad of ['', '   ', null, undefined, 42, {}, []]) {
            const r = await requireFundableVideo(bad, farmer._id, { purpose: 'funding' });
            assert.strictEqual(r.ok, false, `accepted cid=${JSON.stringify(bad)}`);
        }
    });

    await t('a fingerprint with too few frames is refused', async () => {
        await Video.create({
            cid: 'bafyshort', storageType: 'ipfs',
            videoFileHash: 'd'.repeat(64), hashComputedBy: 'server',
            farmerWallet: farmer._id, crop: 'P', location: 'L', purpose: 'funding',
            fingerprint: { frameHashes: ['aaaa', 'bbbb'], nFrames: 2 },
        });
        const r = await requireFundableVideo('bafyshort', farmer._id, { purpose: 'funding' });
        assert.strictEqual(r.ok, false);
        assert.strictEqual(r.code, 'video_not_analysed');
    });

    await t('both money routes go through this one gate', async () => {
        const fr = require('fs').readFileSync('./routes/fundingRequests.js', 'utf8');
        const li = require('fs').readFileSync('./routes/listings.js', 'utf8');
        assert.ok(fr.includes('requireFundableVideo'), 'fundingRequests.js does not use the gate');
        assert.ok(li.includes('requireFundableVideo'), 'listings.js does not use the gate');
        // And neither may still be doing its own partial ownership-only check.
        assert.ok(!/const video = await Video\.findOne\(\{ cid: cid, farmerWallet: req\.user\._id \}\);/.test(fr),
            'fundingRequests.js still has the old ownership-only lookup');
        assert.ok(!/const video = await Video\.findOne\(\{ cid: cid, farmerWallet: req\.user\._id \}\);/.test(li),
            'listings.js still has the old ownership-only lookup');
    });

    await mongoose.disconnect();
    await mem.stop();
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch(async (e) => {
    console.error('harness error:', e.message);
    try { await mongoose.disconnect(); } catch (_) {}
    process.exit(1);
});
