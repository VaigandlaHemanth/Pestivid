/**
 * Integration test for the provenance and anchoring chain, against a real
 * MongoDB (in-memory) so every query, index and schema constraint actually runs.
 *
 * Covers what the unit tests cannot:
 *   - fingerprints round-trip through the Video schema
 *   - duplicate detection finds a re-encoded copy ACROSS farmers
 *   - anchoring excludes client-hashed records
 *   - an inclusion proof regenerated from a stored batch still verifies
 *   - a second anchor run does not re-anchor what is already anchored
 *
 * Pinata is not involved: the pin is a network call with a credential, and the
 * logic worth testing is everything around it.
 *
 *     node test_provenance_integration.js
 */

const assert = require('assert');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

const fpSvc = require('./services/videoFingerprint');
const merkle = require('./services/merkle');

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

const tmp = path.join(os.tmpdir(), `pv_int_${Date.now()}`);
const P = (n) => path.join(tmp, n);
const ff = (args) => execFileSync(fpSvc.ffmpegPath, ['-v', 'error', '-y', ...args], { stdio: 'pipe' });

(async () => {
    fs.mkdirSync(tmp, { recursive: true });
    const mem = await MongoMemoryServer.create();
    await mongoose.connect(mem.getUri(), { dbName: 'pestivid_test' });

    require('./models/User');
    require('./models/Video');
    require('./models/Listing');
    require('./models/FundingRequest');
    require('./models/AnchorBatch');
    require('./models/RetiredFingerprint');

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');
    const AnchorBatch = mongoose.model('AnchorBatch');

    // Services that read models must be required AFTER the models register.
    const provenance = require('./services/provenance');
    const anchor = require('./services/anchor');

    const farmerA = await User.create({
        name: 'Farmer A', email: 'a@test.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const farmerB = await User.create({
        name: 'Farmer B', email: 'b@test.local', password: 'x'.repeat(20), role: 'farmer',
    });

    // Two distinct clips, plus a re-encoded copy of the first.
    ff(['-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=15:duration=4', '-pix_fmt', 'yuv420p', P('v1.mp4')]);
    ff(['-f', 'lavfi', '-i', 'smptebars=size=320x240:rate=15:duration=4', '-pix_fmt', 'yuv420p', P('v2.mp4')]);
    ff(['-i', P('v1.mp4'), '-b:v', '120k', '-pix_fmt', 'yuv420p', P('v1_copy.mp4')]);

    const mkVideo = async (file, farmer, cid, extra = {}) => {
        const analysis = await provenance.analyse(P(file), { farmerId: farmer._id });
        return Video.create({
            cid,
            storageType: 'ipfs',
            videoFileHash: crypto.createHash('sha256').update(fs.readFileSync(P(file))).digest('hex'),
            hashComputedBy: 'server',
            farmerWallet: farmer._id,
            crop: 'Potato',
            location: 'Test District',
            purpose: 'funding',
            fingerprint: analysis.fingerprint,
            provenance: analysis.provenance,
            ...extra,
        });
    };

    // ── fingerprints persist ─────────────────────────────────────────────────
    const v1 = await mkVideo('v1.mp4', farmerA, 'bafyv1');
    await t('fingerprint round-trips through the Video schema', async () => {
        const got = await Video.findById(v1._id).lean();
        assert.ok(Array.isArray(got.fingerprint.frameHashes), 'frameHashes missing');
        assert.ok(got.fingerprint.frameHashes.length >= 4,
            `only ${got.fingerprint.frameHashes.length} frame hashes stored`);
        assert.ok(/^[0-9a-f]{16}$/.test(got.fingerprint.frameHashes[0]));
        assert.strictEqual(got.provenance.reviewState, 'none', 'first upload should be clean');
        assert.deepStrictEqual(got.provenance.flags, []);
    });

    // ── an unrelated video is not flagged ────────────────────────────────────
    const v2 = await mkVideo('v2.mp4', farmerB, 'bafyv2');
    await t('an unrelated video from another farmer is NOT flagged', async () => {
        const got = await Video.findById(v2._id).lean();
        assert.strictEqual(got.provenance.reviewState, 'none',
            `false positive: flags ${JSON.stringify(got.provenance.flags)}`);
    });

    // ── cross-farmer theft is caught ─────────────────────────────────────────
    await t('a re-encoded copy uploaded by ANOTHER farmer is flagged as theft', async () => {
        const a = await provenance.analyse(P('v1_copy.mp4'), { farmerId: farmerB._id });
        assert.strictEqual(a.provenance.reviewState, 'flagged',
            'a stolen video was not flagged');
        assert.ok(a.provenance.flags.includes('matches_another_upload'),
            `wrong flag: ${JSON.stringify(a.provenance.flags)}`);
        assert.strictEqual(a.provenance.similarityKind, 'other_farmer');
        assert.ok(a.provenance.similarityFraction >= 0.6,
            `similarity only ${a.provenance.similarityFraction}`);
        assert.strictEqual(String(a.provenance.similarTo), String(v1._id));
        // Flag names must not assign blame. On a cross-farmer match either party
        // could be the original -- a thief can simply upload first -- so a name
        // like 'matches_another_farmers_video' accuses whoever happened to be
        // second. Adjudication decides, not upload order.
        for (const f of a.provenance.flags) {
            assert.ok(!/thief|stolen|fraud|another_farmer/i.test(f),
                `flag "${f}" assigns blame before any adjudication`);
        }
    });

    // ── same-farmer recycling is a DIFFERENT flag ────────────────────────────
    await t('the same farmer re-uploading their own clip gets the recycling flag', async () => {
        const a = await provenance.analyse(P('v1_copy.mp4'), { farmerId: farmerA._id });
        assert.strictEqual(a.provenance.similarityKind, 'same_farmer');
        assert.ok(a.provenance.flags.includes('duplicate_of_own_video'),
            `wrong flag: ${JSON.stringify(a.provenance.flags)}`);
    });

    // ── reported location is stored, never marked verified ───────────────────
    await t('reported location is stored as-is and rejected when out of range', async () => {
        const ok = await provenance.analyse(P('v2.mp4'), {
            farmerId: farmerA._id,
            reportedLocation: { latitude: 17.385, longitude: 78.4867, accuracy: 12 },
        });
        assert.strictEqual(ok.provenance.reportedLatitude, 17.385);
        assert.strictEqual(ok.provenance.reportedAccuracyMetres, 12);
        const bad = await provenance.analyse(P('v2.mp4'), {
            farmerId: farmerA._id,
            reportedLocation: { latitude: 999, longitude: 12 },
        });
        assert.strictEqual(bad.provenance.reportedLatitude, undefined,
            'an impossible latitude was stored');
    });

    // ── anchoring ────────────────────────────────────────────────────────────
    // A record whose hash came from the client must never be anchored.
    await Video.create({
        cid: 'bafyclient', storageType: 'ipfs', hashComputedBy: 'unverified',
        videoFileHash: 'deadbeef'.repeat(8), farmerWallet: farmerA._id,
        crop: 'Potato', location: 'Test', purpose: 'funding',
    });

    await t('only server-computed hashes are eligible for anchoring', async () => {
        const pending = await anchor.pendingVideos();
        const cids = pending.map((p) => p.cid);
        assert.ok(cids.includes('bafyv1') && cids.includes('bafyv2'),
            `server-hashed videos missing from pending: ${cids}`);
        assert.ok(!cids.includes('bafyclient'),
            'a client-supplied hash was queued for anchoring');
    });

    let batch;
    await t('anchorPending builds a batch and stores the Merkle root', async () => {
        // The OTS stamp needs the network. Skip it here by removing the module
        // from the cache path is not possible cleanly, so accept either outcome
        // and assert on the parts that are ours.
        batch = await anchor.anchorPending();
        assert.ok(batch, 'no batch produced');
        assert.strictEqual(batch.leaves.length, batch.videos.length);
        assert.ok(batch.leaves.length >= 2, `only ${batch.leaves.length} leaves`);
        assert.ok(/^[0-9a-f]{64}$/.test(batch.merkleRoot), 'root is not a sha256 hex');
        const recomputed = merkle.buildTree(
            batch.leaves.map((h) => Buffer.from(h, 'hex'))).root.toString('hex');
        assert.strictEqual(recomputed, batch.merkleRoot,
            'stored root does not match its own stored leaves');
    });

    await t('every record in the batch verifies against the stored root', async () => {
        const { levels, root } = merkle.buildTree(
            batch.leaves.map((h) => Buffer.from(h, 'hex')));
        for (let i = 0; i < batch.videos.length; i++) {
            const proof = merkle.inclusionProof(levels, i);
            assert.ok(merkle.verifyRecord(batch.videos[i].canonicalRecord, proof, root),
                `record ${i} (${batch.videos[i].cid}) failed to verify`);
        }
    });

    await t('proofFor() returns a self-checking proof for a video', async () => {
        const proof = await anchor.proofFor('bafyv1');
        assert.ok(proof, 'no proof returned for an anchored video');
        assert.strictEqual(proof.selfCheck, true, 'proof failed its own self-check');
        assert.strictEqual(proof.merkleRoot, batch.merkleRoot);
        assert.ok(proof.record.includes('bafyv1'));
        assert.ok(Array.isArray(proof.inclusionProof));
        assert.ok(proof.howToVerify.length >= 4, 'verification instructions missing');
    });

    await t('proofFor() is null for a video that was never anchored', async () => {
        assert.strictEqual(await anchor.proofFor('bafyclient'), null);
        assert.strictEqual(await anchor.proofFor('bafy-does-not-exist'), null);
    });

    await t('a second anchor run does not re-anchor the same records', async () => {
        const again = await anchor.anchorPending();
        assert.strictEqual(again, null,
            'the same videos were queued for a second batch');
        assert.strictEqual(await AnchorBatch.countDocuments(), 1);
    });

    await t('a NEW upload after anchoring goes into the next batch only', async () => {
        await mkVideo('v2.mp4', farmerA, 'bafyv3');
        const b2 = await anchor.anchorPending();
        assert.ok(b2, 'new video was not anchored');
        assert.strictEqual(b2.videos.length, 1, `batch 2 had ${b2.videos.length} records`);
        assert.strictEqual(b2.videos[0].cid, 'bafyv3');
        assert.notStrictEqual(b2.merkleRoot, batch.merkleRoot);
    });

    await t('describe() never claims location or recency as verified', async () => {
        const v = await Video.findById(v1._id).lean();
        const d = require('./services/provenance').describe(v);
        const notVerifiedText = d.notVerified.join(' ').toLowerCase();
        assert.ok(notVerifiedText.includes('land'), 'must disclaim showing the farmer\'s land');
        assert.ok(notVerifiedText.includes('recorded recently') || notVerifiedText.includes('recent'),
            'must disclaim recency');
        assert.ok(notVerifiedText.includes('location'), 'must disclaim location');
        const verifiedText = d.verified.join(' ').toLowerCase();
        assert.ok(!verifiedText.includes('location'), 'location must not appear as verified');
        assert.ok(!/\bblockchain verified\b/i.test(verifiedText + notVerifiedText),
            'the phrase "blockchain verified" must not appear');
    });

    await t('describe() reports check COVERAGE, so silence cannot pass as clean', async () => {
        const v = await Video.findById(v1._id).lean();
        const d = require('./services/provenance').describe(v);
        assert.ok(Array.isArray(d.checksCompleted) && d.checksCompleted.length >= 2,
            'checksCompleted missing');
        const reuse = d.checksCompleted.find((c) => /re-use/i.test(c.check));
        assert.strictEqual(reuse.state, 'completed');
        // A record with no fingerprint must SAY the check could not run, rather
        // than looking identical to one that ran and found nothing. Engineering
        // that silence is exactly the attacker's strategy.
        const undecodable = {
            hashComputedBy: 'server', videoFileHash: 'ab', cid: 'x',
            provenance: { flags: ['fingerprint_unavailable'] },
        };
        const d2 = require('./services/provenance').describe(undecodable);
        const reuse2 = d2.checksCompleted.find((c) => /re-use/i.test(c.check));
        assert.strictEqual(reuse2.state, 'could_not_run',
            'an undecodable video looked the same as a checked-and-clean one');
    });

    await t('DELETE does not empty the duplicate index (closes the probe oracle)', async () => {
        const RetiredFingerprint = mongoose.model('RetiredFingerprint');
        // Simulate what the delete route retains.
        const gone = await Video.findById(v1._id).lean();
        await RetiredFingerprint.create({
            frameHashes: gone.fingerprint.frameHashes,
            nFrames: gone.fingerprint.nFrames,
            algorithm: gone.fingerprint.algorithm,
            farmerRef: 'hmac-placeholder',
            reason: 'deleted_by_farmer',
        });
        await Video.deleteOne({ _id: v1._id });

        // The same footage must still be recognised after deletion, otherwise
        // upload-then-delete is a free tuning oracle against the detector.
        const a = await provenance.analyse(P('v1_copy.mp4'), { farmerId: farmerB._id });
        assert.ok(a.provenance.flags.includes('matches_deleted_upload'),
            `deleted footage was not recognised: ${JSON.stringify(a.provenance.flags)}`);
    });

    await mongoose.disconnect();
    await mem.stop();
    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch(async (e) => {
    console.error('harness error:', e.message);
    try { await mongoose.disconnect(); } catch (_) {}
    process.exit(1);
});
