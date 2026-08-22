/**
 * LIVE end-to-end test: real Pinata pin, real ffmpeg, real Bitcoin stamp.
 *
 * This is the one path the other suites could not cover, because it makes a
 * network call with a credential. Everything around it was already tested; this
 * closes the last gap by running the actual chain a farmer's upload takes:
 *
 *   file -> server SHA-256 -> perceptual fingerprint -> duplicate check
 *        -> Pinata pin -> Video record -> Merkle batch -> Bitcoin anchor
 *        -> inclusion proof -> independent verification
 *
 * MongoDB is in-memory so this leaves no trace in a real database. Pinata is
 * REAL, so this pins a small test file to IPFS and then unpins it again.
 *
 * Skips itself with a clear message if PINATA_JWT is absent, so it is safe to
 * leave in CI.
 *
 *     node test_live_pin.js
 */

require('dotenv').config();

const assert = require('assert');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const axios = require('axios');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

const ipfs = require('./services/ipfsUpload');
const fpSvc = require('./services/videoFingerprint');
const merkle = require('./services/merkle');

let pass = 0;
let fail = 0;
const pinnedCids = [];

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

const tmp = path.join(os.tmpdir(), `pv_live_${Date.now()}`);
const P = (n) => path.join(tmp, n);
const ff = (args) => execFileSync(fpSvc.ffmpegPath, ['-v', 'error', '-y', ...args], { stdio: 'pipe' });

/** Leave no litter on the account: unpin whatever this test pinned. */
async function unpinAll() {
    for (const cid of pinnedCids) {
        try {
            await axios.delete(`https://api.pinata.cloud/pinning/unpin/${cid}`, {
                headers: { Authorization: `Bearer ${process.env.PINATA_JWT}` },
                timeout: 20000,
            });
            console.log(`  cleanup: unpinned ${cid}`);
        } catch (e) {
            console.log(`  cleanup: could not unpin ${cid} (${e.response ? e.response.status : e.message})`);
        }
    }
}

(async () => {
    if (!ipfs.pinataConfigured()) {
        console.log('  SKIPPED — PINATA_JWT is not set in backend/.env');
        process.exit(0);
    }

    fs.mkdirSync(tmp, { recursive: true });
    const mem = await MongoMemoryServer.create();
    await mongoose.connect(mem.getUri(), { dbName: 'pestivid_live' });
    require('./models/User');
    require('./models/Video');
    require('./models/Listing');
    require('./models/FundingRequest');
    require('./models/AnchorBatch');
    require('./models/RetiredFingerprint');

    const User = mongoose.model('User');
    const Video = mongoose.model('Video');
    const provenance = require('./services/provenance');
    const anchor = require('./services/anchor');

    const farmer = await User.create({
        name: 'Live Test Farmer', email: 'live@test.local',
        password: 'x'.repeat(20), role: 'farmer',
    });

    // A small, unique clip. Unique so it cannot collide with anything already
    // pinned on the account, and small so the pin is quick.
    const nonce = crypto.randomBytes(4).toString('hex');
    ff(['-f', 'lavfi', '-i', `testsrc2=size=320x240:rate=10:duration=3`,
        '-vf', `drawtext=text=${nonce}:fontsize=28:x=10:y=10:fontcolor=white`,
        '-pix_fmt', 'yuv420p', P('live.mp4')]);
    const sizeKb = fs.statSync(P('live.mp4')).size / 1024;
    console.log(`  test clip: ${sizeKb.toFixed(0)} KB, nonce ${nonce}\n`);

    let serverHash;
    await t('server computes SHA-256 from the received bytes', async () => {
        serverHash = await ipfs.sha256File(P('live.mp4'));
        const expected = crypto.createHash('sha256')
            .update(fs.readFileSync(P('live.mp4'))).digest('hex');
        assert.strictEqual(serverHash, expected);
    });

    let analysis;
    await t('perceptual fingerprint is produced with real ffmpeg', async () => {
        analysis = await provenance.analyse(P('live.mp4'), { farmerId: farmer._id });
        assert.ok(analysis.fingerprint, 'no fingerprint');
        assert.ok(analysis.fingerprint.frameHashes.length >= 3,
            `only ${analysis.fingerprint.frameHashes.length} frames`);
        assert.deepStrictEqual(analysis.provenance.flags, [],
            'a first upload should carry no flags');
    });

    let cid;
    await t('REAL Pinata pin returns a CID', async () => {
        const r = await ipfs.pinToPinata(P('live.mp4'), `pestivid_live_${nonce}.mp4`,
            { test: 'true', nonce });
        cid = r.cid;
        pinnedCids.push(cid);
        assert.ok(cid && cid.length > 20, `implausible cid: ${cid}`);
        console.log(`          cid ${cid}  (${(r.pinSize / 1024).toFixed(0)} KB pinned)`);
    });

    await t('the pinned file is retrievable and its bytes match our hash', async () => {
        // The real integrity claim: fetch it back from a public gateway and
        // recompute. If this fails, the CID we anchor does not name our bytes.
        let data = null;
        const gateways = [
            `https://gateway.pinata.cloud/ipfs/${cid}`,
            `https://ipfs.io/ipfs/${cid}`,
        ];
        for (const g of gateways) {
            try {
                const r = await axios.get(g, { responseType: 'arraybuffer', timeout: 45000 });
                data = Buffer.from(r.data);
                console.log(`          fetched ${(data.length / 1024).toFixed(0)} KB from ${new URL(g).host}`);
                break;
            } catch (e) { /* try the next gateway */ }
        }
        assert.ok(data, 'no gateway served the file (propagation can lag a few seconds)');
        const back = crypto.createHash('sha256').update(data).digest('hex');
        assert.strictEqual(back, serverHash,
            'bytes fetched from IPFS do not match the hash we computed');
    });

    let video;
    await t('Video record stores the server-computed hash and fingerprint', async () => {
        video = await Video.create({
            cid,
            storageType: 'ipfs',
            videoFileHash: serverHash,
            hashComputedBy: 'server',
            farmerWallet: farmer._id,
            crop: 'Potato',
            location: 'Live Test',
            purpose: 'funding',
            fingerprint: analysis.fingerprint,
            provenance: analysis.provenance,
        });
        assert.strictEqual(video.hashComputedBy, 'server');
        assert.strictEqual(video.videoFileHash, serverHash);
    });

    await t('a re-encoded copy of the pinned video is detected', async () => {
        ff(['-i', P('live.mp4'), '-b:v', '90k', '-pix_fmt', 'yuv420p', P('copy.mp4')]);
        const a2 = await provenance.analyse(P('copy.mp4'), { farmerId: farmer._id });
        assert.ok(a2.provenance.flags.includes('duplicate_of_own_video'),
            `not detected: ${JSON.stringify(a2.provenance.flags)}`);
    });

    let batch;
    await t('REAL Bitcoin anchor: batch stamped via OpenTimestamps', async () => {
        batch = await anchor.anchorPending();
        assert.ok(batch, 'no batch produced');
        assert.ok(/^[0-9a-f]{64}$/.test(batch.merkleRoot));
        assert.ok(batch.otsProof, 'no OTS proof stored — the stamp did not happen');
        assert.strictEqual(batch.status, 'pending',
            `expected pending (awaiting a block), got ${batch.status}: ${batch.lastError || ''}`);
        console.log(`          root ${batch.merkleRoot.slice(0, 24)}…  proof ${batch.otsProof.length} b64 chars`);
    });

    await t('inclusion proof verifies independently', async () => {
        const proof = await anchor.proofFor(cid);
        assert.ok(proof, 'no proof for the anchored video');
        assert.strictEqual(proof.selfCheck, true);
        // Re-verify from scratch, the way an outsider would.
        assert.ok(merkle.verifyRecord(proof.record, proof.inclusionProof,
            Buffer.from(proof.merkleRoot, 'hex')), 'independent verification failed');
        assert.ok(proof.record.includes(cid), 'the record does not name this CID');
        assert.ok(proof.record.includes(serverHash), 'the record does not carry our hash');
    });

    await t('the OTS proof deserialises and commits to our root', async () => {
        const OTS = require('opentimestamps');
        const dtf = OTS.DetachedTimestampFile.deserialize(
            Buffer.from(batch.otsProof, 'base64'));
        const digest = Buffer.from(dtf.fileDigest()).toString('hex');
        assert.strictEqual(digest, batch.merkleRoot,
            'the OTS proof commits to a different value than our Merkle root');
    });

    await t('public description does not overclaim', async () => {
        const d = provenance.describe(video.toObject());
        const nv = d.notVerified.join(' ').toLowerCase();
        assert.ok(nv.includes('land'), 'must disclaim showing the farmer\'s land');
        const reuse = d.checksCompleted.find((c) => /re-use/i.test(c.check));
        assert.strictEqual(reuse.state, 'completed');
    });

    await unpinAll();
    await mongoose.disconnect();
    await mem.stop();
    fs.rmSync(tmp, { recursive: true, force: true });
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch(async (e) => {
    console.error('harness error:', e.message);
    await unpinAll();
    try { await mongoose.disconnect(); } catch (_) {}
    process.exit(1);
});
