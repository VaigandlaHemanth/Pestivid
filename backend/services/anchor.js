/**
 * Bitcoin anchoring of video records, via OpenTimestamps.
 *
 * WHAT THIS PROVES
 *   That a specific video record — its CID, its server-computed SHA-256, the
 *   farmer, and the upload time — existed no later than a particular Bitcoin
 *   block. Nobody, including us, can backdate a record or alter one after the
 *   fact without invalidating the proof.
 *
 * WHAT IT DOES NOT PROVE
 *   Anything about what the video shows, where it was filmed, or when it was
 *   filmed. It timestamps our RECEIPT of the file. Provenance is a separate
 *   problem handled in services/provenance.js, and the two must never be
 *   presented as one guarantee.
 *
 * WHY OPENTIMESTAMPS AND NOT A SMART CONTRACT
 *   Free, permanently, with no wallet, no private key on the server, no gas and
 *   no faucet to keep topped up. The alternative considered was a Polygon Amoy
 *   testnet contract, which is free but anchors to a chain that is not
 *   economically secured and can be reset — it produces an explorer link that
 *   looks like proof while proving nothing durable. Polygon mainnet would be real
 *   but costs money per anchor and requires a funded key in the environment.
 *
 *   Measured cost of an OTS stamp: ~2.6 s, 805-byte proof, four independent
 *   calendar servers.
 *
 * WHY BATCHED
 *   One Merkle root per batch, one anchor per root. A single daily anchor covers
 *   any number of videos, and each video still gets an independently checkable
 *   inclusion proof.
 *
 * ONLY SERVER-COMPUTED HASHES ARE ELIGIBLE
 *   A record whose hash came from the client is excluded. Anchoring a hash the
 *   uploader chose would attach a real Bitcoin proof to an unverified claim,
 *   which is worse than no proof because it invites people to rely on it.
 */

const mongoose = require('mongoose');

const merkle = require('./merkle');

let OTS = null;
function ots() {
    if (OTS === null) {
        try {
            OTS = require('opentimestamps');
        } catch (e) {
            OTS = false;
        }
    }
    return OTS;
}

/** Records that are eligible for anchoring and not yet in a batch. */
async function pendingVideos(limit = 5000) {
    const Video = mongoose.model('Video');
    const AnchorBatch = mongoose.model('AnchorBatch');

    const anchoredIds = await AnchorBatch.distinct('videos.video');
    return Video.find({
        _id: { $nin: anchoredIds },
        hashComputedBy: 'server',          // see the note above: not negotiable
        videoFileHash: { $exists: true, $ne: null },
        cid: { $exists: true, $ne: null },
    })
        .sort({ uploadTimestamp: 1 })      // oldest first: batch order is stable
        .limit(limit)
        .select('_id cid videoFileHash farmerWallet uploadTimestamp')
        .lean();
}

// Single-flight guard for anchorPending.
//
// Two overlapping calls -- a cron tick landing on top of a manual admin trigger,
// or two clicks -- each read the same pending set and each build a batch over it.
// The result was duplicate batches covering the same videos, wasted calendar
// submissions, and two competing proofs for one video. Callers that arrive while
// a run is in flight now await THAT run instead of starting a second one.
let inFlight = null;

// Hard ceiling on any call into the OpenTimestamps library.
//
// Two reasons this is not optional. First, the npm `opentimestamps` package was
// last published in 2022 and reaches the calendar servers through the deprecated
// `request` stack; `npm audit` reports 2 critical and 8 moderate advisories, all
// of them in that subtree, and there is no maintained alternative (the only other
// candidate, javascript-opentimestamps, is older still and adds web3). The
// exposure that actually matters is bn.js parsing attestation bytes from a remote
// calendar -- one advisory there is an infinite loop, and that input is remote.
//
// Second, anchorPending() is now single-flight: a call that never settles would
// hold `inFlight` forever and silently stop every future anchor run. A wedged
// calendar must not become a permanent outage.
//
// The timeout does not make the library safe. It bounds the blast radius to one
// skipped batch, which the next run retries.
const OTS_TIMEOUT_MS = Number(process.env.OTS_TIMEOUT_MS || 60_000);

function withTimeout(promise, label, ms = OTS_TIMEOUT_MS) {
    let timer;
    return Promise.race([
        Promise.resolve(promise).finally(() => clearTimeout(timer)),
        new Promise((_, reject) => {
            timer = setTimeout(
                () => reject(new Error(`${label} timed out after ${ms}ms`)),
                ms,
            );
        }),
    ]);
}

/**
 * Build a batch over all pending records, stamp its root, and store it.
 * Returns null when there is nothing to anchor.
 *
 * Concurrency: safe to call from anywhere. Within a process, overlapping calls
 * share one run (see `inFlight`). Across processes, the unique index on
 * merkleRoot rejects a byte-identical duplicate batch and we return the batch
 * that won instead of surfacing a duplicate-key error. A cross-process race
 * where a new upload lands between the two reads still produces two batches with
 * overlapping videos and different roots -- harmless (both proofs are valid, and
 * proofFor prefers the confirmed one) but worth knowing.
 */
async function anchorPending(opts = {}) {
    if (inFlight) return inFlight;
    inFlight = _anchorPending(opts).finally(() => { inFlight = null; });
    return inFlight;
}

async function _anchorPending({ limit = 5000 } = {}) {
    const AnchorBatch = mongoose.model('AnchorBatch');
    const videos = await pendingVideos(limit);
    if (!videos.length) return null;

    const records = videos.map((v) => ({
        video: v._id,
        cid: v.cid,
        canonicalRecord: merkle.canonicalRecord({
            cid: v.cid,
            sha256: v.videoFileHash,
            farmerId: v.farmerWallet,
            uploadedAt: v.uploadTimestamp,
        }),
    }));

    const leaves = records.map((r) => merkle.leafHash(r.canonicalRecord));
    const { root } = merkle.buildTree(leaves);
    const rootHex = root.toString('hex');

    const batch = new AnchorBatch({
        merkleRoot: rootHex,
        leaves: leaves.map((l) => l.toString('hex')),
        videos: records,
        status: 'pending',
    });

    const lib = ots();
    if (!lib) {
        // Still record the batch. The Merkle log is useful on its own — it makes
        // the record set append-only and auditable — and the root can be stamped
        // later once the dependency is available.
        batch.lastError = 'opentimestamps module not installed; root not stamped';
        batch.status = 'failed';
        return saveBatch(batch, AnchorBatch);
    }

    try {
        const dtf = lib.DetachedTimestampFile.fromHash(new lib.Ops.OpSHA256(), root);
        await withTimeout(lib.stamp(dtf), 'ots stamp');
        batch.otsProof = Buffer.from(dtf.serializeToBytes()).toString('base64');
        batch.status = 'pending';          // pending Bitcoin confirmation, normal
        batch.stampedAt = new Date();
    } catch (err) {
        batch.status = 'failed';
        batch.lastError = `stamp failed: ${err.message}`.slice(0, 500);
    }

    return saveBatch(batch, AnchorBatch);
}

/**
 * Save a batch, tolerating the one race the unique merkleRoot index can lose.
 *
 * Another process building the identical batch first is not an error -- it is the
 * index doing its job. Return the winner rather than throwing, so the caller
 * still gets a usable batch.
 */
async function saveBatch(batch, AnchorBatch) {
    try {
        await batch.save();
        return batch;
    } catch (err) {
        if (err && err.code === 11000) {
            const existing = await AnchorBatch.findOne({ merkleRoot: batch.merkleRoot });
            if (existing) {
                console.log(`Anchor batch ${batch.merkleRoot.slice(0, 12)} was `
                    + 'already created by another process; using it.');
                return existing;
            }
        }
        throw err;
    }
}

/**
 * Ask the calendars to upgrade pending proofs to full Bitcoin attestations.
 * Safe and cheap to call on a schedule; a proof that is not yet in a block simply
 * stays pending.
 */
async function upgradePending() {
    const AnchorBatch = mongoose.model('AnchorBatch');
    const lib = ots();
    if (!lib) return { upgraded: 0, checked: 0, reason: 'opentimestamps not installed' };

    const batches = await AnchorBatch.find({ status: 'pending', otsProof: { $ne: null } })
        .sort({ stampedAt: 1 }).limit(50);

    let upgraded = 0;
    for (const b of batches) {
        try {
            const dtf = lib.DetachedTimestampFile.deserialize(
                Buffer.from(b.otsProof, 'base64'));
            const changed = await withTimeout(lib.upgrade(dtf), 'ots upgrade');
            if (changed) {
                b.otsProof = Buffer.from(dtf.serializeToBytes()).toString('base64');
            }
            const info = await withTimeout(lib.verify(dtf), 'ots verify');
            // verify() returns attestation data keyed by chain once a block
            // includes it; an empty result means "not yet", not "invalid".
            const btc = info && (info.bitcoin || info.BITCOIN || info['bitcoin']);
            if (btc && (btc.timestamp || btc.height)) {
                b.status = 'anchored';
                b.upgradedAt = new Date();
                if (btc.height) b.bitcoinBlockHeight = btc.height;
                if (btc.timestamp) b.bitcoinTimestamp = new Date(btc.timestamp * 1000);
                upgraded++;
            }
            await b.save();
        } catch (err) {
            b.lastError = `upgrade failed: ${err.message}`.slice(0, 500);
            await b.save();
        }
    }
    return { upgraded, checked: batches.length };
}

/**
 * Everything an outside party needs to verify one video independently.
 * Returns null when the video has not been anchored yet.
 */
async function proofFor(videoIdOrCid) {
    const AnchorBatch = mongoose.model('AnchorBatch');
    const query = mongoose.isValidObjectId(videoIdOrCid)
        ? { 'videos.video': videoIdOrCid }
        : { 'videos.cid': videoIdOrCid };

    // A CID can legitimately appear in more than one batch: a batch whose OTS
    // submission failed leaves its videos eligible, so the next run re-anchors
    // them. findOne() would then return whichever document Mongo happened to hand
    // back first, which could be the FAILED one -- so the caller would be told a
    // video is unanchored when a good proof exists elsewhere. Prefer a confirmed
    // batch, then a pending one, then anything; and among equals, the earliest,
    // because the earliest confirmed timestamp is the strongest claim.
    const candidates = await AnchorBatch.find(query)
        .sort({ createdAt: 1 })
        .lean();
    if (candidates.length === 0) return null;

    const rank = (b) => ({ anchored: 0, pending: 1, failed: 2 }[b.status] ?? 3);
    const batch = candidates.slice().sort((a, b) => rank(a) - rank(b))[0];

    const index = batch.videos.findIndex((v) => (
        String(v.video) === String(videoIdOrCid) || v.cid === videoIdOrCid));
    if (index < 0) return null;

    const leaves = batch.leaves.map((h) => Buffer.from(h, 'hex'));
    const { levels, root } = merkle.buildTree(leaves);
    const path = merkle.inclusionProof(levels, index);
    const record = batch.videos[index].canonicalRecord;

    // Self-check before handing a proof out. A proof that does not verify locally
    // means the stored batch is inconsistent, and returning it would send someone
    // off to debug our bug.
    const ok = merkle.verifyRecord(record, path, root);

    // What "anchored" actually requires. The route used to report anchored:true
    // for any batch a video appeared in, including one whose OTS submission
    // FAILED -- so a video with no timestamp at all was described as timestamped,
    // permanently. Three things must hold, and all three are reported.
    const confirmed = batch.status === 'anchored' && ok && Boolean(batch.otsProof);

    return {
        confirmed,
        record,
        leafIndex: index,
        leafHash: merkle.leafHash(record).toString('hex'),
        inclusionProof: path,
        merkleRoot: batch.merkleRoot,
        selfCheck: ok,
        batchSize: batch.leaves.length,
        status: batch.status,
        otsProofBase64: batch.otsProof || null,
        bitcoinBlockHeight: batch.bitcoinBlockHeight || null,
        bitcoinTimestamp: batch.bitcoinTimestamp || null,
        stampedAt: batch.stampedAt,
        howToVerify: [
            '1. Recompute the leaf: sha256(0x00 || record) where record is the exact string above.',
            '2. Fold the inclusionProof steps in order: for side "left" compute',
            '   sha256(0x01 || sibling || acc), for "right" sha256(0x01 || acc || sibling).',
            '   The result must equal merkleRoot.',
            '3. Save otsProofBase64 (base64-decoded) as root.ots and run:',
            '   ots verify root.ots  --  it attests merkleRoot to a Bitcoin block.',
            '   Install with: pip install opentimestamps-client',
            '4. status "pending" means the stamp is submitted but not yet in a block;',
            '   OpenTimestamps batches these, so allow a few hours.',
            '5. status "failed" means the stamp was never accepted. Steps 1 and 2 still',
            '   prove this record is in that Merkle root, but nothing ties the root to',
            '   Bitcoin, so there is NO trustworthy time claim. Treat it as unanchored.',
        ],
    };
}

module.exports = { anchorPending, upgradePending, proofFor, pendingVideos };
