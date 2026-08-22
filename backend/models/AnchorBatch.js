/**
 * One anchored batch of video records.
 *
 * A batch holds the Merkle root over every video record accumulated since the
 * last anchor, plus the OpenTimestamps proof for that root. One anchor covers
 * unlimited videos, which is what makes Bitcoin-grade timestamping free here.
 *
 * The leaf list is stored so an inclusion proof can be regenerated on demand
 * rather than kept per-video. Recomputing a proof from ~1000 leaves is
 * microseconds, and storing the list once is far smaller than storing a 10-step
 * proof against every video.
 */

const mongoose = require('mongoose');

const anchorBatchSchema = new mongoose.Schema({
    merkleRoot: {
        type: String,          // hex
        required: true,
        unique: true,
        index: true,
    },

    // Ordered leaf hashes, hex. Order is part of the root's definition, so this
    // array must never be re-sorted or de-duplicated after the root is computed.
    leaves: {
        type: [String],
        required: true,
    },

    // The video records covered, in the same order as `leaves`.
    videos: [{
        video: { type: mongoose.Schema.Types.ObjectId, ref: 'Video' },
        cid: String,
        // The exact canonical string that was hashed. Stored verbatim because a
        // verifier must be able to reproduce the leaf byte-for-byte, and any
        // later change to how we serialise records would otherwise silently
        // invalidate every historical proof.
        canonicalRecord: String,
    }],

    // OpenTimestamps detached proof for merkleRoot, base64. Small (under ~1 KB).
    otsProof: String,

    // 'pending'  submitted to calendar servers, not yet in a Bitcoin block
    // 'anchored' upgraded and confirmed against the Bitcoin chain
    // 'failed'   calendars could not be reached; retry later
    //
    // 'pending' is the normal state for the first few hours. OpenTimestamps
    // batches submissions and waits for a block, so same-day uploads are
    // legitimately not yet Bitcoin-confirmed and the UI must say so rather than
    // implying they are.
    status: {
        type: String,
        enum: ['pending', 'anchored', 'failed'],
        default: 'pending',
        index: true,
    },

    bitcoinBlockHeight: Number,
    bitcoinTimestamp: Date,

    stampedAt: { type: Date, default: Date.now },
    upgradedAt: Date,
    lastError: String,
}, { timestamps: true });

// Look-ups are "which batch contains this video".
anchorBatchSchema.index({ 'videos.video': 1 });
anchorBatchSchema.index({ 'videos.cid': 1 });

module.exports = mongoose.models.AnchorBatch
    || mongoose.model('AnchorBatch', anchorBatchSchema);
