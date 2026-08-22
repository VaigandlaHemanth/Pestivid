/**
 * Fingerprints of deleted or withdrawn videos, retained so the duplicate index
 * cannot be emptied by deletion.
 *
 * WHY THIS EXISTS — it closes a free attack.
 *
 *   DELETE /api/videos/:id removed the Video document, and with it the
 *   fingerprint. That made upload-then-delete an unlimited, zero-cost oracle:
 *   upload a laundered version of a stolen clip, read whether it was flagged,
 *   delete it, adjust the laundering, repeat. An attacker could tune footage
 *   against the detector until it passed, leaving no trace, and the successful
 *   version would then be the only copy in the index.
 *
 *   Retaining the fingerprint means each probe permanently costs the attacker a
 *   detectable artefact, and a deleted video can still be recognised if it comes
 *   back under another account.
 *
 * WHAT IS DELIBERATELY NOT KEPT
 *   No CID, no file hash, no location, no crop, no farmer name. Only the
 *   perceptual hashes, a hashed farmer reference, and timestamps. A farmer who
 *   deletes a video is entitled to have it gone; what remains is the minimum
 *   needed to recognise the same FOOTAGE later, not a shadow copy of the record.
 *   The farmer reference is a one-way HMAC so this collection cannot be used to
 *   enumerate who uploaded what.
 */

const mongoose = require('mongoose');

const retiredFingerprintSchema = new mongoose.Schema({
    // Perceptual hashes only. These identify footage, not a person or a file.
    frameHashes: { type: [String], required: true },
    nFrames: Number,
    durationSeconds: Number,
    algorithm: String,

    // HMAC of the farmer id, not the id itself. Lets us tell "same uploader
    // returning" from "different uploader" without storing who.
    farmerRef: { type: String, index: true },

    retiredAt: { type: Date, default: Date.now, index: true },
    reason: {
        type: String,
        enum: ['deleted_by_farmer', 'withdrawn', 'rejected'],
        default: 'deleted_by_farmer',
    },
}, { timestamps: true });

module.exports = mongoose.models.RetiredFingerprint
    || mongoose.model('RetiredFingerprint', retiredFingerprintSchema);
