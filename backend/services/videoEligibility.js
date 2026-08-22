/**
 * Whether a video may back a money claim.
 *
 * THE BYPASS THIS CLOSES
 *   Every provenance control lives on POST /api/videos/upload: the server hashes
 *   the bytes it received, fingerprints them, and checks them against every other
 *   video. But the legacy POST /api/videos accepts metadata only -- a CID and some
 *   text -- and creates a perfectly valid Video document with no fingerprint and
 *   hashComputedBy 'unverified'.
 *
 *   Both the funding-request and listing routes then validated only that the CID
 *   existed and belonged to the caller. So the whole pipeline could be skipped by
 *   using a different URL:
 *
 *     POST /api/videos            { cid: <anything>, crop, location, purpose }
 *     POST /api/funding-requests  { cid: <same>, amount: 200000, ... }
 *
 *   The result is a funding request whose "evidence" was never hashed by us, never
 *   fingerprinted, and may not even exist on IPFS -- while the UI shows the same
 *   integrity language as a properly uploaded video. That is the worst kind of
 *   hole: not a crash, but a quiet downgrade to no verification at all.
 *
 * WHY ONE SHARED FUNCTION
 *   The rule was duplicated as two partial ownership checks in two routers and
 *   they had already drifted. Anything that can back a money claim now asks the
 *   same question here, so a future money-taking route cannot forget a clause.
 *
 * WHY THIS BLOCKS RATHER THAN FLAGS
 *   Provenance SIGNALS only ever flag, because they are machine judgements about
 *   footage and a false accusation costs a smallholder a season. This is not a
 *   judgement about footage. It asks whether the video went through the pipeline
 *   at all, which is a fact about our own records with no false-positive risk. The
 *   farmer's remedy is immediate and in their hands: upload the video properly.
 */

const mongoose = require('mongoose');

/**
 * @returns {{ok: true, video}} or {ok: false, status, message, code}
 */
async function requireFundableVideo(cid, farmerId, { purpose } = {}) {
    if (!cid || typeof cid !== 'string' || !cid.trim()) {
        return { ok: false, status: 400, code: 'cid_missing', message: 'A video CID is required.' };
    }

    const Video = mongoose.model('Video');
    const video = await Video.findOne({ cid: cid.trim(), farmerWallet: farmerId });

    if (!video) {
        return {
            ok: false, status: 404, code: 'video_not_found',
            message: 'Associated video not found or does not belong to you.',
        };
    }

    // The load-bearing check. 'unverified' means the bytes never passed through
    // this server, so nothing about the file has been established.
    if (video.hashComputedBy !== 'server' || !video.videoFileHash) {
        return {
            ok: false, status: 409, code: 'video_not_verified',
            message: 'This video was recorded before we could verify it, or was added '
                   + 'as metadata only. Please upload the video file again from the '
                   + 'Record page so we can check it, then create your request.',
        };
    }

    // No fingerprint means the re-use check never ran, so this video could be a
    // copy of someone else's and we would not know.
    const fp = video.fingerprint;
    if (!fp || !Array.isArray(fp.frameHashes) || fp.frameHashes.length < 4) {
        return {
            ok: false, status: 409, code: 'video_not_analysed',
            message: 'We could not analyse this video, so it cannot be used to raise '
                   + 'funds yet. Please re-upload it -- if it fails again, the file '
                   + 'may be in a format we cannot read and support can help.',
        };
    }

    // Purpose is what the farmer declared at upload time. Enforcing it stops one
    // video being quietly repurposed from a public showcase into loan evidence.
    if (purpose && video.purpose && video.purpose !== purpose) {
        return {
            ok: false, status: 409, code: 'video_wrong_purpose',
            message: `This video was uploaded for "${video.purpose}", not "${purpose}". `
                   + 'Upload a video for this purpose instead.',
        };
    }

    return { ok: true, video };
}

module.exports = { requireFundableVideo };
