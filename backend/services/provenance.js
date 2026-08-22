/**
 * Provenance checks run at upload time.
 *
 * WHAT THIS IS FOR
 *   Hashing and Bitcoin anchoring prove that bytes existed by a certain time and
 *   have not changed since. They say nothing about whether the farmer shot the
 *   video. The frauds that actually cost investors money are:
 *
 *     F1 recycling     re-uploading last season's clip, or footage of a healthy
 *                      plot to raise money for a failing one
 *     F2 theft         uploading another farmer's video
 *     F3 stock footage uploading something downloaded from the internet
 *
 *   Perceptual fingerprinting catches F1 and F2 well. Nothing cheap catches F3
 *   without a reference corpus, and this module does not pretend otherwise.
 *
 * FLAG, NEVER BLOCK
 *   Every check here returns flags for a human to review. It cannot reject an
 *   upload on its own, and that is deliberate: two neighbouring plots of the same
 *   crop, or the same field photographed in two seasons, can legitimately look
 *   alike, and a wrongly blocked upload can cost a smallholder a season's
 *   funding. The asymmetry of harm points one way.
 *
 * MISSING SIGNAL IS NOT SUSPICION
 *   No location permission, no network during recording, an old phone that
 *   produces an unreadable container -- these produce an ABSENT signal, not a
 *   negative one. Farmers on poor connections must not be treated as fraudsters
 *   for it.
 */

const mongoose = require('mongoose');

const fingerprintSvc = require('./videoFingerprint');

// Fraction of sampled frames that must match before we call it the same footage.
// The measured separation is stark -- a re-encoded copy matches 1.00 and
// unrelated footage 0.00 -- so a high threshold costs no detection power while
// leaving a wide margin against false accusations.
const SIMILARITY_FLAG = Number(process.env.PROVENANCE_SIMILARITY || 0.6);

// Compare against EVERY fingerprinted video, not a recency window. A 5,000-video
// window would let an attacker age a stolen clip out of the index simply by
// waiting, which is a free bypass that costs them nothing.
const COMPARE_LIMIT = Number(process.env.PROVENANCE_COMPARE_LIMIT || 0);

// Absolute number of matching frames that counts as the same source footage
// regardless of how much filler surrounds it. At 1 fps sampling, 6 matched frames
// is six seconds of identical footage, which does not happen by chance.
const MIN_MATCHED_FRAMES = Number(process.env.PROVENANCE_MIN_FRAMES || 6);

/**
 * Fingerprint a video and look for existing footage that matches it.
 *
 * Returns { fingerprint, provenance } ready to store on the Video document.
 * Never throws for provenance reasons: if fingerprinting fails the upload still
 * proceeds with a 'fingerprint_unavailable' flag, because refusing an upload
 * because OUR analysis broke would punish the farmer for our bug.
 */
async function analyse(videoPath, { farmerId, reportedLocation } = {}) {
    const flags = [];
    let fingerprint;

    try {
        fingerprint = await fingerprintSvc.fingerprint(videoPath);
    } catch (err) {
        // A container we cannot decode is a compatibility problem, not fraud.
        return {
            fingerprint: undefined,
            provenance: {
                flags: ['fingerprint_unavailable'],
                reviewState: 'none',
                ...locationFields(reportedLocation),
            },
            note: `Fingerprinting failed: ${err.message}`,
        };
    }

    if (fingerprint.nFrames < 4) {
        // Too few frames to say anything. Very short clips are also low-value
        // evidence for an investor, which is worth surfacing.
        flags.push('too_few_frames');
    }

    const Video = mongoose.model('Video');
    const candidates = await Video.find(
        { 'fingerprint.frameHashes': { $exists: true, $ne: [] } },
        { fingerprint: 1, farmerWallet: 1, cid: 1, uploadTimestamp: 1 },
    ).sort({ uploadTimestamp: -1 }).limit(COMPARE_LIMIT || 0).lean();

    // SCORE CHOICE: matched-frame COUNT relative to the smaller fingerprint, but
    // floored by an absolute count, because a fraction alone is diluted by
    // padding. Measured: appending 30 s of unrelated filler to a stolen clip drops
    // matchedFraction from 1.000 to 0.583 while the theft is unchanged. Counting
    // matched frames means padding cannot hide a match, it can only add
    // non-matching frames alongside it.
    // Deleted videos still count. Otherwise deletion empties the index and the
    // detector can be probed for free -- see models/RetiredFingerprint.js.
    let retired = [];
    try {
        retired = await mongoose.model('RetiredFingerprint').find(
            {}, { frameHashes: 1, nFrames: 1 }).limit(COMPARE_LIMIT || 0).lean();
    } catch (e) {
        // Model not registered (e.g. a narrow unit test). Not fatal.
    }

    let best = null;
    for (const r of retired) {
        const cmp = fingerprintSvc.compare(fingerprint, { frameHashes: r.frameHashes });
        if (!cmp.comparable) continue;
        const score = Math.max(cmp.matchedFraction,
            Math.min(1, cmp.matchedFrames / Math.max(MIN_MATCHED_FRAMES, 1)));
        if (score >= SIMILARITY_FLAG && !flags.includes('matches_deleted_upload')) {
            flags.push('matches_deleted_upload');
        }
    }
    for (const c of candidates) {
        const cmp = fingerprintSvc.compare(fingerprint, c.fingerprint);
        if (!cmp.comparable) continue;
        const absolute = cmp.matchedFrames / Math.max(MIN_MATCHED_FRAMES, 1);
        const score = Math.max(cmp.matchedFraction, Math.min(1, absolute));
        if (!best || score > best.score) best = { cmp, video: c, score };
    }

    const provenance = {
        flags,
        reviewState: 'none',
        ...locationFields(reportedLocation),
    };

    if (best && best.score >= SIMILARITY_FLAG) {
        const sameFarmer = String(best.video.farmerWallet) === String(farmerId);
        provenance.similarTo = best.video._id;
        provenance.similarToFarmer = best.video.farmerWallet;
        provenance.similarityFraction = Number(best.score.toFixed(4));
        provenance.similarityKind = sameFarmer ? 'same_farmer' : 'other_farmer';
        flags.push(sameFarmer ? 'duplicate_of_own_video' : 'matches_another_upload');
        provenance.reviewState = 'flagged';
    }

    return { fingerprint, provenance, matchedVideoId: best && best.score >= SIMILARITY_FLAG
        ? best.video._id : null };
}

/** Location is stored as reported and is never treated as verified. */
function locationFields(loc) {
    if (!loc || typeof loc.latitude !== 'number' || typeof loc.longitude !== 'number') {
        return {};
    }
    if (Math.abs(loc.latitude) > 90 || Math.abs(loc.longitude) > 180) {
        return {};
    }
    return {
        reportedLatitude: loc.latitude,
        reportedLongitude: loc.longitude,
        reportedAccuracyMetres: typeof loc.accuracy === 'number' ? loc.accuracy : undefined,
    };
}

/**
 * Human-readable statement of what is and is not established, for the API and
 * the UI. Written so it cannot be mistaken for a verification badge.
 */
function describe(video) {
    const p = (video && video.provenance) || {};
    const verified = [];
    const notVerified = [
        "That the video shows this farmer's land",
        'That the video was recorded recently',
        'The reported location (device location can be altered)',
    ];

    if (video && video.hashComputedBy === 'server' && video.videoFileHash) {
        verified.push('The file has not been altered since upload (SHA-256 computed by our server)');
    }
    if (video && video.cid) {
        verified.push('The file matches its IPFS address, which anyone can re-check independently');
    }

    // COVERAGE, NOT JUST FINDINGS.
    //
    // Reporting only findings makes "nothing to report" indistinguishable from
    // "we never checked", and engineering that absence is precisely the
    // attacker's strategy: upload a container we cannot decode and the reuse
    // check silently never runs. So each check reports whether it COMPLETED,
    // separately from whether it found anything.
    const fp = (video && video.fingerprint) || {};
    // Read nFrames, NOT frameHashes.length. Callers deliberately do not select the
    // frame hashes -- publishing them would let a fraudster tune footage against
    // the detector -- so depending on the array here made the endpoint report
    // "could not be decoded" for videos that decoded perfectly well. Reporting a
    // completed check as skipped is the exact failure this section exists to stop.
    const nFrames = Number(fp.nFrames || (Array.isArray(fp.frameHashes) ? fp.frameHashes.length : 0));
    const framesOk = nFrames >= 4;
    const flags = p.flags || [];

    const checksCompleted = [
        {
            check: 'File unchanged since upload',
            state: (video && video.hashComputedBy === 'server' && video.videoFileHash)
                ? 'completed' : 'not_available',
            detail: (video && video.hashComputedBy === 'server')
                ? 'SHA-256 computed by our server from the received bytes'
                : 'This record has no server-computed hash',
        },
        (() => {
            // Partial coverage is NOT a pass. If we only sampled the first slice
            // of a video, the rest was never compared, and reporting that as
            // "completed" would let a fraudster hide footage past the sampled
            // range and still show a clean result.
            const dur = Number(fp.durationSeconds || 0);
            const cov = Number(fp.coveredSeconds || 0);
            const partial = framesOk && dur > 0 && cov > 0 && cov < 0.9 * dur;
            return {
                check: 'Re-use and duplicate check',
                state: !framesOk
                    ? (flags.includes('fingerprint_unavailable') ? 'could_not_run' : 'not_available')
                    : (partial ? 'partial' : 'completed'),
                detail: !framesOk
                    ? 'The video could not be decoded, so this check did not run. '
                      + 'That is not a finding against the farmer.'
                    : (partial
                        ? `Only ${Math.round(cov)}s of ${Math.round(dur)}s was sampled, `
                          + 'so part of this video was not compared.'
                        : `${nFrames} frames compared against every other video`
                          + (dur ? `, covering ${Math.round(cov || dur)}s of ${Math.round(dur)}s` : '')),
            };
        })(),
        {
            check: 'Independent timestamp',
            state: 'see_anchor_endpoint',
            detail: 'Bitcoin anchoring is reported separately, per video.',
        },
    ];

    return {
        verified,
        notVerified,
        checksCompleted,
        // flags and reviewState are returned for INTERNAL callers only. The public
        // endpoint must not forward them: there is no reviewer role and no way for
        // a farmer to answer an automated signal, so publishing one would leave an
        // unadjudicated accusation on a real person's record permanently.
        flags,
        reviewState: p.reviewState || 'none',
        disclaimer:
            'Integrity checks show the file has not been changed since we received it. ' +
            "They do not confirm where or when it was filmed, or that it shows " +
            "this farmer's land. Treat the video as one input among several " +
            "before committing money.",
    };
}

module.exports = { analyse, describe, SIMILARITY_FLAG };
