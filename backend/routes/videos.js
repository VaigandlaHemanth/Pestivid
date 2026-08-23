// --- Backend Routes: videos.js ---

const express = require('express');         // Import Express
const router = express.Router();

// Neutralise regex metacharacters: raw user input in $regex is a ReDoS vector.
const safeCrop = (str) => String(str).slice(0, 60).replace(/[^\w\s-]/g, '');

const mongoose = require('mongoose');       // Import Mongoose (needed for ObjectId validation)
const Video = mongoose.model('Video');      // Get the Video Mongoose model
// Needed to tell a farmer their video was flagged. Previously nothing in
// this file notified anyone, so the flag was invisible to its subject.
const Notification = mongoose.model('Notification');
const Listing = mongoose.model('Listing');  // Get Listing model to check if video is used
const FundingRequest = mongoose.model('FundingRequest'); // Get FundingRequest model to check if video is used
const { authenticateToken } = require('./auth'); // Import the authentication middleware
const ipfs = require('../services/ipfsUpload'); // Server-side pinning + hashing
const provenanceSvc = require('../services/provenance'); // Recycle/theft detection
const anchorSvc = require('../services/anchor'); // Merkle log + Bitcoin timestamping
const limits = require('../middleware/rateLimits'); // upload + public-read ceilings

/**
 * First name of a populated user ref, or a fallback.
 * `.name.split(' ')` on an unpopulated ref throws, and a listing endpoint must
 * not 500 because one farmer record is missing a name.
 */
function firstNameOf(userRef, fallback) {
    if (!userRef || typeof userRef !== 'object') return fallback || 'Unknown';
    const n = userRef.name;
    if (typeof n === 'string' && n.trim()) return n.trim().split(' ')[0];
    return userRef.displayIdentifier || fallback || 'Unknown';
}


// Helper function to generate a simulated transaction hash (can be used for video creation if needed)
// Although video creation itself isn't typically a "blockchain transaction" like a purchase or investment,
// linking metadata to a hash could simulate a chain link for the metadata itself.
const generateSimulatedTxHash = (prefix = 'sim_tx') => {
    return `${prefix}_${Date.now().toString(16)}${Math.random().toString(16).substring(2, 12)}`;
};


// --- Video Metadata Routes ---

// @route GET /api/videos
// @desc Get all videos metadata (can be filtered by query parameters)
// @access Public (Accessible to anyone, e.g., for a public AgriStream or browsing all videos)
// Query Params: ?farmerId=<userId>, ?purpose=<agristream|sell|funding>, ?crop=<cropName> etc.
router.get('/', authenticateToken, async (req, res) => {
    try {
        // Build filter object from query parameters
        const filter = {};
        if (req.query.farmerId) {
             // Validate farmerId format if provided
             if (!mongoose.Types.ObjectId.isValid(req.query.farmerId)) {
                 return res.status(400).json({ message: 'Invalid Farmer ID format.' });
             }
            filter.farmerWallet = req.query.farmerId;
        }

        // Whose videos are these? Anything that is not the caller's own is treated
        // as public browsing, which limits BOTH the rows and the fields returned.
        //
        // Before this, GET /api/videos?farmerId=<victim> returned every one of
        // another farmer's videos including their CIDs and file hashes to any
        // authenticated account -- so any signed-up user could enumerate a
        // competitor's private funding evidence and pull the files straight off an
        // IPFS gateway.
        const askingForOwn = req.query.farmerId
            && String(req.query.farmerId) === String(req.user._id);
        const ownScope = Boolean(askingForOwn);

        if (!ownScope) {
            // Public browsing sees only what the farmer published for showcase.
            // A funding or sale video is commercially sensitive until its listing
            // or request is live, and those have their own endpoints.
            filter.purpose = 'agristream';
        }
        if (req.query.purpose) {
             // Validate purpose against schema enum if strict
             if (!['agristream', 'sell', 'funding'].includes(req.query.purpose)) {
                 // Optionally return error or just ignore invalid purpose
                 // return res.status(400).json({ message: 'Invalid purpose filter.' });
             } else {
                 filter.purpose = req.query.purpose;
             }
        }
        if (req.query.crop) {
             filter.crop = { $regex: safeCrop(req.query.crop), $options: 'i' };
        }
        // Add other filters as needed (e.g., location, pesticideCompany)

        // Find video documents based on the filter
        // Populate the farmerWallet field to get the farmer's name (and other public profile info if selected)
        const videos = await Video.find(filter)
                                  .populate('farmerWallet', 'name role displayIdentifier') // Populate farmer's _id, name, role, displayIdentifier
                                  .sort({ uploadTimestamp: -1 }); // Sort by newest upload first

        // Map to format for frontend (e.g., ensure _id is string)
         const formattedVideos = videos.map(video => {
             const base = {
                 _id: video._id.toString(),
                 // The CID is needed even when browsing: it is how the player
                 // addresses the file, and an agristream video is published on
                 // purpose. It is a content address, not a secret.
                 cid: video.cid,
                 storageType: video.storageType,
                 farmerWallet: video.farmerWallet ? video.farmerWallet._id.toString() : null,
                 farmerName: firstNameOf(video.farmerWallet, 'Unknown Farmer'),
                 crop: video.crop,
                 pesticide: video.pesticide,
                 location: video.location,
                 pesticideCompany: video.pesticideCompany,
                 purpose: video.purpose,
                 uploadTimestamp: video.uploadTimestamp ? video.uploadTimestamp.toISOString() : null,
             };
             if (!ownScope) return base;

             // Own videos only: the farmer's "Select Video Evidence" dropdown
             // needs the hash, and the provenance state is theirs to see.
             return {
                 ...base,
                 videoFileHash: video.videoFileHash,
                 hashComputedBy: video.hashComputedBy,
                 fingerprinted: Boolean(video.fingerprint
                     && Array.isArray(video.fingerprint.frameHashes)
                     && video.fingerprint.frameHashes.length >= 4),
             };
         });


        // Send the list of video metadata
        res.json(formattedVideos); // Default status is 200 OK

    } catch (err) {
        console.error('GET /api/videos error:', err);
        res.status(500).json({ message: 'Server error fetching videos.' }); // 500 Internal Server Error
    }
});

// @route GET /api/videos/farmer/:farmerId
// @desc Get videos metadata uploaded by a specific farmer
// @access Private (Requires authentication. User should typically only fetch their own videos)
router.get('/farmer/:farmerId', authenticateToken, async (req, res) => {
    const farmerId = req.params.farmerId; // Get the farmer ID from the URL parameter

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(farmerId)) {
        return res.status(400).json({ message: 'Invalid Farmer ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the farmer whose videos are being requested.
    // Or allow an admin user to view any farmer's videos.
    if (req.user._id.toString() !== farmerId.toString()) {
        console.warn(`Authorization failed: User ${req.user._id} attempted to view videos for user ${farmerId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own videos." }); // 403 Forbidden
    }

    try {
        // Find video documents for the specified farmer
        // Populate farmerWallet to get farmer details if needed (though we just verified the ID matches req.user._id)
        const videos = await Video.find({ farmerWallet: farmerId })
                                  .populate('farmerWallet', 'name role displayIdentifier') // Populate farmer's _id, name, role
                                  .sort({ uploadTimestamp: -1 }); // Sort by newest upload first

        // Map to format for frontend
         const formattedVideos = videos.map(video => ({
             _id: video._id.toString(),
             // Restored: the farmer's own "Select Video Evidence" dropdown keys off
             // vid.cid (index.html:508-521) to attach a recording to a listing or a
             // funding request. Stripping it in Phase 2 silently broke that flow.
             // These are the farmer's own uploads behind authenticateToken, not
             // third-party paid content.
             cid: video.cid,
             storageType: video.storageType,
             videoFileHash: video.videoFileHash,
             farmerWallet: video.farmerWallet ? video.farmerWallet._id.toString() : null, // Send farmer user ID string
             farmerName: firstNameOf(video.farmerWallet, 'Unknown Farmer'),
             crop: video.crop,
             pesticide: video.pesticide,
             location: video.location,
             pesticideCompany: video.pesticideCompany,
             purpose: video.purpose,
             uploadTimestamp: video.uploadTimestamp ? video.uploadTimestamp.toISOString() : null, // Send timestamp as ISO string

             // A farmer's own list has to carry the two facts that decide
             // whether a video can back a funding request, or the app cannot
             // tell them why one is greyed out and has to guess -- which is how
             // it ended up inventing a stricter rule than the server's. Same
             // pair GET /videos already returns for own scope. The bar is here,
             // not the Bitcoin date: see services/videoEligibility.js.
             hashComputedBy: video.hashComputedBy,
             fingerprinted: Boolean(video.fingerprint
                 && Array.isArray(video.fingerprint.frameHashes)
                 && video.fingerprint.frameHashes.length >= 4),
         }));

        // Send the list of video metadata for this farmer
        res.json(formattedVideos); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/videos/farmer/${farmerId} error:`, err);
        res.status(500).json({ message: 'Server error fetching farmer videos.' }); // 500 Internal Server Error
    }
});


// @route POST /api/videos
// @desc Create a new video metadata entry in the database
// @access Private (Requires authentication. Must be a farmer.)
// NOTE: This endpoint typically runs after the actual video file has been successfully uploaded
// to decentralized storage (like Storj or IPFS) from the frontend or a separate upload service.
// The request body should contain the metadata and the storage identifier (CID/Key).
/**
 * POST /api/videos/upload
 *
 * Receives the video FILE, hashes it here, and pins it here. This is the only
 * path that can produce a trustworthy videoFileHash.
 *
 * It replaces a flow where the browser held the Pinata JWT and pinned directly,
 * which exposed the credential to every visitor and left `cid` and
 * `videoFileHash` as attacker-controlled strings in req.body. The blockchain
 * anchoring built on top of this would be meaningless without it: a proof over a
 * hash the uploader chose proves only that they chose it.
 *
 * multipart field: `video`. Metadata fields ride alongside as form fields.
 */
/**
 * GET /api/videos/:cid/provenance
 *
 * States exactly what the platform can and cannot stand behind for one video.
 * Public and unauthenticated on purpose: an investor deciding whether to commit
 * money should be able to check this without an account, and a claim nobody can
 * inspect is not a claim worth making.
 *
 * Deliberately omitted: the perceptual frame hashes. Publishing them would let a
 * fraudster test candidate footage against the duplicate detector until they
 * found something that slips under the threshold.
 */
router.get('/:cid/provenance', limits.publicReadLimiter, async (req, res) => {
    try {
        const video = await Video.findOne({ cid: String(req.params.cid).trim() })
            .select('cid videoFileHash hashComputedBy uploadTimestamp crop location ' +
                    'provenance fingerprint.nFrames fingerprint.algorithm farmerWallet')
            .populate('farmerWallet', 'name')
            .lean();

        if (!video) {
            return res.status(404).json({ message: 'No video with that CID.' });
        }

        const described = provenanceSvc.describe(video);

        return res.json({
            cid: video.cid,
            uploadedAt: video.uploadTimestamp,
            crop: video.crop,
            farmer: video.farmerWallet && video.farmerWallet.name,
            integrity: {
                sha256: video.hashComputedBy === 'server' ? video.videoFileHash : null,
                computedBy: video.hashComputedBy,
                // Anyone can re-run this without trusting us.
                howToCheck: video.hashComputedBy === 'server'
                    ? 'Download the file from any IPFS gateway and run: sha256sum <file>. ' +
                      'It must equal the sha256 above. The CID is itself a content ' +
                      'address, so a mismatch means the file was swapped.'
                    : 'This record predates server-side hashing, so no integrity hash ' +
                      'can be offered for it.',
            },
            verified: described.verified,
            notVerified: described.notVerified,
            // reviewState and raw flags are NOT published here, and that is a
            // deliberate reversal of the first version of this endpoint.
            //
            // There is no reviewer role in this system (see models/User.js: 'admin'
            // was removed on purpose) and no route that writes 'cleared'. So a flag
            // is an unadjudicated machine signal that would sit on a real farmer's
            // public record forever with no way to answer it. A dHash score is not
            // a publishable fact about a person.
            //
            // What IS published is whether the checks completed, so that "nothing
            // reported" cannot be mistaken for "checked and clean" — absence is
            // exactly what an attacker engineers for.
            checksCompleted: described.checksCompleted,
            reportedLocation: (video.provenance && video.provenance.reportedLatitude != null)
                ? {
                    latitude: video.provenance.reportedLatitude,
                    longitude: video.provenance.reportedLongitude,
                    accuracyMetres: video.provenance.reportedAccuracyMetres,
                    status: 'farmer-reported, not verified',
                }
                : null,
            disclaimer: described.disclaimer,
        });
    } catch (err) {
        console.error('Provenance lookup failed:', err.message);
        return res.status(500).json({ message: 'Could not load provenance for this video.' });
    }
});

/**
 * GET /api/videos/:cid/anchor
 *
 * The Bitcoin timestamp proof for one video, in a form anyone can check without
 * trusting this server and without seeing any other farmer's data.
 *
 * Public and unauthenticated on purpose: a proof only has value if the person
 * relying on it can verify it themselves.
 *
 * What it establishes: this exact record — CID, server-computed SHA-256, farmer
 * and upload time — existed no later than the anchored Bitcoin block, and has not
 * been altered since. What it does NOT establish: anything about what the video
 * shows or when it was filmed. See /provenance for that distinction.
 */
router.get('/:cid/anchor', limits.publicReadLimiter, async (req, res) => {
    try {
        const cid = String(req.params.cid).trim();
        const proof = await anchorSvc.proofFor(cid);

        if (!proof) {
            const video = await Video.findOne({ cid }).select('cid hashComputedBy').lean();
            if (!video) {
                return res.status(404).json({ message: 'No video with that CID.' });
            }
            // Not anchored yet is the normal state for a recent upload, and for
            // older records that were never eligible. Say which, so nobody reads
            // "no proof" as "failed verification".
            const eligible = video.hashComputedBy === 'server';
            return res.status(200).json({
                cid,
                anchored: false,
                eligible,
                reason: eligible
                    ? 'This video has not been included in an anchor batch yet. '
                      + 'Batches are anchored periodically; check back later.'
                    : 'This record has no server-computed hash, so it cannot be '
                      + 'anchored. Only videos uploaded through the server-side '
                      + 'upload path are eligible.',
            });
        }

        // `anchored` must mean "there is a Bitcoin timestamp you can check", not
        // merely "a batch exists". A batch whose OTS submission failed used to be
        // reported as anchored:true forever, which is the worst kind of wrong here
        // -- a confident claim with nothing behind it. anchor.proofFor() now
        // computes `confirmed` from status + self-check + presence of a proof.
        const { confirmed, ...rest } = proof;
        return res.json({
            cid,
            anchored: confirmed,
            ...rest,
            ...(confirmed ? {} : {
                reason: proof.status === 'failed'
                    ? 'This record is in a Merkle batch, but the timestamp submission '
                      + 'failed, so nothing ties it to Bitcoin. The inclusion proof '
                      + 'below is still valid; the time claim is not. It will be '
                      + 're-anchored on the next run.'
                    : proof.status === 'pending'
                        ? 'Submitted to the OpenTimestamps calendars and waiting for a '
                          + 'Bitcoin block. This normally takes a few hours.'
                        : 'The stored batch failed its own inclusion self-check, which '
                          + 'means our records are inconsistent. Please report this CID.',
            }),
        });
    } catch (err) {
        console.error('Anchor lookup failed:', err.message);
        return res.status(500).json({ message: 'Could not load the anchor proof.' });
    }
});

/**
 * Provenance review.
 *
 * services/provenance.js sets reviewState='flagged' when a newly uploaded video's
 * perceptual fingerprint closely matches one already on the platform. That was
 * the end of it: nothing listed flagged videos, nothing could record a decision,
 * and the 'cleared'/'rejected' states in the schema were unreachable. So the
 * detection was real and the response to it did not exist.
 *
 * Two routes, admin only:
 *   GET  /api/videos/review-queue     what is waiting
 *   POST /api/videos/:cid/review      record a decision
 *
 * What these deliberately do NOT do: delete a video, block a farmer, or change
 * funding eligibility. A fingerprint match is a similarity heuristic -- a farmer
 * legitimately re-filming the same field on the same day can trip it -- and the
 * cost of a false positive is somebody's funding round. The decision is recorded
 * for a human to act on, and acting on it stays a separate, deliberate step.
 *
 * Still missing, and worth being explicit about rather than pretending otherwise:
 * there is no farmer-facing notice when their video is flagged, no appeal route,
 * and no stated turnaround time. Those are the parts that make review fair to the
 * person being reviewed, and they are product work rather than a bug fix.
 */
function requireAdmin(req, res, next) {
    // req.user.role comes from the database via authenticateToken, not from the
    // token's payload, so a stale token cannot carry a role the user no longer
    // has -- which matters most for exactly this check.
    if (!req.user || req.user.role !== 'admin') {
        console.warn(`Forbidden: non-admin ${req.user && req.user._id} `
                   + `tried to reach a review route.`);
        return res.status(403).json({
            message: 'This is only available to platform reviewers.',
            code: 'admin_only',
        });
    }
    return next();
}

router.get('/review-queue', authenticateToken, requireAdmin, async (req, res) => {
    try {
        const state = ['flagged', 'cleared', 'rejected'].includes(req.query.state)
            ? req.query.state
            : 'flagged';
        const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 50, 1), 200);

        const videos = await Video.find({ 'provenance.reviewState': state })
            // Oldest first: a farmer waiting on a review should not be overtaken
            // by newer uploads.
            .sort({ uploadTimestamp: 1 })
            .limit(limit)
            .select('cid crop location purpose uploadTimestamp farmerWallet '
                  + 'provenance fingerprint.nFrames fingerprint.coveredSeconds '
                  + 'videoFileHash hashComputedBy')
            // The matched video's CID, so the reviewer can pull up both.
            .populate('provenance.similarTo', 'cid crop uploadTimestamp')
            .populate('farmerWallet', 'name displayIdentifier memberSince')
            .lean();

        const items = videos.map((v) => ({
            cid: v.cid,
            crop: v.crop,
            location: v.location,
            purpose: v.purpose,
            uploadedAt: v.uploadTimestamp,
            farmer: v.farmerWallet ? {
                _id: String(v.farmerWallet._id),
                name: v.farmerWallet.name,
                identifier: v.farmerWallet.displayIdentifier,
                memberSince: v.farmerWallet.memberSince,
            } : null,
            flags: (v.provenance && v.provenance.flags) || [],
            reviewState: (v.provenance && v.provenance.reviewState) || 'none',
            // The real field names on the schema are similarTo / similarityFraction
            // / similarityKind -- not matchedCid / matchedFraction, which do not
            // exist and would have serialised as null on every row, quietly
            // hiding the single most useful piece of context a reviewer has.
            similarTo: (v.provenance && v.provenance.similarTo) ? {
                cid: v.provenance.similarTo.cid,
                crop: v.provenance.similarTo.crop,
                uploadedAt: v.provenance.similarTo.uploadTimestamp,
            } : null,
            similarToFarmer: (v.provenance && v.provenance.similarToFarmer)
                ? String(v.provenance.similarToFarmer) : null,
            similarityFraction: (v.provenance && v.provenance.similarityFraction) != null
                ? v.provenance.similarityFraction : null,
            // 'same_farmer' (recycling their own footage) and 'other_farmer'
            // (using someone else's) are different problems and want different
            // decisions, so the reviewer must see which one this is.
            similarityKind: (v.provenance && v.provenance.similarityKind) || null,
            framesCompared: (v.fingerprint && v.fingerprint.nFrames) || 0,
            coveredSeconds: (v.fingerprint && v.fingerprint.coveredSeconds) != null
                ? v.fingerprint.coveredSeconds : null,
            // The farmer's own account of it. A reviewer deciding without reading
            // this is deciding on half the evidence, so it is part of the row
            // rather than something to fetch separately.
            appeal: (v.provenance && v.provenance.appeal
                     && v.provenance.appeal.statement) ? {
                statement: v.provenance.appeal.statement,
                submittedAt: v.provenance.appeal.submittedAt,
                revisions: v.provenance.appeal.revisions || 0,
            } : null,
            // Whether the farmer was ever told. A flag they do not know about
            // should stand out in the queue as a process failure.
            farmerNotifiedAt: (v.provenance && v.provenance.flagNotifiedAt) || null,
            reviewedBy: (v.provenance && v.provenance.reviewedBy)
                ? String(v.provenance.reviewedBy) : null,
            reviewedAt: (v.provenance && v.provenance.reviewedAt) || null,
            reviewNote: (v.provenance && v.provenance.reviewNote) || null,
        }));

        return res.json({
            state,
            count: items.length,
            // Say when the list is truncated, rather than letting a reviewer
            // believe they have seen everything.
            truncated: items.length === limit,
            items,
        });
    } catch (err) {
        console.error('Review queue failed:', err.message);
        return res.status(500).json({ message: 'Could not load the review queue.' });
    }
});

router.post('/:cid/review', authenticateToken, requireAdmin, async (req, res) => {
    const cid = String(req.params.cid || '').trim();
    const { decision, note } = req.body || {};

    if (!['cleared', 'rejected'].includes(decision)) {
        return res.status(400).json({
            message: "decision must be 'cleared' or 'rejected'.",
            code: 'bad_decision',
        });
    }
    // A rejection is the consequential one, so it has to be explained. An
    // unexplained rejection cannot be appealed or audited.
    if (decision === 'rejected' && (!note || String(note).trim().length < 10)) {
        return res.status(400).json({
            message: 'A rejection needs a note of at least 10 characters saying why.',
            code: 'note_required',
        });
    }

    try {
        // Only a FLAGGED video can be decided, and the filter enforces it
        // atomically so two reviewers cannot both record a first decision.
        const updated = await Video.findOneAndUpdate(
            { cid, 'provenance.reviewState': 'flagged' },
            {
                $set: {
                    'provenance.reviewState': decision,
                    'provenance.reviewedBy': req.user._id,
                    'provenance.reviewedAt': new Date(),
                    'provenance.reviewNote': note ? String(note).trim().slice(0, 2000) : undefined,
                },
            },
            { new: true },
        ).select('_id cid provenance').lean();

        if (!updated) {
            // Distinguish the three reasons, so a reviewer is not left guessing.
            const existing = await Video.findOne({ cid })
                .select('cid provenance.reviewState').lean();
            if (!existing) {
                return res.status(404).json({ message: 'No video with that CID.' });
            }
            const current = (existing.provenance && existing.provenance.reviewState) || 'none';
            if (current === 'none') {
                return res.status(409).json({
                    message: 'This video is not flagged, so there is nothing to decide.',
                    code: 'not_flagged',
                });
            }
            return res.status(409).json({
                message: `This video has already been reviewed (${current}).`,
                code: 'already_reviewed',
                reviewState: current,
            });
        }

        // Tell the farmer the outcome. A decision they are not told about is the
        // same secret the flag used to be.
        try {
            const owner = await Video.findOne({ cid }).select('farmerWallet crop').lean();
            if (owner && owner.farmerWallet) {
                await new Notification({
                    recipient: owner.farmerWallet,
                    type: decision === 'cleared' ? 'success' : 'warning',
                    message: decision === 'cleared'
                        ? `Your video for "${owner.crop}" has been checked and cleared. `
                          + 'No further action is needed.'
                        : `After review, your video for "${owner.crop}" cannot be used `
                          + `for funding. Reason: ${String(note).trim()} `
                          + 'If you think this is wrong, you can add an explanation from '
                          + 'the video\'s page and it will be looked at again.',
                    itemId: updated._id,
                    itemType: 'Video',
                    read: false,
                }).save();
            }
        } catch (noteErr) {
            console.error('REVIEW DECISION NOT NOTIFIED:', cid, noteErr.message);
        }

        console.log(`Provenance review: ${cid} -> ${decision} by ${req.user._id}`);
        return res.json({
            message: `Recorded as ${decision}.`,
            cid: updated.cid,
            reviewState: updated.provenance.reviewState,
            reviewedAt: updated.provenance.reviewedAt,
            // Be explicit that recording a decision is not enforcement.
            note: 'This records the decision. It does not delete the video, block '
                + 'the farmer, or change funding eligibility.',
        });
    } catch (err) {
        console.error('Review decision failed:', err.message);
        return res.status(500).json({ message: 'Could not record the decision.' });
    }
});


/**
 * POST /api/videos/:cid/appeal
 *
 * The farmer's side of a provenance flag. Owner only.
 *
 * The reviewer sees a dHash similarity score; only the farmer knows why two
 * clips look alike. "The first take was too dark so I filmed the row again" does
 * not appear in a frame comparison, and it is the usual explanation. Without this
 * route the review had exactly one side of the story, and the person affected had
 * no way to speak at all.
 *
 * Submitting an appeal does NOT change the review state. It attaches a statement
 * the reviewer must see. Deciding stays with the reviewer.
 */
router.post('/:cid/appeal', authenticateToken, async (req, res) => {
    const cid = String(req.params.cid || '').trim();
    const statement = req.body && req.body.statement;

    if (!statement || String(statement).trim().length < 10) {
        return res.status(400).json({
            message: 'Please describe what happened, in at least 10 characters.',
            code: 'statement_required',
        });
    }

    try {
        const video = await Video.findOne({ cid })
            .select('cid farmerWallet provenance crop').lean();
        if (!video) {
            return res.status(404).json({ message: 'No video with that CID.' });
        }
        // Ownership, not role: a farmer may only speak about their own video, and
        // nobody else may put words in their mouth.
        if (String(video.farmerWallet) !== String(req.user._id)) {
            return res.status(403).json({
                message: 'You can only respond about your own video.',
                code: 'not_your_video',
            });
        }

        const state = (video.provenance && video.provenance.reviewState) || 'none';
        if (state === 'none') {
            return res.status(409).json({
                message: 'This video is not flagged, so there is nothing to respond to.',
                code: 'not_flagged',
            });
        }
        if (state === 'cleared') {
            return res.status(409).json({
                message: 'This video has already been cleared. No response is needed.',
                code: 'already_cleared',
            });
        }

        // Deliberately allowed while 'rejected': a rejected farmer is precisely
        // the one who most needs to be able to respond. The statement lands on the
        // record and the video reappears in the reviewer's queue for that state.
        const prior = (video.provenance && video.provenance.appeal) || {};
        const updated = await Video.findOneAndUpdate(
            { cid, farmerWallet: req.user._id },
            {
                $set: {
                    'provenance.appeal.statement': String(statement).trim().slice(0, 2000),
                    'provenance.appeal.submittedAt': new Date(),
                    'provenance.appeal.revisions': (prior.revisions || 0) + (prior.submittedAt ? 1 : 0),
                },
            },
            { new: true },
        ).select('cid provenance.appeal provenance.reviewState').lean();

        console.log(`Appeal recorded for ${cid} by ${req.user._id} (state ${state}).`);
        return res.json({
            message: state === 'rejected'
                ? 'Thank you. Your explanation has been added and this video will be '
                  + 'looked at again.'
                : 'Thank you. Your explanation has been added and will be read before '
                  + 'a decision is made.',
            cid: updated.cid,
            reviewState: updated.provenance.reviewState,
            submittedAt: updated.provenance.appeal.submittedAt,
            // No invented SLA. This is the same figure the notice quotes.
            expectedReviewWithin: '2 working days',
        });
    } catch (err) {
        console.error('Appeal failed:', err.message);
        return res.status(500).json({ message: 'Could not record your response.' });
    }
});


// ── Direct-to-storage upload ────────────────────────────────────────────────
//
// POST /upload sends the file through this API, which works on a box we own and
// does not work on a free serverless host: the request body caps at 4.5 MB and
// a forty-second clip is about 10 MB. These two routes are the path that fits.
//
//   1. the phone asks for a one-use URL          POST /videos/upload-url
//   2. the phone posts the file straight to Pinata
//   3. the phone tells us the CID                POST /videos/confirm-upload
//   4. WE fetch the object back and hash it ourselves
//
// Step 4 is the whole point. Without it the hash would be a number the phone
// sent us, which is exactly the design this file was written to replace. The
// bytes we hash are the bytes that are actually stored, fetched by us.
router.post('/upload-url', authenticateToken, limits.uploadBurstLimiter, async (req, res) => {
    if (req.user.role !== 'farmer') {
        return res.status(403).json({ message: "Forbidden: Only users with the 'farmer' role can upload videos." });
    }
    if (!ipfs.pinataConfigured()) {
        return res.status(503).json({
            message: 'Video storage is not configured on the server. ' +
                     'An administrator must set PINATA_JWT in the backend environment.',
        });
    }
    try {
        const crop = typeof req.body.crop === 'string' ? safeCrop(req.body.crop) : 'video';
        const signed = await ipfs.signUploadUrl({
            filename: `pestivid_${crop}_${Date.now()}.mp4`,
            maxBytes: ipfs.MAX_BYTES,
            expiresSeconds: Number(process.env.UPLOAD_URL_TTL_SECONDS || 120),
        });
        return res.json({
            url: signed.url,
            expiresSeconds: signed.expiresSeconds,
            maxBytes: ipfs.MAX_BYTES,
            // the phone needs to know the field name and that the server will
            // hash the object afterwards, so it can say so on screen
            field: 'file',
            hashedBy: 'server-after-upload',
        });
    } catch (err) {
        console.error('upload-url error:', err.message);
        return res.status(502).json({ message: 'Could not get an upload address from storage. Try again.' });
    }
});

router.post('/confirm-upload', authenticateToken, limits.uploadLimiter, async (req, res) => {
    if (req.user.role !== 'farmer') {
        return res.status(403).json({ message: "Forbidden: Only users with the 'farmer' role can upload videos." });
    }
    const { cid, crop, pesticide, location, pesticideCompany, purpose } = req.body;
    if (!cid || !crop || !location || !purpose) {
        return res.status(400).json({ message: 'Missing required fields (cid, crop, location, purpose).' });
    }
    if (typeof cid !== 'string' || !/^[A-Za-z0-9]{40,80}$/.test(cid)) {
        return res.status(400).json({ message: 'That does not look like a storage identifier.' });
    }
    if (!['agristream', 'sell', 'funding'].includes(purpose)) {
        return res.status(400).json({ message: "Invalid purpose. Must be 'agristream', 'sell', or 'funding'." });
    }
    // Claiming somebody else's CID must not attach their footage to your name.
    const already = await Video.findOne({ cid });
    if (already) {
        return res.status(409).json({ message: `A video with CID "${cid}" already exists.`, cid });
    }

    let tmp = null;
    try {
        const fetched = await ipfs.fetchToTemp(cid, ipfs.MAX_BYTES);
        tmp = fetched.path;

        // hashed here, from the bytes that are actually in storage
        const videoFileHash = await ipfs.sha256File(tmp);
        const existing = await Video.findOne({ videoFileHash });
        if (existing) {
            await ipfs.cleanup(tmp);
            return res.status(409).json({
                message: 'This exact video has already been uploaded.',
                cid: existing.cid,
                uploadTimestamp: existing.uploadTimestamp,
            });
        }

        let analysis = { fingerprint: undefined, provenance: { flags: [], reviewState: 'none' } };
        try {
            let reported;
            if (req.body.latitude && req.body.longitude) {
                reported = {
                    latitude: Number(req.body.latitude),
                    longitude: Number(req.body.longitude),
                    accuracy: req.body.locationAccuracy ? Number(req.body.locationAccuracy) : undefined,
                };
            }
            analysis = await provenanceSvc.analyse(tmp, {
                farmerId: req.user._id,
                reportedLocation: reported,
            });
        } catch (e) {
            console.warn('Provenance analysis failed, continuing:', e.message);
            analysis.provenance.flags = ['analysis_error'];
        }

        const savedVideo = await new Video({
            cid,
            storageType: 'ipfs',
            videoFileHash,
            hashComputedBy: 'server',
            fingerprint: analysis.fingerprint,
            provenance: analysis.provenance,
            farmerWallet: req.user._id,
            crop: String(crop).trim(),
            pesticide: pesticide ? String(pesticide).trim() : undefined,
            location: String(location).trim(),
            pesticideCompany: pesticideCompany ? String(pesticideCompany).trim() : undefined,
            purpose,
        }).save();

        await ipfs.cleanup(tmp);
        return res.status(201).json({
            message: 'Saved. Its date is being written now.',
            cid: savedVideo.cid,
            videoFileHash: savedVideo.videoFileHash,
            hashComputedBy: savedVideo.hashComputedBy,
            uploadTimestamp: savedVideo.uploadTimestamp,
            bytes: fetched.bytes,
        });
    } catch (err) {
        if (tmp) await ipfs.cleanup(tmp);
        if (err.code === 'TOO_BIG') {
            return res.status(413).json({ message: `That video is larger than the ${Math.round(ipfs.MAX_BYTES / 1024 / 1024)} MB limit.` });
        }
        console.error('confirm-upload error:', err.message);
        return res.status(502).json({
            message: 'We could not read the video back from storage, so we have not recorded it. Nothing is half-saved.',
        });
    }
});

router.post('/upload', authenticateToken,
    limits.uploadBurstLimiter, limits.uploadLimiter,
    (req, res, next) => {
    // Role check BEFORE multer runs, so a non-farmer cannot make us accept and
    // write 100 MB to disk just to be rejected afterwards.
    if (req.user.role !== 'farmer') {
        return res.status(403).json({
            message: "Forbidden: Only users with the 'farmer' role can upload videos.",
        });
    }
    if (!ipfs.pinataConfigured()) {
        return res.status(503).json({
            message: 'Video storage is not configured on the server. ' +
                     'An administrator must set PINATA_JWT in the backend environment.',
        });
    }
    ipfs.upload.single('video')(req, res, (err) => {
        if (err) {
            // Distinguish "too big" from "wrong type" so the farmer can act on it.
            const tooBig = err.code === 'LIMIT_FILE_SIZE';
            return res.status(400).json({
                message: tooBig
                    ? `Video is larger than the ${Math.round(ipfs.MAX_BYTES / 1024 / 1024)} MB limit.`
                    : err.message,
            });
        }
        next();
    });
}, async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: 'No video file was received (field name must be "video").' });
    }

    const tmpPath = req.file.path;
    const { crop, pesticide, location, pesticideCompany, purpose } = req.body;

    if (!crop || !location || !purpose) {
        await ipfs.cleanup(tmpPath);
        return res.status(400).json({ message: 'Missing required fields (crop, location, purpose).' });
    }
    if (!['agristream', 'sell', 'funding'].includes(purpose)) {
        await ipfs.cleanup(tmpPath);
        return res.status(400).json({ message: "Invalid purpose. Must be 'agristream', 'sell', or 'funding'." });
    }

    try {
        // Hash BEFORE pinning: if pinning fails we still know what we were given,
        // and the hash is computed from the bytes on our disk rather than a claim.
        const videoFileHash = await ipfs.sha256File(tmpPath);

        const existing = await Video.findOne({ videoFileHash });
        if (existing) {
            await ipfs.cleanup(tmpPath);
            return res.status(409).json({
                message: 'This exact video has already been uploaded.',
                cid: existing.cid,
                uploadTimestamp: existing.uploadTimestamp,
            });
        }

        // Provenance analysis BEFORE pinning: if the video matches existing
        // footage we still store it and flag it, but doing the work first means a
        // fingerprinting bug cannot leave a pinned file with no analysis. It never
        // blocks -- see services/provenance.js for why flagging is the only safe
        // action here.
        let analysis = { fingerprint: undefined, provenance: { flags: [], reviewState: 'none' } };
        try {
            let reported;
            if (req.body.latitude && req.body.longitude) {
                reported = {
                    latitude: Number(req.body.latitude),
                    longitude: Number(req.body.longitude),
                    accuracy: req.body.locationAccuracy ? Number(req.body.locationAccuracy) : undefined,
                };
            }
            analysis = await provenanceSvc.analyse(tmpPath, {
                farmerId: req.user._id,
                reportedLocation: reported,
            });
        } catch (e) {
            console.warn('Provenance analysis failed, continuing:', e.message);
            analysis.provenance.flags = ['analysis_error'];
        }

        const { cid, pinSize } = await ipfs.pinToPinata(
            tmpPath,
            `pestivid_${safeCrop(crop)}_${Date.now()}.mp4`,
            { farmer: String(req.user._id), crop: safeCrop(crop), purpose },
        );

        const duplicateCid = await Video.findOne({ cid });
        if (duplicateCid) {
            await ipfs.cleanup(tmpPath);
            return res.status(409).json({
                message: `A video with CID "${cid}" already exists.`,
                cid,
            });
        }

        const savedVideo = await new Video({
            cid,
            storageType: 'ipfs',
            videoFileHash,                 // computed HERE, from the actual bytes
            hashComputedBy: 'server',
            fingerprint: analysis.fingerprint,
            provenance: analysis.provenance,
            farmerWallet: req.user._id,
            crop: crop.trim(),
            pesticide: pesticide ? pesticide.trim() : undefined,
            location: location.trim(),
            pesticideCompany: pesticideCompany ? pesticideCompany.trim() : undefined,
            purpose,
        }).save();

        await ipfs.cleanup(tmpPath);

        // Tell the farmer if their video was flagged.
        //
        // Until now the flag was invisible to the person it was about. Their video
        // could sit flagged, and later be rejected, without them ever being told
        // it happened or why -- which makes it impossible to correct a mistake and
        // impossible to plan around. A check that only the platform can see is not
        // a safeguard, it is a secret.
        //
        // The wording matters as much as the existence of the notice. This is a
        // similarity heuristic on video frames, and the innocent explanations are
        // the common ones: re-filming the same row, a second take because the
        // first was too dark, the same field two days apart. So the notice states
        // what was observed, not what it means, and does not imply wrongdoing.
        if (savedVideo.provenance && savedVideo.provenance.reviewState === 'flagged') {
            try {
                const sameFarmer = savedVideo.provenance.similarityKind === 'same_farmer';
                await new Notification({
                    recipient: req.user._id,
                    type: 'warning',
                    message: sameFarmer
                        ? `Your video for "${savedVideo.crop}" looks very similar to one `
                          + 'you already uploaded, so it is queued for a quick check. '
                          + 'This is often just a re-filmed clip and is usually fine. '
                          + 'We aim to review within 2 working days. You can add an '
                          + 'explanation from the video\'s page, which we will read '
                          + 'before deciding.'
                        : `Your video for "${savedVideo.crop}" closely matches `
                          + 'another video on the platform, so it is queued for a check '
                          + 'before it can be used for funding. We aim to review within '
                          + '2 working days. If there is an explanation, please add it '
                          + 'from the video\'s page -- we will read it before deciding.',
                    itemId: savedVideo._id,
                    itemType: 'Video',
                    read: false,
                }).save();

                savedVideo.provenance.flagNotifiedAt = new Date();
                await savedVideo.save();
            } catch (noteErr) {
                // A failed notice must not fail the upload -- the file is already
                // pinned. But it must be loud, because an unnotified flag is the
                // exact failure this code exists to prevent.
                console.error('FLAGGED VIDEO NOT NOTIFIED:', savedVideo.cid, noteErr.message);
            }
        }

        return res.status(201).json({
            message: 'Video uploaded, hashed and pinned by the server.',
            video: {
                _id: savedVideo._id,
                cid: savedVideo.cid,
                videoFileHash: savedVideo.videoFileHash,
                hashAlgorithm: 'sha256',
                hashComputedBy: 'server',
                storageType: savedVideo.storageType,
                crop: savedVideo.crop,
                location: savedVideo.location,
                purpose: savedVideo.purpose,
                uploadTimestamp: savedVideo.uploadTimestamp,
                sizeBytes: pinSize,
                // What the platform can and cannot stand behind. Sent on every
                // upload so the UI never has to guess or invent a badge.
                provenance: provenanceSvc.describe(savedVideo),
            },
        });
    } catch (err) {
        await ipfs.cleanup(tmpPath);
        if (err.code === 'PINATA_NOT_CONFIGURED') {
            return res.status(503).json({ message: err.message });
        }
        console.error('Video upload failed:', err.message);
        // Do not leak the Pinata response body: it can echo the Authorization header.
        return res.status(502).json({
            message: 'Upload to decentralised storage failed. The video was not saved.',
        });
    }
});

router.post('/', authenticateToken, async (req, res) => {
    // Authorization: Ensure the authenticated user has the 'farmer' role.
    if (req.user.role !== 'farmer') {
        console.warn(`Authorization failed: User ${req.user._id} with role ${req.user.role} attempted to create video metadata.`);
        return res.status(403).json({ message: "Forbidden: Only users with the 'farmer' role can upload video metadata." }); // 403 Forbidden
    }

    // Extract video metadata from the request body
    const { cid, storageType, videoFileHash, crop, pesticide, location, pesticideCompany, purpose } = req.body;

    // Basic input validation
    if (!cid || !crop || !location || !purpose) {
         return res.status(400).json({ message: "Missing required video metadata fields (cid, crop, location, purpose)." }); // 400 Bad Request
    }
     if (storageType && !['storj', 'ipfs'].includes(storageType)) {
          return res.status(400).json({ message: "Invalid storage type specified." }); // 400 Bad Request
     }
     if (!['agristream', 'sell', 'funding'].includes(purpose)) {
         return res.status(400).json({ message: "Invalid purpose specified. Must be 'agristream', 'sell', or 'funding'." }); // 400 Bad Request
     }

    try {
        // Check if a video entry with this CID already exists (ensures CID uniqueness)
        const existingVideo = await Video.findOne({ cid: cid });
        if (existingVideo) {
            return res.status(409).json({ message: `Video metadata with CID "${cid}" already exists.` }); // 409 Conflict
        }

        // Create a new Video document instance
        const newVideo = new Video({
            cid: cid.trim(),
            storageType: storageType || 'ipfs', // Use provided type or default
            // A client-supplied hash is NOT recorded. The uploader chooses it, so
            // storing it in the same field as a server-computed one would make the
            // two indistinguishable later. Use POST /api/videos/upload to get a
            // verified hash. The CID is kept because a CID is content-addressed and
            // therefore independently re-checkable against the bytes it names.
            videoFileHash: undefined,
            hashComputedBy: 'unverified',
            farmerWallet: req.user._id, // Link the video to the authenticated farmer's user ID
            crop: crop.trim(),
            pesticide: pesticide ? pesticide.trim() : undefined, // Optional
            location: location.trim(),
            pesticideCompany: pesticideCompany ? pesticideCompany.trim() : undefined, // Optional
            purpose: purpose,
            // uploadTimestamp defaults to now in the schema
        });

        // Save the new video metadata document to the database
        const savedVideo = await newVideo.save();

        // Optional: Trigger notifications here (e.g., notify public AgriStream viewers of a new video)
         // const notification = new Notification({
         //     global: true, // This could be a global notification
         //     type: 'video', // Custom notification type
         //     message: `New video uploaded: "${savedVideo.crop}" from a farmer!`,
         //      itemId: savedVideo._id, // Link to the video metadata document
         //      itemType: 'Video',
         //      // Could add a link or context like farmer name: farmerName: req.user.name.split(' ')[0]
         // });
         // await notification.save();
         // console.log(`SIMULATING: Created global notification for new video ${savedVideo._id}`);


        // Populate the farmerWallet field on the saved document before sending it back,
        // so the frontend immediately has the farmer's name associated with the video.
        await savedVideo.populate('farmerWallet', 'name role displayIdentifier');


        // Send a success response with the created video document
        res.status(201).json(savedVideo); // 201 Created

    } catch (err) {
        console.error('POST /api/videos error:', err);
         // Handle Mongoose validation errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
         // Handle duplicate key error (though we check explicitly above, Mongoose might throw it)
          if (err.code === 11000) {
              return res.status(409).json({ message: `Video metadata with CID "${cid}" already exists.` });
         }
        res.status(500).json({ message: 'Server error creating video metadata.' }); // 500 Internal Server Error
    }
});

// @route DELETE /api/videos/:id
// @desc Delete a video metadata entry by ID
// @access Private (Requires authentication. Must be a farmer and own the video. Cannot be used in listings/funding.)
router.delete('/:id', authenticateToken, async (req, res) => {
    const videoId = req.params.id; // Get the video ID from the URL parameter
    const userId = req.user._id;   // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(videoId)) {
        return res.status(400).json({ message: 'Invalid Video ID format.' }); // 400 Bad Request
    }

    try {
        // Find the video by ID
        const video = await Video.findById(videoId);

        // If video is not found, return 404
        if (!video) {
            return res.status(404).json({ message: 'Video not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is the farmer who owns the video.
        if (video.farmerWallet.toString() !== userId.toString()) {
            console.warn(`Authorization failed: User ${userId} attempted to delete video ${videoId} owned by ${video.farmerWallet}`);
            return res.status(403).json({ message: "Forbidden: You can only delete your own videos." }); // 403 Forbidden
        }

        // Check if the video is currently used in any active marketplace listings
        // We check by video's CID and potentially farmer/status
        const isUsedInListing = await Listing.findOne({ cid: video.cid, status: 'active' }); // Assuming one active listing per video
        if (isUsedInListing) {
            return res.status(400).json({ message: `Cannot delete video: It is currently used in active marketplace listing "${isUsedInListing.crop}" (ID: ${isUsedInListing._id}). Please cancel or sell the listing first.` }); // 400 Bad Request
        }

        // Check if the video is currently used in any active or pending funding requests
        const isUsedInFunding = await FundingRequest.findOne({ cid: video.cid, status: { $in: ['pending', 'partially_funded', 'funded', 'growing'] } });
         if (isUsedInFunding) {
            return res.status(400).json({ message: `Cannot delete video: It is currently used in funding request "${isUsedInFunding.title}" (ID: ${isUsedInFunding._id}). Please cancel the request first.` }); // 400 Bad Request
         }


        // --- IMPORTANT: In a real application, you would also need to delete the actual video file ---
        // from the decentralized storage service (Storj/IPFS) here.
        // This typically involves using the SDK/API of the storage service on the backend.
        // This demo backend does NOT implement file deletion logic.
        console.log(`SIMULATING: Backend would now attempt to delete file from storage for CID: ${video.cid} (Storage Type: ${video.storageType})`);
        // Example Placeholder:
        // try {
        //    if (video.storageType === 'storj') {
        //       // Use Storj S3 SDK with credentials from .env to delete the object
        //       // const AWS = require('aws-sdk'); // Needs to be configured similar to frontend upload
        //       // ... s3.deleteObject({ Bucket: process.env.STORJ_BUCKET_NAME, Key: video.cid }).promise();
        //    } else if (video.storageType === 'ipfs') {
        //       // Use Pinata API or similar to unpin the CID
        //       // ... axios.delete(https://api.pinata.cloud/pinning/unpin/${video.cid}, { headers: { Authorization: Bearer ${process.env.PINATA_JWT} } });
        //    }
        //    console.log(Simulated file deletion for CID ${video.cid} successful.);
        // } catch (deleteError) {
        //    console.error(SIMULATION WARNING: Simulated file deletion failed for CID ${video.cid}:, deleteError.message);
            // Decide if you want to prevent deleting the DB record if file deletion fails.
            // For critical data, you might throw an error here. For a demo, maybe just log.
        // }


        // Delete the video document from the database
        // Retain the perceptual fingerprint before the record goes.
        //
        // Without this, upload-then-delete is a free unlimited oracle: probe the
        // duplicate detector with a laundered copy of a stolen clip, read the
        // result, delete, adjust, repeat until it passes. Keeping the frame
        // hashes means every probe costs the attacker a permanent artefact, and a
        // deleted clip is still recognised if it returns under another account.
        //
        // Only the hashes and an HMAC'd uploader reference are kept -- no CID, no
        // file hash, no location. A farmer who deletes a video is entitled to have
        // the record gone; this keeps the minimum needed to recognise the same
        // FOOTAGE, not a shadow copy.
        if (video.fingerprint && Array.isArray(video.fingerprint.frameHashes)
            && video.fingerprint.frameHashes.length) {
            try {
                const RetiredFingerprint = mongoose.model('RetiredFingerprint');
                await RetiredFingerprint.create({
                    frameHashes: video.fingerprint.frameHashes,
                    nFrames: video.fingerprint.nFrames,
                    durationSeconds: video.fingerprint.durationSeconds,
                    algorithm: video.fingerprint.algorithm,
                    farmerRef: require('crypto')

                        .createHmac('sha256', process.env.JWT_SECRET || 'pestivid-retire')
                        .update(String(video.farmerWallet)).digest('hex'),
                    reason: 'deleted_by_farmer',
                });
            } catch (e) {
                // Retention must not block a farmer deleting their own video.
                console.warn('Could not retain retired fingerprint:', e.message);
            }
        }

        await video.deleteOne(); // Use deleteOne() or findByIdAndDelete()

        // Send a success response
        res.json({ message: 'Video metadata removed successfully.' }); // Default status is 200 OK

    } catch (err) {
        console.error(`DELETE /api/videos/${videoId} error:`, err);
        // Handle potential errors during lookup or deletion
        res.status(500).json({ message: 'Server error deleting video metadata.' }); // 500 Internal Server Error
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;