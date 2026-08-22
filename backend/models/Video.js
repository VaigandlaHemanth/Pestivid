// --- Mongoose Model: Video.js ---

const mongoose = require('mongoose'); // Import Mongoose library

// Define the schema for the Video model
const videoSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key.

    cid: {
        type: String,
        required: [true, 'Video CID/Key is required'], // The identifier from storage (Storj object key or IPFS CID)
        unique: true, // Ensure each video record is unique based on its CID/Key
        trim: true
    },
    storageType: {
        type: String,
        required: [true, 'Storage type is required'], // e.g., 'storj', 'ipfs'
        enum: {
            values: ['storj', 'ipfs'],
            message: 'Invalid storage type. Must be "storj" or "ipfs".'
        },
        default: 'ipfs' // Default to IPFS if not specified
    },
    // WHO computed videoFileHash. Load-bearing, not documentation: only
    // 'server' hashes are eligible for blockchain anchoring, because a
    // client-supplied hash is chosen by the uploader and proves nothing. Anchoring
    // one would attach a real Bitcoin proof to an unverified claim.
    hashComputedBy: {
        type: String,
        enum: ['server', 'unverified'],
        default: 'unverified',
        index: true,
    },
    // Perceptual fingerprint: an ordered list of 64-bit dHashes over sampled
    // frames. Unlike videoFileHash this SURVIVES re-encoding, so it is the only
    // signal that catches a farmer re-uploading last season's clip or another
    // farmer's video. Measured on re-encode/rescale/brightness/trim attacks:
    // 100% frame match for a copy, 0% for unrelated footage.
    fingerprint: {
        frameHashes: { type: [String], default: undefined },
        nFrames: Number,
        durationSeconds: Number,
        // How many seconds the sampled frames actually span. Mongoose silently
        // DROPS fields that are not in the schema, so without this the service
        // computed coverage and it never reached the database -- meaning the
        // "was the whole video checked?" disclosure could never say yes.
        coveredSeconds: Number,
        algorithm: String,
    },

    // Provenance signals, kept SEPARATE from integrity. None of these are proven
    // by the hash or by any blockchain anchor, and the UI must not present them
    // as if they were. Every one of them only ever FLAGS: a wrongly blocked
    // upload can cost a smallholder a season's funding, so a human decides.
    provenance: {
        // Another video whose frames match closely enough to suggest the same
        // source footage. Set for both same-farmer (recycling) and cross-farmer
        // (theft) matches -- they are different problems with the same signal.
        similarTo: { type: mongoose.Schema.Types.ObjectId, ref: 'Video' },
        similarToFarmer: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
        similarityFraction: Number,
        similarityKind: { type: String, enum: ['same_farmer', 'other_farmer'] },
        // Farmer-reported, NEVER verified. The browser geolocation API is
        // trivially spoofed with a mock-location app or devtools override, so
        // this is context for a reviewer, not evidence.
        reportedLatitude: Number,
        reportedLongitude: Number,
        reportedAccuracyMetres: Number,
        flags: { type: [String], default: [] },
        reviewState: {
            type: String,
            enum: ['none', 'flagged', 'cleared', 'rejected'],
            default: 'none',
            index: true,
        },

        // Who decided, when, and why.
        //
        // 'cleared' and 'rejected' were in the enum from the start and nothing
        // could ever write them: there was no reviewer, no queue and no route, so
        // a flagged video sat at 'flagged' forever, invisible to everyone and
        // affecting nothing. Detection ran and was then discarded.
        //
        // A decision is recorded, not enforced. Nothing here deletes a video,
        // blocks a farmer or changes eligibility -- that stays a deliberate human
        // step, because the flag is a similarity heuristic and a false positive
        // costs a farmer their funding round.
        reviewedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
        reviewedAt: Date,
        reviewNote: { type: String, maxlength: 2000 },

        // The farmer was told, and when.
        //
        // A flag with no notice is a secret accusation: the video sits flagged,
        // possibly rejected, and the person it is about never learns why or that
        // it happened. notifiedAt records that we actually told them.
        flagNotifiedAt: Date,

        // What the farmer said in response.
        //
        // The reviewer has the fingerprint score; only the farmer has the reason.
        // "I re-filmed the same row because the first clip was too dark" is not
        // visible in a dHash comparison, and it is usually the truth. Without a
        // way to say it, review is one-sided by construction.
        appeal: {
            statement: { type: String, maxlength: 2000 },
            submittedAt: Date,
            // Appeals can be revised until a decision is made; this counts them so
            // an endless-resubmission loop is visible rather than silent.
            revisions: { type: Number, default: 0 },
        },
    },

    videoFileHash: { // SHA256 hash of the video file for integrity verification
        type: String,
        trim: true
        // Not required, as calculating hash might fail or not be needed for all entries.
        // Can add unique: true if you want to prevent uploading the exact same file twice,
        // but this might block re-uploading the same video with different metadata.
    },
    farmerWallet: { // Reference to the User document representing the farmer who uploaded the video
        type: mongoose.Schema.Types.ObjectId, // This is a special Mongoose type for linking documents
        ref: 'User', // This tells Mongoose this field references the 'User' model
        required: [true, 'Farmer ID is required']
    },
    crop: {
        type: String,
        required: [true, 'Crop type is required'],
        trim: true
    },
    pesticide: { // Pesticide used (or 'None', 'Organic', etc.)
        type: String,
        trim: true
        // Not required as per your form, making it optional
    },
    location: { // Field location where the video was recorded
        type: String,
        required: [true, 'Location is required'],
        trim: true
    },
    pesticideCompany: { // Company name for the pesticide used
        type: String,
        trim: true
        // Not required as per your form, making it optional
    },
    purpose: { // Intended purpose of the video upload
        type: String,
        required: [true, 'Purpose is required'],
        enum: {
            values: ['agristream', 'sell', 'funding'], // Values must match your frontend form purposes
            message: 'Invalid purpose. Must be "agristream", "sell", or "funding".'
        },
        default: 'agristream' // Default purpose if not explicitly set
    },
    uploadTimestamp: {
        type: Date,
        default: Date.now // Default to the current date and time when the video metadata is saved
    },
    // Add other fields as needed, e.g.:
    // fileSize: Number, // Size of the uploaded video file
    // duration: Number, // Duration of the video in seconds
    // status: { type: String, enum: ['processing', 'ready', 'error'], default: 'ready' } // Status after upload
});

// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'videos' (lowercase, plural) for this model.
const Video = mongoose.model('Video', videoSchema);

// Export the model so it can be used in other files (like route handlers or other models via ref)
module.exports = Video;
