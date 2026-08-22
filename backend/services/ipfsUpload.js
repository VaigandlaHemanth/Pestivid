/**
 * Server-side IPFS pinning, so the platform can actually attest to a video.
 *
 * WHY THIS EXISTS — it replaces a design that could not be secured.
 *
 * Before this, the browser held the Pinata JWT (hardcoded in index.html) and
 * POSTed the file straight to api.pinata.cloud. Two consequences:
 *
 *   1. The credential shipped to every visitor. Anyone opening devtools could
 *      pin arbitrary content to the project's Pinata account, or delete it.
 *   2. The server never touched the video, so `cid` and `videoFileHash` arrived
 *      from req.body — attacker-controlled. The "integrity hash" proved nothing.
 *
 * (2) is why this had to be fixed before any blockchain anchoring. Anchoring a
 * client-supplied hash to Bitcoin does not make it true; it just attaches a very
 * strong proof to an unverified claim, which is worse than no proof at all
 * because it invites people to trust it.
 *
 * WHY STREAM TO A TEMP FILE RATHER THAN BUFFER IN MEMORY
 * Free hosting tiers run at 512 MB RAM. multer's memoryStorage would hold the
 * whole video in the heap and a couple of concurrent 50 MB uploads would OOM the
 * instance. Disk storage streams instead, so peak memory stays flat and only the
 * hash accumulator lives in RAM.
 *
 * WHAT THIS DOES NOT PROVE
 * That the video shows THIS farmer's land. A hash proves the bytes did not
 * change; it says nothing about where they came from. Recycled or borrowed
 * footage is a real fraud path here and needs GPS/EXIF checks or live capture,
 * not cryptography.
 */

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const axios = require('axios');
const FormData = require('form-data');
const multer = require('multer');

// 100 MB. Large enough for a few minutes of phone video, small enough that a
// free-tier instance and a rural uplink can both survive it.
const MAX_BYTES = Number(process.env.MAX_VIDEO_BYTES || 100 * 1024 * 1024);

const ALLOWED_MIME = new Set([
    'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska',
]);

const upload = multer({
    // Temp directory, not memory: see the note above about 512 MB instances.
    storage: multer.diskStorage({
        destination: (req, file, cb) => cb(null, os.tmpdir()),
        filename: (req, file, cb) =>
            cb(null, `pestivid_${Date.now()}_${crypto.randomBytes(6).toString('hex')}`),
    }),
    limits: { fileSize: MAX_BYTES, files: 1 },
    fileFilter: (req, file, cb) => {
        if (!ALLOWED_MIME.has(file.mimetype)) {
            return cb(new Error(
                `Unsupported video type "${file.mimetype}". Allowed: ${[...ALLOWED_MIME].join(', ')}`));
        }
        cb(null, true);
    },
});

/** SHA-256 of the file on disk, streamed so memory stays flat. */
function sha256File(filePath) {
    return new Promise((resolve, reject) => {
        const h = crypto.createHash('sha256');
        const s = fs.createReadStream(filePath);
        s.on('data', (d) => h.update(d));
        s.on('error', reject);
        s.on('end', () => resolve(h.digest('hex')));
    });
}

function pinataConfigured() {
    return Boolean(process.env.PINATA_JWT && process.env.PINATA_JWT.length > 50);
}

/**
 * Pin a local file to IPFS via Pinata and return its CID.
 * Streams from disk; never loads the file into a Buffer.
 */
async function pinToPinata(filePath, name, metadata = {}) {
    if (!pinataConfigured()) {
        const e = new Error(
            'PINATA_JWT is not configured on the server. Set it in backend/.env — ' +
            'it must never be placed in frontend code.');
        e.code = 'PINATA_NOT_CONFIGURED';
        throw e;
    }

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), { filename: name });
    form.append('pinataMetadata', JSON.stringify({ name, keyvalues: metadata }));
    form.append('pinataOptions', JSON.stringify({ cidVersion: 1 }));

    const resp = await axios.post(
        'https://api.pinata.cloud/pinning/pinFileToIPFS',
        form,
        {
            headers: {
                ...form.getHeaders(),
                Authorization: `Bearer ${process.env.PINATA_JWT}`,
            },
            maxBodyLength: Infinity,
            maxContentLength: Infinity,
            timeout: Number(process.env.PINATA_TIMEOUT_MS || 180000),
        },
    );

    const cid = resp.data && (resp.data.IpfsHash || resp.data.cid);
    if (!cid) {
        throw new Error('Pinata returned no CID; refusing to record an unpinned video.');
    }
    return { cid, pinSize: resp.data.PinSize, pinnedAt: resp.data.Timestamp };
}

/** Always remove the temp file, success or failure. */
async function cleanup(filePath) {
    if (!filePath) return;
    try {
        await fs.promises.unlink(filePath);
    } catch (_) {
        /* already gone, or never created — nothing to do */
    }
}

module.exports = {
    upload,
    sha256File,
    pinToPinata,
    pinataConfigured,
    cleanup,
    MAX_BYTES,
    ALLOWED_MIME,
};
