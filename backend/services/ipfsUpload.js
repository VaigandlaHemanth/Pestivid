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

/**
 * What a phone's recorder actually produces, and what a content sniffer makes
 * of it.
 *
 * MediaRecorder writes a STREAMING WebM: the container header goes out before
 * the track layout is complete, so a sniffer reading the first bytes classifies
 * it as audio/webm. Pinata sniffs, and rejected every single real recording
 * with "Presigned URL does not grant permissions to upload detected MIME type:
 * audio/webm". No farmer could send a video at all.
 *
 * So audio/webm and audio/mp4 are on the list -- not because we want audio, but
 * because that is what a video from a browser looks like to a sniffer. It costs
 * nothing: the server downloads the object back and hashes the bytes itself
 * before recording anything, so the file is checked on its content either way.
 */
const ALLOWED_MIME = new Set([
    'video/mp4', 'video/webm', 'video/quicktime', 'video/x-matroska',
    'audio/webm', 'audio/mp4', 'application/octet-stream',
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


/**
 * Ask Pinata for a one-use upload URL the handset can post straight to.
 *
 * This exists because the video cannot travel through our own API on a free
 * host: a serverless function request body caps at 4.5 MB and a forty-second
 * clip is about 10 MB. The phone therefore uploads to storage directly, and we
 * pull the object back afterwards to hash it -- see fetchToTemp below. The JWT
 * never leaves the server; what the phone receives is a URL that expires.
 */
async function signUploadUrl({ filename, maxBytes = MAX_BYTES, expiresSeconds = 120 }) {
    if (!pinataConfigured()) {
        const e = new Error('PINATA_JWT is not configured on the server.');
        e.code = 'PINATA_NOT_CONFIGURED';
        throw e;
    }
    const resp = await axios.post(
        'https://uploads.pinata.cloud/v3/files/sign',
        {
            network: 'public',
            // `date` is required and undocumented in the guide: without it the
            // endpoint answers 400 invalid_type on body.date. Unix seconds.
            date: Math.floor(Date.now() / 1000),
            expires: expiresSeconds,
            filename,
            max_file_size: maxBytes,
            // ALLOWED_MIME is a Set, and JSON.stringify turns a Set into {}
            allow_mime_types: [...ALLOWED_MIME],
        },
        {
            headers: { Authorization: `Bearer ${process.env.PINATA_JWT}` },
            timeout: Number(process.env.PINATA_TIMEOUT_MS || 30000),
        },
    );
    const url = resp.data && (resp.data.data || resp.data.url || resp.data.signedUrl);
    if (typeof url !== 'string' || !/^https:\/\//.test(url)) {
        throw new Error('Pinata did not return a usable upload URL.');
    }
    return { url, expiresSeconds };
}

/** Where we read a pinned object back from. */
function gatewayUrl(cid) {
    const base = (process.env.PINATA_GATEWAY || 'https://gateway.pinata.cloud').replace(/\/+$/, '');
    return `${base}/ipfs/${cid}`;
}

/**
 * Pull a pinned object back to a temp file so the server can hash the bytes
 * itself.
 *
 * The landing page promises the server hashes the bytes it actually received
 * and never a number the phone sends. Direct-to-storage upload would break that
 * promise unless we fetch the object and hash it here, so we do. It costs one
 * download per video, which is inside the budget, and it is the only reason the
 * claim is allowed to stay on the page.
 *
 * Streams to disk and refuses anything over the size limit rather than reading
 * an unbounded response into memory.
 */
async function fetchToTemp(cid, maxBytes = MAX_BYTES, attempt = 0) {
    const dest = path.join(os.tmpdir(), `pv_${Date.now()}_${Math.random().toString(36).slice(2)}.bin`);
    try {
        const resp = await axios.get(gatewayUrl(cid), {
            responseType: 'stream',
            timeout: Number(process.env.PINATA_TIMEOUT_MS || 180000),
            maxRedirects: 3,
        });
        let seen = 0;
        const out = fs.createWriteStream(dest);
        await new Promise((resolve, reject) => {
            resp.data.on('data', (chunk) => {
                seen += chunk.length;
                if (seen > maxBytes) {
                    resp.data.destroy();
                    out.destroy();
                    reject(Object.assign(new Error('Stored object is larger than the limit.'), { code: 'TOO_BIG' }));
                }
            });
            resp.data.on('error', reject);
            out.on('error', reject);
            out.on('finish', resolve);
            resp.data.pipe(out);
        });
        if (seen === 0) throw new Error('Stored object was empty.');
        return { path: dest, bytes: seen };
    } catch (e) {
        await cleanup(dest);
        // A gateway does not always serve an object the instant it is pinned.
        // That is a timing problem, not a missing file, so back off and try
        // again before telling a farmer their upload failed.
        const status = e.response && e.response.status;
        const retriable = e.code !== 'TOO_BIG'
            && (status === undefined || [404, 408, 425, 429, 500, 502, 503, 504].includes(status));
        if (retriable && attempt < 4) {
            await new Promise(r => setTimeout(r, 900 * (attempt + 1)));
            return fetchToTemp(cid, maxBytes, attempt + 1);
        }
        throw e;
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
    signUploadUrl,
    fetchToTemp,
    gatewayUrl,
};
