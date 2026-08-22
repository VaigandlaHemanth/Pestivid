/**
 * Perceptual fingerprinting of uploaded videos, for recycled/stolen-footage
 * detection.
 *
 * WHY THIS AND NOT A HASH
 *   SHA-256 proves the bytes did not change after upload. It says nothing about
 *   whether the farmer shot the video. Re-encoding a stolen video, or trimming
 *   one second off last season's clip, changes every byte and defeats SHA-256
 *   completely. A perceptual hash survives re-encoding, rescaling and mild
 *   brightness changes, so it is the only cheap control that catches:
 *
 *     - re-uploading last season's footage
 *     - uploading another farmer's video from this platform
 *
 *   It does NOT catch footage downloaded from the internet that we have never
 *   seen before. Nothing cheap does.
 *
 * WHY NO IMAGE LIBRARY
 *   dHash needs a 9x8 grayscale raster. ffmpeg can produce exactly that and
 *   nothing else -- `scale=9:8,format=gray,-f rawvideo` emits 72 raw bytes per
 *   frame. So the decode, the resize and the desaturation all happen in the
 *   bundled static binary, and the hash is plain arithmetic over a Buffer. That
 *   avoids sharp (native build, fails on some hosts) and jimp (slow, large), and
 *   keeps peak memory at a few kilobytes regardless of video size.
 *
 * THRESHOLD CHOICE
 *   Hamming distance over a 64-bit dHash. Distances are interpreted per FRAME
 *   PAIR, then aggregated, because a single matching frame is weak evidence (two
 *   videos of the same green field will share a frame or two) while many matching
 *   frames in order is strong evidence of the same source footage.
 *
 *   Deliberately conservative: a false accusation costs a smallholder a season's
 *   funding, so this module reports a score and never decides on its own.
 */

const { execFile } = require('child_process');
const path = require('path');

const ffmpegPath = require('ffmpeg-static');

// 9x8 grayscale = 72 bytes, the exact input dHash needs.
const HASH_W = 9;
const HASH_H = 8;
const FRAME_BYTES = HASH_W * HASH_H;

// Frames sampled per video, and the sampling rate.
//
// The frame COUNT and the frame RATE have to be chosen together. The first
// version computed fps = count / duration and then capped extraction at
// `-frames:v count`, which is self-consistent -- but when the duration probe
// failed it fell back to fps = 1 while keeping the cap at 12, so only the FIRST
// 12 SECONDS of the video was ever fingerprinted. Everything after that was
// invisible to the duplicate detector, which is a free hiding place: prepend 12
// seconds of anything and a stolen clip stops matching.
//
// So the cap now scales with duration instead of being fixed, and the fps
// fallback is derived from the same numbers.
const DEFAULT_FRAMES = 12;

// Never sample fewer than this, or a long video gets too little coverage to
// compare; never more, or a 30-minute upload costs minutes of CPU.
const MIN_FRAMES = 8;
const MAX_FRAMES = 60;

// Target one frame every N seconds for long videos, so coverage stays even
// rather than clustering at the start.
const SECONDS_PER_FRAME = 5;

// Per-frame Hamming distance at or below which two frames are "the same shot".
// 64-bit dHash: identical images give 0, re-encodes of the same frame typically
// stay under 6, and unrelated images average around 32.
const FRAME_MATCH_MAX = 8;

function runFfmpeg(args, timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
        execFile(ffmpegPath, args, {
            timeout: timeoutMs,
            maxBuffer: 64 * 1024 * 1024,
            encoding: 'buffer',
        }, (err, stdout, stderr) => {
            if (err) {
                const msg = (stderr && stderr.toString().split('\n').slice(-4).join(' ')) || err.message;
                return reject(new Error(`ffmpeg failed: ${msg.slice(0, 300)}`));
            }
            resolve(stdout);
        });
    });
}

/** Duration in seconds, or null if ffmpeg cannot tell us. */
async function probeDuration(videoPath) {
    try {
        // ffmpeg writes the summary to stderr and exits non-zero with -f null,
        // so ask for it the way that exits cleanly.
        const out = await runFfmpeg([
            '-v', 'error', '-i', videoPath, '-f', 'null', '-',
        ]).catch(() => null);
        void out;
    } catch (_) { /* fall through */ }
    // Simpler and reliable: decode the container header only.
    return new Promise((resolve) => {
        execFile(ffmpegPath, ['-i', videoPath], (err, stdout, stderr) => {
            const text = `${stderr || ''}`;
            const m = text.match(/Duration:\s*(\d+):(\d+):(\d+\.?\d*)/);
            if (!m) return resolve(null);
            resolve(Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]));
        });
    });
}

/**
 * Sample `count` evenly spaced frames as 9x8 grayscale rasters.
 * Returns an array of Buffers of length 72.
 */
async function sampleRasters(videoPath, count = DEFAULT_FRAMES) {
    const duration = await probeDuration(videoPath);

    // Decide the frame budget from the actual duration, so coverage spans the
    // WHOLE video rather than its first few seconds.
    let frames = count;
    if (duration && duration > 0) {
        frames = Math.round(duration / SECONDS_PER_FRAME);
        frames = Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, frames));
    }

    // fps filter rather than seeking: one decode pass, and it works even when the
    // container has no seek index (common for MediaRecorder WebM output, which is
    // written as a live stream and often lacks a cues element).
    //
    // When the duration is unknown we CANNOT compute a spreading fps, so sample
    // slowly and raise the cap instead of silently truncating: a low fps with a
    // high cap still walks the whole file.
    const fps = duration && duration > 0
        ? Math.max(frames / duration, 0.02)
        : 0.5;
    const cap = duration && duration > 0 ? frames : MAX_FRAMES;

    const out = await runFfmpeg([
        '-v', 'error',
        '-i', videoPath,
        '-vf', `fps=${fps.toFixed(4)},scale=${HASH_W}:${HASH_H}:flags=area,format=gray`,
        '-frames:v', String(cap),
        '-f', 'rawvideo',
        '-',
    ]);

    const rasters = [];
    for (let o = 0; o + FRAME_BYTES <= out.length; o += FRAME_BYTES) {
        rasters.push(out.subarray(o, o + FRAME_BYTES));
    }
    return { rasters, duration, fps };
}

/**
 * dHash: compare each pixel with its right-hand neighbour, 8 rows of 8
 * comparisons = 64 bits, returned as 16 lowercase hex characters.
 *
 * Gradient-based rather than average-based (aHash) because it is markedly more
 * robust to overall brightness and exposure changes -- and a field video shot at
 * a different hour of the day is exactly a brightness change.
 */
function dHashFromRaster(raster) {
    let hex = '';
    let bits = 0;
    let nibble = 0;
    for (let y = 0; y < HASH_H; y++) {
        for (let x = 0; x < HASH_W - 1; x++) {
            const left = raster[y * HASH_W + x];
            const right = raster[y * HASH_W + x + 1];
            nibble = (nibble << 1) | (left < right ? 1 : 0);
            if (++bits === 4) {
                hex += nibble.toString(16);
                bits = 0;
                nibble = 0;
            }
        }
    }
    return hex;
}

const POPCOUNT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
    POPCOUNT[i] = (i & 1) + POPCOUNT[i >> 1];
}

/** Hamming distance between two 16-hex-char dHashes. 0 = identical, ~32 = unrelated. */
function hamming(a, b) {
    if (!a || !b || a.length !== b.length) return Infinity;
    let d = 0;
    for (let i = 0; i < a.length; i += 2) {
        d += POPCOUNT[parseInt(a.substr(i, 2), 16) ^ parseInt(b.substr(i, 2), 16)];
    }
    return d;
}

/** Fingerprint a video: an ordered list of frame dHashes. */
async function fingerprint(videoPath, count = DEFAULT_FRAMES) {
    const { rasters, duration, fps } = await sampleRasters(videoPath, count);
    if (!rasters.length) {
        throw new Error('No frames could be decoded from the video.');
    }
    // Report the span actually covered. A fingerprint over the first 20% of a
    // video is not a failure, but it is not a clean result either, and the caller
    // must be able to tell the difference.
    const coveredSeconds = duration && duration > 0
        ? Math.min(duration, rasters.length / Math.max(fps, 1e-6))
        : null;

    return {
        frameHashes: rasters.map(dHashFromRaster),
        nFrames: rasters.length,
        durationSeconds: duration,
        coveredSeconds: coveredSeconds != null ? Math.round(coveredSeconds) : undefined,
        algorithm: `dhash64/${HASH_W}x${HASH_H}`,
    };
}

/**
 * Compare two fingerprints.
 *
 * Reports the FRACTION of frames in the shorter fingerprint that have a close
 * match anywhere in the other, plus the best single distance. Order is not
 * required to line up, because trimming or a different start point shifts every
 * index while leaving the content identical.
 */
function compare(a, b, frameMatchMax = FRAME_MATCH_MAX) {
    const A = (a && a.frameHashes) || [];
    const B = (b && b.frameHashes) || [];
    if (!A.length || !B.length) {
        return { matchedFraction: 0, matchedFrames: 0, bestDistance: Infinity, comparable: false };
    }
    const [short, long] = A.length <= B.length ? [A, B] : [B, A];
    let matched = 0;
    let best = Infinity;
    for (const h of short) {
        let localBest = Infinity;
        for (const g of long) {
            const d = hamming(h, g);
            if (d < localBest) localBest = d;
            if (d === 0) break;
        }
        if (localBest < best) best = localBest;
        if (localBest <= frameMatchMax) matched++;
    }
    return {
        matchedFraction: matched / short.length,
        matchedFrames: matched,
        framesCompared: short.length,
        bestDistance: best,
        comparable: true,
    };
}

module.exports = {
    fingerprint,
    compare,
    hamming,
    dHashFromRaster,
    sampleRasters,
    probeDuration,
    ffmpegPath,
    DEFAULT_FRAMES,
    FRAME_MATCH_MAX,
};
