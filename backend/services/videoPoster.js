// --- One frame of the video, small enough to put in a list ---
//
// Every list of videos in this product drew a dark rectangle with a play glyph
// in it. That is an honest placeholder and it is not a thumbnail: five rows of
// identical grey boxes tell a farmer nothing about which clip is which, and tell
// an investor nothing at all.
//
// WHY THE SERVER DOES THIS AND NOT THE BROWSER.
// The browser has the file at upload time and could produce a poster in three
// lines of canvas code. It must not. This product's entire claim is that the
// video is the one thing on the page nobody has to trust -- we fetch the stored
// object back and hash it ourselves precisely so the fingerprint is not a number
// the phone sent us. A client-supplied poster would be an unverified image
// sitting next to that hash, looking like a frame of it. A farmer could send a
// photograph of a healthy field as the poster for a diseased one and every
// screen would show it as evidence. So the poster is cut from the SAME temp file
// the hash was computed from, which is the object that is actually in storage.
//
// No new dependency: ffmpeg-static is already here for videoFingerprint, which
// already decodes frames out of this same file.
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const ffmpegPath = require('ffmpeg-static');

// 480px wide is enough for the largest tile any page draws (the invest detail
// pane) and small enough that the base64 of it belongs in a document. Height
// follows the source aspect ratio rather than being forced: a letterboxed
// thumbnail of a portrait phone video is worse than a tall one.
const WIDTH = 480;
// q=6 on ffmpeg's 2..31 scale. Visibly a photograph, about 20-40 KB.
const QUALITY = 6;
// A hard ceiling on what goes in the document. A 480px JPEG that lands over
// this is a pathological frame (noise, confetti) and is dropped rather than
// stored -- the placeholder is a better outcome than a bloated document.
const MAX_BYTES = 120 * 1024;
// Not frame zero. The first frame of a phone video is very often the lens still
// adjusting: black, blown out, or mid-autofocus. One second in is a picture of
// the field.
const SEEK_SECONDS = 1;
const TIMEOUT_MS = 20000;

function run(args) {
    return new Promise((resolve, reject) => {
        execFile(ffmpegPath, args, { timeout: TIMEOUT_MS, windowsHide: true },
            (err, stdout, stderr) => {
                if (err) {
                    const msg = (stderr || err.message || '').toString();
                    return reject(new Error(`ffmpeg failed: ${msg.slice(0, 300)}`));
                }
                resolve(stdout);
            });
    });
}

/**
 * A JPEG data URI for one frame of the video at `videoPath`, or null.
 *
 * Returns null rather than throwing on every failure mode -- a missing poster
 * is a cosmetic loss and must never fail an upload whose bytes are already
 * stored and hashed. The caller logs and carries on.
 */
async function posterDataUri(videoPath) {
    const out = path.join(os.tmpdir(),
        `pv_poster_${crypto.randomBytes(6).toString('hex')}.jpg`);
    try {
        // -ss BEFORE -i seeks by keyframe without decoding what it skips, which
        // is what makes this cheap. If the clip is shorter than the seek, ffmpeg
        // produces nothing -- so that case retries from the start.
        const attempts = [
            ['-v', 'error', '-ss', String(SEEK_SECONDS), '-i', videoPath],
            ['-v', 'error', '-i', videoPath],
        ];
        let wrote = false;
        for (const head of attempts) {
            try {
                await run([
                    ...head,
                    '-frames:v', '1',
                    // scale to width, height to the even number the aspect ratio
                    // implies: -2 rather than -1 because JPEG chroma subsampling
                    // needs even dimensions.
                    '-vf', `scale=${WIDTH}:-2`,
                    '-q:v', String(QUALITY),
                    '-f', 'image2',
                    '-y', out,
                ]);
            } catch { /* try the next head */ }
            if (fs.existsSync(out) && fs.statSync(out).size > 0) { wrote = true; break; }
        }
        if (!wrote) return null;

        const buf = await fs.promises.readFile(out);
        if (!buf.length || buf.length > MAX_BYTES) return null;
        return `data:image/jpeg;base64,${buf.toString('base64')}`;
    } catch {
        return null;
    } finally {
        try { await fs.promises.unlink(out); } catch { /* already gone */ }
    }
}

module.exports = { posterDataUri, WIDTH, MAX_BYTES };
