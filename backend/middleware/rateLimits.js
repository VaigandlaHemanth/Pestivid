/**
 * Rate limits.
 *
 * WHY THIS WAS URGENT
 *   There was no rate limiting anywhere in the backend, while POST
 *   /api/videos/upload accepts a 100 MB multipart body and writes it to disk
 *   before anything else happens. On a 512 MB free-tier instance with ephemeral
 *   disk, a single authenticated farmer account could fill the disk, exhaust the
 *   monthly bandwidth allowance, or keep every worker busy, from one laptop and a
 *   for-loop. The video pipeline also runs ffmpeg per upload and compares the
 *   fingerprint against every stored video, so each request costs real CPU.
 *
 *   Login needs a separate, tighter limit for a different reason: without one,
 *   password guessing against a known email address is free and unbounded.
 *
 * WHY LIMITS DIFFER PER ROUTE
 *   A limit tight enough to stop upload abuse would break normal browsing, and a
 *   limit loose enough for browsing does nothing against upload abuse. So the
 *   expensive endpoints get their own budgets.
 *
 * WHY KEYED ON USER WHERE POSSIBLE
 *   Rural users are frequently behind carrier-grade NAT, so an IP can represent a
 *   whole district. Keying uploads on the authenticated user id means one abusive
 *   account cannot lock out everyone sharing its IP. Unauthenticated routes have
 *   no choice but to key on IP.
 */

const rl = require('express-rate-limit');
const rateLimit = rl.rateLimit || rl.default || rl;
// Normalises an IPv6 address to its /64 prefix. Without it, a single IPv6 user
// typically controls 2^64 addresses and can defeat any IP-keyed limit by
// rotating within their own prefix -- express-rate-limit raises
// ERR_ERL_KEY_GEN_IPV6 precisely because a raw req.ip key is bypassable.
const ipKeyGenerator = rl.ipKeyGenerator || ((ip) => ip);

const MINUTE = 60 * 1000;
const HOUR = 60 * MINUTE;

/**
 * Prefer the authenticated user; fall back to a prefix-normalised IP.
 *
 * Keying on the user id matters for rural users: carrier-grade NAT can put a
 * whole district behind one address, so an IP-only key would let one abusive
 * account lock out everybody sharing it.
 */
const userOrIp = (req) => (
    (req.user && `u:${String(req.user._id)}`) || `ip:${ipKeyGenerator(req.ip)}`);

const message = (text) => ({ message: text });

/**
 * Uploads. Deliberately generous per hour but tight per minute: a farmer might
 * legitimately upload a handful of videos in a session, and should never
 * accidentally hit this, but nobody uploads 40 videos an hour by hand.
 */
const uploadLimiter = rateLimit({
    windowMs: HOUR,
    limit: Number(process.env.RATE_UPLOAD_PER_HOUR || 20),
    keyGenerator: userOrIp,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: message(
        'Too many uploads in the last hour. Please wait before uploading again. '
        + 'If you need to upload more, contact support.'),
});

const uploadBurstLimiter = rateLimit({
    windowMs: 5 * MINUTE,
    limit: Number(process.env.RATE_UPLOAD_BURST || 5),
    keyGenerator: userOrIp,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: message('Please wait a few minutes before uploading another video.'),
});

/**
 * Login and registration. Keyed on IP because there is no authenticated user
 * yet. skipSuccessfulRequests means a farmer logging in normally never consumes
 * budget -- only failures count, so this throttles guessing rather than use.
 */
const authLimiter = rateLimit({
    windowMs: 15 * MINUTE,
    limit: Number(process.env.RATE_AUTH_PER_15MIN || 10),
    skipSuccessfulRequests: true,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: message(
        'Too many failed sign-in attempts. Please wait 15 minutes and try again.'),
});

/**
 * The public verification endpoints. They are unauthenticated by design, and each
 * anchor lookup rebuilds a Merkle tree, so they need a ceiling. Set high enough
 * that a person checking several videos never notices.
 */
const publicReadLimiter = rateLimit({
    windowMs: MINUTE,
    limit: Number(process.env.RATE_PUBLIC_PER_MIN || 60),
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: message('Too many requests. Please slow down.'),
});

/** Everything else: a backstop, loose enough to be invisible in normal use. */
const generalLimiter = rateLimit({
    windowMs: MINUTE,
    limit: Number(process.env.RATE_GENERAL_PER_MIN || 300),
    keyGenerator: userOrIp,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: message('Too many requests. Please slow down.'),
});

module.exports = {
    uploadLimiter,
    uploadBurstLimiter,
    authLimiter,
    publicReadLimiter,
    generalLimiter,
};
