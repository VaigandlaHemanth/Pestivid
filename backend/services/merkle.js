/**
 * Merkle tree over video records, RFC 6962 (Certificate Transparency) style.
 *
 * WHY A TREE AND NOT ONE ANCHOR PER VIDEO
 *   Anchoring every video separately would mean one Bitcoin timestamp per upload.
 *   Instead we hash all of a batch into a single root and anchor THAT once. One
 *   free daily anchor then covers unlimited videos, and each video gets an
 *   inclusion proof — a short list of sibling hashes — that anyone can check
 *   against the anchored root without trusting us or seeing anyone else's data.
 *
 * WHY RFC 6962 AND NOT NAIVE CONCATENATION
 *   The obvious implementation, H(left || right) with leaves hashed the same way,
 *   is vulnerable to second-preimage attacks: an internal node's hash can be
 *   presented as a leaf, letting someone prove inclusion of data that was never
 *   in the log. RFC 6962 prevents that by DOMAIN-SEPARATING the two cases:
 *
 *       leaf hash      = SHA256(0x00 || data)
 *       internal hash  = SHA256(0x01 || left || right)
 *
 *   The one-byte prefix makes a leaf and an interior node structurally different,
 *   so neither can impersonate the other.
 *
 * ODD NODES
 *   A level with an odd count PROMOTES the last node unchanged rather than
 *   duplicating it. Duplicating the final leaf is the well-known CVE-2012-2459
 *   Bitcoin malleability bug: two different leaf sets can produce the same root.
 *   Promotion keeps the root a unique function of the leaf list.
 */

const crypto = require('crypto');

const LEAF_PREFIX = Buffer.from([0x00]);
const NODE_PREFIX = Buffer.from([0x01]);

const sha256 = (...bufs) => crypto.createHash('sha256').update(Buffer.concat(bufs)).digest();

/** Leaf hash of a record. Domain-separated from internal nodes. */
function leafHash(data) {
    return sha256(LEAF_PREFIX, Buffer.isBuffer(data) ? data : Buffer.from(data, 'utf8'));
}

/** Internal node hash. */
function nodeHash(left, right) {
    return sha256(NODE_PREFIX, left, right);
}

/**
 * Canonical serialisation of a video record for hashing.
 *
 * Field ORDER and formatting are fixed here on purpose. If two parties serialise
 * the same record differently they compute different leaves and the proof fails
 * for no visible reason, so this is the single definition and it must not change
 * without a version bump.
 */
function canonicalRecord({ cid, sha256: hash, farmerId, uploadedAt }) {
    if (!cid || !hash || !farmerId || !uploadedAt) {
        throw new Error('canonicalRecord requires cid, sha256, farmerId and uploadedAt');
    }
    return JSON.stringify({
        v: 1,
        cid: String(cid),
        sha256: String(hash).toLowerCase(),
        farmerId: String(farmerId),
        uploadedAt: new Date(uploadedAt).toISOString(),
    });
}

/**
 * Build a tree from ordered leaf hashes.
 * Returns { root, levels } where levels[0] is the leaves.
 */
function buildTree(leaves) {
    if (!leaves.length) throw new Error('Cannot build a Merkle tree with no leaves');
    const levels = [leaves.slice()];
    while (levels[levels.length - 1].length > 1) {
        const cur = levels[levels.length - 1];
        const next = [];
        for (let i = 0; i < cur.length; i += 2) {
            if (i + 1 < cur.length) {
                next.push(nodeHash(cur[i], cur[i + 1]));
            } else {
                // Promote, do not duplicate. See the note on CVE-2012-2459 above.
                next.push(cur[i]);
            }
        }
        levels.push(next);
    }
    return { root: levels[levels.length - 1][0], levels };
}

/**
 * Inclusion proof for leaf `index`: the sibling hashes needed to recompute the
 * root, each tagged with the side it sits on.
 */
function inclusionProof(levels, index) {
    if (index < 0 || index >= levels[0].length) {
        throw new Error(`leaf index ${index} out of range (${levels[0].length} leaves)`);
    }
    const path = [];
    let i = index;
    for (let d = 0; d < levels.length - 1; d++) {
        const level = levels[d];
        const isRight = i % 2 === 1;
        const siblingIndex = isRight ? i - 1 : i + 1;
        if (siblingIndex < level.length) {
            path.push({
                side: isRight ? 'left' : 'right',
                hash: level[siblingIndex].toString('hex'),
            });
        }
        // A promoted node has no sibling at this level and keeps its position.
        i = Math.floor(i / 2);
    }
    return path;
}

/**
 * Verify that a RECORD is in the log. This is the function an outside verifier
 * should use; it needs no access to the log itself.
 *
 * It takes the record and derives the leaf hash internally, rather than accepting
 * a leaf hash from the caller. That distinction matters: verifyProofFromHash
 * below cannot tell whether the hash it was handed came from leafHash or
 * nodeHash, so a caller could hand it an INTERNAL NODE and get a true result --
 * proving inclusion of something that is not a record. Domain separation means an
 * attacker still cannot produce data hashing to an internal node, so this was
 * never a break of the tree, but an API that makes the mistake possible is an API
 * that will eventually be misused. Hashing here removes the option.
 */
function verifyRecord(record, path, expectedRoot) {
    return verifyProofFromHash(leafHash(record), path, expectedRoot);
}

/**
 * Low-level: recompute a root from a leaf HASH and its proof.
 *
 * Prefer verifyRecord(). Only use this when the leaf hash is already known to
 * have come from leafHash() -- for example inside this service when replaying a
 * stored batch.
 */
function verifyProofFromHash(leaf, path, expectedRoot) {
    let acc = Buffer.isBuffer(leaf) ? leaf : Buffer.from(leaf, 'hex');
    for (const step of path) {
        const sib = Buffer.from(step.hash, 'hex');
        acc = step.side === 'left' ? nodeHash(sib, acc) : nodeHash(acc, sib);
    }
    const want = Buffer.isBuffer(expectedRoot)
        ? expectedRoot
        : Buffer.from(expectedRoot, 'hex');
    return acc.equals(want);
}

module.exports = {
    leafHash,
    nodeHash,
    canonicalRecord,
    buildTree,
    inclusionProof,
    verifyRecord,
    verifyProofFromHash,
    LEAF_PREFIX,
    NODE_PREFIX,
};
