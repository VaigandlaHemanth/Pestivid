/**
 * Merkle tree tests, including the two attacks the design exists to stop.
 *
 *     node test_merkle.js
 */

const assert = require('assert');
const crypto = require('crypto');

const m = require('./services/merkle');

let pass = 0;
let fail = 0;

function t(name, fn) {
    try {
        fn();
        console.log(`  PASS  ${name}`);
        pass++;
    } catch (e) {
        console.log(`  FAIL  ${name}\n          ${e.message}`);
        fail++;
    }
}

const rec = (i) => m.canonicalRecord({
    cid: `bafy${i}`,
    sha256: crypto.createHash('sha256').update(`v${i}`).digest('hex'),
    farmerId: `farmer${i % 3}`,
    uploadedAt: new Date(Date.UTC(2026, 0, 1 + i)).toISOString(),
});

// ── basics ──────────────────────────────────────────────────────────────────
t('single leaf: root equals the leaf', () => {
    const l = m.leafHash(rec(0));
    assert.ok(m.buildTree([l]).root.equals(l));
});

t('inclusion proof verifies for every leaf, at many tree sizes', () => {
    // Odd sizes and powers of two both matter: promotion only happens on odd
    // levels, so a tree of 2 or 4 would never exercise it.
    for (const n of [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 100]) {
        const leaves = Array.from({ length: n }, (_, i) => m.leafHash(rec(i)));
        const { root, levels } = m.buildTree(leaves);
        for (let i = 0; i < n; i++) {
            const proof = m.inclusionProof(levels, i);
            assert.ok(m.verifyProofFromHash(leaves[i], proof, root),
                `proof failed for leaf ${i} of ${n}`);
        }
    }
});

t('a proof for the wrong leaf does NOT verify', () => {
    const leaves = Array.from({ length: 9 }, (_, i) => m.leafHash(rec(i)));
    const { root, levels } = m.buildTree(leaves);
    const proof = m.inclusionProof(levels, 3);
    assert.ok(!m.verifyProofFromHash(leaves[4], proof, root));
});

t('tampering with any proof step breaks verification', () => {
    const leaves = Array.from({ length: 8 }, (_, i) => m.leafHash(rec(i)));
    const { root, levels } = m.buildTree(leaves);
    const proof = m.inclusionProof(levels, 5);
    for (let k = 0; k < proof.length; k++) {
        const bad = proof.map((p, j) => (j === k
            ? { ...p, hash: p.hash.replace(/^./, (c) => (c === '0' ? '1' : '0')) }
            : p));
        assert.ok(!m.verifyProofFromHash(leaves[5], bad, root), `step ${k} tamper undetected`);
    }
});

t('flipping a proof step side breaks verification', () => {
    const leaves = Array.from({ length: 8 }, (_, i) => m.leafHash(rec(i)));
    const { root, levels } = m.buildTree(leaves);
    const proof = m.inclusionProof(levels, 2);
    const flipped = proof.map((p) => ({ ...p, side: p.side === 'left' ? 'right' : 'left' }));
    assert.ok(!m.verifyProofFromHash(leaves[2], flipped, root));
});

// ── the two attacks the design defends against ──────────────────────────────
t('domain separation: leaf and node hashes differ over the same bytes', () => {
    // This is what actually stops a second-preimage attack. Without the 0x00/0x01
    // prefixes, SHA256(L||R) would be valid as both a leaf and an internal node,
    // so an internal node could be passed off as logged data.
    const L = crypto.randomBytes(32);
    const R = crypto.randomBytes(32);
    const asLeaf = m.leafHash(Buffer.concat([L, R])).toString('hex');
    const asNode = m.nodeHash(L, R).toString('hex');
    assert.notStrictEqual(asLeaf, asNode, 'prefixes are not domain-separating');
});

t('ATTACK second-preimage: no RECORD can pose as an internal node', () => {
    // verifyRecord hashes the record itself, so an internal node cannot be
    // supplied as a leaf. An attacker would need data D with
    // SHA256(0x00||D) == SHA256(0x01||L||R), i.e. a SHA-256 collision.
    const leaves = [0, 1, 2, 3].map((i) => m.leafHash(rec(i)));
    const { root, levels } = m.buildTree(leaves);
    const internalHex = levels[1][0].toString('hex');
    const forged = m.inclusionProof(levels, 0).slice(1);
    // Try passing the internal node value as the record itself, in both encodings.
    assert.ok(!m.verifyRecord(internalHex, forged, root));
    assert.ok(!m.verifyRecord(levels[1][0], forged, root));
});

t('verifyRecord accepts a genuine record and rejects a modified one', () => {
    const records = [0, 1, 2, 3, 4].map(rec);
    const leaves = records.map((r) => m.leafHash(r));
    const { root, levels } = m.buildTree(leaves);
    const proof = m.inclusionProof(levels, 2);
    assert.ok(m.verifyRecord(records[2], proof, root), 'genuine record failed');
    const tampered = records[2].replace('bafy2', 'bafy9');
    assert.ok(!m.verifyRecord(tampered, proof, root), 'tampered record verified');
});

t('ATTACK CVE-2012-2459: two different leaf sets cannot share a root', () => {
    // Duplicating the last leaf on an odd level would make [a,b,c] and
    // [a,b,c,c] produce the same root. Promotion prevents it.
    const a = m.leafHash(rec(0));
    const b = m.leafHash(rec(1));
    const c = m.leafHash(rec(2));
    const r3 = m.buildTree([a, b, c]).root.toString('hex');
    const r4 = m.buildTree([a, b, c, c]).root.toString('hex');
    assert.notStrictEqual(r3, r4,
        '[a,b,c] and [a,b,c,c] gave the same root — the tree is malleable');
});

// ── canonicalisation ────────────────────────────────────────────────────────
t('canonicalRecord is stable and order-independent in its input', () => {
    const base = {
        cid: 'bafyX',
        sha256: 'AABBCC',
        farmerId: 'f1',
        uploadedAt: '2026-01-02T03:04:05.000Z',
    };
    const one = m.canonicalRecord(base);
    const two = m.canonicalRecord({
        uploadedAt: base.uploadedAt, farmerId: base.farmerId,
        sha256: base.sha256, cid: base.cid,
    });
    assert.strictEqual(one, two, 'field order in the input changed the output');
    assert.ok(one.includes('"sha256":"aabbcc"'), 'sha256 must be lower-cased');
});

t('canonicalRecord normalises equivalent timestamps', () => {
    const x = m.canonicalRecord({ cid: 'c', sha256: 'ab', farmerId: 'f', uploadedAt: '2026-01-02T03:04:05Z' });
    const y = m.canonicalRecord({ cid: 'c', sha256: 'ab', farmerId: 'f', uploadedAt: new Date('2026-01-02T03:04:05Z') });
    assert.strictEqual(x, y);
});

t('canonicalRecord refuses incomplete input', () => {
    assert.throws(() => m.canonicalRecord({ cid: 'c', sha256: 'ab', farmerId: 'f' }));
    assert.throws(() => m.canonicalRecord({}));
});

t('changing any single field changes the leaf', () => {
    const base = { cid: 'c1', sha256: 'ab', farmerId: 'f1', uploadedAt: '2026-01-01T00:00:00.000Z' };
    const h0 = m.leafHash(m.canonicalRecord(base)).toString('hex');
    for (const [k, v] of Object.entries({
        cid: 'c2', sha256: 'ac', farmerId: 'f2', uploadedAt: '2026-01-01T00:00:01.000Z',
    })) {
        const h = m.leafHash(m.canonicalRecord({ ...base, [k]: v })).toString('hex');
        assert.notStrictEqual(h, h0, `changing ${k} did not change the leaf`);
    }
});

t('empty tree is refused rather than returning a fake root', () => {
    assert.throws(() => m.buildTree([]));
});

t('out-of-range leaf index is refused', () => {
    const leaves = [m.leafHash(rec(0)), m.leafHash(rec(1))];
    const { levels } = m.buildTree(leaves);
    assert.throws(() => m.inclusionProof(levels, 2));
    assert.throws(() => m.inclusionProof(levels, -1));
});

t('proof size grows logarithmically (1000 leaves -> <= 10 steps)', () => {
    const leaves = Array.from({ length: 1000 }, (_, i) => m.leafHash(rec(i)));
    const { levels } = m.buildTree(leaves);
    const len = m.inclusionProof(levels, 500).length;
    assert.ok(len <= 10, `proof was ${len} steps`);
    console.log(`          1000 leaves -> ${len}-step proof`);
});

console.log(`\n${pass}/${pass + fail} checks passed`);
process.exit(fail ? 1 : 0);
