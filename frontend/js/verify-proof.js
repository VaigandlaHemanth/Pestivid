/**
 * Client-side verification of a video's Merkle inclusion proof.
 *
 * ONE IMPLEMENTATION, TWO CALLERS
 *   Used by the in-app "check proof" button (good UX for a logged-in investor)
 *   and by the standalone /verify.html page (works with no login, and is a URL
 *   you can paste into a dispute or an audit trail). Duplicating the folding
 *   logic across the two would create exactly the kind of silent divergence that
 *   makes one of them wrong without erroring — so it lives here once.
 *
 * WHY VERIFY IN THE CLIENT AT ALL
 *   The server already self-checks and returns selfCheck: true. That is worth
 *   nothing to a sceptic: the party storing the data is asserting the data is
 *   fine. Recomputing here means the arithmetic runs on the reader's machine, so
 *   a server that returned a proof which does not fold to its claimed root is
 *   caught rather than believed.
 *
 *   This is not a full trust escape — the page is served from the same origin as
 *   the app, so the same operator controls both. What it does give is a check
 *   that can be independently reproduced with any other SHA-256 tool, and the
 *   step-by-step values needed to do so.
 *
 * RFC 6962 DOMAIN SEPARATION
 *   leaf     = SHA256(0x00 ‖ record)
 *   internal = SHA256(0x01 ‖ left ‖ right)
 *   The one-byte prefixes must match backend/services/merkle.js exactly. Parity
 *   is checked in CI over many tree sizes; if you change one side, change both.
 */
(function (global) {
  'use strict';

  const LEAF = new Uint8Array([0x00]);
  const NODE = new Uint8Array([0x01]);

  const toHex = (buf) => Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0')).join('');

  function hexToBytes(h) {
    if (typeof h !== 'string' || !/^([0-9a-fA-F]{2})*$/.test(h)) {
      throw new Error(`not a hex string: ${String(h).slice(0, 24)}`);
    }
    const out = new Uint8Array(h.length / 2);
    for (let i = 0; i < out.length; i++) {
      out[i] = parseInt(h.substr(i * 2, 2), 16);
    }
    return out;
  }

  async function sha256(...parts) {
    let total = 0;
    for (const p of parts) total += p.length;
    const buf = new Uint8Array(total);
    let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return new Uint8Array(await crypto.subtle.digest('SHA-256', buf));
  }

  const leafHash = (record) =>
    sha256(LEAF, new TextEncoder().encode(record));

  const nodeHash = (l, r) => sha256(NODE, l, r);

  /**
   * Fold `record` through `path` and report the resulting root plus every
   * intermediate value, so a reader can follow or reproduce the computation.
   */
  async function fold(record, path) {
    if (typeof record !== 'string' || !record) {
      throw new Error('record must be the exact canonical string that was hashed');
    }
    if (!Array.isArray(path)) throw new Error('inclusionProof must be an array');

    let acc = await leafHash(record);
    const steps = [{
      label: 'leaf = SHA256(0x00 ‖ record)',
      value: toHex(acc),
    }];

    for (const s of path) {
      if (!s || (s.side !== 'left' && s.side !== 'right')) {
        throw new Error(`proof step has an invalid side: ${JSON.stringify(s)}`);
      }
      const sib = hexToBytes(s.hash);
      acc = s.side === 'left' ? await nodeHash(sib, acc) : await nodeHash(acc, sib);
      steps.push({
        label: s.side === 'left'
          ? 'SHA256(0x01 ‖ sibling ‖ acc)'
          : 'SHA256(0x01 ‖ acc ‖ sibling)',
        value: toHex(acc),
        sibling: s.hash,
      });
    }
    return { root: toHex(acc), steps };
  }

  /**
   * Full check of an /anchor response.
   * Returns { valid, recomputedRoot, claimedRoot, steps, confirmed, reason }.
   */
  async function check(anchorResponse) {
    const a = anchorResponse || {};
    if (!a.anchored) {
      return {
        valid: null,                 // null = nothing to verify yet, not a failure
        pending: true,
        reason: a.reason || 'This video has not been anchored yet.',
      };
    }
    const { root, steps } = await fold(a.record, a.inclusionProof);
    const valid = root === a.merkleRoot;
    return {
      valid,
      pending: false,
      recomputedRoot: root,
      claimedRoot: a.merkleRoot,
      steps,
      batchSize: a.batchSize,
      // 'anchored' means a Bitcoin block includes it. 'pending' is the normal
      // state for a few hours after stamping, not a problem.
      confirmed: a.status === 'anchored',
      bitcoinBlockHeight: a.bitcoinBlockHeight || null,
      bitcoinTimestamp: a.bitcoinTimestamp || null,
      record: a.record,
      howToVerify: a.howToVerify || [],
      reason: valid
        ? 'The record is provably part of the anchored batch.'
        : 'The proof does not fold to the root it claims. Do not rely on this record.',
    };
  }

  /** Fetch and check in one call. `api` defaults to this origin's /api. */
  async function fetchAndCheck(cid, api) {
    const base = api || `${global.location.origin}/api`;
    const res = await fetch(`${base}/videos/${encodeURIComponent(cid)}/anchor`);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.message || `HTTP ${res.status}`);
    return check(body);
  }

  global.VerifyProof = { fold, check, fetchAndCheck, leafHash, nodeHash, toHex, hexToBytes };
})(typeof window !== 'undefined' ? window : globalThis);
