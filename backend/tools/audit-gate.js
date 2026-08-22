#!/usr/bin/env node
/**
 * `npm audit` as a CI gate, with one narrowly scoped exception.
 *
 * Plain `npm audit --audit-level=moderate` exits 1 on this project and always
 * will, so as a gate it is worse than useless: a check that can never pass gets
 * ignored, and then it stops reporting the advisories that DO matter.
 *
 * Every current advisory sits in the dependency subtree of `opentimestamps`,
 * which is how videos get their Bitcoin timestamp. That package was last
 * published in 2022 and reaches the calendar servers through the deprecated
 * `request` stack. There is no maintained replacement -- the only other
 * candidate, javascript-opentimestamps, is older and additionally pulls in web3 --
 * and `npm audit fix` "resolves" it by removing the package, which would remove
 * the anchoring feature.
 *
 * What the exposure actually is:
 *   request SSRF        -- needs an attacker-controlled URL. We only ever POST to
 *                          the library's hardcoded calendar list.
 *   form-data CRLF      -- the vulnerable copy is form-data@2.3.3 under `request`.
 *                          Our OWN uses (Pinata pinning, the AI service) are on
 *                          the patched 4.0.6, which is why this gate compares
 *                          VERSIONS and not just package names.
 *   qs arrayLimit DoS   -- no user input reaches it.
 *   tough-cookie proto  -- cookies from calendar servers only.
 *   bn.js infinite loop -- the one genuinely fed remote bytes, when bitcore-lib
 *                          parses a calendar's attestation. That is why
 *                          services/anchor.js wraps every stamp/upgrade/verify
 *                          call in withTimeout(): a hostile or hanging calendar
 *                          costs one skipped batch, not the anchor job.
 *
 * So the decision is to accept that subtree and gate on everything else. The
 * acceptance is deliberately narrow:
 *
 *   - reachability is computed from the INSTALLED tree (`npm ls --all --json`),
 *     not from npm audit's `effects`, which does not chain reliably to real
 *     top-level dependencies;
 *   - an advisory counts as accepted only if EVERY installed instance matching
 *     its version range sits under `opentimestamps`;
 *   - a package name that appears there for the first time still fails, so
 *     "we accepted this package" cannot drift into "we accept anything under it".
 *
 *     node tools/audit-gate.js
 */

const { execFileSync } = require('child_process');
const semver = require('semver');

const ACCEPTED_ROOT = 'opentimestamps';

// Reviewed advisories inside that subtree (see the header for the reasoning).
const REVIEWED = new Set([
    'bitcore-lib', 'bn.js', 'form-data', 'opentimestamps', 'qs',
    'request', 'request-promise', 'request-promise-core', 'tough-cookie', 'uuid',
]);

const FAIL_AT = ['moderate', 'high', 'critical'];
const WIN = process.platform === 'win32';

function npmJson(args) {
    try {
        return JSON.parse(execFileSync('npm', args, {
            encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, shell: WIN,
        }));
    } catch (err) {
        // npm audit and npm ls both exit non-zero when they have something to
        // report, so the JSON we want is on stdout even on "failure".
        if (err.stdout) return JSON.parse(err.stdout);
        throw err;
    }
}

/**
 * Every installed instance of every package, as name -> [{version, rootPath}].
 * rootPath is the chain of top-level dependency names it hangs from.
 */
function installedInstances() {
    const tree = npmJson(['ls', '--all', '--json']);
    const found = new Map();

    const walk = (node, chain) => {
        for (const [name, dep] of Object.entries(node.dependencies || {})) {
            if (!dep || dep.version === undefined) continue;
            const nextChain = chain.length ? chain : [name];  // first hop = the root
            if (!found.has(name)) found.set(name, []);
            found.get(name).push({ version: dep.version, root: nextChain[0] });
            // Guard against the cycles npm's JSON can contain.
            if (dep.dependencies && chain.length < 24) walk(dep, nextChain);
        }
    };
    walk(tree, []);
    return found;
}

const report = npmJson(['audit', '--json']);
const vulns = report.vulnerabilities || {};
const installed = installedInstances();

// --self-test proves the gate can still FAIL. A guard that cannot fail is
// indistinguishable from no guard, and this one is a single boolean away from
// rubber-stamping everything -- so the ability to block is tested, not assumed.
const SELF_TEST = process.argv.includes('--self-test');

if (SELF_TEST) {
    // A critical advisory on express: a DIRECT dependency, not under
    // opentimestamps. The gate must block it.
    if (!(installed.get('express') || []).length) {
        console.error('self-test needs express installed');
        process.exit(1);
    }
    vulns.express = {
        severity: 'critical',
        via: [{ name: 'express', range: '*', severity: 'critical', title: 'synthetic' }],
    };
    // A NEW name inside the accepted subtree must land in NEEDS REVIEW rather
    // than being waved through with the rest of that subtree.
    vulns['bitcore-message'] = {
        severity: 'high',
        via: [{ name: 'bitcore-message', range: '*', severity: 'high', title: 'synthetic' }],
    };
}

const blocking = [];
const accepted = [];
const unreviewed = [];

for (const [name, v] of Object.entries(vulns)) {
    if (!FAIL_AT.includes(v.severity)) continue;

    // The version ranges this advisory actually covers.
    const ranges = (v.via || [])
        .filter((x) => typeof x === 'object' && x.range)
        .map((x) => x.range);

    const instances = installed.get(name) || [];
    const affected = instances.filter((i) => (
        ranges.length === 0
            ? true                                   // no range given: assume all
            : ranges.some((r) => {
                try { return semver.satisfies(i.version, r, { includePrerelease: true }); }
                catch { return true; }               // unparseable range: be strict
            })
    ));

    // Which top-level dependencies is a VULNERABLE copy reachable from?
    const roots = [...new Set(affected.map((i) => i.root))];

    const detail = { name, severity: v.severity, roots, versions: [...new Set(affected.map((i) => i.version))] };

    if (roots.length === 0) {
        // Reported but no matching installed copy -- do not silently pass it.
        unreviewed.push({ ...detail, note: 'no installed copy matched the advisory range' });
    } else if (roots.every((r) => r === ACCEPTED_ROOT)) {
        (REVIEWED.has(name) ? accepted : unreviewed).push(detail);
    } else {
        blocking.push(detail);
    }
}

const m = report.metadata && report.metadata.vulnerabilities;
if (m) console.log(`npm audit totals: ${JSON.stringify(m)}`);

if (accepted.length) {
    console.log(`\nAccepted -- vulnerable copy exists only under ${ACCEPTED_ROOT},`);
    console.log(`and reviewed in tools/audit-gate.js:`);
    for (const a of accepted) {
        console.log(`  ${a.severity.padEnd(9)} ${a.name}@${a.versions.join(',')}`);
    }
}

let bad = false;

if (unreviewed.length) {
    bad = true;
    console.log(`\nNEEDS REVIEW -- new advisory in the accepted subtree, or no`);
    console.log(`installed copy matched. Review and add to REVIEWED, or fix:`);
    for (const u of unreviewed) {
        console.log(`  ${u.severity.padEnd(9)} ${u.name}  ${u.note || `roots: ${u.roots.join(', ')}`}`);
    }
}

if (blocking.length) {
    bad = true;
    console.log(`\nBLOCKING -- a vulnerable copy is reachable outside ${ACCEPTED_ROOT}:`);
    for (const b of blocking) {
        console.log(`  ${b.severity.padEnd(9)} ${b.name}@${b.versions.join(',')}`
            + `  (via: ${b.roots.join(', ')})`);
    }
    console.log(`\nRun \`npm audit\` for detail. Fix these, or justify them here.`);
}

if (!bad) {
    console.log(`\naudit gate passed -- nothing vulnerable outside the reviewed`);
    console.log(`${ACCEPTED_ROOT} subtree.`);
}
if (SELF_TEST) {
    const caughtOutside = blocking.some((b) => b.name === 'express');
    const caughtNewInside = unreviewed.some((u) => u.name === 'bitcore-message');
    if (!caughtOutside) {
        console.error('SELF-TEST FAILED: a critical advisory on a DIRECT dependency');
        console.error('was not blocked. This gate would pass anything.');
        process.exit(2);
    }
    if (!caughtNewInside) {
        console.error('SELF-TEST FAILED: a new package inside the accepted subtree');
        console.error('was waved through instead of flagged for review.');
        process.exit(2);
    }
    console.log('self-test passed: the gate blocks outside advisories and flags');
    console.log('new names inside the accepted subtree.');
    process.exit(0);
}

process.exit(bad ? 1 : 0);
