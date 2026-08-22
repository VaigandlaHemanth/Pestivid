/**
 * Generate test video clips for manual/live testing.
 *
 * Lives in the repo rather than a scratch dir for two reasons: it needs to
 * resolve ffmpeg-static from backend/node_modules, and inline `node -e` with
 * Windows paths gets mangled by shell escaping (a literal \f in a path becomes a
 * formfeed).
 *
 *   node tools/make_test_clip.js <outDir> new    <name>   fresh synthetic clip
 *   node tools/make_test_clip.js <outDir> copy   <src> <name>   re-encoded copy
 */

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const ff = require('ffmpeg-static');

const [outDir, mode, a, b] = process.argv.slice(2);
if (!outDir || !mode) {
    console.error('usage: make_test_clip.js <outDir> new|copy ...');
    process.exit(2);
}
fs.mkdirSync(outDir, { recursive: true });

function run(args) {
    execFileSync(ff, ['-v', 'error', '-y', ...args], { stdio: 'pipe' });
}

if (mode === 'new') {
    const name = a || 'clip.mp4';
    const pattern = b || 'testsrc2';
    const out = path.join(outDir, name);
    run(['-f', 'lavfi', '-i', `${pattern}=size=480x360:rate=15:duration=5`,
        '-pix_fmt', 'yuv420p', out]);
    console.log(`${out}\t${(fs.statSync(out).size / 1024).toFixed(0)} KB`);
} else if (mode === 'copy') {
    // A re-encode at a different bitrate: every byte changes, so SHA-256 no
    // longer matches, which is exactly the laundering the perceptual
    // fingerprint has to survive.
    const src = path.join(outDir, a);
    const out = path.join(outDir, b || 'copy.mp4');
    run(['-i', src, '-b:v', '150k', '-pix_fmt', 'yuv420p', out]);
    console.log(`${out}\t${(fs.statSync(out).size / 1024).toFixed(0)} KB  (re-encoded copy of ${a})`);
} else {
    console.error(`unknown mode: ${mode}`);
    process.exit(2);
}
