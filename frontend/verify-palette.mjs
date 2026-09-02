// Is the locked palette actually locked?
//
// It said it was, and the live boards carried eight near-duplicates of their own
// tokens across 26 uses. #ded7d0 sat beside the ground #ddd7d1 -- two channel
// values apart, invisible to anyone looking -- and #f2f3f4 stood in for the warm
// surface #f6f3ef, which is a COOL grey in a warm palette. Nobody can see either
// one on its own. That is exactly why they accumulate.
//
// Static: reads the boards, which are the source, so it needs no browser.
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

const TOKENS = {
  '#ddd7d1': 'ground', '#f6f3ef': 'surface', '#1d1a17': 'ink',
  '#006934': 'proved', '#016abe': 'action', '#01579b': 'link',
  '#a71930': 'alarm', '#f2e6cd': 'attention-fill', '#7c4a12': 'attention-ink',
  '#37322d': 'media', '#ffffff': 'white',
};
// Greys and tints the design uses deliberately and that are NOT within reach of a
// token: hairlines, disabled fills, the dark bands. Listed so the check can be
// strict about everything else.
const DELIBERATE = new Set([
  '#fff', '#000', '#4a443d', '#605a53', '#8c857e', '#c3bcb6', '#b9b1a9',
  '#e7e1db', '#eae4de', '#e3ddd6', '#e1e6ec', '#e6ebf0', '#f3f6f9',
  '#141210', '#0e0d0b', '#2a2622', '#012169', '#013f70', '#c9ced4', '#6b7278',
  '#f7e9e6', '#c4bdb6', '#d3ccc5', '#881337',
]);
const NEAR = 12;   // sum of channel deltas below which two colours are the same colour

const rgb = (h) => {
  let s = h.replace('#', '');
  if (s.length === 3) s = [...s].map(c => c + c).join('');
  return [0, 2, 4].map(i => parseInt(s.slice(i, i + 2), 16));
};
const dist = (a, b) => a.reduce((n, v, i) => n + Math.abs(v - b[i]), 0);
const tok = Object.fromEntries(Object.keys(TOKENS).map(k => [k, rgb(k)]));

// Only boards that actually produce a page. Limits.dc.html and Motion.dc.html
// are documentation -- they explain the rules rather than shipping a screen, and
// their pale green marks a passing example. Holding a document to the product's
// palette would be a finding about nothing.
const LIVE = readdirSync('design')
  .filter(f => f.endsWith('.dc.html'))
  .filter(f => /data-page="/.test(readFileSync(path.join('design', f), 'utf8')));
const findings = [];
let colours = 0;

for (const file of LIVE) {
  const src = readFileSync(path.join('design', file), 'utf8');
  // only the shipping panels, and only outside style blocks and comments
  const stripped = src.replace(/<style\b[\s\S]*?<\/style>|<!--[\s\S]*?-->|\/\*[\s\S]*?\*\//g, ' ');
  const seen = new Map();
  for (const m of stripped.matchAll(/#[0-9a-fA-F]{3,6}\b/g)) {
    const c = m[0].toLowerCase();
    if (c.length !== 4 && c.length !== 7) continue;
    seen.set(c, (seen.get(c) || 0) + 1);
  }
  for (const [c, n] of seen) {
    colours++;
    if (c in TOKENS || DELIBERATE.has(c)) continue;
    let v;
    try { v = rgb(c); } catch { continue; }
    for (const [t, tv] of Object.entries(tok)) {
      const d = dist(v, tv);
      if (d > 0 && d <= NEAR) {
        findings.push(`  ${file.padEnd(22)} ${c} x${String(n).padEnd(3)} is ${d} from `
          + `${t} (${TOKENS[t]}) — say ${t}`);
        break;
      }
    }
  }
}

for (const f of findings) console.log(f);
console.log(`\n  ${findings.length} colour(s) within ${NEAR} of a token without being it,`
  + ` across ${LIVE.length} boards`);
process.exit(findings.length ? 1 : 0);
