// Two claims on the motion board are checkable, so check them.
//   1. the dot on screen follows the analytic spring, not something that
//      merely resembles it
//   2. running every animation on the page costs zero layouts
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const DESIGN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');

const solve = (response, zeta) => {
  const w0 = (2 * Math.PI) / response;
  return t => {
    if (zeta < 1) {
      const wd = w0 * Math.sqrt(1 - zeta * zeta);
      return 1 - Math.exp(-zeta * w0 * t) * (Math.cos(wd * t) + (zeta * w0 / wd) * Math.sin(wd * t));
    }
    if (zeta === 1) return 1 - Math.exp(-w0 * t) * (1 + w0 * t);
    const a = w0 * Math.sqrt(zeta * zeta - 1);
    return 1 - Math.exp(-zeta * w0 * t) * (Math.cosh(a * t) + ((zeta * w0) / a) * Math.sinh(a * t));
  };
};
const SPRINGS = { smooth: [0.5, 1.0], snappy: [0.4, 0.85], bouncy: [0.5, 0.7], press: [0.25, 1.0], sheet: [0.55, 0.9] };
const CYCLE = 2400, TRAVEL = 196;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
await page.goto(pathToFileURL(path.join(DESIGN, 'Motion.dc.html')).href, { waitUntil: 'load' });
await page.addScriptTag({ content: shim });
await page.evaluate(() => document.fonts.ready);

// ---- 1. does the pixel match the physics -------------------------------
const settle = {};
for (const [name, [r, z]] of Object.entries(SPRINGS)) {
  const x = solve(r, z);
  let dur = 0;
  for (let t = 0; t < 6; t += 0.002) if (Math.abs(1 - x(t)) > 0.001) dur = t;
  settle[name] = dur + 0.012;
}

const samples = await page.evaluate(async ({ CYCLE, TRAVEL }) => {
  const out = {};
  const dots = [...document.querySelectorAll('.dot')].filter(d => /run-/.test(getComputedStyle(d).animationName));
  for (const d of dots) {
    const name = getComputedStyle(d).animationName.replace('run-', '');
    if (out[name]) continue;
    const anims = d.getAnimations();
    if (!anims.length) { out[name] = null; continue; }
    const a = anims[0];
    a.pause();
    const rows = [];
    for (let i = 0; i <= 24; i++) {
      const ms = (i / 24) * CYCLE * 0.31;      // the outbound leg
      a.currentTime = ms;
      const m = new DOMMatrixReadOnly(getComputedStyle(d).transform);
      rows.push([ms, m.m41 / TRAVEL]);
    }
    a.play();
    out[name] = rows;
  }
  return out;
}, { CYCLE, TRAVEL });

console.log('1. does the motion on screen match the spring solved on paper');
let worst = 0, checked = 0;
for (const [name, rows] of Object.entries(samples)) {
  if (!rows) { console.log(`   ${name.padEnd(7)} NO ANIMATION FOUND`); continue; }
  const [r, z] = SPRINGS[name];
  const x = solve(r, z), dur = settle[name];
  let maxErr = 0;
  for (const [ms, seen] of rows) {
    const frac = Math.min(ms / 1000 / dur, 1);          // keyframe segment is the settle window
    const want = frac >= 1 ? 1 : x(frac * dur);
    maxErr = Math.max(maxErr, Math.abs(seen - want));
  }
  worst = Math.max(worst, maxErr); checked++;
  const verdict = maxErr < 0.02 ? 'matches' : maxErr < 0.05 ? 'close' : 'DIVERGES';
  console.log(`   ${name.padEnd(7)} settle ${String(Math.round(dur * 1000)).padStart(4)}ms   worst error ${(maxErr * 100).toFixed(2)}% of travel   ${verdict}`);
}
console.log(`   ${checked} curves checked, worst deviation ${(worst * 100).toFixed(2)}% of travel\n`);

// ---- 2. does any of it cost layout -------------------------------------
const cdp = await page.context().newCDPSession(page);
await cdp.send('Performance.enable');
const read = async () => Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
const before = await read();
await page.waitForTimeout(4000);
const after = await read();

const running = await page.evaluate(() => document.getAnimations().filter(a => a.playState === 'running').length);
console.log('2. what four seconds of every animation on the page costs');
console.log(`   animations running concurrently  ${running}`);
for (const k of ['LayoutCount', 'RecalcStyleCount', 'LayoutDuration', 'RecalcStyleDuration', 'ScriptDuration']) {
  const d = (after[k] ?? 0) - (before[k] ?? 0);
  const unit = k.endsWith('Duration') ? `${(d * 1000).toFixed(1)} ms` : `${d}`;
  console.log(`   ${k.padEnd(21)} ${unit}`);
}
const layouts = (after.LayoutCount ?? 0) - (before.LayoutCount ?? 0);
console.log(`   => ${layouts === 0 ? 'zero layouts: every animation is compositor-only' : layouts + ' layouts triggered — something is animating a layout property'}`);

// ---- 3. properties actually being animated ------------------------------
const props = await page.evaluate(() => {
  const s = new Set();
  for (const a of document.getAnimations()) {
    const kf = a.effect?.getKeyframes?.() || [];
    for (const k of kf) for (const p of Object.keys(k))
      if (!['offset', 'computedOffset', 'easing', 'composite'].includes(p)) s.add(p);
  }
  return [...s].sort();
});
console.log(`\n3. properties under animation: ${props.join(', ')}`);
const banned = props.filter(p => !['transform', 'opacity'].includes(p));
console.log(`   ${banned.length ? 'OFF-COMPOSITOR: ' + banned.join(', ') : 'transform and opacity only, as specified'}`);

await browser.close();
