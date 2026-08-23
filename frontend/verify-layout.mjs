// Compares the geometry of every element, not the pixels.
//
// Pixel diffing kept arguing with itself: a panel that lands half a pixel over
// re-renders every glyph edge, and no threshold cleanly separates that from
// something that actually moved. Element boxes have no such ambiguity. Walk the
// board panel and the generated page in step, and compare each element's
// position, size, colour and text.
//
//   node frontend/verify-layout.mjs [slug ...]
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import { PAGES } from './build-pages.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DESIGN = path.join(ROOT, 'design');
const APP = path.join(ROOT, 'frontend', 'app');
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
const SIZES = JSON.parse(readFileSync(path.join(APP, 'sizes.json'), 'utf8'));
const originOf = s => (readFileSync(path.join(APP, s + '.html'), 'utf8')
  .match(/generated from design\/([\w.]+\.dc\.html)/) || [])[1];

const WALK = (rootSel) => {
  const root = document.querySelector(rootSel);
  const o = root.getBoundingClientRect();
  const out = [];
  const walk = (el) => {
    const cs = getComputedStyle(el);
    if (cs.display === 'none') return;
    const r = el.getBoundingClientRect();
    let own = '';
    for (const n of el.childNodes) if (n.nodeType === 3) own += n.nodeValue;
    out.push([
      el.tagName.toLowerCase(),
      +(r.x - o.x).toFixed(1), +(r.y - o.y).toFixed(1),
      +r.width.toFixed(1), +r.height.toFixed(1),
      cs.color, cs.backgroundColor, cs.fontSize, cs.fontWeight, cs.fontFamily.slice(0, 24),
      own.replace(/\s+/g, ' ').trim().slice(0, 40),
    ]);
    for (const c of el.children) walk(c);
  };
  walk(root);
  return out;
};

  const settle = async (pg) => {
    // A `ch` measure is taken against the font that is actually loaded, so
    // measuring before the webfont lands reports the fallback's metrics and
    // every width in the hero comes out wrong. Wait for the layout to stop
    // moving rather than for a promise that resolves too early.
    await pg.evaluate(async () => {
      await document.fonts.ready;
      // scrollHeight can be stable while a webfont is still swapping inside a
      // fixed-height hero, so ask for the faces by name as well
      const fams = [...new Set([...document.fonts].map(f => f.family))];
      await Promise.all(fams.flatMap(f => [400, 500, 600, 700, 800]
        .map(w => document.fonts.load(`${w} 20px "${f}"`).catch(() => {}))));
      await document.fonts.ready;
      const deadline = performance.now() + 4000;
      let last = -1, stable = 0;
      while (performance.now() < deadline && stable < 3) {
        const h = document.documentElement.scrollHeight + document.body.scrollWidth;
        stable = h === last ? stable + 1 : 0;
        last = h;
        await new Promise(r => setTimeout(r, 120));
      }
    });
    // a Google-hosted face can still be swapping after all of that; a flat
    // wait is inelegant and it is the only thing that makes the run repeatable
    await pg.waitForTimeout(1500);
  };

const only = process.argv.slice(2);
const slugs = Object.keys(PAGES).filter(s => !only.length || only.includes(s));
const browser = await chromium.launch();
const report = [];

for (const slug of slugs) {
  const size = SIZES[slug], board = originOf(slug);
  const a = await browser.newPage({ viewport: { width: 1500, height: Math.max(size.h, 600) } });
  await a.goto(pathToFileURL(path.join(DESIGN, board)).href, { waitUntil: 'load' });
  await a.addScriptTag({ content: shim });
  await a.evaluate(() => document.querySelectorAll('[data-fold]').forEach(n => n.remove()));
  await settle(a);
  const A = await a.evaluate(WALK, `[data-page="${slug}"]`);
  await a.close();

  const b = await browser.newPage({ viewport: { width: size.w, height: size.h } });
  await b.goto(pathToFileURL(path.join(APP, slug + '.html')).href, { waitUntil: 'load' });
  await settle(b);
  const B = await b.evaluate(WALK, 'body > div');
  await b.close();

  const issues = [];
  if (A.length !== B.length) issues.push({ kind: 'element count', a: A.length, b: B.length });
  const n = Math.min(A.length, B.length);
  const FIELDS = ['tag', 'x', 'y', 'w', 'h', 'color', 'background', 'font-size', 'weight', 'family', 'text'];
  for (let i = 0; i < n; i++) {
    for (let f = 0; f < FIELDS.length; f++) {
      const x = A[i][f], y = B[i][f];
      const same = (f >= 1 && f <= 4) ? Math.abs(x - y) <= 1 : x === y;
      if (!same) issues.push({ i, field: FIELDS[f], a: x, b: y, at: A[i][10].slice(0, 28) || A[i][0] });
    }
  }
  report.push({ slug, elements: A.length, issues });
}
await browser.close();

console.log('page'.padEnd(22) + 'elements   differences');
let perfect = 0;
for (const r of report) {
  const geo = r.issues.filter(i => 'xywh'.includes(i.field) || i.field === 'element count').length;
  const style = r.issues.length - geo;
  if (!r.issues.length) perfect++;
  const verdict = !r.issues.length ? 'every element identical'
    : `${geo} geometry, ${style} style/text`;
  console.log(`${r.slug.padEnd(22)}${String(r.elements).padStart(8)}   ${verdict}`);
  for (const i of r.issues.slice(0, 4)) {
    console.log(`      ${i.kind || i.field}: board ${JSON.stringify(i.a)} vs page ${JSON.stringify(i.b)}  @ "${i.at || ''}"`);
  }
  if (r.issues.length > 4) console.log(`      ... and ${r.issues.length - 4} more`);
}
console.log(`\n${perfect} of ${report.length} pages match the artboard element for element`);
process.exit(perfect === report.length ? 0 : 1);
