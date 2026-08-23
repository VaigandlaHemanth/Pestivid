// Proves the generated pages did not change the design.
//
// For every page: screenshot the marked element inside its artboard, screenshot
// the same element in the generated page at the same width, and diff the pixels.
// A page that drifts fails here rather than in somebody's review.
//
//   node frontend/verify-pages.mjs [slug ...]
import { chromium } from 'playwright';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import { PAGES } from './build-pages.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DESIGN = path.join(ROOT, 'design');
const APP = path.join(ROOT, 'frontend', 'app');
const SHOTS = path.join(ROOT, 'frontend', '.verify');
mkdirSync(SHOTS, { recursive: true });
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
// the size each board actually renders its panel at, recorded by the build
const SIZES = JSON.parse(readFileSync(path.join(APP, 'sizes.json'), 'utf8'));

// which artboard each slug came from, read out of the generated page's own header
const originOf = slug => {
  const s = readFileSync(path.join(APP, slug + '.html'), 'utf8');
  return (s.match(/generated from design\/([\w.]+\.dc\.html)/) || [])[1];
};

const only = process.argv.slice(2);
const slugs = Object.keys(PAGES).filter(s => !only.length || only.includes(s));

const browser = await chromium.launch();
const rows = [];
for (const slug of slugs) {
  const meta = PAGES[slug];
  const size = SIZES[slug];
  const board = originOf(slug);
  if (!board) { rows.push({ slug, err: 'no origin comment' }); continue; }

  // ---- the artboard, with the fold marker hidden so it is a fair comparison
  const a = await browser.newPage({ viewport: { width: 1500, height: Math.max(size.h, 600) }, deviceScaleFactor: 1 });
  await a.goto(pathToFileURL(path.join(DESIGN, board)).href, { waitUntil: 'load' });
  await a.addScriptTag({ content: shim });
  await a.evaluate(() => { document.querySelectorAll('[data-fold]').forEach(n => n.remove()); });
  await a.evaluate(() => document.fonts.ready);
  await a.waitForTimeout(150);
  const elA = await a.$(`[data-page="${slug}"]`);
  if (!elA) { rows.push({ slug, err: `no [data-page] in ${board}` }); await a.close(); continue; }
  await elA.screenshot({ path: path.join(SHOTS, `${slug}.board.png`) });
  await a.close();

  // ---- the generated page, at the width the board was drawn at
  const b = await browser.newPage({ viewport: { width: size.w, height: size.h }, deviceScaleFactor: 1 });
  await b.goto(pathToFileURL(path.join(APP, slug + '.html')).href, { waitUntil: 'load' });
  await b.evaluate(() => document.fonts.ready);
  await b.waitForTimeout(150);
  const elB = await b.$('body > div');
  await elB.screenshot({ path: path.join(SHOTS, `${slug}.page.png`) });
  await b.close();

  rows.push({ slug, board, w: size.w, h: size.h });
}
await browser.close();
writeFileSync(path.join(SHOTS, 'pairs.json'), JSON.stringify(rows, null, 1));
console.log(`captured ${rows.filter(r => !r.err).length} pairs into frontend/.verify/`);
for (const r of rows) if (r.err) console.log(`  ${r.slug}: ${r.err}`);
