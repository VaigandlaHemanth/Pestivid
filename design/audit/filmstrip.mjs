import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
const DESIGN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1400, height: 1000 } });
await p.goto(pathToFileURL(path.join(DESIGN, 'Motion.dc.html')).href, { waitUntil: 'load' });
await p.addScriptTag({ content: shim });
await p.evaluate(() => document.fonts.ready);
await p.waitForTimeout(200);
// freeze every animation so each still is a known instant
const N = 9, SPAN = 900;
for (let i = 0; i < N; i++) {
  const t = (i / (N - 1)) * SPAN;
  await p.evaluate(ms => { for (const a of document.getAnimations()) { a.pause(); a.currentTime = ms; } }, t);
  await p.waitForTimeout(60);
  const el = await p.$('.card');           // the five-spring card
  await el.screenshot({ path: path.join(DESIGN, 'audit', `f${i}.png`) });
}
// and the stamp, which runs on a longer cycle
for (let i = 0; i < 6; i++) {
  const t = 200 + (i / 5) * 1000;
  await p.evaluate(ms => { for (const a of document.getAnimations()) { a.pause(); a.currentTime = ms; } }, t);
  await p.waitForTimeout(50);
  const el = await p.$$('.card');
  await el[6].screenshot({ path: path.join(DESIGN, 'audit', `s${i}.png`) });
}
console.log('frames captured');
await b.close();
