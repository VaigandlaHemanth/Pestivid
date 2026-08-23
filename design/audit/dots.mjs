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
const rows = await p.evaluate(() => {
  const out = {};
  for (const d of document.querySelectorAll('.dot')) {
    const n = getComputedStyle(d).animationName;
    if (!n.startsWith('run-')) continue;
    const a = d.getAnimations()[0]; a.pause();
    const xs = [];
    for (const ms of [0,100,200,300,400,500,600,700,800,900]) {
      a.currentTime = ms;
      xs.push(+new DOMMatrixReadOnly(getComputedStyle(d).transform).m41.toFixed(1));
    }
    out[n.replace('run-','')] = xs;
  }
  return out;
});
console.log('on-screen x in px (travel 196), sampled from the live animation');
console.log('        ' + [0,100,200,300,400,500,600,700,800,900].map(t=>String(t).padStart(6)).join(''));
for (const [k,v] of Object.entries(rows)) console.log('  ' + k.padEnd(7) + v.map(x=>String(x).padStart(6)).join(''));
await b.close();
