import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
const DESIGN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
const b = await chromium.launch();
for (const f of process.argv.slice(2)) {
  const p = await b.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 1 });
  await p.goto(pathToFileURL(path.join(DESIGN, f + '.dc.html')).href, { waitUntil: 'load' });
  await p.addScriptTag({ content: shim });
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(200);
  await p.screenshot({ path: path.join(DESIGN, 'audit', f + '.png'), fullPage: true });
  await p.close();
  console.log('shot', f);
}
await b.close();
