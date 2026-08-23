// Loads every generated page in a browser and fails on anything a real visitor
// would hit: a script error, a missing file, or a page whose module never ran.
import { chromium } from 'playwright';
import { readdirSync, readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const APP = path.join(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), 'frontend', 'app');
// ES modules are blocked over file://, and these pages are served over HTTP in
// production, so serve them here too rather than testing a condition that will
// never occur.
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\//, '') || 'landing.html';
  const file = path.join(APP, rel);
  if (!file.startsWith(APP)) { res.writeHead(403).end(); return; }
  let body;
  try { body = readFileSync(file); } catch { res.writeHead(404).end('not found'); return; }
  res.writeHead(200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' });
  res.end(body);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch();
let bad = 0, n = 0;

for (const f of readdirSync(APP).filter(x => x.endsWith('.html'))) {
  const p = await browser.newPage();
  const errs = [];
  // This harness serves files only. A page reaching for /api is doing the right
  // thing and getting the right answer, so its 404 is not a page defect.
  const ignorable = t => /\/api\//.test(t) || /fonts\.(googleapis|gstatic)/.test(t);
  p.on('console', m => {
    // the message text of a failed fetch does not carry the URL; its location does
    const where = m.location()?.url || '';
    if (m.type() === 'error' && !ignorable(m.text()) && !ignorable(where)) errs.push(m.text().slice(0, 100));
  });
  p.on('pageerror', e => errs.push('JS: ' + String(e).slice(0, 100)));
  p.on('requestfailed', r => { if (!ignorable(r.url())) errs.push('missing ' + r.url().split('/').pop()); });
  await p.goto(base + f, { waitUntil: 'load' });
  await p.waitForTimeout(350);
  const ready = await p.evaluate(() => document.documentElement.dataset.ready || null);
  const title = await p.title();
  n++;
  if (errs.length || !ready || !title) {
    bad++;
    console.log(`  ${f.padEnd(26)} ${ready ? '' : 'module never ran '}${errs.slice(0, 2).join(' | ')}`);
  }
  await p.close();
}
await browser.close();
server.close();
console.log(bad ? `  ${bad} of ${n} pages have a problem` : `  all ${n} pages load clean, wire up, and carry a title`);
process.exit(bad ? 1 : 0);
