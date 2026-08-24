// Do the modules' style-string selectors still match anything?
//
// The behaviour modules find drawn elements by their inline CSS -- things like
// querySelector('div[style*="background: #e7e1db"]') -- because the boards are
// generated and most elements carry no class. That works until somebody edits
// the drawing, and then it fails SILENTLY: the lookup returns null, the feature
// quietly does not exist, and the page still renders.
//
// It has now happened three times:
//   * record.js looked for #37322d / #0e0d0b; the viewfinder is #2a2622, so
//     there was no <video> on the filming screen at all.
//   * setup.js looked for `flex: 1` + `height: 62px`; the code boxes were given
//     a fixed width, so no code input was ever created and the first run had no
//     way to set a password. Nothing logged. The page looked fine.
//   * market.js climbed to closest('div[style*="grid-template-columns"]') and
//     reached the page grid, then deleted the page.
//
// So: pull every style-string selector out of the modules, load the page WITH
// JAVASCRIPT OFF, and check each one still matches something in the drawing.
//
// Javascript off matters. Run against the live DOM this reported four dead
// selectors that all worked: setting bar.style.width re-serialises the whole
// inline style attribute and turns `#01579b` into `rgb(1, 87, 155)`, and
// money.js rewrites the grid it just found. The question a module actually asks
// is "is this in the drawing I was handed", so that is what gets asked here.
import { chromium } from 'playwright';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { needs } from './_needs.mjs';

const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

const ROLE = {
  landing: null, signin: null, signup: null, setup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  thread: 'farmer', market: 'buyer', orders: 'buyer', admin: 'admin',
};
const roleFor = (s) => (s in ROLE ? ROLE[s] : 'farmer');

const PAGES_DIR = 'frontend/app/pages';
const slugs = readdirSync('frontend/app').filter(f => f.endsWith('.html'))
  .map(f => f.replace('.html', '')).sort();

// A shared module belongs to whichever pages import it.
const importsOf = (slug) => {
  const f = path.join(PAGES_DIR, slug + '.js');
  let src = '';
  try { src = readFileSync(f, 'utf8'); } catch { return []; }
  const files = [[slug + '.js', src]];
  for (const m of src.matchAll(/from\s+'\.\/(_[\w-]+\.js)'/g)) {
    try { files.push([m[1], readFileSync(path.join(PAGES_DIR, m[1]), 'utf8')]); } catch {}
  }
  return files;
};

// Every selector string that leans on an inline style, with the line it is on.
const SEL = /(?:querySelector|querySelectorAll|closest|matches)\(\s*'([^']*\[style\*=[^']*)'/g;
const found = new Map();          // slug -> [{file, line, sel}]
for (const slug of slugs) {
  const hits = [];
  for (const [file, src] of importsOf(slug)) {
    for (const m of src.matchAll(SEL)) {
      hits.push({ file, sel: m[1], line: src.slice(0, m.index).split('\n').length });
    }
  }
  if (hits.length) found.set(slug, hits);
}

const b = await chromium.launch();
const tok = {};
let checked = 0, dead = 0;
for (const [slug, hits] of found) {
  const role = roleFor(slug);
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 1000 }, javaScriptEnabled: false });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(300);

  const misses = [];
  for (const h of hits) {
    checked++;
    // closest() and matches() are relative to some element the module already
    // holds, so the honest question for those is whether the SELECTOR matches
    // anything on the page at all -- if it matches nothing, no climb can find it.
    const n = await p.evaluate((sel) => {
      try { return document.querySelectorAll(sel).length; } catch { return -1; }
    }, h.sel);
    if (n === 0) { misses.push(h); dead++; }
    else if (n === -1) { misses.push({ ...h, bad: true }); dead++; }
  }
  if (misses.length) {
    console.log(`\n## ${slug}`);
    for (const m of misses) {
      console.log(`  ${m.file}:${m.line}  ${m.bad ? 'INVALID SELECTOR' : 'matches nothing'}`);
      console.log(`      ${m.sel}`);
    }
  }
  await p.close();
}
await b.close();
console.log(`\n${dead} of ${checked} style-string selectors match nothing, across ${found.size} pages`);
process.exit(dead ? 1 : 0);
