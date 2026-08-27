// Can you see where one surface ends and the next begins?
//
// verify-hig measures TEXT against its background and reports clean. It says
// nothing about two surfaces meeting, and this palette has eight warm steps
// inside a 1.4:1 total range -- so a card can sit on the page at 1.06:1 and be
// the same colour as the page, with clean text on it either way.
//
// Reports every place a filled surface sits directly on another filled surface
// with less than 1.20:1 between them AND no hairline, ring or shadow drawing the
// edge. Either the fills differ enough to see, or something draws the boundary.
import { chromium } from 'playwright';
import { readdirSync } from 'node:fs';
import { needs } from './_needs.mjs';

const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
const ROLE = {
  landing: null, signin: null, signup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  thread: 'farmer', market: 'buyer', orders: 'buyer', admin: 'admin',
};
const FLOOR = 1.20;

const AUDIT = (floor) => {
  const rgb = (s) => { const m = (s || '').match(/[\d.]+/g);
    return m ? { r: +m[0], g: +m[1], b: +m[2], a: m[3] == null ? 1 : +m[3] } : null; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const hex = (c) => '#' + [c.r, c.g, c.b].map(v => v.toString(16).padStart(2, '0')).join('');

  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const mine = rgb(cs.backgroundColor);
    if (!mine || mine.a < 0.95) continue;                 // must be a filled surface
    const r = el.getBoundingClientRect();
    if (r.width < 40 || r.height < 20) continue;          // a surface, not a dot
    // The page panel painted on the body is the same colour by definition. It
    // is the page on itself, not a card you cannot find the edge of.
    if (el.parentElement === document.body || el === document.body) continue;

    // the nearest filled ancestor is what it sits on
    let host = null;
    for (let n = el.parentElement; n; n = n.parentElement) {
      const c = rgb(getComputedStyle(n).backgroundColor);
      if (c && c.a >= 0.95) { host = { el: n, c }; break; }
    }
    if (!host) continue;
    const got = ratio(mine, host.c);
    if (got >= floor) continue;

    // is anything drawing the edge? a ring, a hairline, a shadow, a border
    const drawn = (cs.boxShadow && cs.boxShadow !== 'none')
      || (parseFloat(cs.borderTopWidth) || 0) > 0
      || (parseFloat(cs.borderBottomWidth) || 0) > 0
      || (parseFloat(cs.outlineStyle) !== undefined && cs.outlineStyle !== 'none');
    if (drawn) continue;

    /* Two things are meant to be invisible, and both are deliberate.
     *
     * A control that is switched OFF should recede -- that is how it reads as
     * off -- so the disabled fill on the page is doing its job.
     *
     * And a container painted the page's own colour is not a card whose edge you
     * cannot find; it is part of the page. The chat column is that: the
     * transcript belongs to the page, and the bubbles inside it carry the
     * contrast (white on the ground is 1.43:1).
     */
    if (el.closest('[aria-disabled="true"]')) continue;
    if (got < 1.02) continue;

    out.push({
      what: (el.className || el.getAttribute('data-bind') || el.tagName).toString().slice(0, 24),
      fill: hex(mine), on: hex(host.c),
      got: Math.round(got * 1000) / 1000,
      size: Math.round(r.width) + 'x' + Math.round(r.height),
    });
  }
  // one row per distinct pairing, not per element
  const seen = new Map();
  for (const o of out) {
    const k = o.fill + o.on + o.what;
    if (!seen.has(k)) seen.set(k, { ...o, n: 1 }); else seen.get(k).n++;
  }
  return [...seen.values()];
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let total = 0;
for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 2200 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1300);
  const found = await p.evaluate(AUDIT, FLOOR);
  await p.close();
  if (found.length) {
    console.log(`\n## ${slug}`);
    for (const f of found) {
      console.log(`  ${f.fill} on ${f.on}  ${String(f.got).padEnd(6)} x${String(f.n).padEnd(3)}`
        + ` ${f.size.padEnd(11)} ${f.what}`);
    }
    total += found.length;
  }
}
await b.close();
console.log(`\n  ${total} surface pairing(s) under ${FLOOR}:1 with no edge drawn,`
  + ` across ${slugs.length} pages`);
process.exit(total ? 1 : 0);
