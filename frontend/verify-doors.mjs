// A control that leads to a door this person is refused at.
//
// The buyer's orders page carried "Watch it - check the date" on every receipt,
// and it went to plot.html, which requireUser gates to farmers. So the one
// control that answers "was the lot I paid for real" -- the only reason a
// receipt here is worth more than a receipt anywhere else -- landed a buyer on
// "Not your screen. This page is for a farmer." Four times, once per lot, on
// their own page. Nothing caught it: the destination existed, the link was
// wired, the label was right, and the check for dead destinations only asks
// whether a page is THERE.
//
// So this asks the other question. It reads every page module's requireUser call
// for the roles that page admits, renders each page as the role that owns it, and
// follows every [data-go] on screen: is the role standing here allowed through
// that door? A destination with no gate admits everybody and passes.
import { chromium } from 'playwright';
import { readdirSync, readFileSync } from 'node:fs';
import { needs } from './_needs.mjs';

const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

// Who each page admits, read from the page modules rather than restated here --
// a second copy of this list is a second thing to go stale.
const GATE = {};
for (const f of readdirSync('frontend/app/pages').filter((f) => f.endsWith('.js'))) {
  const src = readFileSync(`frontend/app/pages/${f}`, 'utf8');
  for (const m of src.matchAll(/requireUser\(\s*'([\w-]+)'\s*(?:,\s*\[([^\]]*)\])?/g)) {
    GATE[m[1]] = m[2] ? [...m[2].matchAll(/'([a-z]+)'/g)].map((x) => x[1]) : null;
  }
}
// The leaf pages share one module and pass their slug in.
for (const s of ['leaf-check', 'leaf-result', 'leaf-refusal']) if (!(s in GATE)) GATE[s] = ['farmer'];
for (const s of ['notifications']) if (!(s in GATE)) GATE[s] = null;

const ROLE = {
  landing: null, signin: null, signup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  market: 'buyer', orders: 'buyer', admin: 'admin',
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter((f) => f.endsWith('.html'))
    .map((f) => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0, doors = 0;

for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (!role) { console.log(`  ${slug.padEnd(20)} public`); continue; }
  if (!tok[role]) tok[role] = await login(role);
  const ctx = await b.newContext({ viewport: { width: 1366, height: 900 } });
  const p = await ctx.newPage();
  await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1600);
  const found = await p.evaluate(() => [...document.querySelectorAll('[data-go]')]
    .filter((el) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      return r.width > 1 && r.height > 1 && cs.display !== 'none' && cs.visibility !== 'hidden';
    })
    .map((el) => ({ to: (el.getAttribute('data-go') || '').split(/[?#]/)[0],
                    label: (el.getAttribute('aria-label') || el.textContent || '')
                      .replace(/\s+/g, ' ').trim().slice(0, 46) })));
  await ctx.close();

  const shut = [];
  for (const d of found) {
    if (!d.to || !(d.to in GATE)) continue;      // unknown or ungated
    doors++;
    const allowed = GATE[d.to];
    if (allowed && !allowed.includes(role)) shut.push({ ...d, allowed });
  }
  if (!shut.length) { console.log(`  ${slug.padEnd(20)} ok   ${found.length} door(s)`); continue; }
  bad += shut.length;
  console.log(`  ${slug.padEnd(20)} ${shut.length} door(s) a ${role} is refused at:`);
  for (const s of shut) {
    console.log(`      -> ${s.to}.html  (for ${s.allowed.join('/')})   "${s.label}"`);
  }
}

await b.close();
console.log(`\n  ${bad} control(s) leading somewhere this role is refused,`
  + ` of ${doors} gated door(s) across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
