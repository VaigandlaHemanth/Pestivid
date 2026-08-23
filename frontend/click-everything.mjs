// Clicks every control on every page and reports the ones that do nothing.
//
// A page that renders is not a page that works. This enumerates everything that
// looks interactive, clicks each one on a freshly loaded page, and records what
// happened: a navigation, a DOM change, a script error, or nothing at all.
//
// "Nothing at all" is the finding. A control that looks pressable and isn't is
// worse than no control, because the person concludes the product is broken
// rather than that the button was decoration.
//
//   node frontend/click-everything.mjs            every page
//   node frontend/click-everything.mjs home plots  named pages
import { chromium } from 'playwright';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const API = 'http://127.0.0.1:3001/api';
const APP = 'http://127.0.0.1:3001/app';
const APPDIR = path.join(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), 'frontend', 'app');

const ROLE_OF = {
  landing: null, signin: null, signup: null, 'signin-farmer': null,
  'setup-language': null, 'setup-identity': null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor', 'thread-investor': 'investor',
  market: 'buyer', orders: 'buyer',
  admin: 'admin',
};
const roleFor = (slug) => (slug in ROLE_OF ? ROLE_OF[slug] : 'farmer');

// Pages where clicking would really write something, or really cost money.
// Enumerated and reported, never clicked.
const NO_CLICK = {
  record: 'takes the camera',
  sent: 'uploads to storage, which costs a file of the free allowance',
  signup: 'creates an account on every run',
  'setup-identity': 'creates an account on every run',
  'leaf-result': 'downloads 173 MB of model',
  'leaf-refusal': 'downloads 173 MB of model',
};

const tokens = {};
async function tokenFor(role) {
  if (!role) return null;
  if (tokens[role]) return tokens[role];
  const r = await fetch(API + '/auth/login', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: `demo.${role}@pestivid.sim`, password: 'password123' }),
  });
  if (!r.ok) throw new Error(`login ${role} -> ${r.status}`);
  tokens[role] = await r.json();
  return tokens[role];
}

const ENUMERATE = () => {
  const seen = new Set();
  const out = [];
  const sel = '[data-act], [data-go], [role="button"], [role="checkbox"], [role="radio"], [role="switch"], [role="link"], input, textarea';
  for (const el of document.querySelectorAll(sel)) {
    if (seen.has(el)) continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;
    out.push({
      tag: el.tagName.toLowerCase(),
      kind: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ? 'field' : 'control',
      name: el.getAttribute('aria-label') || el.getAttribute('name')
            || (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 44) || '(no name)',
      go: el.dataset.go || null,
      role: el.getAttribute('role') || null,
      w: Math.round(r.width), h: Math.round(r.height),
    });
  }
  return out;
};

const only = process.argv.slice(2);
const slugs = readdirSync(APPDIR).filter(f => f.endsWith('.html')).map(f => f.replace('.html', ''))
  .filter(s => !only.length || only.includes(s)).sort();

const browser = await chromium.launch();
const findings = [];
let clicked = 0, controls = 0;

for (const slug of slugs) {
  const role = roleFor(slug);
  const session = await tokenFor(role);
  const open = async () => {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    if (session) {
      await page.addInitScript(([t, u]) => {
        localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
      }, [session.token, JSON.stringify(session.user)]);
    }
    return page;
  };

  const page = await open();
  await page.goto(`${APP}/${slug}.html`, { waitUntil: 'load' });
  await page.waitForTimeout(1400);
  const list = await page.evaluate(ENUMERATE);
  await page.close();
  controls += list.length;

  if (NO_CLICK[slug]) {
    console.log(`  ${slug.padEnd(20)} ${String(list.length).padStart(2)} controls  not clicked: ${NO_CLICK[slug]}`);
    continue;
  }

  const dead = [], moved = [], broke = [];
  for (let i = 0; i < list.length; i++) {
    const c = list[i];
    if (c.kind === 'field') continue;                 // fields are exercised by e2e-bind
    const p = await open();
    const errs = [];
    p.on('pageerror', e => errs.push(String(e).slice(0, 70)));
    await p.goto(`${APP}/${slug}.html`, { waitUntil: 'load' });
    await p.waitForTimeout(1200);
    // A control that is already the current one is not dead when clicking it
    // changes nothing -- selecting the row you are already reading is meant to
    // be a no-op. Recorded before the click so the two cases can be told apart.
    const before = await p.evaluate((n) => {
      const sel = '[data-act], [data-go], [role="button"], [role="checkbox"], [role="radio"], [role="switch"], [role="link"], input, textarea';
      const els = [...document.querySelectorAll(sel)].filter(e => {
        const r = e.getBoundingClientRect(); const cs = getComputedStyle(e);
        return r.width >= 2 && r.height >= 2 && cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
      });
      const el = els[n];
      return {
        url: location.pathname,
        html: document.body.innerHTML.length,
        already: el?.getAttribute('aria-current') === 'true' || el?.getAttribute('aria-pressed') === 'true',
      };
    }, i);
    try {
      await p.evaluate((n) => {
        const sel = '[data-act], [data-go], [role="button"], [role="checkbox"], [role="radio"], [role="switch"], [role="link"], input, textarea';
        const els = [...document.querySelectorAll(sel)].filter(e => {
          const r = e.getBoundingClientRect();
          const cs = getComputedStyle(e);
          return r.width >= 2 && r.height >= 2 && cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
        });
        els[n]?.click();
      }, i);
    } catch { /* recorded below */ }
    await p.waitForTimeout(900);
    const after = await p.evaluate(() => ({ url: location.pathname, html: document.body.innerHTML.length }));
    clicked++;
    if (errs.length) broke.push(`${c.name}: ${errs[0]}`);
    else if (after.url !== before.url) moved.push(`${c.name} -> ${after.url.split('/').pop()}`);
    else if (after.html === before.html && !before.already) dead.push(c.name);
    await p.close();
  }

  const line = [`${dead.length} dead`, `${moved.length} navigate`, broke.length ? `${broke.length} ERROR` : null]
    .filter(Boolean).join(', ');
  console.log(`  ${slug.padEnd(20)} ${String(list.length).padStart(2)} controls  ${line}`);
  for (const d of dead) findings.push([slug, 'does nothing', d]);
  for (const b of broke) findings.push([slug, 'script error', b]);
}
await browser.close();

console.log(`\n  ${controls} controls found, ${clicked} clicked`);
if (!findings.length) {
  console.log('  every control either navigates or changes the page');
} else {
  console.log(`  ${findings.length} to fix\n`);
  for (const [slug, kind, what] of findings) {
    console.log(`  ${slug.padEnd(20)} ${kind.padEnd(14)} ${what}`);
  }
}
process.exit(findings.filter(f => f[1] === 'script error').length ? 1 : 0);
