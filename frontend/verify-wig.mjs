// Web Interface Guidelines checks that can be MEASURED rather than eyeballed,
// run against the served pages so what is tested is what ships.
import { chromium } from 'playwright';
import { readdirSync } from 'node:fs';
import { needs } from './_needs.mjs';
// No arguments used to mean NO PAGES, and the summary still read like a pass
// ("0 findings across 0 pages"). It now means every page.
const PAGES = (process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort());
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
// Roles are derived from build-pages.mjs so this map cannot drift out of date.
// It did: a five-entry map meant eighteen pages loaded with no token, bounced
// to sign-in, and were audited as the sign-in page under their own names.
const ROLE = {
  'admin': 'admin',
  'ask': 'farmer',
  'ask-money': 'farmer',
  'confirm-investment': 'investor',
  'home': 'farmer',
  'invest': 'investor',
  'landing': null,
  'leaf-check': 'farmer',
  'market': 'buyer',
  'messages': 'farmer',
  'money': 'farmer',
  'orders': 'buyer',
  'payout': 'farmer',
  'plot': 'farmer',
  'plots': 'farmer',
  'portfolio': 'investor',
  'profile': 'farmer',
  'record': 'farmer',
  'report-harvest': 'farmer',
  'sent': 'farmer',
  'signin': null,
  'signup': null,
  'thread': 'farmer',
};

const b = await chromium.launch();
// Five pages are ABOUT something and are blank without an id.
const QUERY = await needs();
const tokens = {};
let total = 0;
for (const slug of PAGES) {
  const role = ROLE[slug];
  if (role && !tokens[role]) tokens[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tokens[role].token, JSON.stringify(tokens[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1400);

  const f = await p.evaluate(() => {
    const out = [];
    const say = (rule, what) => out.push(`${rule} — ${what}`);
    const txt = (el) => (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40);

    // typography: straight quotes, three dots, non-tabular number columns
    const body = document.body.innerText;
    if (/\.\.\./.test(body)) say('ellipsis', 'literal "..." in copy');
    const straight = body.match(/[A-Za-z]"[A-Za-z ]|[A-Za-z]'[A-Za-z]/g);
    if (straight) say('curly quotes', `${straight.length} straight quote(s), e.g. ${straight[0]}`);

    // forms
    for (const el of document.querySelectorAll('input, textarea')) {
      const id = el.name || el.type || 'input';
      if (!el.getAttribute('aria-label') && !el.labels?.length) say('form label', `<input ${id}> unlabelled`);
      if (!el.getAttribute('autocomplete')) say('autocomplete', `<input ${id}> has no autocomplete`);
      // The ellipsis belongs on an instruction that trails off, not on a
      // SPECIMEN of the value wanted: "98765 43210…" is not a phone number.
      // Same exemption wire.js applies when it builds the field, so the two
      // agree instead of the checker reporting what the code deliberately did.
      const specimen = /^[\d\s+•·.,₹-]+$/.test(el.placeholder || '');
      if (el.placeholder && !specimen && !/[…:]$/.test(el.placeholder)) {
        say('placeholder', `"${el.placeholder}" does not end in an ellipsis`);
      }
      if ((el.type === 'email' || /code|otp|phone|tel/.test(id)) && el.spellcheck) say('spellcheck', `<input ${id}> should disable spellcheck`);
    }

    // controls
    for (const el of document.querySelectorAll('[data-act], [role="button"], [role="switch"], [role="radio"], [role="checkbox"]')) {
      const name = el.getAttribute('aria-label') || txt(el);
      if (!name) say('accessible name', `<${el.tagName.toLowerCase()}> control has none`);
      if (el.tabIndex < 0) say('keyboard', `"${name}" is not focusable`);
      const cs = getComputedStyle(el);
      if (cs.touchAction !== 'manipulation') say('touch-action', `"${name}" lacks touch-action: manipulation`);
    }

    // animation anti-patterns
    for (const el of document.querySelectorAll('body *')) {
      if (/^(script|style|meta|link|title)$/i.test(el.tagName)) continue;
      const cs = getComputedStyle(el);
      // 'all' is the COMPUTED DEFAULT when nothing sets a transition, so it only
      // means "transition: all" was written if a duration was written with it.
      const dur = parseFloat(cs.transitionDuration) || 0;
      if (cs.transitionProperty === 'all' && dur > 0) say('transition: all', txt(el) || el.tagName.toLowerCase());
      if (cs.animationIterationCount === 'infinite') say('infinite loop', txt(el) || el.tagName.toLowerCase());
    }

    // images
    for (const img of document.querySelectorAll('img')) {
      if (!img.getAttribute('alt') && img.getAttribute('alt') !== '') say('img alt', img.src.slice(0, 40));
      if (!img.width || !img.height) say('img dimensions', img.src.slice(0, 40));
    }

    // decorative svg should be hidden from the reader
    let bareSvg = 0;
    for (const svg of document.querySelectorAll('svg')) {
      if (!svg.getAttribute('aria-hidden') && !svg.getAttribute('role') && !svg.querySelector('title')) bareSvg++;
    }
    if (bareSvg) say('aria-hidden', `${bareSvg} decorative <svg> not hidden from readers`);

    // number columns should be tabular
    for (const el of document.querySelectorAll('td, th')) {
      const t = (el.textContent || '').trim();
      // The cell may carry the figure in a child that IS tabular, which is how
      // every money column in this product is built -- so ask whether the
      // rendered digits are tabular, not whether the <td> happens to be.
      const tabular = /tabular-nums/.test(getComputedStyle(el).fontVariantNumeric)
        || [...el.querySelectorAll('*')].some(c => /tabular-nums/.test(getComputedStyle(c).fontVariantNumeric));
      if (/^[₹\d][\d,.\s₹%–-]*$/.test(t) && t.length > 1 && !tabular) {
        say('tabular-nums', `table cell "${t.slice(0, 18)}" is not tabular`);
      }
    }

    // headings
    const hs = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role="heading"]')]
      .map(h => +(h.tagName[0] === 'H' ? h.tagName[1] : h.getAttribute('aria-level') || 2));
    if (!hs.length) say('headings', 'page has no heading of any kind');
    else if (hs[0] !== 1) say('headings', `first heading is level ${hs[0]}`);

    // horizontal overflow
    if (document.documentElement.scrollWidth > window.innerWidth + 1) {
      say('overflow', `page scrolls sideways (${document.documentElement.scrollWidth}px)`);
    }
    return out;
  });

  const uniq = [...new Set(f)];
  total += uniq.length;
  console.log(`\n## ${slug}.html${QUERY[slug] || ''}`);
  if (!uniq.length) console.log('  ✓ pass');
  else uniq.slice(0, 14).forEach(x => console.log(`  ${x}`));
  if (uniq.length > 14) console.log(`  … ${uniq.length - 14} more of the same kinds`);
  await p.close();
}
await b.close();
console.log(`\n${total} findings across ${PAGES.length} pages`);
