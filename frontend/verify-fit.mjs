// Can you reach the controls on the smallest laptop this is for?
//
// The chatbot's composer was cut off the bottom of the screen and the send
// button with it: the page reserved 860px of fixed height on a machine with 768,
// so the one control the page exists for was unreachable without scrolling a
// column that was itself scrolling. Nothing caught it, because every check here
// either measured the page against its own artboard (verify-layout, which
// renders at 1320x880 by definition) or measured colour.
//
// A tall page is NOT a finding. A list of orders should scroll; a landing page
// should scroll. What must never happen is a CONTROL sitting past the fold on a
// page whose content is finite -- and in particular a composer or a submit that
// the page has pinned to its own bottom edge, which scrolling cannot rescue
// when the thing it is pinned inside also scrolls.
//
// 1366x768 is the floor on purpose: it is the commonest laptop resolution still
// in service, and Windows display scaling at 125% turns a 1080p screen into
// about the same usable height.
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

/* Pages whose content is a LIST and is meant to run past the fold. Scrolling is
 * how you read a list; the finding would be the opposite, a list truncated to
 * fit. Named one by one rather than guessed, so a page that starts scrolling by
 * accident still gets reported.
 */
const SCROLLS = new Set(['landing', 'market', 'orders', 'admin', 'invest', 'money',
                         'plots', 'home', 'profile', 'signup', 'signin', 'ask-money',
                         'portfolio', 'plot', 'leaf-check', 'report-harvest']);

const AUDIT = () => {
  const out = { doc: document.documentElement.scrollHeight, view: innerHeight,
                pinned: [], below: [] };
  // A column that scrolls internally and pins something to its bottom edge. If
  // that edge is off-screen the pinned thing is unreachable at any scroll
  // position of the page, because the page is not what is scrolling.
  for (const col of document.querySelectorAll('[data-chatcol]')) {
    const r = col.getBoundingClientRect();
    if (r.bottom > innerHeight + 1) {
      out.pinned.push(`${Math.round(r.bottom - innerHeight)}px of the conversation column`);
    }
  }
  const controls = [...document.querySelectorAll('[data-act], button, [role="button"], a[href]')]
    .filter(e => e.offsetParent && !e.closest('.appbar') && !e.closest('[data-loader]'));
  for (const el of controls) {
    const r = el.getBoundingClientRect();
    if (r.bottom > innerHeight + 1) {
      out.below.push((el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 30)
        || el.getAttribute('aria-label') || el.tagName.toLowerCase());
    }
  }
  return out;
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0;
for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1366, height: 768 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1300);
  const r = await p.evaluate(AUDIT);
  await p.close();

  const issues = [...r.pinned];
  // On a page that is meant to scroll, a control below the fold is just further
  // down the list. On one that is not, it is out of reach.
  if (!SCROLLS.has(slug) && r.below.length) {
    issues.push(`past the fold: ${[...new Set(r.below)].slice(0, 3).join(', ')}`);
  }
  if (issues.length) {
    console.log(`  ${slug.padEnd(20)} doc ${String(r.doc).padEnd(5)} ${issues.join(' · ')}`);
    bad++;
  }
}
await b.close();
console.log(`\n  ${bad} of ${slugs.length} pages put a control out of reach at 1366x768`);
process.exit(bad ? 1 : 0);
