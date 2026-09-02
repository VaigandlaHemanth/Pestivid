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
  market: 'buyer', orders: 'buyer', admin: 'admin',
};

/* Pages whose content is a LIST and is meant to run past the fold. Scrolling is
 * how you read a list; the finding would be the opposite, a list truncated to
 * fit. Named one by one rather than guessed, so a page that starts scrolling by
 * accident still gets reported.
 */
const SCROLLS = new Set(['landing', 'market', 'orders', 'admin', 'invest', 'money',
                         'home', 'profile', 'signup', 'signin', 'ask-money',
                         'portfolio', 'plot', 'leaf-check', 'report-harvest',
                         // Everything that has happened to you, oldest last. It
                         // is a list like orders and admin and was only missing
                         // here because the demo database never held enough
                         // notices to reach the fold.
                         'notifications']);

/* Pages that fix their own height so the page itself never scrolls -- the chat
 * and the chatbot, whose transcript scrolls INSIDE a column of a computed
 * height. For these, fitting is measurable in both directions: content past the
 * bottom edge is unreachable, and ground left over below the content is a page
 * that did not fill the window it was told to fill.
 *
 * The chat column was written as `calc(100dvh - 230px)` behind a
 * `max-height: 830px` media query, which meant: at 975px nothing fired at all
 * and 144px of bare ground sat under the conversation; and where it did fire,
 * 230 was 25px more chrome than the page actually has, so 55px was still dead.
 * A rule that only ever gets eyeballed at one window size gets this wrong every
 * time, so it is measured at four.
 */
const FIXED = new Set(['messages', 'ask']);
const HEIGHTS = [975, 900, 860, 768];
// Under this many leftover pixels is rounding and sub-pixel layout, not a gap
// anybody can see.
const DEAD_FLOOR = 8;

const AUDIT = () => {
  const out = { doc: document.documentElement.scrollHeight, view: innerHeight,
                pinned: [], below: [], dead: null };
  /* How much bare ground is left under the page, and neither of the two obvious
   * ways to ask gives the right answer:
   *
   *  - the gap under the COLUMN counts the 30px of bottom padding its container
   *    legitimately reserves, and reported those designed pixels as a defect on
   *    both pages at every height.
   *  - the gap under the BODY is always zero, because the page root carries
   *    min-height: 100vh. The body reaches the fold whatever the column does;
   *    the hole opens up inside it. This is exactly why the 144px hole was
   *    invisible to a doc-height check.
   *
   * So: the column's own gap, less the padding and margin its ancestors are
   * entitled to. Zero when the page fits, 25 when the constant is 25px out. */
  const col = document.querySelector('[data-chatcol]');
  if (col) {
    let reserved = 0;
    for (let n = col; n && n !== document.body; n = n.parentElement) {
      const cs = getComputedStyle(n.parentElement || n);
      reserved += (parseFloat(cs.paddingBottom) || 0) + (parseFloat(cs.marginBottom) || 0);
    }
    out.dead = Math.round(innerHeight - col.getBoundingClientRect().bottom - reserved);
  }
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
  // A page that fixes its own height is measured at every height it has to
  // survive; the rest at the floor, which is where a control falls off.
  const heights = FIXED.has(slug) ? HEIGHTS : [768];
  const issues = [];
  for (const h of heights) {
    const p = await b.newPage({ viewport: { width: 1366, height: h } });
    if (role) await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
    await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
    await p.waitForTimeout(1300);
    const r = await p.evaluate(AUDIT);
    await p.close();

    const at = heights.length > 1 ? ` at ${h}` : '';
    for (const pin of r.pinned) issues.push(pin + at);
    // On a page that is meant to scroll, a control below the fold is just
    // further down the list. On one that is not, it is out of reach.
    if (!SCROLLS.has(slug) && r.below.length) {
      issues.push(`past the fold${at}: ${[...new Set(r.below)].slice(0, 3).join(', ')}`);
    }
    if (FIXED.has(slug)) {
      if (r.dead !== null && r.dead >= DEAD_FLOOR) {
        issues.push(`${r.dead}px of dead ground below the column${at}`);
      }
      // And the page itself must not have grown a scrollbar it has no use for.
      if (r.doc > r.view + 1) issues.push(`the page scrolls ${r.doc - r.view}px${at}`);
    }
  }
  if (issues.length) {
    console.log(`  ${slug.padEnd(20)} ${issues.join(' · ')}`);
    bad++;
  }
}
await b.close();
console.log(`\n  ${bad} of ${slugs.length} pages fail to fit`
  + ` — controls out of reach at 1366x768, or dead ground at ${HEIGHTS.join('/')}`);
process.exit(bad ? 1 : 0);
