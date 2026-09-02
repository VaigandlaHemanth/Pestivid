// Does the bar say where you are -- and does it say it about the right page?
//
// The bar is drawn three different ways across the boards: .appnav/.appnavOn on
// fourteen of them, .nav/.navOn on two, and a bare inline font-weight on two
// more. All three encode the same fact, so nothing could check it and nothing
// did: four boards drew the underline on "My plots" and two of those four are
// not that page. Standing on home the farmer read "My plots" underlined as the
// screen they were on, directly above a link in the content that also said "My
// plots" and was the one that actually went there.
//
// This asks the rendered page, not the markup: among the bar's nav words, which
// one is presented as current -- heavier, darker, or underlined -- and is it the
// word for this page? Three answers are right:
//   the word for this page          -- plots underlines "My plots"
//   the word for its section        -- a plot detail belongs to My plots
//   nothing at all                  -- home is reached by the wordmark, so no
//                                      word is home and none should claim to be
// Anything else is the bar lying about where you are.
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
// Which nav word owns each page. A page absent here owns no word, and its bar
// must not present one as current.
const OWNS = {
  // home is the list of plots now, so home owns the word.
  home: 'My plots', plot: 'My plots', record: 'My plots', sent: 'My plots',
  money: 'Money', 'ask-money': 'Money', 'report-harvest': 'Money',
  messages: 'Chat',
  // One kind of account now: the two browse screens are the two halves of
  // Market, and the two ledgers are sections of Money.
  invest: 'Market', 'confirm-investment': 'Market', market: 'Market',
  portfolio: 'Money', orders: 'Money', admin: 'Flagged',
};
const WORDS = new Set(Object.values(OWNS));

const READ = (words) => {
  const bar = [...document.querySelectorAll('div, span, a')].filter((el) => {
    const r = el.getBoundingClientRect();
    if (!(r.top >= 0 && r.top < 110 && r.height > 0)) return false;
    const t = el.textContent.trim();
    if (!words.includes(t)) return false;
    /* Not the back control. It sits just under the bar and it NAMES where it
     * goes, so on ask-money its word is "Money" and on the leaf check it is "My
     * plots" -- both of them nav words, both of them bold, and neither of them a
     * claim about where you are. deskNav skips it for the same reason. */
    if (el.closest('[data-chrome], [data-back]')) return false;
    const kid = el.querySelector('div, span');
    return !(kid && kid.textContent.trim() === t);
  });
  return bar.map((el) => {
    const cs = getComputedStyle(el);
    return {
      word: el.textContent.trim(),
      // Any of the three ways a board says "you are here".
      current: Number(cs.fontWeight) >= 600
               || /inset 0 -2px|inset 0px -2px/.test(cs.boxShadow)
               || el.getAttribute('aria-current') === 'page',
      aria: el.getAttribute('aria-current') === 'page',
    };
  });
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter((f) => f.endsWith('.html'))
    .map((f) => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0;

for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role === null) { console.log(`  ${slug.padEnd(20)} no bar`); continue; }
  if (!tok[role]) tok[role] = await login(role);
  const ctx = await b.newContext({ viewport: { width: 1366, height: 900 } });
  const p = await ctx.newPage();
  await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1400);
  const bar = await p.evaluate(READ, [...WORDS]);
  await ctx.close();

  if (!bar.length) { console.log(`  ${slug.padEnd(20)} no nav words in the bar`); continue; }
  const shown = bar.filter((w) => w.current).map((w) => w.word);
  const want = OWNS[slug] || null;
  const line = bar.map((w) => (w.current ? `[${w.word}]` : w.word)).join('  ');

  if (shown.length > 1) {
    bad++; console.log(`  ${slug.padEnd(20)} ${shown.length} words claim to be this page   ${line}`);
  } else if (want && shown[0] !== want) {
    bad++; console.log(`  ${slug.padEnd(20)} says ${shown[0] ? `"${shown[0]}"` : 'nothing'}, this page is "${want}"   ${line}`);
  } else if (!want && shown.length) {
    bad++; console.log(`  ${slug.padEnd(20)} claims "${shown[0]}", which is another page   ${line}`);
  } else if (want && !bar.some((w) => w.word === want && w.aria)) {
    bad++; console.log(`  ${slug.padEnd(20)} looks right but carries no aria-current   ${line}`);
  } else {
    console.log(`  ${slug.padEnd(20)} ok   ${line}`);
  }
}

await b.close();
console.log(`\n  ${bad} page(s) whose bar misstates where you are, across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
