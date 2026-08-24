// Finds things that LOOK pressable and are not wired.
//
// click-everything.mjs enumerates only what a page MARKS as a control, so
// anything drawn to look like a button and never wired is invisible to it: it
// reports "0 controls" instead of "3 dead". That is how three language chips on
// the farmer sign-in page passed every sweep.
//
// This asks the opposite question. It looks for the visual signature of a
// control -- a painted or ringed box holding a short label, or a chevron, or
// link-coloured text -- and reports the ones with no handler behind them.
import { chromium } from 'playwright';
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

const ROLE = {
  landing: null, signin: null, signup: null, 'signin-farmer': null,
  'setup-language': null, 'setup-identity': null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  'thread-investor': 'investor', market: 'buyer', orders: 'buyer', admin: 'admin',
};
const roleFor = (s) => (s in ROLE ? ROLE[s] : 'farmer');

const FIND = () => {
  const out = [];
  const LINK = ['rgb(1, 87, 155)', 'rgb(1, 106, 190)', 'rgb(1, 33, 105)'];
  for (const el of document.querySelectorAll('body div, body span, body a')) {
    if (el.closest('[data-act], [data-go], a[href], [data-fold], [role="button"], [role="link"]')) continue;
    if (el.closest('[data-readout], [data-title]')) continue;
    // A section heading is a heading. It happens to sit beside the first row of
    // its own list, and that row has the chevron.
    if (el.classList.contains('sec') || el.classList.contains('lbl')) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 24 || r.height < 14 || r.width > 900) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;
    const text = (el.textContent || '').replace(/\s+/g, ' ').trim();
    if (!text || text.length > 46) continue;
    if ([...el.children].some(c => (c.textContent || '').trim())) continue;   // leaves only

    const painted = cs.backgroundColor !== 'rgba(0, 0, 0, 0)';
    const ringed = /inset/.test(cs.boxShadow);
    const linked = LINK.includes(cs.color) && Number(cs.fontWeight) >= 600;
    // The chevron has to be a SIBLING of this label, not merely somewhere inside
    // the same card -- otherwise every section heading in a card that contains
    // one row reads as a control.
    const chevron = [...(el.parentElement?.children || [])]
      .some(c => c !== el && c.querySelector?.('svg path[d^="M9 6l6 6-6 6"]'));
    // An all-caps tracked label is a label. This product uses that treatment for
    // its claim tiers -- PROVED, THEIR WORD, NOBODY KNOWS -- and they are
    // headings, not destinations.
    const trackedLabel = parseFloat(cs.letterSpacing) > 0.4 && text === text.toUpperCase();
    // Bold inside a sentence is emphasis, not a link: the parent is prose.
    const insideProse = (el.parentElement?.textContent || '').trim().length > text.length + 12;
    // A painted box that is merely the page's own surface is not a button.
    const surface = ['rgb(246, 243, 239)', 'rgb(255, 255, 255)', 'rgb(221, 215, 209)',
                     'rgb(243, 246, 249)', 'rgba(0, 0, 0, 0)'].includes(cs.backgroundColor);
    const looksLikeControl = linked || chevron || (ringed && !surface) || (painted && !surface);
    if (!looksLikeControl) continue;
    if (trackedLabel || (linked && insideProse)) continue;

    out.push({
      text: text.slice(0, 40),
      why: linked ? 'link-coloured and bold' : chevron ? 'has a chevron'
         : ringed ? `ringed ${cs.boxShadow.slice(0, 22)}` : `filled ${cs.backgroundColor}`,
      size: `${Math.round(r.width)}x${Math.round(r.height)}`,
    });
  }
  return out;
};

const slugs = process.argv.slice(2);
const b = await chromium.launch();
const tok = {};
let total = 0;
for (const slug of slugs) {
  const role = roleFor(slug);
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html`, { waitUntil: 'load' });
  await p.waitForTimeout(1400);
  const found = await p.evaluate(FIND);
  if (found.length) {
    console.log(`\n## ${slug}`);
    for (const f of found.slice(0, 8)) console.log(`  "${f.text}"  ${f.size}  ${f.why}`);
    if (found.length > 8) console.log(`  … ${found.length - 8} more`);
  }
  total += found.length;
  await p.close();
}
await b.close();
console.log(`\n${total} things that look pressable and are not wired, across ${slugs.length} pages`);
