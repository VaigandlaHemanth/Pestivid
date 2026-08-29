// Text that wears the link colour and cannot be clicked.
//
// On the investor's evidence card the middle tier reads "<farmer> told us,
// nobody has checked" in #01579b -- which is --link, the one colour in this
// palette that means "press this". It is a plain div with cursor: auto. The
// three tiers are meant to be read as proved / claimed / unknown, and the
// claimed one was borrowing the colour of a control.
//
// A colour that means one thing has to mean it everywhere, so this asks the
// rendered page: which text is painted --link or --action, and can it be
// pressed? Anything blue and inert is a promise the page does not keep. The
// reverse -- a control that is NOT blue -- is left alone: plenty of controls in
// this product are buttons, rows and cards rather than blue words.
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

const FIND = () => {
  // --link and --action. Both mean "this is a control".
  const BLUE = ['rgb(1, 87, 155)', 'rgb(1, 106, 190)', 'rgb(1, 33, 105)'];
  const out = [];
  for (const el of document.querySelectorAll('div, span, p, h1, h2, h3, td, li')) {
    if (el.children.length) continue;                 // leaf text only
    const t = (el.textContent || '').replace(/\s+/g, ' ').trim();
    if (t.length < 3) continue;
    const cs = getComputedStyle(el);
    if (!BLUE.includes(cs.color)) continue;
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    // Pressable in any of the ways this product makes things pressable.
    if (el.closest('a, button, [role="button"], [data-go], [data-act], [onclick], label, summary')) continue;
    if (el.onclick || el.parentElement?.onclick) continue;
    if (cs.cursor === 'pointer') continue;
    out.push({ t: t.slice(0, 54), color: cs.color });
  }
  return out;
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
  if (role && !tok[role]) tok[role] = await login(role);
  const ctx = await b.newContext({ viewport: { width: 1366, height: 900 } });
  const p = await ctx.newPage();
  if (role) {
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
  }
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1600);
  const found = await p.evaluate(FIND);
  await ctx.close();
  if (!found.length) { console.log(`  ${slug.padEnd(20)} ok`); continue; }
  bad += found.length;
  console.log(`  ${slug.padEnd(20)} ${found.length} blue and inert:`);
  for (const f of found.slice(0, 4)) console.log(`      ${f.color}  "${f.t}"`);
  if (found.length > 4) console.log(`      ... and ${found.length - 4} more`);
}

await b.close();
console.log(`\n  ${bad} piece(s) of text in the link colour that cannot be pressed,`
  + ` across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
