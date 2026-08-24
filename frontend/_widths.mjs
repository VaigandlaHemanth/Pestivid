// Is every page a fixed-width drawing with no responsive behaviour? Measure it.
import { chromium } from 'playwright';
import { readdirSync } from 'node:fs';
import { needs } from './_needs.mjs';
// Five pages are ABOUT something and are blank without an id.
const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
const ROLE = { landing: null, signin: null, signup: null, 'signin-farmer': null,
  setup: null, invest: 'investor', portfolio: 'investor',
  'confirm-investment': 'investor', 'thread': 'investor', market: 'buyer',
  orders: 'buyer', admin: 'admin' };
const b = await chromium.launch(); const tok = {};
const slugs = readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html',''));
let broken = 0, sliver = 0;
for (const slug of slugs.sort()) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const out = {};
  for (const [tag, w] of [['laptop', 1440], ['phone', 390]]) {
    const p = await b.newPage({ viewport: { width: w, height: 900 } });
    if (role) await p.addInitScript(([t,u]) => { localStorage.setItem('pv.token',t); localStorage.setItem('pv.user',u); },
                                   [tok[role].token, JSON.stringify(tok[role].user)]);
    await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
    await p.waitForTimeout(1100);
    out[tag] = await p.evaluate(() => {
      // `body > div` is the app bar, which is full-bleed on every page, so this
      // measured 1440 whether the drawing underneath was 1320 wide or 360. Five
      // pages sat at phone width for a whole pass and this said nothing.
      // Measure the band the content actually occupies instead.
      let l = Infinity, r = -Infinity;
      for (const e of document.querySelectorAll('body *')) {
        const b = e.getBoundingClientRect();
        if (b.width < 2 || b.height < 2) continue;
        const st = getComputedStyle(e);
        if (st.visibility === 'hidden' || st.display === 'none' || st.position === 'fixed') continue;
        l = Math.min(l, b.left); r = Math.max(r, b.right);
      }
      return {
        over: document.documentElement.scrollWidth - window.innerWidth,
        band: Number.isFinite(l) ? Math.round(r - l) : 0,
        drawn: Number(document.body.dataset.designWidth || 0),
      };
    });
    await p.close();
  }
  const overflows = out.phone.over > 1;
  // Two independent signals: what the board was drawn at, and what the page
  // actually fills. confirm-investment is a deliberately narrow card and says so.
  const NARROW_BY_DESIGN = new Set(['confirm-investment']);
  const slivered = !NARROW_BY_DESIGN.has(slug)
    && ((out.laptop.drawn > 0 && out.laptop.drawn < 1000)
        || (out.laptop.band > 0 && out.laptop.band < 900));
  if (overflows) broken++;
  if (slivered) sliver++;
  const flag = [overflows ? `phone overflows by ${out.phone.over}px` : '',
                slivered ? `drawn at ${out.laptop.drawn}px, fills ${out.laptop.band}px of 1440` : '']
                  .filter(Boolean).join(' + ');
  if (flag) console.log(`  ${slug.padEnd(20)} ${flag}`);
}
await b.close();
console.log(`\n  ${broken} pages overflow a phone, ${sliver} show a phone-width sliver on a laptop, of ${slugs.length}`);
