// Shoots a page at laptop AND phone width, so "it looks fine" is never a guess
// made at one size.
import { chromium } from 'playwright';
const OUT = process.argv[2];
const jobs = process.argv.slice(3).map(a => a.split(':'));
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
const b = await chromium.launch();
const tok = {};
for (const [slug, role, query] of jobs) {
  for (const [tag, w, h] of [['laptop', 1440, 900], ['phone', 390, 844]]) {
    const p = await b.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    if (role && role !== 'none') {
      if (!tok[role]) tok[role] = await login(role);
      await p.addInitScript(([t, u]) => {
        localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
      }, [tok[role].token, JSON.stringify(tok[role].user)]);
    }
    await p.goto(`http://127.0.0.1:3001/app/${slug}.html${query || ''}`, { waitUntil: 'load' });
    await p.evaluate(() => document.fonts.ready);
    await p.waitForTimeout(1500);
    const m = await p.evaluate(() => ({
      docW: document.documentElement.scrollWidth,
      rootW: Math.round(document.querySelector('body > div')?.getBoundingClientRect().width || 0),
      sideways: document.documentElement.scrollWidth > window.innerWidth + 1,
    }));
    await p.screenshot({ path: `${OUT}/${slug}-${tag}.png`, fullPage: true });
    console.log(`  ${slug.padEnd(20)} ${tag.padEnd(7)} viewport ${w}  root ${m.rootW}px` +
                (m.sideways ? '  SIDEWAYS SCROLL' : '') +
                (tag === 'laptop' && m.rootW < 500 ? '  ← phone-width page on a laptop' : ''));
    await p.close();
  }
}
await b.close();
