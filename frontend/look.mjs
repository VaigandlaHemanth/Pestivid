// Screenshot live pages into a given directory. Signs in per role first, because
// a page reached without a token screenshots the sign-in screen instead.
import { chromium } from 'playwright';
const OUT = process.argv[2];
const jobs = process.argv.slice(3).map(a => a.split(':'));   // slug:role[:width]
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
const b = await chromium.launch();
for (const [slug, role, w] of jobs) {
  const width = +(w || 0) || (role === 'farmer' ? 390 : 1440);
  const p = await b.newPage({ viewport: { width, height: 900 }, deviceScaleFactor: 2 });
  if (role && role !== 'none') { const s = await login(role);
    await p.addInitScript(([t,u]) => { localStorage.setItem('pv.token',t); localStorage.setItem('pv.user',u); },
                          [s.token, JSON.stringify(s.user)]); }
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html`, { waitUntil: 'load' });
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(1600);
  await p.screenshot({ path: `${OUT}/${slug}.png`, fullPage: true });
  console.log(`  ${slug} ${width}px`);
  await p.close();
}
await b.close();
