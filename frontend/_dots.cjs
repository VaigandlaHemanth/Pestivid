// The middle dot is rationed to one per line. It was the default separator for
// everything, which is how "foo · bar · baz · qux" happens.
const { chromium } = require('playwright');
const { readdirSync } = require('fs');
(async () => {
  const b = await chromium.launch();
  const tk = {};
  const get = async (r) => tk[r] ||= await (await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();
  const ROLE = { landing: null, signin: null, signup: null, invest: 'investor',
    portfolio: 'investor', 'confirm-investment': 'investor', thread: 'farmer',
    market: 'buyer', orders: 'buyer', admin: 'admin' };
  const slugs = readdirSync('app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();
  let bad = 0;
  for (const slug of slugs) {
    const role = slug in ROLE ? ROLE[slug] : 'farmer';
    const t = role ? await get(role) : null;
    const p = await b.newPage({ viewport: { width: 1440, height: 2200 } });
    if (t) await p.addInitScript(([a, u]) => {
      localStorage.setItem('pv.token', a); localStorage.setItem('pv.user', u);
    }, [t.token, JSON.stringify(t.user)]);
    await p.goto('http://127.0.0.1:3001/app/' + slug + '.html');
    await p.waitForTimeout(1200);
    const over = await p.evaluate(() => {
      const out = [];
      for (const el of document.querySelectorAll('body *')) {
        const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('');
        const n = (own.match(/\u00b7/g) || []).length;
        if (n > 1) out.push(n + ' dots: ' + own.trim().replace(/\s+/g, ' ').slice(0, 76));
      }
      return [...new Set(out)];
    });
    await p.close();
    if (over.length) { console.log('## ' + slug); over.forEach(l => console.log('  ' + l)); bad += over.length; }
  }
  await b.close();
  console.log('\n  ' + bad + ' line(s) using the middle dot more than once, of ' + slugs.length + ' pages');
})();
