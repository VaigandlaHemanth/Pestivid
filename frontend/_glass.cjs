// Where glass actually renders, and whether its backdrop earns it.
//
// The rule this product holds itself to: glass needs a backdrop that is both
// FROZEN and VARIED. Over a flat fill it is a tinted rectangle with a blur that
// costs GPU and shows nothing.
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
  let total = 0;
  for (const slug of slugs) {
    const role = slug in ROLE ? ROLE[slug] : 'farmer';
    const t = role ? await get(role) : null;
    const p = await b.newPage({ viewport: { width: 1440, height: 2000 } });
    if (t) await p.addInitScript(([a, u]) => {
      localStorage.setItem('pv.token', a); localStorage.setItem('pv.user', u);
    }, [t.token, JSON.stringify(t.user)]);
    await p.goto('http://127.0.0.1:3001/app/' + slug + '.html');
    await p.waitForTimeout(1300);
    const found = await p.evaluate(() => {
      const out = [];
      for (const el of document.querySelectorAll('body *')) {
        const cs = getComputedStyle(el);
        const bf = cs.backdropFilter || cs.webkitBackdropFilter;
        if (!bf || bf === 'none') continue;
        const r = el.getBoundingClientRect();
        if (r.width < 8 || r.height < 8) continue;
        out.push({
          what: (el.className || el.getAttribute('data-bind') || el.tagName).toString().slice(0, 22),
          size: Math.round(r.width) + 'x' + Math.round(r.height),
          filter: bf.slice(0, 30),
          fill: cs.backgroundColor,
          edge: (cs.boxShadow || 'none').slice(0, 24),
        });
      }
      return out;
    });
    await p.close();
    if (found.length) {
      console.log('## ' + slug + '  (' + found.length + ')');
      for (const g of found) console.log(`   ${g.what.padEnd(22)} ${g.size.padEnd(11)} ${g.filter.padEnd(30)} fill ${g.fill}`);
      total += found.length;
    }
  }
  await b.close();
  console.log('\n  ' + total + ' glass surface(s) rendering, across ' + slugs.length + ' pages');
})();
