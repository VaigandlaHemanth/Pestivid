// Does the back chevron work, on every page that draws one? The user reported
// it dead on some pages and landing somewhere unexpected on others.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
    headers: {'content-type':'application/json'},
    body: JSON.stringify({email:'demo.' + r + '@pestivid.sim',password:'password123'})})).json();
  const toks = {};
  for (const r of ['farmer','investor','buyer','admin']) toks[r] = await login(r);
  // slug -> [where a person came from, whose account]
  const from = {
    record: ['plots','farmer'], ask: ['home','farmer'], 'leaf-check': ['plots','farmer'],
    'ask-money': ['money','farmer'], sent: ['record','farmer'], profile: ['home','farmer'],
    plot: ['plots','farmer'], money: ['home','farmer'], messages: ['home','farmer'],
    plots: ['home','farmer'], payout: ['money','farmer'], setup: ['home','farmer'],
    'report-harvest': ['money','farmer'],
    invest: ['portfolio','investor'], portfolio: ['invest','investor'],
    'confirm-investment': ['invest','investor'],
    market: ['orders','buyer'], orders: ['market','buyer'],
    admin: ['admin','admin'], thread: ['messages','farmer'],
  };
  let dead = 0, wrong = 0;
  const need = await (await import('./_needs.mjs')).needs();
  for (const [slug, [came, role]] of Object.entries(from)) {
    const tok = toks[role];
    const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
    await p.addInitScript(([t,u]) => { localStorage.setItem('pv.token',t); localStorage.setItem('pv.user',u); },
                          [tok.token, JSON.stringify(tok.user)]);
    // arrive the way a person does: from another page
    await p.goto('http://127.0.0.1:3001/app/' + came + '.html');
    await p.waitForTimeout(700);
    // goto() sends no Referer, so without this every page took the FALLBACK
    // path and the test could not tell a working history from a lucky default.
    await p.goto('http://127.0.0.1:3001/app/' + slug + '.html' + (need[slug] || ''),
                 { referer: 'http://127.0.0.1:3001/app/' + came + '.html' });
    await p.waitForTimeout(1300);
    const has = await p.evaluate(() => {
      const c = document.querySelector('[data-chrome="back"]');
      return c ? { wired: c.hasAttribute('data-act') || c.hasAttribute('data-go')
                          || !!c.closest('[data-act]'), } : null;
    });
    if (!has) { console.log('  ' + slug.padEnd(16) + 'no chevron drawn'); await p.close(); continue; }
    if (!has.wired) { console.log('  ' + slug.padEnd(16) + 'DEAD — chevron has no handler'); dead++; await p.close(); continue; }
    await p.click('[data-chrome="back"]');
    await p.waitForTimeout(1200);
    const landed = p.url().split('/app/')[1]?.split('?')[0]?.replace('.html','') || p.url();
    const ok = landed === came;
    if (!ok) wrong++;
    console.log('  ' + slug.padEnd(16) + (ok ? 'ok   ' : 'WRONG') + '  from ' + came + ' -> back landed on ' + landed);
    await p.close();
  }
  await b.close();
  console.log('\n' + dead + ' dead, ' + wrong + ' landing somewhere other than where you came from');
})();
