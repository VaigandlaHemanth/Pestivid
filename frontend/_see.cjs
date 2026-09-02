// Sign in and photograph real pages at a given width. Every laptop conversion
// gets looked at with eyes, at 1440 and at 390, before it counts as done.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const [role, width, ...slugs] = process.argv.slice(2);
const MAIL = { farmer: 'demo.farmer@pestivid.sim', investor: 'demo.investor@pestivid.sim',
               buyer: 'demo.buyer@pestivid.sim', admin: 'demo.admin@pestivid.sim' };

(async () => {
  const b = await chromium.launch();
  // SHOTH overrides the viewport height, because a page that fits a 940px
  // window can still be cut off on a 1366x768 laptop -- which is the floor this
  // product has to fit, not the ceiling. SHOTCLIP photographs the VIEWPORT
  // rather than the full page, because a fullPage shot hides clipping.
  const vh = +(process.env.SHOTH || (+width < 700 ? 844 : 940));
  const p = await b.newPage({ viewport: { width: +width, height: vh },
                              deviceScaleFactor: 2 });
  const errs = [];
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 160)); });
  p.on('pageerror', e => errs.push('PAGEERROR ' + e.message.slice(0, 160)));

  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', MAIL[role]);
  await p.fill('input[type="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });

  for (const slug of slugs) {
    const [name, qs] = slug.split('?');
    await p.goto('http://localhost:3001/app/' + name + '.html' + (qs ? '?' + qs : ''));
    await p.evaluate(() => document.fonts.ready);
    await p.waitForTimeout(1500);
    const w = await p.evaluate(() => Math.max(document.documentElement.scrollWidth,
                                              document.body.scrollWidth));
    const file = `${OUT}/${name}-${width}${process.env.SHOTH ? 'x' + vh : ''}.png`;
    await p.screenshot({ path: file, fullPage: !process.env.SHOTCLIP });
    console.log(`${name.padEnd(16)} ${width}px  scrollWidth ${w}  -> ${file}`);
  }
  if (errs.length) console.log('CONSOLE ERRORS:\n  ' + [...new Set(errs)].join('\n  '));
  await b.close();
})();
