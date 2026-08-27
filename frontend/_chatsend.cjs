const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
  const errs = []; let navs = 0;
  p.on('pageerror', e => errs.push('PAGEERROR ' + e.message.slice(0,140)));
  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', 'demo.farmer@pestivid.sim');
  await p.fill('input[type="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });
  await p.goto('http://localhost:3001/app/messages.html');
  await p.waitForTimeout(1200);
  await p.screenshot({ path: `${OUT}/chat-list.png` });
  await p.click('.n, .nu');
  await p.waitForTimeout(1600);
  await p.screenshot({ path: `${OUT}/chat-open.png` });
  // count navigations from here on: a reload is the flicker
  p.on('framenavigated', f => { if (f === p.mainFrame()) navs++; });
  const before = await p.evaluate(() => document.querySelectorAll('.them,.me').length);
  await p.fill('input[name="reply"]', 'The far end had water on Tuesday. I will film it tomorrow morning.');
  await p.waitForTimeout(200);
  await p.click('[data-send]');
  await p.waitForTimeout(260);
  await p.screenshot({ path: `${OUT}/chat-sending.png` });
  await p.waitForTimeout(2200);
  await p.screenshot({ path: `${OUT}/chat-sent.png` });
  const after = await p.evaluate(() => document.querySelectorAll('.them,.me').length);
  console.log('bubbles', before, '->', after, '| navigations during send:', navs,
              '| days', await p.evaluate(() => [...document.querySelectorAll('.day')].map(d=>d.textContent)));
  if (errs.length) console.log(errs.join('\n'));
  await b.close();
})();
