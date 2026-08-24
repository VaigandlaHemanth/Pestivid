// Ask a real question and watch what the screen does. The composer must stay
// put, the transcript must scroll, the button must come alive when there is
// something to send.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 940 }, deviceScaleFactor: 2 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message.slice(0, 200)));
  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', 'demo.farmer@pestivid.sim');
  await p.fill('input[name="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });
  await p.goto('http://localhost:3001/app/ask.html');
  await p.waitForTimeout(1200);

  const box = () => p.evaluate(() => {
    const s = document.querySelector('[data-send]');
    const t = document.querySelector('[data-transcript]');
    const r = s.getBoundingClientRect();
    return { sendTop: Math.round(r.top), bg: getComputedStyle(s).backgroundColor,
             disabled: s.getAttribute('aria-disabled') || s.dataset.off || '',
             scrollTop: t ? t.scrollTop : -1, scrollH: t ? t.scrollHeight : -1,
             clientH: t ? t.clientHeight : -1, calls: document.querySelectorAll('[data-call]').length };
  });
  console.log('idle      ', JSON.stringify(await box()));

  const f = p.locator('input[name="question"]');
  await f.click(); await f.type('When do I spray for late blight?');
  await p.waitForTimeout(400);
  console.log('typed     ', JSON.stringify(await box()));
  await p.keyboard.press('Enter');
  await p.waitForTimeout(900);
  await p.screenshot({ path: OUT + '/ask-thinking.png', fullPage: true });
  // wait for the pending bubble to be replaced
  for (let i = 0; i < 40; i++) {
    const pend = await p.evaluate(() => document.body.innerText.includes('Looking through the documents'));
    if (!pend) break;
    await p.waitForTimeout(1000);
  }
  await p.waitForTimeout(600);
  console.log('answered  ', JSON.stringify(await box()));
  console.log('last bubble:', (await p.evaluate(() => {
    const b = [...document.querySelectorAll('.them')].pop();
    return b ? b.innerText.replace(/\s+/g, ' ').slice(0, 300) : 'NONE';
  })));
  await p.screenshot({ path: OUT + '/ask-answered.png', fullPage: true });
  if (errs.length) console.log('PAGE ERRORS:', errs.join(' | '));
  await b.close();
})();
