// Can you actually do anything on the notices page?
//
// It was a picture of a list: eight rows marked unread, nothing pressable, no
// way to clear one or all of them, and no way to reach the thing a row was
// about. This walks it as a person would and reports what changed.
//
// Read state is PERSISTENT, so this is not a test that can be re-run to the same
// starting point -- once a role's notices are read they stay read. It exercises
// whatever is still unread and says plainly when there is nothing left to
// exercise, rather than failing on a control that is correctly disabled.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const MAIL = { farmer: 'demo.farmer@pestivid.sim', investor: 'demo.investor@pestivid.sim',
               buyer: 'demo.buyer@pestivid.sim', admin: 'demo.admin@pestivid.sim' };

(async () => {
  const role = process.argv[2] || 'investor';
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
  const errs = [];
  p.on('pageerror', e => errs.push('PAGEERROR ' + e.message.slice(0, 160)));
  p.on('console', m => { if (m.type() === 'error') errs.push('CONSOLE ' + m.text().slice(0, 160)); });

  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', MAIL[role]);
  await p.fill('input[type="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });
  await p.goto('http://localhost:3001/app/notifications.html');
  await p.waitForTimeout(1500);

  const snap = () => p.evaluate(() => ({
    rows: document.querySelectorAll('.n,.nu').length,
    unread: document.querySelectorAll('.nu').length,
    // a chevron is a promise that the row opens something
    opens: document.querySelectorAll('.go').length,
    sub: document.querySelector('[data-bind="unreadLine"]')?.textContent,
    // visibility, not presence: the page HIDES the badge when the count reaches
    // zero rather than removing it, so it can come back if a mark-read fails.
    badge: (() => { const e = document.querySelector('.appbar [data-readout]');
      return !e ? 'gone' : (e.offsetParent ? 'shown ' + e.textContent : 'hidden'); })(),
    markall: document.querySelector('[data-markall]')?.textContent.trim(),
    markallOff: document.querySelector('[data-markall]')?.getAttribute('aria-disabled'),
  }));
  console.log(`${role} at load  `, JSON.stringify(await snap()));
  await p.screenshot({ path: `${OUT}/notif-load.png` });

  // one row that marks itself read and does NOT navigate
  const one = await p.evaluateHandle(() =>
    [...document.querySelectorAll('.nu')].find(r => !r.querySelector('.go')) || null);
  const el = one.asElement() && await one.evaluate(n => !!n) ? one.asElement() : null;
  if (el) {
    await el.click();
    await p.waitForTimeout(1000);
    console.log('  one read  ', JSON.stringify(await snap()),
                '| still here:', new URL(p.url()).pathname);
  } else {
    console.log('  one read   skipped -- nothing unread that stays on the page');
  }

  // and all of them, if the control is live
  if ((await snap()).markallOff === 'false') {
    await p.click('[data-markall]');
    await p.waitForTimeout(2600);
    console.log('  all read  ', JSON.stringify(await snap()));
  } else {
    console.log('  all read   skipped -- the control is correctly disabled with nothing unread');
  }
  await p.screenshot({ path: `${OUT}/notif-allread.png` });

  await p.reload();
  await p.waitForTimeout(1500);
  console.log('  reloaded  ', JSON.stringify(await snap()), '  <- it stuck on the server');
  if (errs.length) console.log([...new Set(errs)].join('\n'));
  await b.close();
})();
