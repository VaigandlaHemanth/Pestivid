// Can you actually do anything on the notices page?
//
// It was a picture of a list: eight rows marked unread, nothing pressable, no
// way to clear one or all of them, and no way to reach the thing a row was
// about. This walks it as a person would and reports what changed.
//
// Read state is PERSISTENT, so this is not a test that can be re-run to the same
// starting point -- once a role's notices are read they stay read. It exercises
// whatever is still unread and says plainly when there is nothing left to
// exercise, rather than failing on a control that is correctly gone.
//
// Two earlier checks here went stale and SILENTLY SKIPPED, which is worse than
// failing -- the run stayed green while testing nothing:
//   - it hunted for an unread row with no chevron, so that clicking it could not
//     navigate. Every row opens something now, so no such row exists. It clicks
//     a row and comes back instead.
//   - it gated the mark-all sweep on aria-disabled="false". The control is no
//     longer disabled when there is nothing to mark; it leaves. Presence is the
//     gate now.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const MAIL = { farmer: 'demo.farmer@pestivid.sim', investor: 'demo.investor@pestivid.sim',
               buyer: 'demo.buyer@pestivid.sim', admin: 'demo.admin@pestivid.sim' };
const NOTICES = 'http://localhost:3001/app/notifications.html';

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
  await p.goto(NOTICES);
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
    // The sweep control is drawn only while it has work. Its absence is the
    // correct state with nothing unread, not a disabled state.
    markall: document.querySelector('[data-markall]')?.textContent.trim() || '(gone)',
    // one glyph per kind, so a message notice must not wear the document
    glyphs: [...document.querySelectorAll('.n .ic svg path:first-child, .nu .ic svg path:first-child')]
      .map(g => ({ 'M4.5': 'proved', 'M7 4': 'money', 'M5 3': 'listing',
                   'M21 ': 'person', 'M12 ': 'wrong' }[g.getAttribute('d').slice(0, 4)] || '?'))
      .join(','),
  }));
  const first = await snap();
  console.log(`${role} at load  `, JSON.stringify(first));
  await p.screenshot({ path: `${OUT}/notif-load.png` });

  let fails = 0;
  const check = (ok, what) => { if (!ok) { fails++; console.log('  FAIL  ' + what); } };
  check(first.rows === 0 || !first.glyphs.includes('?'), 'a row wears a glyph from no kind');

  // One row: it marks itself read on the way out, and the mark survives coming
  // back. Every row navigates now, so this goes there and returns.
  if (first.unread) {
    const head = await p.evaluate(() =>
      document.querySelector('.nu .h')?.textContent.trim().slice(0, 40));
    await p.click('.nu');
    await p.waitForTimeout(1400);
    const went = new URL(p.url()).pathname;
    await p.goto(NOTICES);
    await p.waitForTimeout(1500);
    const now = await snap();
    console.log('  one read  ', JSON.stringify({ unread: now.unread, sub: now.sub }),
                '| it opened:', went);
    check(now.unread === first.unread - 1,
      `clicking one unread row left ${now.unread} unread, expected ${first.unread - 1}`);
    check(went !== '/app/notifications.html', `the row "${head}" went nowhere`);
  } else {
    console.log('  one read   nothing unread for', role, '-- already caught up');
    check(first.markall === '(gone)',
      'nothing unread, yet the sweep control is still on the page: ' + first.markall);
  }

  // And all of them, when the control is there to do it.
  if (await p.$('[data-markall]')) {
    const was = (await snap()).unread;
    await p.click('[data-markall]');
    await p.waitForTimeout(600 + was * 60);
    await p.waitForTimeout(1400);            // the control's own fade out
    const after = await snap();
    console.log('  all read  ', JSON.stringify(after));
    check(after.unread === 0, `swept, yet ${after.unread} rows still unread`);
    check(after.markall === '(gone)', 'nothing left to mark, yet the control stayed');
    check(after.badge === 'hidden' || after.badge === 'gone', 'badge still showing at zero');
  } else {
    console.log('  all read   no control, nothing unread');
  }
  await p.screenshot({ path: `${OUT}/notif-allread.png` });

  await p.reload();
  await p.waitForTimeout(1500);
  const back = await snap();
  console.log('  reloaded  ', JSON.stringify(back), '  <- it stuck on the server');
  check(back.unread === 0, `${back.unread} rows came back unread after a reload`);

  if (errs.length) console.log([...new Set(errs)].join('\n'));
  console.log(fails ? `\n  ${fails} failed` : '\n  all checks passed');
  await b.close();
  process.exit(fails ? 1 : 0);
})();
