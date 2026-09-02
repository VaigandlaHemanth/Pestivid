const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
(async () => {
  // The message this probe sends is a REAL message in a real conversation, and
  // the dev database persists -- so every run used to add a line to the demo
  // transcript. Snapshot the ids first, take back the difference at the end.
  // Dynamic import because the teardown is ESM and this probe is not.
  const { messageIds, removeMessagesNotIn } = await import('./_teardown.mjs');
  const idsBefore = await messageIds();
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
  // The conversation list is [data-row="person"] on the two-pane chat page. It
  // used to be .n/.nu, borrowed from the notices list back when one module drew
  // both -- so after the merge this probe waited 30s for a class that no page
  // draws any more and died on a timeout instead of testing the send.
  // Waiting for the list, not for a stopwatch -- 1200ms was enough for the
  // drawn page and not always for the fetched one.
  await p.waitForSelector('[data-row="person"]', { timeout: 15000 });
  await p.click('[data-row="person"]');
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
  const days = await p.evaluate(() => [...document.querySelectorAll('.day')].map(d => d.textContent));
  console.log('  bubbles', before, '->', after, '| navigations during send:', navs, '| days', days);
  const bad = [];
  if (after <= before) bad.push('the sent message never appeared');
  // A reload is the flicker the send used to cause. Optimistic send means none.
  if (navs > 0) bad.push(navs + ' navigation(s) during send -- that is the flicker');
  if (errs.length) bad.push(...errs);
  console.log(bad.length ? '  FAIL ' + bad.join('; ') : '  passed');
  if (errs.length) console.log(errs.join('\n'));
  await b.close();
  console.log(`  ${await removeMessagesNotIn(idsBefore)} message(s) taken back`);
  process.exit(bad.length ? 1 : 0);
})();
