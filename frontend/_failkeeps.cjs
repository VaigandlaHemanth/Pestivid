// When a write fails, can you still act?
//
// state() replaced everything in the container it was handed, and half the call
// sites hand it the PARENT of a button, because a failure message belongs beside
// the thing that failed. So the failure deleted the thing that failed. On
// confirm-investment an investor whose payment did not go through went from
// three controls to one: "Send Rs 25,000" and "Go back without sending" were
// both replaced by "The money did not move". No retry, and no way off the page.
//
// Every check in this directory passed that screen, because they all look at
// pages as they load and this only exists after a request fails.
//
// Nothing is written here. The request is refused at the wire with route()
// abort, so the failure path runs without moving demo money or creating orders.
const { chromium } = require('playwright');

const CASES = [
  {
    name: 'investment refused',
    role: 'investor',
    url: (ids) => `/app/confirm-investment.html?project=${ids.project}&amount=25000`,
    block: ['**/api/investments**'],
    // check the acknowledgement first, or the send never reaches the request
    prime: async (p) => { await p.click('[data-ack], [data-act][aria-label*="understand" i]'); },
    press: '[data-act][aria-label^="Send "]',
    mustSurvive: [/^Send /, /Go back/],
  },
  {
    name: 'question to a farmer refused',
    role: 'investor',
    url: () => '/app/invest.html',
    // /api/messaging/conversations is what api.messages.open() posts to.
    // Blocking '**/api/messages**' matched nothing, so the request SUCCEEDED
    // and the page navigated away -- and the probe read the empty destination
    // as a screen stripped of its controls.
    block: ['**/api/messaging/conversations'],
    press: '[data-act][aria-label*="question" i], [data-act][aria-label*="ask" i]',
    mustSurvive: [/question|ask/i],
  },
];

const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

(async () => {
  const fr = await (await fetch('http://127.0.0.1:3001/api/funding-requests')).json();
  const list = Array.isArray(fr) ? fr : (fr.projects || fr.requests || []);
  const ids = { project: list[0]?._id || list[0]?.id };
  if (!ids.project) { console.log('  no open funding request to aim at — cannot run'); process.exit(2); }

  const b = await chromium.launch();
  const tok = {};
  let bad = 0, ran = 0;
  for (const c of CASES) {
    if (!tok[c.role]) tok[c.role] = await login(c.role);
    const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
    const errs = [];
    p.on('pageerror', (e) => errs.push(e.message.slice(0, 110)));
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[c.role].token, JSON.stringify(tok[c.role].user)]);
    for (const pattern of c.block) await p.route(pattern, (r) => r.abort('failed'));
    await p.goto('http://127.0.0.1:3001' + c.url(ids), { waitUntil: 'load' });
    await p.waitForTimeout(1700);

    const labels = () => p.evaluate(() => [...document.querySelectorAll('[data-act], [data-go]')]
      .filter((e) => e.offsetParent)
      .map((e) => (e.getAttribute('aria-label') || e.textContent || '').replace(/\s+/g, ' ').trim()));

    const before = await labels();
    const target = await p.$(c.press);
    if (!target) {
      console.log(`  SKIP  ${c.name} — no control matching ${c.press}`);
      await p.close();
      continue;
    }
    ran++;
    if (c.prime) { try { await c.prime(p); await p.waitForTimeout(400); } catch { /* not on this page */ } }
    await target.click();
    await p.waitForTimeout(1800);
    const after = await labels();
    const alert = await p.evaluate(() =>
      document.querySelector('[role="alert"]')?.innerText.replace(/\n/g, ' / ').slice(0, 54) || null);

    const missing = c.mustSurvive.filter((re) => !after.some((l) => re.test(l)));
    const said = alert ? `said "${alert}"` : 'said NOTHING about the failure';
    console.log(`  ${c.name.padEnd(30)} ${before.length} controls -> ${after.length}, ${said}`);
    if (!alert) { console.log('  FAIL  the write failed and the screen did not say so'); bad++; }
    for (const re of missing) {
      console.log(`  FAIL  ${re} is gone after the failure — nothing to retry or leave with`);
      bad++;
    }
    if (errs.length) { console.log('  FAIL  script error: ' + errs.join(' | ')); bad++; }
    await p.close();
  }
  await b.close();
  console.log(bad ? `\n  ${bad} failure(s) leave the screen unusable`
    : `\n  every failed write keeps its controls, across ${ran} path(s)`);
  process.exit(bad ? 1 : 0);
})();
