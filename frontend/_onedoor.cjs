// One create-account page, three roles, each landing where it belongs.
const { chromium } = require('playwright');
const CODE = 'testpass246813';
(async () => {
  const b = await chromium.launch();
  let pass = 0, fail = 0;
  const ok = (what, cond, note) => { cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${what}${note ? '  — ' + note : ''}`); };
  for (const [role, home] of [['farmer','home'], ['investor','invest'], ['buyer','market']]) {
    const stamp = Date.now() + Math.floor(Math.random() * 999);
    const mail = `onedoor.${role}.${stamp}@pestivid.sim`;
    const p = await b.newPage({ viewport: { width: 1440, height: 1400 } });
    const errs = [];
    p.on('pageerror', e => errs.push(e.message.slice(0, 120)));
    await p.goto('http://127.0.0.1:3001/app/signup.html');
    await p.waitForTimeout(900);
    await p.click(`[data-role-pick="${role}"]`);
    await p.fill('input[name="name"]', `One Door ${role} ${stamp}`);
    await p.fill('input[name="email"]', mail);
    await p.fill('input[name="new-password"]', CODE);
    await p.click('[data-ack]');
    await p.getByText('Create the account', { exact: true }).click();
    await p.waitForTimeout(2600);
    ok(`${role} signs up and lands on ${home}`, new RegExp(home + '\.html').test(p.url()),
       p.url().split('/app/')[1]);
    ok(`${role}: no script error`, errs.length === 0, errs[0] || '');
    const r = await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: mail, password: CODE }) });
    const body = await r.json().catch(() => ({}));
    ok(`${role}: the account exists with the right role`,
       r.status === 200 && body.user?.role === role, body.user?.role || r.status);
    await p.close();
  }
  await b.close();
  console.log(`\n  ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
