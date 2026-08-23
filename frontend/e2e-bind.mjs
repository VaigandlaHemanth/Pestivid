// Drives the bound pages against a running backend and checks that what is on
// screen came from the server, not from the placeholder the board was drawn
// with. Needs `node backend/dev-server.js` up on 3001.
import { chromium } from 'playwright';

const API = 'http://127.0.0.1:3001/api';
const APP = 'http://127.0.0.1:3001/app';
const stamp = Date.now();

const post = async (p, body) => {
  const r = await fetch(API + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  return { status: r.status, body: await r.json().catch(() => null) };
};

// a farmer and an investor, each with a known name so the page can be checked
const people = {
  farmer:   { name: `E2E Farmer ${stamp}`,   email: `f${stamp}@e2e.test`, password: 'Password!234', role: 'farmer' },
  investor: { name: `E2E Investor ${stamp}`, email: `i${stamp}@e2e.test`, password: 'Password!234', role: 'investor' },
  buyer:    { name: `E2E Buyer ${stamp}`,    email: `b${stamp}@e2e.test`, password: 'Password!234', role: 'buyer' },
};
for (const [k, p] of Object.entries(people)) {
  const reg = await post('/auth/register', p);
  if (reg.status !== 201 && reg.status !== 200) throw new Error(`register ${k}: ${reg.status} ${JSON.stringify(reg.body)}`);
  const log = await post('/auth/login', { email: p.email, password: p.password });
  if (log.status !== 200) throw new Error(`login ${k}: ${log.status} ${JSON.stringify(log.body)}`);
  p.token = log.body.token; p.user = log.body.user;
}
console.log('  three accounts created and signed in');

const CHECKS = [
  ['profile',        'farmer',   people.farmer.name],
  ['home',           'farmer',   people.farmer.name.split(' ').slice(0, 2).join(' ')],
  ['plots',          'farmer',   'not filmed anything yet'],
  ['money',          'farmer',   null],
  ['messages',       'farmer',   'you have not read'],
  ['payout',         'farmer',   'No season chosen'],
  ['report-harvest', 'farmer',   'No season chosen'],
  ['thread-farmer',  'farmer',   'No conversation chosen'],
  ['plot',           'farmer',   'No videos on this plot'],
  ['sent',           'farmer',   'no clip to send'],
  ['portfolio',      'investor', people.investor.name],
  ['invest',         'investor', null],
  ['confirm-investment', 'investor', 'No season chosen'],
  ['orders',         'buyer',    'not bought a lot yet'],
  ['market',         'buyer',    null],
  // the nine that were placeholders until now
  ['landing',        'farmer',   'A field you can see'],
  ['home-empty',     'farmer',   null],
  ['setup-language', 'farmer',   'Choose your language'],
  ['setup-identity', 'farmer',   'phone number is your account name'],
  ['ask-money-video','farmer',   'Film the field first'],
  ['leaf-result',    'farmer',   'Take a photo of one leaf'],
  ['leaf-refusal',   'farmer',   'Take a photo of one leaf'],
];

const browser = await chromium.launch();
let bad = 0;
for (const [slug, who, expect] of CHECKS) {
  const p = people[who];
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(String(e).slice(0, 110)));
  await page.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t);
    localStorage.setItem('pv.user', u);
  }, [p.token, JSON.stringify(p.user)]);
  await page.goto(`${APP}/${slug}.html`, { waitUntil: 'load' });
  await page.waitForTimeout(1400);
  const text = await page.evaluate(() => document.body.innerText.replace(/\s+/g, ' '));
  // Only strings that cannot be real data. "Alice Farmer" is in the seed set, so
  // matching on it flagged a genuine notification as a leftover placeholder.
  const placeholderLeft = /Charlie Investor|Deccan Cold Storage|98765 43210|Canal plot &mdash;/.test(text);
  const ok = (!expect || text.includes(expect)) && !placeholderLeft && !errs.length;
  if (!ok) {
    bad++;
    console.log(`  ${slug.padEnd(11)} FAIL${errs.length ? ' js: ' + errs[0] : ''}` +
      `${placeholderLeft ? ' — placeholder still on screen' : ''}` +
      `${expect && !text.includes(expect) ? ` — expected "${expect}"` : ''}`);
    console.log(`      saw: ${text.slice(0, 150)}`);
  } else {
    console.log(`  ${slug.padEnd(11)} ok   ${expect ? `"${expect}" is on screen` : 'loaded from the server'}`);
  }
  await page.close();
}
await browser.close();
console.log(bad ? `  ${bad} of ${CHECKS.length} bound pages are wrong` : `  all ${CHECKS.length} bound pages show server data`);
process.exit(bad ? 1 : 0);
