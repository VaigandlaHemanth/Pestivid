// Signs in as each role, opens every page, and reports anything wrong that a
// person would notice: placeholder text still on screen, an em dash where a
// value should be, a script error, or a control with no accessible name.
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const API = 'http://127.0.0.1:3001/api';
const APP = 'http://127.0.0.1:3001/app';
const APPDIR = path.join(path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), 'frontend', 'app');

// Strings that exist only in the artboards. If one survives on a live page the
// screen is showing invented content.
// Strings that exist only in the artboards. Deliberately excludes names that
// are also seeded users -- Alice Farmer, Bob Buyer, Lakshmi Devi and the rest
// are real rows, and flagging them made the audit cry wolf.
const PLACEHOLDERS = [
  '98765 43210', 'Kadapa, Andhra Pradesh',
  // 'Canal plot' was here. It appears inside the landing page's specimen
  // card, which says on its face that it is an example. A plot name in a
  // labelled specimen is not a false claim, and scrubbing it made the card
  // unreadable.
  '9,32,000', '2,58,400', '6,73,600', '4,04,160', '2,69,440', '18.4L',
  '881,204', '878,410', '3f9c2a1e', 'VID_2026', 'IMG_2026',
  'ICAR-CPRI Technical Bulletin', 'Early blight', 'bafybeih',
  'Sneha Reddy', 'Meena Rao', 'Deccan Cold Storage',
];

const ROLE_PAGES = {
  farmer: ['home', 'home-empty', 'plots', 'plot', 'money', 'messages', 'profile', 'record',
           'sent', 'ask', 'leaf-check', 'leaf-check', 'ask-money-video',
           'ask-money-amount', 'ask-money-terms', 'payout', 'report-harvest', 'thread'],
  investor: ['invest', 'portfolio', 'confirm-investment', 'thread'],
  buyer: ['market', 'orders'],
  admin: ['admin'],
};
const PUBLIC = ['landing', 'signin', 'signup', 'signin-farmer', 'setup-language', 'setup-identity'];

const login = async (email) => {
  const r = await fetch(API + '/auth/login', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email, password: 'password123' }),
  });
  if (!r.ok) throw new Error(`login ${email} -> ${r.status}`);
  return r.json();
};

const browser = await chromium.launch();
const problems = [];

async function visit(page, slug, role) {
  const errs = [];
  const onErr = e => errs.push(String(e).slice(0, 80));
  page.on('pageerror', onErr);
  await page.goto(`${APP}/${slug}.html`, { waitUntil: 'load' });
  await page.waitForTimeout(1500);

  const found = await page.evaluate((list) => {
    const text = document.body.innerText;
    const hits = list.filter(s => text.includes(s));
    // an em dash standing alone is a bound field that got nothing
    const dashes = [...document.querySelectorAll('[data-bind]')]
      .filter(e => e.textContent.trim() === '—')
      .map(e => e.dataset.bind);
    // a control a screen reader cannot name
    const unnamed = [...document.querySelectorAll('[data-act]')]
      .filter(e => !e.getAttribute('aria-label') && !e.textContent.trim())
      .length;
    return { hits, dashes, unnamed };
  }, PLACEHOLDERS);

  page.off('pageerror', onErr);
  if (found.hits.length) problems.push([slug, role, 'placeholder', found.hits.join(', ')]);
  if (found.dashes.length) problems.push([slug, role, 'empty field', found.dashes.join(', ')]);
  if (found.unnamed) problems.push([slug, role, 'unnamed control', `${found.unnamed} with no name`]);
  if (errs.length) problems.push([slug, role, 'script error', errs[0]]);
}

for (const slug of PUBLIC) {
  const page = await browser.newPage();
  await visit(page, slug, 'public');
  await page.close();
}
for (const [role, slugs] of Object.entries(ROLE_PAGES)) {
  const s = await login(`demo.${role}@pestivid.sim`);
  const page = await browser.newPage();
  await page.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [s.token, JSON.stringify(s.user)]);
  for (const slug of slugs) await visit(page, slug, role);
  await page.close();
}
await browser.close();

if (!problems.length) {
  console.log('  every page clean: no placeholder text, no empty bound field, no unnamed control, no script error');
} else {
  console.log(`  ${problems.length} problem(s)\n`);
  for (const [slug, role, kind, detail] of problems) {
    console.log(`  ${slug.padEnd(19)} ${role.padEnd(9)} ${kind.padEnd(16)} ${detail.slice(0, 78)}`);
  }
}
process.exit(problems.length ? 1 : 0);
