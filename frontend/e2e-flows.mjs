// Walks the two flows that write, rather than the pages that only read.
//
//   sign up by phone  ->  an account exists and the session works
//   ask for money     ->  three screens, then a FundingRequest on the server
//
// A page that renders is not the same as a page that works, so these drive the
// real controls and then check the server, not the screen.
import { chromium } from 'playwright';
import { randomBytes } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync, statSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import ffmpeg from '../backend/node_modules/ffmpeg-static/index.js';

const API = 'http://127.0.0.1:3001/api';
const APP = 'http://127.0.0.1:3001/app';
let pass = 0, fail = 0;
const ok = (n, c, extra = '') => { c ? pass++ : fail++; console.log(`  ${c ? 'ok  ' : 'FAIL'} ${n}${extra ? '  — ' + extra : ''}`); };

const call = async (p, b, t) => {
  const r = await fetch(API + p, { method: b ? 'POST' : 'GET',
    headers: { 'content-type': 'application/json', ...(t ? { authorization: `Bearer ${t}` } : {}) },
    body: b ? JSON.stringify(b) : undefined });
  return { status: r.status, body: await r.json().catch(() => null) };
};

const browser = await chromium.launch();
const stamp = Date.now();
const digits = String(9000000000 + (stamp % 999999999));
const code = '246813';

// ── 1. signing up by phone number ───────────────────────────────────────────
{
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(String(e).slice(0, 90)));
  await page.goto(`${APP}/setup-identity.html`, { waitUntil: 'load' });
  await page.waitForTimeout(500);

  await page.fill('input[name="name"]', `Testcase Farmer ${stamp}`);
  await page.fill('input[name="tel"]', digits);
  await page.fill('input[name="new-password"]', code);
  await page.getByText('Continue', { exact: true }).click();
  await page.waitForTimeout(2000);

  ok('signing up by phone number lands on the empty home',
     /home-empty|home\.html/.test(page.url()), page.url().split('/app/')[1]);
  ok('no script error while signing up', errs.length === 0, errs[0] || '');

  const login = await call('/auth/login', { email: `${digits}@phone.pestivid.local`, password: code });
  ok('the account really exists on the server', login.status === 200);
  var token = login.body?.token;
  var userId = login.body?.user?._id || login.body?.user?.id;
  await page.close();
}

// -- 2. a real clip, because a random blob is refused (and rightly) ----------
{
  // The server will not fingerprint something that is not a video, and a
  // funding request needs the frame hashes, so generate a genuine tiny clip.
  // The timestamp is drawn into the picture to make every run's bytes unique.
  const made = path.join(os.tmpdir(), "pv_flow_" + stamp + ".mp4");
  execFileSync(ffmpeg, ["-y", "-f", "lavfi",
    "-i", "testsrc=size=320x240:rate=10:duration=3",
    "-vf", "drawtext=text=" + stamp + ":fontsize=20:x=10:y=10",
    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast", made],
    { stdio: "ignore" });
  const bytes = readFileSync(made);
  ok("a real clip was generated to upload", bytes.length > 5000, bytes.length + " bytes");

  const ticket = await call("/videos/upload-url", { crop: "potato" }, token);
  if (ticket.status !== 200) {
    ok("a clip is on the account to attach", false, "upload-url said " + ticket.status);
  } else {
    const form = new FormData();
    form.append("file", new Blob([bytes], { type: "video/mp4" }), "flow.mp4");
    form.append("network", "public");
    const up = await fetch(ticket.body.url, { method: "POST", body: form });
    const cid = (await up.json().catch(() => null))?.data?.cid;
    const rec = cid && await call("/videos/confirm-upload",
      { cid, crop: "potato", location: "flowtest", purpose: "funding" }, token);
    ok("the clip is accepted, hashed and fingerprinted", rec?.status === 201,
       rec ? JSON.stringify(rec.body).slice(0, 100) : "no cid returned");
  }
}

// ── 3. asking for money, across the three screens ───────────────────────────
{
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(String(e).slice(0, 90)));
  await page.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [token, JSON.stringify({ _id: userId, name: 'Testcase Farmer', role: 'farmer' })]);

  await page.goto(`${APP}/ask-money-video.html`, { waitUntil: 'load' });
  await page.waitForTimeout(1500);
  // a video whose date has not landed is deliberately not selectable, so this
  // clicks whatever the page offers and reports what happened
  const selectable = await page.locator('[data-act]').count();
  ok('the video step lists something to choose', selectable > 0, `${selectable} tappable`);

  const text1 = await page.evaluate(() => document.body.innerText.replace(/\s+/g, ' '));
  ok('it says what state each video is in',
     /Date stamped|being written|still checking|Film the field first/.test(text1),
     text1.slice(text1.indexOf('Which field'), text1.indexOf('Which field') + 90));

  await page.evaluate(() => {
    // pick the first video row the page made tappable
    const row = [...document.querySelectorAll('[data-act]')].find(e => /·/.test(e.textContent));
    row?.click();
  });
  await page.waitForTimeout(300);
  const picked = await page.evaluate(() => JSON.parse(sessionStorage.getItem('pv.ask') || '{}').cid || null);
  ok('choosing a video is remembered between screens', Boolean(picked));

  if (picked) {
    await page.evaluate(() => {
      const d = JSON.parse(sessionStorage.getItem('pv.ask'));
      sessionStorage.setItem('pv.ask', JSON.stringify({ ...d, crop: 'Potato', acres: 2 }));
    });
    await page.goto(`${APP}/ask-money-amount.html`, { waitUntil: 'load' });
    await page.waitForTimeout(900);
    await page.fill('input[inputmode="numeric"]', '500000');
    await page.getByText('6 months', { exact: true }).click();
    await page.getByText('No sprays at all', { exact: true }).click();
    await page.getByText('Next', { exact: true }).click();
    await page.waitForTimeout(1200);
    ok('the amount step moves on to the terms', /ask-money-terms/.test(page.url()), page.url().split('/app/')[1]);

    await page.waitForTimeout(900);
    const termsText = await page.evaluate(() => document.body.innerText.replace(/\s+/g, ' '));
    ok('with no finished season it refuses to invent a return',
       /not known|nothing to work it out from/.test(termsText));

    await page.getByText('60%', { exact: true }).click();
    await page.getByText('See what investors will see', { exact: true }).click();
    await page.waitForTimeout(2500);
    ok('no script error across the three steps', errs.length === 0, errs[0] || '');

    const mine = await call(`/funding-requests/farmer/${userId}`, null, token);
    const made = Array.isArray(mine.body) ? mine.body : (mine.body?.projects || []);
    ok('a funding request now exists on the server', made.length > 0,
       made[0] ? `${made[0].title} · ${made[0].amount}` : JSON.stringify(mine.body).slice(0, 80));
    ok('and it carries the amount and share that were chosen',
       made[0]?.amount === 500000 && made[0]?.investorShare === 60,
       made[0] ? `amount ${made[0].amount}, share ${made[0].investorShare}` : '');
  }
  await page.close();
}

await browser.close();
console.log(`\n  ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
