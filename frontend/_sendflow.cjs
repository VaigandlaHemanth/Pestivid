// Walk the send screen with a stubbed clip and a stubbed storage endpoint.
// NOTHING leaves this machine: the ticket's upload URL and the confirm call are
// both intercepted, so no Pinata file is spent and no bogus video is recorded.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const FAKE_CID = 'bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi';

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1366, height: 900 }, deviceScaleFactor: 2 });
  const errs = [];
  p.on('pageerror', e => errs.push('PAGEERROR ' + e.message.slice(0,180)));
  p.on('console', m => { if (m.type()==='error') errs.push('CONSOLE ' + m.text().slice(0,180)); });

  // storage: answer as Pinata would, without being Pinata
  await p.route(u => /pinata|storj|s3|ipfs/i.test(u.href), r =>
    r.fulfill({ status: 200, contentType: 'application/json',
                body: JSON.stringify({ IpfsHash: FAKE_CID }) }));
  // the ticket, so repeated runs do not spend the real route's 5-per-5-minutes
  await p.route('**/api/videos/upload-url', r =>
    r.fulfill({ status: 200, contentType: 'application/json',
                body: JSON.stringify({ url: 'https://api.pinata.cloud/stub', field: 'file',
                                       maxBytes: 100000000 }) }));
  // the record the server would write, without writing one
  await p.route('**/api/videos/confirm-upload', r =>
    r.fulfill({ status: 201, contentType: 'application/json',
                body: JSON.stringify({ _id: 'stub', cid: FAKE_CID, hashComputedBy: 'server',
                                       anchored: false, crop: 'potato' }) }));

  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', 'demo.farmer@pestivid.sim');
  await p.fill('input[type="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });

  // a clip, as the record screen would have left one
  await p.goto('http://localhost:3001/app/record.html');
  await p.evaluate(async () => {
    const bytes = new Uint8Array(1400000); bytes.fill(7);
    const file = new File([bytes], 'clip.mp4', { type: 'video/mp4' });
    const db = await new Promise((res, rej) => { const q = indexedDB.open('pv.clip', 1);
      q.onupgradeneeded = () => { const d = q.result;
        if (!d.objectStoreNames.contains('clip')) d.createObjectStore('clip'); };
      q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    await new Promise((res, rej) => { const tx = db.transaction('clip', 'readwrite');
      tx.objectStore('clip').put({ file, duration: 38, size: file.size }, 'pending');
      tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
    db.close();
  });

  await p.goto('http://localhost:3001/app/sent.html');
  await p.waitForTimeout(1600);
  await p.screenshot({ path: `${OUT}/sent-arrive.png`, fullPage: true });
  const seen = () => p.evaluate(() => ({
    plots: document.querySelectorAll('.pk, .pkOn').length,
    linesOn: document.querySelectorAll('[data-pipeline] .line[data-on]').length,
    ticked: document.querySelectorAll('[data-pipeline] .dotW[data-done], [data-pipeline] .dotW[data-proved]').length,
    loader: document.querySelector('[data-loader]')?.getAttribute('data-loader') || null,
    said: document.querySelector('[data-loader] .said')?.textContent?.slice(0,60) || null,
    keep: document.querySelector('[data-keep]')?.textContent.trim(),
    loaderInCard: !!document.querySelector('[data-pipeline] [data-loader]'),
  }));
  console.log('arrived  ', JSON.stringify(await seen()));

  // pick a plot, then send
  await p.click('.pk');
  await p.waitForTimeout(300);
  await p.click('[data-keep]');
  await p.waitForTimeout(420);
  console.log('sending  ', JSON.stringify(await seen()));
  await p.screenshot({ path: `${OUT}/sent-sending.png`, fullPage: true });
  await p.waitForTimeout(900);
  await p.screenshot({ path: `${OUT}/sent-step1.png`, fullPage: true });
  console.log('step1    ', JSON.stringify(await seen()));
  await p.waitForTimeout(1400);
  console.log('step2    ', JSON.stringify(await seen()));
  await p.screenshot({ path: `${OUT}/sent-step2.png`, fullPage: true });
  await p.waitForTimeout(1600);
  console.log('settled  ', JSON.stringify(await seen()));
  await p.screenshot({ path: `${OUT}/sent-done.png`, fullPage: true });
  const card = await p.$('[data-pipeline]');
  if (card) await card.screenshot({ path: `${OUT}/sent-card.png` });
  await p.waitForTimeout(900);
  if (card) await card.screenshot({ path: `${OUT}/sent-card2.png` });
  if (errs.length) console.log([...new Set(errs)].join('\n'));
  await b.close();
})();
