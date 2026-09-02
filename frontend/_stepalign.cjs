// Do the dots, the connector line and the progress bar share one axis?
//
// The pipeline on the send screen and the four-step loader on the leaf checker
// are both vertical steppers: a marker per row, a line between them, and a bar
// underneath. Three things that have to agree on where the left edge of a column
// is, drawn in three different places, with no check that they do.
//
// Reported by eye and measured here: the centre of every marker, the centre of
// every connector, and the bar.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const FAKE_CID = 'bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi';

const AXIS = () => {
  const box = document.querySelector('[data-pipeline]') || document.querySelector('[data-loadhold]');
  if (!box) return { missing: true };
  const c = (el) => { const r = el.getBoundingClientRect(); return Math.round((r.left + r.right) / 2 * 10) / 10; };
  const markers = [...box.querySelectorAll('.dotOk, .dotW, [data-stepdot]')]
    .filter((e) => e.offsetParent).map((e) => ({ cls: e.className || '(mark)', x: c(e),
      w: Math.round(e.getBoundingClientRect().width) }));
  const lines = [...box.querySelectorAll('.line, [data-stepline]')].filter((e) => e.offsetParent)
    .map((e) => ({ x: c(e), w: Math.round(e.getBoundingClientRect().width),
      on: e.hasAttribute('data-on')
        || (e.firstElementChild && getComputedStyle(e.firstElementChild).transform !== 'matrix(1, 0, 0, 0, 0, 0)'
            && !/matrix\(1, 0, 0, 0/.test(getComputedStyle(e.firstElementChild).transform)) }));
  const bar = box.querySelector('[data-loader] .load, .load, [data-bar]');
  const said = box.querySelector('[data-loader] .said, .said');
  return {
    markers, lines,
    bar: bar ? { left: Math.round(bar.getBoundingClientRect().left), x: c(bar),
      w: Math.round(bar.getBoundingClientRect().width) } : null,
    said: said ? { left: Math.round(said.getBoundingClientRect().left), x: c(said) } : null,
    boxLeft: Math.round(box.getBoundingClientRect().left),
    padLeft: parseFloat(getComputedStyle(box).paddingLeft),
  };
};

const report = (name, m) => {
  if (m.missing) { console.log(`  ${name}: no stepper on screen`); return 0; }
  let bad = 0;
  const xs = m.markers.map((k) => k.x);
  const axis = xs.length ? xs[0] : null;
  console.log(`  ${name}`);
  console.log(`    markers  ${m.markers.map((k) => k.x + '(' + k.w + 'px)').join('  ')}`);
  console.log(`    lines    ${m.lines.length ? m.lines.map((l) => l.x + (l.on ? '*' : '')).join('  ') : '(none drawn)'}`);
  if (m.bar) console.log(`    bar      centre ${m.bar.x}, left ${m.bar.left}, ${m.bar.w}px wide`);
  if (m.said) console.log(`    caption  left ${m.said.left}`);
  // every marker on one axis
  for (const k of m.markers) {
    if (Math.abs(k.x - axis) > 0.6) { console.log(`    FAIL a marker sits at ${k.x}, the first at ${axis}`); bad++; }
  }
  // every connector on that same axis
  for (const l of m.lines) {
    if (Math.abs(l.x - axis) > 0.6) {
      console.log(`    FAIL a connector sits at ${l.x}, the markers at ${axis}`); bad++;
    }
  }
  // the bar starts where the text does, or it reads as detached
  if (m.bar && m.said && Math.abs(m.bar.left - m.said.left) > 2) {
    console.log(`    FAIL the bar starts at ${m.bar.left}, its caption at ${m.said.left}`); bad++;
  }
  return bad;
};

(async () => {
  const b = await chromium.launch();
  let bad = 0;

  // ── the send screen's pipeline ────────────────────────────────────────────
  {
    const p = await b.newPage({ viewport: { width: 1366, height: 900 }, deviceScaleFactor: 2 });
    await p.route((u) => /pinata|storj|s3|ipfs/i.test(u.href), (r) =>
      r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ IpfsHash: FAKE_CID }) }));
    await p.route('**/api/videos/upload-url', (r) =>
      r.fulfill({ status: 200, contentType: 'application/json',
        body: JSON.stringify({ url: 'https://api.pinata.cloud/stub', field: 'file', maxBytes: 100000000 }) }));
    await p.route('**/api/videos/confirm-upload', (r) =>
      r.fulfill({ status: 201, contentType: 'application/json',
        body: JSON.stringify({ _id: 'stub', cid: FAKE_CID, hashComputedBy: 'server', anchored: false, crop: 'potato' }) }));
    await p.goto('http://localhost:3001/app/signin.html');
    await p.fill('input[name="who"]', 'demo.farmer@pestivid.sim');
    await p.fill('input[type="password"]', 'password123');
    await p.keyboard.press('Enter');
    await p.waitForURL((u) => !/signin/.test(u.href), { timeout: 20000 });
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
    await p.waitForTimeout(1700);
    bad += report('sent, at rest', await p.evaluate(AXIS));
    const keep = await p.$('[data-keep]');
    if (keep) { await keep.click(); await p.waitForTimeout(2600); }
    bad += report('sent, sending', await p.evaluate(AXIS));
    await p.screenshot({ path: `${OUT}/align-sent.png` });
    await p.close();
  }

  // ── the leaf checker's four steps ─────────────────────────────────────────
  {
    const tok = await (await fetch('http://127.0.0.1:3001/api/auth/login', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo.farmer@pestivid.sim', password: 'password123' }) })).json();
    const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
      sessionStorage.removeItem('pv.leaf');
      window.__pvHold = new Promise((res) => { window.__pvRelease = res; });
      window.PotatoBrowser = { async load({ onProgress }) {
        onProgress('Getting the checker', 100);
        await window.__pvHold;
        return { async predict() { return { status: 'ok', disease: 'Fungi', confidence: 0.83, ms: 412 }; } };
      } };
    }, [tok.token, JSON.stringify(tok.user)]);
    await p.goto('http://127.0.0.1:3001/app/leaf-check.html', { waitUntil: 'load' });
    await p.waitForTimeout(1500);
    const { execFileSync } = require('child_process');
    const os = require('os'); const path = require('path'); const fs = require('fs');
    const shot = path.join(os.tmpdir(), 'pv_leafprobe_1080x1440.jpg');
    if (!fs.existsSync(shot)) {
      execFileSync(require('../backend/node_modules/ffmpeg-static'),
        ['-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1080x1440:duration=1:rate=1',
          '-frames:v', '1', '-q:v', '5', '-y', shot]);
    }
    await p.setInputFiles('input[type="file"]', shot);
    await p.waitForTimeout(1200);
    bad += report('leaf checker', await p.evaluate(AXIS));
    await p.screenshot({ path: `${OUT}/align-leaf.png` });
    await p.close();
  }

  await b.close();
  console.log(bad ? `\n  ${bad} thing(s) off the column axis` : '\n  every marker, connector and bar on one axis');
  process.exit(bad ? 1 : 0);
})();
