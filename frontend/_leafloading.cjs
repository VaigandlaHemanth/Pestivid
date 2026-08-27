// The state the leaf checker spends MINUTES in, and the only one nothing tested.
//
// _leafstates covers the two screens after the checker runs. Between picking a
// photo and getting an answer there is a 173 MB download, and that screen shipped
// with the progress panel underneath the photograph: the panel's own text ran
// behind the image and was unreadable from the second word on.
//
// The download is stubbed, not performed. window.PotatoBrowser is the whole
// interface _leaf.js loads (load({onProgress}) then predict(file)), so standing a
// fake one up before the page's own script runs reproduces the exact screen at
// any percentage, in about a second, without spending 173 MB.
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const OUT = process.env.SHOTDIR || '.';
const WIDTH = Number(process.argv[3]) || 1440;
const HEIGHT = Number(process.argv[4]) || 900;

/* A photo, and it has to be PORTRAIT: the shape a phone camera produces is the
 * shape that leaves wide dark bars either side of it in a landscape plate, which
 * is the geometry this whole layout turns on. A landscape test frame would fill
 * the plate and hide the bug.
 *
 * Made here rather than committed, so this runs with no arguments like every
 * other probe and no binary sits in the repo for it. ffmpeg-static is already a
 * dependency of the upload path. */
function testPhoto() {
  const out = path.join(os.tmpdir(), 'pv_leafprobe_1080x1440.jpg');
  if (fs.existsSync(out) && fs.statSync(out).size > 0) return out;
  const ff = require('../backend/node_modules/ffmpeg-static');
  execFileSync(ff, ['-v', 'error', '-f', 'lavfi',
    '-i', 'testsrc2=size=1080x1440:duration=1:rate=1', '-frames:v', '1',
    '-q:v', '5', '-y', out]);
  return out;
}
const PHOTO = process.argv[2] || testPhoto();

(async () => {
  const b = await chromium.launch();
  const tok = await (await fetch('http://127.0.0.1:3001/api/auth/login', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: 'demo.farmer@pestivid.sim', password: 'password123' }) })).json();

  const p = await b.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 2 });
  const errs = [];
  p.on('pageerror', e => errs.push(e.message.slice(0, 160)));

  await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    sessionStorage.removeItem('pv.leaf');          // no saved verdict: take a photo
    // The stub. It holds at a percentage until the test lets it go, so the
    // screen can be measured mid-download rather than raced.
    window.__pvHold = new Promise(res => { window.__pvRelease = res; });
    window.PotatoBrowser = {
      async load({ onProgress }) {
        onProgress('Getting the checker', 0);
        for (const pct of [12, 37, 61, 84, 100]) {
          await new Promise(r => setTimeout(r, 60));
          onProgress('Getting the checker', pct);
        }
        await window.__pvHold;                      // park on "Reading your photo"
        return { async predict() {
          return { status: 'ok', disease: 'Fungi', confidence: 0.83,
                   runner_up: 'Bacteria', ms: 412 };
        } };
      },
    };
  }, [tok.token, JSON.stringify(tok.user)]);

  await p.goto('http://localhost:3001/app/leaf-check.html', { waitUntil: 'load' });
  await p.waitForTimeout(1500);

  // The picker is a hidden input the plate clicks for you.
  await p.setInputFiles('input[type="file"]', PHOTO);
  await p.waitForTimeout(900);                      // through the download, onto step 3

  const m = await p.evaluate(() => {
    const plate = document.querySelector('[data-plate]');
    const hold = document.querySelector('[data-loadhold]');
    const img = plate?.querySelector('img');
    const box = (e) => { if (!e) return null; const r = e.getBoundingClientRect();
      return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }; };
    // Is any of the panel's text actually covered by the photograph? Asked of
    // the geometry, not of a screenshot: the note is the widest line and the
    // first thing the image ate.
    const notes = hold ? [...hold.querySelectorAll('div')].filter(d => d.textContent && !d.children.length) : [];
    const ir = img?.getBoundingClientRect();
    const covered = !ir ? [] : notes.filter(n => {
      const r = n.getBoundingClientRect();
      return r.width > 0 && r.right > ir.left && r.left < ir.right
                         && r.bottom > ir.top && r.top < ir.bottom;
    }).map(n => n.textContent.trim().slice(0, 40));
    return {
      plate: box(plate), hold: box(hold), img: box(img),
      holdPosition: hold ? getComputedStyle(hold).position : null,
      // Which paints on top: both positioned means DOM order decides, and the
      // image is prepended, so the panel must be the later one.
      imgIsAfterPanel: !!(img && hold && (img.compareDocumentPosition(hold)
        & Node.DOCUMENT_POSITION_PRECEDING) !== 0),
      covered,
      doc: document.documentElement.scrollHeight, view: innerHeight,
      // Nothing on this page may sit past the fold: it is one screen with one
      // control on it, not a list.
      pastFold: [...document.querySelectorAll('[data-act], button, [role="button"]')]
        .filter(e => e.offsetParent && !e.matches('[data-plate]')
          && e.getBoundingClientRect().bottom > innerHeight + 1)
        .map(e => (e.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 26) || e.tagName),
    };
  });

  await p.screenshot({ path: `${OUT}/leaf-loading.png` });
  await p.evaluate(() => window.__pvRelease());
  await p.waitForTimeout(1800);
  await p.screenshot({ path: `${OUT}/leaf-loading-done.png` });

  console.log(`  plate ${JSON.stringify(m.plate)}`);
  console.log(`  photo ${JSON.stringify(m.img)}`);
  console.log(`  panel ${JSON.stringify(m.hold)}  position: ${m.holdPosition}`);
  console.log(`  page  doc ${m.doc} vs view ${m.view}`);

  const bad = [];
  if (m.covered.length) bad.push(`the photo covers the panel's own words: ${JSON.stringify(m.covered)}`);
  /* Text not overlapping is not the same as a layout that works. The first
   * attempt cleared the panel's words by ONE pixel -- the photo's edge landed at
   * 620 and the text began at 621 -- because the panel's declared 356px width
   * was a content width and its 17px padding sat outside it. So this asks for
   * real air between the two boxes, which is the thing a reader sees. */
  if (m.img && m.hold) {
    const gapX = Math.max(m.img.x, m.hold.x) - Math.min(m.img.x + m.img.w, m.hold.x + m.hold.w);
    const gapY = Math.max(m.img.y, m.hold.y) - Math.min(m.img.y + m.img.h, m.hold.y + m.hold.h);
    const apart = Math.max(gapX, gapY);
    if (apart < 8) {
      bad.push(`the photo and the panel are ${apart}px apart — they need real air, not a hairline`);
    }
  }
  /* Two layouts, two right answers. Over 759px the panel is the plate's second
   * column and stays out of flow; under it the two stack and the panel goes back
   * INTO the flow so the plate can grow to hold both. Asking for `absolute`
   * everywhere failed the stacked layout for doing its job. */
  const stacked = WIDTH <= 759;
  const wantPos = stacked ? 'static' : 'absolute';
  if (m.holdPosition !== wantPos) {
    bad.push(`the panel is position: ${m.holdPosition}, expected ${wantPos} at ${WIDTH}px wide`);
  }
  if (m.img && m.plate && (m.img.h > m.plate.h + 1 || m.img.w > m.plate.w + 1)) {
    bad.push('the photo is larger than the plate it sits in');
  }
  /* One screen is a promise this page only makes on a laptop. A phone scrolls,
   * and the plate is itself wired as the capture control, so it is not a "control
   * past the fold" when it runs on -- it is the surface. */
  if (!stacked) {
    if (m.doc > m.view + 1) bad.push(`the page scrolls ${m.doc - m.view}px at ${WIDTH}x${HEIGHT}`);
    if (m.pastFold.length) bad.push(`past the fold: ${m.pastFold.join(', ')}`);
  }
  if (errs.length) bad.push('script error: ' + errs.join(' | '));

  console.log(bad.length ? bad.map(s => '  FAIL ' + s).join('\n') : '  passed');
  await b.close();
  process.exit(bad.length ? 1 : 0);
})();
