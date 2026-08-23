// Renders every artboard in Chromium and measures what an eye cannot check by
// reading source: real clipping, real computed contrast, real control heights,
// and what happens when a phone's font scale is turned up.
//
//   node design/audit/measure.mjs            all on-canvas artboards
//   node design/audit/measure.mjs Plot Money  named ones only
import { chromium } from 'playwright';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const DESIGN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const canvas = JSON.parse(readFileSync(path.join(DESIGN, 'canvas.json'), 'utf8'));
const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');

const only = process.argv.slice(2);
let boards = canvas.artboards.map(a => ({ file: a.file, w: a.w, h: a.h, title: a.title }));
if (only.length) boards = boards.filter(b => only.some(o => b.file.startsWith(o)));

const PAGE_FN = () => {
  // ---- colour maths -------------------------------------------------------
  const parse = c => {
    const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };
  const over = (fg, bg) => ({
    r: fg.r * fg.a + bg.r * (1 - fg.a),
    g: fg.g * fg.a + bg.g * (1 - fg.a),
    b: fg.b * fg.a + bg.b * (1 - fg.a), a: 1,
  });
  const relLum = c => {
    const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const wcag = (a, b) => {
    const l1 = relLum(a), l2 = relLum(b);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  };
  // APCA-W3 0.1.9
  const apca = (txt, bg) => {
    const sY = c => Math.pow(c / 255, 2.4);
    const Y = c => 0.2126729 * sY(c.r) + 0.7151522 * sY(c.g) + 0.0721750 * sY(c.b);
    let t = Y(txt), b = Y(bg);
    t = t > 0.022 ? t : t + Math.pow(0.022 - t, 1.414);
    b = b > 0.022 ? b : b + Math.pow(0.022 - b, 1.414);
    if (Math.abs(b - t) < 0.0005) return 0;
    let s, o;
    if (b > t) { s = (Math.pow(b, 0.56) - Math.pow(t, 0.57)) * 1.14; o = s < 0.1 ? 0 : s - 0.027; }
    else { s = (Math.pow(b, 0.65) - Math.pow(t, 0.62)) * 1.14; o = s > -0.1 ? 0 : s + 0.027; }
    return o * 100;
  };
  // A hero puts its dark backdrop in an absolutely-positioned sibling, not an
  // ancestor. Walking only the ancestor chain reports white-on-cream and calls
  // a perfectly legible headline a failure, so covering layers count too.
  const ordered = Array.from(document.querySelectorAll('body *'));
  const idx = new Map(ordered.map((e, i) => [e, i]));
  const covers = (a, b) => a.left <= b.left + 0.5 && a.top <= b.top + 0.5 &&
                           a.right >= b.right - 0.5 && a.bottom >= b.bottom - 0.5;

  const effBg = el => {
    const rect = el.getBoundingClientRect();
    const mine = idx.get(el);
    let stack = [], n = el, opaqueAncestor = null;
    while (n && n !== document.documentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c && c.a > 0) { stack.push(c); if (c.a === 1) { opaqueAncestor = n; break; } }
      n = n.parentElement;
    }
    // A covering layer only counts if it paints INSIDE the nearest opaque
    // ancestor — otherwise a hero backdrop would be credited with obscuring
    // the white card sitting on top of it.
    const under = [];
    for (const o of ordered) {
      if (o === el || o.contains(el) || el.contains(o)) continue;
      if (idx.get(o) > mine) continue;
      if (opaqueAncestor && !opaqueAncestor.contains(o)) continue;
      const cs = getComputedStyle(o);
      if (cs.position === 'static' || cs.display === 'none') continue;
      const c = parse(cs.backgroundColor);
      if (!c || c.a === 0) continue;
      // a layer is only as big as its clipping ancestors let it paint
      let r = o.getBoundingClientRect(), q = o.parentElement;
      while (q && q !== document.documentElement) {
        if (getComputedStyle(q).overflow !== 'visible') {
          const qr = q.getBoundingClientRect();
          r = {
            left: Math.max(r.left, qr.left), top: Math.max(r.top, qr.top),
            right: Math.min(r.right, qr.right), bottom: Math.min(r.bottom, qr.bottom),
          };
        }
        q = q.parentElement;
      }
      if (covers(r, rect)) under.push(c);
    }
    let acc = { r: 255, g: 255, b: 255, a: 1 };
    for (let i = stack.length - 1; i >= 0; i--) acc = over(stack[i], acc);
    for (const c of under) acc = over(c, acc);
    return acc;
  };

  // ---- collect ------------------------------------------------------------
  const root = document.querySelector('x-dc > div, x-dc div');
  const rootBox = root ? root.getBoundingClientRect() : { width: 0, height: 0, top: 0, left: 0 };

  const textEls = [];
  const controls = [];
  const clipped = [];
  const all = document.querySelectorAll('body *');

  for (const el of all) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const tag = el.tagName.toLowerCase();
    if (tag === 'helmet' || tag === 'style' || tag === 'script' || tag === 'link') continue;
    const box = el.getBoundingClientRect();

    // own text = direct text node children only
    let own = '';
    for (const n of el.childNodes) if (n.nodeType === 3) own += n.nodeValue;
    own = own.replace(/\s+/g, ' ').trim();

    if (own.length) {
      const fg0 = parse(cs.color) || { r: 0, g: 0, b: 0, a: 1 };
      const bg = effBg(el);
      const fg = fg0.a < 1 ? over(fg0, bg) : fg0;
      textEls.push({
        text: own.slice(0, 60),
        size: parseFloat(cs.fontSize),
        weight: cs.fontWeight,
        color: cs.color, bg: `rgb(${Math.round(bg.r)}, ${Math.round(bg.g)}, ${Math.round(bg.b)})`,
        wcag: +wcag(fg, bg).toFixed(2),
        apca: +apca(fg, bg).toFixed(1),
        tnum: cs.fontVariantNumeric.includes('tabular-nums'),
        hasDigits: /\d/.test(own),
        w: +box.width.toFixed(1), h: +box.height.toFixed(1),
      });
    }

    // control candidate: paints its own background, holds a short label, and
    // has no child that paints its own background
    const selfBg = parse(cs.backgroundColor);
    const ring = cs.boxShadow && cs.boxShadow.includes('inset');
    if ((selfBg && selfBg.a > 0) || ring) {
      const txt = (el.textContent || '').replace(/\s+/g, ' ').trim();
      const childPainted = Array.from(el.children).some(ch => {
        const c = parse(getComputedStyle(ch).backgroundColor); return c && c.a > 0;
      });
      if (txt.length > 0 && txt.length <= 42 && !childPainted && box.height > 0) {
        controls.push({ label: txt.slice(0, 42), w: +box.width.toFixed(1), h: +box.height.toFixed(1) });
      }
    }

    // clipping: an ancestor hides overflow and this element sticks out of it
    if (el.scrollHeight - el.clientHeight > 1 && cs.overflow !== 'visible' && el.clientHeight > 0) {
      clipped.push({
        tag, cls: el.className || '', clientH: el.clientHeight, scrollH: el.scrollHeight,
        cut: el.scrollHeight - el.clientHeight,
        first: (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 50),
      });
    }
  }

  return {
    rootW: +rootBox.width.toFixed(1), rootH: +rootBox.height.toFixed(1),
    docH: document.documentElement.scrollHeight,
    textEls, controls, clipped,
  };
};

const SCALE_FN = (factor) => {
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    const fs = parseFloat(cs.fontSize);
    if (fs) el.style.fontSize = (fs * factor) + 'px';
  }
  const root = document.querySelector('x-dc > div, x-dc div');
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.overflow === 'visible' || el.clientHeight === 0) continue;
    const cut = el.scrollHeight - el.clientHeight;
    if (cut > 2) out.push({
      cut, clientH: el.clientHeight,
      first: (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 55),
    });
  }
  return { clipped: out, docH: document.documentElement.scrollHeight };
};

const browser = await chromium.launch();
const results = [];
for (const b of boards) {
  const page = await browser.newPage({ viewport: { width: Math.max(b.w + 80, 500), height: Math.max(b.h + 80, 700) } });
  await page.goto(pathToFileURL(path.join(DESIGN, b.file)).href, { waitUntil: 'load' });
  await page.addScriptTag({ content: shim });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(120);
  const base = await page.evaluate(PAGE_FN);
  const s13 = await page.evaluate(SCALE_FN, 1.3);
  await page.reload({ waitUntil: 'load' });
  await page.addScriptTag({ content: shim });
  await page.evaluate(() => document.fonts.ready);
  const s20 = await page.evaluate(SCALE_FN, 2.0);
  results.push({ ...b, ...base, scale130: s13, scale200: s20 });
  await page.close();
}
await browser.close();

writeFileSync(path.join(DESIGN, 'audit', 'measurements.json'), JSON.stringify(results, null, 1));
console.log(`measured ${results.length} artboards -> design/audit/measurements.json`);
for (const r of results) {
  const mobile = r.w <= 400;
  const lowApca = r.textEls.filter(t => Math.abs(t.apca) < 60 && t.size < 24).length;
  const small = r.controls.filter(c => c.h < (mobile ? 44 : 24)).length;
  console.log(
    `  ${r.file.replace('.dc.html', '').padEnd(13)} ${String(r.rootW).padStart(6)}x${String(r.rootH).padEnd(6)}` +
    ` declared ${r.w}x${r.h}` +
    `  clip:${r.clipped.length}  apca<60:${lowApca}  target<${mobile ? 44 : 24}:${small}` +
    `  clip@130%:${r.scale130.clipped.length}  clip@200%:${r.scale200.clipped.length}`
  );
}
