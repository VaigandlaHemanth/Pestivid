// Every page, through the Apple HIG rubric, measured rather than eyeballed.
//
// The thresholds are quoted from the guideline documents in
// .claude/skills/apple-design/references/hig:
//
//   accessibility.md  "Text size / Text weight / Minimum contrast ratio:
//                      Up to 17 pts, All, 4.5:1 -- 18 pts, All, 3:1"
//   accessibility.md  "mobile 44x44 pt, 28x28 pt" recommended minimum control size
//   layout.md         "Avoid full-width buttons. Buttons feel at home when they
//                      respect system-defined margins and are inset from the
//                      edges of the screen."
//   writing.md        sentence case for interface text
//
// Run bare it checks every page in frontend/app at both widths. There is no
// mode where it checks nothing and prints a pass.
import { chromium } from 'playwright';
import { readdirSync } from 'node:fs';
import { needs } from './_needs.mjs';

const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

const ROLE = {
  landing: null, signin: null, signup: null, setup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  market: 'buyer', orders: 'buyer', admin: 'admin',
};
const roleFor = (s) => (s in ROLE ? ROLE[s] : 'farmer');

const AUDIT = (touch) => {
  const px = (v) => parseFloat(v) || 0;
  const rgb = (s) => {
    const m = s.match(/[\d.]+/g);
    if (!m) return null;
    return { r: +m[0], g: +m[1], b: +m[2], a: m[3] == null ? 1 : +m[3] };
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
    return (x + 0.05) / (y + 0.05);
  };
  // What is painted behind this text.
  //
  // Walking ANCESTORS alone was wrong and it read as a page-wide failure: the
  // landing hero is light text over a dark panel that is a SIBLING layer, not a
  // parent, so the walk sailed past it to the white body and reported 1.43:1 on
  // seven elements that are in fact fine. elementsFromPoint returns the real
  // paint stack at a point -- ancestors and overlapping layers together, in
  // order -- so read the first opaque background below our own element.
  const behind = (el, r) => {
    const cx = r.left + Math.min(r.width, 40) / 2;
    const cy = r.top + r.height / 2;
    // Only hit-test what is actually on screen. CLAMPING an off-screen point
    // into the viewport was worse than not testing: it sampled whatever
    // happened to be at the bottom edge and reported 1.3:1 on twenty elements
    // that are black on white. The viewport is sized tall enough below that
    // most of a page is in view; anything past it takes the ancestor walk.
    // A point can be inside the viewport and still not be where this element is
    // painted: the two transcripts scroll inside a fixed-height column, so a
    // bubble scrolled out of view keeps its layout box while the composer is
    // what is actually drawn there. Hit-testing that point read a dark bubble's
    // light text against the composer's light ground and called it 1.1:1.
    let clipped = false;
    for (let n = el.parentElement; n && !clipped; n = n.parentElement) {
      const ncs = getComputedStyle(n);
      if (!/auto|scroll|hidden/.test(ncs.overflowY + ncs.overflowX)) continue;
      const nr = n.getBoundingClientRect();
      if (cy < nr.top - 1 || cy > nr.bottom + 1 || cx < nr.left - 1 || cx > nr.right + 1) clipped = true;
    }
    if (!clipped && cx > 1 && cx < innerWidth - 1 && cy > 1 && cy < innerHeight - 1) {
      const stack = document.elementsFromPoint(cx, cy);
      const from = stack.indexOf(el);
      const below = from >= 0 ? stack.slice(from) : [el, ...stack];
      for (const n of below) {
        const cs = getComputedStyle(n);
        if (cs.backgroundImage && cs.backgroundImage !== 'none') return 'image';
        const c = rgb(cs.backgroundColor);
        if (c && c.a >= 0.95) return c;
      }
    }
    for (let n = el; n; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.backgroundImage && cs.backgroundImage !== 'none') return 'image';
      const c = rgb(cs.backgroundColor);
      if (c && c.a >= 0.95) return c;
    }
    return { r: 255, g: 255, b: 255, a: 1 };
  };

  const out = { contrast: [], targets: [], fullWidth: [], caps: [], unmeasured: 0 };
  const panelW = document.body.getBoundingClientRect().width;

  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || px(cs.opacity) < 0.15) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;

    // --- text contrast, own text nodes only -------------------------------
    const own = [...el.childNodes]
      .filter(n => n.nodeType === 3 && n.textContent.trim())
      .map(n => n.textContent.trim()).join(' ');
    if (own) {
      const size = px(cs.fontSize);
      const weight = Number(cs.fontWeight) || 400;
      // WCAG's large-text boundary, which is the stricter reading of the HIG
      // table: 24px, or 18.66px when bold.
      const large = size >= 24 || (size >= 18.66 && weight >= 700);
      const need = large ? 3 : 4.5;
      const bg = behind(el, r);
      const fg = rgb(cs.color);
      // A control that is switched off is SUPPOSED to be low contrast -- that is
      // how it reads as off. WCAG exempts inactive controls and every platform
      // does the same; the ratio that matters is the one it has once enabled.
      const off = el.closest('[aria-disabled="true"], [disabled], .tap[data-off]');
      if (off) { /* exempt while inactive */ }
      else if (bg === 'image' || !fg) { out.unmeasured++; }
      else {
        const got = ratio(fg, bg);
        if (got < need - 0.02) {
          out.contrast.push({ text: own.slice(0, 42), size: Math.round(size * 10) / 10,
                              weight, got: Math.round(got * 100) / 100, need });
        }
      }
    }

    // --- control size and full-bleed --------------------------------------
    const isControl = el.matches('[data-act], [data-go], [role="button"], a[href], button, input, select')
      && !el.closest('[data-act] [data-act]');
    if (isControl) {
      // The real target for a text field is the drawn BOX, not the 21px input
      // sliver inside it: wire.js gives the box `cursor: text` and focuses the
      // input from anywhere in it. Measuring the input alone reported every
      // field on every page as an undersized control.
      let box = r;
      if (el.matches('input, select, textarea')) {
        const fb = el.closest('[data-fieldbox]');
        if (fb) {
          const nr = fb.getBoundingClientRect();
          if (nr.height > box.height) box = nr;
        }
      }
      const min = Math.min(box.width, box.height);
      const floor = touch ? 44 : 28;
      const label = (el.getAttribute('aria-label') || el.textContent || el.getAttribute('placeholder') || '')
        .trim().slice(0, 34);
      // WCAG 2.5.8 and every platform exempt a link inside running text: the
      // target is the line, and padding it out would break the sentence.
      const inline = cs.display.startsWith('inline');
      const parentText = (el.parentElement?.textContent || '').trim();
      const inProse = parentText.length > (el.textContent || '').trim().length + 8;
      if (min < floor - 0.5 && box.width > 4 && !(inline && inProse)) {
        out.targets.push({ label, size: `${Math.round(box.width)}x${Math.round(box.height)}`, floor });
      }
      // "Avoid full-width buttons" -- a filled control running the whole measure
      const filled = (rgb(cs.backgroundColor)?.a || 0) > 0.5;
      if (filled && r.width >= panelW * 0.9 && r.height >= 34 && !touch) {
        out.fullWidth.push({ label, w: Math.round(r.width) });
      }
    }

    // --- sentence case ----------------------------------------------------
    // What the SCREEN says, not what the DOM holds. Three live boards uppercased
    // their table headers and section labels with text-transform, so the text
    // node read "Lot" while the page shouted "LOT", and this check saw nothing.
    const shown = cs.textTransform === 'uppercase' ? own.toUpperCase() : own;
    if (shown.length > 11 && shown === shown.toUpperCase() && /[A-Z]{4,}/.test(shown)
        && !/^[\d\s\W]+$/.test(shown)) {
      out.caps.push(shown.slice(0, 40) + (cs.textTransform === 'uppercase' ? '  (via CSS)' : ''));
    }
  }
  return out;
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
const tally = { contrast: 0, targets: 0, fullWidth: 0, caps: 0, unmeasured: 0 };

for (const slug of slugs) {
  const role = roleFor(slug);
  if (role && !tok[role]) tok[role] = await login(role);
  const lines = [];
  for (const [tag, w, touch] of [['laptop', 1440, false], ['phone', 390, true]]) {
    const p = await b.newPage({ viewport: { width: w, height: touch ? 3200 : 2400 } });
    if (role) await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
    await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
    await p.waitForTimeout(1300);
    const f = await p.evaluate(AUDIT, touch);
    await p.close();
    for (const k of Object.keys(tally)) {
      tally[k] += Array.isArray(f[k]) ? f[k].length : f[k];
    }
    const seen = new Set();
    for (const c of f.contrast) {
      const key = c.text + c.got;
      if (seen.has(key)) continue; seen.add(key);
      lines.push(`  ${tag}  contrast ${c.got}:1 needs ${c.need}:1  ${c.size}px/${c.weight}  "${c.text}"`);
    }
    for (const t of f.targets) lines.push(`  ${tag}  target ${t.size} under ${t.floor}px  "${t.label}"`);
    for (const t of f.fullWidth) lines.push(`  ${tag}  full-bleed button ${t.w}px  "${t.label}"`);
    for (const c of new Set(f.caps)) lines.push(`  ${tag}  ALLCAPS  "${c}"`);
  }
  if (lines.length) console.log(`\n## ${slug}\n${[...new Set(lines)].join('\n')}`);
}
await b.close();
console.log(`\n  ${tally.contrast} contrast, ${tally.targets} small targets,`
  + ` ${tally.fullWidth} full-bleed buttons, ${tally.caps} ALLCAPS`
  + `, across ${slugs.length} pages (${tally.unmeasured} text nodes over imagery, unmeasured)`);
