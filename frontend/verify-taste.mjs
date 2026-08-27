// The rest of the two skills' pre-flight lists, measured on every page.
//
// verify-hig.mjs covers contrast, control size, full-bleed buttons and sentence
// case. verify-palette.mjs covers the colour lock. verify-motion-coverage.mjs
// covers whether a page moves. This is what was left, and it is the half that
// gets asserted by eye and therefore never gets asserted:
//
//   focus visible        WCAG 2.4.7, and the HIG's own keyboard guidance. A
//                        laptop product where Tab shows nothing is unusable
//                        without a mouse.
//   CTA label wrap       taste 4.5: a button label on two lines is broken.
//   duplicate CTA intent taste 4.5: two buttons meaning "contact" is a fail.
//   placeholder contrast taste 4.5 form check. A placeholder is a pseudo
//                        element, so the contrast pass cannot see it.
//   banned strings       taste 9.F: scroll cues, version labels, section-number
//                        eyebrows, "Quietly trusted by".
//   eyebrow budget       taste 4.7: at most one per three sections.
import { chromium } from 'playwright';
import { readdirSync } from 'node:fs';
import { needs } from './_needs.mjs';

const QUERY = await needs();
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

const ROLE = {
  landing: null, signin: null, signup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  thread: 'farmer', market: 'buyer', orders: 'buyer', admin: 'admin',
};

// Same intent, different words. Two of these on one page is the finding.
const INTENT = [
  ['contact', /^(get in touch|contact us|let'?s talk|start a project|reach out)$/i],
  ['signup', /^(try free|get started|sign up free|create an account|start now)$/i],
  ['portfolio', /^(view work|see selected work|browse projects)$/i],
];

const BANNED = [
  ['scroll cue', /^(scroll|scroll down|scroll to explore|scroll to walk through it|↓ ?scroll)$/i],
  ['version label', /^(v\d+(\.\d+)*|beta|alpha|invite-only preview|early access|build \d+)$/i],
  ['section-number eyebrow', /^\d{2,3}\s*[\/·-]\s*\w/],
  ['performative label', /^(quietly (in use at|trusted by)|from the field|field notes|on our desks|currently on the bench)$/i],
];

const AUDIT = () => {
  const out = { wrap: [], intent: [], placeholder: [], banned: [], eyebrows: 0, sections: 0 };
  const px = (v) => parseFloat(v) || 0;
  const rgb = (s) => { const m = (s || '').match(/[\d.]+/g); return m ? { r: +m[0], g: +m[1], b: +m[2], a: m[3] == null ? 1 : +m[3] } : null; };
  const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const opaqueBehind = (el) => {
    for (let n = el; n; n = n.parentElement) {
      const c = rgb(getComputedStyle(n).backgroundColor);
      if (c && c.a >= 0.95) return c;
    }
    return { r: 255, g: 255, b: 255, a: 1 };
  };

  // ---- a button label must fit on one line ---------------------------------
  for (const el of document.querySelectorAll('[data-act], [data-go], button')) {
    const cs = getComputedStyle(el);
    const bg = rgb(cs.backgroundColor);
    if (!bg || bg.a < 0.5) continue;                  // filled controls only
    const r = el.getBoundingClientRect();
    if (r.height < 20 || r.width < 20) continue;
    const text = (el.textContent || '').trim();
    if (!text || text.length > 40) continue;          // a row, not a button
    // Count the LINES the text actually occupies, with a Range over the text
    // nodes. Measuring the element's height instead reported every 44px button
    // with a 16px label as a two-line wrap -- the padding was the second line.
    const nodes = [];
    const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = walk.nextNode(); n; n = walk.nextNode()) if (n.textContent.trim()) nodes.push(n);
    let lines = 0;
    for (const n of nodes) {
      const range = document.createRange();
      range.selectNodeContents(n);
      const rects = [...range.getClientRects()].filter(r => r.width > 1 && r.height > 1);
      // rects on the same baseline are one line
      const tops = new Set(rects.map(r => Math.round(r.top)));
      lines = Math.max(lines, tops.size);
    }
    if (lines > 1) out.wrap.push(`"${text}" over ${lines} lines`);
  }

  // ---- two controls that mean the same thing -------------------------------
  const labels = [...document.querySelectorAll('[data-act], [data-go], button')]
    .map(e => (e.textContent || '').trim()).filter(Boolean);
  out.labelList = labels;

  // ---- a placeholder has to be readable -----------------------------------
  for (const el of document.querySelectorAll('input, textarea')) {
    if (!el.placeholder) continue;
    const cs = getComputedStyle(el, '::placeholder');
    const fg = rgb(cs.color) || rgb(getComputedStyle(el).color);
    if (!fg) continue;
    const got = ratio(fg, opaqueBehind(el));
    if (got < 4.5) out.placeholder.push(`"${el.placeholder.slice(0, 26)}" ${Math.round(got * 100) / 100}:1`);
  }

  // ---- strings the skill bans outright ------------------------------------
  for (const el of document.querySelectorAll('body *')) {
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim();
    if (!own || own.length > 44) continue;
    out.bannedRaw = out.bannedRaw || [];
    out.bannedRaw.push(own);
  }

  /* ---- eyebrow budget: one small wide-tracked caps label per three sections
   *
   * An eyebrow is a LABEL a designer put above a section. A [data-readout] is a
   * value the server produced, and a date stamp that happens to be short and
   * tracked is not a section label -- counting those made the evidence panel's
   * "26 Aug 13:05" an eyebrow.
   */
  const eyeSeen = new Set();
  for (const el of document.querySelectorAll('body *')) {
    if (el.closest('[data-readout], [data-bind], .m')) continue;
    const cs = getComputedStyle(el);
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim();
    if (own && px(cs.fontSize) <= 13.5 && px(cs.letterSpacing) >= 0.5
        && (cs.textTransform === 'uppercase' || own === own.toUpperCase())
        && own.length > 3 && /[A-Za-z]/.test(own) && !/^[\d\s\W]+$/.test(own)) eyeSeen.add(own);
  }
  out.eyebrows = eyeSeen.size;
  out.eyebrowList = [...eyeSeen];
  // A "section" in a product built from divs is a heading, not a <section> tag.
  out.sections = new Set([...document.querySelectorAll('body *')].filter(el => {
    const cs = getComputedStyle(el);
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim();
    return own && px(cs.fontSize) >= 16 && Number(cs.fontWeight) >= 600;
  }).map(el => el.textContent.trim())).size || 1;

  // ---- focus: does Tab show anything? -------------------------------------
  out.focusable = [...document.querySelectorAll('[data-act], [data-go], button, a[href], input, select')]
    .filter(e => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4; }).length;
  return out;
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
const tally = { wrap: 0, intent: 0, placeholder: 0, banned: 0, focus: 0, eyebrow: 0 };

for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 2200 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1300);
  const r = await p.evaluate(AUDIT);

  const lines = [];
  for (const w of new Set(r.wrap)) { lines.push(`  CTA wraps: ${w}`); tally.wrap++; }
  for (const q of new Set(r.placeholder)) { lines.push(`  placeholder contrast: ${q}`); tally.placeholder++; }

  // duplicate intent, per page
  for (const [name, re] of INTENT) {
    const hit = [...new Set((r.labelList || []).filter(l => re.test(l)))];
    if (hit.length > 1) { lines.push(`  two "${name}" CTAs: ${hit.join(' / ')}`); tally.intent++; }
  }
  for (const [name, re] of BANNED) {
    const hit = [...new Set((r.bannedRaw || []).filter(s => re.test(s)))];
    for (const h of hit) { lines.push(`  banned ${name}: "${h}"`); tally.banned++; }
  }
  const budget = Math.ceil(r.sections / 3);
  if (r.eyebrows > budget) {
    lines.push(`  eyebrows: ${r.eyebrows} for ${r.sections} sections, budget ${budget}`);
    tally.eyebrow++;
  }

  /* Focus, with the actual keyboard.
   *
   * el.focus() from script sets :focus but NOT :focus-visible -- the browser
   * only grants that to keyboard interaction -- so a scripted check reported
   * eleven blind controls on a page whose ring works. Press Tab, like a person.
   */
  const blind = [];
  {
    const total = await p.evaluate(() =>
      [...document.querySelectorAll('[data-act], [data-go], button, a[href], input, select')]
        .filter(e => { const r = e.getBoundingClientRect(); return r.width > 4 && r.height > 4; }).length);
    await p.evaluate(() => document.body.setAttribute('tabindex', '-1'));
    await p.evaluate(() => document.body.focus());
    for (let i = 0; i < Math.min(total, 34); i++) {
      await p.keyboard.press('Tab');
      const seen = await p.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return null;
        const r = el.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) return null;
        const cs = getComputedStyle(el);
        // the ring may be drawn on this element or on the field box around it
        const box = el.closest('[data-fieldbox]');
        const bcs = box ? getComputedStyle(box) : null;
        const ringed = (e) => e && e.outlineStyle !== 'none' && parseFloat(e.outlineWidth) >= 1;
        const shows = ringed(cs) || ringed(bcs) || /inset|rgb/.test(cs.boxShadow || '');
        return shows ? null : (el.getAttribute('aria-label') || el.textContent || el.name || el.tagName)
          .toString().trim().slice(0, 26);
      });
      if (seen) blind.push(seen);
    }
  }
  const uniqueBlind = [...new Set(blind)];
  if (uniqueBlind.length) {
    lines.push(`  no visible focus on ${uniqueBlind.length}: ${uniqueBlind.slice(0, 4).join(', ')}`);
    tally.focus += uniqueBlind.length;
  }
  await p.close();
  if (lines.length) console.log(`\n## ${slug}\n${lines.join('\n')}`);
}
await b.close();
console.log(`\n  ${tally.wrap} wrapped CTA, ${tally.intent} duplicate intent,`
  + ` ${tally.placeholder} placeholder contrast, ${tally.banned} banned string,`
  + ` ${tally.eyebrow} over eyebrow budget, ${tally.focus} control(s) with no visible focus`
  + `, across ${slugs.length} pages`);
