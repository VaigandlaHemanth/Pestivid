// A slot drawn to hold a video frame, holding nothing.
//
// Market's own board carries the confession: "carries a video frame at runtime.
// It does not. There is no <video> and no <img> in any generated page except the
// live viewfinder on Record and the captured photo on LeafCheck; this slot is a
// flat #37322d fill, permanently." showPoster() and playsInline() exist now and
// four pages call them -- and the rest still draw the dark rectangle, the play
// triangle and the duration over nothing at all. On the confirmation screen for
// sending somebody money, that empty square sits beside the season's name.
//
// It finds them by what they LOOK like rather than by a list of selectors: a box
// big enough to be a frame, filled with the media token (#37322d) or one of the
// drawn thumb classes, holding no <img> and no <video>. A slot that has a poster
// or a player in it passes; a slot whose page never fills it is a finding.
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
  market: 'buyer', orders: 'buyer', admin: 'admin',
};

const FIND = () => {
  const MEDIA = ['rgb(55, 50, 45)', 'rgb(37, 34, 30)', 'rgb(29, 26, 23)'];
  const CLASS = /\b(thumb|pthumb|tile|plate|lotplate|shot|frame|media)\b/i;
  const out = [];
  const tried = [];
  const shown = [];
  for (const el of document.querySelectorAll('div, a, figure, span')) {
    const r = el.getBoundingClientRect();
    if (r.width < 56 || r.height < 40) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const dark = MEDIA.includes(cs.backgroundColor);
    const named = CLASS.test(String(el.className || '')) || el.hasAttribute('data-thumb')
                  || el.hasAttribute('data-tile') || el.hasAttribute('data-lotplate')
                  || el.hasAttribute('data-lotshot') || el.hasAttribute('data-noposter');
    /* A DARK BOX IS NOT A MEDIA SLOT. The primary button, the language toggle and
     * the landing hero are all this palette's near-black, and the first pass
     * reported every one of them. What makes a slot is holding a frame: it is
     * named as a thumbnail, or it carries a duration readout, or it is a dark box
     * of frame proportions with no words in it at all. */
    const text = (el.textContent || '').replace(/\s+/g, ' ').trim();
    const words = text ? text.split(' ').length : 0;
    const timed = /^\d{1,2}:\d{2}$/.test(text);
    const ratio = r.width / r.height;
    const framey = dark && !words && ratio > 0.8 && ratio < 2.4;
    if (!named && !timed && !framey) continue;
    if (words > 2 && !named) continue;
    // Never a control. A button that happens to be dark and square is a button.
    if (el.closest('button, a, [data-act], [data-go], [role="button"]')
        && !named) continue;
    // Already filled, by a poster, a player, or a background image. The last one
    // matters: the landing hero is a 1366x660 box whose picture is a CSS
    // background, so looking only for <img> and <video> reported the biggest
    // filled surface in the product as an empty slot.
    if (el.querySelector('img, video')) continue;
    /* Something is painted in here. Not only on the box itself: the landing's
     * example record draws a field and a horizon out of gradient bands on four
     * child divs, and the hero's dark panel layers the same way, so looking at
     * the element's own background-image reported both as empty slots. A slot is
     * empty when NOTHING inside it paints. */
    const painted = (n) => {
      const bi = getComputedStyle(n).backgroundImage;
      return bi && bi !== 'none';
    };
    if (painted(el) || [...el.querySelectorAll('*')].some(painted)) continue;
    /* A declared illustration. Not every dark rectangle is a slot waiting for a
     * frame: the landing draws an example record to show what one looks like,
     * and the reason it holds no real frame is written where the decision was
     * made. Declared ones are printed, never silently skipped. */
    if (el.hasAttribute('data-notaslot')) { shown.push(el.getAttribute('data-notaslot')); continue; }
    /* The page DID wire this slot and the record carries no frame. That is a
     * poster missing from the data, not a slot missing from the code, so it is
     * counted and named separately -- cutting a frame for those videos means
     * pulling each file back out of storage, which costs bandwidth somebody
     * pays for, and is a decision rather than a bug. */
    if (el.hasAttribute('data-noposter')) { tried.push(el.tagName); continue; }
    // A nested slot reports once, at the outermost box.
    if (out.some((o) => o.el.contains(el))) continue;
    out.push({ el,
      tag: `${el.tagName}${el.className ? '.' + String(el.className).trim().split(/\s+/)[0] : ''}`,
      w: Math.round(r.width), h: Math.round(r.height),
      near: (el.parentElement?.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 46) });
  }
  return { empty: out.map(({ el, ...rest }) => rest), tried: tried.length, shown: [...new Set(shown)] };
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter((f) => f.endsWith('.html'))
    .map((f) => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0;

for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const ctx = await b.newContext({ viewport: { width: 1366, height: 900 } });
  const p = await ctx.newPage();
  if (role) {
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
  }
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(2200);           // posters are fetched, so give them time
  const { empty, tried, shown } = await p.evaluate(FIND);
  const filled = await p.evaluate(() => document.querySelectorAll('img[data-poster], video').length);
  await ctx.close();

  const note = [filled && `${filled} frame(s) filled`,
                tried && `${tried} wired, no frame in the record`]
    .filter(Boolean).join(', ');
  if (!empty.length) {
    console.log(`  ${slug.padEnd(20)} ok   ${note || 'no media slots'}`);
    for (const d of shown) console.log(`      declared not a slot: ${d}`);
    continue;
  }
  bad += empty.length;
  console.log(`  ${slug.padEnd(20)} ${empty.length} slot(s) the page never wires`
    + (note ? `   (${note})` : ''));
  for (const e of empty.slice(0, 4)) {
    console.log(`      ${e.tag}  ${e.w}\u00d7${e.h}   beside "${e.near}"`);
  }
  if (empty.length > 4) console.log(`      ... and ${empty.length - 4} more`);
}

await b.close();
console.log(`\n  ${bad} slot(s) drawn for a video frame and left empty, across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
