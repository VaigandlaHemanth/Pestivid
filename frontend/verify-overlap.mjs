// Is any text on any page painted over by something else?
//
// The leaf checker's progress panel spent minutes on screen underneath the
// photograph the farmer had just picked, and not one of the twenty checks in
// this directory could see it. They ask about colour, contrast, touch targets,
// reachability, motion, wiring — and every one of them was satisfied, because
// the panel was correctly coloured, correctly sized, correctly wired, and
// invisible. A single line, `into.style.cssText = ...`, had thrown away the
// `position: absolute` its drawing gave it, and an absolutely positioned image
// paints over anything that is not positioned.
//
// So this asks the only question that catches that: for every piece of text on
// the page, is the thing at that point on screen actually this text?
//
// HOW, and why each guard is there:
//
//   - a Range around the text node, not the element's box. An element's box can
//     be far larger than its glyphs, and sampling empty padding finds overlaps
//     that no reader would ever see.
//   - three samples across the text, and all three must be covered. One sample
//     at the centre reports a word clipped by a hairline as a covered sentence.
//   - the covering element must actually be OPAQUE. Hit testing does not care
//     about transparency: a see-through layout wrapper stacked above the text is
//     returned by elementFromPoint while the text reads perfectly through it.
//     So the coverer, or something between it and the text, has to paint.
//   - ancestors and descendants of the text's own element do not count. A parent
//     is not covering its own child.
//
// A modal or a scrim deliberately covering the page underneath is a real
// finding here, so no page in this product opens one over its own content; if
// one ever does, this check will need to learn about it rather than be widened.
//
// WHAT IT DOES NOT COVER, said plainly: every page as it FIRST LOADS. It would
// not, on its own, have caught the bug it was written for -- the leaf checker's
// panel only exists after a photograph is picked, and nothing here picks one.
// A state you have to interact your way into needs a probe that performs the
// interaction, which is what _leafloading.cjs is for. This is the net under all
// the states that need no interaction at all, on every page, at three widths.
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

// Laptop and phone: an overlap is usually a layout that only fails at one width.
const SIZES = [[1440, 900], [1366, 768], [390, 844]];

const AUDIT = () => {
  const out = [];
  /* Is this text actually on screen, or has a scroll container clipped it away?
   *
   * The first run of this check reported twenty-one covered messages, all on the
   * chat page, all false. A transcript scrolls INSIDE a fixed-height column, and
   * a message scrolled off the top of that column still has a rectangle -- one
   * that sits behind the app bar, which is opaque and is genuinely the thing at
   * that point on screen. The text was not covered; it was somewhere else.
   *
   * So every ancestor that clips gets to narrow where its descendants can be
   * seen, and text outside that is not a finding.
   */
  const onScreen = (box, el) => {
    let clip = { top: 0, left: 0, bottom: innerHeight, right: innerWidth };
    for (let n = el.parentElement; n && n !== document.documentElement; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.overflow === 'visible' && cs.overflowX === 'visible' && cs.overflowY === 'visible') continue;
      const r = n.getBoundingClientRect();
      clip = {
        top: Math.max(clip.top, r.top), left: Math.max(clip.left, r.left),
        bottom: Math.min(clip.bottom, r.bottom), right: Math.min(clip.right, r.right),
      };
    }
    // Most of the text has to survive the clip, not a sliver of it.
    const w = Math.min(box.right, clip.right) - Math.max(box.left, clip.left);
    const h = Math.min(box.bottom, clip.bottom) - Math.max(box.top, clip.top);
    return w > box.width * 0.5 && h > box.height * 0.5;
  };
  const opaque = (el) => {
    for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.backgroundImage !== 'none') return true;
      const m = cs.backgroundColor.match(/rgba?\(([^)]+)\)/);
      if (m) {
        const parts = m[1].split(',').map((s) => parseFloat(s));
        // Anything over a quarter opaque will hide text under it.
        if (parts.length < 4 || parts[3] > 0.25) return true;
      }
      if (/^(IMG|VIDEO|CANVAS|SVG)$/i.test(n.tagName)) return true;
      // Stop at the first painted layer; going further up would credit the page
      // background for covering its own text.
      if (cs.position !== 'static') break;
    }
    return false;
  };

  for (const el of document.querySelectorAll('body *')) {
    if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') continue;
    /* Text hidden ON PURPOSE is not text painted over -- it is not painted.
     *
     * A screen-reader-only heading is a 1x1 box with clip-path: inset(50%), which
     * keeps it in the document and out of the picture. Measuring the TEXT NODE
     * still returns the words at their natural size, so the chat page's hidden
     * h1 was reported as "Chat covered by div" at all three widths. The element's
     * own box, and its clip, are what say whether anybody can see it. */
    const own = el.getBoundingClientRect();
    const ecs = getComputedStyle(el);
    if (own.width < 6 || own.height < 6) continue;
    if (ecs.clipPath && ecs.clipPath !== 'none') continue;
    for (const node of el.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE) continue;
      const text = node.textContent.replace(/\s+/g, ' ').trim();
      if (!text) continue;
      const r = document.createRange();
      r.selectNodeContents(node);
      const box = r.getBoundingClientRect();
      r.detach?.();
      if (box.width < 6 || box.height < 6) continue;
      if (box.bottom < 0 || box.top > innerHeight || box.right < 0 || box.left > innerWidth) continue;
      if (!onScreen(box, el)) continue;

      const y = box.top + box.height / 2;
      const xs = [box.left + box.width * 0.2, box.left + box.width * 0.5,
                  box.left + box.width * 0.8];
      let coveredBy = null;
      let all = true;
      for (const x of xs) {
        if (x < 0 || x > innerWidth - 1 || y < 0 || y > innerHeight - 1) { all = false; break; }
        const top = document.elementFromPoint(x, y);
        if (!top || top === el || el.contains(top) || top.contains(el)) { all = false; break; }
        if (!opaque(top)) { all = false; break; }
        coveredBy = top;
      }
      if (all && coveredBy) {
        out.push({
          text: text.slice(0, 46),
          by: coveredBy.tagName.toLowerCase()
            + (coveredBy.getAttributeNames().filter((a) => a.startsWith('data-'))[0]
              ? `[${coveredBy.getAttributeNames().filter((a) => a.startsWith('data-'))[0]}]` : '')
            + (coveredBy.className && typeof coveredBy.className === 'string'
              ? `.${coveredBy.className.split(' ')[0]}` : ''),
        });
      }
    }
  }
  return out;
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter((f) => f.endsWith('.html'))
    .map((f) => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let found = 0;
for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  for (const [w, h] of SIZES) {
    const p = await b.newPage({ viewport: { width: w, height: h } });
    if (role) {
      await p.addInitScript(([t, u]) => {
        localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
      }, [tok[role].token, JSON.stringify(tok[role].user)]);
    }
    await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
    await p.waitForTimeout(1300);
    const hits = await p.evaluate(AUDIT);
    await p.close();
    for (const hit of hits) {
      console.log(`  ${slug.padEnd(20)} ${String(w).padEnd(5)} "${hit.text}"  covered by ${hit.by}`);
      found++;
    }
  }
}
await b.close();
console.log(`\n  ${found} piece(s) of text painted over, across ${slugs.length} pages`
  + ` at ${SIZES.map(([w]) => w).join('/')}px`);
process.exit(found ? 1 : 0);
