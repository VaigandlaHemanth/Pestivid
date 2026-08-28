// Does any page show INVENTED data as if it were real, before its own fetch
// answers?
//
// Every bound list is drawn with example rows, because that is what an artboard
// is for, and the page then replaces them. Measured on the notifications screen,
// the examples were on screen at full opacity for 126ms first -- five notices
// about a "Canal plot" and a "Meena Rao" putting Rs 50,000 into a "North plot",
// none of which happened -- and only then did the real ones fade in. On a
// product whose whole claim is that its records are real, a tenth of a second of
// fiction is not a cosmetic flicker.
//
// HOW IT DECIDES. Not by comparing text: the demo seed deliberately reuses the
// board's own copy, so a drawn row can flash and still look identical to the
// real one that replaced it. What matters is not whether the words coincide, it
// is that the row on screen did not come from the server.
//
// So it watches the two DOM methods that throw drawn rows away -- remove() and
// replaceChildren() -- and records, at the moment each fires, what was standing
// there: its text, whether it was visible, and whether JS had made it. Anything
// the page DELETES that the page did not CREATE was drawn, was on screen, and
// was not data. If it is not marked [data-specimen] -- which responsive.css
// hides before the first paint -- it is a finding.
//
// This needs no list of selectors and no knowledge of repeat()/rows(), so it
// cannot quietly go stale the way a hardcoded probe does.
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

const WATCH = () => {
  window.__drawn = [];
  const made = new WeakSet();
  // Anything JS builds is tagged, so what is left over is what the board drew.
  const ce = Document.prototype.createElement;
  Document.prototype.createElement = function (...a) { const el = ce.apply(this, a); made.add(el); return el; };
  const cn = Node.prototype.cloneNode;
  Node.prototype.cloneNode = function (...a) { const el = cn.apply(this, a); if (el.nodeType === 1) made.add(el); return el; };

  const seen = (el) => {
    if (el?.nodeType !== 1 || made.has(el)) return;
    const txt = (el.textContent || '').replace(/\s+/g, ' ').trim();
    if (txt.length < 8 || !/[a-z]{3}/i.test(txt)) return;
    const cs = getComputedStyle(el);
    const box = el.getBoundingClientRect();
    window.__drawn.push({
      txt,
      tag: `${el.tagName}${el.className ? '.' + String(el.className).trim().split(/\s+/).join('.') : ''}`,
      marked: !!el.closest?.('[data-specimen]'),
      shown: cs.display !== 'none' && cs.visibility !== 'hidden'
             && Number(cs.opacity) > 0.05 && box.width > 1 && box.height > 1,
    });
  };

  const rm = Element.prototype.remove;
  Element.prototype.remove = function () { seen(this); return rm.call(this); };
  const rc = Element.prototype.replaceChildren;
  Element.prototype.replaceChildren = function (...a) {
    for (const c of [...this.children]) if (!a.includes(c)) seen(c);
    return rc.apply(this, a);
  };
  const rch = Element.prototype.removeChild || Node.prototype.removeChild;
  Node.prototype.removeChild = function (c) { seen(c); return rch.call(this, c); };
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter((f) => f.endsWith('.html'))
    .map((f) => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0, ok = 0;

for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const ctx = await b.newContext({ viewport: { width: 1366, height: 900 } });
  const p = await ctx.newPage();
  await p.addInitScript(WATCH);
  if (role) {
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
  }
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1800);
  const drawn = await p.evaluate(() => window.__drawn || []);
  // THE OTHER DIRECTION, and the dangerous one. Hiding [data-specimen] is only
  // safe while nothing real wears the mark: a template that keeps it hands it to
  // every clone, and the whole list renders invisible with no error anywhere.
  // Anything still marked and still carrying text after the page has settled is
  // that failure, so it fails the run louder than the flash it was meant to fix.
  const stuck = await p.evaluate(() => [...document.querySelectorAll('[data-specimen]')]
    .filter((el) => (el.textContent || '').trim().length > 3
                    && getComputedStyle(el).visibility === 'hidden')
    .map((el) => `${el.tagName}${el.className ? '.' + String(el.className).trim().split(/\s+/)[0] : ''}`
                 + `  "${(el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 44)}"`));
  await ctx.close();
  if (stuck.length) {
    bad += stuck.length;
    console.log(`  ${slug.padEnd(20)} ${stuck.length} MARKED element(s) still hidden after the page settled`);
    console.log('      a template kept the mark, so its clones are invisible:');
    for (const t of stuck.slice(0, 4)) console.log(`      ${t}`);
    continue;
  }

  // One entry per distinct element shape: five notice rows are one finding.
  const group = new Map();
  for (const d of drawn.filter((d) => d.shown && !d.marked)) {
    const g = group.get(d.tag) || [];
    g.push(d.txt); group.set(d.tag, g);
  }
  const good = drawn.filter((d) => d.marked).length;
  ok += good;
  if (!group.size) {
    console.log(`  ${slug.padEnd(20)} clean${good ? `  (${good} drawn row(s), marked)` : ''}`);
    continue;
  }
  bad += group.size;
  console.log(`  ${slug.padEnd(20)} ${group.size} drawn thing(s) on screen, then deleted as the data arrived:`);
  for (const [tag, txts] of group) {
    console.log(`      ${tag}   ${txts.length > 1 ? `${txts.length} of them` : ''}`);
    console.log(`          "${txts[0].slice(0, 68)}"`);
  }
}

await b.close();
console.log(`\n  ${bad} drawn element(s) shown as real before the server answered`
  + `, ${ok} correctly marked, across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
