// Does every style-based selector still match the drawing it was written for?
//
// money.js looked for the harvest button with
// `closest('div[style*="background: #fff"]')`. The two "Report the harvest"
// buttons were later made one button, which turned that one blue, and the
// selector matched nothing from then on: the button was neither wired nor
// removed, silently, on the money screen.
//
// This is checked against the BOARDS, not a running page. A page module sets
// inline styles as it works -- `bar.style.width = '31%'` rewrites the whole
// attribute into rgb() form -- so a live check cannot tell "this selector is
// stale" from "this selector already did its job". The drawing is the thing the
// selector was written against, and the drawing does not move under it.
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

// Which board each page's markup comes from, so a selector is checked against
// the drawing it actually runs on.
const BOARD = {
  landing: 'Main', signin: 'Login', signup: 'Login', home: 'Home', plots: 'Plots',
  plot: 'Plot', record: 'Record', sent: 'Sent', money: 'Money',
  'report-harvest': 'Confirm', 'ask-money': 'AskMoney', 'leaf-check': 'LeafCheck',
  ask: 'Ask', messages: 'Messages', notifications: 'Messages',
  profile: 'Profile', invest: 'Invest', portfolio: 'Portfolio', market: 'Market',
  orders: 'Orders', admin: 'Admin', 'confirm-investment': 'Invest',
};
// Shared modules, and which pages they run on.
const SHARED = { _leaf: ['leaf-check'], _notices: ['notifications'] };

const board = (name) => readFileSync(path.join('design', name + '.dc.html'), 'utf8');
const files = readdirSync('frontend/app/pages').filter(f => f.endsWith('.js'));

let dead = 0, checked = 0;
for (const f of files) {
  const slug = f.replace('.js', '');
  const targets = SHARED[slug] || (BOARD[slug] ? [slug] : null);
  if (!targets) continue;
  const src = readFileSync(path.join('frontend/app/pages', f), 'utf8');
  const lines = src.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.trim();
    if (t.startsWith('//') || t.startsWith('*')) continue;
    for (const m of line.matchAll(/\[style\*="([^"]+)"\]/g)) {
      const needle = m[1];
      /* Any colour-keyed selector at all is the finding.
       *
       * The first version of this check asked "does that colour appear anywhere
       * in the board", and it PASSED the exact bug it was written for: money.js
       * looked for the harvest button by `background: #fff`, the button turned
       * blue, and some other white element in the same board satisfied the
       * search. A static check cannot know which element was meant.
       *
       * So the population is zero instead. Eleven of these existed; each one is
       * a data-* mark now, and a mark cannot drift when a fill changes.
       * A STRUCTURAL hint -- "display: flex", "padding", "grid-template-columns"
       * -- is fine and is not counted: it describes shape, not paint.
       */
      if (!/#[0-9a-fA-F]{3,6}|inset/.test(needle)) continue;
      checked++;
      dead++;
      console.log(`  ${f}:${i + 1}  keyed on paint: [style*="${needle}"]`
        + `  — give the element a data-* mark instead`);
    }
  }
}
/* And the same question about DESTINATIONS: does every page a module sends you
 * to still exist?
 *
 * The two chat screens were merged into one and thread.html was retired. Two
 * modules went on pointing at it -- invest.js and market.js, both on the "ask
 * the farmer a question" action that is the only way a buyer or an investor can
 * reach a farmer -- so the primary secondary action on two pages led to a 404
 * for as long as nobody clicked it. Nothing caught it: it is a string, it parses,
 * the page renders, and the check that clicks everything cannot follow a
 * location.href that only runs after a request succeeds.
 *
 * A page count going DOWN is the moment this matters, and this product has been
 * cutting pages on purpose.
 */
const pages = new Set(readdirSync('frontend/app')
  .filter((f) => f.endsWith('.html')).map((f) => f.replace('.html', '')));
let gone = 0, links = 0;
for (const f of readdirSync('frontend/app/pages').filter((n) => n.endsWith('.js'))
  .map((n) => path.join('frontend/app/pages', n))
  .concat(['frontend/app/chrome.js', 'frontend/app/bind.js', 'frontend/app/wire.js'])) {
  let src;
  try { src = readFileSync(f, 'utf8'); } catch { continue; }
  src.split('\n').forEach((line, i) => {
    const t = line.trim();
    if (t.startsWith('//') || t.startsWith('*')) return;
    // location.href = './slug.html'  and  location.replace('./slug.html')
    for (const m of line.matchAll(/\.\/([a-z0-9-]+)\.html/g)) {
      links++;
      if (!pages.has(m[1])) {
        gone++;
        console.log(`  ${f}:${i + 1}  goes to ./${m[1]}.html, which is not a page any more`);
      }
    }
    /* goes(el, 'slug', ...) — the helper that wires a destination.
     *
     * The second argument, found by balancing brackets rather than by a comma.
     * `goes(slot(el, 'asked'), 'messages', ...)` is a real line in money.js, and
     * a regex reading up to the first comma reported "asked" as a missing page
     * on a call whose destination is perfectly good. A check that cries wolf on
     * a correct line is how a harness gets ignored. */
    for (const at of [...line.matchAll(/\bgoes\(/g)].map((m) => m.index + m[0].length)) {
      let depth = 0, start = at, arg = 0, dest = null;
      for (let c = at; c < line.length && dest === null; c++) {
        const ch = line[c];
        if (ch === '(' || ch === '[') depth++;
        else if (ch === ')' && depth === 0) break;
        else if (ch === ')' || ch === ']') depth--;
        else if (ch === ',' && depth === 0) {
          arg++;
          if (arg === 1) dest = line.slice(start, c);
          start = c + 1;
        }
      }
      const q = dest && dest.trim().match(/^'([a-z0-9-]+)'$/);
      if (!q) continue;                 // a variable destination; nothing to check
      links++;
      if (!pages.has(q[1])) {
        gone++;
        console.log(`  ${f}:${i + 1}  wires a link to "${q[1]}", which is not a page any more`);
      }
    }
  });
}

console.log(`
  ${dead} selector(s) keyed on a colour instead of a mark,`
  + ` of ${checked} found`);
console.log(`  ${gone} destination(s) that no longer exist, of ${links} named`
  + ` across ${pages.size} pages`);
process.exit(dead || gone ? 1 : 0);
