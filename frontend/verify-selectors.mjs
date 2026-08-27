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
  ask: 'Ask', messages: 'Messages', notifications: 'Messages', thread: 'Thread',
  profile: 'Profile', invest: 'Invest', portfolio: 'Portfolio', market: 'Market',
  orders: 'Orders', admin: 'Admin', 'confirm-investment': 'Invest',
};
// Shared modules, and which pages they run on.
const SHARED = { _leaf: ['leaf-check'], _thread: ['thread'], _notices: ['messages', 'notifications'] };

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
console.log(`
  ${dead} selector(s) keyed on a colour instead of a mark,`
  + ` of ${checked} found`);
process.exit(dead ? 1 : 0);
