// Does every money box group its digits the way its own specimen does?
//
// Two classes of defect, both reported by the owner on report-harvest, both
// likely to exist on other pages for the same reasons.
//
// 1. THE COMMAS. The harvest form printed "2,58,400" on the artboard and gave a
//    bare "945000" the moment somebody typed. Any field that takes an amount has
//    the same trap: the drawn specimen is grouped and the typed value is not, so
//    the one figure the reader is responsible for is the unformatted one. A field
//    marked inputmode="numeric" whose label or placeholder is a money figure has
//    to group as you type.
//
// A SECOND CHECK WAS TRIED AND WITHDRAWN. The report-harvest misalignment --
// an amber panel pushing the form 100px below the button it was about -- is a
// real class of defect, and a heuristic for it ("compare the first painted
// descendant of each column") reported three findings of which two were false:
// invest, whose column boxes are dead level at 60/60 and 203/203 and differ only
// in text baseline, and landing, whose offset is its deliberate asymmetric hero.
// One real finding out of three is not a check, it is noise -- and this repo
// already learned that a check which cries wolf gets ignored, which is worse
// than not having it. Column alignment is judged by eye.
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

const AUDIT = async () => {
  const out = { money: [] };

  /* ---- 1. money boxes -------------------------------------------------
   * Typed digit by digit, because that is when grouping has to happen. A field
   * that only formats on blur still shows a farmer an ungrouped number while
   * they are checking it.
   */
  /* The rule is narrow on purpose: IF THE DRAWN SPECIMEN IS GROUPED, the typed
   * value must be too. That is the defect exactly -- a placeholder reading
   * "5,00,000" beside a box that answers "945000" -- and it excludes the
   * numeric fields that are not money. "Until when?" takes a number of months
   * and its specimen is "12"; a comma there would be nonsense. */
  const looksLikeMoney = (i) => /\d,\d\d/.test(i.placeholder || '');
  for (const i of document.querySelectorAll('input')) {
    if (i.type === 'password' || !looksLikeMoney(i)) continue;
    // a six-figure amount is the shortest that must show a comma in en-IN
    const had = i.value;
    i.focus();
    i.value = '';
    for (const ch of '945000') {
      i.value += ch;
      i.dispatchEvent(new Event('input', { bubbles: true }));
    }
    await new Promise(r => setTimeout(r, 60));
    const got = i.value;
    i.value = had;
    i.dispatchEvent(new Event('input', { bubbles: true }));
    if (!/,/.test(got)) {
      out.money.push(`${i.getAttribute('aria-label') || i.name || 'a field'} -> "${got}"`);
    }
  }

  return out;
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bad = 0;
for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1366, height: 900 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1400);
  const r = await p.evaluate(AUDIT);
  await p.close();

  const lines = r.money.map(m => `no grouping: ${m}`);
  if (lines.length) {
    console.log(`  ${slug.padEnd(20)} ${lines.join('  ·  ')}`);
    bad += lines.length;
  }
}
await b.close();
console.log(`\n  ${bad} money box(es) whose specimen is grouped and whose value is not,`
  + ` across ${slugs.length} pages`);
process.exit(bad ? 1 : 0);
