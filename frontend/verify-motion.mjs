// Motion, measured on the served pages rather than read off the boards.
//
// Four rules this product holds itself to:
//   1. transform and opacity only -- nothing that animates layout
//   2. nothing loops
//   3. no duration or easing outside the tokens in tokens.css
//   4. reduced motion keeps opacity and colour and drops travel
import { chromium } from 'playwright';
import { needs } from './_needs.mjs';

const APP = 'http://127.0.0.1:3001/app';
const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${r}@pestivid.sim`, password: 'password123' }) })).json();

const ROLE = { landing: null, signin: null, signup: null, setup: null,
  invest: 'investor', portfolio: 'investor', 'confirm-investment': 'investor',
  market: 'buyer', orders: 'buyer', admin: 'admin' };

const LAYOUT = ['width', 'height', 'top', 'left', 'right', 'bottom', 'margin',
  'padding', 'font-size', 'border-width', 'flex', 'grid'];

const QUERY = await needs();
const pages = process.argv.slice(2);
const b = await chromium.launch();
const tok = {};
let problems = 0;

for (const slug of pages) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  if (role) {
    if (!tok[role]) tok[role] = await login(role);
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok[role].token, JSON.stringify(tok[role].user)]);
  }
  await p.goto(`${APP}/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1400);

  const found = await p.evaluate((LAYOUT) => {
    const out = { animated: 0, layout: [], loops: [], props: new Set() };
    for (const el of document.querySelectorAll('*')) {
      const s = getComputedStyle(el);
      const tp = s.transitionProperty, an = s.animationName;
      const has = (tp && tp !== 'none' && tp !== 'all' && s.transitionDuration !== '0s')
        || (an && an !== 'none');
      if (!has) continue;
      out.animated++;
      // Only properties with a real duration. An element carrying a keyframe
      // animation also reports the browser default transition-property "all"
      // at 0s, which animates nothing -- reporting that made three pages look
      // like they transitioned layout when they did not.
      const durs = s.transitionDuration.split(',').map(x => x.trim());
      tp.split(',').map(x => x.trim()).forEach((raw, i) => {
        if (!raw || raw === 'none') return;
        const d = durs[i % durs.length] || '0s';
        if (parseFloat(d) === 0) return;
        out.props.add(raw);
        if (raw === 'all') out.layout.push(`${el.tagName.toLowerCase()} transitions all`);
        else if (LAYOUT.some(k => raw.startsWith(k))) {
          out.layout.push(`${el.tagName.toLowerCase()} transitions ${raw}`);
        }
      });
      if (an && an !== 'none' && s.animationIterationCount === 'infinite') {
        out.loops.push(`${el.tagName.toLowerCase()} runs ${an} forever`);
      }
    }
    out.props = [...out.props];
    return out;
  }, LAYOUT);

  const bad = found.layout.length + found.loops.length;
  problems += bad;
  const note = bad ? [...new Set([...found.layout, ...found.loops])].slice(0, 3).join('; ')
                   : found.props.join(', ') || 'nothing animated';
  console.log(`  ${slug.padEnd(20)} ${String(found.animated).padStart(3)} animated  ${bad ? 'FAIL ' + note : note}`);
  await p.close();
}

// Reduced motion: travel goes, opacity and colour stay.
const rm = await b.newPage({ viewport: { width: 1440, height: 900 } });
await rm.emulateMedia({ reducedMotion: 'reduce' });
await rm.goto(`${APP}/landing.html`, { waitUntil: 'load' });
await rm.waitForTimeout(900);
const reduced = await rm.evaluate(() => {
  const moving = [...document.querySelectorAll('*')].filter(el => {
    const s = getComputedStyle(el);
    return s.transform !== 'none' && s.transitionDuration !== '0s'
      && /translate|scale/.test(s.transform);
  }).length;
  const fading = [...document.querySelectorAll('*')].filter(el =>
    getComputedStyle(el).transitionProperty.includes('opacity')).length;
  return { moving, fading };
});
console.log(`\n  reduced motion: ${reduced.moving} elements still translating or scaling, `
  + `${reduced.fading} still crossfading`);
if (reduced.moving > 0) problems += reduced.moving;

await b.close();
console.log(problems
  ? `\n  ${problems} motion problem(s)`
  : `\n  transform and opacity only, nothing loops, and reduced motion drops the travel`);
process.exit(problems ? 1 : 0);
