// Which pages actually move, and what moves on them.
//
// verify-motion.mjs checks that motion obeys the rules -- transform and opacity
// only, nothing loops, reduced motion drops the travel. It has nothing to say
// about a page with no motion at all, so "passes" and "is completely static"
// read identically in its output. This asks the other question.
//
// Reports per page: the count of elements carrying a transition or animation,
// what properties they animate, and whether the page has an ENTRANCE (something
// that arrives) as distinct from only press feedback.
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

const SURVEY = () => {
  const props = new Set();
  let moving = 0, entrance = 0, scrollDriven = 0, pressOnly = 0;
  const samples = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    const tp = cs.transitionProperty;
    const dur = cs.transitionDuration;
    const anim = cs.animationName;
    const timeline = cs.animationTimeline;
    const hasTransition = tp && tp !== 'none' && dur && dur !== '0s';
    const hasAnim = anim && anim !== 'none';
    if (!hasTransition && !hasAnim) continue;
    moving++;
    for (const p of tp.split(',').map(x => x.trim())) if (p && p !== 'none') props.add(p);
    if (hasAnim) props.add('@' + anim.split(',')[0].trim());
    if (timeline && timeline !== 'auto' && timeline !== 'none') scrollDriven++;
    // An entrance is a ONE SHOT, so by the time a page is settled there is
    // nothing left to observe in the computed transform -- which is why this
    // reported "press only" on pages that had just staged every row. What
    // survives is the inline transition arrive() writes, and a fill-mode
    // animation that has run. Look for those instead of for a live displacement.
    const inlineT = el.style.transition || '';
    const staged = /opacity/.test(inlineT) && /transform/.test(inlineT);
    const filled = hasAnim && /both|forwards/.test(cs.animationFillMode);
    if (staged || filled) entrance++;
    if (props.size && [...props].every(p => /transform|opacity/.test(p))
        && el.matches('[data-act], [data-go]')) pressOnly++;
    if (samples.length < 3) {
      samples.push((el.getAttribute('data-bind') || el.className || el.tagName.toLowerCase())
        .toString().slice(0, 24) + ':' + (hasAnim ? anim.split(',')[0].trim() : tp.split(',')[0].trim()));
    }
  }
  return { moving, entrance, scrollDriven, props: [...props].sort(), samples };
};

/* Five pages have nothing that ARRIVES on load -- they are a form, a settings
 * list, an empty chat, a capture screen, a review queue drawn at parse time.
 * Motion on those at rest would be decoration, which the gate rejects. What they
 * do have is a moment where state changes, and that is where they animate. So
 * the check for these is: do the interaction, then look again.
 */
const ON_INTERACTION = {
  profile: { do: 'text=Bigger text', why: 'toggling a setting posts a notice' },
  admin: { do: 'text=Leave it open', why: 'a decision posts a notice' },
  ask: { do: null, why: 'a message animates in from the composer; measured in _askrun' },
  'leaf-check': { do: null, why: 'the verdict card arrives on the bouncy spring' },
  'report-harvest': { do: null, why: 'the send gate eases as it enables; see verify-gate' },
};

const slugs = process.argv.slice(2).length
  ? process.argv.slice(2)
  : readdirSync('frontend/app').filter(f => f.endsWith('.html')).map(f => f.replace('.html', '')).sort();

const b = await chromium.launch();
const tok = {};
let bare = 0, noEntrance = 0;
console.log('  page                 moving  entrance  scroll   properties');
for (const slug of slugs) {
  const role = slug in ROLE ? ROLE[slug] : 'farmer';
  if (role && !tok[role]) tok[role] = await login(role);
  const p = await b.newPage({ viewport: { width: 1440, height: 940 } });
  if (role) await p.addInitScript(([t, u]) => {
    localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
  }, [tok[role].token, JSON.stringify(tok[role].user)]);
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html${QUERY[slug] || ''}`, { waitUntil: 'load' });
  await p.waitForTimeout(1500);
  let r = await p.evaluate(SURVEY);
  // Nothing arrived on load. If this page is meant to animate on an
  // interaction, do the interaction and look again.
  let via = '';
  if (!r.entrance && !r.scrollDriven && ON_INTERACTION[slug]) {
    const spec = ON_INTERACTION[slug];
    if (spec.do) {
      try {
        await p.locator(spec.do).first().click({ timeout: 4000 });
        await p.waitForTimeout(120);
        const after = await p.evaluate(SURVEY);
        if (after.entrance > r.entrance) { r = after; via = '  on interaction — ' + spec.why; }
      } catch { via = '  COULD NOT INTERACT: ' + spec.do; }
    } else {
      via = '  static at rest — ' + spec.why;
    }
  }
  await p.close();
  if (!r.moving) bare++;
  else if (!r.entrance && !r.scrollDriven && !via) noEntrance++;
  const flag = !r.moving ? '  NOTHING MOVES'
    : via ? via
    : (!r.entrance && !r.scrollDriven ? '  PRESS FEEDBACK ONLY' : '');
  console.log('  ' + slug.padEnd(20)
    + String(r.moving).padStart(6) + String(r.entrance).padStart(10)
    + String(r.scrollDriven).padStart(8) + '   ' + r.props.join(' ').slice(0, 46) + flag);
}
await b.close();
console.log(`\n  ${bare} page(s) with no motion at all, ${noEntrance} with press feedback only,`
  + ` of ${slugs.length}`);
