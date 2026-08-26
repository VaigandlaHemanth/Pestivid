// The mechanical half of the taste-skill pre-flight, run against what ships.
// Only the checks that generalise past a landing page: the Section 9 AI tells,
// the colour and shape locks, and the CTA rules.
const { chromium } = require('playwright');
const { readdirSync } = require('fs');
const TOKENS = new Set(['rgb(221, 215, 209)','rgb(246, 243, 239)','rgb(29, 26, 23)',
  'rgb(0, 105, 52)','rgb(1, 106, 190)','rgb(1, 87, 155)','rgb(167, 25, 48)',
  'rgb(242, 230, 205)','rgb(124, 74, 18)','rgb(55, 50, 45)','rgb(255, 255, 255)',
  'rgba(0, 0, 0, 0)','rgb(74, 68, 61)','rgb(96, 90, 83)','rgb(195, 188, 182)',
  'rgb(231, 225, 219)','rgb(234, 228, 222)','rgb(225, 230, 236)','rgb(220, 214, 201)',
  'rgb(1, 33, 105)','rgb(247, 233, 230)','rgb(14, 13, 11)','rgb(42, 38, 34)','rgb(0, 0, 0)']);

const AUDIT = (TOKEN_LIST) => {
  const TOKENS = new Set(TOKEN_LIST);
  const text = document.body.innerText;
  const out = { dashes: [], dots: [], scrollCue: [], numbered: [], wraps: [], radii: {}, offPalette: {} };
  // 9.G em-dash: zero
  for (const m of text.matchAll(/[^\n]{0,28}[—–][^\n]{0,28}/g)) out.dashes.push(m[0].trim());
  // 9.F middle dot: max one per line
  for (const line of text.split('\n')) {
    const n = (line.match(/·/g) || []).length;
    if (n > 1) out.dots.push(n + '× ' + line.trim().slice(0, 58));
  }
  // 9.F scroll cues and section-number eyebrows
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (/^(scroll|↓\s*scroll|scroll to)/i.test(t)) out.scrollCue.push(t.slice(0, 40));
    if (/^\d{2,3}\s*[\/·-]\s*\w/.test(t)) out.numbered.push(t.slice(0, 40));
  }
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    // 4.4 shape lock
    const rad = cs.borderTopLeftRadius;
    if (rad && rad !== '0px') out.radii[rad] = (out.radii[rad] || 0) + 1;
    // 4.2 colour lock: anything painted outside the token set
    for (const c of [cs.backgroundColor, cs.color]) {
      if (c && !TOKENS.has(c)) out.offPalette[c] = (out.offPalette[c] || 0) + 1;
    }
    // 4.5 CTA wrap: a button label on more than one line
    if (el.matches('[data-act], [data-go], button') && el.children.length === 0) {
      const t = (el.textContent || '').trim();
      const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
      if (t && t.length < 40 && r.height > lh * 1.7) out.wraps.push(t.slice(0, 34) + ' (' + Math.round(r.height) + 'px)');
    }
  }
  return out;
};

(async () => {
  const b = await chromium.launch();
  const tk = {};
  const get = async (r) => tk[r] ||= await (await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
    headers: {'content-type':'application/json'},
    body: JSON.stringify({email:`demo.${r}@pestivid.sim`,password:'password123'})})).json();
  const ROLE = { landing:null, signin:null, signup:null, setup:null, invest:'investor',
    portfolio:'investor', 'confirm-investment':'investor', thread:'farmer', market:'buyer',
    orders:'buyer', admin:'admin' };
  const NEEDS = require('child_process');
  const slugs = process.argv.slice(2).length ? process.argv.slice(2)
    : readdirSync('app').filter(f => f.endsWith('.html')).map(f => f.replace('.html','')).sort();
  const tally = { dashes:0, dots:0, scrollCue:0, numbered:0, wraps:0 };
  const radii = {}, off = {};
  for (const slug of slugs) {
    const role = slug in ROLE ? ROLE[slug] : 'farmer';
    const t = role ? await get(role) : null;
    const p = await b.newPage({ viewport: { width: 1440, height: 2000 } });
    if (t) await p.addInitScript(([a,u]) => { localStorage.setItem('pv.token',a); localStorage.setItem('pv.user',u); },
                                 [t.token, JSON.stringify(t.user)]);
    await p.goto('http://127.0.0.1:3001/app/' + slug + '.html');
    await p.waitForTimeout(1400);
    const r = await p.evaluate(AUDIT, [...TOKENS]);
    await p.close();
    for (const k of Object.keys(tally)) tally[k] += r[k].length;
    for (const [k,v] of Object.entries(r.radii)) radii[k] = (radii[k]||0) + v;
    for (const [k,v] of Object.entries(r.offPalette)) off[k] = (off[k]||0) + v;
    const lines = [];
    for (const d of r.dashes.slice(0,2)) lines.push('  em-dash      ' + d);
    for (const d of r.dots.slice(0,2)) lines.push('  middle dots  ' + d);
    for (const d of r.scrollCue) lines.push('  scroll cue   ' + d);
    for (const d of r.numbered) lines.push('  numbered     ' + d);
    for (const d of r.wraps) lines.push('  CTA wraps    ' + d);
    if (lines.length) console.log('## ' + slug + '\n' + lines.join('\n'));
  }
  await b.close();
  console.log('\n  radii in use: ' + Object.entries(radii).sort((a,c)=>c[1]-a[1]).map(([k,v])=>k+'×'+v).join(' '));
  console.log('  off-palette colours: ' + (Object.keys(off).length
    ? Object.entries(off).sort((a,c)=>c[1]-a[1]).slice(0,6).map(([k,v])=>k+'×'+v).join('  ') : 'none'));
  console.log(`\n  ${tally.dashes} em-dash, ${tally.dots} multi-dot lines, ${tally.scrollCue} scroll cues,`
    + ` ${tally.numbered} numbered eyebrows, ${tally.wraps} wrapped CTAs`);
})();
