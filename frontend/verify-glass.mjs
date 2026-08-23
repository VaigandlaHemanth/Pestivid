// The worst case for a plate over a frame nobody chose is an overexposed one.
// Puts pure white behind every plate, samples the plate's real composited
// pixels, and reports the WCAG ratio against the text colour that sits on it.
import { chromium } from 'playwright';
const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const L = ([r,g,b]) => 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b);
const ratio = (a,b) => { const [x,y] = [L(a),L(b)].sort((p,q)=>q-p); return (x+0.05)/(y+0.05); };

const b = await chromium.launch();
const login = async (role) => (await fetch('http://127.0.0.1:3001/api/auth/login', {
  method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${role}@pestivid.sim`, password: 'password123' }) })).json();

for (const [slug, role] of [['leaf-result','farmer'], ['market','buyer'], ['invest','investor'], ['landing',null]]) {
  const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
  if (role) { const s = await login(role);
    await p.addInitScript(([t,u]) => { localStorage.setItem('pv.token',t); localStorage.setItem('pv.user',u); },
                          [s.token, JSON.stringify(s.user)]); }
  await p.goto(`http://127.0.0.1:3001/app/${slug}.html`, { waitUntil: 'load' });
  await p.waitForTimeout(1200);
  const rows = await p.evaluate(() => {
    const out = [];
    for (const g of document.querySelectorAll('.gplate')) {
      const host = g.offsetParent || g.parentElement;
      if (host) { host.style.background = '#fff'; host.style.backgroundImage = 'none'; }
      const t = g.querySelector('*') || g;
      out.push({ ink: getComputedStyle(t).color, w: Math.round(g.getBoundingClientRect().width) });
    }
    return out;
  });
  await p.waitForTimeout(400);
  const els = await p.$$('.gplate');
  for (let i = 0; i < els.length; i++) {
    const box = await els[i].boundingBox();
    if (!box || box.width < 8) continue;
    const buf = await p.screenshot({ clip: { x: box.x + box.width/2, y: box.y + 2, width: 4, height: 3 } });
    // decode the 4x3 PNG by re-reading it through the page
    const px = await p.evaluate(async (b64) => {
      const img = new Image(); img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      c.getContext('2d').drawImage(img, 0, 0);
      const d = c.getContext('2d').getImageData(0, 0, img.width, img.height).data;
      return [d[0], d[1], d[2]];
    }, buf.toString('base64'));
    const ink = rows[i].ink.match(/\d+/g).slice(0,3).map(Number);
    const r = ratio(px, ink);
    console.log(`  ${slug.padEnd(13)} plate ${i}  plate pixel rgb(${px}) vs ink rgb(${ink})  ${r.toFixed(2)}:1  ${r >= 4.5 ? 'AA' : 'FAILS AA'}`);
  }
  await p.close();
}
await b.close();
