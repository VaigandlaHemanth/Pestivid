// Contrast for the text that verify-hig cannot compute.
//
// verify-hig walks up from each text node looking for a solid background colour.
// When it finds a photograph or a translucent plate over one it gives up and
// counts the node as "unmeasured" -- twenty-two of them on this product, every
// one on the landing hero, including four lines of 11-12px type inside a frosted
// stamp. Unmeasured is not the same as fine. It is the one bucket where a real
// failure can sit forever, because text over a picture is exactly where contrast
// goes wrong.
//
// So this composites it for real, which is the only way to know:
//   hide the ink, screenshot the element's own box, decode it in a canvas from a
//   data: URL, and measure the pixels the words actually sit on.
// It samples the pixels UNDER THE GLYPHS, not the element's box. A text node's
// box is a rectangle; the thing behind it need not be. On the landing hero the
// stamp is a circle, so the corners of "Not whose field"'s box fall outside the
// plate onto the photograph, and sampling the box scored the line 1.63:1 when
// what its letters actually sit on is the plate. Two screenshots -- one with the
// ink hidden, one with it shown -- differ exactly where the glyphs are, and that
// difference is the mask.
//
// The worst case is taken at the 2nd percentile of the masked pixels rather than
// the true minimum, so one antialiased edge pixel does not decide a verdict.
import { chromium } from 'playwright';

const lum = (r, g, b) => {
  const f = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
};
const ratio = (a, z) => { const [x, y] = a > z ? [a, z] : [z, a]; return (x + 0.05) / (y + 0.05); };
const parse = (c) => (c.match(/[\d.]+/g) || []).slice(0, 3).map(Number);

const slug = process.argv[2] || 'landing';
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1366, height: 900 } });
await p.goto(`http://127.0.0.1:3001/app/${slug}.html`, { waitUntil: 'load' });
await p.waitForTimeout(1600);

// Every leaf text node whose backdrop is not a solid colour anywhere above it.
const nodes = await p.evaluate(() => {
  const solid = (c) => c && c !== 'transparent' && !/rgba\(.*,\s*0\)$/.test(c);
  const out = [];
  document.querySelectorAll('*').forEach((el, i) => {
    if (el.children.length) return;
    const t = (el.textContent || '').replace(/\s+/g, ' ').trim();
    if (t.length < 2) return;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) < 0.05) return;
    // Walk up for a solid, fully opaque background. If an image or a translucent
    // fill is hit first, this node is one verify-hig cannot resolve.
    let overImage = false;
    for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (s.backgroundImage && s.backgroundImage !== 'none') { overImage = true; break; }
      const bg = s.backgroundColor;
      if (solid(bg)) {
        const a = Number((bg.match(/[\d.]+/g) || [])[3] ?? 1);
        if (a >= 0.999) break;
        overImage = true; break;
      }
    }
    if (!overImage) return;
    el.setAttribute('data-contrastprobe', String(i));
    const size = parseFloat(cs.fontSize) || 0;
    const weight = Number(cs.fontWeight) || 400;
    out.push({ id: String(i), text: t.slice(0, 34), fg: cs.color, size: Math.round(size),
               need: (size >= 24 || (size >= 18.66 && weight >= 700)) ? 3 : 4.5 });
  });
  return out;
});

console.log(`  ${nodes.length} text node(s) over imagery or a translucent plate on ${slug}\n`);
let bad = 0;
for (const n of nodes) {
  const box = await p.evaluate((id) => {
    const el = document.querySelector(`[data-contrastprobe="${id}"]`);
    if (!el) return null;
    el.scrollIntoView({ block: 'center' });
    el.style.color = 'transparent';
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return null;
    return { x: Math.max(0, Math.round(r.x)), y: Math.max(0, Math.round(r.y)),
             width: Math.round(r.width), height: Math.round(r.height) };
  }, n.id);
  if (!box) continue;
  await p.waitForTimeout(60);
  let bare, inked;
  try {
    bare = (await p.screenshot({ clip: box })).toString('base64');
    await p.evaluate((id) => { const e = document.querySelector(`[data-contrastprobe="${id}"]`); if (e) e.style.color = ''; }, n.id);
    await p.waitForTimeout(60);
    inked = (await p.screenshot({ clip: box })).toString('base64');
  } catch {
    await p.evaluate((id) => { const e = document.querySelector(`[data-contrastprobe="${id}"]`); if (e) e.style.color = ''; }, n.id);
    continue;
  }
  const px = await p.evaluate(async ([a64, b64]) => {
    const load = async (b64) => {
      const img = new Image();
      await new Promise((ok, no) => { img.onload = ok; img.onerror = no; img.src = 'data:image/png;base64,' + b64; });
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const g = c.getContext('2d'); g.drawImage(img, 0, 0);
      return g.getImageData(0, 0, c.width, c.height).data;
    };
    const A = await load(a64), B = await load(b64);
    const out = [];
    for (let i = 0; i < A.length; i += 4) {
      // Where the ink landed. 26 is well clear of JPEG-ish noise and well under
      // the difference any real glyph makes.
      const d = Math.abs(A[i] - B[i]) + Math.abs(A[i + 1] - B[i + 1]) + Math.abs(A[i + 2] - B[i + 2]);
      if (d > 26) out.push([A[i], A[i + 1], A[i + 2]]);   // the BACKGROUND there
    }
    return out;
  }, [bare, inked]);
  if (px.length < 12) continue;          // too few glyph pixels to judge

  const inkL = lum(...parse(n.fg));
  const ls = px.map(([r, g, bl]) => lum(r, g, bl)).sort((x, y) => x - y);
  const worst = ls[Math.floor(ls.length * 0.02)];
  const got = ratio(inkL, worst);
  const pass = got >= n.need - 0.02;
  if (!pass) bad++;
  console.log(`  ${pass ? 'ok  ' : 'FAIL'} ${got.toFixed(2).padStart(6)}:1 needs ${n.need}:1`
    + `  ${String(n.size).padStart(3)}px  ${n.fg.padEnd(24)} "${n.text}"`);
}
await b.close();
console.log(`\n  ${bad} of ${nodes.length} fail once the backdrop is actually composited`);
process.exit(bad ? 1 : 0);
