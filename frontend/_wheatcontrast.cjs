// The harvest button, at rest and with the pointer on it, on both screens that
// carry the card. Shot at 3x so the wheat can actually be judged.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
let failed = 0;
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1366, height: 900 }, deviceScaleFactor: 3 });
  await p.goto('http://localhost:3001/app/signin.html');
  await p.fill('input[name="who"]', 'demo.farmer@pestivid.sim');
  await p.fill('input[type="password"]', 'password123');
  await p.keyboard.press('Enter');
  await p.waitForURL(u => !/signin/.test(u.href), { timeout: 20000 });
  for (const slug of ['money', 'home']) {
    await p.goto(`http://localhost:3001/app/${slug}.html`);
    await p.waitForTimeout(1700);
    const card = await p.$('[data-todoband]');
    if (!card) { console.log(slug, '-- no harvest card'); continue; }
    await card.scrollIntoViewIfNeeded();
    await p.waitForTimeout(250);
    const h = await p.evaluateHandle(() =>
      document.querySelector('[data-todoband] [data-act], [data-todoband] [data-go]'));
    const btn = h.asElement();
    if (!btn) { console.log(slug, '-- no button in the card'); continue; }

    await btn.screenshot({ path: `${OUT}/wheat-${slug}-rest.png` });
    await card.screenshot({ path: `${OUT}/card-${slug}-rest.png` });
    await btn.hover();
    await p.waitForTimeout(600);
    await btn.screenshot({ path: `${OUT}/wheat-${slug}-hover.png` });
    await card.screenshot({ path: `${OUT}/card-${slug}-hover.png` });

    // and prove the label is never sitting on the wheat: sample the mask at the
    // centre of the button, where the text is
    const m = await p.evaluate(() => {
      const el = document.querySelector('[data-todoband] [data-act], [data-todoband] [data-go]');
      const cs = getComputedStyle(el, '::after');
      const r = el.getBoundingClientRect();
      return { w: Math.round(r.width), h: Math.round(r.height),
               band: cs.height, img: cs.backgroundImage.slice(0, 34),
               mask: (cs.maskImage || cs.webkitMaskImage || 'none').slice(0, 40) };
    });
    console.log(`  ${slug.padEnd(6)} button ${m.w}x${m.h}  band ${m.band}`);

    // The real question is not the mask value, it is whether white text still
    // clears 4.5:1 over whatever the wheat does to the blue underneath it. So
    // sample the rendered pixels along the label's own band and take the worst.
    // The label is hidden for the shot. Sampling with it visible measured its
    // own anti-aliased edges -- rgb(200,219,235) is the white of a glyph
    // feathering into blue, not the background -- and every such pixel looks
    // like a contrast failure. With the text gone, every pixel in that band IS
    // the background it would have sat on.
    await p.evaluate(() => {
      const el = document.querySelector('[data-todoband] [data-act], [data-todoband] [data-go]');
      for (const kid of el.children) kid.style.visibility = 'hidden';
    });
    await p.waitForTimeout(120);
    const shot = await btn.screenshot();
    await p.evaluate(() => {
      const el = document.querySelector('[data-todoband] [data-act], [data-todoband] [data-go]');
      for (const kid of el.children) kid.style.visibility = '';
    });
    // Only the LABEL's own box matters. The first measurement sampled the whole
    // button and reported 3.48:1 at x=12 -- inside the 26px padding, where there
    // is no text to fail. What the wheat does out there is decoration; what it
    // does behind a glyph is contrast.
    const zone = await p.evaluate(() => {
      const el = document.querySelector('[data-todoband] [data-act], [data-todoband] [data-go]');
      const b = el.getBoundingClientRect();
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const kid of el.children) {
        const r = kid.getBoundingClientRect();
        x0 = Math.min(x0, r.left - b.left); x1 = Math.max(x1, r.right - b.left);
        y0 = Math.min(y0, r.top - b.top);   y1 = Math.max(y1, r.bottom - b.top);
      }
      return { x0, x1, y0, y1, dpr: devicePixelRatio };
    });
    const worst = await p.evaluate(async ([b64, z]) => {
      const img = new Image();
      img.src = 'data:image/png;base64,' + b64;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      const lum = ({ r, gr, bl }) => {
        const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
        return 0.2126 * f(r) + 0.7152 * f(gr) + 0.0722 * f(bl);
      };
      // exactly the label's box, in device pixels
      const s = img.width / (z.x1 + (z.x0));      // unused guard, see below
      const k = z.dpr;
      const y0 = Math.max(0, Math.round(z.y0 * k)), y1 = Math.min(img.height - 1, Math.round(z.y1 * k));
      const xa = Math.max(0, Math.round(z.x0 * k)), xb = Math.min(img.width - 1, Math.round(z.x1 * k));
      let lo = Infinity, at = null;
      for (let y = y0; y <= y1; y += 1) {
        for (let x = xa; x <= xb; x += 1) {
          const d = g.getImageData(x, y, 1, 1).data;
          if (d[3] < 250) continue;                    // the rounded corners
          const L = lum({ r: d[0], gr: d[1], bl: d[2] });
          const ratio = 1.05 / (L + 0.05);
          if (ratio < lo) { lo = ratio; at = `${x},${y} rgb(${d[0]},${d[1]},${d[2]})`; }
        }
      }
      return { ratio: Number.isFinite(lo) ? Math.round(lo * 100) / 100 : null, at };
    }, [shot.toString('base64'), zone]);
    console.log(`         label box ${Math.round(zone.x0)}..${Math.round(zone.x1)}px`
      + `  white on the lightest pixel behind it: ${worst.ratio}:1  (${worst.at})`
      + `  ${worst.ratio >= 4.5 ? 'passes AA' : 'FAILS AA'}`);
    // 4.5:1, not 3:1: the label is 16.5px bold and WCAG's large-text allowance
    // starts at 18.66px bold, so this does not qualify for the lower bar.
    if (!(worst.ratio >= 4.5)) failed++;
  }
  await b.close();
  console.log('');
  console.log(failed
    ? `  ${failed} button(s) whose decoration breaks the label's contrast`
    : '  the wheat never takes the label below 4.5:1');
  process.exit(failed ? 1 : 0);
})();
