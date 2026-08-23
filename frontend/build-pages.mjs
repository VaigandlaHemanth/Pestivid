// Cuts real pages out of the design artboards.
//
// The artboards in design/ stay the single source of truth for markup. Nothing
// here retypes a screen, so a page cannot quietly drift from the board it was
// signed off as -- and verify-pages.mjs renders both and diffs the pixels to
// prove it.
//
// Extraction runs inside Chromium rather than over the text, because several
// boards use <sc-for> and {{...}}. Reading outerHTML back after the canvas shim
// has run means the page ships exactly what the board renders, templates
// already resolved into the placeholder content that was reviewed.
//
//   node frontend/build-pages.mjs
import { chromium } from 'playwright';
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DESIGN = path.join(ROOT, 'design');
const OUT = path.join(ROOT, 'frontend', 'app');
mkdirSync(OUT, { recursive: true });
mkdirSync(path.join(OUT, 'pages'), { recursive: true });

export const PAGES = {
  'landing':            { title: 'Pestivid', role: 'public' },
  'signin':             { title: 'Sign in', role: 'public' },
  'signin-farmer':      { title: 'Sign in', role: 'public' },
  'setup-language':     { title: 'Choose your language', role: 'public' },
  'setup-identity':     { title: 'Who are you?', role: 'public' },
  'home-empty':         { title: 'Pestivid', role: 'farmer' },
  'home':               { title: 'Pestivid', role: 'farmer' },
  'record':             { title: 'Record a video', role: 'farmer' },
  'sent':               { title: 'That is saved', role: 'farmer' },
  'plots':              { title: 'My plots', role: 'farmer' },
  'plot':               { title: 'Plot', role: 'farmer' },
  'ask-money-video':    { title: 'Ask for money', role: 'farmer' },
  'ask-money-amount':   { title: 'How much do you need?', role: 'farmer' },
  'ask-money-terms':    { title: 'How do they get paid?', role: 'farmer' },
  'money':              { title: 'Money', role: 'farmer' },
  'payout':             { title: 'Check who gets what', role: 'farmer' },
  'messages':           { title: 'Messages', role: 'farmer' },
  'thread-farmer':      { title: 'Message', role: 'farmer' },
  'thread-investor':    { title: 'Message', role: 'investor' },
  'ask':                { title: 'Ask a question', role: 'farmer' },
  'profile':            { title: 'You and your settings', role: 'farmer' },
  'leaf-result':        { title: 'Leaf check', role: 'farmer' },
  'leaf-refusal':       { title: 'Leaf check', role: 'farmer' },
  'invest':             { title: 'Open for funding', role: 'investor' },
  'confirm-investment': { title: 'Confirm your investment', role: 'investor' },
  'report-harvest':     { title: 'Report the harvest', role: 'farmer' },
  'portfolio':          { title: 'Your seasons', role: 'investor' },
  'market':             { title: 'Lots for sale', role: 'buyer' },
  'orders':             { title: 'What you bought', role: 'buyer' },
  'admin':              { title: 'Flagged by the system', role: 'admin' },
};

if (import.meta.url === pathToFileURL(process.argv[1]).href) await build();

export async function build() {
  const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
  const boards = readdirSync(DESIGN).filter(f => f.endsWith('.dc.html'));
  const browser = await chromium.launch();
  const found = {};

  for (const file of boards) {
    const page = await browser.newPage({ viewport: { width: 1500, height: 1200 } });
    await page.goto(pathToFileURL(path.join(DESIGN, file)).href, { waitUntil: 'load' });
    await page.addScriptTag({ content: shim });
    await page.evaluate(() => document.fonts.ready);
    // the dashed fold line is a note to a designer, not part of the product
    await page.evaluate(() => document.querySelectorAll('[data-fold]').forEach(n => n.remove()));
    const got = await page.evaluate(() => {
      const css = [...document.querySelectorAll('style')]
        .filter(s => !s.dataset.injected).map(s => s.textContent).join('\n');
      const links = [...document.querySelectorAll('link[rel="stylesheet"]')].map(l => l.outerHTML).join('\n  ');
      const out = [];
      // A panel cut out of a board loses everything it was inheriting from the
      // board root -- most visibly the font stack, which turned one page into
      // Times. Carry the inherited values across explicitly.
      // font-size and line-height are deliberately absent: a computed
      // line-height comes back as a px number, and pinning that on the wrapper
      // makes every descendant inherit an absolute leading instead of `normal`.
      const INHERIT = ['font-family', 'font-weight', 'font-style', 'letter-spacing',
        'color', 'text-align', 'font-variant-numeric', 'font-feature-settings',
        '-webkit-font-smoothing'];
      for (const el of document.querySelectorAll('[data-page]')) {
        const r = el.getBoundingClientRect();
        const own = el.getAttribute('style') || '';
        const mine = getComputedStyle(el), up = el.parentElement ? getComputedStyle(el.parentElement) : null;
        const inherited = up ? INHERIT
          .filter(k => !new RegExp('(^|;)\s*' + k + '\s*:').test(own))
          .filter(k => mine.getPropertyValue(k) === up.getPropertyValue(k))
          // a font stack contains double quotes, which would close style="
          .map(k => `${k}: ${up.getPropertyValue(k).replace(/"/g, "'")}`).join('; ') : '';
        out.push({ slug: el.dataset.page, html: el.outerHTML, inherited,
                   w: Math.round(r.width), h: Math.round(r.height) });
      }
      return { css, links, out };
    });
    for (const p of got.out) found[p.slug] = { ...p, css: got.css, links: got.links, from: file };
    await page.close();
  }
  await browser.close();

  let wrote = 0;
  const missing = [], sizes = {};
  for (const [slug, meta] of Object.entries(PAGES)) {
    const f = found[slug];
    if (!f) { missing.push(slug); continue; }
    sizes[slug] = { w: f.w, h: f.h };

    // The board was drawn at a fixed width. A page has to fill a handset and
    // centre on anything wider; at the design width the two are identical,
    // which is what verify-pages.mjs measures.
    const open = f.html.slice(0, f.html.indexOf('>') + 1);
    let style = (open.match(/style="([^"]*)"/) || [, ''])[1];
    style = style.replace(/\bwidth:\s*\d+px;?/, '');
    style = style.replace(/\b(min-)?height:\s*\d+px;?/, '');
    // A desktop board drawn at 1280 or 1320 was being clamped to that width and
    // centred, which left a band of bare ground down both sides of the window.
    // Let it fill instead; at the design width the two are identical, which is
    // what verify-layout measures. A phone board keeps its width and centres,
    // because that is what a phone screen is.
    const clamp = f.w > 400 ? 'max-width: none;' : `max-width: ${f.w}px;`;
    style = `box-sizing: border-box; width: 100%; ${clamp} margin-left: auto; margin-right: auto; min-height: 100vh; `
      + (f.inherited ? f.inherited + '; ' : '') + style;
    let open2 = open.replace(/\sdata-page="[^"]*"/, '');
    open2 = /style="/.test(open2) ? open2.replace(/style="[^"]*"/, `style="${style}"`)
                                  : open2.replace(/^<div/, `<div style="${style}"`);
    const body = open2 + f.html.slice(open.length);

    writeFileSync(path.join(OUT, slug + '.html'), `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="${f.w <= 400 ? '#f6f3ef' : '#ffffff'}">
  <title>${meta.title}</title>
  <!-- generated from design/${f.from} by frontend/build-pages.mjs -- do not edit -->
  ${f.links}
  <link rel="stylesheet" href="./tokens.css">
  <style>
${f.css.trimEnd()}
  </style>
</head>
<body data-page="${slug}" data-role="${meta.role}" data-design-width="${f.w}" data-design-height="${f.h}">
${body}
<script type="module" src="./pages/${slug}.js"></script>
</body>
</html>
`);
    wrote++;
  }
  writeFileSync(path.join(OUT, 'sizes.json'), JSON.stringify(sizes, null, 1));
  console.log(`generated ${wrote} pages into frontend/app/`);
  if (missing.length) console.log(`  NOT FOUND in any artboard: ${missing.join(', ')}`);
  const extra = Object.keys(found).filter(s => !PAGES[s]);
  if (extra.length) console.log(`  marked but not listed: ${extra.join(', ')}`);
  return sizes;
}
