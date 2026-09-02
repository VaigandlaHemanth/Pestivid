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
import { readFileSync, writeFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
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
  'signup':             { title: 'Create an account', role: 'public' },
  // 'setup' was the farmer's own account page: a language picker, a phone
  // number and a six-digit code standing in for a password. A phone login needs
  // an OTP this product has no way to send, and one role having its own front
  // door is exactly the split it was reported as. Everybody uses signup.
  // ONE KIND OF ACCOUNT. farmer, investor and buyer were three bars for one
  // person, so every signed-in screen is role 'any' except the reviewer's
  // queue. The word 'farmer' below is kept only where a page still means a
  // field's owner and nobody else -- there are none left; they are all 'any'.
  'home':               { title: 'Home', role: 'any' },
  'record':             { title: 'Record a video', role: 'any' },
  'sent':               { title: 'That is saved', role: 'any' },
  // 'plots' was a second list of the same eight videos home already shows, in the
  // same order, every row going to the same plot page. It printed each video's
  // length twice -- on the thumbnail and again in a "Length" column -- and left
  // out which plot the video was of, on a page called My plots. Home's table has
  // the plot name. Two pages cannot both be the list, so the nav word "My plots"
  // points at home and this one is gone. Plots.dc.html stays where it is, the way
  // Setup.dc.html did.
  'plot':               { title: 'Plot', role: 'any' },
  'ask-money':          { title: 'Ask for money', role: 'any' },
  'money':              { title: 'Money', role: 'any' },
  // Shared by all four roles, so not the farmer's. A buyer reaching a page
  // marked 'farmer' got the farmer's nav bar and thought they were in the wrong
  // app -- which they were.
  'messages':           { title: 'Messages', role: 'any' },
  'notifications':      { title: 'What has happened', role: 'any' },
  'ask':                { title: 'Ask a question', role: 'any' },
  'profile':            { title: 'You and your settings', role: 'any' },
  'leaf-check':        { title: 'Check a leaf', role: 'any' },
  'invest':             { title: 'Open for funding', role: 'any' },
  'confirm-investment': { title: 'Confirm your investment', role: 'any' },
  'report-harvest':     { title: 'Report the harvest', role: 'any' },
  'portfolio':          { title: 'Your seasons', role: 'any' },
  'market':             { title: 'Lots for sale', role: 'any' },
  'orders':             { title: 'What you bought', role: 'any' },
  'admin':              { title: 'Flagged by the system', role: 'admin' },
};

/* ── what a crawler and a link preview see ────────────────────────────────
 * Only the landing page is worth indexing: everything else is behind a token,
 * and an indexed sign-in shell is noise that competes with the page you want
 * found. So landing gets the full set and the other twenty-three get noindex.
 *
 * Absolute URLs need an origin, and this repo has no domain yet. Rather than
 * ship a guessed one -- a wrong canonical is worse than none, it points every
 * signal at a URL that does not exist -- canonical, og:url and the sitemap are
 * emitted only when PESTIVID_SITE is set at build time:
 *
 *     PESTIVID_SITE=https://pestivid.example node frontend/build-pages.mjs
 */
const SITE = (process.env.PESTIVID_SITE || '').replace(/\/+$/, '');

const SEO_TITLE = 'Pestivid — farm video evidence with a date nobody can move';
const SEO_DESC = 'Farmers film their crop. Pestivid fingerprints the file the moment it '
  + 'arrives and writes that fingerprint into Bitcoin, so the date cannot be changed later.';

function seoHead(slug) {
  const out = ['  <link rel="icon" href="./favicon.svg" type="image/svg+xml">'];
  if (slug !== 'landing') {
    // Behind a token, and nothing here is a landing page for anything.
    out.push('  <meta name="robots" content="noindex, follow">');
    return out.join('\n');
  }
  out.push(`  <meta name="description" content="${SEO_DESC}">`);
  out.push('  <meta name="robots" content="index, follow">');
  if (SITE) out.push(`  <link rel="canonical" href="${SITE}/">`);
  out.push('  <meta property="og:type" content="website">');
  out.push('  <meta property="og:site_name" content="Pestivid">');
  out.push(`  <meta property="og:title" content="${SEO_TITLE}">`);
  out.push(`  <meta property="og:description" content="${SEO_DESC}">`);
  if (SITE) {
    out.push(`  <meta property="og:url" content="${SITE}/">`);
    out.push(`  <meta property="og:image" content="${SITE}/app/og.png">`);
    out.push('  <meta property="og:image:width" content="1200">');
    out.push('  <meta property="og:image:height" content="630">');
    out.push('  <meta property="og:image:alt" content="Pestivid: a field you can see, a date nobody can move.">');
    out.push('  <meta name="twitter:card" content="summary_large_image">');
  } else {
    // A relative og:image is ignored by most scrapers, so claim nothing.
    out.push('  <meta name="twitter:card" content="summary">');
  }
  out.push(`  <meta name="twitter:title" content="${SEO_TITLE}">`);
  out.push(`  <meta name="twitter:description" content="${SEO_DESC}">`);
  // Organization and WebSite only. No HowTo (deprecated) and no FAQPage (rich
  // results retired), and nothing this product cannot substantiate: no ratings,
  // no address, no founder, no trial results.
  const graph = [
    { '@type': 'Organization', name: 'Pestivid', description: SEO_DESC,
      ...(SITE ? { url: SITE + '/', logo: SITE + '/app/favicon.svg' } : {}) },
    { '@type': 'WebSite', name: 'Pestivid', inLanguage: 'en-IN',
      ...(SITE ? { url: SITE + '/' } : {}) },
  ];
  out.push('  <script type="application/ld+json">'
    + JSON.stringify({ '@context': 'https://schema.org', '@graph': graph })
    + '</script>');
  return out.join('\n');
}

function writeCrawlerFiles(slugs) {
  const lines = ['User-agent: *', 'Allow: /', '',
    '# Everything under /app/ except the landing page carries a noindex meta tag:',
    '# they need a token to say anything, and CSS and JS have to stay crawlable',
    '# for the landing page to render, so nothing is Disallowed here.'];
  if (SITE) lines.push('', `Sitemap: ${SITE}/sitemap.xml`);
  writeFileSync(path.join(OUT, '..', 'robots.txt'), lines.join('\n') + '\n');

  if (!SITE) {
    // A sitemap from an earlier build still names that earlier origin, so it
    // goes rather than lingering as a file full of URLs nobody serves.
    rmSync(path.join(OUT, '..', 'sitemap.xml'), { force: true });
    return 'robots.txt (no sitemap: PESTIVID_SITE is not set)';
  }
  const urls = [`${SITE}/`];
  writeFileSync(path.join(OUT, '..', 'sitemap.xml'),
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    + '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
    + urls.map(u => `  <url><loc>${u}</loc></url>`).join('\n')
    + '\n</urlset>\n');
  return `robots.txt and sitemap.xml (${urls.length} url)`;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) await build();

export async function build() {
  const shim = readFileSync(path.join(DESIGN, 'audit', 'shim.js'), 'utf8');
  // Which board a page comes from must not depend on directory order. A copy of
  // a board left in design/ declares the same data-page slugs, and whichever
  // the listing happened to reach first became that page's origin -- so a
  // throwaway file silently decided what a real page was generated from, and
  // the generated page then pointed at a board that no longer existed.
  //
  // canvas.json is the authoritative set of artboards, so it goes first and in
  // its own order. Anything else in design/ follows, and files starting with an
  // underscore are scratch and never considered.
  const onCanvas = JSON.parse(readFileSync(path.join(DESIGN, 'canvas.json'), 'utf8'))
    .artboards.map(a => a.file);
  const present = new Set(readdirSync(DESIGN).filter(f => f.endsWith('.dc.html') && !f.startsWith('_')));
  const boards = [...onCanvas.filter(f => present.has(f)),
                  ...[...present].filter(f => !onCanvas.includes(f)).sort()];
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
    // The root says which kind of surface it is, so responsive.css can adapt a
    // desktop layout without also rewriting a phone one that is already narrow.
    let open2 = open.replace(/\sdata-page="[^"]*"/, ` data-surface="${f.w > 400 ? 'desk' : 'phone'}" data-w="${f.w}"`);
    open2 = /style="/.test(open2) ? open2.replace(/style="[^"]*"/, `style="${style}"`)
                                  : open2.replace(/^<div/, `<div style="${style}"`);
    const body = open2 + f.html.slice(open.length);

    writeFileSync(path.join(OUT, slug + '.html'), `<!doctype html>
<html lang="en-IN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="${f.w <= 400 ? '#f6f3ef' : '#ffffff'}">
  <title>${slug === 'landing' ? SEO_TITLE : meta.title + ' · Pestivid'}</title>
  <!-- generated from design/${f.from} by frontend/build-pages.mjs -- do not edit -->
${seoHead(slug)}
  ${f.links}
  <link rel="stylesheet" href="./tokens.css">
  <!-- After tokens, because it adapts what the boards pin. Nothing in it fires
       at a design's own width. -->
  <link rel="stylesheet" href="./loaders.css">
  <link rel="stylesheet" href="./responsive.css">
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
  console.log(`  ${writeCrawlerFiles(Object.keys(PAGES))}`);
  if (missing.length) console.log(`  NOT FOUND in any artboard: ${missing.join(', ')}`);
  const extra = Object.keys(found).filter(s => !PAGES[s]);
  if (extra.length) console.log(`  marked but not listed: ${extra.join(', ')}`);
  return sizes;
}
