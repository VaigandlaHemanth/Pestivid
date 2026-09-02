// The measurable half of the seo-page checklist, run against what ships.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  await p.goto('http://localhost:3001/');
  await p.waitForTimeout(2200);
  console.log(await p.evaluate(() => {
    const q = (s) => document.querySelector(s);
    const heads = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
      .map(h => h.tagName + ' ' + h.textContent.trim().slice(0, 50));
    const imgs = [...document.querySelectorAll('img')].map(i =>
      (i.getAttribute('alt') === null ? 'NO ALT' : 'alt ok') + ' · '
      + (i.width && i.height ? 'sized' : 'UNSIZED') + ' · ' + (i.currentSrc || i.src).slice(-40));
    const vids = [...document.querySelectorAll('video')].map(v =>
      (v.getAttribute('aria-label') || v.getAttribute('title') ? 'labelled' : 'UNLABELLED')
      + ' · ' + (v.hasAttribute('muted') || v.muted ? 'muted' : 'NOT MUTED')
      + ' · ' + (v.getAttribute('poster') ? 'poster' : 'no poster'));
    const title = document.title;
    const desc = q('meta[name="description"]')?.content || '';
    return [
      'title            ' + title.length + ' chars' + (title.length >= 40 && title.length <= 62 ? ' ok' : ' OUT OF RANGE'),
      'description      ' + desc.length + ' chars' + (desc.length >= 120 && desc.length <= 165 ? ' ok' : ' OUT OF RANGE'),
      'html lang        ' + (document.documentElement.lang || 'MISSING'),
      'h1 count         ' + document.querySelectorAll('h1').length,
      'headings         ' + (heads.length ? heads.join(' | ') : 'NONE — the whole page is divs'),
      'images           ' + (imgs.length ? imgs.join('\n                 ') : 'none'),
      'videos           ' + (vids.length ? vids.join('\n                 ') : 'none'),
      'ld+json          ' + document.querySelectorAll('script[type="application/ld+json"]').length,
      'render-blocking  ' + [...document.querySelectorAll('link[rel=stylesheet]')].map(l => l.href.replace(/^https?:\/\//,'').slice(0,44)).join(', '),
      'deferred scripts ' + [...document.querySelectorAll('script[src]')].map(s => (s.defer||s.type==='module'?'ok':'BLOCKING') + ' ' + s.src.split('/').pop()).join(', '),
      // Absolute-URL signals. build-pages omits these unless PESTIVID_SITE is
      // set, because a canonical pointing at a URL that does not exist is worse
      // than no canonical -- so "not set" is the correct state until deploy, and
      // it is reported rather than quietly passed.
      'canonical        ' + (q('link[rel=canonical]')?.href
        || 'not set - build with PESTIVID_SITE=https://your.domain'),
      'og:image         ' + (q('meta[property="og:image"]')?.content
        ? q('meta[property="og:image"]').content + ' · '
          + (q('meta[property="og:image:width"]')?.content || 'NO WIDTH') + 'x'
          + (q('meta[property="og:image:height"]')?.content || 'NO HEIGHT')
          + (q('meta[property="og:image:alt"]') ? ' · alt ok' : ' · NO ALT')
        : 'not set - needs PESTIVID_SITE'),
      'twitter:card     ' + (() => {
        const card = q('meta[name="twitter:card"]')?.content || 'MISSING';
        const want = q('meta[property="og:image"]') ? 'summary_large_image' : 'summary';
        return card + (card === want ? ' ok' : ' SHOULD BE ' + want);
      })(),
      'ld+json types    ' + (() => {
        const el = q('script[type="application/ld+json"]');
        if (!el) return 'NONE';
        try {
          const d = JSON.parse(el.textContent);
          const g = d['@graph'] || [d];
          const bad = g.map(x => x['@type']).filter(t => /HowTo|FAQPage/.test(t));
          return g.map(x => x['@type']).join(', ')
            + (bad.length ? '  DEPRECATED: ' + bad.join(', ') : '  ok');
        } catch (e) { return 'DOES NOT PARSE: ' + e.message; }
      })(),
      // A <br> between two word characters is read as no space at all by a
      // crawler, a screen reader and innerText alike. The h1 shipped as
      // "A field you can see.A date nobody can move."
      'joined by <br>   ' + (() => {
        const bad = [...document.querySelectorAll('body *')]
          .filter(e => /[A-Za-z0-9,;:)\.]<br\s*\/?>[A-Za-z0-9(]/.test(e.innerHTML))
          .map(e => e.textContent.trim().slice(0, 40));
        return bad.length ? bad.length + ' FOUND: ' + bad[0] : 'none';
      })(),
    ].join('\n');
  }));
  await b.close();
})();
