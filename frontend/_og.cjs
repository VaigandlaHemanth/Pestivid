// The share card, rendered rather than described: og:image has to be a raster
// file that actually exists, or a scraper shows a blank rectangle.
const { chromium } = require('playwright');
const { pathToFileURL } = require('url');

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1200, height: 630 } });
  await p.goto(pathToFileURL(process.argv[2]).href);
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(1200);
  await p.screenshot({ path: 'frontend/app/og.png' });
  await b.close();
  console.log('frontend/app/og.png written');
})();
