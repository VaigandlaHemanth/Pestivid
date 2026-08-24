// Screenshot a local HTML file. Used to render design proposals for review.
const { chromium } = require('playwright');
const { pathToFileURL } = require('url');

(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1700, height: 1200 }, deviceScaleFactor: 2 });
  await p.goto(pathToFileURL(process.argv[2]).href);
  await p.evaluate(() => document.fonts.ready);
  await p.waitForTimeout(1400);
  await p.screenshot({ path: process.argv[3], fullPage: true });
  console.log('shot ->', process.argv[3]);
  await b.close();
})();
