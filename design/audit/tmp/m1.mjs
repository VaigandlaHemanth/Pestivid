import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const DESIGN = 'C:/Users/ASUS/Desktop/pestivid orginal/p_pro/design';
const shim = readFileSync(path.join(DESIGN,'audit','shim.js'),'utf8');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1800 } });
await page.goto(pathToFileURL(path.join(DESIGN,'AskMoney.dc.html')).href, { waitUntil: 'load' });
await page.addScriptTag({ content: shim });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(300);
const r = await page.evaluate(() => {
  const out = [];
  for (const el of document.querySelectorAll('.cp, .cpOn')) {
    const b = el.getBoundingClientRect();
    out.push({ t: el.textContent.trim(), w: +b.width.toFixed(1), h: +b.height.toFixed(1) });
  }
  const panel = document.querySelector('[data-page="ask-money-amount"] div[style*="flex-grow: 1"]');
  const pb = panel.getBoundingClientRect();
  const cs = getComputedStyle(panel);
  const rows = [...document.querySelectorAll('[data-page="ask-money-amount"] div[style*="display: flex; gap: 8px; margin-top: 8px;"]')];
  const rowInfo = rows.map(x => { const b=x.getBoundingClientRect(); return { w:+b.width.toFixed(1), sw:x.scrollWidth, h:+b.height.toFixed(1) }; });
  const roi = document.querySelector('[data-bind="roi.value"]');
  const rb = roi.getBoundingClientRect();
  return { out, panelW: +pb.width.toFixed(1), pad: cs.padding, clientW: panel.clientWidth, rowInfo, roi: { w:+rb.width.toFixed(1), h:+rb.height.toFixed(1), fs: getComputedStyle(roi).fontSize } };
});
console.log(JSON.stringify(r, null, 1));
await browser.close();
