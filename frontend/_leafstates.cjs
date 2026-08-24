// The two screens a farmer actually reaches after the checker runs. Seeded
// rather than downloaded, because the layout is what is being judged here.
const { chromium } = require('playwright');
const OUT = process.env.SHOTDIR || '.';
const CASES = [
  ['verdict', { status: 'ok', disease: 'Fungi', confidence: 0.83, runner_up: 'Bacteria',
                ms: 412, file: 'IMG_20260824_0914.jpg' }],
  ['refusal', { status: 'not_a_leaf', confidence: 0.21, ms: 380, file: 'IMG_20260824_0930.jpg',
                message: 'The whole leaf has to be in the frame, with a little space around it.' }],
];
(async () => {
  const b = await chromium.launch();
  const tok = await (await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
    headers: {'content-type':'application/json'},
    body: JSON.stringify({email:'demo.farmer@pestivid.sim',password:'password123'})})).json();
  for (const [name, v] of CASES) {
    const p = await b.newPage({ viewport: { width: 1440, height: 940 }, deviceScaleFactor: 2 });
    const errs = [];
    p.on('pageerror', e => errs.push(e.message.slice(0, 160)));
    await p.addInitScript(([t,u,verdict]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
      sessionStorage.setItem('pv.leaf', verdict);
    }, [tok.token, JSON.stringify(tok.user), JSON.stringify(v)]);
    await p.goto('http://localhost:3001/app/leaf-check.html');
    await p.waitForTimeout(1600);
    const seen = await p.evaluate(() => {
      const vis = s => [...document.querySelectorAll(s)].some(e => e.offsetParent !== null || getComputedStyle(e).display !== 'none');
      return { capture: vis('[data-capture]'), verdict: vis('[data-verdictcard]'),
               treatment: vis('[data-treatment]'), retake: vis('[data-retake]'),
               ask: vis('[data-askcard]'), framing: vis('[data-framing]'),
               name: document.querySelector('[data-bind="verdict.name"]')?.textContent.trim(),
               mancozeb: document.body.innerText.includes('Mancozeb') };
    });
    console.log(name.padEnd(9), JSON.stringify(seen));
    if (errs.length) console.log('  ERRORS', errs.join(' | '));
    await p.screenshot({ path: `${OUT}/leaf-${name}.png`, fullPage: true });
    await p.close();
  }
  await b.close();
})();
