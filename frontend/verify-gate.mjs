// Proves the acknowledgement gate: measures the button before and after the tick,
// and checks the transition is eased rather than a hard cut.
import { chromium } from 'playwright';
const lin = c => { c/=255; return c<=0.04045 ? c/12.92 : ((c+0.055)/1.055)**2.4; };
const L = ([r,g,b]) => 0.2126*lin(r)+0.7152*lin(g)+0.0722*lin(b);
const ratio = (a,b) => { const [x,y]=[L(a),L(b)].sort((p,q)=>q-p); return (x+0.05)/(y+0.05); };
const rgb = s => s.match(/\d+/g).slice(0,3).map(Number);

const b = await chromium.launch();
const s = await (await fetch('http://127.0.0.1:3001/api/auth/login', { method:'POST',
  headers:{'content-type':'application/json'},
  body: JSON.stringify({ email:'demo.investor@pestivid.sim', password:'password123' }) })).json();
const projects = await (await fetch('http://127.0.0.1:3001/api/funding-requests')).json();
const id = projects[0]._id || projects[0].id;

const p = await b.newPage({ viewport:{ width:1440, height:900 }, deviceScaleFactor:2 });
await p.addInitScript(([t,u])=>{localStorage.setItem('pv.token',t);localStorage.setItem('pv.user',u);},
                      [s.token, JSON.stringify(s.user)]);
await p.goto(`http://127.0.0.1:3001/app/confirm-investment.html?project=${id}&amount=50000`);
await p.waitForTimeout(1500);

const read = () => p.evaluate(() => {
  const send = document.querySelector('[data-send]');
  const label = send?.querySelector('div');
  const box = document.querySelector('[data-box]');
  const tick = box?.querySelector('svg');
  const row = document.querySelector('[data-ack]');
  const r = row?.getBoundingClientRect();
  return {
    bg: getComputedStyle(send).backgroundColor,
    ink: getComputedStyle(label).color,
    trans: getComputedStyle(send).transitionProperty + ' ' + getComputedStyle(send).transitionDuration,
    boxBg: getComputedStyle(box).backgroundColor,
    tickOpacity: getComputedStyle(tick).opacity,
    disabled: send.getAttribute('aria-disabled'),
    checked: row?.getAttribute('aria-checked'),
    rowH: Math.round(r?.height || 0), rowW: Math.round(r?.width || 0),
    boxW: Math.round(box.getBoundingClientRect().width),
  };
});

const before = await read();
console.log(`  before   disabled=${before.disabled}  button ${before.bg} / ${before.ink}` +
            `  ${ratio(rgb(before.bg), rgb(before.ink)).toFixed(2)}:1` +
            `  ${ratio(rgb(before.bg), rgb(before.ink)) >= 4.5 ? 'AA' : 'below AA (a disabled control, exempt)'}`);
console.log(`  target   ack row ${before.rowW}x${before.rowH}  box ${before.boxW}px  tick opacity ${before.tickOpacity}`);
console.log(`  eased    ${before.trans}`);

await p.click('[data-ack]');
await p.waitForTimeout(60);
const mid = await read();
await p.waitForTimeout(500);
const after = await read();
console.log(`  after    disabled=${after.disabled} checked=${after.checked}  button ${after.bg} / ${after.ink}` +
            `  ${ratio(rgb(after.bg), rgb(after.ink)).toFixed(2)}:1` +
            `  ${ratio(rgb(after.bg), rgb(after.ink)) >= 4.5 ? 'AA' : 'FAILS'}`);
console.log(`  mid-flight button was ${mid.bg} -- ${mid.bg !== before.bg && mid.bg !== after.bg ? 'interpolating, not a cut' : 'HARD CUT'}`);
console.log(`  tick     ${before.tickOpacity} -> ${after.tickOpacity}`);
await p.screenshot({ path: process.argv[2], fullPage: false });
await b.close();
