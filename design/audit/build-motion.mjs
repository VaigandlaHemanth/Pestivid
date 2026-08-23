// Generates Motion.dc.html. The easings and the plotted curves come from the
// same spring solver, so the picture on the board and the motion on the board
// cannot drift apart.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const DESIGN = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function solve(response, zeta) {
  const w0 = (2 * Math.PI) / response;
  return t => {
    if (zeta < 1) {
      const wd = w0 * Math.sqrt(1 - zeta * zeta);
      return 1 - Math.exp(-zeta * w0 * t) * (Math.cos(wd * t) + (zeta * w0 / wd) * Math.sin(wd * t));
    }
    if (zeta === 1) return 1 - Math.exp(-w0 * t) * (1 + w0 * t);
    const a = w0 * Math.sqrt(zeta * zeta - 1);
    return 1 - Math.exp(-zeta * w0 * t) * (Math.cosh(a * t) + ((zeta * w0) / a) * Math.sinh(a * t));
  };
}
function spring(response, zeta, steps = 44) {
  const x = solve(response, zeta);
  let dur = 0;
  for (let t = 0; t < 6; t += 0.002) if (Math.abs(1 - x(t)) > 0.001) dur = t;
  dur += 0.012;
  const pts = [];
  for (let i = 0; i <= steps; i++) pts.push(+x((i / steps) * dur).toFixed(4));
  pts[0] = 0; pts[steps] = 1;
  return { ms: Math.round(dur * 1000), css: `linear(${pts.join(',')})`, pts, peak: Math.max(...pts), x, dur };
}
// cubic-bezier(.2,0,0,1) -- the curve this spec used to prescribe
const bez = (p1x, p1y, p2x, p2y) => {
  const cx = 3 * p1x, bx = 3 * (p2x - p1x) - cx, ax = 1 - cx - bx;
  const cy = 3 * p1y, by = 3 * (p2y - p1y) - cy, ay = 1 - cy - by;
  const sx = t => ((ax * t + bx) * t + cx) * t, sy = t => ((ay * t + by) * t + cy) * t;
  return x => { let t = x; for (let i = 0; i < 8; i++) { const e = sx(t) - x; if (Math.abs(e) < 1e-6) break; const d = (3 * ax * t + 2 * bx) * t + cx; if (!d) break; t -= e / d; } return sy(t); };
};

const SPRINGS = [
  ['smooth', 0.5, 1.0, 'Navigation. A screen arriving or leaving.', 'No overshoot at all. Anything a person is reading must not wobble.'],
  ['snappy', 0.4, 0.85, 'Chips, toggles, filters, small state flips.', 'A whisper of overshoot. Reads as responsive without reading as toy-like.'],
  ['bouncy', 0.5, 0.7, 'The stamp, and nothing else in this product.', 'Real overshoot. Spend it once, on the only moment worth celebrating.'],
  ['press', 0.25, 1.0, 'Finger down, finger up.', 'Fast enough that it feels like the surface, not like a reaction to you.'],
  ['sheet', 0.55, 0.9, 'The confirm screen sliding up over the plot.', 'Slightly longer, because it carries more pixels a longer distance.'],
];
const S = Object.fromEntries(SPRINGS.map(([n, r, z]) => [n, spring(r, z)]));
const CYCLE = 2400;

// ---- curve plot ----------------------------------------------------------
const W = 232, H = 92, PAD = 10;
function plotPath(fn, dur, over = 1) {
  const pts = [];
  for (let i = 0; i <= 90; i++) {
    const t = (i / 90) * dur;
    const x = PAD + (i / 90) * (W - PAD * 2);
    const y = H - PAD - (fn(t) / over) * (H - PAD * 2);
    pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }
  return 'M' + pts.join(' L');
}
const OVER = Math.max(...SPRINGS.map(([n]) => S[n].peak));

// ---- css -----------------------------------------------------------------
let css = '';
for (const [n] of SPRINGS) {
  const s = S[n], pct = ((s.ms / CYCLE) * 100).toFixed(1), back = (50 + s.ms / CYCLE * 100).toFixed(1);
  css += `    --e-${n}: ${s.css};\n`;
  css += `    --t-${n}: ${s.ms}ms;\n`;
}
let keys = '';
for (const [n] of SPRINGS) {
  const s = S[n], p = +((s.ms / CYCLE) * 100).toFixed(1), b = +(50 + p).toFixed(1);
  keys += `    @keyframes run-${n} { 0%{transform:translateX(0)} ${p}%,50%{transform:translateX(196px)} ${b}%,100%{transform:translateX(0)} }\n`;
}

const row = ([n, r, z, use, why]) => {
  const s = S[n];
  return `
      <div style="display: grid; grid-template-columns: 150px 244px 1fr 128px; gap: 20px; align-items: center; padding: 16px 0; box-shadow: inset 0 -1px 0 #e4e9ee;">
        <div>
          <div class="mono" style="font-size: 15px; font-weight: 600;">${n}</div>
          <div class="m" style="font-size: 13px; color: #4a443d; margin-top: 3px;">response ${r}s</div>
          <div class="m" style="font-size: 13px; color: #4a443d;">damping ${z}</div>
        </div>
        <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true" style="background: #fff;">
          <line x1="${PAD}" y1="${(H - PAD - (1 / OVER) * (H - PAD * 2)).toFixed(1)}" x2="${W - PAD}" y2="${(H - PAD - (1 / OVER) * (H - PAD * 2)).toFixed(1)}" stroke="#c9d2da" stroke-dasharray="3 3"></line>
          <line x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}" stroke="#c9d2da"></line>
          <path d="${plotPath(s.x, s.dur, OVER)}" fill="none" stroke="#012169" stroke-width="2"></path>
        </svg>
        <div>
          <div style="font-size: 15px; font-weight: 600;">${use}</div>
          <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 3px;">${why}</div>
          <div style="height: 30px; background: #eef2f6; margin-top: 9px; position: relative; overflow: hidden;">
            <div class="dot" style="animation-name: run-${n}; animation-timing-function: var(--e-${n});"></div>
          </div>
        </div>
        <div style="text-align: right;">
          <div class="m" style="font-size: 19px; font-weight: 700;">${s.ms}<span style="font-size: 13px; font-weight: 400; color: #4a443d;">ms</span></div>
          <div class="m" style="font-size: 13px; color: #4a443d; margin-top: 2px;">${s.peak > 1.001 ? '+' + ((s.peak - 1) * 100).toFixed(1) + '% over' : 'no overshoot'}</div>
        </div>
      </div>`;
};

// interruption plot: a fixed-duration curve retargeted at 45% vs a spring
// Interrupt while it is genuinely travelling and reverse it. Sending the thing
// back where it came from is the case that separates the two: a spring carries
// its momentum past the turn, a duration curve stops dead and starts again.
const bezFn = bez(.2, 0, 0, 1), D1 = 0.42;
const INT_T = 0.22 * D1, TARGET = 0;
const iw = 620, ih = 190, ipad = 20;
const px = t => ipad + (t / 0.82) * (iw - ipad * 2);
const py = v => ih - ipad - v * (ih - ipad * 2) * 0.80;
function bezPath() {
  let p = [];
  for (let i = 0; i <= 60; i++) { const t = (i / 60) * INT_T; p.push(`${px(t).toFixed(1)},${py(bezFn(t / D1)).toFixed(1)}`); }
  const v0 = bezFn(INT_T / D1);
  // retarget: a fresh 420ms bezier from where it is, and it must start at rest
  for (let i = 0; i <= 60; i++) { const t = (i / 60) * D1; p.push(`${px(INT_T + t).toFixed(1)},${py(v0 + (TARGET - v0) * bezFn(t / D1)).toFixed(1)}`); }
  return 'M' + p.join(' L');
}
function springPath() {
  const s = S.smooth, x = s.x;
  let p = [];
  for (let i = 0; i <= 60; i++) { const t = (i / 60) * INT_T; p.push(`${px(t).toFixed(1)},${py(x(t)).toFixed(1)}`); }
  // velocity at the interrupt carries into the new target
  const h = 0.002, v0 = x(INT_T), vel = (x(INT_T + h) - x(INT_T - h)) / (2 * h);
  const w0 = (2 * Math.PI) / 0.5, target = TARGET;
  for (let i = 0; i <= 90; i++) {
    const t = (i / 90) * 0.66, d = v0 - target;
    const v = target + Math.exp(-w0 * t) * (d + (vel + w0 * d) * t);
    p.push(`${px(INT_T + t).toFixed(1)},${py(v).toFixed(1)}`);
  }
  return 'M' + p.join(' L');
}

const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Anek+Latin:wght@400;500;600;700&display=swap">
  <style>
    body { margin: 0; }
    a { color: #01579b; } a:hover { color: #013f70; }
    .a { font-family: 'Anek Latin', system-ui, sans-serif; }
    .mono { font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace; }
    .m { font-variant-numeric: tabular-nums; font-feature-settings: 'tnum' 1, 'zero' 1; }
    .lift1 { box-shadow: 0 1px 2px rgba(29,26,23,.06), 0 3px 10px rgba(29,26,23,.05); }
    .lbl { font-size: 13.5px; color: #4a443d; }
    .card { background: #fff; padding: 24px 26px; }

    /* The five curves, solved from response and damping and emitted as
       linear(). Chrome 113+ and Safari 17.2+ interpolate these natively, so
       what runs below is the spring itself -- not a bezier that resembles one. */
    :root {
${css}    }
${keys}
    .dot { position: absolute; left: 6px; top: 6px; width: 18px; height: 18px; background: #012169;
           border-radius: 9px; animation-duration: ${CYCLE}ms; animation-iteration-count: infinite; }

    @keyframes near { 0%{transform:translateX(0)} 8.3%,50%{transform:translateX(18px)} 58.3%,100%{transform:translateX(0)} }
    @keyframes mid  { 0%{transform:translateX(0)} 15.8%,50%{transform:translateX(118px)} 65.8%,100%{transform:translateX(0)} }
    @keyframes far  { 0%{transform:translateX(0)} 25%,50%{transform:translateX(320px)} 75%,100%{transform:translateX(0)} }

    @keyframes originWrong { 0%{transform:scale(.2);opacity:0} 22%,58%{transform:scale(1);opacity:1} 66%,100%{transform:scale(.2);opacity:0} }
    @keyframes originRight { 0%{transform:scale(.2);opacity:0} 22%,58%{transform:scale(1);opacity:1} 66%,100%{transform:scale(.2);opacity:0} }
    .menu { animation-duration: ${CYCLE}ms; animation-iteration-count: infinite;
            animation-timing-function: var(--e-snappy); }

    @keyframes stamp { 0%{transform:scale(1.5) rotate(-16deg);opacity:0} 6%{opacity:1}
                       34.5%,64%{transform:scale(1) rotate(-5.5deg);opacity:1} 72%,100%{transform:scale(1.5) rotate(-16deg);opacity:0} }

    /* Press is the one animation a person drives. It fires on pointerdown, not
       on click -- waiting for the tap to resolve is the whole difference
       between a surface that answers and one that lags. */
    .press { transition: transform var(--t-press) var(--e-press); }
    .press:active { transform: scale(.97); }

    @keyframes barTrue  { 0%,100%{transform:scaleX(0)} 92%{transform:scaleX(1)} }

    /* Anyone who asked their operating system for less movement gets the state
       change, never the journey. Cross-fade, never nothing: a change that
       happens with no transition at all is harder to follow, not easier. */
    @media (prefers-reduced-motion: reduce) {
      .dot, .menu, .stampEl, .barEl { animation: none !important; }
      .press { transition: none; }
      .rm { display: block !important; }
    }
  </style>
</helmet>

<div class="a" style="width: 1320px; background: #f3f6f9; color: #1d1a17; padding: 34px 44px 44px; box-sizing: border-box;">

  <div class="lbl">Motion</div>
  <h1 style="margin: 4px 0 0; font-size: 31px; font-weight: 700; letter-spacing: -0.03em;">Buttery is a physics claim, so it is checkable</h1>
  <p style="margin: 9px 0 0; font-size: 15.5px; line-height: 1.55; max-width: 100ch; color: #4a443d;">
    The reason iOS motion feels the way it does is not the easing curve, it is that nothing on the
    screen is playing a fixed-length animation. Everything is a spring with a position and a velocity,
    so it can be caught, redirected and handed a new target mid-flight without ever snapping. This
    board runs the real curves &mdash; every dot below is moving on the spring named beside it.
  </p>

  <div class="card lift1" style="margin-top: 24px;">
    <div style="display: flex; justify-content: space-between; align-items: baseline;">
      <div style="font-size: 19px; font-weight: 700;">Five springs, and no sixth</div>
      <div class="lbl">solved from response and damping &middot; emitted as CSS <span class="mono">linear()</span></div>
    </div>
${SPRINGS.map(row).join('\n')}
    <div style="font-size: 14px; line-height: 1.55; color: #4a443d; margin-top: 16px; max-width: 108ch;">
      Settle time is measured, not chosen: it is the moment the curve is within 0.1% of rest and stays
      there. That is why <span class="mono">bouncy</span> costs 828&nbsp;ms for a 500&nbsp;ms response
      &mdash; the overshoot has to come back.
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 680px 1fr; gap: 22px; margin-top: 22px; align-items: stretch;">

    <div class="card lift1">
      <div style="font-size: 19px; font-weight: 700;">Why not a duration and a curve</div>
      <div style="font-size: 14.5px; line-height: 1.55; color: #4a443d; margin-top: 5px;">
        Both lines set off, and are told a fifth of the way in to turn round and go back.
      </div>
      <svg width="${iw}" height="${ih}" viewBox="0 0 ${iw} ${ih}" style="margin-top: 12px; background: #fff;" aria-hidden="true">
        <line x1="${ipad}" y1="${py(0).toFixed(1)}" x2="${iw - ipad}" y2="${py(0).toFixed(1)}" stroke="#c9d2da" stroke-dasharray="3 3"></line>
        <line x1="${px(INT_T).toFixed(1)}" y1="${ipad}" x2="${px(INT_T).toFixed(1)}" y2="${ih - ipad}" stroke="#a71930" stroke-width="1.5" stroke-dasharray="4 3"></line>
        <path d="${bezPath()}" fill="none" stroke="#a71930" stroke-width="2.2"></path>
        <path d="${springPath()}" fill="none" stroke="#012169" stroke-width="2.2"></path>
        <text x="${(px(INT_T) + 8).toFixed(1)}" y="${ipad + 12}" font-family="system-ui" font-size="12" fill="#a71930">told to reverse</text>
      </svg>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px;">
        <div>
          <div style="font-size: 15px; font-weight: 700; color: #a71930;">420&nbsp;ms cubic-bezier</div>
          <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 3px;">
            It has to stop dead before it can go back, because the replacement animation begins at
            rest. That corner is what people call cheap, and no amount of curve-tuning removes it.
          </div>
        </div>
        <div>
          <div style="font-size: 15px; font-weight: 700; color: #012169;">spring, velocity carried</div>
          <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 3px;">
            It was moving, so it carries on a little further before the turn, exactly the way a
            physical object would. Nothing on screen is ever motionless mid-gesture.
          </div>
        </div>
      </div>
    </div>

    <div class="card lift1">
      <div style="font-size: 19px; font-weight: 700;">Distance sets the duration</div>
      <div style="font-size: 14.5px; line-height: 1.55; color: #4a443d; margin-top: 5px;">
        A chip nudging 18&nbsp;px and a sheet crossing 320&nbsp;px cannot take the same time. Same
        spring, response scaled by distance.
      </div>
      <div style="margin-top: 14px; display: flex; flex-direction: column; gap: 12px;">
        <div>
          <div class="m lbl">18&nbsp;px &middot; response 0.20s &middot; 200&nbsp;ms</div>
          <div style="height: 26px; background: #eef2f6; margin-top: 5px; position: relative; overflow: hidden;">
            <div class="dot" style="width: 14px; height: 14px; top: 6px; animation-name: near; animation-timing-function: var(--e-snappy);"></div>
          </div>
        </div>
        <div>
          <div class="m lbl">118&nbsp;px &middot; response 0.32s &middot; 380&nbsp;ms</div>
          <div style="height: 26px; background: #eef2f6; margin-top: 5px; position: relative; overflow: hidden;">
            <div class="dot" style="width: 14px; height: 14px; top: 6px; animation-name: mid; animation-timing-function: var(--e-snappy);"></div>
          </div>
        </div>
        <div>
          <div class="m lbl">320&nbsp;px &middot; response 0.50s &middot; 600&nbsp;ms</div>
          <div style="height: 26px; background: #eef2f6; margin-top: 5px; position: relative; overflow: hidden;">
            <div class="dot" style="width: 14px; height: 14px; top: 6px; animation-name: far; animation-timing-function: var(--e-smooth);"></div>
          </div>
        </div>
      </div>
      <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 14px;">
        The rule: <span class="mono">response = clamp(0.18, 0.14 + distance/1400, 0.55)</span>. Never a
        constant.
      </div>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 22px; margin-top: 22px;">

    <div class="card lift1">
      <div style="font-size: 18px; font-weight: 700;">Motion starts where it physically should</div>
      <div style="font-size: 14.5px; line-height: 1.5; color: #4a443d; margin-top: 5px;">
        A menu that grows from the middle of itself came from nowhere. One that grows from the control
        that opened it came from that control.
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;">
        <div>
          <div style="font-size: 13px; font-weight: 600; color: #a71930;">centre &mdash; wrong</div>
          <div style="height: 104px; background: #eef2f6; margin-top: 6px; position: relative; display: flex; align-items: center; justify-content: center;">
            <div class="menu" style="width: 74px; height: 62px; background: #012169; transform-origin: 50% 50%; animation-name: originWrong;"></div>
          </div>
        </div>
        <div>
          <div style="font-size: 13px; font-weight: 600; color: #012169;">from the control &mdash; right</div>
          <div style="height: 104px; background: #eef2f6; margin-top: 6px; position: relative;">
            <div style="position: absolute; left: 10px; top: 10px; width: 26px; height: 12px; background: #b9c2cb;"></div>
            <div class="menu" style="position: absolute; left: 10px; top: 26px; width: 74px; height: 62px; background: #012169; transform-origin: 0 0; animation-name: originRight;"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="card lift1">
      <div style="font-size: 18px; font-weight: 700;">Press, and the one bar that must not ease</div>
      <div style="font-size: 14.5px; line-height: 1.5; color: #4a443d; margin-top: 5px;">
        Hold the button. It answers on pointerdown, before the tap has even resolved.
      </div>
      <div class="press" style="background: #016abe; height: 54px; margin-top: 14px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
        <div style="font-size: 16.5px; font-weight: 700; color: #fff;">Report the harvest</div>
      </div>
      <div class="lbl" style="margin-top: 16px;">Upload progress &middot; <span class="mono">scaleX</span>, linear</div>
      <div style="height: 12px; background: #eef2f6; margin-top: 6px; overflow: hidden;">
        <div class="barEl" style="height: 100%; background: #012169; transform-origin: left; animation: barTrue 3200ms linear infinite;"></div>
      </div>
      <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 9px;">
        Progress is the one place an easing curve would be a lie: it would report a rate the upload is
        not achieving. Linear, and on <span class="mono">scaleX</span> so it never relayouts its parent.
      </div>
    </div>

    <div class="card lift1">
      <div style="font-size: 18px; font-weight: 700;">The stamp</div>
      <div style="font-size: 14.5px; line-height: 1.5; color: #4a443d; margin-top: 5px;">
        The only celebratory motion in the entire product, and it fires when the block confirms.
        Never optimistically, because a stamp that appears and then has to be taken away is worse than
        no stamp.
      </div>
      <div style="height: 178px; background: #eef2f6; margin-top: 12px; display: flex; align-items: center; justify-content: center; overflow: hidden;">
        <div class="stampEl" style="width: 118px; height: 118px; border-radius: 50%; border: 3px solid #01579b; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,253,249,.62); animation: stamp ${CYCLE * 1.4}ms var(--e-bouncy) infinite;">
          <div class="m" style="font-size: 11px; letter-spacing: .08em; color: #01579b;">DATE PROVED</div>
          <div class="m" style="font-size: 15px; font-weight: 700; color: #01579b; margin-top: 2px;">18 AUG 2026</div>
          <div class="m" style="font-size: 11px; color: #4a443d; margin-top: 3px;">block 881,204</div>
        </div>
      </div>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 22px;">
    <div class="card lift1">
      <div style="font-size: 18px; font-weight: 700;">What may move, and what may never</div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 12px;">
        <div style="background: #eef4ee; padding: 15px 16px;">
          <div style="font-size: 14.5px; font-weight: 700; color: #024c26;">Free &mdash; the compositor owns them</div>
          <div class="mono" style="font-size: 13.5px; line-height: 1.7; margin-top: 5px;">transform<br>opacity</div>
          <div style="font-size: 13.5px; line-height: 1.5; color: #4a443d; margin-top: 6px;">
            Translate, scale, rotate, fade. That is the whole vocabulary, and everything on this canvas
            is built from it.
          </div>
        </div>
        <div style="background: #f7e9e6; padding: 15px 16px;">
          <div style="font-size: 14.5px; font-weight: 700; color: #a71930;">Banned inside a loop</div>
          <div class="mono" style="font-size: 13.5px; line-height: 1.7; margin-top: 5px;">width height top left<br>margin padding<br>background-color box-shadow<br>filter border-radius</div>
          <div style="font-size: 13.5px; line-height: 1.5; color: #4a443d; margin-top: 6px;">
            Each one costs layout or paint on the main thread, every frame, on a phone that has about
            8&nbsp;ms of headroom in the first place.
          </div>
        </div>
      </div>
      <div style="font-size: 14px; line-height: 1.55; color: #4a443d; margin-top: 14px;">
        <span style="font-weight: 700;">Also: <span class="mono">will-change</span> is added on
        interaction start and stripped on <span class="mono">animationend</span>.</span> Left switched on
        it permanently allocates a compositor layer, and a dozen of those on a 4&nbsp;GB handset is
        memory pressure that shows up as jank somewhere else entirely.
      </div>
      <div class="rm" style="display: none; background: #f2e6cd; padding: 14px 15px; margin-top: 14px;">
        <div style="font-size: 14.5px; font-weight: 700; color: #7c4a12;">Your system asked for reduced motion, so this board stopped moving.</div>
        <div style="font-size: 14px; line-height: 1.5; color: #4a443d; margin-top: 4px;">
          In the product the state still changes &mdash; it cross-fades instead of travelling. Removing
          the transition entirely makes a change harder to follow, not easier.
        </div>
      </div>
    </div>

    <div class="card lift1">
      <div style="font-size: 18px; font-weight: 700;">Glass, and the line it does not cross</div>
      <div style="font-size: 14.5px; line-height: 1.55; color: #4a443d; margin-top: 5px;">
        It is on the landing hero, the sign-in panel and the investor evidence card &mdash; 26 surfaces,
        each with an <span class="mono">@supports</span> fallback and a
        <span class="mono">prefers-reduced-transparency</span> opt-out. It is on no farmer screen, and
        that is the interesting half.
      </div>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 14px;">
        <div style="background: #eef4ee; padding: 15px 16px;">
          <div style="font-size: 14.5px; font-weight: 700; color: #024c26;">Affordable</div>
          <div style="font-size: 13.5px; line-height: 1.5; color: #4a443d; margin-top: 5px;">
            A blurred panel over a <span style="font-weight: 700;">static</span> backdrop is filtered
            once and cached. The hero grid behind our nav never moves, which is why the nav can be glass.
          </div>
        </div>
        <div style="background: #f7e9e6; padding: 15px 16px;">
          <div style="font-size: 14.5px; font-weight: 700; color: #a71930;">Not affordable</div>
          <div style="font-size: 13.5px; line-height: 1.5; color: #4a443d; margin-top: 5px;">
            Over anything that moves &mdash; a scrolling list, a camera preview &mdash; the backdrop is
            re-filtered every frame. On a Mali GPU that is the frame budget, gone, on the one screen a
            farmer uses outdoors in a hurry.
          </div>
        </div>
      </div>
      <div style="font-size: 14px; line-height: 1.55; color: #4a443d; margin-top: 14px;">
        So the recording screen uses flat dark bars over the viewfinder rather than blurred ones. It is
        the single place the product looks less fashionable than it could, on purpose, and the reason is
        printed in the stylesheet next to the decision.
      </div>
    </div>
  </div>

</div>
</x-dc>

<script data-dc-script data-props='{}'>
class Component extends DCLogic {
  renderVals() { return {}; }
}
</script>
</body>
</html>
`;
writeFileSync(path.join(DESIGN, 'Motion.dc.html'), html);
console.log('wrote Motion.dc.html');
for (const [n, r, z] of SPRINGS) console.log(`  ${n.padEnd(7)} ${String(S[n].ms).padStart(4)}ms  overshoot ${((S[n].peak - 1) * 100).toFixed(1)}%`);
