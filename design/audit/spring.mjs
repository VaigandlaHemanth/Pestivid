// Turn a spring the way iOS describes one (response + damping fraction) into a
// CSS linear() easing. Chrome 113+ / Safari 17.2+ interpolate it natively, so
// the curve on screen is the real spring, not a bezier that resembles one.
function spring(response, zeta, steps = 40) {
  const w0 = (2 * Math.PI) / response;
  const x = t => {
    if (zeta < 1) {
      const wd = w0 * Math.sqrt(1 - zeta * zeta);
      return 1 - Math.exp(-zeta * w0 * t) * (Math.cos(wd * t) + (zeta * w0 / wd) * Math.sin(wd * t));
    }
    if (zeta === 1) return 1 - Math.exp(-w0 * t) * (1 + w0 * t);
    const a = w0 * Math.sqrt(zeta * zeta - 1), c1 = 1, c2 = (zeta * w0) / a;
    return 1 - Math.exp(-zeta * w0 * t) * (c1 * Math.cosh(a * t) + c2 * Math.sinh(a * t));
  };
  // settle time: last moment it is more than 0.1% from rest
  let dur = 0;
  for (let t = 0; t < 6; t += 0.002) if (Math.abs(1 - x(t)) > 0.001) dur = t;
  dur += 0.01;
  const pts = [];
  for (let i = 0; i <= steps; i++) pts.push(+x((i / steps) * dur).toFixed(4));
  pts[0] = 0; pts[steps] = 1;
  return { ms: Math.round(dur * 1000), css: `linear(${pts.join(',')})`, peak: Math.max(...pts) };
}
for (const [name, r, z] of [['smooth', .5, 1], ['snappy', .4, .85], ['bouncy', .5, .7], ['press', .25, 1], ['sheet', .55, .9]]) {
  const s = spring(r, z);
  console.log(`${name.padEnd(7)} response ${r}s  damping ${z}   settles ${String(s.ms).padStart(4)}ms   overshoot ${((s.peak-1)*100).toFixed(1)}%`);
  console.log(`  ${s.css}\n`);
}
