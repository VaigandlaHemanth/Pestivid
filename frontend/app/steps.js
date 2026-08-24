// The Multi Step Loader, from design/Components.dc.html.
//
// The board's own argument for it: "A 173 MB download on a metered prepaid pack
// is a financial event, not a loading state. Named steps and a real percentage
// — an indeterminate spinner past ten seconds fails Nielsen's own threshold,
// and this takes minutes."
//
// What was there instead was one line of text with a percentage appended, which
// is the spinner argument in a different costume: it never said which of the
// four things was happening, and a farmer watching a prepaid pack drain needs
// to know that the asking and the waiting are already behind them.
const TICK = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f6f3ef" stroke-width="3.2"><path d="M5 12.5l4.5 4.5L19 7.5"></path></svg>';

/**
 * @param into      element to render into (replaced).
 * @param steps     [{ label, note }] in order.
 * @returns { at(i, pct, note), fail(i, why), done() }
 */
export function multiStep(into, steps) {
  if (!into) return { at() {}, fail() {}, done() {} };

  into.replaceChildren();
  into.style.cssText = 'background: #f6f3ef; padding: 16px 17px; margin: 16px 20px;';
  const rows = steps.map((s, i) => {
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; gap: 12px; align-items: flex-start;'
      + (i ? ' margin-top: 13px;' : '');

    // The dot is ink when a step is finished, not green: green in this product
    // means a fact anybody can check without trusting us, and "we downloaded a
    // file onto your phone" is our word for it.
    const dot = document.createElement('div');
    dot.style.cssText = 'width: 20px; height: 20px; flex-shrink: 0; margin-top: 2px;'
      + ' box-shadow: inset 0 0 0 2px #78716a; display: flex; align-items: center;'
      + ' justify-content: center; transition: background-color var(--t-release, 260ms) var(--e-press, ease);';

    const body = document.createElement('div');
    const label = document.createElement('div');
    label.style.cssText = 'font-size: 15.5px; font-weight: 600; color: #605a53;';
    label.textContent = s.label;
    const note = document.createElement('div');
    note.style.cssText = 'font-size: 14px; line-height: 1.45; margin-top: 2px; color: #605a53;';
    if (s.note) note.textContent = s.note; else note.style.display = 'none';

    // A real percentage, on a bar that rides on transform so it cannot relayout
    // the page it is sitting in.
    const track = document.createElement('div');
    track.style.cssText = 'height: 6px; background: #d3ccc5; margin-top: 8px; overflow: hidden; display: none;';
    const fill = document.createElement('div');
    fill.style.cssText = 'width: 100%; height: 6px; background: #016abe;'
      + ' transform-origin: left; transform: scaleX(0); transition: transform 240ms linear;';
    track.append(fill);

    body.append(label, note, track);
    row.append(dot, body);
    into.append(row);
    return { row, dot, label, note, track, fill };
  });

  const settle = (i, state) => {
    const r = rows[i];
    if (!r) return;
    if (state === 'done') {
      r.dot.style.background = '#1d1a17';
      r.dot.style.boxShadow = 'none';
      r.dot.innerHTML = TICK;
      r.label.style.color = '#1d1a17';
      r.track.style.display = 'none';
    } else if (state === 'now') {
      r.dot.style.background = 'transparent';
      r.dot.style.boxShadow = 'inset 0 0 0 2px #016abe';
      r.dot.innerHTML = '';
      r.label.style.color = '#1d1a17';
    } else if (state === 'failed') {
      r.dot.style.background = '#a71930';
      r.dot.style.boxShadow = 'none';
      r.dot.innerHTML = '';
      r.label.style.color = '#a71930';
      r.track.style.display = 'none';
    }
  };

  return {
    /** Mark step i as the one happening, everything before it finished. */
    at(i, pct, note) {
      for (let n = 0; n < rows.length; n++) settle(n, n < i ? 'done' : n === i ? 'now' : 'ahead');
      const r = rows[i];
      if (!r) return;
      if (note) { r.note.textContent = note; r.note.style.display = ''; }
      if (pct == null) { r.track.style.display = 'none'; return; }
      r.track.style.display = '';
      r.fill.style.transform = `scaleX(${Math.max(0, Math.min(1, pct / 100))})`;
      into.setAttribute('aria-valuenow', String(Math.round(pct)));
    },
    fail(i, why) {
      settle(i, 'failed');
      if (rows[i] && why) { rows[i].note.textContent = why; rows[i].note.style.display = ''; }
    },
    done() {
      for (let n = 0; n < rows.length; n++) settle(n, 'done');
    },
  };
}
