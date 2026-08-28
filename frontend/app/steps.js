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
  if (!into) return { at() {}, fail() {}, done() {}, retire() {}, bare() {} };

  // What the host was DRAWN as, kept so it can be handed back untouched.
  const drawn = into.getAttribute('style') || '';

  into.replaceChildren();
  /* Paint, and only paint.
   *
   * This line was `into.style.cssText = 'background: ...; padding: ...; margin:
   * ...'`, and cssText is a REPLACEMENT: it wiped the host's own drawn style. On
   * the leaf checker that host is drawn
   *
   *     position: absolute; left: 20px; right: 20px; bottom: 44px
   *
   * pinned inside the viewfinder. Losing it turned the panel into a static block
   * at the top of the plate, where the photograph -- absolutely positioned, and
   * so painted above anything that is not -- covered it. Every line of the
   * panel's own text ran behind the image: "173 MB, once. It never downloads on
   * its o…" was the last a farmer could read of it, on the screen that exists to
   * explain a 173 MB download.
   *
   * Three properties are three properties. */
  into.style.background = '#f6f3ef';
  into.style.padding = '16px 17px';
  /* The side margin is for a host with no geometry of its own -- it lines the
   * panel up with padded content around it. A host its own drawing positioned
   * has an inset already, and must not be shoved off it by a second one.
   *
   * The test is what the DRAWING says, not what getComputedStyle reports. On a
   * phone a stylesheet lays this panel out in the flow under the photo, so the
   * computed position is `static` there -- and asking the computed value wrote an
   * inline `margin: 16px 20px` which then outranked the stylesheet's own margin
   * and put the panel back on top of the photo. The question being asked is "did
   * its drawing give it geometry", and only the drawing can answer that. */
  if (!/position:\s*(absolute|fixed|relative)/.test(drawn)) into.style.margin = '16px 20px';
  /* A connector between the steps, and it FILLS as each one lands.
   *
   * The send screen's pipeline has had this from the start -- the line travels
   * from one marker to the next as the step above it completes -- and these four
   * steps, which are the ones a farmer actually waits minutes in front of, had
   * four markers floating with nothing joining them. Same design, same motion:
   * a 2px track in the marker column, an inner bar scaled from the top.
   *
   * 9px of left margin, not 10: the marker here is 20px wide so its centre is at
   * 10, and a 2px line centres on that at 9. The send screen's marker is 22px and
   * its line sits at 10 for the same reason. Getting this wrong by one pixel is
   * exactly the kind of drift that gets reported as "the alignment is off".
   */
  const lines = [];
  const connector = () => {
    const line = document.createElement('div');
    line.setAttribute('data-stepline', '');
    line.style.cssText = 'width: 2px; height: 18px; margin: 3px 0 3px 9px;'
      + ' background: #d3ccc5; position: relative; overflow: hidden;';
    const fill = document.createElement('div');
    fill.style.cssText = 'position: absolute; inset: 0; background: #1d1a17;'
      + ' transform-origin: top; transform: scaleY(0);'
      + ' transition: transform var(--t-smooth, 746ms) var(--e-smooth, ease);';
    line.append(fill);
    lines.push({ line, fill });
    return line;
  };

  const rows = steps.map((s, i) => {
    // The connector supplies the gap between rows, so the row no longer carries
    // its own 13px top margin -- two spacings stacked would double it.
    if (i) into.append(connector());

    const row = document.createElement('div');
    row.style.cssText = 'display: flex; gap: 12px; align-items: flex-start;';

    // The dot is ink when a step is finished, not green: green in this product
    // means a fact anybody can check without trusting us, and "we downloaded a
    // file onto your phone" is our word for it.
    const dot = document.createElement('div');
    dot.setAttribute('data-stepdot', '');
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
    /* The connector BELOW a step fills when that step lands, so the line travels
     * from the marker that just finished down to the one now working -- which is
     * the direction a reader's eye is already going. lines[i] is the connector
     * after row i, because the first row has none above it. */
    const below = lines[i];
    if (below) {
      below.fill.style.transform = state === 'done' ? 'scaleY(1)' : 'scaleY(0)';
      // A failed step's line stops being a promise that the next one is coming.
      below.line.style.background = state === 'failed' ? '#e0b4ac' : '#d3ccc5';
    }
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
    /** Hand the host back exactly as it was drawn, keeping what is in it now. */
    bare() {
      into.setAttribute('style', drawn);
    },
    /* The download is history the moment there is an answer on screen.
     *
     * Nothing retired this, so a finished checklist -- four ticks and "173 MB,
     * once" -- stayed sitting on the verdict screen for as long as the farmer
     * read their diagnosis, and collided with the filename plate that arrives
     * with it. Opacity only, then the host goes back to being the empty drawn
     * div it started as. */
    retire() {
      into.style.transition = 'opacity var(--t-snappy, 568ms) var(--e-snappy, ease)';
      into.style.opacity = '0';
      const gone = () => { into.replaceChildren(); into.setAttribute('style', drawn); };
      into.addEventListener('transitionend', gone, { once: true });
      // transitionend never arrives under reduced motion, which suppresses the
      // transition outright.
      setTimeout(gone, 900);
    },
  };
}
