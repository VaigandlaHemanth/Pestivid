// The farmer's own videos.
//
// Before this the page listed them and let you do nothing with them: zero
// controls, so the back arrow, the header and every row were decoration. It
// also repeated one long status sentence on every row, which turns a list of
// four videos into a wall of the same blue paragraph four times.
import { requireUser, api, load, state } from './_guard.js';
import { repeatRows, bind } from '../bind.js';
import { whenShort, dateState } from '../api.js';
import { appChrome } from '../chrome.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('plots', ['farmer']);

/** 38 -> "0:38". The tile is drawn for a duration; this is the real one. */
const clock = (sec) => {
  const n = Math.round(Number(sec));
  if (!Number.isFinite(n) || n <= 0) return null;
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`;
};

if (ctx) {
  appChrome(ctx.root, { back: 'home', user: ctx.user });
  goes(ctx.root.querySelector('[data-recordcta]'), 'record', 'Record a video');
  press(ctx.root);

  load(ctx.root, async () => {
    const videos = await api.videos.mine(ctx.user._id || ctx.user.id);

    if (!videos.length) {
      return state(ctx.root, 'empty', 'You have not filmed anything yet',
        'Walk across your field and record forty seconds. Nobody will fund a season on an empty account.',
        { label: 'Record a video', go: 'record' });
    }

    // The heading is a state and the count is a figure beside it. Binding the
    // count into the title left the artboard's separate "2" sitting next to
    // it, so the page read "4 videos on our server  2".
    const states = videos.map(v => dateState(v));
    const proved = states.filter(s => s.kind === 'proved').length;
    bind(ctx.root, {
      group: { title: 'On our server', count: String(videos.length) },
      onPhone: {
        note: proved === videos.length
          ? 'Every one of these has its date written into a Bitcoin block. Nobody can move them now, including us.'
          : proved > 0
            ? `${proved} of ${videos.length} already have their date written into a Bitcoin block. The rest are waiting for the next batch, usually within a day.`
            : 'All of these are with us and being batched. Their dates are usually written into a Bitcoin block within a day.',
      },
    });

    // The board drew two groups -- on the phone, and sent -- to show both
    // states at once. There is one real list, so the second heading goes.
    const second = [...ctx.root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Sent and safe');
    second?.parentElement?.parentElement?.remove();

    repeatRows(ctx.root, '.row', videos.map((v, i) => ({
      v,
      name: v.crop || v.location || 'Untitled plot',
      when: whenShort(v.uploadTimestamp),
      dur: clock(v.durationSeconds),
      s: states[i],
    })), (el, row) => {
      const ttl = el.querySelector('.ttl'); if (ttl) ttl.textContent = row.name;
      const lbl = el.querySelector('.lbl'); if (lbl) lbl.textContent = row.when;

      // The duration, in the tile corner and in its own column. Marked on the
      // board rather than found by a style string: the laptop layout moved the
      // tile's background into a class, the old selector matched nothing, and
      // every row went on showing the artboard's 0:41.
      const dur = el.querySelector('[data-dur]');
      if (dur) { if (row.dur) dur.textContent = row.dur; else dur.remove(); }
      const len = el.querySelector('[data-len]');
      if (len) {
        if (row.v.durationSeconds) len.textContent = `${Math.round(row.v.durationSeconds)} sec`;
        else len.textContent = 'not recorded';
      }

      // The state, not the explanation. Four rows repeating "date being
      // written, usually by tomorrow" says it three times too often; the
      // sentence above the list carries it once.
      const status = el.querySelector('[data-state]') || [...el.querySelectorAll('div')]
        .reverse().find(d => !d.children.length && d.textContent.trim());
      if (status) {
        status.textContent = row.s.short || row.s.text;
        // not #01579b: link blue on a state turns a reading into a link
        status.style.color = row.s.kind === 'proved' ? '#006934'
                          : row.s.kind === 'waiting' ? '#4a443d' : '#7c4a12';
      }

      // A row about one video has to open that video.
      goes(el, `plot?name=${encodeURIComponent(row.name)}`, `${row.name}, ${row.when}`);
    });

    press(ctx.root);
  });
}
