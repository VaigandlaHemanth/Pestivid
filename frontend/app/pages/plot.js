import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { whenShort, dateState, rupees } from '../api.js';

const ctx = requireUser('plot', ['farmer']);
if (ctx) load(ctx.root, async () => {
  // There is no Plot entity on the server; a plot is the crop-and-location
  // string a video carries. Grouping by it here is honest about that.
  const key = new URLSearchParams(location.search).get('name');
  const all = await api.videos.mine(ctx.user._id || ctx.user.id);
  const mine = key ? all.filter(v => (v.crop || v.location) === key) : all;
  if (!mine.length) {
    return state(ctx.root, 'empty', 'No videos on this plot',
      'Film it once and this page fills itself.');
  }
  const first = mine[0];
  bind(ctx.root, { plot: {
    name: key || first.crop || first.location || 'Plot',
    meta: [first.location, first.crop].filter(Boolean).join(' · '),
    stage: `${mine.length} video${mine.length === 1 ? '' : 's'} so far`,
    videosLabel: 'Videos of this plot',
  } });
  const rows = [...ctx.root.querySelectorAll('.vid')];
  repeat(rows[0]?.parentElement, mine.map(v => {
    const s = dateState(v);
    return { when: whenShort(v.uploadTimestamp), status: s.text, kind: s.kind };
  }), (el, r) => {
    const m = el.querySelector('.m'); if (m) m.textContent = r.when;
    const status = [...el.querySelectorAll('div')].reverse().find(d => !d.children.length && d.textContent.trim());
    if (status) { status.textContent = r.status; status.style.color = r.kind === 'proved' ? '#006934' : '#4a443d'; }
  });
});
