import { requireUser, api, load, state } from './_guard.js';
import { repeat, bind } from '../bind.js';
import { whenShort, dateState } from '../api.js';

const ctx = requireUser('plots', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const videos = await api.videos.mine(ctx.user._id || ctx.user.id);
  const groups = [...ctx.root.querySelectorAll('.row')].map(r => r.parentElement);
  const container = groups[0];
  if (!videos.length) {
    return state(ctx.root, 'empty', 'You have not filmed anything yet',
      'Walk across your field and record forty seconds. Nobody will fund a season on an empty account.');
  }
  // every group after the first held a second example; one list is the truth
  [...new Set(groups)].slice(1).forEach(g => g.remove());
  repeat(container, videos.map(v => {
    const s = dateState(v);
    return { name: v.crop || v.location || 'Untitled plot', when: whenShort(v.uploadTimestamp), status: s.text, kind: s.kind };
  }), (el, row) => {
    const ttl = el.querySelector('.ttl'); if (ttl) ttl.textContent = row.name;
    const lbl = el.querySelector('.lbl'); if (lbl) lbl.textContent = row.when;
    const last = el.querySelectorAll('div');
    const status = [...last].reverse().find(d => !d.children.length && d.textContent.trim());
    if (status) {
      status.textContent = row.status;
      status.style.color = row.kind === 'proved' ? '#006934' : row.kind === 'waiting' ? '#01579b' : '#7c4a12';
    }
  });
});
