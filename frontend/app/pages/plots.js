import { requireUser, api, load, state } from './_guard.js';
import { repeatRows, bind } from '../bind.js';
import { whenShort, dateState } from '../api.js';

const ctx = requireUser('plots', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const videos = await api.videos.mine(ctx.user._id || ctx.user.id);

  // one list, so the drawn group heading has to describe it honestly rather
  // than still saying "still on your phone" about videos already sent
  if (!videos.length) {
    return state(ctx.root, 'empty', 'You have not filmed anything yet',
      'Walk across your field and record forty seconds. Nobody will fund a season on an empty account.');
  }
  bind(ctx.root, { group: { title: `${videos.length} video${videos.length === 1 ? '' : 's'} on our server` },
                   onPhone: { note: 'Every one of these is with us. Their dates are being written or already written.' } });
  // The board split the list in two -- on the phone, and sent -- to show both
  // states side by side. There is one real list, so the second heading and its
  // note go with the second group rather than being left describing nothing.
  const second = [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'Sent and safe');
  second?.parentElement?.parentElement?.remove();
  repeatRows(ctx.root, '.row', videos.map(v => {
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
