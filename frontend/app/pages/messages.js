import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { whenShort } from '../api.js';

const ctx = requireUser('messages');
if (ctx) load(ctx.root, async () => {
  const id = ctx.user._id || ctx.user.id;
  const [threads, notes] = await Promise.all([
    api.messages.threads(id).catch(() => []),
    api.notifications.mine(id).catch(() => []),
  ]);
  const unread = (notes || []).filter(n => !n.read).length;
  bind(ctx.root, { unreadLine: unread
    ? `${unread} you have not read`
    : 'Nothing unread' });

  const rows = [...ctx.root.querySelectorAll('.n, .nu')];
  const container = rows[0]?.parentElement;
  const items = [
    ...(notes || []).map(n => ({ head: n.title || n.message, body: n.detail || '', when: whenShort(n.createdAt), unread: !n.read })),
    ...(threads || []).map(t => ({ head: t.otherName || 'A message', body: t.lastMessageSnippet || '', when: whenShort(t.lastMessageTimestamp), unread: false, go: `thread-farmer.html?c=${t._id}` })),
  ];
  if (!items.length) {
    return state(container, 'empty', 'Nothing yet',
      'Only your own plots and your own money appear here. We do not send offers.');
  }
  repeat(container, items, (el, r) => {
    el.className = r.unread ? 'nu' : 'n';
    const h = el.querySelector('.h'); if (h) h.textContent = r.head;
    const b = el.querySelector('.b'); if (b) b.textContent = r.body;
    const t = el.querySelector('.t'); if (t) t.textContent = r.when;
    if (r.go) { el.dataset.go = r.go; el.setAttribute('data-act', ''); }
  });
});
