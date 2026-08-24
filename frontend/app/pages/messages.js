import { requireUser, api, load, state } from './_guard.js';
import { appChrome } from '../chrome.js';
import { press } from '../wire.js';
import { bind, repeatRows } from '../bind.js';
import { whenShort } from '../api.js';

const ctx = requireUser('messages');
if (ctx) { appChrome(ctx.root, { back: 'home', user: ctx.user }); press(ctx.root); }
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

  const items = [
    ...(notes || []).map(n => ({ head: n.title || n.message, body: n.title ? (n.message || '') : '', when: whenShort(n.timestamp || n.createdAt), unread: !n.read })),
    ...(threads || []).map(t => ({ head: t.otherName || 'A message', body: t.lastMessageSnippet || '', when: whenShort(t.lastMessageTimestamp), unread: false, go: `thread.html?c=${t._id}` })),
  ];
  if (!items.length) {
    return state(ctx.root, 'empty', 'Nothing yet',
      'Only your own plots and your own money appear here. We do not send offers.',
        { label: 'Ask a question', go: 'ask' });
  }
  repeatRows(ctx.root, '.n, .nu', items, (el, r) => {
    el.className = r.unread ? 'nu' : 'n';
    const h = el.querySelector('.h'); if (h) h.textContent = r.head;
    // An empty detail wrote an em dash on its own line. Nothing to say means
    // nothing on screen, not a placeholder standing where prose would be.
    const b = el.querySelector('.b');
    if (b) { if (r.body) b.textContent = r.body; else b.remove(); }
    const t = el.querySelector('.t');
    if (t) { if (r.when) t.textContent = r.when; else t.remove(); }

    // Green is for a fact anybody can check without us. Only one kind of row
    // here qualifies -- a video's date landing in a Bitcoin block -- so only
    // that row gets it, and the rest keep ink.
    const tick = el.querySelector('svg');
    if (tick && /block/i.test(`${r.head} ${r.body}`)) {
      tick.setAttribute('stroke', '#006934');
      tick.setAttribute('stroke-width', '2.4');
    }
    if (r.go) { el.dataset.go = r.go; el.setAttribute('data-act', ''); }
  });
});
