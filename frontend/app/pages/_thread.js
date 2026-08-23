// Both ends of a conversation are the same screen with a different strip under
// the name, so they are the same module.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { whenShort } from '../api.js';

export function thread(slug) {
  const ctx = requireUser(slug);
  if (!ctx) return;
  load(ctx.root, async () => {
    const cid = new URLSearchParams(location.search).get('c');
    if (!cid) return state(ctx.root, 'empty', 'No conversation chosen', 'Open a message from the messages list.');
    const msgs = await api.messages.inThread(cid);
    const me = String(ctx.user._id || ctx.user.id);
    const other = msgs.find(m => String(m.sender?._id || m.sender) !== me);
    bind(ctx.root, { other: {
      name: other?.sender?.name || 'Them',
      context: `${msgs.length} message${msgs.length === 1 ? '' : 's'}`,
    } });
    const bubbles = [...ctx.root.querySelectorAll('.them, .me')];
    const container = bubbles[0]?.parentElement;
    if (!msgs.length) {
      return state(container, 'empty', 'Nothing said yet', 'Write the first message.');
    }
    repeat(container, msgs.map(m => ({
      mine: String(m.sender?._id || m.sender) === me,
      text: m.text, when: whenShort(m.timestamp), read: m.read,
    })), (el, r) => {
      el.className = r.mine ? 'me' : 'them';
      const t = el.querySelector('.t'); if (t) t.textContent = r.text;
      const w = el.querySelector('.when, .m'); if (w) w.textContent = r.when + (r.mine && r.read ? ' · read' : '');
    });
    await api.messages.markRead(cid).catch(() => {});
  });
}
