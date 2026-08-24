// Both ends of a conversation are the same screen with a different strip under
// the name, so they are the same module.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeatRows } from '../bind.js';
import { whenShort } from '../api.js';
import { acts, asField, press } from '../wire.js';

export function thread(slug) {
  const ctx = requireUser(slug);
  if (!ctx) return;
  load(ctx.root, async () => {
    const cid = new URLSearchParams(location.search).get('c');
    if (!cid) return state(ctx.root, 'empty', 'No conversation chosen', 'Open a message from the messages list.');
    const msgs = await api.messages.inThread(cid);
    const me = String(ctx.user._id || ctx.user.id);
    const other = msgs.find(m => String(m.sender?._id || m.sender) !== me);
    const otherName = other?.senderName || other?.sender?.name || 'Them';
    bind(ctx.root, { other: {
      name: otherName,
      context: `${msgs.length} message${msgs.length === 1 ? '' : 's'}`,
    } });
    // The avatar showed the artboard's "R" whoever the conversation was with.
    const avatar = [...ctx.root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && /^[A-Z]$/.test(d.textContent.trim()));
    if (avatar) avatar.textContent = (otherName.trim()[0] || '?').toUpperCase();
    // repeatRows, never repeat(): repeat() ends in container.replaceChildren()
    // and the container here is the whole bubble column, which also holds the
    // spacer, the "Tap one to answer" prompt with its three chips, and on the
    // investor side the panel explaining that she does not have to answer. All
    // of it was deleted at load -- served live, this page reported zero chips.
    const bubbles = [...ctx.root.querySelectorAll('.them, .me')];
    if (!msgs.length) {
      const first = bubbles[0];
      bubbles.slice(1).forEach(b => b.remove());
      if (first) state(first, 'empty', 'Nothing said yet', 'Write the first message.');
    } else {
      repeatRows(ctx.root, '.them, .me', msgs.map(m => ({
        mine: String(m.sender?._id || m.sender) === me,
        text: m.text, when: whenShort(m.timestamp), read: m.read,
      })), (el, r) => {
        el.className = r.mine ? 'me' : 'them';
        const t = el.querySelector('.t'); if (t) t.textContent = r.text;
        const w = el.querySelector('.when, .m');
        if (w) {
          w.textContent = r.when + (r.mine && r.read ? ' · read' : '');
          // repeatRows clones ONE template for both sides, and the first bubble
          // in the board is a received one -- so its meta colour, chosen for a
          // light bubble, came out as #4a443d on the near-black sent bubble:
          // 1.8:1, effectively invisible. The side decides the colour.
          w.style.color = r.mine ? '#c4bdb6' : '#605a53';
        }
        // The read tick belongs to a sent message and the template may carry it
        // either way round.
        const tick = el.querySelector('svg');
        if (tick) {
          tick.style.display = r.mine && r.read ? '' : 'none';
          tick.setAttribute('stroke', '#c4bdb6');
        }
      });
    }

    // ---- replying ----------------------------------------------------
    // The chips are canned replies, which is the point of them on a phone held
    // one-handed. They send; they do not merely fill the field.
    const composer = ctx.root.querySelector('[data-reply]');
    const input = composer && asField(composer, {
      name: 'reply', enterKeyHint: 'send',
      placeholder: 'Write a message', label: 'Your reply',
    });

    let sending = false;
    const send = async (text) => {
      if (!text || sending) return;
      sending = true;
      try {
        await api.messages.send(cid, { text });
        location.reload();          // the thread is the record; re-read it
      } catch (err) {
        sending = false;
        const [h, d] = [err.rateLimited ? 'Too many just now' : 'That did not send', err.message];
        state(ctx.root.querySelector('[data-sendfail]') || ctx.root, 'failed', h, d);
      }
    };

    // Canned replies are the point of the chips on a phone held one-handed in a
    // field. "Yes, I will" is not what an investor asking a question needs, so
    // they belong to the farmer's side of the same page.
    if (ctx.user.role !== 'farmer') {
      const group = ctx.root.querySelector('.chip')?.parentElement?.parentElement;
      group?.remove();
    }
    for (const chip of ctx.root.querySelectorAll('.chip')) {
      acts(chip, chip.textContent.trim(), () => send(chip.textContent.trim()));
    }
    // Inactive until there is something to send -- the same rule as the chat.
    const sendBtn = ctx.root.querySelector('[data-send]');
    const paintSend = () => {
      const has = Boolean(input?.value.trim());
      if (sendBtn) {
        // each panel keeps its own accent; only the inactive grey is shared
        sendBtn.style.background = has
          ? (sendBtn.hasAttribute('data-navy') ? '#012169' : '#016abe')
          : '#c3bcb6';
        sendBtn.setAttribute('aria-disabled', String(!has));
      }
    };
    input?.addEventListener('input', paintSend);
    paintSend();
    if (sendBtn) acts(sendBtn, 'Send your reply', () => {
      const v = input?.value.trim();
      if (!v) { input?.focus(); return; }
      input.value = ''; paintSend(); send(v);
    });
    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { const v = input.value.trim(); input.value = ''; send(v); }
    });

    press(ctx.root);
    await api.messages.markRead(cid).catch(() => {});
  });
}
