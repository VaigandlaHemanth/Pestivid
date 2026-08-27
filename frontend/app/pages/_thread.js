// Both ends of a conversation are the same screen with a different strip under
// the name, so they are the same module.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeatRows } from '../bind.js';
import { whenShort, dayMonth } from '../api.js';
import { acts, asField, press } from '../wire.js';

export function thread(slug) {
  const ctx = requireUser(slug);
  if (!ctx) return;
  load(ctx.root, async () => {
    const cid = new URLSearchParams(location.search).get('c');
    if (!cid) return state(ctx.root, 'empty', 'No conversation chosen', 'Open a message from the messages list.');
    const me = String(ctx.user._id || ctx.user.id);
    // The conversation list is where the other person's NAME lives. The message
    // endpoint answers a bare array, so a conversation nobody has written in
    // yet -- which is every conversation a buyer has just opened from a lot --
    // had no name at all and the header said "Them".
    const [msgs, threads] = await Promise.all([
      api.messages.inThread(cid),
      api.messages.threads(me).catch(() => []),
    ]);
    const listed = (threads || []).find(t => String(t._id || t.id) === String(cid));
    const other = msgs.find(m => String(m.sender?._id || m.sender) !== me);
    const otherName = other?.senderName || other?.sender?.name
      || listed?.otherName || 'Them';
    let sent = msgs.length;
    bind(ctx.root, { other: {
      name: otherName,
      context: `${msgs.length} message${msgs.length === 1 ? '' : 's'}`,
    } });
    // The avatar showed the artboard's "R" whoever the conversation was with.
    // Marked, not guessed: the first childless single capital letter on the page
    // is the nav bar's own avatar, which is the signed-in farmer, not the person
    // being talked to. Relabelling that one renamed the wrong human.
    const avatar = ctx.root.querySelector('[data-avatar]')
      || [...ctx.root.querySelectorAll('div')]
        .find(d => d.children.length === 0 && !d.closest('.appbar')
                   && /^[A-Z]$/.test(d.textContent.trim()));
    if (avatar) avatar.textContent = (otherName.trim()[0] || '?').toUpperCase();
    // repeatRows, never repeat(): repeat() ends in container.replaceChildren()
    // and the container here is the whole bubble column, which also holds the
    // spacer, the "Tap one to answer" prompt with its three chips, and on the
    // investor side the panel explaining that she does not have to answer. All
    // of it was deleted at load -- served live, this page reported zero chips.
    /* ---- one bubble ---------------------------------------------------
     * Named, because three things paint a bubble now: the transcript on load,
     * the message you have just typed, and that same message once the server
     * has taken it. They must agree to the pixel or sending flickers.
     */
    const paint = (el, r) => {
      el.className = r.mine ? 'me' : 'them';
      const t = el.querySelector('.t'); if (t) t.textContent = r.text;
      const w = el.querySelector('.when, .m');
      if (w) {
        // The day is the separator's job. Inside the bubble it is the clock and
        // nothing else -- "17 August, 1:40 pm" on every line of a conversation
        // held in one afternoon is the same six words twenty times.
        w.textContent = r.time + (r.mine && r.read ? ' · read' : '');
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
    };

    const clock = (ts) => ts
      ? new Date(ts).toLocaleTimeString('en-IN', { hour: 'numeric', minute: '2-digit' }).toLowerCase()
      : '';
    // Which day a message belongs to, in the words a reader uses for it.
    const dayOf = (ts) => {
      if (!ts) return '';
      const d = new Date(ts), now = new Date();
      const days = Math.round((new Date(now.toDateString()) - new Date(d.toDateString())) / 86400000);
      return days === 0 ? 'Today' : days === 1 ? 'Yesterday' : dayMonth(ts);
    };
    const row = (m) => ({
      mine: String(m.sender?._id || m.sender) === me,
      text: m.text, time: clock(m.timestamp), day: dayOf(m.timestamp),
      when: whenShort(m.timestamp), read: m.read,
    });

    const tape = ctx.root.querySelector('[data-transcript]');
    // The board draws one day label so the spacing can be judged; the real ones
    // are computed from the messages.
    const dayTpl = ctx.root.querySelector('.day');
    const daySep = (label) => {
      const el = dayTpl ? dayTpl.cloneNode(true) : document.createElement('div');
      el.className = 'day';
      el.textContent = label;
      return el;
    };
    dayTpl?.remove();

    const bubbles = [...ctx.root.querySelectorAll('.them, .me')];
    const tpl = bubbles[0]?.cloneNode(true) || null;
    if (!msgs.length) {
      const first = bubbles[0];
      bubbles.slice(1).forEach(b => b.remove());
      if (first) state(first, 'empty', 'Nothing said yet', 'Write the first message.');
    } else {
      const rows = msgs.map(row);
      const made = repeatRows(ctx.root, '.them, .me', rows, paint) || [];
      // A day label goes in front of the first message of each day. Inserted
      // after the fact rather than inside repeatRows, which repeats one
      // template and cannot emit a different element between rows.
      let seen = null;
      made.forEach((el, i) => {
        el.dataset.day = rows[i].day || '';
        if (rows[i].day && rows[i].day !== seen) {
          seen = rows[i].day;
          el.before(daySep(seen));
        }
      });
    }

    /* ---- appending one, without reloading the page --------------------
     * Sending used to end in location.reload(): the whole document was thrown
     * away and rebuilt, which is the flicker -- a white flash, the transcript
     * gone, the scroll back at the top, and the message just written
     * reappearing half a second later. A chat is the one place where the state
     * is already in the client's hands, so it is added in place and confirmed.
     */
    const addBubble = (r) => {
      if (!tpl || !tape) return null;
      const el = tpl.cloneNode(true);
      paint(el, r);
      const last = [...tape.querySelectorAll('.them, .me')].pop();
      if (r.day && r.day !== last?.dataset.day) tape.append(daySep(r.day));
      el.dataset.day = r.day || '';
      // It comes up from where the composer is, which is where the eye already
      // is. Transform and opacity only, and it is one short beat -- a message
      // appearing, not a panel arriving.
      const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.style.opacity = '0';
      if (!still) el.style.transform = 'translateY(10px) scale(.97)';
      el.style.transition = 'opacity var(--t-press, 120ms) var(--e-smooth, ease),'
        + ' transform var(--t-snappy, 568ms) var(--e-snappy, ease)';
      tape.append(el);
      requestAnimationFrame(() => {
        el.style.opacity = r.pending ? '0.55' : '1';
        el.style.transform = 'none';
      });
      tape.scrollTo({ top: tape.scrollHeight, behavior: still ? 'auto' : 'smooth' });
      return el;
    };

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
      const now = Date.now();
      // On screen immediately, at reduced opacity: it is written, not yet
      // acknowledged. This is what makes sending feel instant.
      const el = addBubble({ mine: true, text, time: clock(now), day: dayOf(now),
                             read: false, pending: true });
      try {
        const saved = await api.messages.send(cid, { text });
        const ts = saved?.timestamp || saved?.message?.timestamp || now;
        if (el) {
          paint(el, { mine: true, text, time: clock(ts), read: false });
          el.style.opacity = '1';        // acknowledged
        }
        // The strip under the name counts the messages, so it cannot stay a
        // message behind the transcript it is describing.
        sent += 1;
        bind(ctx.root, { other: { name: otherName, context: `${sent} messages` } });
        paintChips();
      } catch (err) {
        // It did not send, so it does not stay on screen pretending it did.
        el?.remove();
        const [h, d] = [err.rateLimited ? 'Too many just now' : 'That did not send', err.message];
        state(ctx.root.querySelector('[data-sendfail]') || ctx.root, 'failed', h, d);
      } finally { sending = false; }
    };

    // The transcript scrolls now, so a conversation opens in the middle of
    // itself unless it is sent to the end. The newest message is the one being
    // answered, so that is what has to be on screen.
    if (tape) requestAnimationFrame(() => { tape.scrollTop = tape.scrollHeight; });

    // Canned replies are the point of the chips on a phone held one-handed in a
    // field. "Yes, I will" is not what an investor asking a question needs, so
    // they belong to the farmer's side of the same page.
    const chipGroup = ctx.root.querySelector('.chip')?.parentElement?.parentElement;
    if (ctx.user.role !== 'farmer') chipGroup?.remove();
    /* And "Tap one to answer" standing under a message YOU just sent is the
     * screen offering to answer itself. The chips are for a question waiting on
     * you, so they are there when the last thing said came from the other side
     * and gone when it did not -- which also gives a short laptop back the
     * hundred pixels the transcript wanted. */
    const paintChips = () => {
      if (!chipGroup) return;
      const last = [...ctx.root.querySelectorAll('.them, .me')].pop();
      chipGroup.style.display = last && last.classList.contains('them') ? '' : 'none';
    };
    paintChips();
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
