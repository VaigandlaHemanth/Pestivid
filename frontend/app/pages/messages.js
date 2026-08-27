// Chat: the people on the left, the talking on the right, on one page.
//
// This was two pages. A list you tapped, then a conversation you had navigated
// to, then the back button to change who you were talking to. Every chat anybody
// has used puts them side by side and makes switching a click, so that is what
// this is -- and the page count went 24 to 22 on the way (payout went the same
// week, for the same reason: a second screen whose only new information fits on
// the first).
//
// Three pieces of copy went with the merge, all of them named:
//   "16 messages"  -- a count of the thing you are looking at.
//   "These are kept on our server and are not scrambled. Assume we could read
//    them."  -- a standing disclosure, printed above the keyboard on every
//    single reply. It belongs in the privacy copy, once.
//   "Words only"  -- the absence of a paperclip already says it.
// The one fact worth keeping is that only people who funded you can write here,
// because it explains why the list is short. It sits at the top of the list.
import { requireUser, api, load, state } from './_guard.js';
import { bind, rows, arrive } from '../bind.js';
import { whenShort, dayMonth } from '../api.js';
import { acts, asField, press } from '../wire.js';
import { appChrome } from '../chrome.js';

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

const ctx = requireUser('messages');
if (ctx) {
  appChrome(ctx.root, { back: 'history', user: ctx.user });

  load(ctx.root, async () => {
    const root = ctx.root;
    const me = String(ctx.user._id || ctx.user.id);
    const threads = (await api.messages.threads(me).catch(() => [])) || [];

    if (!threads.length) {
      root.querySelector('[data-chatcol]')?.remove();
      return state(root, 'empty', 'No conversations yet',
        'People with money in one of your seasons can write to you here. Nobody else can start one.');
    }

    /* The templates, taken BEFORE anything is rendered. The right pane is
     * rebuilt every time a different person is picked, so the empty bubble and
     * the empty day label have to survive the first render -- repeatRows and
     * rows() both consume their template. */
    const tape = root.querySelector('[data-transcript]');
    const bubbleTpl = root.querySelector('.them')?.cloneNode(true) || null;
    const dayTpl = root.querySelector('.day')?.cloneNode(true) || null;
    const chipGroup = root.querySelector('[data-chips]');
    const spacer = root.querySelector('[data-spacer]');
    [...root.querySelectorAll('.them, .me, .day')].forEach(el => el.remove());

    // ---- one bubble ------------------------------------------------------
    const paint = (el, r) => {
      el.className = r.mine ? 'me' : 'them';
      const t = el.querySelector('.bt'); if (t) t.textContent = r.text;
      const w = el.querySelector('.when, .m');
      if (w) {
        // The day is the separator's job. Inside the bubble it is the clock.
        w.textContent = r.time + (r.mine && r.read ? ' · read' : '');
        // One template serves both sides, and the drawn one is a received
        // bubble -- so its meta colour, chosen for a light fill, came out as
        // #4a443d on near-black: 1.8:1. The side decides the colour.
        w.style.color = r.mine ? '#c4bdb6' : '#605a53';
      }
      const tick = el.querySelector('svg');
      if (tick) {
        tick.style.display = r.mine && r.read ? '' : 'none';
        tick.setAttribute('stroke', '#c4bdb6');
      }
      el.dataset.day = r.day || '';
    };
    const daySep = (label) => {
      const el = dayTpl ? dayTpl.cloneNode(true) : document.createElement('div');
      el.className = 'day';
      el.textContent = label;
      return el;
    };
    const row = (m) => ({
      mine: String(m.sender?._id || m.sender) === me,
      text: m.text, time: clock(m.timestamp), day: dayOf(m.timestamp), read: m.read,
    });

    /* A bubble arriving. It comes up from where the composer is, which is where
     * the eye already is: transform and opacity only, one short beat. */
    const addBubble = (r, animate) => {
      if (!bubbleTpl || !tape) return null;
      const el = bubbleTpl.cloneNode(true);
      paint(el, r);
      const last = [...tape.querySelectorAll('.them, .me')].pop();
      if (r.day && r.day !== last?.dataset.day) tape.append(daySep(r.day));
      tape.append(el);
      if (animate) {
        const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
        el.style.opacity = '0';
        if (!still) el.style.transform = 'translateY(10px) scale(.97)';
        el.style.transition = 'opacity var(--t-press, 120ms) var(--e-smooth, ease),'
          + ' transform var(--t-snappy, 568ms) var(--e-snappy, ease)';
        requestAnimationFrame(() => {
          el.style.opacity = r.pending ? '0.55' : '1';
          el.style.transform = 'none';
        });
      }
      return el;
    };
    const toBottom = (smooth) => {
      if (!tape) return;
      tape.scrollTo({ top: tape.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    };

    // ---- which conversation is open --------------------------------------
    const want = new URLSearchParams(location.search).get('c');
    let current = threads.find(t => String(t._id) === String(want)) || threads[0];
    let sending = false;

    const paintChips = () => {
      if (!chipGroup) return;
      // "Tap one to answer" under a message YOU sent is the screen offering to
      // answer itself. And they are the farmer's canned replies, not an
      // investor's -- "Yes, I will" is not what somebody asking a question needs.
      const last = [...root.querySelectorAll('.them, .me')].pop();
      const show = ctx.user.role === 'farmer' && last && last.classList.contains('them');
      chipGroup.style.display = show ? '' : 'none';
    };

    const people = rows(root, 'person', threads.map(t => ({
      initial: (String(t.otherName || '?').trim()[0] || '?').toUpperCase(),
      name: t.otherName || 'A message',
      snippet: (t.lastMessageSnippet || '').replace(/\.{3,}$/, '…') || 'No messages yet',
      when: whenShort(t.lastMessageTimestamp).replace(/^Today, /, ''),
    })));

    const paintPeople = () => {
      people.forEach((el, i) => {
        const on = String(threads[i]._id) === String(current._id);
        el.style.background = on ? '#eae4de' : 'transparent';
        el.style.boxShadow = `inset 3px 0 0 ${on ? '#016abe' : 'transparent'},`
          + ' inset 0 -1px 0 #ddd7d1';
        if (on) el.setAttribute('aria-current', 'true');
        else el.removeAttribute('aria-current');
      });
    };

    /* Opening one. No page load: the URL is corrected so the conversation can
     * still be linked to and reloaded, and only the right pane is rebuilt. */
    const open = async (t) => {
      current = t;
      paintPeople();
      history.replaceState(null, '', `./messages.html?c=${t._id}`);
      bind(root, { other: {
        name: t.otherName || 'Them',
        // Not a message count. WHY this person can write to you, which is the
        // one thing about them the screen can usefully say.
        context: 'Put money into one of your seasons',
      } });
      const av = root.querySelector('[data-avatar]');
      if (av) av.textContent = (String(t.otherName || '?').trim()[0] || '?').toUpperCase();

      [...tape.querySelectorAll('.them, .me, .day')].forEach(el => el.remove());
      const msgs = (await api.messages.inThread(t._id).catch(() => [])) || [];
      if (!msgs.length) {
        const first = addBubble({ mine: false, text: 'Nothing said yet. Write the first message.',
                                 time: '', day: '' }, false);
        if (first) first.style.opacity = '.75';
      } else {
        const made = msgs.map(m => addBubble(row(m), false)).filter(Boolean);
        arrive(made.slice(-6));
      }
      paintChips();
      toBottom(false);
      await api.messages.markRead(t._id).catch(() => {});
      // the row is no longer unread once it has been read
      const idx = threads.indexOf(t);
      if (idx >= 0) { threads[idx].unread = false; threads[idx].unreadCount = 0; }
    };

    people.forEach((el, i) => acts(el, `Open the conversation with ${threads[i].otherName || 'them'}`,
      () => { if (String(threads[i]._id) !== String(current._id)) open(threads[i]); }));

    // ---- writing ---------------------------------------------------------
    const composer = root.querySelector('[data-reply]');
    const input = composer && asField(composer, {
      name: 'reply', enterKeyHint: 'send',
      placeholder: 'Write a message', label: 'Your reply',
    });

    const send = async (text) => {
      if (!text || sending) return;
      sending = true;
      const now = Date.now();
      // On screen at once, at reduced opacity: written, not yet acknowledged.
      // Sending used to end in location.reload(), which is what the flicker was.
      const el = addBubble({ mine: true, text, time: clock(now), day: dayOf(now),
                             read: false, pending: true }, true);
      toBottom(true);
      try {
        const saved = await api.messages.send(current._id, { text });
        const ts = saved?.timestamp || saved?.message?.timestamp || now;
        if (el) { paint(el, { mine: true, text, time: clock(ts), read: false }); el.style.opacity = '1'; }
        // the left-hand row is the record of what was last said
        const idx = threads.indexOf(current);
        const snip = people[idx]?.querySelector('[data-slot="snippet"]');
        const when = people[idx]?.querySelector('[data-slot="when"]');
        if (snip) snip.textContent = text.length > 50 ? text.slice(0, 50) + '…' : text;
        if (when) when.textContent = clock(ts);
        paintChips();
      } catch (err) {
        el?.remove();                       // it did not send, so it does not stay
        const [h, d] = [err.rateLimited ? 'Too many just now' : 'That did not send', err.message];
        state(root.querySelector('[data-sendfail]') || root, 'failed', h, d);
      } finally { sending = false; }
    };

    const sendBtn = root.querySelector('[data-send]');
    const paintSend = () => {
      const has = Boolean(input?.value.trim());
      if (!sendBtn) return;
      sendBtn.style.background = has ? '#016abe' : '#c3bcb6';
      sendBtn.setAttribute('aria-disabled', String(!has));
    };
    input?.addEventListener('input', paintSend);
    paintSend();
    if (sendBtn) acts(sendBtn, 'Send your reply', () => {
      const v = input?.value.trim();
      if (!v) { input?.focus(); return; }
      input.value = ''; paintSend(); send(v);
    });
    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { const v = input.value.trim(); input.value = ''; paintSend(); send(v); }
    });
    for (const chip of root.querySelectorAll('.chip')) {
      acts(chip, chip.textContent.trim(), () => send(chip.textContent.trim()));
    }

    if (spacer) spacer.remove();     // the pane is a fixed height; nothing to push
    await open(current);
    press(root);
  });
}
