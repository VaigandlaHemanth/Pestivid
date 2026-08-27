// The row shared by Messages and What has happened.
//
// These were one page, and the conversations were appended AFTER the notices --
// so a buyer with one chat and eighteen notices found the chat at row nineteen
// and reported, correctly, that the product had no chat. They are two surfaces
// now: Messages is people, and the envelope in the bar leads to the notices,
// which is what its unread badge was counting all along.
//
// And until now the notices page was a picture of a list. Eight rows marked
// unread, nothing pressable, no way to clear one or all of them, and no way to
// reach the thing a row was about: it reported at you and could not be answered.
// Every row is a control now.
import { requireUser, api, load, state } from './_guard.js';
import { appChrome } from '../chrome.js';
import { acts, press } from '../wire.js';
import { repeatRows, arrive } from '../bind.js';
import { whenShort, noticeForRole } from '../api.js';

/* Which glyph a row wears.
 *
 * The board draws five, one per kind: a tick for a date that landed, a rupee for
 * money, a document for a listing, a speech bubble for a person, a warning
 * triangle for something wrong. repeatRows clones the FIRST row as its
 * template, so every row wore the tick -- and a tick means "proved" everywhere
 * else in this product.
 *
 * Scoped to .ic: a row also carries a chevron now, and an unscoped svg search
 * collected icon, chevron, icon, chevron -- so the fifth "kind" was the third
 * row's chevron and two kinds vanished.
 */
function glyphSet(root) {
  const out = {};
  const drawn = [...root.querySelectorAll('.n .ic svg, .nu .ic svg')];
  const KINDS = ['proved', 'money', 'listing', 'person', 'wrong'];
  drawn.forEach((g, i) => { if (KINDS[i]) out[KINDS[i]] = g.cloneNode(true); });
  return out;
}

function kindOf(r) {
  if (r.go) return 'person';
  const txt = `${r.head} ${r.body}`.toLowerCase();
  // Somebody wrote to you. r.go marks a conversation ROW on the messages page;
  // on the notices page the same event arrives as a notice, and it was wearing
  // the document glyph -- so "Demo wrote: what price are you asking" was filed
  // as paperwork.
  if (/\bwrote\b|asked you|wrote to you/.test(txt)) return 'person';
  if (/\bblock\b/.test(txt) || /date .*(landed|written)/.test(txt)) return 'proved';
  if (/bought|paid|funded|asking for money|investor/.test(txt)) return 'money';
  if (/listed|listing/.test(txt)) return 'listing';
  if (/failed|queried|problem|cannot|refus/.test(txt)) return 'wrong';
  return 'listing';
}

/* Where a notice leads, when it leads anywhere.
 *
 * Only pairs that really exist: a screen that shows the thing the notice is
 * about, to the role reading it. Everything else has no destination and says so
 * by having no chevron, so you can tell before you press rather than after --
 * pressing one of those marks it read and leaves you where you are, which is
 * what a notification centre is supposed to do.
 */
function destination(n, role) {
  // An admin's notices are the queue they exist to act on, so all of them lead
  // to the one screen that lets them. Without this every row on the admin's
  // notices page was a dead end.
  if (role === 'admin') return n.itemType ? 'admin' : null;
  switch (n.itemType) {
    case 'Message': return 'messages';
    case 'FundingRequest':
      return role === 'investor' ? 'invest' : role === 'farmer' ? 'money' : null;
    case 'Investment':
      return role === 'investor' ? 'portfolio' : role === 'farmer' ? 'money' : null;
    case 'Listing':
      if (role === 'buyer') return n.type === 'purchase' ? 'orders' : 'market';
      if (role === 'farmer') return n.type === 'purchase' ? 'money' : 'plots';
      return null;
    default: return null;
  }
}

/**
 * @param slug  the page this runs on. Only 'notifications' now: the
 *              conversations list moved into the two-pane chat page.
 */
export function notices(slug, kind = 'notices') {
  const ctx = requireUser(slug);
  if (!ctx) return;
  appChrome(ctx.root, { back: 'history', user: ctx.user });
  press(ctx.root);

  load(ctx.root, async () => {
    const id = ctx.user._id || ctx.user.id;
    const role = ctx.user.role;
    const glyphs = glyphSet(ctx.root);

    // Both pages show the unread COUNT of notices, because that is what the
    // envelope badge counts and the two must never disagree. Which means both
    // count the same SET -- see noticeForRole in api.js.
    const all = await api.notifications.mine(id).catch(() => []);
    const notes = (all || []).filter(n => noticeForRole(n, role));
    const unreadOf = () => notes.filter(n => !n.read).length;

    const sub = ctx.root.querySelector('[data-bind="unreadLine"]');
    const badge = ctx.root.querySelector('.appbar [data-readout]');
    /* One number, three places: the line under the heading, the badge on the
     * envelope, and the rail's control. They were only ever written once, at
     * load, so marking a row read left both readouts contradicting the page
     * they were describing. */
    const say = (n) => {
      if (sub && kind === 'notices') {
        sub.textContent = n ? `${n} you have not read` : 'Nothing unread';
      }
      if (badge) {
        if (n > 0) { badge.textContent = n > 9 ? '9+' : String(n); badge.style.display = ''; }
        else badge.style.display = 'none';
      }
    };

    /* One kind of row now. This module used to serve the conversations list as
     * well ('people'), until the chat became a two-pane page of its own -- so
     * that branch described a page that no longer exists and has gone with it. */
    const items = notes.map(n => ({
      id: n._id,
      head: n.title || n.message,
      body: n.title ? (n.message || '') : '',
      when: whenShort(n.timestamp || n.createdAt),
      unread: !n.read,
      opens: destination(n, role),
    }));
    say(unreadOf());

    if (!items.length) {
      return state(ctx.root, 'empty', 'Nothing has happened yet',
        'Dates landing in a block, money moving, a lot selling. Only your own, and never an advertisement.');
    }

    /* ---- marking one read ---------------------------------------------
     * On screen first, then on the server. The row settles from unread to read
     * in place: same box, same padding, only the blue bar fading out, so a row
     * you just pressed cannot twitch. If the server refuses, the bar comes back
     * -- a row that says "read" when the server still says otherwise is a lie
     * that survives the next reload.
     */
    const markRead = async (el, nid) => {
      const rec = notes.find(n => String(n._id) === String(nid));
      if (!rec || rec.read) return true;
      rec.read = true;
      /* The inline transition is not decoration here, it is a repair.
       *
       * arrive() gives every row an INLINE `transition: opacity, transform` for
       * its staged entrance, and an inline declaration outranks the board's
       * `transition: box-shadow` on .n/.nu -- so the blue bar snapped off
       * instead of fading, and the one piece of feedback this press has was
       * lost to a rule written for a different moment. The entrance is long
       * finished by the time anything is pressed, so its transition is replaced
       * rather than fought with. */
      el.style.transition = 'box-shadow var(--t-smooth, 746ms) var(--e-smooth, ease)';
      el.className = 'n';
      say(unreadOf());
      try {
        await api.notifications.read(nid);
        return true;
      } catch (err) {
        rec.read = false;
        el.className = 'nu';       // the bar comes back the same way it left
        say(unreadOf());
        state(ctx.root.querySelector('[data-markall]')?.parentElement || ctx.root,
          'failed', 'That did not save', err.message);
        return false;
      }
    };

    const painter = (el, r) => {
      el.className = r.unread ? 'nu' : 'n';
      const h = el.querySelector('.h'); if (h) h.textContent = r.head;
      // An empty detail wrote a dash on its own line. Nothing to say means
      // nothing on screen, not a placeholder standing where prose would be.
      const b = el.querySelector('.b');
      if (b) { if (r.body) b.textContent = r.body; else b.remove(); }
      const t = el.querySelector('.t');
      if (t) { if (r.when) t.textContent = r.when; else t.remove(); }

      const kindName = kindOf(r);
      const drawnGlyph = el.querySelector('.ic svg');
      const want = glyphs[kindName];
      if (drawnGlyph && want) {
        const g = want.cloneNode(true);
        // Green is for a fact anybody can check without us, so only a date that
        // landed in a block gets it.
        g.setAttribute('stroke', kindName === 'proved' ? '#006934'
          : kindName === 'wrong' ? '#a71930' : '#1d1a17');
        g.setAttribute('aria-hidden', 'true');
        g.setAttribute('focusable', 'false');
        drawnGlyph.replaceWith(g);
      }

      const opens = r.go || r.opens || null;
      // The chevron is a promise that there is somewhere to go. A row with
      // nowhere to go does not make it.
      if (!opens) el.querySelector('.go')?.remove();
      if (r.id) el.dataset.nid = r.id;

      const label = opens
        ? `${r.head}. Opens ${String(opens).split('?')[0]}`
        : `${r.head}. Mark as read`;
      acts(el, label, async () => {
        if (r.id) {
          const ok = await markRead(el, r.id);
          if (!ok) return;
        }
        if (opens) el.dataset.go = opens;      // wire.js navigates on data-go
        if (opens) location.href = `./${String(opens).split('?')[0]}.html`
          + (String(opens).includes('?') ? '?' + String(opens).split('?')[1] : '');
      });
    };

    arrive(repeatRows(ctx.root, '.n, .nu', items, painter) || []);

    /* ---- clearing all of them -----------------------------------------
     * Marking eight rows one at a time is not a feature. The rows settle in
     * sequence rather than all at once, because eight bars vanishing on the
     * same frame reads as the list being wiped rather than caught up on.
     */
    const markAll = ctx.root.querySelector('[data-markall]');
    if (markAll && kind === 'notices') {
      /* When there is nothing unread this control used to relabel itself to
       * "Nothing unread" -- which put those two words on the page twice, once as
       * the line under the heading and once as a grey button that did nothing
       * when pressed. The line under the heading is the readout; a control with
       * no work left is not a readout, so it leaves instead.
       *
       * It leaves on opacity alone, and only if it was already on screen -- a
       * page that opens with nothing unread never draws it, so there is nothing
       * to fade. */
      const paintAll = () => {
        if (unreadOf()) {
          markAll.firstElementChild?.replaceChildren(
            document.createTextNode('Mark all as read'));
          markAll.style.boxShadow = 'inset 0 0 0 1.5px #016abe';
          return;
        }
        if (!markAll.isConnected) return;
        markAll.setAttribute('aria-hidden', 'true');
        markAll.style.pointerEvents = 'none';
        markAll.style.transition = 'opacity var(--t-snappy, 568ms) var(--e-snappy, ease)';
        markAll.style.opacity = '0';
        const gone = () => markAll.remove();
        markAll.addEventListener('transitionend', gone, { once: true });
        // transitionend never arrives under reduced motion, where the
        // transition is suppressed outright.
        setTimeout(gone, 900);
      };
      // At load, nothing unread means the control was never wanted at all.
      if (!unreadOf()) markAll.remove(); else paintAll();
      acts(markAll, 'Mark all as read', async () => {
        const rows = [...ctx.root.querySelectorAll('.nu')];
        if (!rows.length) return;
        for (let i = 0; i < rows.length; i++) {
          const el = rows[i];
          const nid = el.dataset.nid;
          if (!nid) continue;
          // 45ms is the stagger arrive() already uses for a list, so a row
          // settling and a row arriving move at the same cadence.
          await new Promise(r => setTimeout(r, i ? 45 : 0));
          await markRead(el, nid);
        }
        paintAll();
      });
    } else {
      markAll?.remove();
    }

    press(ctx.root);
  });
}
