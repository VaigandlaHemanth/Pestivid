// The row shared by Messages and What has happened.
//
// These were one page, and the conversations were appended AFTER the notices --
// so a buyer with one chat and eighteen notices found the chat at row nineteen
// and reported, correctly, that the product had no chat. They are two surfaces
// now: Messages is people, and the envelope in the bar leads to the notices,
// which is what its unread badge was counting all along.
import { requireUser, api, load, state } from './_guard.js';
import { appChrome } from '../chrome.js';
import { press } from '../wire.js';
import { repeatRows, arrive } from '../bind.js';
import { whenShort } from '../api.js';

/* Which glyph a row wears.
 *
 * The board draws five, one per kind: a tick for a date that landed, a rupee for
 * money, a document for a listing, a speech bubble for a person, a warning
 * triangle for something wrong. repeatRows clones the FIRST row as its
 * template, so every row wore the tick -- and a tick means "proved" everywhere
 * else in this product.
 */
function glyphSet(root) {
  const out = {};
  const drawn = [...root.querySelectorAll('.n svg, .nu svg')];
  const KINDS = ['proved', 'money', 'listing', 'person', 'wrong'];
  drawn.forEach((g, i) => { if (KINDS[i]) out[KINDS[i]] = g.cloneNode(true); });
  return out;
}

function kindOf(r) {
  if (r.go) return 'person';
  const txt = `${r.head} ${r.body}`.toLowerCase();
  if (/\bblock\b/.test(txt) || /date .*(landed|written)/.test(txt)) return 'proved';
  if (/bought|paid|funded|asking for money|investor/.test(txt)) return 'money';
  if (/listed|listing/.test(txt)) return 'listing';
  if (/failed|queried|problem|cannot|refus/.test(txt)) return 'wrong';
  return 'listing';
}

function painter(glyphs) {
  return (el, r) => {
    el.className = r.unread ? 'nu' : 'n';
    const h = el.querySelector('.h'); if (h) h.textContent = r.head;
    // An empty detail wrote a dash on its own line. Nothing to say means nothing
    // on screen, not a placeholder standing where prose would be.
    const b = el.querySelector('.b');
    if (b) { if (r.body) b.textContent = r.body; else b.remove(); }
    const t = el.querySelector('.t');
    if (t) { if (r.when) t.textContent = r.when; else t.remove(); }

    const kind = kindOf(r);
    const drawnGlyph = el.querySelector('svg');
    const want = glyphs[kind];
    if (drawnGlyph && want) {
      const g = want.cloneNode(true);
      // Green is for a fact anybody can check without us, so only a date that
      // landed in a block gets it.
      g.setAttribute('stroke', kind === 'proved' ? '#006934'
        : kind === 'wrong' ? '#a71930' : '#1d1a17');
      g.setAttribute('aria-hidden', 'true');
      g.setAttribute('focusable', 'false');
      drawnGlyph.replaceWith(g);
    }
    if (r.go) { el.dataset.go = r.go; el.setAttribute('data-act', ''); }
  };
}

/**
 * @param slug   'messages' or 'notifications'
 * @param kind   'people' for conversations, 'notices' for what the system said
 */
export function notices(slug, kind) {
  const ctx = requireUser(slug);
  if (!ctx) return;
  appChrome(ctx.root, { back: 'history', user: ctx.user });
  press(ctx.root);

  load(ctx.root, async () => {
    const id = ctx.user._id || ctx.user.id;
    const glyphs = glyphSet(ctx.root);

    // Both pages show the unread COUNT of notices, because that is what the
    // envelope badge counts and the two must never disagree.
    const notes = await api.notifications.mine(id).catch(() => []);
    const unread = (notes || []).filter(n => !n.read).length;

    let items;
    if (kind === 'people') {
      const threads = await api.messages.threads(id).catch(() => []);
      items = (threads || []).map(t => ({
        head: t.otherName || 'A message',
        body: t.lastMessageSnippet || '',
        when: whenShort(t.lastMessageTimestamp),
        unread: Boolean(t.unread || t.unreadCount),
        // The slug, not the filename: wire.js turns `thread?c=1` into
        // `./thread.html?c=1`. Passing `thread.html?c=1` made it thread.html.html.
        go: `thread?c=${t._id}`,
      }));
    } else {
      items = (notes || []).map(n => ({
        head: n.title || n.message,
        body: n.title ? (n.message || '') : '',
        when: whenShort(n.timestamp || n.createdAt),
        unread: !n.read,
      }));
    }

    const sub = ctx.root.querySelector('[data-bind="unreadLine"]');
    if (sub) {
      sub.textContent = kind === 'people'
        ? (items.length === 1 ? 'One conversation' : `${items.length} conversations`)
        : (unread ? `${unread} you have not read` : 'Nothing unread');
    }

    if (!items.length) {
      return kind === 'people'
        ? state(ctx.root, 'empty', 'No conversations yet',
            'People with money in one of your seasons can write to you here. Nobody else can start one.')
        : state(ctx.root, 'empty', 'Nothing has happened yet',
            'Dates landing in a block, money moving, a lot selling. Only your own, and never an advertisement.');
    }
    arrive(repeatRows(ctx.root, '.n, .nu', items, painter(glyphs)) || []);
    press(ctx.root);
  });
}
