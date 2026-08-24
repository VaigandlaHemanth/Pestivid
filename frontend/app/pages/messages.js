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

  // The board draws FIVE glyphs, one per kind of row: a tick for a date that
  // landed, a rupee for money, a document for a listing, a speech bubble for a
  // person, a warning triangle for something wrong. repeatRows clones the FIRST
  // row as its template, so all seven rows on this page wore the tick -- and a
  // tick means "proved" everywhere else in this product.
  const glyphs = {};
  {
    const drawn = [...ctx.root.querySelectorAll('.n svg, .nu svg')];
    const KINDS = ['proved', 'money', 'listing', 'person', 'wrong'];
    drawn.forEach((g, i) => { if (KINDS[i]) glyphs[KINDS[i]] = g.cloneNode(true); });
  }
  const kindOf = (r) => {
    if (r.go) return 'person';
    const txt = `${r.head} ${r.body}`.toLowerCase();
    if (/\bblock\b/.test(txt) || /date .*(landed|written)/.test(txt)) return 'proved';
    if (/bought|paid|funded|asking for money|investor/.test(txt)) return 'money';
    if (/listed|listing/.test(txt)) return 'listing';
    if (/failed|queried|problem|cannot|refus/.test(txt)) return 'wrong';
    return 'listing';
  };

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

    // The glyph says which kind of thing the row is. Green is for a fact anybody
    // can check without us, so only a date that landed in a block gets it.
    //
    // The test here was /\bblock\b/ and had been written into the file with two
    // literal BACKSPACE characters instead of the word boundaries -- my own
    // Python escaping -- so it never matched anything and the green never
    // appeared. It is the only such corruption in the tree; checked.
    const kind = kindOf(r);
    const drawnGlyph = el.querySelector('svg');
    const want = glyphs[kind];
    if (drawnGlyph && want) {
      const g = want.cloneNode(true);
      g.setAttribute('stroke', kind === 'proved' ? '#006934'
        : kind === 'wrong' ? '#a71930' : '#1d1a17');
      g.setAttribute('aria-hidden', 'true');
      g.setAttribute('focusable', 'false');
      drawnGlyph.replaceWith(g);
    }
    if (r.go) { el.dataset.go = r.go; el.setAttribute('data-act', ''); }
  });
});
