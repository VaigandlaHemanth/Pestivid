// The review queue.
//
// This page had no controls at all: four buttons that decide what happens to a
// named farmer's video, a "Resubmit that batch" button, and a four-item bar
// that looked like tabs. The bar was the worst of it -- one item was underlined
// as selected while the page below showed three of the four categories at once,
// so the selected state described a filter that does not exist, and clicking any
// of the other three did nothing because what they named was already on screen.
// It is an index now, with the real counts.
//
// The buttons are the sharp end of this product. One of them publicly marks a
// named farmer's video as not their footage, and the copy under it says every
// viewer including the buyer who already paid will see that note. There is no
// route behind any of them, so each says what it would do and refuses to
// pretend it did it -- silently swallowing the tap on THIS page is the one
// place that would be unforgivable.
import { requireUser, api, load, state } from './_guard.js';
import { acts, press } from '../wire.js';

const ctx = requireUser('admin', ['admin']);

const NOT_YET = [
  'Nothing has been written',
  'This decision has no route behind it yet, so nothing was recorded and nobody was told. '
  + 'It is listed here so it can be built, not so it can look built.',
];

if (ctx) {
  const root = ctx.root;
  press(root);

  load(root, async () => {
    const { bind } = await import('../bind.js');
    bind(root, { me: { line: `${ctx.user.name} · admin · every action here is written to the audit collection under your name` } });

    // GET /videos/review-queue answers { state, count, truncated, items } -- not
    // an array and not { videos }. Reading it as either left items as an object
    // and items.filter threw on the page, in front of the admin.
    const q = await api.admin.flagged();
    const items = Array.isArray(q) ? q
                : Array.isArray(q?.items) ? q.items
                : Array.isArray(q?.videos) ? q.videos : [];
    const reported = Number.isFinite(q?.count) ? q.count : items.length;

    // ---- the index -----------------------------------------------------
    // Real counts where the server reports them, and a plain zero where it does
    // not -- rather than the artboard's 2 / 1 / 3 / 0.
    const kinds = {
      duplicate: items.filter(v => v.flagReason === 'duplicate' || v.duplicateOf).length || (reported && !items.length ? reported : 0),
      unverified: items.filter(v => v.hashComputedBy && v.hashComputedBy !== 'server').length,
      late: items.filter(v => v.anchorLate).length,
      location: items.filter(v => v.locationImpossible).length,
    };
    const LABEL = {
      duplicate: 'Same footage, two farmers',
      unverified: 'Hash not computed by us',
      late: 'Timestamps running late',
      location: 'Location looks impossible',
    };
    for (const [kind, n] of Object.entries(kinds)) {
      const el = root.querySelector(`[data-count="${kind}"]`);
      if (!el) continue;
      el.textContent = `${LABEL[kind]} · ${n}`;
      // the one with anything in it reads as current; the rest are just counts
      el.className = n > 0 ? 'tabOn' : 'tab';
    }

    // A panel with nothing in it must not go on showing the artboard's figures.
    // "Timestamps running late -- three batches, 41 videos" over an index that
    // says 0 is the same contradiction the money screen opened with.
    for (const [kind, n] of Object.entries(kinds)) {
      if (n > 0) continue;
      root.querySelector(`[data-panel="${kind}"]`)?.remove();
    }

    // With nothing flagged -- which is the normal state -- every panel in that
    // row goes, and "What this page cannot do" was left as a 400px dark block
    // alone at the left of a 1440px window, under two full-width cards. It is
    // the most important statement on the page; when it is the only thing left
    // in its row it takes the row.
    const cannot = [...root.querySelectorAll('div')]
      .find(d => d.firstElementChild?.textContent.trim() === 'What this page cannot do');
    const column = cannot?.parentElement;
    const rowOf = column?.parentElement;
    if (cannot && column && rowOf) {
      const siblingsLeft = [...rowOf.children].filter(c => c !== column && c.textContent.trim()).length;
      if (!siblingsLeft) {
        column.style.width = 'auto';
        column.style.flexGrow = '1';
      }
    }

    // ---- the decisions -------------------------------------------------
    for (const btn of root.querySelectorAll('.btn')) {
      const label = btn.textContent.trim();
      acts(btn, label, () => {
        let note = btn.parentElement?.querySelector('[data-note]');
        if (!note) {
          note = document.createElement('div');
          note.setAttribute('data-note', '');
          btn.parentElement?.after(note);
        }
        state(note, 'waiting', NOT_YET[0], `“${label}”, ${NOT_YET[1]}`);
      });
    }

    if (!items.length) {
      // Not state(ctx.root, ...): that replaced the page root and took the
      // heading, the nav and the index with it, so an admin lost every way off
      // the screen. The message replaces the QUEUE, which is the empty thing.
      const queue = root.querySelector('table')?.closest('div')
        || root.querySelector('[data-count]')?.parentElement?.nextElementSibling
        || root.lastElementChild;
      return state(queue, 'empty', 'The system has flagged nothing',
        'No duplicate footage, no unverified hashes, no late timestamps. This page being empty is the normal state.');
    }

    press(root);
  });
}
