// The farmer's home screen.
//
// This page rendered correctly and did nothing: click-everything.mjs found ZERO
// controls on it. Nine things looked pressable -- five menu rows, two header
// icons, the harvest button, the speak button -- and not one of them was wired.
// Binding data is not the same as building a page.
import { requireUser, api, load } from './_guard.js';
import { bind, oneByText } from '../bind.js';
import { press, goes } from '../wire.js';

const ctx = requireUser('home', ['farmer']);

if (ctx) {
  const root = ctx.root;

  // ---- navigation ------------------------------------------------------
  // Each row is a whole-row target, not a link on the label: a 44px glyph
  // inside a 76px row means most of what looks pressable is not.
  const rows = [
    ['Record a video', 'record'],
    ['My plots',       'plots'],
    ['Money',          'money'],
    ['Check a leaf',   'leaf-result'],
    ['Ask a question', 'ask'],
  ];
  for (const [label, dest] of rows) {
    const el = oneByText(label, root)?.closest('.row');
    if (el) goes(el, dest, label);
  }

  // The header glyphs. Both boxes are 44px now; the mail badge inside the
  // first one is a readout and must not swallow the press.
  const header = root.querySelector('div[style*="justify-content: space-between"]');
  const glyphs = header ? [...header.querySelectorAll('div[style*="width: 44px"]')] : [];
  if (glyphs[0]) goes(glyphs[0], 'messages', 'Messages');
  if (glyphs[1]) goes(glyphs[1], 'profile', 'Your profile');

  goes(oneByText('Report the harvest', root)?.parentElement, 'report-harvest', 'Report the harvest');

  // Read-aloud and speech input were here. Both are gone -- the drawn speaker
  // on every row and the "Speak instead of typing" block have left the board
  // too, rather than being left visible and dead.

  press(root);

  // ---- data ------------------------------------------------------------
  load(root, async () => {
    const [me, videos, projects] = await Promise.all([
      api.auth.me(),
      api.videos.mine(ctx.user._id || ctx.user.id),
      api.projects.mine(ctx.user._id || ctx.user.id).catch(() => []),
    ]);
    const due = (projects || []).find(p => p.status === 'funded' && !p.harvestReportedAt);
    bind(root, {
      whoWhere: me.name,
      todo: { headline: due ? `${due.title} is ready to harvest` : 'Nothing needs you today' },
      plots: {
        waiting: videos.length
          ? `${videos.length} video${videos.length === 1 ? '' : 's'} filed`
          : 'Nothing filed yet',
      },
    });

    // Do not leave a call to action pointing at nothing. The whole dark block
    // goes, not just the button: "Needs you today" over an empty space reads
    // like something failed to load.
    if (!due) {
      const btn = oneByText('Report the harvest', root);
      btn?.closest('div[style*="background: #016abe"]')?.remove();
    }

    // The unread badge showed the artboard's "2" whatever the truth was. There
    // is no unread-count route, so it counts unread notifications, which is
    // what the envelope actually leads to.
    const badge = root.querySelector('[data-readout]');
    if (badge) {
      const notes = await api.notifications.mine(ctx.user._id || ctx.user.id).catch(() => []);
      const unread = (notes || []).filter(n => !n.read && !n.isRead).length;
      if (unread > 0) badge.textContent = unread > 9 ? '9+' : String(unread);
      else badge.remove();
    }
  });
}
