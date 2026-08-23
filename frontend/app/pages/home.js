// The farmer's home screen.
//
// This page rendered correctly and did nothing: click-everything.mjs found ZERO
// controls on it. Nine things looked pressable -- five menu rows, two header
// icons, the harvest button, the speak button -- and not one of them was wired.
// Binding data is not the same as building a page.
import { requireUser, api, load } from './_guard.js';
import { bind, oneByText } from '../bind.js';
import { press, goes, acts, sayable } from '../wire.js';

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

  // ---- read aloud ------------------------------------------------------
  // The speaker on every row used to be a drawing. Read-aloud matters more
  // here than anywhere else in the product, so it is real now -- but only
  // when the device can actually speak the language. sayable() returns false
  // when no matching voice is installed, and then the button is removed
  // rather than left there to do nothing.
  for (const [label] of rows) {
    const row = oneByText(label, root)?.closest('.row');
    const btn = row?.querySelector('.say');
    if (!btn) continue;
    const line = [label, row.querySelector('.nat')?.textContent?.trim()]
      .filter(Boolean).join('. ');
    if (!sayable(btn, line, `Read aloud: ${label}`)) btn.remove();
  }

  // ---- speech input ----------------------------------------------------
  // Same rule. Chrome has webkitSpeechRecognition, most other browsers do not,
  // so the button exists only where it can work. It hands what it hears to the
  // chatbot, which is the one screen on this app that takes a sentence.
  const speak = oneByText('Speak instead of typing', root)?.parentElement;
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (speak && !Rec) {
    speak.parentElement?.remove();
  } else if (speak) {
    acts(speak, 'Speak instead of typing', () => {
      sessionStorage.setItem('pv.listen', '1');
      location.href = './ask.html';
    });
  }

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
