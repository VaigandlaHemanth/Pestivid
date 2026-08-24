// The farmer's home screen.
//
// This page rendered correctly and did nothing: click-everything.mjs found ZERO
// controls on it. Nine things looked pressable -- five menu rows, two header
// icons, the harvest button, the speak button -- and not one of them was wired.
// Binding data is not the same as building a page.
import { requireUser, api, load } from './_guard.js';
import { bind, oneByText } from '../bind.js';
import { press, goes } from '../wire.js';
import { rupees } from '../api.js';

// Only the seasons that are still open are worth summarising: a closed one is
// history and belongs on the money screen, not on the row that leads to it.
// FundingRequest.status is one of pending | partially_funded | funded |
// completed | cancelled, and the money that has arrived is `fundedAmount`.
// Written from the model rather than guessed: reading a field that does not
// exist is how the portfolio page came to show three wrong numbers.
const OPEN = ['pending', 'partially_funded', 'funded'];
function moneyLine(projects) {
  const open = (projects || []).filter(p => OPEN.includes(p.status));
  if (!open.length) return 'Nothing asked for yet';
  const asked = open.reduce((a, p) => a + (p.amount || 0), 0);
  const inHand = open.reduce((a, p) => a + (p.fundedAmount || 0), 0);
  if (!asked) return `${open.length} season${open.length === 1 ? '' : 's'} open`;
  return `${rupees(inHand)} in of ${rupees(asked)} asked for`;
}

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
    ['Check a leaf',   'leaf-check'],
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

  // Wired below, in the data callback: this ran at setup, before anything knew
  // WHICH season needed reporting, so it linked to report-harvest with no id --
  // and that page, opened without one, is a dead end saying "No season chosen".
  // Every one of the three routes into it had the same fault.

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
    const reportBtn = oneByText('Report the harvest', root)?.parentElement;
    if (reportBtn && due) goes(reportBtn, `report-harvest?project=${due._id}`, 'Report the harvest');
    bind(root, {
      whoWhere: me.name,
      todo: { headline: due ? `${due.title} is ready to harvest` : 'Nothing needs you today' },
      plots: {
        waiting: videos.length
          ? `${videos.length} video${videos.length === 1 ? '' : 's'} filed`
          : 'Nothing filed yet',
      },
      // The most useful line on a farmer's home screen: how much has actually
      // come in against what was asked for. The row carried no line at all.
      money: { line: moneyLine(projects) },
    });

    // The empty first run used to be a page of its own, home-empty.html, which
    // said something home's empty state did not: WHY it is empty and what to do
    // about it. A first run is a state of this screen, not another destination,
    // so the words moved here and the page went away.
    const nothingYet = !videos.length;
    if (!due) {
      const btn = oneByText('Report the harvest', root);
      btn?.closest('div[style*="background: #016abe"]')?.remove();
    }
    if (nothingYet) {
      const band = root.querySelector('div[style*="background: #1d1a17"]');
      const head = root.querySelector('[data-bind="todo.headline"]');
      const kicker = head?.previousElementSibling;
      if (kicker) kicker.textContent = 'Start here';
      if (head) head.textContent = 'Walk across your field and film it';
      if (band && head) {
        const why = document.createElement('div');
        why.style.cssText = 'font-size: 15px; line-height: 1.5; margin-top: 7px; color: #c4bdb6;';
        why.textContent = 'This is empty because you have not filmed anything yet. '
          + 'Forty seconds is enough, and the date is fixed the moment it reaches us.';
        head.after(why);
        const go = document.createElement('div');
        go.style.cssText = 'background: #016abe; min-height: 56px; margin-top: 14px; display: flex;'
          + ' align-items: center; justify-content: center; font-size: 17px; font-weight: 700; color: #fff;';
        go.textContent = 'Film your field';
        band.append(go);
        goes(go, 'record', 'Film your field');
        press(root);
      }
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
