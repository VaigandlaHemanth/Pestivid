// What an investor has funded, including what failed.
//
// Three of the four numbers on this page were wrong, in a way that always read
// as good news. portfolio.js reduced over `i.paidOut` and filtered on
// `!i.settledAt`, and neither field exists anywhere in the backend -- the API
// sends status, payoutAmount, payoutDate and progress. So "Paid back to you so
// far" was always ₹0, "Still out in unfinished seasons" always equalled the
// total put in (because !undefined is true for every row), and every "Back to
// you" cell read "Not yet".
//
// The row loop also wrote only three of the five cells, so the status pill and
// the "Waiting on" sentence kept the artboard's text on every row -- four
// different seasons all "Growing", all "Week 11 of 18", all waiting on Alice.
import { requireUser, api, load, state } from './_guard.js';
import { bind, arrive } from '../bind.js';
import { rupees, dayMonth } from '../api.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('portfolio');

// The four states the API actually reports, and the pill each one wears. No
// green anywhere: whether money reached your bank is a thing you take our word
// for, and green in this product means a fact you can check without us.
const PILL = {
  harvested: ['Paid out',     '#eae4de', '#1d1a17', 'inset 0 0 0 1.5px #1d1a17'],
  // a hairline, not none: #eae4de on the surface is 1.14:1 and the edge has
  // to be drawn or the pill is a smudge (verify-surfaces)
  growing:   ['Growing',      '#eae4de', '#4a443d', 'inset 0 0 0 1px #c3bcb6'],
  active:    ['Still raising', '#f2e6cd', '#7c4a12', 'none'],
  cancelled: ['Crop failed',  '#f7e9e6', '#a71930', 'none'],
};

if (ctx) {
  const root = ctx.root;
  goes([...root.querySelectorAll('div')].find(d => d.children.length === 0 && d.textContent.trim() === 'Browse'),
       'invest', 'Browse seasons');
  goes([...root.querySelectorAll('div')].find(d => d.children.length === 0 && d.textContent.trim() === 'Messages'),
       'messages', 'Messages');
  goes([...root.querySelectorAll('div')].find(d => d.children.length === 0 && d.textContent.trim() === 'Browse them'),
       'invest', 'Browse the open seasons');
  press(root);

  load(root, async () => {
    const initial = (ctx.user.name || '?').trim()[0].toUpperCase();
    const mine = (await api.investments.mine(ctx.user._id || ctx.user.id)) || [];

    const done = (i) => i.status === 'harvested' || i.status === 'cancelled';
    const putIn = mine.reduce((a, i) => a + (i.amount || 0), 0);
    const back = mine.reduce((a, i) => a + (i.payoutAmount || 0), 0);
    const openMoney = mine.filter(i => !done(i)).reduce((a, i) => a + (i.amount || 0), 0);

    const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
    bind(root, {
      me: { initial },
      // The rows are INVESTMENTS, and two of them can be in the same season --
      // "Organic Wheat Expansion" appeared twice, at ₹55,000 and ₹1,00,000, under
      // a heading that called them three seasons. It counts what it lists.
      heading: mine.length === 1 ? 'Your one investment'
             : `Your ${WORDS[mine.length] || mine.length} investment${mine.length === 1 ? '' : 's'}`,
      investor: { name: ctx.user.name },
      labels: { putIn: 'You have put in, all together' },
      // The note described the artboard's mix -- "two have not finished and one
      // failed outright" -- whatever the real one was.
      noReturn: (() => {
        const open = mine.filter(i => !done(i)).length;
        const failed = mine.filter(i => i.status === 'cancelled').length;
        /* THE CARD EXPLAINED A STATE THAT WAS NOT ON THE PAGE.
         *
         * "Why a failed season stays on this page" is drawn once, in the present
         * tense, as though one were sitting in the table above it -- and with
         * three healthy rows an investor reads it, looks for the failure, and
         * finds none. It is worth saying either way, because it is a promise
         * about what this page will not hide; it just has to say which one it is.
         */
        bind(ctx.root, { fail: failed ? {
          head: failed === 1 ? 'Why a failed season stays on this page'
                             : 'Why failed seasons stay on this page',
          lead: failed === 1
            ? 'It stays here permanently, in the same table and the same type size as the one that paid.'
            : 'They stay here permanently, in the same table and the same type size as the ones that paid.',
        } : {
          head: 'A failed season would stay on this page',
          lead: 'None of yours has failed. If one does it stays here permanently, in the same table and the same type size as the one that paid.',
        } });
        const parts = [];
        // word numbers, to match the heading -- "2 of these seasons" next to
        // "Your three seasons" reads like two different pages
        const w = (n) => WORDS[n] || String(n);
        if (open) parts.push(open === 1 ? 'One of these has not finished'
                                        : `${w(open)[0].toUpperCase()}${w(open).slice(1)} of these have not finished`);
        if (failed) parts.push(failed === 1 ? 'one failed outright' : `${w(failed)} failed outright`);
        const lead = parts.length ? parts.join(' and ') + '. ' : '';
        return lead + 'A single percentage would average them together into a number that describes none of them.';
      })(),
    });

    const tiles = [...root.querySelectorAll('.m')].filter(e => /₹/.test(e.textContent));
    if (tiles[0]) tiles[0].textContent = rupees(putIn);
    if (tiles[1]) tiles[1].textContent = rupees(openMoney);
    if (tiles[2]) tiles[2].textContent = rupees(back);

    const table = root.querySelector('table');
    const body = table?.querySelector('tr')?.parentElement;
    if (!mine.length) {
      return state(table?.parentElement || root, 'empty', 'You have not funded a season yet',
        'When you do, every one you fund stays on this page, including any that fail.',
        { label: 'See what is open', go: 'invest' });
    }

    const header = body.children[0];
    const tpl = body.children[1]?.cloneNode(true);
    tpl?.removeAttribute('data-specimen');      // the clones carry real stakes
    if (!tpl) return;
    body.replaceChildren(header);

    // The table replaces a blank, so the rows arrive rather than appear.
    const built = [];
    for (const inv of mine) {
      const tr = tpl.cloneNode(true);
      const tds = tr.querySelectorAll('td');

      // season, farmer, and how much evidence there is
      const cell0 = tds[0]?.querySelectorAll('div');
      if (cell0?.[0]) cell0[0].textContent = inv.projectTitle || 'Season';
      if (cell0?.[1]) {
        const who = [inv.farmerName, inv.location].filter(Boolean).join(' · ');
        if (who) cell0[1].textContent = who; else cell0[1].remove();
      }
      if (cell0?.[2]) {
        const n = inv.videoCount;
        if (n > 0) cell0[2].textContent = `${n} dated video${n === 1 ? '' : 's'} on the record`;
        else cell0[2].remove();      // no count to state
      }

      if (tds[1]) tds[1].textContent = rupees(inv.amount);

      // the pill and the line under it, which the old loop never touched -- so
      // every row said "Growing / Week 11 of 18" whatever had happened
      const [label, bg, ink, ring] = PILL[inv.status] || PILL.growing;
      const pill = tds[2]?.querySelector('.pill');
      if (pill) {
        pill.textContent = label;
        pill.style.background = bg;
        pill.style.color = ink;
        pill.style.boxShadow = ring;
      }
      const under = tds[2]?.querySelectorAll('div')[1];
      if (under) {
        if (inv.status === 'harvested' && inv.payoutDate) under.textContent = dayMonth(inv.payoutDate);
        // progress is a bare number. "50" under a "Growing" pill says nothing.
        else if (Number.isFinite(inv.progress)) under.textContent = `${inv.progress}% of the way through`;
        else under.remove();         // no week count exists on the server
      }

      // what is being waited on. One sentence per state, or nothing.
      if (tds[3]) {
        const waiting = {
          harvested: 'Nothing. This season is closed.',
          cancelled: 'Nothing. The crop failed and the season is closed.',
          active: 'The season to fill. Your money has not reached the farmer yet and comes back if it does not.',
          growing: `${inv.farmerName || 'The farmer'} to report what the harvest sold for.`,
        }[inv.status] || 'The farmer to report the harvest.';
        tds[3].textContent = waiting;
      }

      if (tds[4]) {
        // A column that is money on one row and words on the next: the drawn
        // cell was styled for "Not yet", so a real payout landed in it grey and
        // without tabular figures, and would not line up under the row above.
        const paid = inv.payoutAmount != null && inv.payoutAmount > 0;
        tds[4].textContent = paid ? rupees(inv.payoutAmount)
          : inv.status === 'cancelled' ? 'Nothing' : 'Not yet';
        tds[4].classList.toggle('m', paid);
        tds[4].style.color = paid ? '#1d1a17' : '#605a53';
        tds[4].style.fontWeight = paid ? '600' : '';
      }

      body.append(tr);
      built.push(tr);
    }
    arrive(built);

    press(root);
  });
}
