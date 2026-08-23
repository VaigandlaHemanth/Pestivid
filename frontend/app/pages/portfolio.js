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
import { bind } from '../bind.js';
import { rupees, dayMonth } from '../api.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('portfolio', ['investor']);

// The four states the API actually reports, and the pill each one wears. No
// green anywhere: whether money reached your bank is a thing you take our word
// for, and green in this product means a fact you can check without us.
const PILL = {
  harvested: ['Paid out',     '#eae4de', '#1d1a17', 'inset 0 0 0 1.5px #1d1a17'],
  growing:   ['Growing',      '#eae4de', '#4a443d', 'none'],
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
      heading: mine.length === 1 ? 'Your one season'
             : `Your ${WORDS[mine.length] || mine.length} seasons`,
      investor: { name: ctx.user.name },
      labels: { putIn: 'You have put in, all together' },
      // The note described the artboard's mix -- "two have not finished and one
      // failed outright" -- whatever the real one was.
      noReturn: (() => {
        const open = mine.filter(i => !done(i)).length;
        const failed = mine.filter(i => i.status === 'cancelled').length;
        const parts = [];
        // word numbers, to match the heading -- "2 of these seasons" next to
        // "Your three seasons" reads like two different pages
        const w = (n) => WORDS[n] || String(n);
        if (open) parts.push(open === 1 ? 'One of these seasons has not finished'
                                        : `${w(open)[0].toUpperCase()}${w(open).slice(1)} of these seasons have not finished`);
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
        'When you do, every one you fund stays on this page — including any that fail.',
        { label: 'See what is open', go: 'invest' });
    }

    const header = body.children[0];
    const tpl = body.children[1]?.cloneNode(true);
    if (!tpl) return;
    body.replaceChildren(header);

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
        tds[4].textContent = inv.payoutAmount != null && inv.payoutAmount > 0
          ? rupees(inv.payoutAmount)
          : inv.status === 'cancelled' ? 'Nothing' : 'Not yet';
      }

      body.append(tr);
    }

    press(root);
  });
}
