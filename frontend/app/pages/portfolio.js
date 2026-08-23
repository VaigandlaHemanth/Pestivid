import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('portfolio', ['investor']);
if (ctx) load(ctx.root, async () => {
  // the nav shows who is signed in, on every desktop page that has one
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();

  const mine = await api.investments.mine(ctx.user._id || ctx.user.id);
  const putIn = mine.reduce((a, i) => a + (i.amount || 0), 0);
  const back = mine.reduce((a, i) => a + (i.paidOut || 0), 0);
  const openMoney = mine.filter(i => !i.settledAt).reduce((a, i) => a + (i.amount || 0), 0);
  bind(ctx.root, { me: { initial }, investor: { name: ctx.user.name }, labels: { putIn: 'You have put in, all together' } });

  const tiles = [...ctx.root.querySelectorAll('.m')].filter(e => /₹/.test(e.textContent));
  if (tiles[0]) tiles[0].textContent = rupees(putIn);
  if (tiles[1]) tiles[1].textContent = rupees(openMoney);
  if (tiles[2]) tiles[2].textContent = rupees(back);

  const table = ctx.root.querySelector('table');
  const body = table?.querySelector('tr')?.parentElement;
  if (!mine.length) {
    return state(ctx.root.querySelector('table')?.parentElement || ctx.root, 'empty', 'You have not funded a season yet',
      'When you do, every one you fund stays on this page — including any that fail.');
  }
  // keep the header row, repeat the first data row
  const header = body.children[0];
  const tpl = body.children[1]?.cloneNode(true);
  if (!tpl) return;
  body.replaceChildren(header);
  for (const inv of mine) {
    const tr = tpl.cloneNode(true);
    const tds = tr.querySelectorAll('td');
    tds[0]?.querySelector('div')?.replaceChildren(document.createTextNode(inv.projectTitle || 'Season'));
    if (tds[1]) tds[1].textContent = rupees(inv.amount);
    if (tds[4]) tds[4].textContent = inv.paidOut != null ? rupees(inv.paidOut) : 'Not yet';
    body.append(tr);
  }
});
