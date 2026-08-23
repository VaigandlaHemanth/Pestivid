import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { rupees, dayMonth } from '../api.js';

const ctx = requireUser('orders', ['buyer']);
if (ctx) load(ctx.root, async () => {
  // the nav shows who is signed in, on every desktop page that has one
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();

  const buys = await api.purchases.asBuyer(ctx.user._id || ctx.user.id);
  bind(ctx.root, { me: { initial }, buyer: { line: `${ctx.user.name} · buyer since ${dayMonth(ctx.user.memberSince)}` } });
  const tiles = [...ctx.root.querySelectorAll('.m')].filter(e => e.style.fontSize === '29px');
  if (tiles[0]) tiles[0].textContent = String(buys.length);
  if (tiles[1]) tiles[1].textContent = rupees(buys.reduce((a, p) => a + (p.price || 0), 0));
  if (!buys.length) {
    return state(ctx.root.querySelector('table')?.parentElement || ctx.root, 'empty', 'You have not bought a lot yet',
      'Nothing here yet. When you buy, the receipt and the video it was sold on stay on this page.');
  }
  const body = ctx.root.querySelector('table tr')?.parentElement;
  const header = body.children[0], tpl = body.children[1]?.cloneNode(true);
  if (!tpl) return;
  body.replaceChildren(header);
  for (const p of buys) {
    const tr = tpl.cloneNode(true), tds = tr.querySelectorAll('td');
    tds[0]?.querySelector('div')?.replaceChildren(document.createTextNode(p.crop || 'Lot'));
    tds[1]?.querySelector('.m')?.replaceChildren(document.createTextNode(rupees(p.price)));
    if (tds[2]) tds[2].textContent = dayMonth(p.purchaseDate);
    // the pesticide field is what the farmer typed; blank is blank, not none
    if (tds[4]) tds[4].textContent = p.pesticide
      ? [p.pesticide, p.pesticideCompany].filter(Boolean).join(' · ')
      : 'They left this blank';
    body.append(tr);
  }
});
