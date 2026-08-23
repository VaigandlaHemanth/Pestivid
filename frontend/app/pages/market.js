import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('market', ['buyer', 'investor']);
if (ctx) load(ctx.root, async () => {
  bind(ctx.root, { me: { line: `${ctx.user.name} · ${ctx.user.role}` } });
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();
  const avatar = [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'B');
  if (avatar) avatar.textContent = initial;

  const lots = await api.listings.all();
  const active = (lots || []).filter(l => (l.status || 'active') === 'active');
  // the sc-for is marked in the artboard, so the row template is unambiguous
  const list = ctx.root.querySelector('[data-list="lots"]');
  if (!active.length) {
    return state(list || ctx.root, 'empty', 'No lots are for sale',
      'Nobody has produce listed this week.');
  }
  repeat(list, active.map(l => ({
    who: l.farmerName || 'Farmer',
    where: l.location || '',
    // A lot is sold whole for an offer inside a range. There is no unit rate,
    // because Purchase has no quantity and dividing by weight would invent one.
    price: `${rupees(l.minPrice)}–${rupees(l.maxPrice)}`,
    qty: l.crop || '',
    stamp: l.cid ? 'Dated video attached' : 'No video — not listed as proved',
  })));
});
