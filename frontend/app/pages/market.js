import { requireUser, api, load, state } from './_guard.js';
import { rupees } from '../api.js';

const ctx = requireUser('market', ['buyer', 'investor']);
if (ctx) load(ctx.root, async () => {
  // the nav shows who is signed in, on every desktop page that has one
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();
  const { bind } = await import('../bind.js');
  bind(ctx.root, { me: { initial } });

  const lots = await api.listings.all();
  const active = (lots || []).filter(l => l.status === 'active');
  if (!active.length) {
    return state(ctx.root, 'empty', 'No lots are for sale',
      'Nobody has produce listed this week.');
  }
  const first = ctx.root.querySelector('.lift1');
  const container = first?.parentElement;
  const { repeat } = await import('../bind.js');
  repeat(container, active.map(l => ({
    who: l.farmerName || 'Farmer', where: l.location || '',
    // a lot is sold whole for an offer inside a range; there is no unit rate
    price: `${rupees(l.minPrice)}–${rupees(l.maxPrice)}`,
    qty: l.crop || '', dur: '', stamp: l.cid ? 'Dated video attached' : 'No video — not listed as proved',
  })));
});
