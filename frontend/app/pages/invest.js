import { requireUser, api, load, state } from './_guard.js';
import { repeat, bind } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('invest', ['investor']);
if (ctx) load(ctx.root, async () => {
  // the nav shows who is signed in, on every desktop page that has one
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();

  const all = await api.projects.open();
  const open = (all || []).filter(p => (p.status || 'open') !== 'cancelled');
  const list = ctx.root.querySelector('[data-list="projects"]');
  bind(ctx.root, { me: { name: ctx.user.name.split(' ')[0], initial }, openCount: open.length
    ? `${open.length} season${open.length === 1 ? '' : 's'} looking for money`
    : 'No seasons are open right now' });
  if (!open.length) {
    return state(list, 'empty', 'Nothing is open for funding',
      'No farmer is raising money this week. Nothing has gone wrong.');
  }
  // the detail column always shows one project; default to the first open one
  const lot = open[0];
  bind(ctx.root, { lot: {
    season: [lot.crop, lot.timeline && `${lot.timeline} months`].filter(Boolean).join(' · '),
    title: lot.title,
    farmer: lot.farmerName || 'The farmer',
    since: lot.description ? `· ${lot.description.slice(0, 48)}` : '',
    needed: rupees(Math.max(0, (lot.amount || 0) - (lot.fundedAmount || 0))),
    goal: `of ${rupees(lot.amount)}`,
  } });

  repeat(list, open.map(p => ({
    title: p.title,
    meta: [p.acres && `${p.acres} acres`, p.method, p.timeline && `${p.timeline} months`].filter(Boolean).join(' · '),
    raised: rupees(p.fundedAmount || 0),
    goal: `of ${rupees(p.amount)}`,
    pct: Math.min(100, Math.round(100 * (p.fundedAmount || 0) / (p.amount || 1))) + '%',
  })), (el, row) => {
    const bar = el.querySelector('div[style*="background: #01579b"]');
    if (bar) bar.style.width = row.pct;
  });
});
