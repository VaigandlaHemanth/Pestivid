import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('money', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const id = ctx.user._id || ctx.user.id;
  const [projects, listings] = await Promise.all([
    api.projects.mine(id).catch(() => []),
    api.listings.mine(id).catch(() => []),
  ]);
  const due = (projects || []).find(p => p.status === 'funded' && !p.harvestReportedAt);
  const raising = (projects || []).find(p => (p.status || 'open') === 'open');
  const settled = (projects || []).find(p => p.harvestReportedAt);

  const investors = settled
    ? Math.round(((settled.harvestRevenue || 0) - (settled.inputCostBasis || 0)) * (settled.investorShare || 0) / 100)
    : null;

  bind(ctx.root, {
    due: { line: due
      ? `Your investors are waiting to be paid on ${due.title}.`
      : 'Nobody is waiting on you.' },
    raise: {
      amount: raising ? rupees(raising.amount) : '—',
      state: raising
        ? `${rupees(raising.fundedAmount || 0)} raised of ${rupees(raising.amount)}`
        : 'You are not raising money',
    },
    lastSeason: {
      investors: investors == null ? '—' : rupees(investors),
      kept: settled ? rupees((settled.harvestRevenue || 0) - investors) : '—',
    },
  });
  if (!projects?.length && !listings?.length) {
    state(ctx.root, 'empty', 'No money has moved yet',
      'When you raise for a season or sell a lot, every figure lands on this page.');
  }
});
