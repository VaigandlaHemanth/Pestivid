// Who gets what, from the server's arithmetic rather than the phone's.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeatRows } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('payout', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const id = new URLSearchParams(location.search).get('project');
  if (!id) return state(ctx.root, 'empty', 'No season chosen', 'Open this from the season you are reporting on.',
        { label: 'Back to your seasons', go: 'money' });
  const [project, investments] = await Promise.all([
    api.projects.one(id), api.investments.onProject(id),
  ]);
  const revenue = project.harvestRevenue || 0, costs = project.inputCostBasis || 0;
  const profit = Math.max(0, revenue - costs);
  const share = project.investorShare || 0;
  const pool = investments.reduce((a, i) => a + (i.amount || 0), 0);
  const toInvestors = Math.round(profit * share / 100);

  bind(ctx.root, {
    revenue: rupees(revenue), costs: rupees(costs),
    toInvestors: rupees(toInvestors), youKeep: rupees(profit - toInvestors),
    countLine: `${investments.length} investor${investments.length === 1 ? '' : 's'}`,
  });

  repeatRows(ctx.root, '.p1, .p2', investments.map(inv => ({
    name: inv.investorName || 'Investor',
    // each share is that investor's slice of the pool, not an equal split
    pct: pool ? Math.round(100 * (inv.amount || 0) / pool) : 0,
    amount: pool ? Math.round(toInvestors * (inv.amount || 0) / pool) : 0,
    put: inv.amount || 0,
  })), (el, r) => {
    const [initial] = el.querySelectorAll('.in');
    if (initial) initial.textContent = (r.name[0] || '?').toUpperCase();
    const divs = el.querySelectorAll('div');
    const name = [...divs].find(d => !d.children.length && /^[A-Z]/.test(d.textContent.trim()) && d !== initial);
    if (name) name.textContent = r.name;
    const meta = el.querySelector('.m');
    if (meta) meta.textContent = `Put in ${rupees(r.put)} · ${r.pct}% of the pool`;
    const money = [...el.querySelectorAll('.m')].pop();
    if (money) money.textContent = rupees(r.amount);
  });
});
