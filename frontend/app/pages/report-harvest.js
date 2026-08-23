// The two numbers that decide what four people are paid. The arithmetic shown
// here is the same arithmetic the server will do; it is shown first so a typo
// is caught before it is irreversible.
import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('report-harvest', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const id = new URLSearchParams(location.search).get('project');
  if (!id) return state(ctx.root, 'empty', 'No season chosen', 'Open this from the season you are reporting on.');
  const project = await api.projects.one(id);
  const share = project.investorShare || 0;

  const fields = [...ctx.root.querySelectorAll('[data-bind="revenue"], [data-bind="costs"]')];
  const inputs = fields.map(f => {
    const cs = getComputedStyle(f);
    const i = document.createElement('input');
    i.type = 'text'; i.inputMode = 'numeric'; i.autocomplete = 'off';
    i.value = '';
    i.placeholder = '₹';
    i.style.cssText = `all: unset; display: block; width: 100%; font: ${cs.font}; color: ${cs.color};`;
    f.replaceChildren(i);
    return i;
  });

  const recompute = () => {
    const [rev, cost] = inputs.map(i => Number(String(i.value).replace(/[^\d]/g, '')) || 0);
    const profit = Math.max(0, rev - cost);
    const toInv = Math.round(profit * share / 100);
    bind(ctx.root, { profit: rupees(profit), toInvestors: rupees(toInv), youKeep: rupees(profit - toInv) });
  };
  inputs.forEach(i => i.addEventListener('input', recompute));
  recompute();

  const send = [...ctx.root.querySelectorAll('div')].find(d => d.textContent.trim() === 'Send these numbers');
  send?.setAttribute('data-act', '');
  send?.addEventListener('click', () => {
    const [rev, cost] = inputs.map(i => Number(String(i.value).replace(/[^\d]/g, '')) || 0);
    if (!rev) return state(ctx.root, 'waiting', 'Tell us what you sold it for', 'Without that figure nobody can be paid.');
    location.href = `./payout.html?project=${id}&revenue=${rev}&costs=${cost}`;
  });
});
