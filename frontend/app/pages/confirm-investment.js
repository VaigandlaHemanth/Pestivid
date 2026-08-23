import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { rupees } from '../api.js';

const ctx = requireUser('confirm-investment', ['investor']);
if (ctx) load(ctx.root, async () => {
  const q = new URLSearchParams(location.search);
  const id = q.get('project'), amount = Number(q.get('amount')) || 0;
  if (!id) return state(ctx.root, 'empty', 'No season chosen', 'Open this from a season you were reading.');
  const p = await api.projects.one(id);
  bind(ctx.root, {
    lot: {
      title: p.title,
      meta: [p.farmerName, p.acres && `${p.acres} acres`, p.crop].filter(Boolean).join(' · '),
      needed: rupees(Math.max(0, (p.amount || 0) - (p.fundedAmount || 0))),
    },
    amount: rupees(amount),
  });

  // The button stays inactive until the loss is acknowledged. That is the point
  // of the screen, so it is enforced here and not merely drawn.
  const box = ctx.root.querySelector('div[style*="inset 0 0 0 2px #a71930"]');
  const btn = [...ctx.root.querySelectorAll('div')]
    .find(d => /^Send /.test(d.textContent.trim()) && !d.children.length);
  const shell = btn?.parentElement;
  let agreed = false;

  const paint = () => {
    if (!shell) return;
    shell.style.background = agreed ? '#016abe' : '#c9ced4';
    if (btn) btn.style.color = agreed ? '#fff' : '#6b7278';
    shell.setAttribute('aria-disabled', String(!agreed));
  };
  box?.setAttribute('data-act', '');
  box?.setAttribute('role', 'checkbox');
  box?.addEventListener('click', () => {
    agreed = !agreed;
    box.setAttribute('aria-checked', String(agreed));
    box.style.background = agreed ? '#a71930' : '#fff';
    paint();
  });
  paint();

  shell?.addEventListener('click', async () => {
    if (!agreed) return;
    try {
      await api.investments.create({ projectId: id, amount });
      location.href = './portfolio.html';
    } catch (err) {
      state(ctx.root, 'failed', 'The money did not move', err.message);
    }
  });
});
