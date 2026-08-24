// The two numbers that decide what the people who funded this season are paid.
// The arithmetic shown here is the same arithmetic the server will do; it is
// shown first so a typo is caught before it is irreversible.
//
// Three figures on this screen were the artboard's and were never bound: "your
// four investors", "your investors get 60%", and "plus the ₹5,00,000 they
// already gave you". On a season with one investor, a 15% share and ₹3,00,000
// raised, all three were wrong -- on the one screen in the product a farmer
// cannot take back.
import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { asField, acts, press } from '../wire.js';
import { rupees } from '../api.js';

const ctx = requireUser('report-harvest', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const id = new URLSearchParams(location.search).get('project');
  if (!id) return state(ctx.root, 'empty', 'No season chosen', 'Open this from the season you are reporting on.',
        { label: 'Back to your seasons', go: 'money' });
  const [project, investors] = await Promise.all([
    api.projects.one(id),
    api.investments.onProject(id).catch(() => []),
  ]);
  const share = project.investorShare || 0;
  const n = investors.length;
  bind(ctx.root, {
    whoIsPaid: n
      ? `These two numbers decide what your ${n === 1 ? 'one investor is' : n + ' investors are'} paid.`
      : 'These two numbers decide what anyone who funded this season is paid.',
    shareLine: `Your investors get ${share}%`,
    principal: rupees(project.fundedAmount || project.amount || 0),
  });

  const fields = [...ctx.root.querySelectorAll('[data-bind="revenue"], [data-bind="costs"]')];
  const inputs = fields.map((f, i) => asField(f, {
    // A bare ₹ is not a specimen of anything.
    inputMode: 'numeric', autocomplete: 'off',
    placeholder: i === 0 ? '9,32,000' : '2,58,400',
    label: i === 0 ? 'What did you sell it for' : 'What did it cost you to grow',
  }));

  const recompute = () => {
    const [rev, cost] = inputs.map(i => Number(String(i.value).replace(/[^\d]/g, '')) || 0);
    const profit = Math.max(0, rev - cost);
    const toInv = Math.round(profit * share / 100);
    bind(ctx.root, { profit: rupees(profit), toInvestors: rupees(toInv), youKeep: rupees(profit - toInv) });
  };
  inputs.forEach(i => i.addEventListener('input', recompute));
  recompute();

  // Crop failure. The route has always accepted total_loss; this screen had no
  // way to say it, and its own gate refused to move without a sale figure -- so
  // the one outcome the whole product warns about could not be reported.
  const failRow = ctx.root.querySelector('[data-failed]');
  const failBox = ctx.root.querySelector('[data-failbox]');
  let failed = false;
  const paintFail = () => {
    if (failBox) {
      failBox.style.background = failed ? '#1d1a17' : 'transparent';
      failBox.style.boxShadow = failed ? 'inset 0 0 0 2px #1d1a17' : 'inset 0 0 0 2px #78716a';
    }
    failRow?.setAttribute('aria-checked', String(failed));
    // With nothing sold there is nothing to type, so the boxes stop asking.
    inputs.forEach((i) => {
      if (!i) return;
      i.disabled = failed && i === inputs[0];
      if (failed && i === inputs[0]) i.value = '';
    });
    recompute();
  };
  if (failRow) {
    failRow.setAttribute('role', 'checkbox');
    acts(failRow, 'The crop failed, there was nothing to sell', () => {
      failed = !failed;
      paintFail();
    });
    paintFail();
  }

  const send = [...ctx.root.querySelectorAll('div')].find(d => d.textContent.trim() === 'Send these numbers');
  send?.setAttribute('data-act', '');
  send?.addEventListener('click', () => {
    const [rev, cost] = inputs.map(i => Number(String(i.value).replace(/[^\d]/g, '')) || 0);
    if (failed) {
      location.href = `./payout.html?project=${id}&revenue=0&costs=${cost}&failed=1`;
      return;
    }
    if (!rev) {
      return state(ctx.root, 'waiting', 'Tell us what you sold it for',
        'Without that figure nobody can be paid. If the crop failed and there was nothing to sell, '
        + 'tick the line above instead.');
    }
    location.href = `./payout.html?project=${id}&revenue=${rev}&costs=${cost}`;
  });

  press(ctx.root);
});
