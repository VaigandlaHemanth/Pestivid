// The two numbers that decide what the people who funded this season are paid,
// and — since payout.html was merged into this screen — who gets what, and the
// send itself.
//
// The arithmetic shown here is the same arithmetic the server will do; it is
// shown first so a typo is caught before it is irreversible.
//
// Three figures on this screen were the artboard's and were never bound: "your
// four investors", "your investors get 60%", and "plus the ₹5,00,000 they
// already gave you". On a season with one investor, a 15% share and ₹3,00,000
// raised, all three were wrong -- on the one screen in the product a farmer
// cannot take back.
import { requireUser, api, load, state } from './_guard.js';
import { bind, rows } from '../bind.js';
import { asField, acts, press, digitsOnly } from '../wire.js';
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
  const done = Boolean(project.harvestReportedAt);
  const pool = investors.reduce((a, i) => a + (i.amount || 0), 0);
  bind(ctx.root, {
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

  const read = () => inputs.map(i => Number(digitsOnly(i?.value || '')) || 0);

  /* Who gets what. This was the entire content of the second screen: each
   * person's slice of the POOL, not an equal split.
   *
   * Built ONCE. rows() clones a template, fills it and removes the template,
   * and it ends in arrive() -- so calling it on every keystroke would re-run the
   * staged entrance while somebody is still typing. Only the amount changes as
   * the numbers change, so only the amount is rewritten.
   */
  const payeeRows = rows(ctx.root, 'payee', investors.map(inv => ({
    pct: (pool ? Math.round(100 * (inv.amount || 0) / pool) : 0) + '%',
    name: inv.investorName || inv.investor?.name || 'An investor',
    put: `Put in ${rupees(inv.amount || 0)}`,
    amount: rupees(0),
  })));
  if (!investors.length) ctx.root.querySelector('[data-sec="payees"]')?.remove();

  const recompute = () => {
    const [rev, cost] = read();
    // A failed crop sold for nothing, whatever is still typed in the box.
    const profit = Math.max(0, (failed ? 0 : rev) - cost);
    const toInv = Math.round(profit * share / 100);
    bind(ctx.root, { profit: rupees(profit), toInvestors: rupees(toInv), youKeep: rupees(profit - toInv) });
    payeeRows.forEach((el, i) => {
      const put = investors[i]?.amount || 0;
      const cell = el.querySelector('[data-slot="amount"]');
      if (cell) cell.textContent = rupees(pool ? Math.round(toInv * put / pool) : 0);
    });
  };

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
      // A filled square with nothing in it does not read as "ticked" -- the same
      // thing the create-account box was fixed for, and this one still had it.
      failBox.innerHTML = failed
        ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff"'
          + ' stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"'
          + ' style="display: block; margin: 3px" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7"></path></svg>'
        : '';
    }
    failRow?.setAttribute('aria-checked', String(failed));
    /* The box stops asking, and REMEMBERS.
     *
     * It used to do `i.value = ''` on tick, so ticking the line threw away the
     * figure that had just been typed and unticking gave back an empty box. A
     * farmer who ticked it to read the explanation lost their work. Disabled is
     * enough to say "not being asked for"; the arithmetic already treats a
     * failed crop as nothing sold, so the value does not need to be destroyed
     * to be ignored. */
    if (inputs[0]) inputs[0].disabled = failed;
    recompute();
  };
  if (failRow) {
    failRow.setAttribute('role', 'checkbox');
    acts(failRow, 'The crop failed, there was nothing to sell', () => {
      failed = !failed;
      paintFail();
    });
  }
  // Typing changes every figure below, so typing recomputes them. groupLive
  // adds its own listener for the commas; this is the arithmetic.
  inputs.forEach(i => i?.addEventListener('input', recompute));
  if (failRow) paintFail(); else recompute();

  /* ---- sending, which used to be a second screen ---------------------
   * report-harvest handed its two numbers to payout.html in the query string
   * and payout did the POST. payout showed those same two numbers again plus
   * the per-investor split; the split is on this screen now, so the second
   * screen had nothing left to add and a click on a page that repeats the
   * previous one is not a safeguard.
   */
  const sendBtn = ctx.root.querySelector('[data-send]');
  const label = sendBtn?.firstElementChild;

  if (done) {
    // Already reported. The form is a record now: it must not offer to do it
    // again, and it must not imply the figures can still be changed.
    const when = new Date(project.harvestReportedAt)
      .toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' });
    if (inputs[0]) inputs[0].value = (project.harvestRevenue || 0).toLocaleString('en-IN');
    if (inputs[1]) inputs[1].value = (project.inputCostBasis || 0).toLocaleString('en-IN');
    inputs.forEach(i => { if (i) i.disabled = true; });
    failRow?.remove();
    sendBtn?.remove();
    recompute();
    bind(ctx.root, { warning: `This was sent on ${when}. Each person has been told their `
      + 'amount and the figures cannot be changed here.' });
    const t = ctx.root.querySelector('[data-title]');
    if (t) t.textContent = 'What you reported';
  } else if (sendBtn) {
    acts(sendBtn, 'Send these numbers', async () => {
      const [rev, cost] = read();
      if (!failed && !rev) {
        return state(ctx.root, 'waiting', 'Tell us what you sold it for',
          'Without that figure nobody can be paid. If the crop failed and there was nothing to sell, '
          + 'tick the line above instead.');
      }
      const revenue = failed ? 0 : rev;
      const was = label ? label.textContent : '';
      if (label) label.textContent = 'Sending…';
      sendBtn.setAttribute('aria-disabled', 'true');
      try {
        await api.projects.reportHarvest(id, {
          harvestRevenue: revenue,
          inputCostBasis: cost,
          // FundingRequest.outcome is one of harvested | partial_loss |
          // total_loss. 'profit' and 'loss' are not values it accepts, and the
          // route answered 400 -- written from the enum this time, not guessed.
          outcome: revenue === 0 ? 'total_loss'
            : (revenue > cost ? 'harvested' : 'partial_loss'),
        });
        location.href = './money.html?settled=' + id;
      } catch (err) {
        if (label) label.textContent = was;
        sendBtn.removeAttribute('aria-disabled');
        // Beside the button, which is where the reader is looking.
        let holder = ctx.root.querySelector('[data-sendfail]');
        if (!holder) {
          holder = document.createElement('div');
          holder.setAttribute('data-sendfail', '');
          sendBtn.after(holder);
        }
        state(holder, 'failed', 'That was not accepted', err.message);
      }
    });
  }

  press(ctx.root);
});
