// Who gets what, from the server's arithmetic rather than the phone's.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeatRows } from '../bind.js';
import { rupees } from '../api.js';
import { appChrome } from '../chrome.js';
import { acts, press } from '../wire.js';

const ctx = requireUser('payout', ['farmer']);
if (ctx) appChrome(ctx.root, { back: 'money', user: ctx.user });
if (ctx) load(ctx.root, async () => {
  const id = new URLSearchParams(location.search).get('project');
  if (!id) {
    // state() keeps the page header, and this one is drawn with a specimen --
    // "Canal plot · nothing sent yet" -- so "No season chosen" arrived under
    // the name of a season.
    bind(ctx.root, { seasonLine: 'None chosen yet' });
    return state(ctx.root, 'empty', 'No season chosen',
      'Open this from the season you are reporting on.',
      { label: 'Back to your seasons', go: 'money' });
  }
  const [project, investments] = await Promise.all([
    api.projects.one(id), api.investments.onProject(id),
  ]);

  // report-harvest hands its two numbers over in the query, because this screen
  // is the CONFIRMATION for them: it shows every person's share before anything
  // is committed. Only when there is neither a reported harvest nor a pair of
  // figures to preview is there nothing to divide.
  const q = new URLSearchParams(location.search);
  const previewRev = Number(q.get('revenue'));
  const previewCost = Number(q.get('costs'));
  // ?failed=1 is a deliberate report of a total loss: nothing sold. Without it,
  // revenue 0 means nobody has said anything yet.
  const declaredLoss = q.get('failed') === '1';
  const preview = !project.harvestReportedAt
    && (declaredLoss || (Number.isFinite(previewRev) && previewRev > 0));

  if (!project.harvestReportedAt && !preview) {
    bind(ctx.root, { seasonLine: `${project.title || 'Season'} · harvest not reported` });
    return state(ctx.root, 'waiting', 'Nothing to divide yet',
      'Tell us what this season sold for and what it cost you, and this screen will show every '
      + 'person’s share before anything is sent.',
      { label: 'Report the harvest', go: `report-harvest?project=${id}` });
  }
  const revenue = preview ? previewRev : (project.harvestRevenue || 0);
  const costs = preview ? (Number.isFinite(previewCost) ? previewCost : 0)
                        : (project.inputCostBasis || 0);
  const profit = Math.max(0, revenue - costs);
  const share = project.investorShare || 0;
  const pool = investments.reduce((a, i) => a + (i.amount || 0), 0);
  const toInvestors = Math.round(profit * share / 100);

  bind(ctx.root, {
    // The sub-line was drawn, not bound, so every payout screen ever rendered
    // said "Canal plot · nothing sent yet" whatever season it was showing.
    seasonLine: `${project.title || 'Season'} · `
      + (project.harvestReportedAt ? 'harvest reported' : 'nothing sent yet'),
    revenue: rupees(revenue), costs: rupees(costs),
    toInvestors: rupees(toInvestors), youKeep: rupees(profit - toInvestors),
    countLine: `${investments.length} investor${investments.length === 1 ? '' : 's'}`,
    // "All four together" was drawn as words and stayed four however many
    // people had actually put money in.
    allLine: investments.length === 1 ? 'That one investor'
      : `All ${investments.length} of them together`,
  });

  repeatRows(ctx.root, '.p1, .p2', investments.map(inv => ({
    name: inv.investorName || 'Investor',
    // each share is that investor's slice of the pool, not an equal split
    pct: pool ? Math.round(100 * (inv.amount || 0) / pool) : 0,
    amount: pool ? Math.round(toInvestors * (inv.amount || 0) / pool) : 0,
    put: inv.amount || 0,
  })), (el, r) => {
    // The chip carries the stake, not an initial. A letter says nothing; the
    // percentage is the reason the figure on the right is the size it is.
    const stake = el.querySelector('.in');
    if (stake) stake.textContent = `${r.pct}%`;
    const name = [...el.querySelectorAll('div')]
      .find(d => !d.children.length && d !== stake && !d.classList.contains('m'));
    if (name) name.textContent = r.name;
    const meta = el.querySelector('.m');
    if (meta) meta.textContent = `Put in ${rupees(r.put)}`;
    const money = [...el.querySelectorAll('.m')].pop();
    if (money) money.textContent = rupees(r.amount);
  });

  // "Send it" was drawn and never wired, so the harvest could not be reported
  // from the interface at all: report-harvest handed its numbers here and the
  // flow stopped at a button that did nothing. click-everything did not catch
  // it because without ?project= this page only ever shows its empty state, so
  // this button was never on the page it enumerated.
  const label = [...ctx.root.querySelectorAll('div')]
    .find(d => !d.children.length && d.textContent.trim() === 'Send it');
  const sendBtn = label?.parentElement;
  if (sendBtn && !project.harvestReportedAt) {
    acts(sendBtn, 'Send it', async () => {
      const was = label.textContent;
      label.textContent = 'Sending…';
      try {
        await api.projects.reportHarvest(id, {
          harvestRevenue: revenue,
          inputCostBasis: costs,
          // FundingRequest.outcome is one of harvested | partial_loss |
          // total_loss. 'profit' and 'loss' are not values it accepts, and the
          // route answered 400 -- written from the enum this time, not guessed.
          outcome: revenue === 0 ? 'total_loss'
            : (revenue > costs ? 'harvested' : 'partial_loss'),
        });
        location.href = './money.html?settled=' + id;
      } catch (err) {
        label.textContent = was;
        // Beside the button, which is where the reader is looking. It used to
        // land next to the investor count, well above the fold.
        let holder = ctx.root.querySelector('[data-sendfail]');
        if (!holder) {
          holder = document.createElement('div');
          holder.setAttribute('data-sendfail', '');
          sendBtn.after(holder);
        }
        state(holder, 'failed', 'That was not accepted', err.message);
      }
    });
  } else if (sendBtn && project.harvestReportedAt) {
    // Already reported. The button must not offer to do it twice.
    sendBtn.remove();
  }

  // Once the harvest is reported this screen stops being a confirmation and
  // becomes the record. It went on saying "This cannot be undone. Once you send
  // it..." about something already sent, and offering to change figures the
  // server will not let anybody change.
  const done = Boolean(project.harvestReportedAt);
  bind(ctx.root, {
    warning: done
      ? 'This was sent on ' + new Date(project.harvestReportedAt)
          .toLocaleDateString('en-IN', { day: 'numeric', month: 'long', year: 'numeric' })
        + '. Each person has been told their amount and the figures cannot be changed here.'
      : undefined,
    backLabel: done ? 'Back to your seasons' : 'Not yet, go back',
  });
  const change = ctx.root.querySelector('[data-change]');
  if (change) {
    if (done) change.remove();
    else acts(change, 'Change these numbers',
      () => { location.href = `./report-harvest.html?project=${id}`; });
  }

  const back = [...ctx.root.querySelectorAll('div')]
    .find(d => !d.children.length && /Not yet, go back|Back to your seasons/.test(d.textContent.trim()));
  if (back) acts(back.parentElement || back, back.textContent.trim(),
    () => { location.href = done ? './money.html' : `./report-harvest.html?project=${id}`; });

  press(ctx.root);
});
