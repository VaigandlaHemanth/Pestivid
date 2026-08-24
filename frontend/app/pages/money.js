// The farmer's money.
//
// This page was contradicting itself on screen. It bound the summary lines and
// left every ROW the artboard drew: it said "No season open / Nothing being
// raised / You are not raising money" and then listed "North plot, 1,80,000,
// still raising, of 5,00,000" directly underneath. It said "No season closed
// yet / Nothing paid out yet / Nothing yet" and then listed three named
// investors with rupee amounts marked "sent".
//
// A summary that disagrees with the rows under it is worse than either alone,
// because the reader cannot tell which is the product and which is the mock.
// Every block now renders from data or is removed, and the board names its own
// parts so the code addresses them by name rather than by position.
import { requireUser, api, load, state } from './_guard.js';
import { bind, rows, slot, dropSection } from '../bind.js';
import { rupees } from '../api.js';
import { appChrome } from '../chrome.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('money', ['farmer']);

if (ctx) {
  const root = ctx.root;
  appChrome(root, { back: 'home', user: ctx.user });
  press(root);

  load(root, async () => {
    const id = ctx.user._id || ctx.user.id;
    const [projects, listings] = await Promise.all([
      api.projects.mine(id).catch(() => []),
      api.listings.mine(id).catch(() => []),
    ]);
    const all = projects || [];
    const lots = listings || [];
    const due = all.find(p => p.status === 'funded' && !p.harvestReportedAt);
    const open = all.filter(p => !p.harvestReportedAt);
    const settled = all.find(p => p.harvestReportedAt);

    // ---- the one thing that needs them today -------------------------
    bind(root, {
      due: {
        headline: due ? `${due.title} is ready to harvest` : 'Nothing needs you today',
        line: due
          ? 'Your investors are waiting to be paid. Tell us what you sold it for and what it cost you to grow.'
          : 'Nobody is waiting on you.',
      },
    });
    const report = [...root.querySelectorAll('div')]
      .find(d => d.textContent.trim() === 'Report the harvest')?.closest('div[style*="background: #fff"]');
    if (due) goes(report, 'report-harvest', 'Report the harvest');
    else report?.remove();

    // ---- money coming in ---------------------------------------------
    // Every season that has not been settled is money still coming in, whether
    // it is part-raised or fully raised and growing.
    if (!open.length) {
      dropSection(root, 'raise', 'raise', 'raise-alt');
    } else {
      root.querySelectorAll('[data-row="raise-alt"]').forEach(n => n.remove());
      rows(root, 'raise', open.map(p => {
        const got = p.fundedAmount || 0;
        const full = got >= (p.amount || 0);
        return {
          name: p.title,
          amount: rupees(p.amount),
          state: full ? 'All of it raised' : `${rupees(got)} raised so far`,
          // A share is a real term of the request. Shown when the request
          // states one, removed when it does not -- the artboard's "they keep
          // 60%" was sitting on a row that also said nothing was being raised.
          share: p.investorShare != null ? `they keep ${p.investorShare}%` : null,
          pct: Math.max(2, Math.min(100, Math.round((got / (p.amount || 1)) * 100))),
          full,
        };
      }), (el, row) => {
        const bar = slot(el, 'bar');
        if (bar) {
          bar.style.width = row.pct + '%';
          // near-black for a season that is fully raised, blue while it is
          // still open: the same two states the artboard drew.
          bar.style.background = row.full ? '#1d1a17' : '#01579b';
        }
        const st = slot(el, 'state');
        // weight without the link colour: it is a state, not somewhere to go
        if (st && !row.full) { st.style.color = '#4a443d'; st.style.fontWeight = '600'; }
      });
    }

    // ---- produce you are selling -------------------------------------
    if (!lots.length) {
      dropSection(root, 'sell', 'sell');
    } else {
      rows(root, 'sell', lots.map(l => ({
        crop: l.crop || l.title || 'Lot',
        price: l.pricePerQuintal != null ? rupees(l.pricePerQuintal) : null,
        per: l.pricePerQuintal != null ? 'per quintal' : null,
        qty: l.quantityQuintal != null ? `${l.quantityQuintal} quintal ready` : null,
        // Nobody has asked is a fact; "2 buyers have asked" because the artboard
        // said so is the same kind of lie as inventing a hash.
        asked: l.enquiryCount > 0
          ? `${l.enquiryCount} buyer${l.enquiryCount === 1 ? ' has' : 's have'} asked about this`
          : null,
      })), (el, row) => {
        if (row.asked) goes(slot(el, 'asked'), 'messages', 'Buyers who asked about this lot');
      });
    }

    // ---- what you paid out last season -------------------------------
    if (!settled) {
      // Nothing has closed, so there is nothing to pay out. Three named
      // investors marked "sent" under "No season closed yet" is the
      // contradiction this page opened with.
      dropSection(root, 'paid', 'payout');
      root.querySelector('[data-bind="lastSeason.name"]')?.parentElement?.remove();
      slot(root, 'investors')?.closest('div[style*="padding"]')?.remove();
    } else {
      const revenue = settled.harvestRevenue || 0;
      const cost = settled.inputCostBasis || 0;
      const pool = Math.max(0, revenue - cost);
      const toInvestors = Math.round(pool * (settled.investorShare || 0) / 100);
      bind(root, {
        lastSeason: {
          name: `${settled.title}, ${new Date(settled.harvestReportedAt)
            .toLocaleDateString('en-IN', { month: 'long', year: 'numeric' })}`,
          investors: rupees(toInvestors),
          kept: rupees(revenue - toInvestors),
          basis: `Worked out from what you told us: ${rupees(revenue)} sold, ${rupees(cost)} spent growing it.`,
        },
      });

      const stakes = (await api.investments.onProject(settled._id).catch(() => [])) || [];
      const put = stakes.reduce((t, s) => t + (s.amount || 0), 0) || 1;
      rows(root, 'payout', stakes.map(s => {
        const name = s.investorName || s.investor?.name || 'An investor';
        return {
          initial: name.trim()[0]?.toUpperCase() || '?',
          name,
          put: `${rupees(s.amount || 0)} put in`,
          amount: rupees(Math.round(toInvestors * (s.amount || 0) / put)),
        };
      }));
    }

    if (!all.length && !lots.length) {
      state(root, 'empty', 'No money has moved yet',
        'When you raise for a season or sell a lot, every figure lands on this page.',
        { label: 'Ask for money', go: 'ask-money' });
    }
    press(root);
  });
}
