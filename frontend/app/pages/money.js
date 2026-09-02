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
import { showPoster, bind, rows, slot, dropSection } from '../bind.js';
import { rupees, rupeeRange, whenShort, dayMonth } from '../api.js';
import { appChrome } from '../chrome.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('money');

if (ctx) {
  const root = ctx.root;
  appChrome(root, { back: 'home', user: ctx.user });
  press(root);

  load(root, async () => {
    const id = ctx.user._id || ctx.user.id;
    const [projects, listings, stakes, bought, ledger] = await Promise.all([
      api.projects.mine(id).catch(() => []),
      api.listings.mine(id).catch(() => []),
      // One account grows, funds and buys, so the page reads all three.
      api.investments.mine(id).catch(() => []),
      api.purchases.asBuyer(id).catch(() => []),
      api.money.transactions(id).catch(() => []),
    ]);
    const all = projects || [];
    const lots = listings || [];
    const due = all.find(p => p.status === 'funded' && !p.harvestReportedAt);
    const open = all.filter(p => !p.harvestReportedAt);
    const settled = all.find(p => p.harvestReportedAt);

    // ---- the one thing that needs them today -------------------------
    /* No season waiting means no notice. It is drawn on --attention-fill, which
     * this palette keeps for "take care", and a box in that colour saying
     * "Nothing needs you today" is a warning about nothing -- the surest way to
     * teach somebody to stop reading the warnings. The row simply is not there. */
    const band = root.querySelector('[data-todoband]');
    if (!due) band?.remove();
    else {
      bind(root, {
        due: {
          headline: `${due.title} is ready to harvest`,
          // Shorter than the banner's line was: this is a notice beside a button,
          // not a hero paragraph. What it costs to get wrong -- that it can only
          // be done once -- stays.
          line: 'Your investors are waiting to be paid. You can only do this once.',
        },
      });
    }
    // Found by its mark. This searched for a div whose inline style contained
    // `background: #fff`, and the button stopped being white the moment the two
    // "Report the harvest" buttons were made one button -- so it silently found
    // nothing, and neither the wiring nor the removal happened.
    const report = root.querySelector('[data-report]')
      || [...root.querySelectorAll('div')]
        .find(d => d.textContent.trim() === 'Report the harvest')?.parentElement;
    if (due) goes(report, `report-harvest?project=${due._id}`, 'Report the harvest');
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
          // "they keep 20%" on the farmer's OWN money page: the row is a season
          // title, an amount and a share, and there is no "they" anywhere in it
          // to point at. On the one screen about who gets paid what, the pronoun
          // has to name somebody.
          share: p.investorShare != null ? `investors keep ${p.investorShare}%` : null,
          // The 2% floor exists so a tiny amount is still visible. It must not
          // apply to nothing: a blue nub on a season with zero raised claims
          // money arrived. Zero is drawn as zero.
          pct: got > 0
            ? Math.max(2, Math.min(100, Math.round((got / (p.amount || 1)) * 100)))
            : 0,
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
      /* THE HEADING HAS TO SURVIVE THE DATA UNDER IT.
       *
       * "Produce you are selling" is drawn once and was never rebound, so a
       * farmer with four lots of which three had already sold read a heading
       * saying all four were on sale. Each row does say "Sold", which makes the
       * heading the only untrue line in the section -- and it is the line in the
       * largest type. */
      const onSale = lots.filter(l => String(l.status || '') !== 'sold').length;
      bind(root, { sell: { title:
        onSale === lots.length ? 'Produce you are selling'
          : onSale === 0 ? 'Produce you have sold'
          : `Produce you listed · ${onSale} still for sale` } });
      rows(root, 'sell', lots.map(l => ({
        crop: l.crop || l.title || 'Lot',
        // A Listing carries minPrice and maxPrice for the whole lot. It has no
        // pricePerQuintal and no quantityQuintal -- the quantity field is
        // commented out in the model -- so every one of these slots was read as
        // null and REMOVED, and the rows showed a crop name and nothing else.
        price: l.minPrice != null && l.maxPrice != null
          ? rupeeRange(l.minPrice, l.maxPrice)
          : (l.minPrice != null ? rupees(l.minPrice) : null),
        per: l.minPrice != null ? 'for the whole lot' : null,
        qty: l.status === 'sold' ? 'Sold' : (l.createdAt ? `Listed ${whenShort(l.createdAt)}` : null),
        // Nobody has asked is a fact; "2 buyers have asked" because the artboard
        // said so is the same kind of lie as inventing a hash.
        asked: l.enquiryCount > 0
          ? `${l.enquiryCount} buyer${l.enquiryCount === 1 ? ' has' : 's have'} asked about this`
          : null,
      })), (el, row, i) => {
        if (row.asked) goes(slot(el, 'asked'), 'messages', 'Buyers who asked about this lot');
        // The 62px square beside each lot was drawn to hold a frame of the video
        // the lot was listed with, and held a flat fill instead. The listings
        // route carries the frame now, cut server-side from the stored object.
        showPoster(el.querySelector('[data-lotthumb]'), lots[i]);
      });
    }

    // ---- seasons you funded ------------------------------------------
    // The portfolio page's rows, here in brief: this page is the one answer to
    // where the money is, and money you put into somebody else's season is
    // as much yours as money raised on your own.
    const funded = (stakes || []).filter(s => s && s.status !== 'cancelled' || (s && s.amount));
    if (!funded.length) {
      dropSection(root, 'funded', 'funded');
      root.querySelector('[data-more="portfolio"]')?.remove();
    } else {
      rows(root, 'funded', funded.map(s => {
        const paid = s.payoutAmount != null && s.payoutAmount > 0;
        return {
          name: s.projectTitle || 'A season',
          amount: rupees(s.amount || 0),
          state: s.status === 'harvested' ? `Paid out${s.payoutDate ? ' · ' + dayMonth(s.payoutDate) : ''}`
            : s.status === 'cancelled' ? 'Failed'
            : `Growing${Number.isFinite(s.progress) ? ` · ${s.progress}% of the way through` : ''}`,
          back: paid ? `${rupees(s.payoutAmount)} back` : s.status === 'cancelled' ? 'nothing back' : 'nothing back yet',
          who: s.farmerName ? `${s.farmerName}’s season` : null,
        };
      }));
      // drawn as specimens until this moment: nobody knew whether there was anything to show
      for (const el of root.querySelectorAll('[data-sec="funded"], [data-more="portfolio"]')) el.removeAttribute('data-specimen');
      goes(root.querySelector('[data-more="portfolio"]'), 'portfolio', 'Every season you funded');
    }

    // ---- lots you bought ---------------------------------------------
    const buys = bought || [];
    if (!buys.length) {
      dropSection(root, 'bought', 'bought');
      root.querySelector('[data-more="orders"]')?.remove();
    } else {
      rows(root, 'bought', buys.map(p => ({
        crop: p.crop || 'A lot',
        price: rupees(p.price || 0),
        when: p.purchaseDate ? `Bought ${dayMonth(p.purchaseDate)}` : null,
        from: p.farmerName ? `from ${p.farmerName}` : null,
      })), (el, row, i) => showPoster(el.querySelector('[data-boughtthumb]'), buys[i]));
      for (const el of root.querySelectorAll('[data-sec="bought"], [data-more="orders"]')) el.removeAttribute('data-specimen');
      goes(root.querySelector('[data-more="orders"]'), 'orders', 'Every lot you bought');
    }

    // ---- all together ------------------------------------------------
    // The server's own transaction ledger, both directions. It is the one
    // list that already knows every rupee that moved, whichever screen moved it.
    const tx = ledger || [];
    const sum = (types) => tx.filter(t => types.includes(t.type)).reduce((a, t) => a + (t.amount || 0), 0);
    const stillOut = funded.filter(s => s.status !== 'harvested' && s.status !== 'cancelled')
      .reduce((a, s) => a + (s.amount || 0), 0);
    if (!tx.length) {
      root.querySelectorAll('[data-totals]').forEach(el => el.remove());
    } else {
      bind(root, { totals: {
        in: rupees(sum(['sale', 'payout'])),
        out: rupees(sum(['purchase', 'investment'])),
        open: rupees(stillOut),
      } });
      root.querySelectorAll('[data-totals]').forEach(el => el.removeAttribute('data-specimen'));
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

    if (!all.length && !lots.length && !funded.length && !buys.length) {
      state(root, 'empty', 'No money has moved yet',
        'When you raise for a season, sell a lot, fund somebody else’s season or buy '
        + 'their produce, every figure lands on this page.',
        { label: 'See the market', go: 'market' });
    }
    // With no settled season the rail is empty, and a 372px column of nothing
    // beside the work is worse than no column. The main takes the width.
    const grid = root.querySelector('div[style*="grid-template-columns: 1fr 372px"]');
    const rail = grid?.lastElementChild;
    if (grid && rail && !rail.textContent.trim()) {
      rail.remove();
      grid.style.gridTemplateColumns = 'minmax(0, 1fr)';
    }

    press(root);
  });
}
