import { requireUser, api, load, state } from './_guard.js';
import { bind, arrive, playsInline, state as showState } from '../bind.js';
import { rupees, dayMonth } from '../api.js';
import { goes, acts, press } from '../wire.js';

// A CSV row separator, named so no editing pass can put a real newline
// inside the string literal -- which is exactly how this file broke once.
const NEWLINE = String.fromCharCode(13, 10);

const ctx = requireUser('orders', ['buyer']);

// The nav and the three "Watch it - check the date" links were painted to read
// unmistakably as links and did nothing: the board carried no data-act and this
// module emitted none, so wire.js never promoted them.
if (ctx) {
  const navItem = (t) => [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === t);
  goes(navItem('Lots for sale'), 'market', 'Lots for sale');
  goes(navItem('Messages'), 'messages', 'Messages');
  press(ctx.root);
}

if (ctx) load(ctx.root, async () => {
  // the nav shows who is signed in, on every desktop page that has one
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();

  const buys = await api.purchases.asBuyer(ctx.user._id || ctx.user.id);
  bind(ctx.root, { me: { initial }, buyer: { line: `${ctx.user.name} · buyer since ${dayMonth(ctx.user.memberSince)}` } });
  // Addressed by name, not by type size. This looked up '29px' and the board's
  // figures became 32px when the KPI cards turned into a ledger band -- so the
  // page went on showing "4 lots bought, 3,86,000 paid" from the artboard while
  // the table below it said the buyer had bought nothing.
  const fig = (n) => ctx.root.querySelector(`[data-fig="${n}"]`);
  const tiles = [fig('count'), fig('paid')];
  if (tiles[0]) tiles[0].textContent = String(buys.length);
  if (tiles[1]) tiles[1].textContent = rupees(buys.reduce((a, p) => a + (p.price || 0), 0));
  // The receipt panel shows the newest purchase. With no purchases it says so,
  // because deClaimProps has already stripped the drawn hash and an em dash
  // where a hash belongs reads as a fault.
  const top = buys[0];
  const short = (v) => (v ? `${String(v).slice(0, 8)}…${String(v).slice(-4)}` : 'not recorded');
  bind(ctx.root, { receipt: top ? {
    head: `Receipt, ${top.crop || 'lot'}, ${dayMonth(top.purchaseDate)}`,
    paid: rupees(top.price),
    tx: short(top.txHash), hash: short(top.videoFileHash), cid: short(top.cid),
    block: top.blockHeight ? `Block ${Number(top.blockHeight).toLocaleString('en-IN')}` : 'Not written into a block yet',
  } : {
    // An em dash where money belongs reads as a fault, which is the same
    // reason the hash rows below say it in words.
    head: 'No receipt yet', paid: 'No purchase yet',
    tx: 'No purchase yet', hash: 'No purchase yet', cid: 'No purchase yet',
    block: 'No purchase yet',
  } });

  // The green tick beside the date is the proved mark. Beside "once its date
  // lands" it says the opposite of the words next to it.
  for (const row of ctx.root.querySelectorAll('tr')) {
    const tick = row.querySelector('svg[stroke="#006934"], svg[stroke="#024c26"]');
    if (tick && !/block\s*\d/.test(row.textContent)) tick.remove();
  }

  if (!buys.length) {
    return state(ctx.root.querySelector('table')?.parentElement || ctx.root, 'empty', 'You have not bought a lot yet',
      'Nothing here yet. When you buy, the receipt and the video it was sold on stay on this page.');
  }
  const body = ctx.root.querySelector('table tr')?.parentElement;
  const header = body.children[0], tpl = body.children[1]?.cloneNode(true);
  tpl?.removeAttribute('data-specimen');        // the clones carry real orders
  if (!tpl) return;
  body.replaceChildren(header);
  // The table replaces a blank, so the rows arrive rather than appear.
  const built = [];
  for (const p of buys) {
    const tr = tpl.cloneNode(true), tds = tr.querySelectorAll('td');
    tds[0]?.querySelector('div')?.replaceChildren(document.createTextNode(p.crop || 'Lot'));
    // The farmer and where the lot came from. Unmarked on the board, so every
    // row said "Alice Farmer · Kadapa" whoever had actually grown it.
    const who = tr.querySelector('[data-who]');
    if (who) {
      const line = [p.farmerName, p.location].filter(Boolean).join(' · ');
      if (line) who.textContent = line; else who.remove();
    }
    tds[1]?.querySelector('.m')?.replaceChildren(document.createTextNode(rupees(p.price)));
    if (tds[2]) tds[2].textContent = dayMonth(p.purchaseDate);
    // the pesticide field is what the farmer typed; blank is blank, not none
    if (tds[4]) tds[4].textContent = p.pesticide
      ? [p.pesticide, p.pesticideCompany].filter(Boolean).join(' · ')
      : 'They left this blank';
    // The range the farmer asked, and what the record can say about the date.
    // Both were the first row's values printed on every row.
    const asked = tr.querySelector('[data-asked]');
    if (asked) {
      if (p.minPrice != null && p.maxPrice != null) {
        asked.textContent = `She asked ${rupees(p.minPrice)}, ${rupees(p.maxPrice)}`;
      } else asked.remove();
    }
    const dated = tr.querySelector('[data-dated]');
    if (dated) {
      /* The short form in the cell, the explanation once above the table.
       * Four rows each carrying "On our server, its date has not landed in a
       * block yet" is eleven words repeated four times, wrapping to two lines
       * every time -- the same thing plots.js and plot.js already avoid by
       * using dateState()'s short form. "Watch it, check the date" underneath
       * is what the row is FOR, and it was competing with the sentence. */
      dated.textContent = p.blockHeight
        ? `Block ${Number(p.blockHeight).toLocaleString('en-IN')}`
        : 'Date being written';
      dated.style.color = p.blockHeight ? '#006934' : '#4a443d';
    }

    /* "Watch it - check the date" IS the row. It is the only reason a receipt in
     * this product is worth more than a receipt anywhere else, and it went to
     * `plot?name=<crop>` -- plot.html, which is gated to farmers. So a buyer
     * pressing the one control that answers "was the thing I paid for real"
     * landed on "Not your screen. This page is for a farmer." Four times, once
     * per lot, on the buyer's own orders page.
     *
     * It plays here instead. The address comes from the public provenance route,
     * fetched on the press rather than four times on load, because most visits to
     * this page are not going to watch anything. If the record has no address, the
     * row says so where the player would have been rather than opening an empty
     * box or going quiet. */
    const watch = tr.querySelector('.act');
    if (watch) {
      if (!p.cid) { watch.remove(); }
      else {
        let player = null;
        acts(watch, `Watch the video for ${p.crop || 'this lot'} and check its date`, async () => {
          if (player) { player.toggle(); return; }
          const was = watch.textContent;
          watch.textContent = 'Fetching the record…';
          try {
            const v = await api.videos.provenance(p.cid);
            watch.textContent = was;
            if (!v?.gateway) throw new Error('This record carries no address for the file.');
            /* A player belongs in a CELL, not after a row. playsInline inserts a
             * div after whatever it is given, and a div after a <tr> inside a
             * <tbody> is not a thing HTML has: the browser hoists it out of the
             * table and the video ends up above the receipt. So the row gets a
             * row of its own, spanning every column, and the player opens against
             * an anchor inside that cell. */
            const holder = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 5;
            cell.style.cssText = 'padding: 0 0 16px;';
            holder.append(cell);
            tr.after(holder);
            const anchor = document.createElement('div');
            cell.append(anchor);
            player = playsInline(anchor, { gateway: v.gateway });
            // aria-expanded belongs on the control somebody presses, not on the
            // empty div the panel is measured against.
            anchor.removeAttribute('aria-expanded');
            watch.setAttribute('aria-expanded', 'true');
            const was2 = player.toggle;
            player.toggle = () => { was2(); watch.setAttribute('aria-expanded',
              watch.getAttribute('aria-expanded') === 'true' ? 'false' : 'true'); };
            player.open?.();
          } catch (err) {
            watch.textContent = was;
            const box = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 5;
            box.append(cell);
            tr.after(box);
            showState(cell, 'failed', 'This one will not play', err.message);
          }
        });
      }
    }
    body.append(tr);
    built.push(tr);
  }
  arrive(built);

  // "Download" is the record of everything held about these purchases. It is
  // built here from what is already on screen -- no route needed, and nothing
  // leaves the browser.
  // Not querySelector('.act'): that is the first "Watch it" link in the table,
  // so the download handler was attached to a row and the button got nothing.
  const dl = [...ctx.root.querySelectorAll('.act')]
    .find(a => a.textContent.trim() === 'Download');
  if (dl) {
    acts(dl, 'Download your purchase record', () => {
      const rows = [['crop', 'price', 'bought', 'pesticide', 'company', 'cid']];
      for (const p of buys) {
        rows.push([p.crop || '', p.price ?? '', p.purchaseDate || '',
                   p.pesticide || '', p.pesticideCompany || '', p.cid || '']);
      }
      const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join(NEWLINE);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
      a.download = 'pestivid-purchases.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    });
  }

  press(ctx.root);
});

// Same as the landing hero: a filename, a hash and a block number drawn as
