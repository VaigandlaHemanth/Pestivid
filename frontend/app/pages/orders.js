import { requireUser, api, load, state } from './_guard.js';
import { bind, arrive } from '../bind.js';
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
      dated.textContent = p.blockHeight
        ? `Block ${Number(p.blockHeight).toLocaleString('en-IN')} · you can check this yourself`
        : 'On our server · its date has not landed in a block yet';
      dated.style.color = p.blockHeight ? '#006934' : '#4a443d';
    }

    // "Watch it - check the date" is the whole point of this row: it is how a
    // buyer verifies the lot they paid for. It goes to the provenance record,
    // and where there is no video it is removed rather than left to be pressed.
    const watch = tr.querySelector('.act');
    if (watch) {
      if (p.cid) goes(watch, `plot?name=${encodeURIComponent(p.crop || '')}`,
                      `Watch the video for ${p.crop || 'this lot'} and check its date`);
      else watch.remove();
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
