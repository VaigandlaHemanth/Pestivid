// The buyer's market: every lot for sale, and one of them open on the right.
//
// The page listed lots and did nothing with them -- zero controls, so the nav,
// every row, the offer field and "Send this offer" were all decoration. The
// detail column showed the artboard's lot whichever row you were looking at,
// because nothing ever swapped it.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat, rows as rows2, slot as slot2 } from '../bind.js';
import { rupees, whenShort, dateState } from '../api.js';
import { promote, goes, acts, asField, press } from '../wire.js';

const ctx = requireUser('market', ['buyer', 'investor']);

if (ctx) {
  const root = ctx.root;
  const navItem = (t) => [...root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === t);
  goes(navItem('My orders'), 'orders', 'My orders');
  goes(navItem('Messages'), 'messages', 'Messages');
  press(root);

  load(root, async () => {
    bind(root, { me: { line: `${ctx.user.name} · ${ctx.user.role}` } });
    const initial = (ctx.user.name || '?').trim()[0].toUpperCase();
    const avatar = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'B');
    if (avatar) avatar.textContent = initial;

    const lots = await api.listings.all();
    // Cheapest first, which is what the heading above the list claims.
    const active = (lots || [])
      .filter(l => (l.status || 'active') === 'active')
      .sort((a, b) => (a.minPrice || 0) - (b.minPrice || 0));

    const list = root.querySelector('[data-list="lots"]');
    // Was closest('div[style*="grid-template-columns"]') from inside the panel.
    // The PAGE is a three-column grid, so on a week with nothing listed that
    // climbed past the panel and removed the whole page -- filters, listings
    // and all -- and then wrote the empty message into a detached node. The
    // market rendered as a nav bar over nothing, and every checker passed it.
    const detail = root.querySelector('[data-detail]');
    const listings = root.querySelector('[data-listings]');

    if (!active.length) {
      detail?.remove();
      root.querySelector('[data-bind="list.sub"]')?.remove();
      bind(root, { list: { title: 'Nothing is for sale this week' } });
      // The rail's drawn counts -- 14 potato, 9 wheat, 6 corn -- are specimens
      // that nobody counted, and the code that replaces them with real ones
      // runs below this return. So they have to go here too, or an empty week
      // shows a filter offering fourteen lots of potato.
      const crops = root.querySelector('[data-crops]');
      if (crops) {
        const none = document.createElement('div');
        none.style.cssText = 'font-size: 14.5px; line-height: 1.5; color: #605a53;';
        none.textContent = 'Nothing to filter yet.';
        crops.replaceChildren(none);
      }
      root.querySelector('[data-sort]')?.remove();
      // Into the listings column, which is still attached.
      return state(list?.parentElement || listings || root, 'empty', 'No lots are for sale',
        'Nobody has produce listed this week. Nothing has gone wrong, and the filters on the '
        + 'left will start counting as soon as somebody lists something.');
    }

    // ---- the filter rail ---------------------------------------------
    // The drawn rail counted 14 potato, 9 wheat and 6 corn lots that nobody had
    // counted, filtered on an organic flag no Listing carries, and had a
    // distance slider over a distance nothing in this product can compute.
    // What survives is what the data supports: crop, and whether there is a
    // dated video.
    const crops = [...new Set(active.map(l => l.crop).filter(Boolean))].sort();
    const picked = new Set();
    let datedOnly = false;

    const TICK = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.4"><path d="M4 12.5 L9.5 18 L20 6.5"></path></svg>';
    const paintBox = (box, on) => {
      if (!box) return;
      box.style.background = on ? '#1d1a17' : 'transparent';
      box.style.boxShadow = on ? 'none' : 'inset 0 0 0 1.5px #b9c2cb';
      box.style.display = 'flex';
      box.style.alignItems = 'center';
      box.style.justifyContent = 'center';
      box.innerHTML = on ? TICK : '';
    };

    rows2(root, 'crop', crops.map(c => ({
      name: c,
      count: String(active.filter(l => l.crop === c).length),
    })), (el, row) => {
      const box = slot2(el, 'box');
      paintBox(box, false);
      acts(el, `${row.name} lots`, () => {
        if (picked.has(row.name)) picked.delete(row.name); else picked.add(row.name);
        paintBox(box, picked.has(row.name));
        apply();
      });
    });

    const dated = root.querySelector('[data-dated]');
    const datedBox = dated?.querySelector('[data-slot="box"]');
    paintBox(datedBox, false);
    if (dated) acts(dated, 'Only lots with a dated video', () => {
      datedOnly = !datedOnly;
      paintBox(datedBox, datedOnly);
      apply();
    });

    let shown = active;
    function apply() {
      shown = active.filter(l =>
        (!picked.size || picked.has(l.crop)) && (!datedOnly || Boolean(l.cid)));
      draw();
    }

    function draw() {
      bind(root, { list: {
        title: picked.size ? [...picked].join(', ') : 'Everything for sale',
        sub: shown.length === active.length
          ? `${active.length} lot${active.length === 1 ? '' : 's'} listed`
          : `${shown.length} of ${active.length} lots`,
      } });
      if (!shown.length) {
        state(list, 'empty', 'Nothing matches that',
          'Loosen a filter and they come back. Nothing has been removed.');
        return;
      }
      paintList();
    }

    function paintList() {
    repeat(list, shown.map(l => ({
      who: l.farmerName || 'Farmer',
      where: l.location || '',
      // A lot is sold whole for an offer inside a range. There is no unit rate,
      // because Purchase has no quantity and dividing by weight would invent one.
      price: `${rupees(l.minPrice)}, ${rupees(l.maxPrice)}`,
      qty: l.crop || '',
      stamp: l.cid ? 'Dated video attached' : 'No video, not listed as proved',
    })));

    const rows = [...(list?.children || [])];
    const select = (i) => rows.forEach((el, n) => {
      el.style.boxShadow = n === i
        ? 'inset 3px 0 0 #016abe, inset 0 -1px 0 #e4e9ee'
        : 'inset 0 -1px 0 #e4e9ee';
      el.setAttribute('aria-current', n === i ? 'true' : 'false');
    });
    rows.forEach((el, i) => {
      promote(el, `${shown[i].crop || 'Lot'} from ${shown[i].farmerName || 'a farmer'}`);
      el.addEventListener('click', () => { select(i); show(shown[i]); });
    });
    select(0);
    show(shown[0]);
    press(root);
    }

    let open = active[0];

    // ---- the offer ---------------------------------------------------
    const offerBox = root.querySelector('[data-offer]');
    const offer = offerBox && asField(offerBox, {
      type: 'text', name: 'offer', inputMode: 'numeric',
      label: 'What you are offering, in rupees', placeholder: '48,000',
    });

    const bar = root.querySelector('[data-range]');
    const inside = root.querySelector('[data-inside]');
    const loLabel = root.querySelector('[data-lo]');
    const hiLabel = root.querySelector('[data-hi]');
    const sendRow = root.querySelector('[data-send]');

    const digits = (s) => Number(String(s || '').replace(/[^\d]/g, '')) || 0;

    const paintOffer = () => {
      const n = digits(offer?.value);
      const lo = open.minPrice || 0, hi = open.maxPrice || 0;
      if (bar && hi > lo) {
        const pct = Math.max(0, Math.min(100, ((n - lo) / (hi - lo)) * 100));
        bar.style.width = pct + '%';
      }
      if (inside) {
        inside.textContent = !n
          ? 'Type what you are willing to pay for the whole lot.'
          : n < lo
            ? `Below her lowest of ${rupees(lo)}. She can still see it, but she is unlikely to accept.`
            : n > hi
              ? `Above her asking price of ${rupees(hi)}. She can accept it, and you may be paying more than you need to.`
              : 'Inside her range, so she can accept it. She is not obliged to, she can refuse or wait for a better offer, and there is no auction clock pushing either of you.';
      }
    };

    function show(l) {
    // Clicking a row replaces this whole column. A browsing investor does that
    // tens of times a day, so the change is announced and not performed: 140ms
    // of opacity, no travel, nothing to wait for.
    const col = ctx.root.querySelector('[data-bind="lot.who"], [data-bind="lot.title"]')
      ?.closest('div[style*="padding"]');
    if (col) {
      col.style.transition = 'opacity 140ms var(--e-smooth, ease)';
      col.style.opacity = '.55';
      requestAnimationFrame(() => { col.style.opacity = '1'; });
    }

      open = l;
      // Reaching the farmer. An investor has had this on their screen all
      // along; a buyer could only bid, with no way to ask anything first.
      const msgRow = root.querySelector('[data-message]');
      if (msgRow) {
        if (l.farmerWallet) {
          promote(msgRow, l.farmerName ? `Message ${l.farmerName}` : 'Message the farmer');
          // assigned, not added: this runs again on every lot click
          msgRow.onclick = async () => {
            try {
              const conv = await api.messages.open({ targetUserId: l.farmerWallet });
              // messages.html, not thread.html -- the two chat screens became
              // one two-pane page and thread was retired. See invest.js, which
              // had the same dangling link to the same removed page.
              location.href = `./messages.html?c=${conv._id || conv.id}`;
            } catch (err) {
              state(msgRow, 'failed', 'Could not open the conversation', err.message);
            }
          };
        } else {
          msgRow.style.display = 'none';
        }
      }
      const proved = l.cid ? dateState({ cid: l.cid, anchored: l.anchored, blockHeight: l.blockHeight }) : null;
      bind(root, {
        lot: {
          who: [l.farmerName, l.location].filter(Boolean).join(' · '),
          ask: l.farmerName ? `Message ${l.farmerName}` : 'Message the farmer',
          accept: `${rupees(l.minPrice)}, ${rupees(l.maxPrice)}`,
          what: l.crop ? `${l.crop}, sold whole` : 'Sold whole',
          grown: l.method ? `${l.method}, farmer's word` : 'Not stated',
          // never a hash the server did not give us
          file: l.cid ? `sha256 ${String(l.videoFileHash || '').slice(0, 8)}…` : 'No video on this lot',
          when: l.createdAt ? whenShort(l.createdAt) : '',
          proved: proved ? proved.text : 'There is no video, so there is nothing dated to check.',
        },
      });
      // the range labels belong to the lot, not to the artboard's lot
      if (loLabel) loLabel.textContent = `${rupees(l.minPrice)} lowest`;
      if (hiLabel) hiLabel.textContent = `${rupees(l.maxPrice)} asking`;
      // seeded and normalised as money -- a bare 28000 in a field labelled
      // "what are you offering" reads like a part number
      if (offer) offer.value = l.minPrice ? Number(l.minPrice).toLocaleString('en-IN') : '';
      paintOffer();
    }

    offer?.addEventListener('input', paintOffer);
    offer?.addEventListener('blur', () => {
      const n = digits(offer.value);
      offer.value = n ? n.toLocaleString('en-IN') : '';
      paintOffer();
    });

    let busy = false;
    acts(sendRow, 'Send this offer', async () => {
      if (busy) return;
      const n = digits(offer?.value);
      const label = sendRow.querySelector('div');
      if (!n) {
        state(sendRow.parentElement, 'waiting', 'Name a figure first',
          'Type what you are willing to pay for the whole lot.');
        return;
      }
      busy = true;
      const was = label?.textContent;
      if (label) label.textContent = 'Sending…';
      try {
        await api.purchases.create({ listingId: open._id || open.id, offerPrice: n });
        if (label) label.textContent = 'Offer sent';
      } catch (err) {
        if (label) label.textContent = was;
        busy = false;
        state(sendRow.parentElement, 'failed', 'The offer did not send', err.message);
      }
    });

    apply();
  });
}
