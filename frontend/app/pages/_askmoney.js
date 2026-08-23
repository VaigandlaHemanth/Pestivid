// Asking for money, across three screens.
//
// FundingRequest refuses to save without a cid, so the video is question one --
// which is also the honest ordering, because it is the only thing on the page
// an investor can check without trusting the farmer.
//
// The answers are held in sessionStorage between the three steps and posted
// once, at the end. Nothing is created halfway.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat } from '../bind.js';
import { rupees, whenShort, dateState } from '../api.js';
import { asField } from '../wire.js';

const KEY = 'pv.ask';
const draft = {
  get() { try { return JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch { return {}; } },
  put(patch) { sessionStorage.setItem(KEY, JSON.stringify({ ...draft.get(), ...patch })); },
  clear() { sessionStorage.removeItem(KEY); },
};

const byLabel = (root, t) => [...root.querySelectorAll('div')]
  .find(d => d.children.length === 0 && d.textContent.trim() === t);

const chips = (root, cls, onPick) => {
  const list = [...root.querySelectorAll(`.${cls}, .${cls}On`)];
  list.forEach((el) => {
    el.setAttribute('data-act', '');
    el.setAttribute('role', 'radio');
    el.tabIndex = 0;
    el.addEventListener('click', () => {
      list.forEach(o => { o.className = cls; o.setAttribute('aria-checked', 'false'); });
      el.className = cls + 'On';
      el.setAttribute('aria-checked', 'true');
      onPick(el.textContent.trim(), el);
    });
  });
  return list;
};

// ── step 1: which video ─────────────────────────────────────────────────────
export function stepVideo() {
  const ctx = requireUser('ask-money-video', ['farmer']);
  if (!ctx) return;
  load(ctx.root, async () => {
    const vids = await api.videos.mine(ctx.user._id || ctx.user.id);
    bind(ctx.root, { step1: { hint: vids.length
      ? 'Pick a video of it. You cannot ask for money without one.'
      : 'You have no videos yet, and money cannot be asked for without one.' } });
    const rows = [...ctx.root.querySelectorAll('div[style*="background: #f6f3ef"][style*="display: flex"]')];
    const container = rows[0]?.parentElement;
    if (!vids.length) {
      return state(container || ctx.root, 'empty', 'Film the field first',
        'One forty-second walk across it is enough. Come back here afterwards.');
    }
    // The bar is the server's, not ours: hashed by us and fingerprinted.
    // See services/videoEligibility.js -- the Bitcoin date arrives afterwards
    // and is not what a funding request is checked against.
    repeat(container, vids.map(v => {
      const s = dateState(v);
      const usable = v.hashComputedBy === 'server' && v.fingerprinted;
      return {
        label: `${v.crop || v.location || 'Plot'} · ${whenShort(v.uploadTimestamp)}`,
        status: usable ? s.text : 'We are still checking this one',
        ready: usable, proved: s.kind === 'proved', v,
      };
    }), (el, r) => {
      const m = el.querySelector('.m');
      if (m) m.textContent = r.label;
      const status = [...el.querySelectorAll('div')].reverse()
        .find(d => !d.children.length && d.textContent.trim() && d !== m);
      if (status) {
        status.textContent = r.status;
        status.style.color = r.proved ? '#006934' : r.ready ? '#4a443d' : '#78716a';
      }
      // A video we have not finished checking is shown, not hidden, with why
      if (!r.ready) { el.style.opacity = '.62'; return; }
      el.setAttribute('data-act', '');
      el.addEventListener('click', () => {
        draft.put({ cid: r.v.cid, crop: r.v.crop, location: r.v.location });
        [...container.children].forEach(c => { c.style.boxShadow = ''; });
        el.style.boxShadow = 'inset 0 0 0 2px #1d1a17';
      });
    });

    // crop and acres, the two words beside the video
    const cropBox = byLabel(ctx.root, 'Potato');
    const acreBox = byLabel(ctx.root, '2 acres');
    const crop = asField(cropBox, { placeholder: 'Potato', label: 'What are you growing' });
    const acres = asField(acreBox, { placeholder: '2', inputMode: 'decimal', label: 'How much land, in acres' });
    crop?.addEventListener('input', () => draft.put({ crop: crop.value.trim() }));
    acres?.addEventListener('input', () => draft.put({ acres: Number(acres.value) || 0 }));

    const next = byLabel(ctx.root, 'Next');
    next?.setAttribute('data-act', '');
    next?.addEventListener('click', () => {
      const d = draft.get();
      if (!d.cid) return state(ctx.root, 'waiting', 'Pick a video first',
        'The server will not accept a request without one, and neither will an investor.');
      location.href = './ask-money-amount.html';
    });
  });
}

// ── step 2: how much, how long, how grown ───────────────────────────────────
export function stepAmount() {
  const ctx = requireUser('ask-money-amount', ['farmer']);
  if (!ctx) return;
  load(ctx.root, async () => {
    const d = draft.get();
    if (!d.cid) { location.replace('./ask-money-video.html'); return; }
    bind(ctx.root, { amount: d.amount ? rupees(d.amount) : '' });

    const box = ctx.root.querySelector('[data-bind="amount"]');
    const cs = box && getComputedStyle(box);
    const amountField = asField(box, { inputMode: 'numeric', placeholder: '₹',
      value: d.amount || '', label: 'How much you need' });
    amountField?.addEventListener('input', () =>
      draft.put({ amount: Number(String(amountField.value).replace(/[^\d]/g, '')) || 0 }));

    // the growing method is a five-value enum on the server; these are its
    // words translated into words a farmer uses
    const METHOD = {
      'Normal, with sprays': 'conventional',
      'No sprays at all': 'organic',
      'Water beds': 'hydroponic',
      'Fish and plants': 'aquaponic',
      'Building the soil back': 'regenerative',
    };
    chips(ctx.root, 'cp', (text) => {
      if (METHOD[text]) draft.put({ method: METHOD[text] });
      const months = /^(\d+) months?$/.exec(text);
      if (months) draft.put({ timeline: Number(months[1]) });
    });

    const notes = [...ctx.root.querySelectorAll('div')]
      .find(e => /Canal water has been steady/.test(e.textContent) && e.children.length === 0);
    const ta = asField(notes, { multiline: true, rows: 3, value: d.description || '',
      placeholder: 'What happened last year on this plot', label: 'Anything they should know' });
    ta?.addEventListener('input', () => draft.put({ description: ta.value.slice(0, 1000) }));

    const next = byLabel(ctx.root, 'Next');
    next?.setAttribute('data-act', '');
    next?.addEventListener('click', () => {
      const cur = draft.get();
      if (!cur.amount) return state(ctx.root, 'waiting', 'How much do you need?',
        'Without an amount there is nothing for anyone to fund.');
      if (!cur.timeline) draft.put({ timeline: 6 });
      if (!cur.method) draft.put({ method: 'conventional' });
      location.href = './ask-money-terms.html';
    });
  });
}

// ── step 3: how they get paid, then submit ──────────────────────────────────
export function stepTerms() {
  const ctx = requireUser('ask-money-terms', ['farmer']);
  if (!ctx) return;
  load(ctx.root, async () => {
    const d = draft.get();
    if (!d.cid || !d.amount) { location.replace('./ask-money-video.html'); return; }

    // The roi field is required and it is a projection, so rather than let a
    // farmer invent one it is worked out from her own last settled season. If
    // there isn't one, the screen says so instead of showing a number.
    const past = await api.projects.mine(ctx.user._id || ctx.user.id).catch(() => []);
    const settled = (past || []).find(p => p.harvestReportedAt && p.harvestRevenue);
    let roi = null;
    if (settled) {
      const profit = Math.max(0, (settled.harvestRevenue || 0) - (settled.inputCostBasis || 0));
      const pool = settled.amount || 1;
      roi = Math.round(100 * profit * ((settled.investorShare || 0) / 100) / pool);
    }
    bind(ctx.root, {
      roi: {
        value: roi == null ? 'not known' : `${roi}%`,
        basis: settled
          ? `worked out from ${rupees(settled.harvestRevenue)} sold and ${rupees(settled.inputCostBasis)} spent last season`
          : 'you have not finished a season with us, so there is nothing to work it out from',
      },
    });
    draft.put({ roi: roi == null ? 0 : roi });

    // profit share or full repayment. The server treats them as equivalent;
    // they are not, and each card already says what happens if the crop fails.
    const cards = [...ctx.root.querySelectorAll('div[style*="background: #f6f3ef"][style*="padding: 15px 16px"]')];
    let mode = 'profit_share';
    cards.forEach((card, i) => {
      card.setAttribute('data-act', '');
      card.setAttribute('role', 'radio');
      card.addEventListener('click', () => {
        mode = i === 0 ? 'profit_share' : 'full_repayment';
        cards.forEach((c, k) => {
          c.style.boxShadow = k === i ? 'inset 0 0 0 2px #1d1a17' : '';
          const dot = c.querySelector('div[style*="border-radius: 10px"]');
          if (dot) {
            dot.style.background = k === i ? '#1d1a17' : 'transparent';
            dot.style.boxShadow = k === i ? '' : 'inset 0 0 0 2px #78716a';
          }
        });
        draft.put({ settlementMode: mode });
      });
    });
    draft.put({ settlementMode: mode });

    chips(ctx.root, 'cp', (text) => {
      const pct = /^(\d+)%$/.exec(text);
      if (pct) draft.put({ investorShare: Number(pct[1]) });
    });

    const submit = byLabel(ctx.root, 'See what investors will see');
    submit?.setAttribute('data-act', '');
    submit?.addEventListener('click', async () => {
      const f = draft.get();
      if (!f.investorShare) return state(ctx.root, 'waiting', 'Pick a share',
        'What fraction of the profit goes to the people funding you?');
      const label = submit.textContent;
      submit.textContent = 'Sending…';
      try {
        const created = await api.projects.create({
          title: `${f.crop || 'Crop'} — ${f.acres || 1} acres`,
          crop: f.crop || 'Crop',
          acres: f.acres || 1,
          amount: f.amount,
          method: f.method || 'conventional',
          description: f.description || 'No notes given.',
          timeline: f.timeline || 6,
          roi: f.roi || 0,
          investorShare: f.investorShare,
          settlementMode: f.settlementMode || 'profit_share',
          cid: f.cid,
        });
        draft.clear();
        location.href = `./money.html?created=${created._id || ''}`;
      } catch (err) {
        submit.textContent = label;
        state(ctx.root, 'failed', 'That was not accepted', err.message);
      }
    });
  });
}
