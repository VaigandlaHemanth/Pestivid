// Asking for money: one screen, three steps.
//
// It was three URLs -- ask-money-video, ask-money-amount, ask-money-terms --
// for one form, which is why the answers lived in sessionStorage: they had to
// survive two page loads. The steps change in place now, so a dropped signal
// halfway through costs nothing. The draft is still written, but as a save
// against an accidental reload rather than as a handoff.
//
// FundingRequest refuses to save without a cid, so the video is question one --
// which is also the honest ordering, because it is the only thing on the page
// an investor can check without trusting the farmer. The settlement choice is
// last and alone: it is the one question on which a farmer can lose their land.
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeat, showPoster } from '../bind.js';
import { rupees, whenShort, dateState } from '../api.js';
import { asField, press } from '../wire.js';

const KEY = 'pv.ask';
const draft = {
  get() { try { return JSON.parse(sessionStorage.getItem(KEY)) || {}; } catch { return {}; } },
  put(patch) { sessionStorage.setItem(KEY, JSON.stringify({ ...draft.get(), ...patch })); },
  clear() { sessionStorage.removeItem(KEY); },
};

const byLabel = (root, t) => [...root.querySelectorAll('div')]
  .find(d => d.children.length === 0 && d.textContent.trim() === t);

const chips = (root, cls, onPick, group) => {
  const list = [...root.querySelectorAll(group ? `[data-chip="${group}"]` : `.${cls}, .${cls}On`)];
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

const ctx = requireUser('ask-money', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const root = ctx.root;
  const steps = [1, 2, 3].map(n => root.querySelector(`[data-step="${n}"]`));
  const segs = [1, 2, 3].map(n => root.querySelector(`[data-seg="${n}"]`));
  const nextLabel = root.querySelector('[data-next]');
  const nextBtn = nextLabel?.parentElement;
  const LABEL = ['Next', 'Next', 'See what investors will see'];

  // Each step gets its own place to be told what is missing, so a message about
  // the amount cannot appear under the settlement cards.
  const notices = steps.map((s) => {
    const n = document.createElement('div');
    s?.append(n);
    return n;
  });

  let at = 1;
  const show = (n) => {
    at = n;
    steps.forEach((s, i) => { if (s) s.style.display = i + 1 === n ? '' : 'none'; });
    segs.forEach((s, i) => { if (s) s.style.background = i + 1 <= n ? '#1d1a17' : '#d3ccc5'; });
    if (nextLabel) nextLabel.textContent = LABEL[n - 1];
    if (n === 3) paintAmount();
    // Moving between steps changes the content of one place rather than the
    // page, so put the reader at the question that just arrived.
    steps[n - 1]?.querySelector('[data-stepq]')?.scrollIntoView({ block: 'nearest' });
  };
  const stop = (n, headline, detail) => {
    state(notices[n - 1], 'waiting', headline, detail);
    return false;
  };

  // -- step 1: which video ---------------------------------------------------
  const s1 = steps[0];
  const vids = await api.videos.mine(ctx.user._id || ctx.user.id);
  bind(root, { step1: { hint: vids.length
    ? 'Pick a video of it. You cannot ask for money without one.'
    : 'You have no videos yet, and money cannot be asked for without one.' } });
  // Marked, not keyed on a fill. A selector that names a colour breaks the day
  // the colour moves, silently, which is how the harvest button on the money
  // screen stopped being wired at all.
  const rows = [...s1.querySelectorAll('[data-vidrow]')];
  const container = rows[0]?.parentElement;
  if (!vids.length) {
    state(container || s1, 'empty', 'Film the field first',
      'One forty-second walk across it is enough. Come back here afterwards.');
  } else {
    // The bar is the server's, not ours: hashed by us and fingerprinted.
    // See services/videoEligibility.js -- the Bitcoin date arrives afterwards
    // and is not what a funding request is checked against.
    repeat(container, vids.map(v => {
      const s = dateState(v);
      const usable = v.hashComputedBy === 'server' && v.fingerprinted;
      return {
        label: `${v.crop || v.location || 'Plot'} · ${whenShort(v.uploadTimestamp)}`,
        // The short form: five rows repeating the same eleven words is the
        // explanation four times too often. See plots.js.
        status: usable ? (s.short || s.text) : 'We are still checking this one',
        ready: usable, proved: s.kind === 'proved', v,
      };
    }), (el, r) => {
      // One real frame in the thumbnail, when the server could cut one.
      showPoster(el.querySelector('[data-thumb]'), r.v);
      // repeat() clones the FIRST drawn row as its template, and that row was
      // drawn selected -- so every video arrived wearing the 2px ring and all
      // four looked chosen before anything was tapped.
      el.style.boxShadow = '';
      // The green tick belongs to a proved date and nothing else. It was drawn
      // beside the status line unconditionally, so a row reading "date being
      // written, usually by tomorrow" carried a green tick next to it.
      const tick = el.querySelector('svg');
      if (tick) tick.style.display = r.proved ? '' : 'none';
      const m = el.querySelector('.m');
      if (m) m.textContent = r.label;
      const status = [...el.querySelectorAll('div')].reverse()
        .find(d => !d.children.length && d.textContent.trim() && d !== m);
      if (status) {
        status.textContent = r.status;
        status.style.color = r.proved ? '#006934' : '#4a443d';
      }
      // A video we have not finished checking is shown, not hidden, with why
      if (!r.ready) return;   // no opacity wash: it drops the row's own text below AA
      el.setAttribute('data-act', '');
      el.addEventListener('click', () => {
        draft.put({ cid: r.v.cid, crop: r.v.crop, location: r.v.location });
        [...container.children].forEach(c => { c.style.boxShadow = ''; });
        el.style.boxShadow = 'inset 0 0 0 2px #1d1a17';
      });
    });
  }

  const crop = asField(byLabel(s1, 'Potato'), { placeholder: 'Potato', label: 'What are you growing' });
  const acres = asField(byLabel(s1, '2 acres'),
    { placeholder: '2', inputMode: 'decimal', label: 'How much land, in acres' });
  crop?.addEventListener('input', () => draft.put({ crop: crop.value.trim() }));
  acres?.addEventListener('input', () => draft.put({ acres: Number(acres.value) || 0 }));

  // -- step 2: how much, how long, how grown ---------------------------------
  const s2 = steps[1];
  const d0 = draft.get();
  bind(root, { amount: d0.amount ? rupees(d0.amount) : '' });
  const amountField = asField(s2.querySelector('[data-bind="amount"]'),
    // A bare ₹ is not a specimen of anything. A number is.
    { inputMode: 'numeric', placeholder: '5,00,000', value: d0.amount || '',
      label: 'How much you need' });
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
  // Two questions, two groups. Selecting a growing method must not clear the
  // timeline, which is what one shared group did.
  // "Other" was a chip that selected, stored nothing, and let the six-month
  // default through -- so a farmer who needed twelve months tapped it, watched
  // it go black, and got six. It opens a field now.
  const otherWrap = document.createElement('div');
  const otherLabel = document.createElement('label');
  otherLabel.textContent = 'How many months?';
  otherLabel.style.cssText = 'display: block; font-size: 14px; font-weight: 600; margin-top: 10px;';
  const otherMonths = document.createElement('input');
  otherMonths.type = 'text';
  otherMonths.inputMode = 'numeric';
  otherMonths.autocomplete = 'off';
  otherMonths.spellcheck = false;
  otherMonths.placeholder = '12';
  otherMonths.id = 'ask-other-months';
  otherMonths.style.cssText = 'width: 100%; box-sizing: border-box; background: #f6f3ef; height: 52px;'
    + ' margin-top: 6px; padding: 0 14px; border: 0; box-shadow: inset 0 0 0 1px #c3bcb6;'
    + ' font: inherit; font-size: 17px; color: #1d1a17;';
  otherLabel.htmlFor = otherMonths.id;
  otherWrap.append(otherLabel, otherMonths);
  otherWrap.style.display = 'none';
  otherMonths.addEventListener('input', () => {
    otherMonths.value = otherMonths.value.replace(/[^\d]/g, '').slice(0, 2);
    const n = Number(otherMonths.value);
    if (n >= 1 && n <= 24) draft.put({ timeline: n });
  });

  const timelineChips = chips(s2, 'cp', (text) => {
    const months = /^(\d+) months?$/.exec(text);
    if (months) { draft.put({ timeline: Number(months[1]) }); otherWrap.style.display = 'none'; return; }
    if (text === 'Other') {
      otherWrap.style.display = '';
      otherMonths.focus();
      const n = Number(otherMonths.value);
      draft.put({ timeline: n >= 1 && n <= 24 ? n : 0 });
    }
  }, 'timeline');
  timelineChips[0]?.parentElement?.after(otherWrap);

  // The board draws "6 months" and "Normal, with sprays" already selected, and
  // those are real defaults -- but nothing wrote them to the draft, so the step
  // gate below would have stopped a farmer who simply accepted what was on
  // screen. Whatever is drawn selected is what the draft starts with.
  const drawn = (group, read) => {
    const on = s2.querySelector(`[data-chip="${group}"].cpOn`);
    if (on) read(on.textContent.trim());
  };
  drawn('timeline', (t) => {
    const m = /^(\d+) months?$/.exec(t);
    if (m) draft.put({ timeline: Number(m[1]) });
  });
  drawn('method', (t) => { if (METHOD[t]) draft.put({ method: METHOD[t] }); });
  chips(s2, 'cp', (text) => {
    if (METHOD[text]) draft.put({ method: METHOD[text] });
  }, 'method');

  const notes = [...s2.querySelectorAll('div')]
    .find(e => /Canal water has been steady/.test(e.textContent) && e.children.length === 0);
  const ta = asField(notes, { multiline: true, rows: 3, value: d0.description || '',
    placeholder: 'What happened last year on this plot', label: 'Anything they should know' });
  ta?.addEventListener('input', () => draft.put({ description: ta.value.slice(0, 1000) }));

  // -- step 3: how they get paid ---------------------------------------------
  const s3 = steps[2];
  // The roi field is required and it is a projection, so rather than let a
  // farmer invent one it is worked out from her own last settled season. If
  // there is no such season, the screen says so instead of showing a number.
  const past = await api.projects.mine(ctx.user._id || ctx.user.id).catch(() => []);
  const settled = (past || []).find(p => p.harvestReportedAt && p.harvestRevenue);
  let roi = null;
  if (settled) {
    const profit = Math.max(0, (settled.harvestRevenue || 0) - (settled.inputCostBasis || 0));
    const pool = settled.amount || 1;
    roi = Math.round(100 * profit * ((settled.investorShare || 0) / 100) / pool);
  }
  // The paragraph under the figure claimed she had sold a specific amount last
  // Rabi. That was the artboard's specimen, and on a first season it was a
  // fabricated statement about her own finances sitting directly beneath the
  // words "not known". paintMode() below is the one place that writes this
  // panel -- two painters over the same three fields is how they drift apart.
  draft.put({ roi: roi == null ? 0 : roi });

  // Both sentences about what she owes quoted the artboard's five lakh.
  const paintAmount = () => bind(root, {
    terms: { amount: draft.get().amount ? rupees(draft.get().amount) : 'the amount' },
  });

  // profit share or full repayment. The server treats them as equivalent;
  // they are not, and each card already says what happens if the crop fails.
  const cards = [...s3.querySelectorAll('[data-methodcard]')];
  let mode = 'profit_share';
  cards.forEach((card, i) => {
    card.setAttribute('data-act', '');
    card.setAttribute('role', 'radio');
    card.setAttribute('aria-checked', String(i === 0));
    card.addEventListener('click', () => {
      mode = i === 0 ? 'profit_share' : 'full_repayment';
      cards.forEach((c, k) => {
        c.style.boxShadow = k === i ? 'inset 0 0 0 2px #1d1a17' : '';
        c.setAttribute('aria-checked', String(k === i));
        const dot = c.querySelector('div[style*="border-radius: 10px"]');
        if (dot) {
          dot.style.background = k === i ? '#1d1a17' : 'transparent';
          dot.style.boxShadow = k === i ? '' : 'inset 0 0 0 2px #78716a';
        }
      });
      draft.put({ settlementMode: mode });
      paintMode();
    });
  });
  draft.put({ settlementMode: mode });

  // Under full repayment there is no profit to take a share of, so the chips
  // stop being a question -- they used to stay on screen and selectable, and
  // whatever was tapped was sent to the server alongside a repayment agreement.
  const shareBlock = s3.querySelector('[data-share]');
  const paintMode = () => {
    const full = mode === 'full_repayment';
    if (shareBlock) shareBlock.style.display = full ? 'none' : '';
    if (full) draft.put({ investorShare: 0 });
    bind(root, { roi: {
      value: full ? 'the whole amount back' : (roi == null ? 'not known' : `${roi}%`),
      basis: full
        ? 'and nothing on top of it, because you are repaying rather than sharing'
        : (settled
          ? `worked out from ${rupees(settled.harvestRevenue)} sold and ${rupees(settled.inputCostBasis)} spent last season`
          : 'this is your first season with us'),
      source: full
        ? 'What they get back does not depend on the harvest under this choice. That is the whole '
          + 'difference between the two, and it is the reason the card above says it in red.'
        : (settled
          ? `You sold ${rupees(settled.harvestRevenue)} and spent ${rupees(settled.inputCostBasis)} last season, `
            + 'and we filled this in from those numbers so it is not a figure pulled from the air.'
          : 'There is nothing to work it out from, so we are not putting a number here, and an '
            + 'investor will read the same two words you are reading.'),
    } });
  };
  paintMode();

  // Scoped to step 3: the share chips carry no data-chip group, so an unscoped
  // lookup swept up step 2's timeline and method chips as well.
  chips(s3, 'cp', (text) => {
    const pct = /^(\d+)%$/.exec(text);
    if (pct) draft.put({ investorShare: Number(pct[1]) });
  });

  // -- the one button --------------------------------------------------------
  const submit = async () => {
    const f = draft.get();
    if (f.settlementMode !== 'full_repayment' && !f.investorShare) {
      return stop(3, 'Pick a share',
        'What fraction of the profit goes to the people funding you?');
    }
    const label = nextLabel.textContent;
    nextLabel.textContent = 'Sending…';
    try {
      const created = await api.projects.create({
        title: `${f.crop || 'Crop'}, ${f.acres || 1} acres`,
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
      nextLabel.textContent = label;
      state(notices[2], 'failed', 'That was not accepted', err.message);
    }
  };

  nextBtn?.setAttribute('data-act', '');
  nextBtn?.addEventListener('click', () => {
    const d = draft.get();
    if (at === 1) {
      if (!d.cid) return stop(1, 'Pick a video first',
        'The server will not accept a request without one, and neither will an investor.');
      return show(2);
    }
    if (at === 2) {
      if (!d.amount) return stop(2, 'How much do you need?',
        'Without an amount there is nothing for anyone to fund.');
      if (!d.timeline) {
        return stop(2, 'How many months?',
          'You picked Other, so put the number in and the date investors see will be right.');
      }
      if (!d.method) draft.put({ method: 'conventional' });
      return show(3);
    }
    return submit();
  });

  // The chevron steps back through the form before it leaves it, which is what
  // it did when each step was its own URL and the browser's own back worked.
  const back = root.querySelector('[data-chrome="back"]');
  if (back) {
    const chrome = back.cloneNode(true);
    back.replaceWith(chrome);
    chrome.setAttribute('data-act', '');
    chrome.setAttribute('aria-label', 'Back');
    chrome.addEventListener('click', () => {
      if (at > 1) return show(at - 1);
      location.href = './money.html';
    });
  }

  show(1);
  press(root);
});
