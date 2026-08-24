import { requireUser, api, load, state } from './_guard.js';
import { repeat, bind } from '../bind.js';
import { rupees } from '../api.js';
import { promote, goes, press } from '../wire.js';

const ctx = requireUser('invest', ['investor']);
if (ctx) load(ctx.root, async () => {
  const initial = (ctx.user.name || '?').trim()[0].toUpperCase();
  const all = await api.projects.open();
  const open = (all || []).filter(p => (p.status || 'open') !== 'cancelled');
  const list = ctx.root.querySelector('[data-list="projects"]');

  bind(ctx.root, {
    me: { name: (ctx.user.name || '').split(' ')[0], initial },
    openCount: open.length
      ? `${open.length} season${open.length === 1 ? '' : 's'} looking for money`
      : 'No seasons are open right now',
  });
  if (!open.length) {
    return state(list, 'empty', 'Nothing is open for funding',
      'No farmer is raising money this week. Nothing has gone wrong.');
  }

  repeat(list, open.map(p => ({
    title: p.title,
    meta: [p.acres && `${p.acres} acres`, p.method, p.timeline && `${p.timeline} months`].filter(Boolean).join(' · '),
    raised: rupees(p.fundedAmount || 0),
    goal: `of ${rupees(p.amount)}`,
    pct: Math.min(100, Math.round(100 * (p.fundedAmount || 0) / (p.amount || 1))) + '%',
  })), (el, row) => {
    const bar = el.querySelector('div[style*="background: #01579b"]');
    if (bar) bar.style.width = row.pct;
  });

  // repeat() clones the first drawn row, and on the board that row is the
  // SELECTED one -- so every row arrived wearing the selection fill and the
  // blue border, and the thing you were actually looking at was indicated by
  // nothing at all. Selection is state; the template carries none of it.
  const rows = [...list.children];
  const select = (i) => rows.forEach((el, n) => {
    const on = n === i;
    el.style.background = on ? '#eae4de' : '#fff';
    el.style.borderLeftColor = on ? '#012169' : 'transparent';
    el.setAttribute('aria-current', on ? 'true' : 'false');
  });
  rows.forEach((el, i) => {
    promote(el, open[i]?.title || 'This season');
    el.addEventListener('click', () => { select(i); show(open[i]); });
  });
  select(0);
  await show(open[0]);
  press(ctx.root);

  async function show(p) {
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

    const farmer = p.farmerName || 'The farmer';
    const needed = Math.max(0, (p.amount || 0) - (p.fundedAmount || 0));
    // What we would put in is what is still needed, capped at a sane first
    // step. Nothing is committed here; the next screen restates it.
    const offer = Math.min(needed, 50000) || needed;

    // The evidence block only claims what the video record actually says. If a
    // date has not landed in a block yet, it says so instead of printing one.
    // GET /videos/:cid/provenance answers { uploadedAt, integrity: { sha256 } }
    // and carries NO anchor information; the block lives on
    // GET /videos/:cid/anchor as { anchored, blockHeight }. This file read
    // video.uploadTimestamp, video.videoFileHash, video.anchored and
    // video.blockHeight -- none of which that response has -- so the evidence
    // chain printed "not recorded" three times and "No fingerprint on the
    // record" on the one panel this whole product exists to show, while the
    // hash sat in the response it had just fetched.
    let video = null;
    let anchor = null;
    if (p.cid) {
      [video, anchor] = await Promise.all([
        api.videos.provenance(p.cid).catch(() => null),
        api.videos.anchor(p.cid).catch(() => null),
      ]);
    }
    const sha = video?.integrity?.sha256 || null;
    const shortSha = sha ? `sha256 ${sha.slice(0, 8)}…${sha.slice(-4)}` : null;
    const anchored = Boolean(anchor?.anchored && anchor?.blockHeight);

    bind(ctx.root, {
      lot: {
        season: [p.crop, p.timeline && `${p.timeline} months`].filter(Boolean).join(' · '),
        title: p.title,
        farmer,
        since: p.acres ? `· ${p.acres} acres` : '',
        needed: rupees(needed),
        goal: `of ${rupees(p.amount)}`,
      },
      file: {
        meta: shortSha || 'No file record',
        count: p.cid ? '1 of 1' : '—',
      },
      proved: {
        line: anchored
          ? `This exact video file has not been altered. Its date is written into Bitcoin block ${Number(anchor.blockHeight).toLocaleString('en-IN')}.`
          : 'This exact video file has not been altered. Its date has not landed in a block yet, so there is no block number to check — usually by tomorrow.',
      },
      told: {
        ask: `Ask ${farmer.split(' ')[0]} a question`,
        title: `${farmer} told us — nobody has checked`,
        body: [
          'That this is their land',
          p.acres ? `that it is ${p.acres} acres` : null,
          p.crop ? `that it is ${p.crop.toLowerCase()}` : null,
          'and the sowing date.',
        ].filter(Boolean).join(', ') + ' A phone\u2019s reported location can be faked.',
      },
      offer: {
        title: `If you put in ${rupees(offer)}`,
        back: rupees(offer),
        share: `${p.investorShare || 0}%`,
        when: `Paid after ${farmer} reports the harvest.`,
      },
      told2: {},
    });

    // ---- the evidence chain -------------------------------------------
    // Three events with real times, or removed. The green dot is the only green
    // on the page and it belongs to the block alone: the two steps above it are
    // our own word, so they are ink.
    const chain = ctx.root.querySelector('[data-chain]');
    if (chain && !video) {
      // No video record, so the chain has no events. Three rows of "not
      // recorded" is worse than not asking the question: the tier block below
      // already says there is nothing to check.
      chain.remove();
    } else if (chain) {
      const when = (iso) => (iso ? new Date(iso).toLocaleString('en-IN', {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false,
      }).toUpperCase().replace(',', '') : null);
      const set = (kind, time, note, head) => {
        const w = chain.querySelector(`[data-chain-when="${kind}"]`);
        const n = chain.querySelector(`[data-chain-note="${kind}"]`);
        const h = chain.querySelector(`[data-chain-head="${kind}"]`);
        if (w) w.textContent = time == null ? 'not recorded' : time;
        if (n && note != null) n.textContent = note;
        if (h && head) h.textContent = head;
      };
      const filmed = when(video?.uploadedAt);
      set('filmed', filmed, null);
      set('hashed', filmed, shortSha || 'No fingerprint on the record');
      if (anchored) {
        set('block', when(anchor.anchoredAt) || filmed,
            `block ${Number(anchor.blockHeight).toLocaleString('en-IN')} — check it on any explorer`);
      } else {
        // The date has not landed, so the green goes: it is the one token this
        // product spends on a fact anybody can check, and there is not one yet.
        const dot = chain.querySelector('[data-chain-dot="block"]');
        if (dot) dot.style.background = '#78716a';
        const head = chain.querySelector('[data-chain-head="block"]');
        if (head) { head.style.color = '#4a443d'; head.textContent = 'Not in a block yet'; }
        // Not "not recorded": nothing has happened yet, which is different from
        // something happening and us failing to write it down.
        set('block', '', 'The proof job runs once a day, so usually by tomorrow.');
      }
    }

    const go = [...ctx.root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'See the exact amount');
    if (go) goes(go.parentElement, `confirm-investment?project=${p._id || p.id}&amount=${offer}`,
                 'See the exact amount');

    // "Ask <farmer> a question" is a lifted white bar with an icon sitting
    // directly under the primary button -- it reads as the page's secondary
    // action and did nothing at all.
    const askRow = ctx.root.querySelector('[data-bind="told.ask"]')?.parentElement;
    if (askRow && p.farmerWallet) {
      promote(askRow, `Ask ${farmer.split(' ')[0]} a question`);
      // assigned, not added: show() runs again on every row click, and
      // addEventListener would stack one handler per click.
      askRow.onclick = async () => {
        try {
          const conv = await api.messages.open({ targetUserId: p.farmerWallet });
          location.href = `./thread.html?c=${conv._id || conv.id}`;
        } catch (err) {
          state(askRow, 'failed', 'Could not open the conversation', err.message);
        }
      };
    } else if (askRow) {
      askRow.remove();       // no farmer to ask
    }
  }
});
