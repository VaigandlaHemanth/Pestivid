// The screen where money actually moves.
//
// The gate on the whole transaction was a 22px empty square: under the 24px
// desktop floor, with the 293px label outside the hit area, no keyboard path
// even though the JS set role=checkbox, and "ticked" communicated by a red
// square appearing inside an already-red panel. The enable moment -- the most
// consequential state change in this product -- was a hard cut from #c9ced4 to
// #016abe, two hexes in no token, and it introduced a third primary colour to a
// single flow: the button that leads here is ink on Invest, and this one turned
// blue.
import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { rupees } from '../api.js';
import { acts, goes, press } from '../wire.js';

const ctx = requireUser('confirm-investment', ['investor']);

if (ctx) {
  const root = ctx.root;
  goes(root.querySelector('[data-back]'), 'invest', 'Go back without sending');
  press(root);

  load(root, async () => {
    const q = new URLSearchParams(location.search);
    const id = q.get('project');
    const amount = Number(q.get('amount')) || 0;
    if (!id) {
      return state(root, 'empty', 'No season chosen',
        'Open this from a season you were reading.',
        { label: 'See what is open', go: 'invest' });
    }

    const p = await api.projects.one(id);
    bind(root, {
      lot: {
        title: p.title,
        meta: [p.farmerName, p.acres && `${p.acres} acre${p.acres === 1 ? '' : 's'}`, p.crop].filter(Boolean).join(' · '),
        needed: rupees(Math.max(0, (p.amount || 0) - (p.fundedAmount || 0))),
        // These two were not bound at all -- the artboard's "60%" and
        // "Harvest reported, about 7 weeks" were shown to an investor about to
        // send money into a season whose real share is whatever the farmer
        // agreed. Two of the four figures on this screen were decoration.
        share: p.investorShare != null ? `${p.investorShare}%` : 'not stated',
        when: p.settlementMode === 'full_repayment'
          ? 'Harvest reported, the whole amount comes back'
          : (p.timeline
            ? `Harvest reported, about ${p.timeline} month${p.timeline === 1 ? '' : 's'}`
            : 'When the harvest is reported'),
      },
      amount: rupees(amount),
    });

    // The acknowledgement. The whole row is the control, so the sentence is part
    // of the target rather than decoration beside it.
    const ackRow = root.querySelector('[data-ack]');
    const box = ackRow?.querySelector('[data-box]');
    const tick = box?.querySelector('svg');
    const send = root.querySelector('[data-send]');
    const label = send?.querySelector('div');
    let agreed = false;

    const paint = () => {
      if (send) {
        send.style.background = agreed ? '#1d1a17' : '#d3ccc5';
        send.setAttribute('aria-disabled', String(!agreed));
        send.style.cursor = agreed ? 'pointer' : 'default';
      }
      if (label) label.style.color = agreed ? '#f6f3ef' : '#4a443d';
      if (box) box.style.background = agreed ? '#a71930' : '#fff';
      if (tick) {
        tick.style.opacity = agreed ? '1' : '0';
        tick.style.transform = agreed ? 'scale(1)' : 'scale(.6)';
      }
    };

    if (ackRow) {
      ackRow.setAttribute('role', 'checkbox');
      ackRow.setAttribute('aria-checked', 'false');
      acts(ackRow, 'I understand I could lose this money', () => {
        agreed = !agreed;
        ackRow.setAttribute('aria-checked', String(agreed));
        paint();
      });
    }
    paint();

    let busy = false;
    acts(send, `Send ${rupees(amount)}`, async () => {
      // Refusing silently is what a disabled button does, and this one is not
      // visibly a button until it is armed -- so say why nothing happened.
      if (!agreed) {
        ackRow?.scrollIntoView({ block: 'center', behavior: 'smooth' });
        if (box) {
          // one nudge on the box, not a colour change: the panel is already red
          box.style.transform = 'scale(1.12)';
          setTimeout(() => { box.style.transform = 'scale(1)'; }, 180);
        }
        return;
      }
      if (busy) return;
      busy = true;
      const was = label?.textContent;
      if (label) label.textContent = 'Sending…';
      try {
        await api.investments.create({ projectId: id, amount });
        if (label) label.textContent = 'Sent';
        location.href = './portfolio.html';
      } catch (err) {
        if (label) label.textContent = was;
        busy = false;
        state(send.parentElement, 'failed', 'The money did not move', err.message);
      }
    });

    press(root);
  });
}
