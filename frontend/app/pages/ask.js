// The agronomy chat. The interesting state here is the one the design did not
// have: a free inference tier rate-limits per minute and per day, so this
// screen has to be able to say "not now" in words a farmer can act on.
import { requireUser, api, load, state } from './_guard.js';
import { oneByText, reason } from '../bind.js';
import { asField, acts, press } from '../wire.js';

const ctx = requireUser('ask', ['farmer', 'investor', 'buyer', 'admin']);
// The board drew an example exchange so the bubbles could be judged. Keep the
// opening line, drop the rest: a farmer must not read a worked example as an
// answer they were given.
if (!ctx) { /* requireUser has already sent them to sign in */ } else {

const root = ctx.root;
const thread = root.querySelector('.them')?.parentElement;
const field = oneByText('Type your question', root);
const send = root.querySelector('[data-send]');

// The Kisan Call Centre panel is drawn INSIDE one of the example bubbles, so
// stripping the examples deleted the one actionable thing on the screen -- a
// farmer who cannot read a bottle label had nothing to tap. It is not an
// example, it is the standing fallback for every time this chat cannot answer,
// so it is lifted out and kept above the composer where it is always reachable.
const callPanel = root.querySelector('[data-call]')?.closest('div[style*="background: #e7e1db"]');
// the flex ROW that holds the field and the button, so the panel goes above it
// as a sibling -- inserting into the row put it beside the input and squashed it
const composerRow = root.querySelector('[data-send]')?.closest('div[style*="align-items: stretch"]');

if (thread) {
  const bubbles = [...thread.querySelectorAll('.them, .me')];
  bubbles.slice(1).forEach(b => b.remove());
}
if (callPanel && composerRow?.parentElement) {
  callPanel.style.marginTop = '0';
  callPanel.style.marginBottom = '12px';
  composerRow.parentElement.insertBefore(callPanel, composerRow);
}

function bubble(kind, text, source) {
  const el = document.createElement('div');
  el.className = kind;                        // .them and .me are the board's
  const p = document.createElement('div');
  p.className = 'p'; p.textContent = text;
  el.append(p);
  if (source) {
    const s = document.createElement('div');
    s.style.cssText = 'margin-top: 11px; padding-top: 10px; min-height: 44px; box-sizing: border-box; display: flex; align-items: center; box-shadow: inset 0 1px 0 #dcd6c9;';
    const t = document.createElement('div');
    t.className = 'm';
    t.style.cssText = 'font-size: 12.5px; line-height: 1.5; color: #4a443d;';
    t.textContent = source;
    s.append(t); el.append(s);
  }
  thread?.insertBefore(el, thread.querySelector('div[style*="flex-grow: 1"]'));
  return el;
}

let asking = false;
async function ask(text) {
  if (!text || asking) return;
  asking = true;
  bubble('me', text);
  const pending = bubble('them', 'Looking through the documents…');
  try {
    const r = await api.ai.ask(text, []);
    pending.remove();
    bubble('them', r.answer || r.message || 'The documents do not cover that.', r.source || r.citation);
  } catch (err) {
    pending.remove();
    const [h, d] = reason(err);
    const box = bubble('them', `${h}. ${d}`);
    box.style.background = err.rateLimited ? '#f2e6cd' : '#f7e9e6';
    // A refusal is the system working. Point at the number that always answers.
    if (err.rateLimited || err.offline) {
      // The board's own refusal panel already has this number as a tel: link.
      // Restating it as prose in a bubble made it dead in a second place, so
      // this scrolls to the live one instead of printing a copy.
      const panel = root.querySelector('[data-call]')?.closest('div[style*="background: #e7e1db"]');
      if (panel) {
        bubble('them', 'The Kisan Call Centre answers when we cannot. Their number is on this screen — tap it to call.');
        panel.scrollIntoView({ block: 'center', behavior: 'smooth' });
      } else {
        bubble('them', 'Kisan Call Centre — 1800 180 1551. Free, 6 am to 10 pm, and they speak Telugu.');
      }
    }
  } finally { asking = false; }
}

const input = asField(field, { name: 'question', enterKeyHint: 'send',
  placeholder: 'Type your question', label: 'Your question' });
if (input) {
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { ask(input.value.trim()); input.value = ''; }
  });
}

/* ---- the composer button -------------------------------------------------
 * The board drew a microphone and the caption promised speech. There is no
 * speech capture in this product and there is not going to be one for now, so
 * the microphone has left the board and the button is the arrow it always
 * should have been: it sends what is typed. The caption no longer promises
 * anything it cannot do.
 * ------------------------------------------------------------------------- */
const submit = () => { const t = input?.value.trim(); if (input) input.value = ''; ask(t); };

// Inactive until there is something to send, the same way the investor's
// confirm button waits for its acknowledgement -- a full-strength blue button
// that silently does nothing is the defect this whole pass has been removing.
const paintSend = () => {
  const has = Boolean(input?.value.trim());
  send.style.background = has ? '#016abe' : '#c3bcb6';
  send.setAttribute('aria-disabled', String(!has));
};
input?.addEventListener('input', paintSend);
paintSend();

acts(send, 'Send your question', () => {
  if (!input?.value.trim()) {
    // Say why nothing happened, once, next to the thing that is missing.
    let hint = root.querySelector('[data-hint]');
    if (!hint) {
      hint = document.createElement('div');
      hint.setAttribute('data-hint', '');
      hint.style.cssText = 'font-size: 13.5px; line-height: 1.45; margin-top: 8px; color: #7c4a12;';
      send.closest('div[style*="align-items: stretch"]')?.after(hint);
    }
    hint.textContent = 'Type a question first, or tap one of the suggestions above.';
    input?.focus();
    return;
  }
  root.querySelector('[data-hint]')?.remove();
  submit();
  paintSend();
});

// The leaf screen hands a question over with the verdict already attached, so
// the farmer does not explain the photo twice.
const handed = sessionStorage.getItem('pv.askText');
if (handed) {
  sessionStorage.removeItem('pv.askText');
  if (input) input.value = handed;
  setTimeout(() => ask(handed), 200);
}

press(root);

// the suggestion chips are real questions
for (const chip of root.querySelectorAll('.chip')) {
  chip.setAttribute('data-act', '');
  chip.setAttribute('role', 'button');
  chip.tabIndex = 0;
  chip.addEventListener('click', () => ask(chip.textContent.trim()));
}

}
