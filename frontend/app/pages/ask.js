// The agronomy chat. The interesting state here is the one the design did not
// have: a free inference tier rate-limits per minute and per day, so this
// screen has to be able to say "not now" in words a farmer can act on.
import { requireUser, api, load, state } from './_guard.js';
import { oneByText, reason } from '../bind.js';
import { asField, acts, press } from '../wire.js';
import { plainText } from '../bind.js';

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
const callPanel = root.querySelector('[data-call]')?.closest('[data-callpanel]');
// the flex ROW that holds the field and the button, so the panel goes above it
// as a sibling -- inserting into the row put it beside the input and squashed it
const composerRow = root.querySelector('[data-send]')?.closest('div[style*="align-items: stretch"]');

if (thread) {
  const bubbles = [...thread.querySelectorAll('.them, .me')];
  bubbles.slice(1).forEach(b => b.remove());
  /* Before the first question there is one bubble in a column drawn for a
   * whole conversation, and it sat at the top with four hundred pixels of
   * nothing under it. A spacer at each end centres it instead, so the empty
   * state reads as a conversation about to start rather than a hole -- which is
   * what every chat does before its first message. The second spacer is
   * removed the moment anything is said. */
  const tail = document.createElement('div');
  tail.setAttribute('data-spacer-end', '');
  tail.style.flexGrow = '1';
  thread.append(tail);
}
// On the laptop board the panel is already in the rail, standing beside the
// conversation instead of inside it, so there is nothing to lift.
if (callPanel && !callPanel.closest('.rail') && composerRow?.parentElement) {
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
  // A bubble arriving with no bridge is the case this rule exists for. It comes
  // up from where the composer is, which is where the reader is looking.
  el.style.opacity = '0';
  el.style.transform = 'translateY(8px)';
  el.style.transition = 'opacity var(--t-press, 120ms) var(--e-smooth, ease),'
    + ' transform var(--t-snappy, 568ms) var(--e-snappy, ease)';
  // The spacer is the FIRST child now, not the last: it pushes a short
  // conversation down onto the composer and collapses once the transcript
  // overflows. So a new bubble is appended, and inserting before the spacer
  // would put every answer above the greeting.
  // The opening line was centred while nothing had been said. Once something
  // has, the transcript is a transcript: it grows from the composer upward.
  thread?.querySelector('[data-spacer-end]')?.remove();
  thread?.append(el);
  requestAnimationFrame(() => { el.style.opacity = '1'; el.style.transform = 'none'; });
  // The transcript scrolls now, so a new message can land below the fold. Keep
  // the newest one in view -- and jump rather than glide when travel is off.
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  el.scrollIntoView({ block: 'end', behavior: still ? 'auto' : 'smooth' });
  return el;
}

/* Where the sentence came from -----------------------------------------------
 * r.source is the PIPELINE name ('rag'), not a citation -- printing it put the
 * word "rag" under an answer as though it were a document. A real citation only
 * exists when the retrieval server was up and returned chunks, which carry a
 * page number. When it did not, the answer is general knowledge and the screen
 * has to say so rather than imply a document nobody quoted.
 */
function provenance(r) {
  const hits = Array.isArray(r.retrieved) ? r.retrieved.filter(d => d && d.page != null) : [];
  if (hits.length) {
    const pages = [...new Set(hits.map(d => d.page))].slice(0, 3).join(', ');
    return 'Government document, page ' + pages;
  }
  return 'General farming knowledge, not quoted from a document.';
}

// What has been said so far, so a follow-up question makes sense. The route
// bounds it to the last eight turns; this keeps them in order.
const said = [];

// plainText is shared with the leaf checker; see app/bind.js.


let asking = false;
async function ask(text) {
  if (!text || asking) return;
  asking = true;
  bubble('me', text);
  said.push({ role: 'user', content: text });
  const pending = bubble('them', 'Looking through the documents…');
  try {
    const r = await api.ai.ask(text, said.slice(0, -1));
    pending.remove();
    const answer = plainText(r.answer || r.message) || 'The documents do not cover that.';
    said.push({ role: 'assistant', content: answer });
    bubble('them', answer, provenance(r));
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
      const panel = root.querySelector('[data-call]')?.closest('[data-callpanel]');
      if (panel) {
        bubble('them', 'The Kisan Call Centre answers when we cannot. Their number is on this screen, tap it to call.');
        panel.scrollIntoView({ block: 'center', behavior: 'smooth' });
      } else {
        bubble('them', 'Kisan Call Centre, 1800 180 1551. Free, 6 am to 10 pm, and they speak Telugu.');
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
