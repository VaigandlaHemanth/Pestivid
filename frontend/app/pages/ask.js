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
 * The board drew a microphone and the caption said "Hold the blue button and
 * speak", over a handler that sent whatever was typed. There was no speech
 * capture anywhere in frontend/, and home.js has been writing sessionStorage
 * 'pv.listen' since it was built with nothing on earth reading it.
 *
 * Voice is the stated accommodation for a farmer who cannot type Telugu, so it
 * is real here where the browser has the API. Where it does not, the mic is
 * swapped for the arrow and the caption stops promising speech -- rather than
 * leaving a microphone glyph over a text send, which is the version that tells
 * a farmer the app is broken.
 * ------------------------------------------------------------------------- */
const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
const mic = root.querySelector('[data-mic]');
const arrow = root.querySelector('[data-arrow]');
const caption = root.querySelector('[data-caption]');

const submit = () => { const t = input?.value.trim(); if (input) input.value = ''; ask(t); };

if (!Rec) {
  if (mic) mic.style.display = 'none';
  if (arrow) arrow.style.display = '';
  if (caption) {
    caption.textContent = 'This phone will not let a web page listen, so type the question. '
      + 'This is not a doctor or an agronomist — check anything important with your extension officer.';
  }
  acts(send, 'Send your question', submit);
} else {
  const rec = new Rec();
  rec.lang = document.documentElement.lang || localStorage.getItem('pv.lang') || 'en-IN';
  rec.interimResults = true;
  rec.continuous = false;
  let listening = false;

  const stopLook = () => {
    listening = false;
    send.setAttribute('aria-pressed', 'false');
    send.style.background = '#016abe';
  };
  rec.onresult = (e) => {
    let heard = '';
    for (const r of e.results) heard += r[0].transcript;
    if (input) input.value = heard.trim();
  };
  rec.onerror = (e) => {
    stopLook();
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      // Refused the microphone: say so once, and leave the typed path working.
      if (caption) caption.textContent = 'You did not allow the microphone, so type the question instead.';
    }
  };
  rec.onend = () => {
    stopLook();
    // Speaking then falling silent means "send it" -- a farmer mid-field should
    // not have to find a second button afterwards.
    if (input?.value.trim()) submit();
  };

  send.setAttribute('aria-pressed', 'false');
  acts(send, 'Speak your question', () => {
    if (listening) { rec.stop(); return; }
    if (input?.value.trim()) { submit(); return; }   // something typed: send it
    try {
      rec.start();
      listening = true;
      send.setAttribute('aria-pressed', 'true');
      send.style.background = '#a71930';             // recording, same red as Record
    } catch { stopLook(); }
  });

  // home.js's "Speak instead of typing" sets this and nothing ever read it.
  if (sessionStorage.getItem('pv.listen') === '1') {
    sessionStorage.removeItem('pv.listen');
    setTimeout(() => send.click(), 250);
  }
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
