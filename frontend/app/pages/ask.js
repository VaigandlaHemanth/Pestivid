// The agronomy chat. The interesting state here is the one the design did not
// have: a free inference tier rate-limits per minute and per day, so this
// screen has to be able to say "not now" in words a farmer can act on.
import { requireUser, api, load, state } from './_guard.js';
import { oneByText, reason } from '../bind.js';

const ctx = requireUser('ask', ['farmer', 'investor', 'buyer', 'admin']);
if (!ctx) { /* requireUser has already sent them to sign in */ } else {

const root = ctx.root;
const thread = root.querySelector('.them')?.parentElement;
const field = oneByText('Type your question', root);
const send = root.querySelector('div[style*="background: #016abe"]');

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
      bubble('them', 'Kisan Call Centre — 1800 180 1551. Free, 6 am to 10 pm, and they speak Telugu.');
    }
  } finally { asking = false; }
}

// the drawn field becomes a real one, in the box it already occupies
if (field) {
  const input = document.createElement('input');
  const cs = getComputedStyle(field);
  input.type = 'text'; input.name = 'question'; input.enterKeyHint = 'send';
  input.placeholder = 'Type your question';
  input.style.cssText = `all: unset; display: block; width: 100%; font: ${cs.font}; color: #1d1a17;`;
  field.replaceChildren(input);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { ask(input.value.trim()); input.value = ''; }
  });
  send?.setAttribute('data-act', '');
  send?.addEventListener('click', () => { ask(input.value.trim()); input.value = ''; });
}

// the suggestion chips are real questions
for (const chip of root.querySelectorAll('.chip')) {
  chip.setAttribute('data-act', '');
  chip.setAttribute('role', 'button');
  chip.tabIndex = 0;
  chip.addEventListener('click', () => ask(chip.textContent.trim()));
}

}
