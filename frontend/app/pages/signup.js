// Create an account, for an investor or a buyer.
//
// The farmer route is the phone flow; this is the desktop one, and it did not
// exist at all before. The order on the page is deliberate: the loss risk, then
// what we hold, then the fields. Nothing is created until the risk line is
// ticked, and that is enforced here rather than only drawn.
import { api, session } from '../api.js';
import { wire, asField } from '../wire.js';
import { state, oneByText } from '../bind.js';

const root = wire('signup');

const name = asField(oneByText('Ravi Kumar', root), {
  type: 'text', name: 'name', autocomplete: 'name',
  placeholder: 'Your name', label: 'Your name',
});
const email = asField(oneByText('ravi@example.com', root), {
  type: 'email', name: 'email', autocomplete: 'username',
  inputMode: 'email', placeholder: 'you@example.com', label: 'Email address',
});
const pass = asField(oneByText('••••••••', root), {
  type: 'password', name: 'new-password', autocomplete: 'new-password',
  placeholder: 'Eight or more', label: 'Password',
});

// role
let role = 'investor';
const cards = [...root.querySelectorAll('[data-role-pick]')];
const paintRole = () => cards.forEach((c) => {
  const mine = c.dataset.rolePick === role;
  c.style.boxShadow = mine ? 'inset 0 0 0 2px #1d1a17' : '';
  c.setAttribute('aria-checked', String(mine));
  const dot = c.querySelector('div[style*="border-radius: 10px"]');
  if (dot) {
    dot.style.background = mine ? '#1d1a17' : 'transparent';
    dot.style.boxShadow = mine ? '' : 'inset 0 0 0 2px #78716a';
  }
});
cards.forEach((c) => {
  c.setAttribute('data-act', '');
  c.setAttribute('role', 'radio');
  c.tabIndex = 0;
  c.addEventListener('click', () => { role = c.dataset.rolePick; paintRole(); });
});
paintRole();

// the acknowledgement, and the button it gates
const ack = root.querySelector('[data-ack]');
const ackBox = ack?.querySelector('div[style*="inset 0 0 0 2px #a71930"]');
const label = oneByText('Create the account', root);
const button = label?.parentElement;
let agreed = false;

const paintButton = () => {
  if (!button) return;
  button.style.background = agreed ? '#016abe' : '#c9ced4';
  if (label) label.style.color = agreed ? '#fff' : '#6b7278';
  button.setAttribute('aria-disabled', String(!agreed));
};
if (ack) {
  ack.setAttribute('data-act', '');
  ack.setAttribute('role', 'checkbox');
  ack.tabIndex = 0;
  ack.addEventListener('click', () => {
    agreed = !agreed;
    ack.setAttribute('aria-checked', String(agreed));
    if (ackBox) {
      // A filled red square with nothing in it does not read as "ticked".
      ackBox.style.background = agreed ? '#a71930' : '#fff';
      ackBox.innerHTML = agreed
        ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3.2" style="display:block;margin:3px"><path d="M5 12.5l4.5 4.5L19 7"></path></svg>'
        : '';
    }
    paintButton();
  });
}
// "Show" reveals the password on the sign-in screen and did nothing here, on
// the screen where somebody is TYPING a password for the first time and most
// wants to check it.
const show = oneByText('Show', root);
if (show && pass) {
  show.setAttribute('data-act', '');
  show.setAttribute('role', 'switch');
  show.setAttribute('aria-checked', 'false');
  show.addEventListener('click', (e) => {
    e.stopPropagation();
    const on = pass.type === 'password';
    pass.type = on ? 'text' : 'password';
    show.textContent = on ? 'Hide' : 'Show';
    show.setAttribute('aria-checked', String(on));
  });
}

button?.setAttribute('data-act', '');
button?.setAttribute('role', 'button');
if (button) button.tabIndex = 0;
paintButton();

const errorSlot = document.createElement('div');
button?.after(errorSlot);
const fail = (h, d) => state(errorSlot, 'failed', h, d);
const clearError = () => errorSlot.replaceChildren();

let busy = false;
button?.addEventListener('click', async () => {
  if (!agreed || busy) return;
  const n = name?.value.trim(), e = email?.value.trim(), pw = pass?.value || '';
  if (!n) return fail('We need a name', 'It is what a farmer sees when you fund their season.');
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(e || '')) return fail('That address does not look right', 'Check it and try again.');
  // the server's floor is 6; 8 is what the screen asks for, so hold the screen's line
  if (pw.length < 8) return fail('Eight characters or more', 'Longer is better. A passphrase beats a clever short one.');

  busy = true; clearError();
  const was = label.textContent;
  label.textContent = 'Creating…';
  try {
    await api.auth.register({ name: n, email: e, password: pw, role });
    const r = await api.auth.login(e, pw);
    session.set(r.token, r.user);
    location.href = role === 'buyer' ? './market.html' : './invest.html';
  } catch (err) {
    label.textContent = was;
    fail(err.status === 409 ? 'That address already has an account' : 'The account was not created',
         err.status === 409 ? 'Sign in with it instead.' : err.message);
  } finally { busy = false; }
});

for (const f of [name, email, pass]) f?.addEventListener('input', clearError);
const back = oneByText('Sign in', root);
if (back) { back.setAttribute('data-act', ''); back.dataset.go = 'signin'; }

name?.focus();
