// Sign in. The only page that may talk to the API without a token.
import { api, session } from '../api.js';
import { wire } from '../wire.js';
import { state } from '../bind.js';

const root = wire('signin');
const form = document.querySelector('body > div');

// The board draws fields, not inputs. Turn each drawn field into a real one so
// a password manager, an autofill and a keyboard all work, without moving a
// pixel: the input inherits the box it replaces.
function asInput(el, { type, name, autocomplete, inputmode, placeholder }) {
  if (!el || el.querySelector('input')) return el?.querySelector('input');
  const input = document.createElement('input');
  const cs = getComputedStyle(el);
  input.type = type; input.name = name; input.autocomplete = autocomplete;
  if (inputmode) input.inputMode = inputmode;
  input.placeholder = placeholder || '';
  input.value = el.textContent.trim().startsWith('•') ? '' : el.textContent.trim();
  input.style.cssText = `all: unset; display: block; width: 100%; box-sizing: border-box;
    font: ${cs.font}; color: ${cs.color}; letter-spacing: ${cs.letterSpacing};`;
  el.replaceChildren(input);
  return input;
}

const byText = (t) => [...form.querySelectorAll('div')]
  .find(d => d.children.length === 0 && d.textContent.trim() === t);

const email = asInput(byText('charlie@example.com'), { type: 'email', name: 'email', autocomplete: 'username', inputmode: 'email', placeholder: 'you@example.com' });
const pass = asInput(byText('••••••••'), { type: 'password', name: 'password', autocomplete: 'current-password', placeholder: 'Your password' });

const button = [...form.querySelectorAll('div')].find(d => d.textContent.trim() === 'Sign in' && d.children.length === 0);
if (button) {
  button.setAttribute('data-act', '');
  button.setAttribute('role', 'button');
  button.tabIndex = 0;
}

let busy = false;
async function submit() {
  if (busy) return;
  busy = true;
  const label = button.textContent;
  button.textContent = 'Signing in…';           // spinner state keeps the label
  try {
    const r = await api.auth.login(email.value.trim(), pass.value);
    session.set(r.token, r.user);
    const home = { farmer: 'home', investor: 'invest', buyer: 'market', admin: 'admin' }[r.user.role] || 'home';
    location.href = `./${home}.html`;
  } catch (err) {
    button.textContent = label;
    state(form, 'failed', err.status === 400 || err.status === 401
      ? 'That did not match' : 'Could not sign you in',
      err.status === 400 || err.status === 401
        ? 'Check the address and the password. We do not say which of the two was wrong, on purpose.'
        : err.message);
  } finally { busy = false; }
}

button?.addEventListener('click', submit);
form.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
