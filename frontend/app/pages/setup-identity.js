// Creating the account: a name, a phone number, and a six-number code.
//
// The server stores an email and a password, so the phone number becomes the
// account name and the code becomes the password. The screen says that out
// loud, because a person who does not know what their account name is cannot
// get back in.
//
// KNOWN WEAKNESS, stated rather than hidden: six digits is a million guesses,
// which is thin for a sole credential. It is survivable only because sign-in is
// rate-limited server-side, and it wants replacing with a one-time code sent to
// the phone. That route does not exist yet.
import { api, session } from '../api.js';
import { wire, asField } from '../wire.js';
import { state } from '../bind.js';

const root = wire('setup-identity');

const byText = (t) => [...root.querySelectorAll('div')]
  .find(d => d.children.length === 0 && d.textContent.trim() === t);

if (root) {
  const name = asField(byText('Alice Farmer'),
    { type: 'text', name: 'name', autocomplete: 'name', placeholder: 'Your name', label: 'Your name' });
  const phone = asField(byText('98765 43210'),
    { type: 'tel', name: 'tel', autocomplete: 'tel', inputMode: 'numeric', placeholder: '98765 43210', label: 'Your phone number' });

  // six drawn boxes become one field, so a password manager and a numeric
  // keypad both work; the boxes stay as the visual
  const boxes = [...root.querySelectorAll('div[style*="flex: 1"][style*="height: 62px"]')];
  const strip = boxes[0]?.parentElement;
  let code = null;
  if (strip) {
    code = document.createElement('input');
    code.type = 'password';
    code.name = 'new-password';
    code.autocomplete = 'new-password';
    code.inputMode = 'numeric';
    code.maxLength = 6;
    code.setAttribute('aria-label', 'Your six number code');
    code.style.cssText = 'position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; border: 0;';
    strip.style.position = 'relative';
    strip.append(code);
    const paint = () => boxes.forEach((b, i) => {
      const filled = i < code.value.length;
      b.textContent = filled ? '•' : '';
      b.style.boxShadow = i === code.value.length
        ? 'inset 0 0 0 2px #1d1a17' : 'inset 0 0 0 1px #c3bcb6';
    });
    code.addEventListener('input', () => {
      code.value = code.value.replace(/[^\d]/g, '').slice(0, 6);
      paint();
    });
    strip.setAttribute('data-act', '');
    strip.setAttribute('aria-label', 'Your six number code');
    strip.addEventListener('click', () => code.focus());
    paint();
  }

  const holder = document.createElement('div');
  root.querySelector('div[style*="flex-grow: 1"]')?.append(holder);
  state(holder, 'empty', 'Your phone number is your account name',
    'That is how you sign in on a new phone. Write it down with the six numbers if you need to.');

  const go = byText('Continue');
  go?.setAttribute('data-act', '');
  go?.addEventListener('click', async () => {
    const digits = (phone?.value || '').replace(/[^\d]/g, '');
    if (!name?.value.trim()) return state(holder, 'waiting', 'We need your name', 'It is what a farmer or an investor sees.');
    if (digits.length < 10) return state(holder, 'waiting', 'That phone number looks short', 'Ten digits, no country code.');
    if ((code?.value || '').length !== 6) return state(holder, 'waiting', 'Six numbers', 'Pick a code you will remember and that is not your bank PIN.');

    const label = go.textContent;
    go.textContent = 'Creating…';
    try {
      const email = `${digits}@phone.pestivid.local`;
      await api.auth.register({ name: name.value.trim(), email, password: code.value, role: 'farmer', phone: digits });
      const r = await api.auth.login(email, code.value);
      session.set(r.token, r.user);
      location.href = './home.html';   // home carries the first-run state now
    } catch (err) {
      go.textContent = label;
      state(holder, 'failed',
        err.status === 409 ? 'That number already has an account' : 'The account was not created',
        err.status === 409 ? 'Sign in with it instead.' : err.message);
    }
  });
}
