// Sign in. The only page allowed to talk to the API without a token.
import { api, session } from '../api.js';
import { wire, asField } from '../wire.js';
import { state, oneByText } from '../bind.js';

const root = wire('signin');

const email = asField(oneByText('charlie@example.com', root), {
  type: 'email', name: 'email', autocomplete: 'username',
  inputMode: 'email', placeholder: 'you@example.com', label: 'Email address',
});
const pass = asField(oneByText('••••••••', root), {
  type: 'password', name: 'password', autocomplete: 'current-password',
  placeholder: 'Your password', label: 'Password',
});

const button = oneByText('Sign in', root);
const shell = button?.parentElement;
button?.setAttribute('data-act', '');
button?.setAttribute('role', 'button');
if (button) button.tabIndex = 0;

// "Show" has to do something or it should not be there
const show = oneByText('Show', root);
if (show && pass) {
  show.setAttribute('data-act', '');
  show.setAttribute('role', 'switch');
  show.setAttribute('aria-checked', 'false');
  show.addEventListener('click', () => {
    const on = pass.type === 'password';
    pass.type = on ? 'text' : 'password';
    show.textContent = on ? 'Hide' : 'Show';
    show.setAttribute('aria-checked', String(on));
  });
}

let busy = false;
async function submit() {
  if (busy || !email || !pass) return;
  busy = true;
  const label = button.textContent;
  button.textContent = 'Signing in…';
  try {
    const r = await api.auth.login(email.value.trim(), pass.value);
    session.set(r.token, r.user);
    const home = { farmer: 'home', investor: 'invest', buyer: 'market', admin: 'admin' }[r.user.role] || 'home';
    location.href = `./${home}.html`;
  } catch (err) {
    button.textContent = label;
    const bad = err.status === 400 || err.status === 401;
    // Which of the two was wrong is not said, on purpose: saying it tells an
    // attacker whether the address exists.
    state(root, 'failed',
      bad ? 'That did not match' : 'Could not sign you in',
      bad ? 'Check the address and the password.' : err.message);
  } finally { busy = false; }
}

button?.addEventListener('click', submit);
shell?.addEventListener('click', submit);
for (const f of [email, pass]) {
  f?.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
}
email?.focus();

// The hero carries a "record of evidence" prop with a filename, a hash and a
// block number on it. On a public page those are invented figures dressed as
// proof, which is the one thing this product must not do. Replace them with a
