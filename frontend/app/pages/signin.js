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

// The label is a leaf div inside the filled box, and the box is what looks like
// the button, so the box is what takes the handler. Its PARENT was taking it
// before, which is the whole card: clicking the email field submitted the form.
const label = oneByText('Sign in', root);
const box = label?.parentElement;
const button = box && getComputedStyle(box).backgroundColor !== 'rgba(0, 0, 0, 0)' ? box : label;
button?.setAttribute('data-act', '');
button?.setAttribute('role', 'button');
if (button) button.tabIndex = 0;

// Errors go in their own slot inside the card. state() replaces whatever it is
// handed, so handing it the page root deleted the form and left a lone message
// on an empty page.
const errorSlot = document.createElement('div');
button?.after(errorSlot);
const fail = (head, body) => state(errorSlot, 'failed', head, body);
const clearError = () => errorSlot.replaceChildren();

// "Show" has to do something or it should not be drawn.
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

let busy = false;
async function submit() {
  if (busy || !email || !pass || !button) return;
  if (!email.value.trim() || !pass.value) {
    return fail('Both fields, please', 'We cannot check an address without a password.');
  }
  busy = true;
  clearError();
  const was = label.textContent;
  label.textContent = 'Signing in…';
  try {
    const r = await api.auth.login(email.value.trim(), pass.value);
    session.set(r.token, r.user);
    const home = { farmer: 'home', investor: 'invest', buyer: 'market', admin: 'admin' }[r.user.role] || 'home';
    location.href = `./${home}.html`;
  } catch (err) {
    label.textContent = was;
    const bad = err.status === 400 || err.status === 401;
    // Which of the two was wrong is not said, on purpose: saying it tells
    // somebody guessing whether the address exists.
    fail(bad ? 'That did not match' : 'Could not sign you in',
         bad ? 'Check the address and the password, then try again.' : err.message);
  } finally { busy = false; }
}

button?.addEventListener('click', submit);
for (const f of [email, pass]) {
  f?.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
  f?.addEventListener('input', clearError);
}
// "Create an account" pointed nowhere; the page it needs now exists.
const create = oneByText('Create an account', root);
if (create) { create.setAttribute('data-act', ''); create.dataset.go = 'signup'; }
// There is no phone app. A farmer signs in on this same site, with a phone
// number instead of an address, so the link goes to that screen.
const farmerIn = oneByText('Farmer sign-in', root);
if (farmerIn) { farmerIn.setAttribute('data-act', ''); farmerIn.dataset.go = 'signin-farmer'; }

// "Forgot?" was link-coloured, bold, and pointed at nothing: there is no reset
// route in this product. Saying that beats a link that swallows the click --
// and it is the same answer the profile screen gives for the farmer's code.
const forgot = oneByText('Forgot?', root);
if (forgot) {
  forgot.setAttribute('data-act', '');
  forgot.addEventListener('click', () => {
    const slot = forgot.closest('div')?.parentElement;
    let holder = slot?.querySelector('[data-forgot]');
    if (!holder) {
      holder = document.createElement('div');
      holder.setAttribute('data-forgot', '');
      slot?.after(holder);
    }
    state(holder, 'waiting', 'No reset from here yet',
      'There is no way to reset a password in the app yet. Write to us and we will do it by hand — '
      + 'we would rather say that than open a screen that cannot finish.');
  });
}

email?.focus();
