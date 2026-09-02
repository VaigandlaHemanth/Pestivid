// Sign in. The only page allowed to talk to the API without a token.
import { api, session } from '../api.js';
import { wire, asField } from '../wire.js';
import { state, oneByText } from '../bind.js';

const root = wire('signin');

/* One field, one credential: the address the account was made with.
 *
 * It used to show `98765 43210` and tell a screen reader "your phone number or
 * email address", under a label reading "Email address" and a sentence saying
 * "sign in with the email address you made the account with". Three of those
 * four said email and the example said phone.
 *
 * The phone was not merely inconsistent, it was impossible. Nothing in this
 * codebase can create a phone account any more -- setup-identity, which did,
 * is gone -- and the database holds none. A farmer following the example was
 * being invited to type a credential that could only ever come back "no
 * account". `ravi@example.com` is the same example the create-account panel
 * below already uses.
 */
const who = asField(oneByText('ravi@example.com', root), {
  type: 'email', name: 'who', autocomplete: 'username',
  // you@, not ravi@: the board's specimen is a filled-in example, but a
  // placeholder speaks to the person about to type. The create-account panel on
  // this same page already says you@example.com, and two panels of one page
  // holding up two different examples is one of them being wrong.
  placeholder: 'you@example.com', label: 'Your email address',
});
const pass = asField(oneByText('••••••••', root), {
  type: 'password', name: 'password', autocomplete: 'current-password',
  placeholder: 'Your password', label: 'Your password',
});

// The label is a leaf div inside the filled box, and the box is what looks like
// the button, so the box is what takes the handler. Its PARENT was taking it
// before, which is the whole card: clicking the field submitted the form.
// The page's own heading now says "Sign in" too, and oneByText returns the
// FIRST match in document order -- so a text lookup here would have handed the
// handler to the h1 and left the most important button in the product dead.
// The board marks it.
const label = root?.querySelector('[data-submit]')
  ? [...root.querySelectorAll('[data-submit] div, [data-submit]')]
      .find(e => e.textContent.trim() === 'Sign in')
  : oneByText('Sign in', root);
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
  if (busy || !who || !pass || !button) return;
  if (!who.value.trim() || !pass.value) {
    return fail('Both boxes, please', 'We cannot look you up without the code or password that goes with it.');
  }
  busy = true;
  clearError();
  const was = label.textContent;
  label.textContent = 'Signing in…';
  try {
    /* Ten digits used to be turned into `<digits>@phone.pestivid.local`, the
     * address setup-identity gave a phone account. That page is retired, nothing
     * else can register one, and the database holds zero of them -- so the
     * branch could only ever build an address that does not exist and hand back
     * "no account". Dead code that fabricates a credential is worse than none:
     * it made a phone number look like a thing you could sign in with. */
    const r = await api.auth.login(who.value.trim(), pass.value);
    session.set(r.token, r.user);
    const home = { farmer: 'home', investor: 'invest', buyer: 'market', admin: 'admin' }[r.user.role] || 'home';
    location.href = `./${home}.html`;
  } catch (err) {
    label.textContent = was;
    const bad = err.status === 400 || err.status === 401;
    // Which of the two was wrong is not said, on purpose: saying it tells
    // somebody guessing whether the address exists.
    fail(bad ? 'That did not match' : 'Could not sign you in',
         bad ? 'Check the number or address, and the code or password, then try again.' : err.message);
  } finally { busy = false; }
}

button?.addEventListener('click', submit);
for (const f of [who, pass]) {
  f?.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
  f?.addEventListener('input', clearError);
}
// "Create an account" pointed nowhere; the page it needs now exists.
const create = oneByText('Create an account', root);
if (create) { create.setAttribute('data-act', ''); create.dataset.go = 'signup'; }

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
      'There is no way to reset a password in the app yet. Write to us and we will do it by hand, '
      + 'we would rather say that than open a screen that cannot finish.');
  });
}

who?.focus();
