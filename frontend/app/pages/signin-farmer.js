// Farmer sign-in: a phone number and the six-number code, on this same site.
//
// This screen used to say phone sign-in was not connected, which was true when
// there was no way to make a farmer account. setup-identity now registers one,
// keyed on an address derived from the phone number, so signing in is the same
// derivation run backwards. No server route was missing; the screen was.
//
// And there is no phone app to send anybody to. These are the farmer's screens,
// the same site laid out for a 360px handset.
import { api, session } from '../api.js';
import { wire, asField } from '../wire.js';
import { languagePicker } from '../lang.js';
import { state, oneByText } from '../bind.js';

const root = wire('signin-farmer');
if (root) {
  // The same picker the setup and profile screens use. These three chips were
  // plain divs, so they looked pressable and were not -- on the first screen a
  // farmer sees, where choosing a language is the whole point of them being
  // there. No notice panel here: this screen has no room and the choice is
  // restated on setup.
  languagePicker(root, null);

  const phone = asField(oneByText('+91 98765 43210', root) || oneByText('98765 43210', root), {
    type: 'tel', name: 'tel', autocomplete: 'tel', inputMode: 'numeric',
    placeholder: '98765 43210', label: 'Your phone number',
  });

  const code = asField(oneByText('••••••', root), {
    type: 'password', name: 'code', autocomplete: 'current-password',
    inputMode: 'numeric', maxLength: 6, placeholder: '••••••',
    label: 'Your six number code',
  });

  const errorSlot = document.createElement('div');
  code?.closest('div[style*="margin-top"]')?.after(errorSlot) || root.append(errorSlot);
  const fail = (h, d) => state(errorSlot, 'failed', h, d);

  const label = oneByText('Sign in', root);
  const button = label?.parentElement;
  button?.setAttribute('data-act', '');
  button?.setAttribute('role', 'button');
  if (button) button.tabIndex = 0;

  let busy = false;
  async function submit() {
    if (busy) return;
    const digits = (phone?.value || '').replace(/[^\d]/g, '').slice(-10);
    const pin = code?.value || '';
    if (digits.length < 10) return fail('That number looks short', 'Ten digits, no country code.');
    if (pin.length < 6) return fail('Six numbers', 'The code you picked when you made the account.');
    busy = true;
    errorSlot.replaceChildren();
    const was = label.textContent;
    label.textContent = 'Signing in…';
    try {
      // the same derivation setup-identity used when the account was made
      const r = await api.auth.login(`${digits}@phone.pestivid.local`, pin);
      session.set(r.token, r.user);
      location.href = './home.html';
    } catch (err) {
      label.textContent = was;
      fail(err.status === 400 || err.status === 401 ? 'That did not match' : 'Could not sign you in',
           err.status === 400 || err.status === 401
             ? 'Check the number and the six digits. If you have never made an account, make one first.'
             : err.message);
    } finally { busy = false; }
  }
  button?.addEventListener('click', submit);
  for (const f of [phone, code]) f?.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });

  const make = oneByText('Create an account', root) || oneByText('New here?', root);
  if (make) { make.setAttribute('data-act', ''); make.dataset.go = 'setup-language'; }
  phone?.focus();
}
