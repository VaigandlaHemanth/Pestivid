// The first screen anybody sees, before there is an account.
//
// Choosing a language is real and it persists. What it cannot yet do is change
// the words, because only the English strings exist — so the screen says that
// rather than switching to Telugu and showing English anyway.
import { wire, acts, press } from '../wire.js';
import { languagePicker } from '../lang.js';

const root = wire('setup-language');
if (root) {
  // The picker is shared with the profile screen -- it was implemented here and
  // nowhere else, so the identical chips on profile.html were dead.
  const notice = document.createElement('div');
  languagePicker(root, notice);
  root.querySelector('div[style*="flex-grow: 1"]')?.after(notice);

  const next = [...root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'Continue');
  if (next) acts(next.parentElement || next, 'Continue',
                 () => { location.href = './setup-identity.html'; });

  // The read-aloud button is gone from the board along with the rest of the
  // voice work, so there is nothing to wire here.

  press(root);
}
