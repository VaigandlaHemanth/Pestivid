// The first screen anybody sees, before there is an account.
//
// Choosing a language is real and it persists. What it cannot yet do is change
// the words, because only the English strings exist — so the screen says that
// rather than switching to Telugu and showing English anyway.
import { wire } from '../wire.js';
import { state } from '../bind.js';

const LANGS = {
  English: 'en', 'తెలుగు': 'te', 'हिन्दी': 'hi', 'தமிழ்': 'ta',
  'বাংলা': 'bn', 'ಕನ್ನಡ': 'kn', 'मराठी': 'mr',
};

const root = wire('setup-language');
if (root) {
  const chips = [...root.querySelectorAll('.lang, .langOn')];
  const current = localStorage.getItem('pv.lang') || 'en';
  const notice = document.createElement('div');

  const paint = (code) => {
    chips.forEach((c) => {
      const mine = LANGS[c.textContent.trim()] === code;
      c.className = mine ? 'langOn' : 'lang';
      c.setAttribute('aria-checked', String(mine));
    });
    document.documentElement.lang = code;
    if (code === 'en') { notice.replaceChildren(); return; }
    state(notice, 'waiting', 'The words are still in English',
      'Your choice is saved and the app will use it as soon as the translations exist. Showing you an English screen and calling it Telugu would be worse than admitting this.');
  };

  chips.forEach((c) => {
    c.setAttribute('data-act', '');
    c.setAttribute('role', 'radio');
    c.tabIndex = 0;
    c.addEventListener('click', () => {
      const code = LANGS[c.textContent.trim()];
      if (!code) return;
      localStorage.setItem('pv.lang', code);
      paint(code);
    });
  });

  root.querySelector('div[style*="flex-grow: 1"]')?.after(notice);
  paint(current);

  const next = [...root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'Continue');
  next?.setAttribute('data-act', '');
  next?.addEventListener('click', () => { location.href = './setup-identity.html'; });

  // read-aloud is a recorded voice we do not have; say so rather than firing
  // the robot speech synthesiser the copy promises it is not
  const aloud = [...root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'Read this to me');
  aloud?.setAttribute('data-act', '');
  aloud?.addEventListener('click', () => {
    const holder = document.createElement('div');
    aloud.closest('div[style*="flex-shrink: 0"]')?.prepend(holder);
    state(holder, 'waiting', 'There is no recording yet',
      'The screen promises a recorded human voice, not a robot one, and we have not recorded it. Using the phone’s synthesiser here would break that promise quietly.');
  });
}
