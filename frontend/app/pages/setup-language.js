// The first screen anybody sees, before there is an account.
//
// Choosing a language is real and it persists. What it cannot yet do is change
// the words, because only the English strings exist — so the screen says that
// rather than switching to Telugu and showing English anyway.
import { wire, sayable, acts, press } from '../wire.js';
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

  // Read-aloud used to refuse: the copy promised a recorded human voice, which
  // nobody has recorded, so the button explained itself instead of speaking.
  // Home now reads aloud with the device voice, gated on a voice for the
  // language existing -- and two screens taking opposite positions on the same
  // question is the real inconsistency. So this one speaks too, the label no
  // longer implies a person read it, and where the device has no voice the
  // button is removed rather than left to explain.
  const aloud = root.querySelector('[data-aloud]')?.parentElement;
  const heading = root.querySelector('h1, [style*="font-size: 26px"], [style*="font-size: 25px"]');
  const line = [heading?.textContent, ...[...root.querySelectorAll('.lang, .langOn')].map(c => c.textContent)]
    .filter(Boolean).map(x => x.trim()).join('. ');
  if (aloud && !sayable(aloud, line, 'Read this screen aloud')) aloud.remove();

  press(root);
}
