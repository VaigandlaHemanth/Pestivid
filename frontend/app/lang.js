// The language picker, used on two screens.
//
// setup-language.js drove these exact .lang / .langOn chips and profile.js did
// not, so the same component worked on the setup screen and was dead on the
// settings screen -- a farmer who came back to change language tapped seven
// times and nothing moved. One implementation, so that cannot happen again.
import { acts } from './wire.js';
import { state } from './bind.js';

export const LANGS = {
  English: 'en', 'తెలుగు': 'te', 'हिन्दी': 'hi', 'தமிழ்': 'ta',
  'বাংলা': 'bn', 'ಕನ್ನಡ': 'kn', 'मराठी': 'mr',
};

/**
 * Wire every language chip under `root`.
 *
 * The choice is real and it persists. What it cannot do yet is change the
 * words, because only the English strings exist -- so the screen says that
 * rather than switching to Telugu and showing English anyway. Both screens now
 * make the same admission in the same words, which they did not before.
 *
 * @param noticeInto where to put that admission. Omit to skip it.
 */
export function languagePicker(root, noticeInto) {
  const chips = [...root.querySelectorAll('.lang, .langOn')];
  if (!chips.length) return null;

  const notice = document.createElement('div');
  if (noticeInto) noticeInto.append(notice);

  const paint = (code) => {
    for (const c of chips) {
      const mine = LANGS[c.textContent.trim()] === code;
      c.className = mine ? 'langOn' : 'lang';
      c.setAttribute('aria-checked', String(mine));
    }
    document.documentElement.lang = code;
    if (!noticeInto) return;
    if (code === 'en') { notice.replaceChildren(); return; }
    state(notice, 'waiting', 'The words are still in English',
      'Your choice is saved and the app will use it as soon as the translations exist. '
      + 'Showing you an English screen and calling it Telugu would be worse than admitting this.');
  };

  for (const c of chips) {
    c.setAttribute('role', 'radio');
    acts(c, c.textContent.trim(), () => {
      const code = LANGS[c.textContent.trim()];
      if (!code) return;
      localStorage.setItem('pv.lang', code);
      paint(code);
    });
  }

  paint(localStorage.getItem('pv.lang') || 'en');
  return { chips, paint };
}
