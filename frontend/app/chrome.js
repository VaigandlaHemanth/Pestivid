// The app chrome that repeats on almost every screen: back, messages, you.
//
// The back chevron is drawn on 13 artboards, 17 times, and was wired on none of
// them -- 22px of SVG with no target around it and no handler on it. Rather
// than repeat that in every page module, the boards now mark the three roles
// (data-chrome="back" | "mail" | "you") and this wires whichever are present.
import { goes, acts } from './wire.js';
import { api } from './api.js';

/**
 * @param root  the page root from wire()/requireUser()
 * @param opts.back  where the chevron goes. A slug, or 'history' to go back.
 *                   Prefer a slug: history.back() from a page opened directly
 *                   leaves the person on whatever was there before the app.
 * @param opts.user  the signed-in user, for the unread count.
 */
export function appChrome(root, opts = {}) {
  if (!root) return;

  const back = root.querySelector('[data-chrome="back"]');
  if (back) {
    // Back means back to where you came FROM. It used to mean "go to the slug
    // this page happens to declare", so opening a plot from the profile screen
    // and pressing back landed you somewhere you had never been. The declared
    // slug is now only the fallback -- for a screen opened cold, from a link or
    // a bookmark, where there is no history of ours to return to.
    const fallback = (opts.back && opts.back !== 'history') ? opts.back : 'home';
    acts(back, 'Back', () => {
      const cameFromUs = document.referrer
        && new URL(document.referrer, location.href).host === location.host
        && !/\/(signin|signup)\.html/.test(document.referrer);
      if (history.length > 1 && cameFromUs) history.back();
      else location.href = `./${fallback}.html`;
    });
  }

  const mail = root.querySelector('[data-chrome="mail"]');
  if (mail) {
    goes(mail, 'messages', 'Messages');
    // The badge showed the artboard's number on every screen. There is no
    // unread-count route, so it counts unread notifications -- which is what
    // the envelope leads to. No count, no badge: a red dot that always says
    // something is waiting teaches people to ignore it.
    const badge = mail.querySelector('[data-readout]');
    const id = opts.user && (opts.user._id || opts.user.id);
    if (badge && id) {
      api.notifications.mine(id)
        .then(list => {
          const n = (list || []).filter(x => !x.read && !x.isRead).length;
          if (n > 0) badge.textContent = n > 9 ? '9+' : String(n);
          else badge.remove();
        })
        .catch(() => badge.remove());
    } else if (badge) {
      badge.remove();
    }
  }

  const you = root.querySelector('[data-chrome="you"]');
  if (you) goes(you, 'profile', 'Your profile');

  deskNav(root, opts.user);
}

/**
 * The desktop header: wordmark, two or three destinations, and the signed-in
 * person.
 *
 * Every page wired this itself, by searching for its own labels -- and invest.js
 * wired none of it, so an investor who landed on Browse could not reach
 * Portfolio or Messages at all. The avatar was wired nowhere but the farmer's
 * phone chrome, so a buyer could not open their profile.
 *
 * One place, keyed on the label, so a page cannot forget. The label for the page
 * you are already on is not made into a link -- it is marked as the current one,
 * which is what a screen reader needs and what the underline already says.
 */
const DEST = {
  Browse: 'invest',
  Portfolio: 'portfolio',
  Messages: 'messages',
  'Buy produce': 'market',
  'Lots for sale': 'market',
  'My orders': 'orders',
  'What you bought': 'orders',
};

export function deskNav(root, user) {
  const here = (document.body.dataset.page || '').trim();

  // The header found by GEOMETRY, not by guessing a selector. The first element
  // matching div[style*="justify-content: space-between"] on the portfolio page
  // is a figures row further down, so a selector-based lookup wired nothing
  // there at all. Anything sitting in the top 110px of the page is the header.
  const inHeader = (el) => {
    const r = el.getBoundingClientRect();
    return r.top >= 0 && r.top < 110 && r.height > 0;
  };
  const labels = [...root.querySelectorAll('div, span, a')]
    .filter(el => inHeader(el))
    .filter(el => {
      const t = el.textContent.trim();
      if (!(t in DEST) && t !== 'Pestivid') return false;
      // The outermost element whose whole text is the label -- not its wrapper,
      // and not a child that repeats it.
      const kid = el.querySelector('div, span');
      return !(kid && kid.textContent.trim() === t);
    });

  const HOME = { farmer: 'home', investor: 'invest', buyer: 'market', admin: 'admin' };
  for (const el of labels) {
    const t = el.textContent.trim();
    const dest = t === 'Pestivid' ? (HOME[user?.role] || 'home') : DEST[t];
    if (!dest) continue;
    if (dest === here) { el.setAttribute('aria-current', 'page'); continue; }
    if (el.hasAttribute('data-go') || el.closest('[data-act]')) continue;
    goes(el, dest, t === 'Pestivid' ? 'Pestivid, home' : t);
  }

  // The avatar, or the name beside it. A buyer had no way to their own profile.
  const avatar = [...root.querySelectorAll('div')]
    .filter(el => inHeader(el))
    .find(d => /border-radius: 1[4-9]px|border-radius: 2[0-9]px/.test(d.getAttribute('style') || ''));
  if (avatar && here !== 'profile' && !avatar.closest('[data-act]')) {
    goes(avatar, 'profile', 'You and your settings');
  }
}

