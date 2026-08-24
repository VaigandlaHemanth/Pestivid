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
/* The back chevron, wired once wherever it is drawn -------------------------
 * Back means back to where you came FROM. It used to mean "go to the slug this
 * page happens to declare", so opening a plot from the profile screen and
 * pressing back landed you somewhere you had never been. The declared slug is
 * only the fallback now -- for a screen opened cold, from a link or a bookmark,
 * where there is no history of ours to return to.
 *
 * This lives on its own because only SEVEN pages call appChrome, so the chevron
 * on record, ask, leaf-check, setup and report-harvest had no handler at all:
 * five screens where the only way out of a dead end did nothing when pressed.
 * deskNav calls this for every page; a page with a better fallback than its
 * role's home can still say so, and the later call just updates the fallback.
 */
export function wireBack(root, fallback) {
  const back = root?.querySelector('[data-chrome="back"]');
  if (!back) return;
  if (fallback) back.dataset.backto = fallback;
  if (back.dataset.backWired) return;
  back.dataset.backWired = '1';
  acts(back, 'Back', () => {
    const cameFromUs = document.referrer
      && new URL(document.referrer, location.href).host === location.host
      && !/\/(signin|signup)\.html/.test(document.referrer);
    if (history.length > 1 && cameFromUs) history.back();
    else location.href = `./${back.dataset.backto || 'home'}.html`;
  });
}

export function appChrome(root, opts = {}) {
  if (!root) return;

  wireBack(root, (opts.back && opts.back !== 'history') ? opts.back : null);

  const mail = root.querySelector('[data-chrome="mail"]');
  if (mail) {
    goes(mail, 'messages', 'Messages');
    // The count is deskNav's, below: it runs on every page, and doing it here
    // as well meant seven pages asked the server for the same list twice.
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
  // the farmer's three, now that the farmer has the same bar as everybody else
  'My plots': 'plots',
  Money: 'money',
  Browse: 'invest',
  Portfolio: 'portfolio',
  Messages: 'messages',
  'Buy produce': 'market',
  'Lots for sale': 'market',
  'My orders': 'orders',
  'What you bought': 'orders',
};

// Where a page falls back to when it was opened cold, per page. Only the ones
// whose answer is not simply the role's home need an entry.
const BACK = {
  record: 'plots', 'leaf-check': 'plots', sent: 'record', plot: 'plots',
  payout: 'money', 'report-harvest': 'money', 'ask-money': 'money',
  thread: 'messages', setup: 'signin', profile: 'home', ask: 'home',
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
  wireBack(root, BACK[here] || HOME[user?.role] || 'home');
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

  /* The two readouts in the bar ------------------------------------------
   * appChrome() does the badge, but only seven pages call it, and NOTHING did
   * the avatar except home.js in its own copy. So the envelope showed the
   * artboard's "2" on record, ask, leaf-check and ask-money, and the avatar
   * showed the artboard's "A" on nine of the ten farmer screens whoever was
   * signed in. deskNav runs on every page, from wire.js, so both belong here
   * and appChrome no longer repeats the badge.
   */
  const initial = root.querySelector('.appbar [data-initial], [data-chrome="you"] [data-initial]');
  const name = (user?.name || '').trim();
  if (initial && name) initial.textContent = name[0].toUpperCase();

  const badge = root.querySelector('.appbar [data-readout], [data-chrome="mail"] [data-readout]');
  const id = user && (user._id || user.id);
  if (badge && id) {
    // No unread-count route, so this counts unread notifications -- which is
    // where the envelope goes. No count, no badge: a red dot that always says
    // something is waiting teaches people to ignore it.
    api.notifications.mine(id)
      .then((list) => {
        const n = (list || []).filter(x => !x.read && !x.isRead).length;
        if (n > 0) badge.textContent = n > 9 ? '9+' : String(n);
        else badge.remove();
      })
      .catch(() => badge.remove());
  } else if (badge) {
    badge.remove();
  }
}

