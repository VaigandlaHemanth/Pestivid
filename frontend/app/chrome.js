// The app chrome that repeats on almost every screen: back, messages, you.
//
// The back chevron is drawn on 13 artboards, 17 times, and was wired on none of
// them -- 22px of SVG with no target around it and no handler on it. Rather
// than repeat that in every page module, the boards now mark the three roles
// (data-chrome="back" | "mail" | "you") and this wires whichever are present.
import { goes, acts } from './wire.js';
import { api, noticeForRole } from './api.js';

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
/* What each destination is CALLED, so the link can say where it goes.
 *
 * The words are the ones those screens use about themselves, not invented
 * synonyms: "My plots" is the nav word for plots, "Film your field" is the
 * record page's own heading.
 */
const BACKWORD = {
  plots: 'My plots', money: 'Money', home: 'Home', record: 'Film your field',
  messages: 'Chat', invest: 'Browse', market: 'Buy produce', orders: 'My orders',
  portfolio: 'Portfolio', admin: 'Flagged', signin: 'Sign in', profile: 'You',
};

export function wireBack(root, fallback) {
  const back = root?.querySelector('[data-chrome="back"]');
  if (!back) return;
  if (fallback) back.dataset.backto = fallback;
  const to = back.dataset.backto || 'home';

  /* IT GOES WHERE IT SAYS IT GOES.
   *
   * This used to prefer history.back() and fall back to the slug, which was the
   * right call for a bare chevron: a chevron promises nothing in particular, so
   * "wherever you came from" is the most useful thing it can mean.
   *
   * The control is a labelled link now -- it reads "My plots" -- and a link that
   * says My plots and lands you on the profile screen because that is where you
   * happened to come from is simply lying. The browser's own back button already
   * does "where I came from", on every page, for everyone. This does the thing
   * it is named after.
   */
  const word = BACKWORD[to] || 'Home';
  const slot = back.querySelector('[data-backword]');
  if (slot) slot.textContent = word;
  if (back.tagName === 'A') back.setAttribute('href', `./${to}.html`);
  back.setAttribute('aria-label', `Back to ${word}`);
  if (back.dataset.backWired) return;
  back.dataset.backWired = '1';
  // A drawn <div> on a board not yet moved over still needs a handler; a real
  // link does not, and must not have one -- it would break middle-click and
  // open-in-new-tab, which is the whole reason it is an anchor.
  if (back.tagName !== 'A') {
    acts(back, `Back to ${word}`, () => { location.href = `./${to}.html`; });
  }
}

export function appChrome(root, opts = {}) {
  if (!root) return;

  wireBack(root, (opts.back && opts.back !== 'history') ? opts.back : null);

  const mail = root.querySelector('[data-chrome="mail"]');
  if (mail) {
    // The badge counts unread NOTICES, so the envelope goes where they are.
    // It used to land on Messages, which is what conflated the two.
    goes(mail, 'notifications', 'What has happened');
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
  // "Messages" was the word until the list and the conversation became one
  // two-pane page. The bar says Chat, and the old word still resolves because
  // three boards may not have caught up.
  Chat: 'messages',
  Messages: 'messages',
  'Buy produce': 'market',
  'Lots for sale': 'market',
  'My orders': 'orders',
  'What you bought': 'orders',
  // The reviewer's own screen. It had no word at all, which is why the admin's
  // bar was borrowing other people's.
  Flagged: 'admin',
};

/* THE NAV BELONGS TO WHOEVER IS SIGNED IN -------------------------------------
 *
 * messages and profile are shared by all four roles, but their boards
 * draw ONE bar and it is the farmer's. deskNav only wired the labels it found,
 * so a buyer who clicked Chat arrived on a page headed "My plots · Money ·
 * Chat" -- someone else's app, with their own avatar in the corner. The bar
 * is relabelled for the signed-in role before anything is wired.
 *
 * Three slots, because the boards draw three. Chat is last for everybody: it is
 * the one destination all four roles share.
 */
const NAV = {
  farmer:   ['My plots', 'Money', 'Chat'],
  investor: ['Browse', 'Portfolio', 'Chat'],
  buyer:    ['Buy produce', 'My orders', 'Chat'],
  /* The reviewer has TWO destinations, not three, and both of the three it used
   * to be were refused.
   *
   * 'Browse' is the investor's word and 'Buy produce' is the buyer's; invest.html
   * and market.html are gated to those roles, so an admin pressing either landed
   * on "Not your screen". Two thirds of this bar was a dead end into a refusal,
   * and the combination -- half investor, half buyer -- belonged to nobody.
   *
   * A reviewer's screens are the flagged queue and chat. The bell already
   * carries notifications and the avatar carries the profile, so those are not
   * slots. relabel() removes the third drawn slot rather than leaving a word on
   * it that goes somewhere this person is not allowed. */
  admin:    ['Flagged', 'Chat'],
};

function relabel(root, user, inHeader) {
  const want = NAV[user?.role];
  if (!want) return;
  /* The BAR, not "the top 110px". The page heading on the chat page is the word
   * "Chat", it sits just under the bar, and it is a known destination -- so
   * the geometric filter found four nav slots where there are three, the count
   * check bailed, and a buyer went on seeing the farmer's bar. */
  const realBar = root.querySelector('.appbar, .bar, header');
  const bar = realBar || root;
  const slots = [...bar.querySelectorAll('div, span, a')]
    .filter(el => inHeader(el))
    .filter(el => {
      const t = el.textContent.trim();
      if (!(t in DEST)) return false;
      const kid = el.querySelector('div, span');
      return !(kid && kid.textContent.trim() === t);
    });
  /* A role may want FEWER slots than the boards drew.
   *
   * The reviewer wants two where every board draws three, and the old
   * `slots.length !== want.length` bail left all three carrying the words they
   * were drawn with -- which is exactly how an admin ended up with the
   * investor's "Browse" and the buyer's "Buy produce", both of which refuse
   * them. Surplus slots are removed rather than left pointing somewhere this
   * person is not allowed to go.
   *
   * Only ever inside a REAL bar. Without one this falls back to the whole page
   * root, and the note above records what that costs: the chat page's own
   * heading is the word "Chat", it sits inside the geometric filter, and it
   * would be removed as a surplus nav slot. More slots than a role wants is a
   * bar this code does not understand, so it is left alone. */
  if (slots.length < want.length) return;     // an unfamiliar bar is left alone
  if (slots.length > want.length && !realBar) return;
  slots.forEach((el, i) => {
    if (i >= want.length) { el.remove(); return; }
    if (el.textContent.trim() !== want[i]) el.textContent = want[i];
  });
}

// Where a page falls back to when it was opened cold, per page. Only the ones
// whose answer is not simply the role's home need an entry.
const BACK = {
  record: 'plots', 'leaf-check': 'plots', sent: 'record', plot: 'plots',
  'report-harvest': 'money', 'ask-money': 'money',
  setup: 'signin', profile: 'home', ask: 'home',
  notifications: 'home',
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
  relabel(root, user, inHeader);

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
    /* Never the back link.
     *
     * This scanner claims anything in the top 110px whose text is a known
     * destination. The back control now sits in the content, just under the bar,
     * and says where it goes -- so on ask-money its word was "Money", DEST has
     * "Money", and it was wired as a nav destination on top of its own handler.
     * Pressing it on step three then stepped back to step two AND navigated to
     * money.html, because preventDefault stops the anchor, not another listener.
     * The back link is chrome with wiring of its own; this owns the nav words. */
    if (el.hasAttribute('data-go') || el.closest('[data-act]')
      || el.closest('[data-chrome]')) continue;
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

  /* And the bell itself, not just the number on it.
   *
   * appChrome() wires the bell, and only seven pages call appChrome. Everywhere
   * else it was an icon carrying a red unread count that did nothing when
   * pressed -- home, invest, portfolio, market and orders, which is most of what
   * an investor or a buyer ever looks at. The badge was already found here for
   * exactly this element; the door belongs next to it. */
  const bell = root.querySelector('[data-chrome="mail"]');
  if (bell && here !== 'notifications' && !bell.hasAttribute('data-act')
      && !bell.hasAttribute('data-go')) {
    goes(bell, 'notifications', 'What has happened');
  }

  const badge = root.querySelector('.appbar [data-readout], [data-chrome="mail"] [data-readout]');
  const id = user && (user._id || user.id);
  if (badge && id) {
    // No unread-count route, so this counts unread notifications -- which is
    // where the envelope goes. No count, no badge: a red dot that always says
    // something is waiting teaches people to ignore it.
    api.notifications.mine(id)
      .then((list) => {
        // The same filter the notices page applies, or the badge counts
        // broadcasts this reader will never be shown.
        const n = (list || [])
          .filter(x => noticeForRole(x, user?.role))
          .filter(x => !x.read && !x.isRead).length;
        if (n > 0) badge.textContent = n > 9 ? '9+' : String(n);
        else badge.remove();
      })
      .catch(() => badge.remove());
  } else if (badge) {
    badge.remove();
  }
}

