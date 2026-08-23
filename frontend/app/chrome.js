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
    if (!opts.back || opts.back === 'history') {
      acts(back, 'Back', () => {
        // Only trust history when it is ours. Otherwise the chevron would walk
        // somebody out of the app, which is not what a back arrow inside a
        // screen promises.
        if (history.length > 1 && document.referrer.includes(location.host)) history.back();
        else location.href = './home.html';
      });
    } else {
      goes(back, opts.back, 'Back');
    }
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
}
