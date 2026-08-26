// Settings, and the record of what we hold.
//
// Served live this page had exactly ONE control on it: "Sign out". Seven
// language chips, all three toggles, "Remove it to free space", "Change your
// code" and "Your videos and money records" were inert -- and the language
// picker is the worst of them, because setup-language.js was already driving
// the identical chips, so the same component worked on one screen and was dead
// on the other. It is one shared module now.
import { requireUser, api, session, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { appChrome } from '../chrome.js';
import { acts, goes, press, asField } from '../wire.js';
import { languagePicker } from '../lang.js';

const ctx = requireUser('profile');

// What each toggle actually controls, and what it is honest to claim.
const SWITCHES = [
  {
    match: /wi-?fi only/i,
    key: 'pv.wifiOnly',
    // A page cannot hold an upload back until the handset has wi-fi -- that
    // needs a service worker with a background sync queue, and there is none.
    // So the preference is stored and the screen says what it does and does
    // not do, rather than implying a queue that is not there.
    note: ['This is remembered, not enforced yet',
           'Nothing here can hold an upload back until you have wi-fi, that needs the app to keep '
           + 'working after you close it, which it does not do yet. Your choice is saved for when it can.'],
  },
  {
    // Real, and cheap: browser zoom on the page root enlarges px type as well
    // as rem, which matters because these boards are drawn in px throughout.
    // Safe to ship because measure.mjs already proves every board survives
    // 130% scale without clipping -- clip@130% is 0 on all 25 of them.
    match: /bigger text/i,
    key: 'pv.bigText',
    zoom: 1.3,
    note: null,
  },
  {
    match: /tell me when|notify/i,
    key: 'pv.notify',
    note: null,          // the messages screen is where these land; nothing to disclaim
  },
  {
    match: /keep .*on this phone|keep a copy/i,
    key: 'pv.keepLocal',
    note: null,
  },
];

// Applied on every page, not just this one, or the setting would be a note to
// itself. Kept here because this is the screen that owns it.
export function applyBigText() {
  const on = localStorage.getItem('pv.bigText') === '1';
  document.documentElement.style.zoom = on ? '1.3' : '';
}

if (ctx) {
  const root = ctx.root;
  appChrome(root, { back: 'home', user: ctx.user });

  load(root, async () => {
    const me = await api.auth.me();
    const initial = (me.name || '?').trim()[0].toUpperCase();
    // Marked, because the nav bar now has an initial too and it comes first in
    // document order: an unmarked lookup set the bar's and left the card's as
    // the artboard's A.
    const avatar = root.querySelector('[data-avatar]') || [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && /^[A-Z]$/.test(d.textContent.trim()));
    if (avatar) avatar.textContent = initial;
    bind(root, {
      name: me.name,
      phone: me.phone || 'No number on file',
      where: me.location || 'No district on file',
      // the model is a local file, so its size is a fact about this handset
      model: { state: localStorage.getItem('pv.model')
        ? '173 MB downloaded · works offline'
        : 'Not downloaded yet' },
    });

    // ---- language ----------------------------------------------------
    const langRow = root.querySelector('.lang, .langOn');
    languagePicker(root, langRow?.parentElement?.parentElement || null);

    // ---- the switches ------------------------------------------------
    for (const track of root.querySelectorAll('[data-toggle]')) {
      const row = track.closest('.row, .row2') || track.parentElement;
      const label = row?.querySelector('.lab')?.textContent.trim() || 'This setting';
      const spec = SWITCHES.find(s => s.match.test(label));
      const knob = track.querySelector('[data-knob]');

      // The board drew each switch in whatever state read best; the stored
      // preference decides it, and an unset one starts off.
      let on = spec ? localStorage.getItem(spec.key) === '1' : false;
      // The whole row is the switch, so the row carries the role and the state.
      // Leaving role="switch" on the pill after the row took over the click made
      // it a switch with no name and no way to focus it -- worse than either.
      const swap = (row && row !== track) ? row : track;
      const paint = () => {
        track.style.background = on ? '#016abe' : '#c3bcb6';
        if (knob) knob.style.transform = on ? 'translateX(20px)' : 'translateX(0)';
        swap.setAttribute('aria-checked', String(on));
      };
      swap.setAttribute('role', 'switch');
      if (swap !== track) {
        track.removeAttribute('role');
        track.setAttribute('aria-hidden', 'true');
      }
      const val = row?.querySelector('.val');
      const describe = () => {
        if (spec?.zoom && val) val.textContent = on ? 'Currently larger' : 'Currently normal';
      };
      paint();
      describe();

      let notice = null;
      // A 50x30 pill is under the 44pt touch minimum and stretching it into an
      // oval to satisfy the number would be worse. The row is the control, the
      // way a Settings row is on a phone: the label toggles it too. `row` above
      // is the same element -- there is no second one to declare.
      acts(swap, label, () => {
        on = !on;
        if (spec) localStorage.setItem(spec.key, on ? '1' : '0');
        paint();
        describe();
        if (spec?.zoom) {
          applyBigText();
          // say what it did, because the page moving under you needs a reason
          if (!notice) { notice = document.createElement('div'); row.after(notice); }
          if (on) state(notice, 'waiting', 'Everything is bigger now',
            'Every screen is enlarged, not just this one. Turn it off here to go back.');
          else notice.replaceChildren();
        }
        if (!spec?.note) return;
        if (!notice) { notice = document.createElement('div'); row.after(notice); }
        if (on) state(notice, 'waiting', spec.note[0], spec.note[1]);
        else notice.replaceChildren();
      });
    }

    // ---- the leaf model ----------------------------------------------
    // "Remove it to free space" is 173 MB sitting on a handset that may not
    // have it to spare, so it has to work. It asks once.
    const remove = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Remove it to free space');
    if (remove) {
      if (!localStorage.getItem('pv.model')) {
        remove.remove();                 // nothing downloaded, nothing to free
      } else {
        let armed = false;
        acts(remove, 'Remove the leaf checker to free space', () => {
          if (!armed) {
            armed = true;
            remove.textContent = 'Tap again to remove it';
            setTimeout(() => {
              if (!armed || !remove.isConnected) return;
              armed = false;
              remove.textContent = 'Remove it to free space';
            }, 4000);
            return;
          }
          localStorage.removeItem('pv.model');
          bind(root, { model: { state: 'Not downloaded yet' } });
          remove.remove();
        });
      }
    }

    // ---- the two rows that lead somewhere ----------------------------
    const rowFor = (text) => [...root.querySelectorAll('.lab')]
      .find(d => d.textContent.trim() === text)?.closest('.row, .row2');

    // ---- changing the code -------------------------------------------
    // This row said "there is no way to change the code in the app yet". There
    // is: POST /auth/change-password has been there all along and api.js
    // already wrapped it. The route demanded eight characters while
    // registration demands six, so the one credential the product lets a
    // farmer choose was refused by the route that changes it. Both are six now.
    const isFarmer = ctx.user.role === 'farmer';
    // (the wording follows the role: a farmer picks a code, others a password)

    const codeRow = rowFor('Change your code');
    if (codeRow) acts(codeRow, 'Change your code', () => {
      if (codeRow.nextElementSibling?.hasAttribute?.('data-codeform')) {
        codeRow.nextElementSibling.remove();
        return;
      }
      const form = document.createElement('div');
      form.setAttribute('data-codeform', '');
      form.style.cssText = 'background: #f6f3ef; padding: 16px 18px;'
        + ' box-shadow: inset 0 -1px 0 #c3bcb6;';

      const line = (text, css) => {
        const d = document.createElement('div');
        d.style.cssText = css;
        d.textContent = text;
        return d;
      };
      const field = (labelText, name) => {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'margin-top: 12px;';
        wrap.append(line(labelText, 'font-size: 14.5px; font-weight: 600;'));
        const box = document.createElement('div');
        box.style.cssText = 'background: #fff; min-height: 52px; margin-top: 6px;'
          + ' display: flex; align-items: center; padding: 0 14px;'
          + ' box-shadow: inset 0 0 0 1px #c3bcb6;';
        const slot = document.createElement('div');
        slot.style.cssText = 'font-size: 17px; flex: 1 1 auto; min-width: 0;';
        box.append(slot);
        wrap.append(box);
        const input = asField(slot, {
          type: 'password', name,
          autocomplete: name === 'current-password' ? 'current-password' : 'new-password',
          // inputMode only, never maxLength. Capping this at six truncated
          // "password123" to "passwo" and the route rightly said the current
          // password was wrong -- a farmer whose credential is longer than six
          // could not have got past this box at all.
          inputMode: isFarmer ? 'numeric' : undefined,
          placeholder: isFarmer ? 'Six numbers or more' : 'At least six characters',
          label: labelText,
        });
        return { wrap, input };
      };

      const now = field('What you use now', 'current-password');
      const next = field('What you want instead', 'new-password');
      const button = line('Change it',
        'background: #1d1a17; color: #fff; min-height: 52px; margin-top: 14px;'
        + ' display: flex; align-items: center; justify-content: center;'
        + ' font-size: 16px; font-weight: 600;');
      const note = document.createElement('div');

      form.append(
        line('Six characters or more. Changing it signs you out on every other phone.',
          'font-size: 14px; line-height: 1.5; color: #4a443d;'),
        now.wrap, next.wrap, button, note,
      );
      codeRow.after(form);
      press(root);

      acts(button, 'Change it', async () => {
        const a = now.input?.value || '';
        const b = next.input?.value || '';
        if (!a || !b) {
          return state(note, 'waiting', 'Both boxes',
            'We need the one you use now and the one you want instead.');
        }
        if (b.length < 6) {
          return state(note, 'waiting', 'Six or more',
            'Anything shorter than six characters is refused.');
        }
        const was = button.textContent;
        button.textContent = 'Changing…';
        try {
          await api.auth.changePassword(a, b);
          button.remove();
          state(note, 'proved', 'Changed',
            'Use the new one next time. Every other phone signed in as you is signed out.');
        } catch (err) {
          button.textContent = was;
          state(note, 'failed', 'Not changed', err.message);
        }
      });
    });

    // ---- the export ---------------------------------------------------
    // "Download everything we hold" navigated to the plots list. It downloads
    // now, built in the browser from the same routes the screens already use,
    // so it exposes nothing that was not already this account's to read.
    const dlRow = rowFor('Your videos and money records');
    if (dlRow) acts(dlRow, 'Download everything we hold', async () => {
      let holder = dlRow.nextElementSibling?.hasAttribute?.('data-note')
        ? dlRow.nextElementSibling : null;
      if (!holder) {
        holder = document.createElement('div');
        holder.setAttribute('data-note', '');
        dlRow.after(holder);
      }
      state(holder, 'waiting', 'Collecting it', 'Built on your phone. Nothing is sent anywhere.');

      const id = ctx.user._id || ctx.user.id;
      const safe = (p) => p.catch(() => null);
      const [me, videos, seasons, investments, purchases, notes] = await Promise.all([
        safe(api.auth.me()),
        safe(api.videos.mine(id)),
        safe(api.projects.mine(id)),
        safe(api.investments.mine(id)),
        safe(api.purchases.asBuyer(id)),
        safe(api.notifications.mine(id)),
      ]);
      const doc = {
        exported: new Date().toISOString(),
        about: 'Everything Pestivid holds about this account, as the app can read it.',
        you: me, videos, seasons, investments, purchases, notifications: notes,
      };
      const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
      const href = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = href;
      a.download = 'pestivid-'
        + String(me?.name || 'account').replace(/[^a-z0-9]+/gi, '-').toLowerCase() + '.json';
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 4000);

      const n = (list) => (list || []).length;
      state(holder, 'proved', 'Downloaded',
        n(videos) + ' videos, ' + n(seasons) + ' seasons, ' + n(investments) + ' investments and '
        + n(purchases) + ' purchases, as a JSON file in your downloads.');
    });

    const out = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Sign out');
    if (out) acts(out, 'Sign out', () => { session.clear(); location.href = './signin.html'; });

    press(root);
  });
}
