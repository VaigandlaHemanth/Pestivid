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
import { acts, goes, press } from '../wire.js';
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
           'Nothing here can hold an upload back until you have wi-fi — that needs the app to keep '
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
    const avatar = [...root.querySelectorAll('div')]
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
      const paint = () => {
        track.style.background = on ? '#016abe' : '#c3bcb6';
        if (knob) knob.style.transform = on ? 'translateX(20px)' : 'translateX(0)';
        track.setAttribute('aria-checked', String(on));
      };
      track.setAttribute('role', 'switch');
      const val = row?.querySelector('.val');
      const describe = () => {
        if (spec?.zoom && val) val.textContent = on ? 'Currently larger' : 'Currently normal';
      };
      paint();
      describe();

      let notice = null;
      acts(track, label, () => {
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

    // Changing the code is changing the password, and there is no route for it.
    // Saying so beats a row that swallows the tap.
    const code = rowFor('Change your code');
    if (code) acts(code, 'Change your code', () => {
      let holder = code.nextElementSibling?.hasAttribute?.('data-note')
        ? code.nextElementSibling : null;
      if (!holder) {
        holder = document.createElement('div');
        holder.setAttribute('data-note', '');
        code.after(holder);
      }
      state(holder, 'waiting', 'Not from here yet',
        'There is no way to change the code in the app yet. Until there is, ask us and we will '
        + 'reset it — we would rather tell you that than open a screen that cannot finish.');
    });

    goes(rowFor('Your videos and money records'), 'plots', 'Your videos and money records');

    const out = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Sign out');
    if (out) acts(out, 'Sign out', () => { session.clear(); location.href = './signin.html'; });

    press(root);
  });
}
