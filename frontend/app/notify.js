// A notice, the moment it arrives.
//
// The bell counted unread notices and the list showed them, but nothing told
// you one had just happened: a buyer's message or an investor's money landed
// silently, and you found out when you next pressed the bell. This asks the
// server every few seconds, keeps the badge honest on every page, and shows
// each new arrival as a banner at the top right -- the way a Mac shows one.
//
// What "the way a Mac shows one" means here, so nobody has to guess later:
//   - it comes in from the right edge and leaves by the same edge (§7 of the
//     apple-design notes: exit along the path you entered);
//   - frosted, not opaque: the page stays legible through it, and the banner
//     reads as a layer over the page rather than a box on it;
//   - the app mark on the left, a short title and one or two lines of body;
//   - it waits, then goes on its own. Hovering holds it. The little close mark
//     appears at the top-left corner on hover, where a Mac puts it;
//   - you can swipe it away to the right, and it follows the pointer 1:1,
//     resists if pushed the wrong way, and finishes at the speed you threw it;
//   - pressing it opens the thing it is about, if there is a screen for that.
//
// Under reduced motion the travel goes and the fade stays, which is what
// tokens.css already does for every other transition in this product.
import { api, noticeForRole, noticeKind, noticeDestination, whenShort } from './api.js';

const POLL_MS = 12000;      // often enough that a reply feels answered
const SHOW_MS = 6500;       // macOS banners hold for about six seconds
const MAX_ON_SCREEN = 3;
const STILL = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// One baseline per person PER TAB, kept across pages: the newest notice this
// tab has already been shown. Without it every page load would re-announce the
// same backlog, and eighteen unread notices would arrive as eighteen banners.
//
// sessionStorage, not localStorage. Shared across tabs, the first tab to poll
// moved the baseline past an arrival and every other tab then read it as old:
// measured with home and the chat open side by side, home raised the banner
// and the chat never learned a message had come. Each tab keeps its own.
const seenKey = (uid) => `pv.notice.seen.${uid}`;
const store = () => { try { return sessionStorage; } catch { return null; } };

/* Pages that handle a kind of notice themselves say so here, and get the
 * arrival as an event instead of a banner. The notices page IS the list, and
 * the chat already shows a message where it belongs. */
const listeners = new Set();
export function onNotice(fn) { listeners.add(fn); return () => listeners.delete(fn); }

const QUIET = {
  notifications: () => true,                       // the page is the answer
  messages: (n) => n.type === 'message',           // the chat shows it in place
};

/* ── the look, injected once ────────────────────────────────────────────── */
function styles() {
  if (document.getElementById('pv-toast-css')) return;
  const s = document.createElement('style');
  s.id = 'pv-toast-css';
  s.textContent = `
[data-toasts] { position: fixed; top: 72px; right: 16px; z-index: 1000;
  width: min(372px, calc(100vw - 32px)); display: flex; flex-direction: column;
  gap: 10px; pointer-events: none; }
.pv-toast { pointer-events: auto; position: relative; display: grid;
  grid-template-columns: 38px minmax(0, 1fr); gap: 12px; align-items: start;
  padding: 12px 14px 13px 12px; border-radius: 16px; color: #1d1a17;
  background: rgba(246, 243, 239, .8);
  -webkit-backdrop-filter: blur(28px) saturate(170%); backdrop-filter: blur(28px) saturate(170%);
  box-shadow: 0 0 0 1px rgba(29, 26, 23, .09), inset 0 1px 0 rgba(255, 255, 255, .65),
    0 2px 6px rgba(29, 26, 23, .08), 0 18px 48px rgba(29, 26, 23, .2);
  font: 400 14px/1.4 system-ui, "Anek Latin", "Noto Sans", sans-serif;
  cursor: pointer; touch-action: pan-y; user-select: none; -webkit-user-select: none;
  will-change: transform, opacity; outline-offset: 3px; }
.pv-toast .ic { width: 38px; height: 38px; border-radius: 10px; background: #1d1a17;
  display: flex; align-items: center; justify-content: center; }
.pv-toast .ic svg { width: 21px; height: 21px; }
.pv-toast .ttl { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; min-width: 0; }
.pv-toast .app { font-size: 13px; font-weight: 600; letter-spacing: .01em; }
.pv-toast .when { font-size: 12px; color: #605a53; flex-shrink: 0; font-variant-numeric: tabular-nums; }
.pv-toast .msg { margin-top: 1px; font-size: 14px; line-height: 1.4; overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
.pv-toast .x { position: absolute; top: -8px; left: -8px; width: 24px; height: 24px;
  border-radius: 12px; background: #f6f3ef; color: #1d1a17; display: flex; align-items: center;
  justify-content: center; box-shadow: 0 0 0 1px rgba(29, 26, 23, .14), 0 2px 6px rgba(29, 26, 23, .18);
  opacity: 0; transform: scale(.8); transition: opacity 120ms ease, transform 120ms ease; cursor: pointer; }
.pv-toast:hover .x, .pv-toast:focus-within .x, .pv-toast:focus-visible .x { opacity: 1; transform: none; }
@media (prefers-reduced-transparency: reduce) {
  .pv-toast { background: #f6f3ef; -webkit-backdrop-filter: none; backdrop-filter: none; } }
@media (prefers-contrast: more) {
  .pv-toast { background: #fffefb; box-shadow: 0 0 0 1.5px #1d1a17; } }
@media (max-width: 600px) {
  [data-toasts] { top: 8px; right: 8px; left: 8px; width: auto; } }`;
  document.head.append(s);
}

/* The glyphs, the same drawings the notices page uses, one per kind. */
const GLYPH = {
  person:  '<path d="M21 12a8.5 8.5 0 0 1-9 8.4 9 9 0 0 1-3.9-.9L3 21l1.9-5.1A8.5 8.5 0 0 1 12 3.5a8.5 8.5 0 0 1 9 8.5z"/>',
  money:   '<path d="M7 4h9"/><path d="M7 8.5h9"/><path d="M7 13h4.5c2.5 0 4.5-1.9 4.5-4.3S14 4 11.5 4"/><path d="M7 13l8 7"/>',
  listing: '<path d="M5 3h11l3 3v15H5z"/><path d="M16 3v3h3"/><path d="M8.5 12.5h7M8.5 16h4"/>',
  proved:  '<path d="M4.5 12.5l5 5L19.5 7"/>',
  wrong:   '<path d="M12 3 L22 20 H2 Z"/><path d="M12 9.5v5M12 17.5v.01"/>',
  bell:    '<path d="M18 8.5a6 6 0 0 0-12 0c0 5-2 6.5-2 6.5h16s-2-1.5-2-6.5"/><path d="M10.3 19a2 2 0 0 0 3.4 0"/>',
};
const TITLE = {
  person: 'Chat', money: 'Money', listing: 'Market', proved: 'Date proved', wrong: 'Something went wrong',
};
const STROKE = { proved: '#5fbe8a', wrong: '#f0a0aa' };

function stage() {
  let el = document.querySelector('[data-toasts]');
  if (el) return el;
  el = document.createElement('div');
  el.setAttribute('data-toasts', '');
  el.setAttribute('role', 'status');
  el.setAttribute('aria-live', 'polite');
  document.body.append(el);
  return el;
}

/* FLIP: mutate, then let everything that moved travel from where it was.
 * Prepending a banner would otherwise teleport the ones under it 10px plus a
 * banner's height downwards; a Mac slides them. */
function shifting(mutate) {
  const host = stage();
  const kids = [...host.children];
  const before = new Map(kids.map(k => [k, k.getBoundingClientRect().top]));
  mutate();
  if (STILL()) return;
  for (const k of kids) {
    if (!k.isConnected || k.dataset.leaving) continue;
    const d = before.get(k) - k.getBoundingClientRect().top;
    if (!d) continue;
    k.style.transition = 'none';
    k.style.transform = `translateY(${d}px)`;
    void k.offsetHeight;                       // commit the start position
    k.style.transition = 'transform var(--t-snappy, 568ms) var(--e-snappy, ease)';
    k.style.transform = '';
  }
}

/* Momentum, the way iOS projects a flick: where would it stop on its own. */
const project = (v, rate = 0.998) => (v / 1000) * rate / (1 - rate);
const rubber = (over, size, c = 0.55) => (over * size * c) / (size + c * Math.abs(over));

/**
 * One banner. `n` is a notice as the server sends it; `opts.role` decides the
 * destination; `opts.onOpen(n)` runs before navigation.
 */
export function banner(n, opts = {}) {
  styles();
  const host = stage();
  const kind = opts.kind || noticeKind(n);
  const dest = opts.kind ? null : noticeDestination(n, opts.role);

  const el = document.createElement('div');
  el.className = 'pv-toast';
  el.setAttribute('role', 'button');
  el.tabIndex = 0;
  el.setAttribute('aria-label', `${TITLE[kind] || 'Pestivid'}: ${n.message || n.title || ''}.`
    + (dest ? ` Opens ${dest}.` : ' Press to mark as read.'));
  const stroke = STROKE[kind] || '#f6f3ef';
  el.innerHTML = `
    <div class="ic"><svg viewBox="0 0 24 24" fill="none" stroke="${stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${GLYPH[kind] || GLYPH.bell}</svg></div>
    <div style="min-width: 0;">
      <div class="ttl"><span class="app"></span><span class="when"></span></div>
      <div class="msg"></div>
    </div>
    <div class="x" role="button" tabindex="-1" aria-label="Dismiss"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg></div>`;
  el.querySelector('.app').textContent = TITLE[kind] || 'Pestivid';
  el.querySelector('.when').textContent = opts.when || 'now';
  el.querySelector('.msg').textContent = n.message || n.title || '';

  // ---- arriving: from the right edge, on the product's snappy spring
  const still = STILL();
  el.style.opacity = '0';
  if (!still) el.style.transform = 'translateX(calc(100% + 24px))';
  shifting(() => host.prepend(el));
  requestAnimationFrame(() => {
    el.style.transition = 'transform var(--t-snappy, 568ms) var(--e-snappy, ease),'
      + ' opacity var(--t-press, 120ms) var(--e-smooth, ease)';
    el.style.opacity = '1';
    el.style.transform = '';
  });

  // ---- leaving: the same edge it came from
  let gone = false;
  const leave = (velocity = 0) => {
    if (gone) return; gone = true;
    clearTimeout(timer);
    el.dataset.leaving = '1';
    el.style.pointerEvents = 'none';
    // Faster if thrown. The exit is a fixed 260ms release; a hard flick shortens
    // it, so the banner keeps the speed the finger gave it (§5, velocity handoff).
    const ms = velocity > 1500 ? 160 : velocity > 700 ? 200 : 260;
    el.style.transition = `transform ${ms}ms var(--e-press, ease), opacity ${ms}ms var(--e-press, ease)`;
    el.style.opacity = '0';
    if (!still) el.style.transform = 'translateX(calc(100% + 24px))';
    const done = () => shifting(() => el.remove());
    el.addEventListener('transitionend', done, { once: true });
    setTimeout(done, ms + 80);                 // transitionend never fires under reduced motion
  };

  // ---- it waits, unless you are looking at it
  let timer = 0, left = SHOW_MS, since = performance.now();
  const hold = () => { clearTimeout(timer); left = Math.max(800, left - (performance.now() - since)); };
  const go = () => { since = performance.now(); clearTimeout(timer); timer = setTimeout(() => leave(), left); };
  go();
  el.addEventListener('pointerenter', hold);
  el.addEventListener('pointerleave', () => { if (sx == null) go(); });
  el.addEventListener('focus', hold);
  el.addEventListener('blur', go);

  // ---- opening it
  const open = () => {
    if (gone) return;
    opts.onOpen?.(n);
    leave();
    if (dest) location.href = `./${dest}.html`;
  };
  el.querySelector('.x').addEventListener('click', (e) => { e.stopPropagation(); leave(); });
  el.querySelector('.x').addEventListener('pointerdown', (e) => e.stopPropagation());
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
    if (e.key === 'Escape') leave();
  });

  // ---- swiping it away: 1:1 with the pointer, resisting the wrong way
  let sx = null, dx = 0, vx = 0, lastX = 0, lastT = 0, dragging = false, moved = false;
  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    sx = e.clientX; lastX = sx; lastT = e.timeStamp; dx = 0; vx = 0; dragging = false;
    el.setPointerCapture(e.pointerId);
    hold();
  });
  el.addEventListener('pointermove', (e) => {
    if (sx == null) return;
    dx = e.clientX - sx;
    const dt = e.timeStamp - lastT;
    if (dt > 0) vx = (e.clientX - lastX) / dt * 1000;
    lastX = e.clientX; lastT = e.timeStamp;
    if (!dragging && Math.abs(dx) < 8) return;       // hysteresis before committing
    dragging = true; moved = true;
    const w = el.offsetWidth;
    const x = dx > 0 ? dx : rubber(dx, w);
    el.style.transition = 'none';
    el.style.transform = `translateX(${x}px)`;
    el.style.opacity = String(Math.max(.25, 1 - Math.max(0, dx) / (w * 1.1)));
  });
  const release = () => {
    if (sx == null) return;
    sx = null;
    if (!dragging) { go(); return; }
    const w = el.offsetWidth;
    // Decide by where it is GOING, not where it is: a short fast flick dismisses.
    const rests = dx + project(vx);
    if (rests > w * 0.45 || vx > 600) { leave(Math.abs(vx)); return; }
    el.style.transition = 'transform var(--t-snappy, 568ms) var(--e-snappy, ease),'
      + ' opacity var(--t-press, 120ms) var(--e-smooth, ease)';
    el.style.transform = '';
    el.style.opacity = '1';
    go();
  };
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);
  el.addEventListener('click', () => { if (moved) { moved = false; return; } open(); });

  return { el, leave };
}

/* ── the poll, and the badge ────────────────────────────────────────────── */

/**
 * Start watching for this person's notices. Called once per page by deskNav.
 *
 * @param root  the page root (holds the bell badge)
 * @param user  the signed-in user
 */
export function watchNotices(root, user) {
  const uid = user && (user._id || user.id);
  const badge = root.querySelector('.appbar [data-readout], [data-chrome="mail"] [data-readout]');
  if (!uid) { badge?.remove(); return; }
  const role = user.role;
  const page = (document.body.dataset.page || '').trim();
  const quiet = QUIET[page] || (() => false);

  const paintBadge = (n) => {
    if (!badge) return;
    // Drawn as "2" and hidden before the first paint; until the fetch answers
    // nobody knows whether anything is waiting. Hidden rather than removed, so
    // the count can come back when something arrives.
    badge.removeAttribute('data-specimen');
    if (n > 0) { badge.textContent = n > 9 ? '9+' : String(n); badge.style.display = ''; }
    else badge.style.display = 'none';
  };

  const shown = new Set();
  let inFlight = false;
  const poll = async () => {
    if (inFlight || document.visibilityState === 'hidden') return;
    inFlight = true;
    try {
      const list = ((await api.notifications.mine(uid)) || []).filter(n => noticeForRole(n, role));
      paintBadge(list.filter(n => !n.read && !n.isRead).length);

      const newest = list.reduce((m, n) => {
        const t = Date.parse(n.timestamp || n.createdAt || 0) || 0;
        return t > m ? t : m;
      }, 0);
      const seen = Number(store()?.getItem(seenKey(uid)) || 0);
      if (!seen) {
        // First time on this device: everything here is backlog, and the bell
        // already says so. Only what arrives from now on is announced.
        if (newest) store()?.setItem(seenKey(uid), String(newest));
        return;
      }
      const fresh = list
        .filter(n => (Date.parse(n.timestamp || n.createdAt || 0) || 0) > seen)
        .filter(n => !n.read && !n.isRead && !shown.has(String(n._id)))
        .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));   // oldest first, so the newest ends on top
      if (newest > seen) store()?.setItem(seenKey(uid), String(newest));
      if (!fresh.length) return;
      fresh.forEach(n => shown.add(String(n._id)));

      for (const fn of listeners) { try { fn(fresh); } catch (e) { console.error(e); } }
      const loud = fresh.filter(n => !quiet(n));
      if (!loud.length) return;

      /* Three at most on screen. More than that at once -- the first sign-in of
       * the day after a busy night -- becomes two banners and one saying how
       * many more, rather than a column that covers the page. */
      const room = Math.max(0, MAX_ON_SCREEN - document.querySelectorAll('.pv-toast:not([data-leaving])').length);
      const show = loud.length > room ? loud.slice(-(Math.max(0, room - 1))) : loud;
      const rest = loud.length - show.length;
      for (const n of show) {
        const age = Date.now() - (Date.parse(n.timestamp) || 0);
        banner(n, {
          role,
          // "now" for what just happened; the clock for what happened while
          // this tab was closed and is only being announced on return.
          when: age < 90000 ? 'now' : whenShort(n.timestamp).replace(/^Today, /, ''),
          onOpen: (x) => {
            api.notifications.read(x._id).catch(() => {});
            const cur = Number(badge?.textContent) || 0;
            if (cur) paintBadge(cur - 1);
          },
        });
      }
      if (rest > 0 && room > 0) {
        banner({ message: `${rest} more ${rest === 1 ? 'thing has' : 'things have'} happened. The bell has them all.` },
               { kind: 'bell', when: 'now' });
      }
    } catch (e) {
      // A failed poll is not an event. The next one is 12 seconds away.
      if (e?.status === 401) return;             // api.js has already sent them to sign in
    } finally { inFlight = false; }
  };

  poll();
  const every = setInterval(poll, POLL_MS);
  addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') poll(); });
  addEventListener('pagehide', () => clearInterval(every));
  return poll;
}
