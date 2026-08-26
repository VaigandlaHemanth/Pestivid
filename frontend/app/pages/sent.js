// The screen straight after filming: name the plot, then send.
//
// Four things were wrong here, and the first is the worst thing found anywhere
// in this app.
//
// 1. The screen exists to collect one answer -- "Which plot was that?" -- and
//    the answer was thrown away. The four options were never wired, and the
//    upload went out with `crop: 'crop', location: 'unknown'` every time. So
//    the provenance record this entire product is built on had no field
//    attached to it, while the copy directly under the question promised
//    "we ask now because in a week nobody remembers which field a video was".
//
// 2. The send button broke visually the instant it was tapped. The control was
//    found with `.find(d => d.textContent.trim() === 'Keep it and send')`,
//    which returns the OUTER blue box, not the label inside it -- so
//    `send.textContent = 'Sending…'` deleted the styled label and 18px/700
//    white became 16px/400 near-black on #016abe: 3.14:1, failing AA, at the
//    one moment the farmer is watching the button.
//
// 3. markDone(0) and markDone(1) ran unconditionally, so "Fingerprinted the
//    moment it arrives" was ticked whether or not the server had hashed it.
//
// 4. The first step's tick was --proved green. See the board comment.
import { requireUser, api, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { sendVideo } from '../api.js';
import { appChrome } from '../chrome.js';
import { acts, press } from '../wire.js';
import { takeClip, dropClip } from '../clip.js';

const ctx = requireUser('sent', ['farmer']);

const TICK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3"><path d="M5 12.5l4.5 4.5L19 7"></path></svg>';

if (ctx) {
  appChrome(ctx.root, { back: 'record', user: ctx.user });
  press(ctx.root);

  load(ctx.root, async () => {
    const root = ctx.root;
    // Handed over by the record screen through IndexedDB. It used to be read
    // off `window`, which the navigation from record.html had already thrown
    // away -- so this screen always said there was no clip.
    const clip = await takeClip();
    bind(root, { clip: { line: clip
      ? `${Math.round(clip.duration || 0)} seconds · ${(clip.size / 1e6).toFixed(1)} MB · on your phone`
      : 'Nothing filmed yet' } });

    if (!clip) {
      // The page heading is "That is saved". It is not, and a title claiming a
      // success above a body reporting nothing is the screen contradicting
      // itself in the two places a reader looks first.
      const title = root.querySelector('[data-title]');
      if (title) title.textContent = 'Nothing to send';
      return state(root, 'empty', 'There is no clip to send',
        'Film your field first. Nothing has been lost, there was simply nothing here.',
        { label: 'Film the field', go: 'record' });
    }

    // ---- which plot -------------------------------------------------
    const opts = [...root.querySelectorAll('.pk, .pkOn')];
    const escape = opts.pop();                 // "A plot not in this list"
    const plots = (await api.projects.mine(ctx.user._id || ctx.user.id).catch(() => [])) || [];
    let chosen = null;

    if (!plots.length) {
      // A question with no real answers is a call to action pointing at
      // nothing. The block goes rather than offering invented plot names --
      // and a farmer's first video legitimately has no plot yet, because a
      // funding request needs a video cid before it can exist.
      opts.forEach(o => o.remove());
      escape?.parentElement?.parentElement?.remove();
    } else {
      opts.forEach((el, i) => {
        const p = plots[i];
        if (!p) { el.remove(); return; }
        el.textContent = p.title;
        el.className = 'pk';                   // nothing is pre-picked on a live page
        acts(el, p.title, () => {
          opts.forEach(o => { if (o.isConnected) o.className = 'pk'; });
          el.className = 'pkOn';
          chosen = p;
          warn?.replaceChildren();
        });
      });
      if (escape) {
        escape.className = 'pk';
        escape.style.color = '#01579b';
        escape.style.fontWeight = '600';
        acts(escape, 'A plot not in this list', () => { location.href = './plots.html'; });
      }
    }

    // Somewhere local to say "pick one first". state() ends in
    // replaceChildren(), so handing it the page root wipes the screen the
    // farmer still has to use.
    let warn = null;
    const picker = opts.find(o => o.isConnected)?.parentElement;
    if (picker) {
      warn = document.createElement('div');
      picker.insertAdjacentElement('afterend', warn);
    }

    // ---- the timeline -----------------------------------------------
    const steps = [...root.querySelectorAll('.dotW')];
    const markDone = (i, proved) => {
      const d = steps[i];
      if (!d) return;
      d.className = proved ? 'dotD' : 'dotOk';
      d.innerHTML = TICK;
      // The farmer is watching this step complete, so it completes visibly.
      const mark = d.firstElementChild;
      if (mark) {
        mark.style.opacity = '0';
        mark.style.transform = 'scale(.5)';
        mark.style.transition = 'opacity var(--t-press, 120ms) var(--e-smooth, ease),'
          + ' transform var(--t-bouncy, 830ms) var(--e-bouncy, ease)';
        requestAnimationFrame(() => { mark.style.opacity = '1'; mark.style.transform = 'none'; });
      }
    };

    // ---- send -------------------------------------------------------
    // The LABEL, not the box around it.
    const send = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Keep it and send');
    const button = send?.parentElement;

    // "Throw this one away" was drawn in alarm red and never handled: the one
    // control on this screen whose whole purpose is that nothing has left the
    // phone yet. click-everything skips this page (it spends storage), so
    // nothing caught it.
    const bin = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Throw this one away');
    if (bin) {
      acts(bin.parentElement || bin, 'Throw this one away', async () => {
        await dropClip();
        location.href = './home.html';
      });
    }

    acts(button, 'Keep it and send', async () => {
      if (plots.length && !chosen) {
        if (warn) {
          state(warn, 'waiting', 'Pick the plot first',
            'Tap the field this video is of. In a week nobody remembers which one it was.');
          warn.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
        return;
      }
      const label = send.textContent;
      send.textContent = 'Sending…';
      try {
        const saved = await sendVideo(
          clip.file,
          chosen
            // `location` is required by POST /videos and a funding request has
            // no location field, so the title stands in -- it is required on
            // the model, so it is never empty.
            ? { crop: chosen.crop || 'crop', location: chosen.location || chosen.title, purpose: 'agristream' }
            : { crop: clip.crop || 'crop', location: clip.location || 'unknown', purpose: 'agristream' },
          (p) => { send.textContent = `Sending… ${Math.round(p * 100)}%`; },
        );
        const hashed = saved.hashComputedBy === 'server';
        markDone(0);                 // sent to us: we watched it arrive
        // Only tick the fingerprint step if the server actually did it. Ticking
        // it either way is the app claiming something it has not checked.
        if (hashed) markDone(1);
        send.textContent = 'Sent';
        bind(root, { clip: { line: hashed
          ? 'Stored, and we hashed it ourselves'
          : 'Stored, but the hash is unverified' } });
      } catch (err) {
        send.textContent = label;
        state(warn || root, err.offline ? 'waiting' : 'failed',
          err.offline ? 'No signal yet' : 'That did not send',
          err.offline
            ? 'It is still on your phone and will go on its own when you have a bar.'
            : err.message);
      }
    });

    press(root);
  });
}
