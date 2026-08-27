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
import { bind, writing, working } from '../bind.js';
import { sendVideo } from '../api.js';
import { appChrome } from '../chrome.js';
import { acts, press } from '../wire.js';
import { takeClip, dropClip } from '../clip.js';

const ctx = requireUser('sent', ['farmer']);

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
      // The whole left column goes, by its mark. Climbing two parents from the
      // escape row was the same question asked fragilely: it happened to be the
      // block, until the block became a grid cell.
      (ctx.root.querySelector('[data-picker]')
        || escape?.parentElement?.parentElement)?.remove();
      opts.forEach(o => { if (o.isConnected) o.remove(); });
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

    /* ---- the timeline -------------------------------------------------
     * Four things happen to this file in order, with real time between them,
     * and the screen used to swap three class names on one frame and call that
     * reporting. Now it plays: the line between two steps draws downward, the
     * step it reaches ticks, the next line starts.
     *
     * Every duration comes off the tokens, so this and every other movement in
     * the product are timed by the same two numbers.
     */
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const token = (name, fallback) => {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      const n = parseFloat(v);
      return Number.isFinite(n) ? (/ms$/.test(v) ? n : n * 1000) : fallback;
    };
    const T_SMOOTH = token('--t-smooth', 746);
    const T_PRESS = token('--t-press', 120);
    // Under reduced motion the states still change; they just arrive without
    // being waited for.
    const beat = (d) => new Promise(r => setTimeout(r, still ? 0 : d));

    const steps = [...root.querySelectorAll('[data-pipeline] .st')];
    const lines = [...root.querySelectorAll('[data-pipeline] .line')];

    /** Draw the line BELOW step i, downward, and wait for it to arrive. */
    const drawLine = async (i) => {
      const line = lines[i];
      if (!line || line.hasAttribute('data-on')) return;   // already drawn
      line.setAttribute('data-on', '');
      await beat(T_SMOOTH);
    };
    /** Tick step i. `proved` is the one step green is allowed on. */
    const tick = async (i, proved) => {
      const dot = steps[i]?.querySelector('.dotW');
      if (dot) dot.setAttribute(proved ? 'data-proved' : 'data-done', '');
      // The next line starts as the tick lands rather than after its spring has
      // finished settling -- otherwise the sequence reads as four pauses.
      await beat(T_PRESS);
    };

    /* ---- send ---------------------------------------------------------
     * Both controls are found by their mark, not by the words printed in them.
     * The previous version searched the page for the string "Keep it and send"
     * -- so the first thing the handler did, setting that text to "Sending...",
     * destroyed the thing that found it, and any wording change would have
     * silently unwired the only button on the screen.
     */
    const button = root.querySelector('[data-keep]');
    const send = button?.firstElementChild;        // the LABEL, not the box

    // "Throw this one away" was drawn in alarm red and never handled: the one
    // control on this screen whose whole purpose is that nothing has left the
    // phone yet. click-everything skips this page (it spends storage), so
    // nothing caught it.
    const bin = root.querySelector('[data-bin]');
    if (bin) {
      acts(bin, 'Throw this one away', async () => {
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
      button.setAttribute('aria-disabled', 'true');
      // The file is leaving the phone, so the line from "saved on your phone" to
      // "sent to us" starts drawing now -- not awaited, because it runs while
      // the upload does. And the pill goes INSIDE the timeline card, next to the
      // step it is about: this was reported as "I did not even see a loader",
      // and it was correct -- it was a band at the top of a page the reader had
      // scrolled past.
      drawLine(0);
      const stop = working(root, 'Sending it to us now.');
      const said = root.querySelector('[data-loader] .said');
      try {
        const saved = await sendVideo(
          clip.file,
          chosen
            // `location` is required by POST /videos and a funding request has
            // no location field, so the title stands in -- it is required on
            // the model, so it is never empty.
            ? { crop: chosen.crop || 'crop', location: chosen.location || chosen.title, purpose: 'agristream' }
            : { crop: clip.crop || 'crop', location: clip.location || 'unknown', purpose: 'agristream' },
          (p) => {
            const pct = Math.round(p * 100);
            send.textContent = `Sending… ${pct}%`;
            if (said) said.textContent = `Sending it to us now. ${pct}%`;
          },
        );
        const hashed = saved.hashComputedBy === 'server';
        stop();
        send.textContent = 'Sent';
        button.style.background = '#1d1a17';
        /* And the screen stops offering what it can no longer do.
         *
         * "Throw this one away" stayed live after a successful send, next to a
         * band reading "you cannot delete it later" -- the page contradicting
         * itself in two adjacent boxes, with the red button being the lie. The
         * warning was about a decision that has now been made, so both go, and
         * the picker locks: choosing a different plot after the upload has
         * carried the old one changes nothing and looks like it changes
         * everything. */
        root.querySelector('[data-warn-delete]')?.remove();
        bin?.remove();
        const phoneCard = root.querySelector('[data-rail-phone]');
        if (phoneCard) {
          phoneCard.querySelector('.railh').textContent = 'It is with us now';
          phoneCard.querySelector('.railp').textContent = 'The file is on our server with its '
            + 'fingerprint taken. It is off your phone, and there is nothing left for you to do '
            + 'but wait for the date.';
        }
        button.style.width = '100%';
        for (const o of root.querySelectorAll('[data-picker] .pk, [data-picker] .pkOn')) {
          o.setAttribute('aria-disabled', 'true');
          o.removeAttribute('data-act');
        }
        bind(root, { clip: { line: hashed
          ? 'Stored, and we hashed it ourselves'
          : 'Stored, but the hash is unverified' } });

        // Now it plays, in the order it happened.
        await drawLine(0);                      // already drawing; returns at once
        await tick(1);                          // sent to us: we watched it arrive
        // Only tick the fingerprint step if the server actually did it. Ticking
        // it either way is the app claiming something it has not checked -- and
        // the line cannot travel PAST a step that has not happened, so an
        // unconfirmed hash stops the sequence here and says so.
        if (!hashed) {
          working(root, 'It is with us and it is not going anywhere. We have not been '
            + 'able to confirm the fingerprint yet, and we will tell you when we have.');
          return;
        }
        await drawLine(1);
        await tick(2);
        await drawLine(2);
        // The fourth step has not happened and will not for hours: the day's
        // hashes go into one Bitcoin block. This is the one place in the product
        // where the wait IS the subject, so the pencil says so rather than a
        // sentence sitting still. It stays until the farmer leaves the screen.
        writing(root, 'The date is being written into a Bitcoin block with the rest '
          + 'of today’s, usually by tomorrow. You can close this.');
      } catch (err) {
        stop();
        send.textContent = label;
        button.removeAttribute('aria-disabled');
        // It did not arrive, so the line saying it did is taken back.
        lines[0]?.removeAttribute('data-on');
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
