// The screen straight after filming: name the plot, then send.
import { requireUser, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { sendVideo } from '../api.js';

const ctx = requireUser('sent', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const clip = window.__pvClip || null;      // handed over by the record screen
  bind(ctx.root, { clip: { line: clip
    ? `${Math.round(clip.duration || 0)} seconds · ${(clip.size / 1e6).toFixed(1)} MB · on your phone`
    : 'Nothing filmed yet' } });

  if (!clip) {
    return state(ctx.root, 'empty', 'There is no clip to send',
      'Film your field first. Nothing has been lost — there was simply nothing here.');
  }

  const steps = [...ctx.root.querySelectorAll('.dotW')];
  const markDone = (i) => {
    const d = steps[i];
    if (!d) return;
    d.className = 'dotD';
    d.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3"><path d="M5 12.5l4.5 4.5L19 7"></path></svg>';
  };

  const send = [...ctx.root.querySelectorAll('div')]
    .find(d => d.textContent.trim() === 'Keep it and send');
  send?.setAttribute('data-act', '');
  send?.addEventListener('click', async () => {
    const label = send.textContent;
    send.textContent = 'Sending…';
    try {
      const saved = await sendVideo(
        clip.file,
        { crop: clip.crop || 'crop', location: clip.location || 'unknown', purpose: 'agristream' },
        (p) => { send.textContent = `Sending… ${Math.round(p * 100)}%`; },
      );
      markDone(0);
      markDone(1);
      send.textContent = 'Sent';
      bind(ctx.root, { clip: { line: saved.hashComputedBy === 'server'
        ? 'Stored, and we hashed it ourselves'
        : 'Stored, but the hash is unverified' } });
    } catch (err) {
      send.textContent = label;
      state(ctx.root, err.offline ? 'waiting' : 'failed',
        err.offline ? 'No signal yet' : 'That did not send',
        err.offline
          ? 'It is still on your phone and will go on its own when you have a bar.'
          : err.message);
    }
  });
});
