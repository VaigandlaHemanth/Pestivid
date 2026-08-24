// Filming. The camera is the product; everything else here is restraint.
//
// Nothing leaves the handset on this screen. The clip is handed to the next one
// in memory, which is what lets that screen offer "throw this one away" as a
// real choice rather than a request to delete something already sent.
import { requireUser, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { acts, press } from '../wire.js';
import { putClip } from '../clip.js';

/**
 * The camera is unavailable. Say why, and give the farmer the way through that
 * the old copy promised and did not provide: pick a file the camera app made.
 * It runs through the same upload and the same server-side hash, so nothing
 * about the evidence is weaker -- only the filming happened elsewhere.
 */
function refuse(headline, detail) {
  const pick = document.createElement('input');
  pick.type = 'file';
  pick.accept = 'video/*';
  pick.style.cssText = 'position: absolute; width: 1px; height: 1px; opacity: 0;';
  pick.addEventListener('change', async () => {
    const file = pick.files?.[0];
    if (!file) return;
    try {
      await putClip({ file, size: file.size, duration: 0, fromCameraApp: true });
      location.href = './sent.html';
    } catch {
      state(ctx.root, 'failed', 'That file could not be held',
        'This browser will not keep a video between screens. Try a different browser.');
    }
  });
  ctx.root.append(pick);
  // The panel is drawn 800px tall because it is a viewfinder. With no camera
  // there is nothing to view, and the leftover was a thousand pixels of black
  // under a red box. The page becomes as tall as what it actually has to say.
  ctx.root.style.minHeight = 'auto';
  return state(ctx.root, 'failed', headline, detail,
    { label: 'Pick a file instead', act: () => pick.click() });
}

const ctx = requireUser('record', ['farmer']);
if (ctx) load(ctx.root, async () => {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    return refuse('This phone will not record here',
      'The browser does not give a web page the camera. Film with your camera app, then pick the '
      + 'file here — it goes through exactly the same check, and the date is fixed when it reaches us.');
  }

  const slot = ctx.root.querySelector('div[style*="#37322d"], div[style*="#0e0d0b"]');
  let chunks = [];
  let started = 0;
  let timer = null;
  let stream;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
      audio: true,
    });
  } catch {
    return refuse('We cannot see through the camera',
      'You said no, or another app is holding it. Nothing is being recorded. Allow the camera in '
      + 'your browser settings and open this screen again, or pick a file you already filmed.');
  }

  const view = document.createElement('video');
  view.autoplay = true;
  view.muted = true;
  view.playsInline = true;
  view.srcObject = stream;
  view.style.cssText = 'position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;';
  slot?.prepend(view);

  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9' : 'video/webm';
  const rec = new MediaRecorder(stream, { mimeType: mime });
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };

  // [data-progress], not a colour match. The record dot is #a71930 too and sits
  // earlier in the document, so the colour selector returned the dot and the
  // elapsed bar never moved once.
  const bar = ctx.root.querySelector('[data-progress]');

  const tick = () => {
    const secs = (Date.now() - started) / 1000;
    const bytes = chunks.reduce((a, c) => a + c.size, 0);
    bind(ctx.root, { clip: { size: `${(bytes / 1e6).toFixed(1)} MB` } });
    // scaleX, never width: a bar animated on width relayouts its parent every
    // frame, and this runs while the camera already owns the CPU.
    if (bar) bar.style.transform = `scaleX(${Math.min(1, secs / 45)})`;
  };

  let thrown = false;
  rec.onstop = () => {
    clearInterval(timer);
    stream.getTracks().forEach(t => t.stop());
    if (thrown) { location.href = './home.html'; return; }
    const blob = new Blob(chunks, { type: rec.mimeType });
    // Not window.__pvClip: a navigation destroys window, so every clip filmed
    // arrived at a send screen that said there was no clip. See app/clip.js.
    putClip({
      file: new File([blob], 'clip.webm', { type: blob.type }),
      size: blob.size,
      duration: (Date.now() - started) / 1000,
    }).then(() => { location.href = './sent.html'; })
      .catch(() => state(ctx.root, 'failed', 'The clip could not be held',
        'It was filmed but this browser will not keep it between screens, so it has not been sent. '
        + 'Nothing left your phone.'));
  };

  started = Date.now();
  rec.start(1000);
  tick();                       // the board draws the bar part-filled; zero it
  timer = setInterval(tick, 500);

  const byLabel = (t) => [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === t);

  const control = (t) => byLabel(t)?.parentElement || null;

  acts(control('Stop'), 'Stop recording', () => {
    if (rec.state !== 'inactive') rec.stop();
  });

  // Throwing the take away is irreversible and sits 56px from Stop, on a phone
  // held one-handed while walking. It asks once.
  let armed = false;
  const binLabel = byLabel('Throw away');
  acts(control('Throw away'), 'Throw this recording away', () => {
    if (!armed) {
      armed = true;
      binLabel.textContent = 'Tap again to lose it';
      setTimeout(() => {
        if (!armed) return;
        armed = false;
        if (binLabel.isConnected) binLabel.textContent = 'Throw away';
      }, 4000);
      return;
    }
    thrown = true;
    chunks = [];
    if (rec.state !== 'inactive') rec.stop(); else location.href = './home.html';
  });

  press(ctx.root);
});
