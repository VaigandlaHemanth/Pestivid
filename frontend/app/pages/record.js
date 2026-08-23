// Filming. The camera is the product; everything else here is restraint.
//
// Nothing leaves the handset on this screen. The clip is handed to the next one
// in memory, which is what lets that screen offer "throw this one away" as a
// real choice rather than a request to delete something already sent.
import { requireUser, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { acts, press } from '../wire.js';

const ctx = requireUser('record', ['farmer']);
if (ctx) load(ctx.root, async () => {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    return state(ctx.root, 'failed', 'This phone will not record here',
      'The browser does not give a web page the camera. Film with the camera app and send the file instead.');
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
    return state(ctx.root, 'failed', 'We cannot see through the camera',
      'You said no, or another app is holding it. Nothing is being recorded.');
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
    window.__pvClip = {
      file: new File([blob], 'clip.webm', { type: blob.type }),
      size: blob.size,
      duration: (Date.now() - started) / 1000,
    };
    location.href = './sent.html';
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
