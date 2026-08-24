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

  // The board marks it. This used to look for a background colour that is not
  // on the Record board at all -- #37322d and #0e0d0b, while the viewfinder is
  // #2a2622 -- so the lookup returned null, no <video> was ever inserted, and
  // the camera ran behind a blank rectangle.
  const slot = ctx.root.querySelector('[data-viewfinder]');
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

  // mp4 first. A streaming WebM's container header goes out before the track
  // layout is settled, so a content sniffer reads it as audio/webm -- which is
  // exactly what storage said when it rejected every real recording. An mp4
  // sniffs as video, and it is also the only thing iOS Safari will record.
  const WANT = [
    'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
    'video/mp4',
    'video/webm;codecs=vp9,opus',
    'video/webm',
  ];
  const mime = WANT.find(t => MediaRecorder.isTypeSupported(t)) || '';
  const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };

  // [data-progress], not a colour match. The record dot is #a71930 too and sits
  // earlier in the document, so the colour selector returned the dot and the
  // elapsed bar never moved once.
  const bar = ctx.root.querySelector('[data-progress]');

  const clock = (secs) => {
    const m = Math.floor(secs / 60);
    const s2 = Math.floor(secs % 60);
    return `${m}:${String(s2).padStart(2, '0')}`;
  };

  const tick = () => {
    const secs = (Date.now() - started) / 1000;
    const bytes = chunks.reduce((a, c) => a + c.size, 0);
    // The elapsed clock had no binding at all, so it read the artboard's 0:22
    // for the whole recording however long a farmer walked.
    bind(ctx.root, { clip: { size: `${(bytes / 1e6).toFixed(1)} MB`, elapsed: clock(secs) } });
    // scaleX, never width: a bar animated on width relayouts its parent every
    // frame, and this runs while the camera already owns the CPU.
    if (bar) bar.style.transform = `scaleX(${Math.min(1, secs / 45)})`;
  };

  let thrown = false;
  rec.onstop = () => {
    clearInterval(timer);
    stream.getTracks().forEach(t => t.stop());
    if (thrown) { location.href = './home.html'; return; }
    // The base type, without the codecs parameter: what goes on the wire has to
    // match what the upload ticket allows.
    const base = (rec.mimeType || 'video/webm').split(';')[0];
    const blob = new Blob(chunks, { type: base });
    // Not window.__pvClip: a navigation destroys window, so every clip filmed
    // arrived at a send screen that said there was no clip. See app/clip.js.
    putClip({
      file: new File([blob], base === 'video/mp4' ? 'clip.mp4' : 'clip.webm', { type: base }),
      size: blob.size,
      duration: (Date.now() - started) / 1000,
    }).then(() => { location.href = './sent.html'; })
      .catch(() => state(ctx.root, 'failed', 'The clip could not be held',
        'It was filmed but this browser will not keep it between screens, so it has not been sent. '
        + 'Nothing left your phone.'));
  };

  const byLabel = (t) => [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === t);

  const control = (t) => byLabel(t)?.parentElement || null;

  // ---- the shutter -----------------------------------------------------
  // Recording used to begin the instant the page loaded. A farmer arrived on a
  // screen that was already filming them walking up to the field, and the only
  // way out was to stop or throw it away. The camera previews; the farmer
  // decides when it starts.
  const shutter = ctx.root.querySelector('[data-shutter]');
  const shutterIcon = ctx.root.querySelector('[data-shutter-icon]');
  const shutterLabel = ctx.root.querySelector('[data-shutter-label]');
  const dot = ctx.root.querySelector('[data-reddot]');
  const paintIdle = () => {
    if (shutterIcon) {
      shutterIcon.style.borderRadius = '50%';
      shutterIcon.style.width = '36px';
      shutterIcon.style.height = '36px';
    }
    if (shutterLabel) shutterLabel.textContent = 'Record';
    if (bar) bar.style.transform = 'scaleX(0)';
    // The red dot means "recording". Idle, it is not.
    if (dot) dot.style.background = 'rgba(255,255,255,.28)';
    bind(ctx.root, { clip: { elapsed: '0:00', size: '0.0 MB' } });
  };
  const paintRolling = () => {
    if (shutterIcon) {
      shutterIcon.style.borderRadius = '4px';
      shutterIcon.style.width = '34px';
      shutterIcon.style.height = '34px';
    }
    if (shutterLabel) shutterLabel.textContent = 'Stop';
    if (dot) dot.style.background = '#a71930';
  };
  paintIdle();

  const begin = () => {
    if (rec.state !== 'inactive') return;
    chunks = [];
    started = Date.now();
    rec.start(1000);
    paintRolling();
    tick();
    timer = setInterval(tick, 500);
  };

  acts(shutter, 'Record', () => {
    if (rec.state === 'inactive') begin();
    else rec.stop();
  });

  // The microphone. This slot held "Read to me", the last of the voice work,
  // wired to nothing. Whether your own voice goes into the file is a real
  // choice on a screen you use while talking to somebody.
  const micBtn = ctx.root.querySelector('[data-mic]');
  const micLabel = ctx.root.querySelector('[data-miclabel]');
  let sound = true;
  const paintMic = () => {
    if (micLabel) micLabel.textContent = sound ? 'Sound on' : 'Sound off';
    if (micBtn) micBtn.style.background = sound ? 'rgba(255,255,255,.13)' : 'rgba(167,25,48,.28)';
    micBtn?.setAttribute('aria-pressed', String(!sound));
    stream.getAudioTracks().forEach(t => { t.enabled = sound; });
  };
  if (micBtn) { acts(micBtn, 'Sound', () => { sound = !sound; paintMic(); }); paintMic(); }

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
