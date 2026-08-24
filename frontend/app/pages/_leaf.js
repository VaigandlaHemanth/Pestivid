// Leaf check, both outcomes.
//
// The classifier runs on the handset -- js/potato-browser.js, DINOv2 features
// and a calibrated linear probe -- so a photograph never leaves the phone and
// there is no per-question API cost. That also means the 173 MB backbone has to
// be downloaded once, which the screen has to be honest about on a metered
// connection.
//
// predict() returns one of three states and the design has a page for each
// side of the split:
//   ok          -> leaf-result
//   not_a_leaf  -> leaf-refusal
//   uncertain   -> leaf-refusal
//
// The refusal is the system working, so it is never styled as an error.
import { requireUser, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { asField, acts, press } from '../wire.js';

const KEY = 'pv.leaf';
const MODEL_ROOT = '/model';

const WORDS = {
  Bacteria:    'Bacterial spot or wilt. Look for dark, wet-looking patches with a yellow halo.',
  Fungi:       'A fungal infection. Rings or powder on the older, lower leaves.',
  Healthy:     'Nothing wrong with this leaf.',
  Nematode:    'Root damage showing in the leaf. Check the roots for knots.',
  Pest:        'Insect damage rather than disease. Look under the leaf for what is eating it.',
  Phytopthora: 'Late blight. This moves fast in wet weather and needs acting on today.',
  Virus:       'A virus. It cannot be sprayed away; the plant will not recover.',
};

async function classifier(onProgress) {
  if (!window.PotatoBrowser) {
    await new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = '/js/potato-browser.js';
      s.onload = res; s.onerror = () => rej(new Error('The leaf checker could not be loaded.'));
      document.head.append(s);
    });
  }
  return window.PotatoBrowser.load({
    modelDir: MODEL_ROOT,
    resizeUrl: '/js/pil-resize.js',
    onProgress,
  });
}

export function leaf(slug) {
  const ctx = requireUser(slug, ['farmer']);
  if (!ctx) return;
  const wantRefusal = slug === 'leaf-refusal';

  load(ctx.root, async () => {
    const saved = JSON.parse(sessionStorage.getItem(KEY) || 'null');

    // Each page shows one side of the split, so land on the right one.
    if (saved) {
      const isRefusal = saved.status !== 'ok';
      if (isRefusal !== wantRefusal) {
        location.replace(isRefusal ? './leaf-refusal.html' : './leaf-result.html');
        return;
      }
      render(saved);
      return;
    }

    // No verdict yet, so the drawn example must go: a farmer must not read a
    // worked diagnosis as one the checker gave them.
    bind(ctx.root, {
      shot: { file: 'No photo yet', where: 'Nothing checked yet' },
      verdict: { name: 'No diagnosis yet', note: 'Take a photo and the checker will name what it sees, or refuse to.' },
      refusal: { headline: 'Nothing checked yet' },
    });

    // No verdict yet: this page is where a photograph is taken.
    const slot = ctx.root.querySelector('div[style*="#37322d"]');
    // ---- asking about the result -------------------------------------
  // Three suggested questions, a field and a send button, all drawn and none of
  // them wired. They hand the question to the chatbot with the verdict already
  // attached, which is the point of them: the farmer should not have to explain
  // the photo again.
  const askAbout = (q) => {
    if (!q) return;
    const verdict = ctx.root.querySelector('[data-bind="verdict.name"]')?.textContent.trim();
    const full = verdict ? `${q} (about a leaf the checker called ${verdict})` : q;
    sessionStorage.setItem('pv.askText', full);   // not 'pv.ask' -- _askmoney.js owns that
    location.href = './ask.html';
  };
  for (const row of ctx.root.querySelectorAll('[data-ask]')) {
    acts(row, row.textContent.trim(), () => askAbout(row.textContent.trim()));
  }
  const askField = ctx.root.querySelector('[data-askfield]');
  const askInput = askField && asField(askField, {
    name: 'question', enterKeyHint: 'send',
    placeholder: 'Ask something else', label: 'Your question about this leaf',
  });
  const askSend = ctx.root.querySelector('[data-asksend]');
  if (askSend) {
    const paint = () => {
      const has = Boolean(askInput?.value.trim());
      askSend.style.background = has ? '#016abe' : '#c3bcb6';
      askSend.setAttribute('aria-disabled', String(!has));
    };
    askInput?.addEventListener('input', paint);
    paint();
    acts(askSend, 'Send your question', () => {
      const v = askInput?.value.trim();
      if (!v) { askInput?.focus(); return; }
      askAbout(v);
    });
    askInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') askAbout(askInput.value.trim());
    });
  }

  const picker = document.createElement('input');
    picker.type = 'file';
    // A bare file input has no name for a reader and no autocomplete for the
    // browser; both are one line each.
    picker.setAttribute('aria-label', 'Choose a photo of one leaf');
    picker.setAttribute('autocomplete', 'off');
    picker.accept = 'image/*';
    picker.capture = 'environment';
    picker.className = 'sr';
    ctx.root.append(picker);

    const holder = document.createElement('div');
    ctx.root.prepend(holder);
    state(holder, 'empty', 'Take a photo of one leaf',
      'The whole leaf in the frame with a little space around it, in even daylight. Do not zoom in — filling the frame is what makes the checker refuse.');
    holder.firstElementChild.setAttribute('data-act', '');
    holder.firstElementChild.addEventListener('click', () => picker.click());

    picker.addEventListener('change', async () => {
      const file = picker.files?.[0];
      if (!file) return;
      const preview = URL.createObjectURL(file);
      if (slot) {
        const img = document.createElement('img');
        img.src = preview;
        img.alt = 'The leaf you photographed';
        img.style.cssText = 'position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;';
        slot.prepend(img);
      }
      state(holder, 'waiting', 'Checking on your phone', 'Starting the checker…');
      try {
        const clf = await classifier((msg, pct) => {
          state(holder, 'waiting', 'Checking on your phone',
            `${msg}${pct != null ? ` — ${Math.round(pct)}%` : ''}. This happens once; after that it works with no signal.`);
        });
        const verdict = await clf.predict(file);
        verdict.file = file.name;
        sessionStorage.setItem(KEY, JSON.stringify(verdict));
        localStorage.setItem('pv.model', '1');
        const isRefusal = verdict.status !== 'ok';
        location.replace(isRefusal ? './leaf-refusal.html' : './leaf-result.html');
      } catch (err) {
        state(holder, 'failed', 'The checker did not run',
          `${err.message} Nothing has been sent anywhere — the photo never left your phone.`);
      }
    });
  });

  function render(v) {
    const pct = v.confidence != null ? `${Math.round(v.confidence * 100)}% sure` : '';
    bind(ctx.root, {
      shot: { file: v.file || 'your photo', where: `Checked on your phone · ${v.ms} ms` },
      verdict: {
        name: v.status === 'ok' ? v.disease : '—',
        note: v.status === 'ok'
          ? `${WORDS[v.disease] || ''} ${pct}${v.runner_up ? `, and it could be ${v.runner_up}` : ''}.`.trim()
          : '',
      },
      refusal: { headline: v.status === 'not_a_leaf'
        ? 'That does not look like a potato leaf'
        : 'Not sure enough to name anything' },
    });
    // The refusal already carries the correct wording from the model, which
    // says explicitly not to fill the frame. Use its words, not ours.
    if (v.status !== 'ok' && v.message) {
      const note = ctx.root.querySelector('[data-bind="refusal.headline"]')?.parentElement
        ?.querySelector('div:not([data-bind])');
      if (note) note.textContent = v.message;
    }
    const again = document.createElement('div');
    ctx.root.append(again);
    state(again, 'empty', 'Check another leaf', 'This clears the result and opens the camera.');
    const box = again.firstElementChild;
    box.setAttribute('data-act', '');
    box.addEventListener('click', () => {
      sessionStorage.removeItem(KEY);
      location.replace('./leaf-result.html');
    });
  }
}
