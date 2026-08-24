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
//   ok          -> leaf-check
//   not_a_leaf  -> the retake guidance on this same page
//   uncertain   -> the same
//
// The refusal is the system working, so it is never styled as an error.
import { requireUser, load, state } from './_guard.js';
import { bind } from '../bind.js';
import { asField, acts, press } from '../wire.js';
import { multiStep } from '../steps.js';
import { plainText } from '../bind.js';

// A refusal shows the retake guidance that used to live on a page of its own.
function revealRetake(on) {
  const block = document.querySelector('[data-retake]');
  if (block) block.style.display = on ? '' : 'none';
}

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
  // One page, two verdicts. It used to location.replace() between two URLs
  // for the same screen; a refusal is a verdict, not a destination.
  const wantRefusal = false;

  load(ctx.root, async () => {
    const saved = JSON.parse(sessionStorage.getItem(KEY) || 'null');

    // Each page shows one side of the split, so land on the right one.
    if (saved) {
      const isRefusal = saved.status !== 'ok';
      if (isRefusal !== wantRefusal) {
        revealRetake(isRefusal);
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
      // The Multi Step Loader from design/Components.dc.html. One line of text
      // with a percentage on the end was the indeterminate-spinner argument in
      // another costume: it never said WHICH of the four things was happening,
      // and 173 MB on a metered prepaid pack is a financial event, not a
      // loading state.
      const steps = multiStep(holder, [
        { label: 'Asked you first', note: '173 MB, once. It never downloads on its own.' },
        { label: 'Getting the checker', note: '' },
        { label: 'Reading your photo', note: '' },
        { label: 'Ready to use offline', note: 'It works with no signal from here on.' },
      ]);
      steps.at(1, 0, 'Starting');
      try {
        const clf = await classifier((msg, pct) => {
          steps.at(1, pct, pct != null ? `${msg} — ${Math.round(pct)}% of 173 MB` : msg);
        });
        steps.at(2, null, 'On this phone. Nothing is uploaded.');
        const verdict = await clf.predict(file);
        verdict.file = file.name;
        sessionStorage.setItem(KEY, JSON.stringify(verdict));
        localStorage.setItem('pv.model', '1');
        steps.done();
        // render() was only ever called on LOAD, from saved state. A fresh
        // check stored its verdict and never drew it, so a farmer waited out a
        // 173 MB download, picked a leaf, and the screen went on saying "No
        // diagnosis yet". The answer existed and was never shown.
        render(verdict);
        // The one moment in this product that has waited minutes for an answer.
        // 830ms on the bouncy spring, once, on the verdict only.
        const card = ctx.root.querySelector('[data-bind="verdict.name"]')?.closest('div[style*="background"]');
        if (card) {
          card.style.transform = 'scale(.97)';
          card.style.opacity = '0';
          card.style.transition = 'opacity var(--t-press, 120ms) var(--e-smooth, ease),'
            + ' transform var(--t-bouncy, 830ms) var(--e-bouncy, ease)';
          requestAnimationFrame(() => { card.style.opacity = '1'; card.style.transform = 'none'; });
        }
        const isRefusal = verdict.status !== 'ok';
        revealRetake(isRefusal);
      } catch (err) {
        state(holder, 'failed', 'The checker did not run',
          `${err.message} Nothing has been sent anywhere — the photo never left your phone.`);
      }
    });
  });

  function render(v) {
    const pct = v.confidence != null ? `${Math.round(v.confidence * 100)}% sure` : '';
    const ok = v.status === 'ok';

    // A REFUSAL IS A VERDICT AND HAS TO SAY SO.
    //
    // refusal.headline only exists inside the retake block, which is hidden
    // until a refusal earns it -- so a refused photo rendered the verdict card
    // as "Most likely / — / —" and then went on to show the full Mancozeb spray
    // guidance underneath. Spray advice for a disease nobody named is the most
    // dangerous thing this screen could do. The refusal states itself in the
    // verdict card, and the treatment sections go.
    const headline = v.status === 'not_a_leaf'
      ? 'That does not look like a potato leaf'
      : 'Not sure enough to name anything';
    bind(ctx.root, {
      shot: { file: v.file || 'your photo', where: `Checked on your phone · ${v.ms} ms` },
      verdict: {
        name: ok ? v.disease : headline,
        note: ok
          ? `${WORDS[v.disease] || ''} ${pct}${v.runner_up ? `, and it could be ${v.runner_up}` : ''}.`.trim()
          : plainText(v.message) || 'Take it again with the whole leaf in frame and a little space around it.',
      },
      refusal: { headline },
    });
    for (const sec of ctx.root.querySelectorAll('[data-treatment]')) {
      sec.style.display = ok ? '' : 'none';
    }
    // "Most likely" is a claim about a diagnosis. There is not one.
    const kicker = ctx.root.querySelector('[data-bind="verdict.name"]')?.previousElementSibling;
    if (kicker && !ok) kicker.textContent = 'The checker refused';
    const again = document.createElement('div');
    ctx.root.append(again);
    state(again, 'empty', 'Check another leaf', 'This clears the result and opens the camera.');
    const box = again.firstElementChild;
    box.setAttribute('data-act', '');
    box.addEventListener('click', () => {
      sessionStorage.removeItem(KEY);
      location.replace('./leaf-check.html');
    });
  }
}
