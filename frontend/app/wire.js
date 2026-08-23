// Shared conduct for every generated page.
//
// The markup is cut from the artboards and must not be edited, so behaviour is
// attached here by attribute rather than by rewriting the screens. A page's own
// module adds the data binding; this file adds what every page needs.

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Press answers on pointerdown, not on click. Waiting for a tap to resolve is
 *  the difference between a surface that answers and one that lags. */
function pressFeedback(root) {
  if (REDUCED) return;
  const down = e => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    el.setAttribute('data-pressed', '');
  };
  const up = () => root.querySelectorAll('[data-pressed]').forEach(el => el.removeAttribute('data-pressed'));
  root.addEventListener('pointerdown', down, { passive: true });
  addEventListener('pointerup', up, { passive: true });
  addEventListener('pointercancel', up, { passive: true });
}

/** A div that behaves like a button has to answer a keyboard and announce
 *  itself. The artboards are div soup by necessity; this repairs it at runtime
 *  rather than by editing markup that has to stay pixel-identical. */
function promoteControls(root) {
  for (const el of root.querySelectorAll('[data-act]')) {
    if (!el.hasAttribute('role')) el.setAttribute('role', el.dataset.act === 'link' ? 'link' : 'button');
    if (!el.hasAttribute('tabindex')) el.tabIndex = 0;
    if (!el.getAttribute('aria-label') && !el.textContent.trim()) {
      console.warn('control with no accessible name', el);
    }
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); }
    });
  }
}

/** Navigation, declared in markup: data-go="plot" moves to that page. */
function links(root) {
  root.addEventListener('click', e => {
    const el = e.target.closest('[data-go]');
    if (!el) return;
    const to = el.dataset.go;
    location.href = /^https?:|^\//.test(to) ? to : `./${to}.html`;
  });
}

/** Fills text and attributes from a plain object: data-text="farmer.name". */
export function fill(root, data) {
  for (const el of root.querySelectorAll('[data-text]')) {
    const v = el.dataset.text.split('.').reduce((o, k) => (o == null ? o : o[k]), data);
    if (v !== undefined && v !== null) el.textContent = String(v);
  }
  for (const el of root.querySelectorAll('[data-hidden-if]')) {
    const v = el.dataset.hiddenIf.split('.').reduce((o, k) => (o == null ? o : o[k]), data);
    el.hidden = Boolean(v);
  }
}

/**
 * Turn a drawn field into a real one, in the box it already occupies.
 *
 * The boards draw fields as a flex row: an icon, a text div, sometimes a "Show"
 * link. Replacing the text div's contents with a block-level input at width:100%
 * blew the row apart -- the icon wrapped onto its own line -- and the input
 * inherited the div's letter-spacing, so a password placeholder came out spaced
 * like a display heading. Both are fixed here rather than in five page modules.
 */
export function asField(el, opts = {}) {
  if (!el) return null;
  const existing = el.querySelector('input, textarea');
  if (existing) return existing;
  const cs = getComputedStyle(el);
  const input = document.createElement(opts.multiline ? 'textarea' : 'input');
  if (!opts.multiline) input.type = opts.type || 'text';
  for (const k of ['name', 'placeholder', 'autocomplete', 'inputMode', 'maxLength', 'enterKeyHint', 'rows']) {
    if (opts[k] != null) input[k] = opts[k];
  }
  if (opts.label) input.setAttribute('aria-label', opts.label);
  input.value = opts.value || '';
  // keep the row intact: the field is a flex item that may shrink
  el.style.flex = el.style.flex || '1 1 auto';
  el.style.minWidth = '0';
  input.style.cssText = [
    'all: unset', 'display: block', 'width: 100%', 'box-sizing: border-box',
    `font-family: ${cs.fontFamily}`, `font-size: ${cs.fontSize}`, `font-weight: ${cs.fontWeight}`,
    'letter-spacing: normal', `color: ${cs.color}`, 'background: transparent',
    opts.multiline ? 'resize: vertical' : '',
  ].filter(Boolean).join('; ') + ';';
  el.replaceChildren(input);
  return input;
}

/**
 * Strip invented proof out of the drawn props.
 *
 * Several boards carry a "record of evidence" prop with a filename, a SHA-256
 * and a Bitcoin block number written on it, so the layout could be judged. On a
 * live page those are made-up figures dressed as proof, which is the single
 * thing this product must not do. They are replaced with a label saying what
 * would go there.
 *
 * Walks text nodes rather than elements: the props use <br> separators, so the
 * element has children and an element-level scan skips it entirely.
 */
export function deClaimProps(root) {
  const SWAP = [
    [/VID_\d{8}_\d{6}_IST\.mp4/g, 'the video file'],
    [/IMG_\d{8}_\d{4}_IST\.jpg/g, 'your photo'],
    [/sha256\s+[0-9a-f]{8}(?:…|\.\.\.)?[0-9a-f]{0,4}/gi, 'sha256 of the file itself'],
    [/\b[0-9a-f]{8}…[0-9a-f]{4}\b/g, '—'],
    [/\bbafybeih…?[0-9a-z]*/gi, '—'],
    [/\bblock\s*(?:9xx,xxx|[\d,]{4,})/gi, 'block number once its date lands'],
    [/\bBlock\s*(?:9xx,xxx|[\d,]{4,})/g, 'Block number once its date lands'],
    [/\b(?:0:41|0:38|0:36|0:44|0:29|0:35|0:22)\b/g, '—'],
    [/\b1080×1920\b/g, ''],
    [/\b30fps\b/g, ''],
    [/\b10\.1 MB\b/g, ''],
    [/no\.\s*0{3,}\d+/gi, 'no. —'],
    // the boards also write it bare, and inside a sentence
    [/\b8(?:81|78),\d{3}\b/g, 'the block'],
    [/\bCanal plot\b/g, 'a plot'],
    [/ICAR-CPRI Technical Bulletin 78, p\.34/g, 'the document the answer came from'],
    [/\bEarly blight\b/g, 'the diagnosis'],
  ];
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walk.nextNode()) nodes.push(walk.currentNode);
  for (const n of nodes) {
    let v = n.nodeValue;
    if (!v || !v.trim()) continue;
    let out = v;
    for (const [re, to] of SWAP) out = out.replace(re, to);
    if (out !== v) n.nodeValue = out.replace(/\s*·\s*·\s*/g, ' · ').replace(/\s{2,}/g, ' ');
  }
}

export function wire(slug) {
  const root = document.querySelector('body > div');
  if (!root) return;
  pressFeedback(root);
  promoteControls(root);
  links(root);
  deClaimProps(root);
  document.documentElement.dataset.ready = slug;
  return root;
}
