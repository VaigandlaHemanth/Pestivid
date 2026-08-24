import { deskNav } from './chrome.js';
import { session } from './api.js';
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
/** Promote anything that BECOMES a control after setup.
 *
 *  promoteControls() runs once, and page modules mark their controls after
 *  that -- 24 call sites across 10 modules set data-act by hand, so none of
 *  them got a role, a tabindex or a keyboard handler. Patching 24 call sites
 *  fixes today; watching the attribute fixes tomorrow's too.
 */
function watchControls(root) {
  const seen = new WeakSet();
  const fix = (el) => {
    if (seen.has(el)) return;
    seen.add(el);
    if (!el.hasAttribute('role')) el.setAttribute('role', el.dataset.act === 'link' ? 'link' : 'button');
    if (!el.hasAttribute('tabindex')) el.tabIndex = 0;
    if (el.dataset.kb) return;
    el.dataset.kb = '1';
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); }
    });
  };
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === 'attributes' && r.target.hasAttribute?.('data-act')) fix(r.target);
      for (const n of r.addedNodes || []) {
        if (n.nodeType !== 1) continue;
        if (n.hasAttribute?.('data-act')) fix(n);
        n.querySelectorAll?.('[data-act]').forEach(fix);
      }
    }
  }).observe(root, { subtree: true, childList: true, attributes: true, attributeFilter: ['data-act'] });
}

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
    if (/^https?:|^\//.test(to)) { location.href = to; return; }
    // A destination may carry a query -- data-go="plot?cid=bafy..." -- and the
    // extension belongs on the slug, not on the end of the whole string.
    const q = to.indexOf('?');
    location.href = q < 0 ? `./${to}.html` : `./${to.slice(0, q)}.html${to.slice(q)}`;
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
  // Two guidelines, applied once here instead of at thirty call sites:
  // a field with no autocomplete makes the browser guess, and a placeholder
  // that does not trail off reads as a value that is already there.
  if (!('autocomplete' in opts)) opts.autocomplete = 'off';
  // An ellipsis belongs on an instruction that trails off, not on a specimen of
  // the value. "98765 43210…" and "••••••…" read as part of what you are meant
  // to type, which is worse than no ellipsis at all.
  const specimen = /^[\d\s+•·.,₹-]+$/.test(opts.placeholder || '');
  if (opts.placeholder && !specimen && !/[…:]$/.test(opts.placeholder)) opts.placeholder += '…';
  // A code, an address or a phone number is not prose; do not underline it red.
  if (/code|otp|phone|tel|email|offer|amount|reply|question/.test(String(opts.name || opts.type || ''))) {
    opts.spellcheck = false;
  }
  if (!el) return null;
  const existing = el.querySelector('input, textarea');
  if (existing) return existing;
  const cs = getComputedStyle(el);
  const input = document.createElement(opts.multiline ? 'textarea' : 'input');
  if (!opts.multiline) input.type = opts.type || 'text';
  for (const k of ['name', 'placeholder', 'autocomplete', 'inputMode', 'maxLength', 'enterKeyHint', 'rows', 'spellcheck']) {
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
  // Token-by-token substitution turned the prop into word salad
  // ("the video file - . sha256 of the file itself block number on"), which is
  // worse than the invented figures it replaced. So: if a run of text is made
  // ONLY of prop tokens, the whole run goes and one honest line takes its
  // place. A token sitting inside a real sentence is still swapped in place.
  const PROP = /VID_\d{8}_\d{6}_IST\.mp4|IMG_\d{8}_\d{4}_IST\.jpg|sha256\s+[0-9a-f]{4,}(?:…|\.\.\.)?[0-9a-f]*|[0-9a-f]{8}…[0-9a-f]{4}|bafybeih…?[0-9a-z]*|[Bb]lock\s*(?:9xx,xxx|[\d,]{4,})|8(?:81|78),\d{3}|1080×1920|30fps|10\.1 MB|no\.\s*0{3,}\d+/g;
  const FILLER = /^[\s·,.\-–|/]*$/;

  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walk.nextNode()) nodes.push(walk.currentNode);

  // group the runs that belong to one prop: same parent, nothing but tokens
  const holders = new Set();
  for (const n of nodes) {
    const v = n.nodeValue;
    if (!v || !v.trim() || !PROP.test(v)) { PROP.lastIndex = 0; continue; }
    PROP.lastIndex = 0;
    const rest = v.replace(PROP, '');
    if (FILLER.test(rest)) holders.add(n.parentElement);
    else n.nodeValue = v.replace(PROP, (m) => IN_SENTENCE(m));
    PROP.lastIndex = 0;
  }

  for (const el of holders) {
    // The replacement is a sentence and needs room for one. Inside a 68px
    // thumbnail it rendered as a wrapped fragment of itself, which is worse
    // than the invented figure it replaced -- so in a box that narrow the
    // decoration is removed instead of re-lettered.
    // A BOUND field is about to be filled with the real thing, so it must
    // survive. Removing it left the orders receipt with four labels and four
    // blank values -- Transaction, Video file SHA-256, Stored at, Date written
    // into Bitcoin, all empty -- which reads as a page that failed to load.
    if (el.hasAttribute('data-bind')) { el.textContent = ''; continue; }
    const box = el.getBoundingClientRect();
    const holder = el.closest('[style*="overflow: hidden"]');
    if (box.width < 190 || (holder && holder.getBoundingClientRect().width < 190)) {
      // A LABELLED value must not be removed, for the same reason a bound one
      // must not: it orphans the label. The landing page's "Record of evidence"
      // lost its Fingerprint value that way -- a card whose entire argument is
      // the fingerprint, showing the word and then nothing. So when the holder
      // has a sibling that carries text, it gets the short honest value; only a
      // lone decoration is removed.
      const labelled = [...(el.parentElement?.children || [])]
        .some(sib => sib !== el && sib.textContent.trim());
      if (labelled) {
        el.textContent = SHORT_VALUE(el.textContent);
        el.setAttribute('data-declaimed', '');
        continue;
      }
      el.remove();
      continue;
    }
    el.textContent = 'The file, its fingerprint and the block its date went into';
    el.setAttribute('data-declaimed', '');
  }

  // What a stripped prop says when it is a VALUE beside its own label, where
  // "its fingerprint" next to "Fingerprint" would be a tautology and a sentence
  // would not fit. It says what will be there instead of pretending it is.
  function SHORT_VALUE(v) {
    if (/sha256|[0-9a-f]{8}…/i.test(v)) return 'computed on our server';
    if (/^\s*[Bb]lock/.test(v)) return 'once its date lands';
    if (/^VID_/.test(v)) return 'the file we received';
    if (/^IMG_/.test(v)) return 'the photo you took';
    if (/bafybeih/.test(v)) return 'pinned on IPFS';
    return 'filled in from the real record';
  }

  function IN_SENTENCE(m) {
    if (/^sha256/i.test(m)) return 'its fingerprint';
    if (/^[Bb]lock/.test(m)) return m[0] === 'B' ? 'Block number, once its date lands' : 'block number, once its date lands';
    if (/^VID_/.test(m)) return 'the video file';
    if (/^IMG_/.test(m)) return 'your photo';
    return '';
  }
}

/* ------------------------------------------------------------------ *
 * Making a div into a control.
 *
 * wire() runs promoteControls() once, at page setup, before a page module has
 * marked anything -- so a module that adds data-act afterwards got the click
 * delegation (that is on the root) but never the role, the tabindex or the
 * keyboard handler. Every page needs to mark its own controls after wire(),
 * so marking and promoting happen together here.
 * ------------------------------------------------------------------ */

/** Promote one element that has just been marked as a control. */
export function promote(el, name) {
  if (!el) return null;
  el.setAttribute('data-act', el.dataset.act || '');
  if (!el.hasAttribute('role')) el.setAttribute('role', 'button');
  if (!el.hasAttribute('tabindex')) el.tabIndex = 0;
  if (name && !el.getAttribute('aria-label')) el.setAttribute('aria-label', name);
  if (!el.dataset.kb) {
    el.dataset.kb = '1';
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); }
    });
  }
  return el;
}

/** A control inside another control has to stop there. The read-aloud button
 *  sits inside a whole-row link, so without this a press on the speaker also
 *  navigates -- and the row wins, because navigation happens first. Applied
 *  automatically, since the alternative is remembering it at every call site. */
function containIfNested(el) {
  if (!el.parentElement?.closest('[data-act]')) return;
  el.addEventListener('click', e => e.stopPropagation());
  el.addEventListener('pointerdown', e => e.stopPropagation());
}

/** A control that goes somewhere. The whole row, not the label inside it: a
 *  44px glyph in a 76px row leaves most of what looks pressable dead. */
export function goes(el, dest, name) {
  if (!el) return null;
  el.dataset.go = dest;
  promote(el, name);
  containIfNested(el);
  return el;
}

/** A control that does something here. */
export function acts(el, name, fn) {
  if (!el) return null;
  promote(el, name);
  containIfNested(el);
  el.addEventListener('click', fn);
  return el;
}

/** Press feedback for controls a module marked after wire() ran. */
export function press(root) { pressFeedback(root); }

/* ------------------------------------------------------------------ *
 * What the artboards cannot carry, because they are drawn.
 * ------------------------------------------------------------------ */

/** The page's title, announced as one.
 *
 *  The boards are div soup by necessity, so most pages handed a screen reader
 *  no heading at all -- 31 pages with nothing to navigate by. The board marks
 *  its title with data-title and this gives it the role, rather than retagging
 *  it: verify-layout compares tag names, and an attribute changes nothing that
 *  is drawn. Three boards already use a real <h1>; those are left alone.
 */
function heading(root) {
  if (root.querySelector('h1')) return;
  const t = root.querySelector('[data-title]');
  if (!t) return;
  t.setAttribute('role', 'heading');
  t.setAttribute('aria-level', '1');
}

/** Icons drawn beside their own label are decoration.
 *
 *  Every glyph in this product sits next to the words it illustrates, so a
 *  reader that announces them says everything twice. One with a <title> in it,
 *  or one that is the whole of a control, is left alone.
 */
function hideDecoration(root) {
  for (const svg of root.querySelectorAll('svg')) {
    if (svg.querySelector('title') || svg.getAttribute('role')) continue;
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');   // IE/old-Edge put SVGs in the tab order
  }
}

export function wire(slug) {
  const root = document.querySelector('body > div');
  if (!root) return;
  // The "bigger text" setting lives on the profile screen but has to hold on
  // every screen, or it is a note to itself. Zoom enlarges px type as well as
  // rem, which matters because these boards are drawn in px throughout.
  if (localStorage.getItem('pv.bigText') === '1') document.documentElement.style.zoom = '1.3';
  pressFeedback(root);
  promoteControls(root);
  links(root);
  deClaimProps(root);
  heading(root);
  hideDecoration(root);
  watchControls(root);
  // The desktop header, wired here rather than per page. Only seven of the
  // twenty-four pages called appChrome, and invest.js -- the investor's first
  // screen -- wired no navigation at all, so Portfolio and Messages were plain
  // text an investor could not click.
  deskNav(root, session.user);
  document.documentElement.dataset.ready = slug;
  return root;
}
