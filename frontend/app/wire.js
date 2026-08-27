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
/* ════ MONEY GROUPS ITSELF ══════════════════════════════════════════════════
 *
 * The harvest form drew "2,58,400" on the artboard and answered a bare "945000"
 * the moment somebody typed -- two figures on one screen formatted two ways, and
 * the one the farmer was responsible for was the unformatted one. The same trap
 * was live on ask-money, whose "How much do you need?" specimen reads
 * "5,00,000".
 *
 * So it is not a thing a page has to remember. If the SPECIMEN is grouped, the
 * field groups: asField sees the comma in the placeholder and wires it. Any new
 * money box gets it by being drawn with a grouped specimen, and verify-forms
 * holds the same rule from the outside.
 *
 * en-IN is lakh grouping -- three digits then pairs -- which is what rupees()
 * already uses for every printed figure in this product.
 */
export const digitsOnly = (v) => String(v).replace(/[^\d]/g, '');
export const grouped = (v) => {
  const d = digitsOnly(v).replace(/^0+(?=\d)/, '');
  return d ? Number(d).toLocaleString('en-IN') : '';
};

/* Reformatting rewrites the whole value, which sends the caret to the end. Fine
 * while typing at the end, wrong the moment somebody corrects a digit in the
 * middle. So the position is kept in DIGITS -- the one unit the commas cannot
 * move -- and mapped back afterwards.
 */
export function groupLive(input) {
  if (!input || input.__grouped) return input;
  input.__grouped = true;
  input.addEventListener('input', () => {
    const before = input.value;
    const caret = input.selectionStart ?? before.length;
    const digitsBefore = digitsOnly(before.slice(0, caret)).length;
    const next = grouped(before);
    if (next === before) return;
    input.value = next;
    let seen = 0, pos = next.length;
    for (let i = 0; i < next.length; i++) {
      if (/\d/.test(next[i])) seen++;
      if (seen === digitsBefore) { pos = i + 1; break; }
    }
    if (digitsBefore === 0) pos = 0;
    try { input.setSelectionRange(pos, pos); } catch { /* not a text input */ }
  });
  return input;
}

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

  // A grouped specimen is a promise about the value. Kept here so no page has to
  // remember, and so a new money box gets it just by being drawn with commas.
  if (/\d,\d\d/.test(input.placeholder || '')) groupLive(input);

  // The tap target is the BOX, not the text slot inside it.
  //
  // `el` is the drawn text node -- 21px tall inside a field that looks 48px --
  // so the input landed as a sliver in the middle of it and a finger aimed at
  // the top or bottom third of the field hit nothing. That is what "I cannot
  // type in that text box" was. The drawn geometry is not moved, because
  // verify-layout compares it: instead the visible box focuses the input, the
  // same way the six-digit code strip already does.
  // Found by SIZE, walking up a few levels: a field box is taller than its text
  // slot and still field-sized. A selector on background colour matched the
  // whole 269px footer band instead, and put a text cursor on it.
  const mine = el.getBoundingClientRect().height;
  let box = null;
  for (let n = el.parentElement, hop = 0; n && hop < 4; n = n.parentElement, hop++) {
    const h = n.getBoundingClientRect().height;
    if (h > mine + 4 && h <= Math.max(96, mine * 4)) { box = n; break; }
    if (h > Math.max(96, mine * 4)) break;
  }
  if (box) {
    // Named, not inferred. `cursor: text` is an INHERITED property, so every
    // descendant of the box computes to it and a walk looking for it matches the
    // input's own parent -- which is the 17px sliver, not the 56px box.
    box.setAttribute('data-fieldbox', '');
    box.style.cursor = 'text';
    box.addEventListener('mousedown', (e) => {
      if (e.target === input || input.contains(e.target)) return;
      // A control inside the box -- a Show link, a send arrow -- keeps its click.
      if (e.target.closest?.('[data-act]') && !box.matches('[data-act]')) return;
      e.preventDefault();
      input.focus();
    });
  }
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
  const FILLER = /^[\s·,.\-, |/]*$/;

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

/* The painted box, not the word inside it.
 *
 * Half the boards draw a button as a filled div with a bare label div inside,
 * and a module marks the LABEL -- so "Continue" was a 77x24 control sitting in
 * the middle of a 568x60 blue rectangle, and clicking the rectangle anywhere
 * except the seven characters of the word did nothing at all. That is a dead
 * button as far as a user is concerned, and the HIG minimum control size
 * (28x28 with a pointer, 44x44 by touch) is the measure that catches it.
 *
 * Climbs ONE level, and only when there is no ambiguity: the parent is painted
 * or ringed, it is bigger, it holds nothing but this label, and it is not
 * already a control itself. A list row with several children is left alone.
 */
function paintedTarget(el) {
  const up = el.parentElement;
  if (!up || up.matches('[data-act], [data-go], a[href], button')) return el;
  // A parent that holds the label plus a decorative glyph is still one button.
  // What disqualifies it is another CONTROL or other text.
  const siblings = [...up.children].filter(n => n !== el);
  if (siblings.some(n => n.matches('[data-act], [data-go], a[href], button, input'))) return el;
  if ((up.textContent || '').trim() !== (el.textContent || '').trim()) return el;
  const mine = el.getBoundingClientRect();
  const theirs = up.getBoundingClientRect();
  if (theirs.height <= mine.height + 3 && theirs.width <= mine.width + 3) return el;
  if (theirs.height > 96) return el;
  const cs = getComputedStyle(up);
  const bg = cs.backgroundColor || '';
  const painted = bg && bg !== 'transparent' && !/rgba\(0, 0, 0, 0\)/.test(bg);
  const ringed = cs.boxShadow && cs.boxShadow !== 'none';
  const own = getComputedStyle(el).backgroundColor;
  if (own && own !== bg && !/rgba\(0, 0, 0, 0\)/.test(own)) return el;
  // An UNPAINTED parent counts too when it holds nothing but these words: a
  // market row is a 59px band with a 26px label in it, and the dead 33px was
  // the part of the row a thumb actually lands on.
  if (!painted && !ringed && theirs.height < mine.height + 12) return el;
  return up;
}

/** Promote one element that has just been marked as a control. */
export function promote(el, name) {
  if (!el) return null;
  el = paintedTarget(el);
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
  const target = promote(el, name);
  target.dataset.go = dest;
  containIfNested(target);
  return el;
}

/** A control that does something here. */
export function acts(el, name, fn) {
  if (!el) return null;
  // promote() may hand the role to the painted box around a bare label. The
  // handler has to go on the same element, or the padding stays dead.
  const target = promote(el, name);
  containIfNested(target);
  target.addEventListener('click', fn);
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
