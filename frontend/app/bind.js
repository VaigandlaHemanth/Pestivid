// Data binding for markup that must not be edited.
//
// The screens are cut from the artboards, so nothing here rewrites them. The
// artboards carry inert attributes -- data-bind, data-repeat, data-when -- that
// change nothing at rest and are filled in at runtime.
//
// One rule throughout: a value that did not arrive is never left showing the
// placeholder it was drawn with. Placeholder money on a money screen is worse
// than an empty one.

const dig = (obj, path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);

/** Fill one subtree from one object. */
export function bind(root, data) {
  // Two calls over the same root must not wipe each other. A path whose FIRST
  // segment is absent from this call's data was not addressed by it, so it is
  // left alone; a path that is addressed and empty becomes an em dash. Without
  // that distinction a second bind() blanked everything the first had filled,
  // which is exactly what emptied the investor nav.
  const addressed = (path) => Object.prototype.hasOwnProperty.call(data || {}, path.split('.')[0]);
  for (const el of root.querySelectorAll('[data-bind]')) {
    if (!addressed(el.dataset.bind)) continue;
    const v = dig(data, el.dataset.bind);
    el.textContent = v == null || v === '' ? 'not yet' : String(v);
    el.setAttribute('data-filled', '');
  }
  for (const el of root.querySelectorAll('[data-bind-html]')) {
    const v = dig(data, el.dataset.bindHtml);
    if (v != null) { el.textContent = String(v); el.setAttribute('data-filled', ''); }
  }
  for (const el of root.querySelectorAll('[data-when]')) {
    let key = el.dataset.when, want = true;
    if (key.startsWith('!')) { key = key.slice(1); want = false; }
    el.hidden = Boolean(dig(data, key)) !== want;
  }
  for (const el of root.querySelectorAll('[data-style]')) {
    const [prop, path] = el.dataset.style.split(':');
    const v = dig(data, path.trim());
    if (v != null) el.style.setProperty(prop.trim(), String(v));
  }
  for (const el of root.querySelectorAll('[data-goto]')) {
    const v = dig(data, el.dataset.goto);
    if (v != null) el.dataset.go = String(v);
  }
  return root;
}

/* A list arriving from the server ---------------------------------------------
 *
 * Every list on this product is empty until a fetch resolves, and then the rows
 * appear all at once with no bridge. verify-motion-coverage put a number on it:
 * seventeen of twenty-four pages had press feedback and nothing else, so the one
 * moment where the screen changes shape was the one moment with no motion.
 *
 * One recipe, applied in the shared row helper rather than seventeen times:
 *   purpose    preventing a jarring change -- rows replace a blank
 *   frequency  once per page load, which is where a stagger is affordable
 *   budget     568ms on the snappy spring, the token this product already uses
 *   stagger    45ms, capped at eight rows so a long list does not crawl
 *
 * transform and opacity only, one shot, and under reduced motion the fade stays
 * while the travel goes -- the rule this repo already holds itself to.
 */
const STILL = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
export function arrive(els) {
  if (!els || !els.length) return;
  const still = STILL();
  els.forEach((el, i) => {
    if (!el || el.nodeType !== 1) return;
    const wait = Math.min(i, 8) * 45;
    el.style.opacity = '0';
    if (!still) el.style.transform = 'translateY(8px)';
    el.style.transition = `opacity var(--t-press, 120ms) var(--e-smooth, ease) ${wait}ms,`
      + ` transform var(--t-snappy, 568ms) var(--e-snappy, ease) ${wait}ms`;
  });
  requestAnimationFrame(() => {
    for (const el of els) {
      if (!el || el.nodeType !== 1) continue;
      el.style.opacity = '1';
      el.style.transform = 'none';
    }
  });
}

/**
 * Repeat the first child of a container once per row.
 *
 * The artboards were drawn with two or three example rows so the spacing could
 * be judged; the first is the template and the rest are removed. Order matters:
 * the template is taken before anything is cleared.
 */
export function repeat(container, rows, decorate) {
  if (!container) return;
  if (!container.__tpl) {
    const first = container.firstElementChild;
    if (!first) return;
    container.__tpl = first.cloneNode(true);
  }
  container.replaceChildren();
  const made = rows.map((row, i) => {
    const el = container.__tpl.cloneNode(true);
    bind(el, row);
    if (decorate) decorate(el, row, i);
    container.append(el);
    return el;
  });
  arrive(made);
}

/**
 * Replace a subtree with an honest message. Used for empty, failed and
 * rate-limited: three different things that must not look like each other.
 */
/**
 * A message that replaces whatever it is handed.
 *
 * @param action optional {label, go} -- a way out. An empty state that names an
 *   action ("Film your field first") and then offers no way to take it leaves
 *   the person on a blank page with no exit, which is what this produced on the
 *   send screen: one box at the top of 1,200px of nothing. If the state names a
 *   thing to do, it has to be doable from here.
 */
export function state(container, kind, headline, detail, action) {
  if (!container) return;
  const tone = {
    empty:   ['#eae4de', '#1d1a17', '#4a443d'],
    failed:  ['#f7e9e6', '#a71930', '#4a443d'],
    waiting: ['#f2e6cd', '#7c4a12', '#4a443d'],
  }[kind] || ['#eae4de', '#1d1a17', '#4a443d'];
  const box = document.createElement('div');
  box.setAttribute('role', kind === 'failed' ? 'alert' : 'status');
  // The 20px side margin exists so a box dropped into an UNPADDED container
  // lines up with padded content around it. In a container that is already
  // padded it indents twice, which is what made the setup screen's notice sit
  // 20px inside every panel above it. So: ask the container.
  // Nine call sites append an unpadded holder div into a padded panel, so
  // asking the container alone still indented twice. Climb while the padding is
  // zero: the question is "am I already inside something padded", not "is my
  // immediate parent padded".
  let side = 20;
  for (let n = container instanceof Element ? container : null, hop = 0;
       n && hop < 4; n = n.parentElement, hop++) {
    const pl = parseFloat(getComputedStyle(n).paddingLeft) || 0;
    if (pl > 8) { side = 0; break; }
  }
  // These arrive in place of something the reader was looking at -- an error
  // over a form, an empty state over a list -- so they come in rather than
  // teleport. Opacity and transform only, and the travel is 6px: this is a
  // message appearing, not a panel sliding.
  // A sentence set across 1400px is not a sentence anybody reads. These boxes
  // were drawn for a 360px panel and inherited the whole laptop width when the
  // farmer pages widened.
  // The hairline is part of the box, not a stylesheet rule: cssText serialises
  // the background to rgb(), so nothing keyed on the hex could reach it, and the
  // amber attention band sat at 1.154:1 against the page it was warning about.
  const edge = kind === 'failed' ? '#e0b4ac' : kind === 'waiting' ? '#d9c69b' : '#c3bcb6';
  box.style.cssText = `background: ${tone[0]}; padding: 18px 20px; margin: 16px ${side}px;`
    + ` box-shadow: inset 0 0 0 1px ${edge};`
    + ' max-width: 720px; box-sizing: border-box;'
    + ' opacity: 0; transform: translateY(6px);'
    + ' transition: opacity var(--t-press, 120ms) var(--e-smooth, ease),'
    + ' transform var(--t-release, 260ms) var(--e-snappy, ease);';
  requestAnimationFrame(() => {
    box.style.opacity = '1';
    box.style.transform = 'none';
  });
  const h = document.createElement('div');
  h.style.cssText = `font-size: 15.5px; font-weight: 700; color: ${tone[1]};`;
  h.textContent = headline;
  const p = document.createElement('div');
  p.style.cssText = `font-size: 14.5px; line-height: 1.5; margin-top: 4px; color: ${tone[2]};`;
  p.textContent = detail;
  box.style.display = 'flex';
  box.style.flexDirection = 'column';
  box.append(h, p);
  // An action is either somewhere to go or something to do. The record screen's
  // camera refusal needs the second kind: it opens a file picker rather than
  // navigating, and without it that screen was a red panel over a black void
  // with no way forward at all.
  if (action && action.label && (action.go || action.act)) {
    const a = document.createElement('div');
    if (action.go) a.dataset.go = action.go;
    if (action.act) a.addEventListener('click', action.act);
    a.dataset.act = '';
    a.setAttribute('role', 'button');
    a.tabIndex = 0;
    // A button is the width of its label plus room to hit it, not the width of
    // whatever it happens to sit in.
    a.style.cssText = 'margin-top: 14px; min-height: 48px; background: #1d1a17; color: #fff;'
      + ' display: inline-flex; align-self: flex-start; padding: 0 28px;'
      + ' align-items: center; justify-content: center;'
      + ' font-size: 16px; font-weight: 600; cursor: pointer;';
    a.textContent = action.label;
    a.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); a.click(); }
    });
    box.append(a);
  }
  // An empty state must not replace the PAGE. Nine call sites handed this the
  // page root, and replaceChildren() then took the header, the heading and the
  // back arrow with it -- so "No season chosen" arrived on a screen with no
  // title and no way off it. Handed a page root, it keeps the header and
  // replaces what is below.
  if (container.matches?.('body > div') && container.children.length > 1) {
    const keep = [];
    for (const child of container.children) {
      // the header is whatever holds the title or the chrome
      if (child.querySelector?.('[data-title], [data-chrome]') || child.matches?.('[data-title], [data-chrome]')) {
        keep.push(child);
        continue;
      }
      if (!keep.length && child === container.firstElementChild) keep.push(child);
    }
    if (keep.length && keep.length < container.children.length) {
      container.replaceChildren(...keep, box);
      return;
    }
  }
  container.replaceChildren(box);
}

/** The message for a failure, in the words the screens use elsewhere. */
export function reason(err) {
  if (err?.offline) return ['No connection', 'Your phone cannot reach us. Nothing is lost, try again when you have a bar of signal.'];
  if (err?.rateLimited) return ['Too many questions just now', 'We are on a free allowance and it is used up for the minute. Wait a moment and ask again.'];
  if (err?.status === 404) return ['Not there', 'That is not something we hold.'];
  if (err?.status >= 500) return ['Our end broke', 'This is our fault, not yours. Nothing you sent has been lost.'];
  return ['That did not work', err?.message || 'Something went wrong and we do not know what.'];
}

/** Wraps a page load so a thrown error always lands somewhere visible. */
/* A LOADER, while a fetch is in flight -------------------------------------
 *
 * load() handled the failure and showed nothing at all while waiting, so every
 * page sat with the artboard's drawn example on screen until real data replaced
 * it -- which reads as "these are your figures" for as long as the fetch takes.
 *
 * Two shapes, and which one is not a style choice:
 *   working()  the pill. Work in flight with no figure to report.
 *   writing()  the pencil. A date being WRITTEN: a video waiting for the day's
 *              hashes to land in a Bitcoin block. The product already says
 *              "date being written, usually by tomorrow" in words.
 *
 * Both are in app/loaders.css with the attribution and the reason they loop.
 */
const PENCIL = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" class="pencil pencil--sm" aria-hidden="true" focusable="false">
  <defs><clipPath id="pv-eraser"><rect height="30" width="30" ry="5" rx="5"></rect></clipPath></defs>
  <circle transform="rotate(-113,100,100)" stroke-linecap="round" stroke-dashoffset="439.82" stroke-dasharray="439.82 439.82" stroke-width="4" stroke="currentColor" fill="none" r="70" class="pencil__stroke"></circle>
  <g transform="translate(100,100)" class="pencil__rotate">
    <g fill="none">
      <circle transform="rotate(-90)" stroke-dashoffset="402" stroke-dasharray="402.12 402.12" stroke-width="30" stroke="#016abe" r="64" class="pencil__body1"></circle>
      <circle transform="rotate(-90)" stroke-dashoffset="465" stroke-dasharray="464.96 464.96" stroke-width="10" stroke="#0a86e0" r="74" class="pencil__body2"></circle>
      <circle transform="rotate(-90)" stroke-dashoffset="339" stroke-dasharray="339.29 339.29" stroke-width="10" stroke="#01579b" r="54" class="pencil__body3"></circle>
    </g>
    <g transform="rotate(-90) translate(49,0)" class="pencil__eraser">
      <g class="pencil__eraser-skew">
        <rect height="30" width="30" ry="5" rx="5" fill="#a71930"></rect>
        <rect clip-path="url(#pv-eraser)" height="30" width="5" fill="#8a1428"></rect>
        <rect height="20" width="30" fill="#e3ddd6"></rect>
        <rect height="20" width="15" fill="#c3bcb6"></rect>
        <rect height="20" width="5" fill="#d3ccc5"></rect>
      </g>
    </g>
    <g transform="rotate(-90) translate(49,-30)" class="pencil__point">
      <polygon points="15 0,30 30,0 30" fill="#e8cfa6"></polygon>
      <polygon points="15 0,6 30,0 30" fill="#c9a262"></polygon>
      <polygon points="15 0,20 10,10 10" fill="#1d1a17"></polygon>
    </g>
  </g>
</svg>`;

const PILL = `<div class="loader" aria-hidden="true">
  <span class="loader-text">working</span>
  <span class="load"></span>
</div>`;

function loaderBox(kind, said) {
  const box = document.createElement('div');
  box.setAttribute('data-loader', kind);
  // status, not alert: it is progress, and it must not interrupt a reader.
  box.setAttribute('role', 'status');
  box.setAttribute('aria-live', 'polite');
  box.innerHTML = (kind === 'writing' ? PENCIL : PILL)
    + `<div class="said">${said}</div>`;
  return box;
}

/* Where a loader goes, without disturbing anything.
 *
 * Two attempts, both wrong, and the second is the instructive one:
 *   append to the container -> it landed at the very bottom-left of the page,
 *     past every rail card, unpadded. A message about the list you are waiting
 *     for, a thousand pixels below the list.
 *   prepend to the content column -> the "column" resolved to the two-column
 *     GRID on pages whose board has no .main, so the loader became a grid item
 *     and pushed the table into the rail's track. A loading state must never
 *     move the layout it is loading.
 *
 * So it does not enter the layout at all. It is a band directly after the app
 * bar, full width with the page's own side padding, and it cannot be anyone's
 * grid or flex child because the bar's parent is the page's column stack.
 */
function loaderSlot(container) {
  const bar = container.querySelector('.appbar, .bar, header');
  return { parent: (bar && bar.parentElement) || container, after: bar || null };
}
function placeLoader(container, box) {
  const { parent, after } = loaderSlot(container);
  box.style.padding = '20px 28px';
  if (after && after.parentElement === parent) after.after(box);
  else parent.prepend(box);
}

/** The pill, while something is in flight. Returns a function that removes it. */
export function working(container, said = 'Fetching this from the server.') {
  if (!container) return () => {};
  const box = loaderBox('working', said);
  placeLoader(container, box);
  return () => box.remove();
}

/** The pencil, for a date being written into a block. */
export function writing(container, said = 'The date is being written. Usually by tomorrow.') {
  if (!container) return () => {};
  const box = loaderBox('writing', said);
  placeLoader(container, box);
  return () => box.remove();
}

export async function load(container, fn) {
  // A loader only appears if the work outlasts a beat. Showing one for a fetch
  // that resolves in 80ms is a flash of noise, so it waits 220ms first.
  let stop = () => {};
  const timer = setTimeout(() => { stop = working(container); }, 220);
  try {
    await fn();
  } catch (err) {
    clearTimeout(timer); stop();
    console.error(err);
    const [h, d] = reason(err);

    // A failed load used to leave every DRAWN figure on screen and append the
    // message at the bottom. On confirm-investment that meant an investor read
    // "You are sending ₹50,000 / your share 60% / this project still needs
    // ₹3,20,000" -- all of it the artboard's -- with the failure notice BELOW
    // the send button. Every field the page meant to fill and did not now says
    // so, in place.
    const unfilled = [...container.querySelectorAll('[data-bind], [data-bind-html]')]
      .filter(el => !el.hasAttribute('data-filled'));
    for (const el of unfilled) el.textContent = 'not loaded';

    // And you can leave, but you cannot act. Navigation stays live; anything
    // that would commit is refused, because the page does not know what it is
    // acting on.
    const box = document.createElement('div');
    container.append(box);
    for (const el of container.querySelectorAll('[data-act]')) {
      if (el.closest('[data-chrome]') || el.hasAttribute('data-chrome')
          || el.hasAttribute('data-back') || box.contains(el)) continue;
      el.setAttribute('aria-disabled', 'true');
      el.style.opacity = '0.55';
      el.style.pointerEvents = 'none';
    }
    state(box, err?.rateLimited ? 'waiting' : 'failed', h, d);
    // Put the reader at the message rather than wherever they happened to be.
    box.scrollIntoView({ block: 'center' });
  } finally {
    // Both paths: a loader that outlives its fetch is a page that looks stuck.
    clearTimeout(timer);
    stop();
  }
}

/** The container a repeated row lives in, found from any one of its rows. */
export const listOf = (sel, root = document) => root.querySelector(sel)?.parentElement || null;

/** Every element whose own text is exactly this, useful on div-only markup. */
export const byText = (text, root = document) =>
  [...root.querySelectorAll('div,td,span')].filter(el =>
    el.children.length === 0 && el.textContent.trim() === text);

/** The first of those. */
export const oneByText = (text, root = document) => byText(text, root)[0] || null;

/**
 * Repeat only the elements matching `selector`, leaving their siblings alone.
 *
 * repeat() assumes the container holds nothing but rows. Several boards do not
 * work that way: the plot list, the payout list and the message list all put a
 * section heading and its note in the same parent as the rows. Taking the first
 * child as the template there cloned the heading once per row and dropped the
 * rows entirely.
 *
 * This takes the first match as the template, removes the rest of the matches,
 * and inserts the clones where the first one stood.
 */
export function repeatRows(root, selector, rows, decorate) {
  const found = [...root.querySelectorAll(selector)];
  if (!found.length) return null;
  const anchor = document.createComment('rows');
  found[0].before(anchor);
  const tpl = found[0].cloneNode(true);
  found.forEach(el => el.remove());
  const made = rows.map((row, i) => {
    const el = tpl.cloneNode(true);
    bind(el, row);
    if (decorate) decorate(el, row, i);
    anchor.before(el);
    return el;
  });
  anchor.remove();
  arrive(made);
  return made;
}

/* ------------------------------------------------------------------ *
 * Named rows.
 *
 * repeatRows() takes a CSS selector and its callers were matching substrings of
 * the style attribute, then picking children by position -- "the first leaf div
 * that has text". That put a crop name inside a thumbnail and dropped a price,
 * and it means any change to the drawing silently rewires the code.
 *
 * So the artboards name their own parts instead: data-row="payout" marks a
 * repeatable row, data-slot="amount" names a field in it, data-sec="paid" marks
 * the heading that introduces the group. None of it paints anything.
 * ------------------------------------------------------------------ */

/**
 * Clone a named row once per item and fill its named slots.
 *
 * A slot whose value is `null` is REMOVED rather than left showing whatever the
 * artboard drew -- that is the difference between "this lot has no price yet"
 * and "this lot costs 1,250 rupees because the mock said so". A slot the item
 * does not mention at all is left alone.
 *
 * @returns the elements created, in order.
 */
export function rows(root, name, items, decorate) {
  const all = [...root.querySelectorAll(`[data-row="${name}"]`)];
  const tpl = all[0];
  if (!tpl) return [];
  all.slice(1).forEach(n => n.remove());     // extra examples of the same row
  const parent = tpl.parentElement;
  const anchor = tpl.nextSibling;
  if (!items.length) { tpl.remove(); return []; }

  const made = [];
  for (const item of items) {
    const el = tpl.cloneNode(true);
    for (const slot of el.querySelectorAll('[data-slot]')) {
      const v = item[slot.dataset.slot];
      if (v === undefined) continue;
      if (v === null) { slot.remove(); continue; }
      slot.textContent = String(v);
    }
    parent.insertBefore(el, anchor);
    decorate?.(el, item);
    made.push(el);
  }
  tpl.remove();
  arrive(made);
  return made;
}

/** One slot inside a row built by rows(). */
export const slot = (el, name) => el?.querySelector(`[data-slot="${name}"]`) || null;

/**
 * Remove a whole section: the data-sec heading that introduces it and every
 * named row that belongs to it. Used when there is genuinely nothing to show --
 * an empty section with the artboard's examples still in it is how this page
 * ended up saying "No season closed yet" above three investors marked "sent".
 */
export function dropSection(root, sec, ...rowNames) {
  root.querySelector(`[data-sec="${sec}"]`)?.remove();
  for (const n of rowNames)
    root.querySelectorAll(`[data-row="${n}"]`).forEach(el => el.remove());
}

/**
 * Markdown that a model emitted anyway, turned back into readable text.
 *
 * Two places show model prose: the chatbot's bubbles and the leaf checker's
 * refusal. Both set textContent, so "- **Check the label**" reached a farmer
 * with every asterisk and hyphen intact. Text in, text out -- no HTML is ever
 * built from model output.
 *
 * Written with String.fromCharCode for the newline and the bullet: composing
 * this file from a script twice turned \n inside a regex literal into a real
 * line break, and the second time it shipped a SyntaxError to two pages.
 */
export function plainText(md) {
  const NL = String.fromCharCode(10);
  const BULLET = String.fromCharCode(8226, 32);
  return String(md || '')
    .replace(/```[\s\S]*?```/g, m => m.replace(/```/g, "").trim())
    .replace(/^#{1,6}[ \t]*/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    // The trailing lookahead allows punctuation, or "your *extension officer*."
    // keeps its asterisks -- which is exactly how it arrives.
    .replace(/(^|[ \t])\*([^*\r\n]+)\*(?=[ \t.,;:!?)\]]|$)/gm, '$1$2')
    .replace(/(^|[ \t])_([^_\r\n]+)_(?=[ \t.,;:!?)\]]|$)/gm, '$1$2')
    .replace(/^[ \t]*[-*+][ \t]+/gm, BULLET)
    .replace(/\u2011/g, '-')
    .replace(/(\r?\n){3,}/g, NL + NL)
    .trim();
}
