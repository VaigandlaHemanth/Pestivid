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
    el.textContent = v == null || v === '' ? '—' : String(v);
  }
  for (const el of root.querySelectorAll('[data-bind-html]')) {
    const v = dig(data, el.dataset.bindHtml);
    if (v != null) el.textContent = String(v);
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
  rows.forEach((row, i) => {
    const el = container.__tpl.cloneNode(true);
    bind(el, row);
    if (decorate) decorate(el, row, i);
    container.append(el);
  });
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
  box.style.cssText = `background: ${tone[0]}; padding: 16px 17px; margin: 16px 20px;`;
  const h = document.createElement('div');
  h.style.cssText = `font-size: 15.5px; font-weight: 700; color: ${tone[1]};`;
  h.textContent = headline;
  const p = document.createElement('div');
  p.style.cssText = `font-size: 14.5px; line-height: 1.5; margin-top: 4px; color: ${tone[2]};`;
  p.textContent = detail;
  box.append(h, p);
  if (action && action.label && action.go) {
    const a = document.createElement('div');
    a.dataset.go = action.go;
    a.dataset.act = '';
    a.setAttribute('role', 'button');
    a.tabIndex = 0;
    a.style.cssText = 'margin-top: 13px; min-height: 48px; background: #1d1a17; color: #fff;'
      + ' display: flex; align-items: center; justify-content: center;'
      + ' font-size: 16px; font-weight: 600; cursor: pointer;';
    a.textContent = action.label;
    a.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); a.click(); }
    });
    box.append(a);
  }
  container.replaceChildren(box);
}

/** The message for a failure, in the words the screens use elsewhere. */
export function reason(err) {
  if (err?.offline) return ['No connection', 'Your phone cannot reach us. Nothing is lost — try again when you have a bar of signal.'];
  if (err?.rateLimited) return ['Too many questions just now', 'We are on a free allowance and it is used up for the minute. Wait a moment and ask again.'];
  if (err?.status === 404) return ['Not there', 'That is not something we hold.'];
  if (err?.status >= 500) return ['Our end broke', 'This is our fault, not yours. Nothing you sent has been lost.'];
  return ['That did not work', err?.message || 'Something went wrong and we do not know what.'];
}

/** Wraps a page load so a thrown error always lands somewhere visible. */
export async function load(container, fn) {
  try {
    await fn();
  } catch (err) {
    console.error(err);
    const [h, d] = reason(err);
    // append rather than replace: losing signal should not also lose the nav
    const holder = document.createElement('div');
    container.append(holder);
    state(holder, err?.rateLimited ? 'waiting' : 'failed', h, d);
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
