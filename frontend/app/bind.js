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
  for (const el of root.querySelectorAll('[data-bind]')) {
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
export function state(container, kind, headline, detail) {
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
