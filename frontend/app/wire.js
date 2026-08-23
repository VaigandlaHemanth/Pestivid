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

export function wire(slug) {
  const root = document.querySelector('body > div');
  if (!root) return;
  pressFeedback(root);
  promoteControls(root);
  links(root);
  document.documentElement.dataset.ready = slug;
}
