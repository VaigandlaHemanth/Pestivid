import { requireUser, api, load, state } from './_guard.js';

const ctx = requireUser('admin', ['admin']);
if (ctx) load(ctx.root, async () => {
  const { bind } = await import('../bind.js');
  bind(ctx.root, { me: { line: `${ctx.user.name} · admin · every action here is written to the audit collection under your name` } });
  const q = await api.admin.flagged();
  const items = q?.videos || q || [];
  const tab = ctx.root.querySelector('.tabOn');

  if (!items.length) {
    // Not state(ctx.root, ...): that ends in replaceChildren() on the page root,
    // which took the heading, the nav and the tabs with it -- the page reported
    // no heading of any kind, and an admin lost every way off the screen. The
    // message replaces the QUEUE, which is the thing that is empty.
    const queue = ctx.root.querySelector('table')?.parentElement
      || ctx.root.querySelector('[data-list]')
      || tab?.closest('div')?.nextElementSibling
      || ctx.root.lastElementChild;
    if (tab) tab.textContent = 'Same footage, two farmers · 0';
    return state(queue, 'empty', 'The system has flagged nothing',
      'No duplicate footage, no unverified hashes, no late timestamps. This page being empty is the normal state.');
  }
  if (tab) tab.textContent = `Same footage, two farmers · ${items.length}`;
});
