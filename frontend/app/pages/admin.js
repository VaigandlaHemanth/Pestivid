import { requireUser, api, load, state } from './_guard.js';

const ctx = requireUser('admin', ['admin']);
if (ctx) load(ctx.root, async () => {
  const { bind } = await import('../bind.js');
  bind(ctx.root, { me: { line: `${ctx.user.name} · admin · every action here is written to the audit collection under your name` } });
  const q = await api.admin.flagged();
  const items = q?.videos || q || [];
  if (!items.length) {
    return state(ctx.root, 'empty', 'The system has flagged nothing',
      'No duplicate footage, no unverified hashes, no late timestamps. This page being empty is the normal state.');
  }
  const tab = ctx.root.querySelector('.tabOn');
  if (tab) tab.textContent = `Same footage, two farmers · ${items.length}`;
});
