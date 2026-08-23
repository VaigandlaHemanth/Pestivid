import { requireUser, api, session, load } from './_guard.js';
import { bind } from '../bind.js';

const ctx = requireUser('profile');
if (ctx) load(ctx.root, async () => {
  const me = await api.auth.me();
  bind(ctx.root, {
    name: me.name,
    phone: me.phone || 'No number on file',
    where: me.location || 'No district on file',
    // the model is a local file, so its size is a fact about this handset
    model: { state: localStorage.getItem('pv.model') ? '173 MB downloaded · works offline' : 'Not downloaded yet' },
  });
  const out = [...ctx.root.querySelectorAll('div')].find(d => d.textContent.trim() === 'Sign out' && !d.children.length);
  out?.setAttribute('data-act', '');
  out?.addEventListener('click', () => { session.clear(); location.href = './signin.html'; });
});
