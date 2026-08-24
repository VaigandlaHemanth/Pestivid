// The home screen before anything has been filmed.
//
// It is not a separate destination: if the farmer has videos, they belong on the
// real home screen, so this redirects rather than showing a stale empty state.
import { requireUser, api, load } from './_guard.js';
import { bind } from '../bind.js';
import { appChrome } from '../chrome.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('home-empty', ['farmer']);

// The four rows on the first screen after signup were zebra bars, and not one
// was wired: only the last carried a chevron, so three read as inert labels
// while the one that looked pressable was equally dead. This is the screen a
// farmer sees before anything has happened, so every row it draws has to work.
if (ctx) {
  appChrome(ctx.root, { back: 'home', user: ctx.user });
  for (const [label, dest] of [
    ['My plots', 'plots'],
    ['Money', 'money'],
    ['Check a leaf', 'leaf-result'],
    ['Ask a question', 'ask'],
  ]) {
    const row = [...ctx.root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === label)?.closest('.row0');
    if (row) goes(row, dest, label);
  }
  press(ctx.root);
}

if (ctx) load(ctx.root, async () => {
  const [me, videos] = await Promise.all([
    api.auth.me(),
    api.videos.mine(ctx.user._id || ctx.user.id).catch(() => []),
  ]);
  if (videos.length) { location.replace('./home.html'); return; }
  bind(ctx.root, { whoWhere: me.name });
  const who = [...ctx.root.querySelectorAll('div')]
    .find(d => d.children.length === 0 && d.textContent.trim() === 'Alice');
  if (who) who.textContent = (me.name || '').split(' ')[0] || me.name;
  const leaf = [...ctx.root.querySelectorAll('.m')]
    .find(d => /173 MB/.test(d.textContent));
  if (leaf) leaf.textContent = localStorage.getItem('pv.model') ? 'Ready · works offline' : 'Get it · 173 MB';
});
