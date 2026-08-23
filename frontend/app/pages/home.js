import { requireUser, api, load, state } from './_guard.js';
import { bind, oneByText } from '../bind.js';

const ctx = requireUser('home', ['farmer']);
if (ctx) load(ctx.root, async () => {
  const [me, videos, projects] = await Promise.all([
    api.auth.me(),
    api.videos.mine(ctx.user._id || ctx.user.id),
    api.projects.mine(ctx.user._id || ctx.user.id).catch(() => []),
  ]);
  // "needs you today" is the one project whose harvest is due and unreported
  const due = (projects || []).find(p => p.status === 'funded' && !p.harvestReportedAt);
  bind(ctx.root, {
    whoWhere: [me.name, me.location].filter(Boolean).join(' · '),
    'todo': { headline: due ? `${due.title} is ready to harvest` : 'Nothing needs you today' },
    plots: { waiting: videos.length ? `${videos.length} video${videos.length === 1 ? '' : 's'}` : 'None yet' },
  });
  if (!due) {
    // do not leave a call to action pointing at nothing
    const btn = oneByText('Report the harvest', ctx.root);
    btn?.closest('div[style*="background: #016abe"]')?.remove();
  }
});
