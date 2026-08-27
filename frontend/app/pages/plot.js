// One plot: every video filed against it, and the money riding on it.
//
// The page had zero controls -- the back arrow, both buttons and every video
// row were decoration. The season band was worse: "Week 11 of 18", "Ready to
// harvest" and a bar at 61%, none of which exists. There is no Plot entity on
// the server and a video record carries no season length, so the week count had
// nothing behind it and "Ready to harvest" was a static claim printed on every
// plot -- on the same screen that ends "Nobody has visited this field."
import { requireUser, api, load, state } from './_guard.js';
import { bind, repeatRows, showPoster } from '../bind.js';
import { whenShort, dateState, rupees } from '../api.js';
import { appChrome } from '../chrome.js';
import { goes, press } from '../wire.js';

const ctx = requireUser('plot', ['farmer']);

if (ctx) {
  const root = ctx.root;
  appChrome(root, { back: 'plots', user: ctx.user });
  press(root);

  load(root, async () => {
    // A plot is the crop-and-location string a video carries, so that string is
    // the key. plots.js used to link with ?cid=, which this page never read.
    const q = new URLSearchParams(location.search);
    const key = q.get('name');
    const all = await api.videos.mine(ctx.user._id || ctx.user.id);

    // Without ?name= this page has no plot. It used to fall back to `all` and
    // present the farmer's whole library under the heading "4 videos on this
    // plot" -- four different crops, called one field.
    if (!key) {
      // state() keeps the page header, and this header is a DRAWN specimen --
      // "Canal plot / Two acres, potato, sown 12 June" -- so an honest empty
      // state arrived under a plot that does not exist.
      bind(root, { plot: { name: 'A plot', meta: 'None chosen yet',
        stage: '', dates: '', videosLabel: '' } });
      return state(root, 'empty', 'No plot chosen',
        'This screen shows one plot at a time. Pick the one you want from your plots.',
        { label: 'See my plots', go: 'plots' });
    }
    const mine = all.filter(v => (v.crop || v.location) === key);

    if (!mine.length) {
      return state(root, 'empty', 'No videos on this plot',
        'Film it once and this page fills itself.',
        { label: 'Record a video', go: 'record' });
    }

    const first = mine[0];
    const written = mine.filter(v => v.anchored && v.blockHeight).length;
    bind(root, { plot: {
      name: key || first.crop || first.location || 'Plot',
      // The name IS usually the crop, so listing it again under itself read
      // "Lettuce / Sunny Acres · Lettuce".
      meta: [first.location, first.crop].filter(Boolean)
        .filter(v => v !== (key || first.crop)).join(' · ') || 'Where you told us it is',
      // What this page actually knows, instead of a season length nobody stores.
      stage: `${mine.length} video${mine.length === 1 ? '' : 's'} on this plot`,
      dates: `${written} of ${mine.length} date${mine.length === 1 ? '' : 's'} written`,
      videosLabel: 'Videos of this plot',
    } });

    const bar = root.querySelector('[data-progress]');
    if (bar) bar.style.transform = `scaleX(${mine.length ? written / mine.length : 0})`;

    repeatRows(root, '.vid', mine.map(v => {
      const s = dateState(v);
      return { v, when: whenShort(v.uploadTimestamp), short: s.short || s.text, kind: s.kind };
    }), (el, r) => {
      const m = el.querySelector('.m'); if (m) m.textContent = r.when;
      // One real frame in the thumbnail, when the server could cut one.
      showPoster(el.querySelector('.pthumb'), r.v);
      const dur = el.querySelector('.dur');
      // The chip shows a real duration or none. The artboard's 0:41 repeated on
      // every row is one invented number printed three times.
      if (dur) {
        const n = Math.round(Number(r.v.durationSeconds));
        if (Number.isFinite(n) && n > 0) {
          dur.textContent = `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`;
          el.querySelector('[data-mark]')?.remove();
        } else {
          dur.remove();          // the neutral video mark carries the tile alone
        }
      }
      const status = [...el.querySelectorAll('div')].reverse()
        .find(d => !d.children.length && d.textContent.trim());
      if (status) {
        // The short form. Four rows repeating "On our server - date being
        // written, usually by tomorrow" is one sentence said three times too
        // often; the band above the list already carries the count.
        status.textContent = r.short;
        status.style.color = r.kind === 'proved' ? '#006934' : '#4a443d';
      }
      // The green tick is the proved mark. Beside "date being written" it says
      // the opposite of the words next to it, so it only appears when the date
      // really is in a block.
      const tick = el.querySelector('svg[stroke="#006934"]');
      if (tick && r.kind !== 'proved') tick.remove();
    });

    // ---- the money on this plot --------------------------------------
    // Two full-width blue buttons of the same size read as one repeated
    // control, so the board now has one primary. The consequential action is
    // reporting the harvest: nothing is paid to anybody until it happens.
    const projects = (await api.projects.mine(ctx.user._id || ctx.user.id).catch(() => [])) || [];
    const here = projects.find(p => p.title === key) || projects.find(p => !p.harvestReportedAt);
    // .pcard, not .card: the shared laptop shell owns .railcard now and this
    // board's own money card needed a name of its own.
    const money = root.querySelector('.pcard');

    if (!here) {
      // No season is riding on this plot, so there is nothing to report and no
      // raised figure to print. The artboard's 5,00,000 and "Four investors
      // keep 60%" are not this plot's numbers.
      money?.remove();
      [...root.querySelectorAll('.sec')].find(d => d.textContent.trim() === 'Money on this plot')?.remove();
    } else {
      const leafs = [...money.querySelectorAll('div')].filter(d => !d.children.length && d.textContent.trim());
      const raised = leafs.find(d => /^₹/.test(d.textContent.trim()));
      if (raised) raised.textContent = rupees(here.fundedAmount || 0);
      const share = leafs.find(d => /% of the profit/.test(d.textContent));
      if (share) share.textContent = here.investorShare != null
        ? `${here.investorShare}% of the profit` : 'a share of the profit';
      const who = leafs.find(d => /investors keep/.test(d.textContent));
      if (who) who.textContent = 'Investors keep';

      const report = [...root.querySelectorAll('div')]
        .find(d => d.children.length === 0 && d.textContent.trim() === 'Report the harvest')?.parentElement;
      if (here.harvestReportedAt) report?.remove();
      else goes(report, `report-harvest?project=${here._id}`, 'Report the harvest');
    }

    goes([...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === 'Record another one here')?.parentElement,
      'record', 'Record another video here');

    press(root);
  });
}
