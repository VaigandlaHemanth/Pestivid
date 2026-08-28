// Does the back link work, and does it go where it says?
//
// It used to be a bare chevron in the app bar, and the contract this probe held
// was "back lands where you came from" -- the right promise for a control that
// names nothing. The chevron has moved into the content and become a labelled
// link: "My plots", "Money", "Film your field". A link that says My plots and
// lands you on the profile screen because that is where you happened to come
// from is lying, so the contract changed with it:
//
//   the link lands on the page its own word names, from wherever you arrived.
//
// The browser's back button still does "where I came from", on every page, for
// everyone. That is not this control's job.
//
// This also stopped recognising the control at all when it became an <a href>:
// it looked for data-act or data-go, an anchor carries neither, and eleven
// working links reported DEAD. A real link IS the wiring.
const { chromium } = require('playwright');

// what each destination is called, mirroring BACKWORD in chrome.js
const WORD = {
  plots: 'My plots', money: 'Money', home: 'Home', record: 'Film your field',
  messages: 'Chat', invest: 'Browse', market: 'Buy produce', orders: 'My orders',
  portfolio: 'Portfolio', admin: 'Flagged', profile: 'You',
};

(async () => {
  const b = await chromium.launch();
  const login = async (r) => (await fetch('http://127.0.0.1:3001/api/auth/login', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: 'demo.' + r + '@pestivid.sim', password: 'password123' }) })).json();
  const toks = {};
  for (const r of ['farmer', 'investor', 'buyer', 'admin']) toks[r] = await login(r);

  // slug -> [a page a person plausibly arrived FROM, whose account]. Arriving
  // from somewhere is still worth doing: it is the case where a history-based
  // control would quietly pass while going somewhere its label never claimed.
  const from = {
    record: ['plots', 'farmer'], ask: ['home', 'farmer'], 'leaf-check': ['plots', 'farmer'],
    'ask-money': ['money', 'farmer'], sent: ['record', 'farmer'], profile: ['home', 'farmer'],
    plot: ['plots', 'farmer'], money: ['home', 'farmer'], messages: ['home', 'farmer'],
    plots: ['home', 'farmer'], 'report-harvest': ['money', 'farmer'],
    notifications: ['home', 'farmer'],
  };
  let dead = 0, wrong = 0, unnamed = 0;
  const need = await (await import('./_needs.mjs')).needs();

  for (const [slug, [came, role]] of Object.entries(from)) {
    const tok = toks[role];
    const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
    await p.addInitScript(([t, u]) => {
      localStorage.setItem('pv.token', t); localStorage.setItem('pv.user', u);
    }, [tok.token, JSON.stringify(tok.user)]);
    // arrive the way a person does: from another page
    await p.goto('http://127.0.0.1:3001/app/' + came + '.html');
    await p.waitForTimeout(700);
    await p.goto('http://127.0.0.1:3001/app/' + slug + '.html' + (need[slug] || ''),
      { referer: 'http://127.0.0.1:3001/app/' + came + '.html' });
    await p.waitForTimeout(1300);

    const has = await p.evaluate(() => {
      const c = document.querySelector('[data-chrome="back"]');
      if (!c) return null;
      return {
        // an <a href> is wired by being an anchor; anything else needs a handler
        wired: (c.tagName === 'A' && !!c.getAttribute('href'))
          || c.hasAttribute('data-act') || c.hasAttribute('data-go') || !!c.closest('[data-act]'),
        word: (c.querySelector('[data-backword]') || c).textContent.replace(/\s+/g, ' ').trim(),
        inBar: !!c.closest('.appbar'),
      };
    });
    if (!has) { console.log('  ' + slug.padEnd(16) + 'no back link drawn'); await p.close(); continue; }
    if (!has.wired) {
      console.log('  ' + slug.padEnd(16) + 'DEAD — nothing wires it'); dead++; await p.close(); continue;
    }
    if (has.inBar) {
      console.log('  ' + slug.padEnd(16) + 'still in the app bar, left of the wordmark'); wrong++;
    }

    await p.click('[data-chrome="back"]');
    await p.waitForTimeout(1300);
    const landed = p.url().split('/app/')[1]?.split('?')[0]?.replace('.html', '') || p.url();
    // The promise is the WORD, not the history.
    const promised = Object.keys(WORD).find((k) => WORD[k] === has.word);
    if (!promised) {
      console.log('  ' + slug.padEnd(16) + 'says "' + has.word + '", which names no page'); unnamed++;
    } else if (landed !== promised) {
      console.log('  ' + slug.padEnd(16) + 'WRONG  says "' + has.word + '" (' + promised
        + ') but landed on ' + landed); wrong++;
    } else {
      console.log('  ' + slug.padEnd(16) + 'ok     came from ' + came.padEnd(8)
        + ' says "' + has.word + '" -> ' + landed);
    }
    await p.close();
  }
  await b.close();
  console.log('\n  ' + dead + ' dead, ' + wrong + ' not going where they say, '
    + unnamed + ' naming no page');
  process.exit(dead || wrong || unnamed ? 1 : 0);
})();
