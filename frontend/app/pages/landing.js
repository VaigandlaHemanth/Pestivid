// The public page. Two figures on it are real and everything else is copy.
//
// GET /funding-requests is one of the few routes that needs no token, which is
// what makes a live number possible here at all.
import { api } from '../api.js';
import { wire } from '../wire.js';
import { bind } from '../bind.js';
import { rupees } from '../api.js';

const root = wire('landing');
if (root) {
  (async () => {
    try {
      const projects = await api.projects.open();
      const raised = (projects || []).reduce((a, p) => a + (p.fundedAmount || 0), 0);
      const asked = (projects || []).reduce((a, p) => a + (p.amount || 0), 0);
      // lakhs, because that is how the figure is spoken about here
      const lakh = (n) => n >= 100000 ? `₹${(n / 100000).toFixed(1)}L` : rupees(n);
      bind(root, { funded: lakh(raised) });
      const bar = root.querySelector('div[style*="background: #01579b"][style*="height: 100%"]')
        || root.querySelector('div[style*="height: 8px"] > div');
      if (bar) {
        bar.style.transformOrigin = 'left';
        bar.style.width = asked ? `${Math.min(100, Math.round(100 * raised / asked))}%` : '0%';
      }
    } catch {
      // A marketing page that cannot reach the API should still read fine, so
      // the figure goes blank rather than the page shouting about it.
      bind(root, { funded: '—' });
    }
  })();

  // the two calls to action are the only controls here
  for (const [label, to] of [['I farm — get funded', 'setup'],
                             ['I want to invest', 'signin'],
                             ['Sign in', 'signin']]) {
    const el = [...root.querySelectorAll('div')]
      .find(d => d.children.length === 0 && d.textContent.trim() === label);
    if (el) { el.setAttribute('data-act', ''); el.dataset.go = to; }
  }
}

// The hero carries a "record of evidence" prop with a filename, a hash and a
// block number on it. On a public page those are invented figures dressed as
// proof, which is the one thing this product must not do, so deClaimProps in
// wire.js swaps the whole run for a sentence naming what a record holds. The
// board keeps the filled-in example because an artboard is a drawing; the live
// page does not, because a live page is a claim.
