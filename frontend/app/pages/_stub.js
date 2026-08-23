// Pages that are drawn but not yet reading anything.
//
// A page showing invented numbers is worse than one that admits it is a mockup,
// so each says so in place rather than letting a farmer read "₹5,00,000" and
// believe it. Delete the call as each screen is wired.
import { wire } from '../wire.js';
import { state } from '../bind.js';

export function notWired(slug, why) {
  const root = wire(slug);
  if (!root) return null;
  const holder = document.createElement('div');
  root.prepend(holder);
  state(holder, 'waiting', 'Nothing on this screen is real yet',
    why || 'Every figure here is placeholder content from the design. It is not your data.');
  return root;
}
