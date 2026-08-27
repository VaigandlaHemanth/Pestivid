// Every page behind a sign-in shares the same first move.
import { api, session } from '../api.js';
import { wire } from '../wire.js';
import { load, state } from '../bind.js';

export function requireUser(slug, roles) {
  const root = wire(slug);
  if (!session.token) { location.href = './signin.html'; return null; }
  const user = session.user;
  if (roles && user && !roles.includes(user.role)) {
    // "a admin", "a investor". The article depends on the word that follows it,
    // and three of the four roles here begin with a vowel.
    const an = (w) => (/^[aeiou]/i.test(w) ? 'an ' : 'a ') + w;
    state(root, 'failed', 'Not your screen',
      `This page is for ${roles.map(an).join(' or ')}. You are signed in as ${an(user.role)}.`);
    return null;
  }
  return { root, user };
}
export { api, session, load, state };
