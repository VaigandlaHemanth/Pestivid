// Every page behind a sign-in shares the same first move.
import { api, session } from '../api.js';
import { wire } from '../wire.js';
import { load, state } from '../bind.js';

export function requireUser(slug, roles) {
  const root = wire(slug);
  if (!session.token) { location.href = './signin.html'; return null; }
  const user = session.user;
  if (roles && user && !roles.includes(user.role)) {
    state(root, 'failed', 'Not your screen', `This page is for a ${roles.join(' or ')}. You are signed in as a ${user.role}.`);
    return null;
  }
  return { root, user };
}
export { api, session, load, state };
