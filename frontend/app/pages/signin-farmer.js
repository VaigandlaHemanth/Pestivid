// Farmer sign-in: a phone number and a four-number code, nothing else.
import { api, session } from '../api.js';
import { wire } from '../wire.js';
import { state } from '../bind.js';
wire('signin-farmer');
const root = document.querySelector('body > div');
// The account is keyed on email server-side; a phone-only farmer signs in with
// the address the server made for them. Until that route exists, say so rather
// than pretending the field works.
state(root, 'waiting', 'Phone sign-in is not connected yet',
  'The server signs people in by email address. A phone-number route has to exist before this screen can do anything, and drawing a working keypad over a missing route would be a lie.');
