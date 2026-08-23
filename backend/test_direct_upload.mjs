// A real round trip through the free-tier upload path.
//
//   ask for a one-use URL -> post the file straight to storage ->
//   tell the server the id -> the server fetches it back and hashes it
//
// The last step is the one worth testing: the hash the server records must be
// the hash of the bytes in storage, not anything the client said. So this
// computes the digest locally and requires the server to arrive at the same one
// without ever being told it.
//
//   node backend/test_direct_upload.mjs        (needs the dev server on 3001)
import { createHash, randomBytes } from 'node:crypto';

const API = process.env.PV_API || 'http://127.0.0.1:3001/api';
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  cond ? pass++ : fail++;
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${extra ? '  — ' + extra : ''}`);
};

const call = async (path, body, token) => {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify(body),
  });
  return { status: r.status, body: await r.json().catch(() => null) };
};

const stamp = Date.now();
const email = `upl${stamp}@e2e.test`;
await call('/auth/register', { name: `Upload ${stamp}`, email, password: 'Password!234', role: 'farmer' });
const login = await call('/auth/login', { email, password: 'Password!234' });
const token = login.body?.token;
ok('a farmer can sign in', Boolean(token));
if (!token) process.exit(1);

// A small unique payload with a video mime type. Unique so it can never collide
// with an existing hash, small so the test costs almost nothing of a 1 GB free
// allowance.
const bytes = Buffer.concat([Buffer.from('\x1aE\xdf\xa3'), randomBytes(2048)]);
const localDigest = createHash('sha256').update(bytes).digest('hex');

const ticket = await call('/videos/upload-url', { crop: 'testcrop' }, token);
ok('the server issues a one-use upload URL', ticket.status === 200 && /^https:\/\//.test(ticket.body?.url || ''),
   ticket.status !== 200 ? JSON.stringify(ticket.body).slice(0, 90) : '');
if (ticket.status !== 200) process.exit(1);
ok('the URL expires', Number(ticket.body.expiresSeconds) > 0 && Number(ticket.body.expiresSeconds) <= 600);
ok('the URL is not our own API', !ticket.body.url.includes('/api/'));
ok('the ticket says who hashes it', ticket.body.hashedBy === 'server-after-upload');

const form = new FormData();
form.append('file', new Blob([bytes], { type: 'video/webm' }), 'probe.webm');
form.append('network', 'public');
const up = await fetch(ticket.body.url, { method: 'POST', body: form });
const upBody = await up.json().catch(() => null);
const cid = upBody?.data?.cid || upBody?.cid || upBody?.IpfsHash;
ok('the file goes straight to storage, not through us', up.ok && Boolean(cid),
   up.ok ? '' : `${up.status} ${JSON.stringify(upBody).slice(0, 90)}`);
if (!cid) process.exit(1);

// The client never sends a hash. If the server records the right one it can
// only have got it by reading the object itself.
const done = await call('/videos/confirm-upload',
  { cid, crop: 'testcrop', location: 'e2e', purpose: 'agristream' }, token);
ok('the server records the video', done.status === 201, done.status !== 201 ? JSON.stringify(done.body).slice(0, 110) : '');
ok('it hashed the bytes itself', done.body?.hashComputedBy === 'server');
ok('and got the same digest we computed locally', done.body?.videoFileHash === localDigest,
   done.body?.videoFileHash ? `server ${String(done.body.videoFileHash).slice(0, 16)}… local ${localDigest.slice(0, 16)}…` : '');
ok('it reports the size it actually read', done.body?.bytes === bytes.length,
   `read ${done.body?.bytes}, sent ${bytes.length}`);

const again = await call('/videos/confirm-upload',
  { cid, crop: 'testcrop', location: 'e2e', purpose: 'agristream' }, token);
ok('the same object cannot be claimed twice', again.status === 409);

console.log(`\n  ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
