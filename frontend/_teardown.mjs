// Takes back what the flow test left behind.
//
// e2e-flows signs up a real account on every run, on purpose: the thing being
// tested is that a NEW farmer lands on the empty home, and you only get to be
// new once. What it never did was clean up, and the dev database persists — so
// after a dozen runs the demo investor's chat list was nine "Testcase Farmer
// 1787845685112" rows against three real ones, and twelve of the twenty-one
// videos on the server belonged to a test. The demo was mostly test residue.
//
// This removes those accounts and the records that hang off them. It is
// deliberately narrow:
//
//   - it refuses to run against anything but the local dev instance, so it can
//     never be pointed at a real database by an environment variable;
//   - it matches ONLY /^Testcase Farmer \d+$/ — the exact name e2e-flows
//     generates. A real account cannot be named that by accident, and nothing
//     seeded is;
//   - it deletes by owner id, never by a pattern on the records themselves.
//
// Nothing here touches the four demo accounts or anything they own. If a sweep
// ever removes too much, `node backend/dev-server.js --fresh` rebuilds the whole
// seed from scratch.
import mongoose from '../backend/node_modules/mongoose/index.js';
import { pathToFileURL } from 'node:url';

const URI = 'mongodb://127.0.0.1:27017/pestivid';
const MADE_BY_THE_TEST = /^Testcase Farmer \d+$/;

// Which field in each collection points back at the account that owns the row.
// Listed rather than guessed, so a collection nobody thought about is left alone
// instead of being swept by a lucky field name.
const OWNED_BY = [
  ['videos', ['farmerWallet', 'farmer', 'uploadedBy']],
  ['conversations', ['participants', 'farmer', 'investor', 'buyer']],
  ['messages', ['sender', 'recipient', 'conversation']],
  ['notifications', ['user', 'recipient', 'farmer']],
  ['fundingrequests', ['farmer', 'farmerWallet']],
  ['investments', ['farmer', 'investor']],
  ['orders', ['farmer', 'buyer']],
  ['listings', ['farmer', 'farmerWallet']],
];

export async function sweep({ quiet = false } = {}) {
  const say = (s) => { if (!quiet) console.log(s); };
  await mongoose.connect(URI);
  const db = mongoose.connection.db;
  try {
    const users = db.collection('users');
    const doomed = await users.find({ name: MADE_BY_THE_TEST })
      .project({ _id: 1, name: 1 }).toArray();
    if (!doomed.length) { say('  nothing left behind by the flow test'); return 0; }

    const ids = doomed.map(u => u._id);
    // Conversations first, so their messages can be found by conversation id
    // before the conversation rows go.
    const convIds = (await db.collection('conversations')
      .find({ $or: [{ participants: { $in: ids } }, { farmer: { $in: ids } },
                    { investor: { $in: ids } }, { buyer: { $in: ids } }] })
      .project({ _id: 1 }).toArray()).map(c => c._id);

    let removed = 0;
    for (const [name, fields] of OWNED_BY) {
      if (!(await db.listCollections({ name }).hasNext())) continue;
      const or = fields.map(f => ({ [f]: { $in: f === 'conversation' ? convIds : ids } }));
      const r = await db.collection(name).deleteMany({ $or: or });
      if (r.deletedCount) { say(`  ${name.padEnd(16)} ${r.deletedCount} removed`); removed += r.deletedCount; }
    }
    const u = await users.deleteMany({ _id: { $in: ids }, name: MADE_BY_THE_TEST });
    say(`  ${'users'.padEnd(16)} ${u.deletedCount} removed`
      + `  (${doomed.map(d => d.name.replace('Testcase Farmer ', '#')).join(' ')})`);
    return removed + u.deletedCount;
  } finally {
    await mongoose.disconnect();
  }
}

/* And the same idea for a probe that SENDS something.
 *
 * _chatsend types a real message into a real conversation, so every run added a
 * line to the demo transcript -- five copies of "The far end had water on
 * Tuesday" by the time anybody looked.
 *
 * This deletes by id difference, never by matching the text. Matching text would
 * mean deciding that an English sentence in a chat log was written by a machine,
 * and some of the residue already on this server is indistinguishable from
 * somebody typing in the browser. A probe may only take back the exact rows it
 * can prove it added.
 */
export async function messageIds() {
  await mongoose.connect(URI);
  try {
    const rows = await mongoose.connection.db.collection('messages')
      .find({}).project({ _id: 1 }).toArray();
    return new Set(rows.map(r => String(r._id)));
  } finally { await mongoose.disconnect(); }
}

export async function removeMessagesNotIn(before, { quiet = false } = {}) {
  await mongoose.connect(URI);
  try {
    const col = mongoose.connection.db.collection('messages');
    const rows = await col.find({}).project({ _id: 1, text: 1, conversationId: 1 }).toArray();
    const mine = rows.filter(r => !before.has(String(r._id)));
    if (!mine.length) return 0;
    await col.deleteMany({ _id: { $in: mine.map(r => r._id) } });
    // The conversation still quotes the deleted message as its last one. A
    // probe's sentence sat in the demo's chat list for a day this way. Each
    // touched conversation takes its last line back from the messages that
    // remain, in the route's own 50-character form.
    const convs = [...new Set(mine.map(r => String(r.conversationId)))];
    for (const id of convs) {
      const cid = new mongoose.Types.ObjectId(id);
      const last = (await col.find({ conversationId: cid }).sort({ timestamp: -1 }).limit(1).toArray())[0];
      if (!last) continue;
      const text = String(last.text);
      await mongoose.connection.db.collection('conversations').updateOne({ _id: cid }, { $set: {
        lastMessageSnippet: text.substring(0, 50) + (text.length > 50 ? '…' : ''),
        lastMessageTimestamp: last.timestamp,
      } });
    }
    if (!quiet) {
      for (const r of mine) {
        console.log(`  took back: ${JSON.stringify(String(r.text).slice(0, 52))}`);
      }
    }
    return mine.length;
  } finally { await mongoose.disconnect(); }
}

// Run directly to sweep whatever earlier runs left behind. pathToFileURL, not a
// hand-built file:// string: a Windows path becomes file:///C:/… with three
// slashes, and the hand-built two-slash version silently never matched, so this
// did nothing at all when run on its own.
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const n = await sweep();
  console.log(`\n  ${n} record(s) removed — the demo is back to its seeded state`);
}
