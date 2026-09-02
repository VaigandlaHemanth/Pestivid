// EXTRA DEMO DATA, ADDED TO A DATABASE THAT ALREADY HAS SOME.
//
// seed.js runs once, into an empty database, and wipes what is there first.
// That is the wrong tool for a database you are living in: by the time the
// screens were being looked at page by page, four of its five market lots had
// been BOUGHT by test runs and the market read as empty. Re-seeding would have
// thrown away every video, purchase and notice with them.
//
// So this script only ever inserts, and every insert is guarded: listings by
// their txHash, conversations by their pair of participants, notices by their
// exact sentence. Run it as many times as you like -- the second run says
// "already there" and writes nothing.
//
//   node backend/demo-data.js            (against a running dev-server)
//
// It needs the dev server up, because that is what holds the database open.
const mongoose = require('mongoose');

const User = require('./models/User');
const Video = require('./models/Video');
const Listing = require('./models/Listing');
const Conversation = require('./models/Conversation');
const Message = require('./models/Message');
const Notification = require('./models/Notification');
const FundingRequest = require('./models/FundingRequest');

const URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/pestivid';
const days = (n) => new Date(Date.now() - n * 24 * 60 * 60 * 1000);

let added = 0;
let kept = 0;
const say = (what, did) => {
  if (did) { added += 1; console.log('  added   ' + what); }
  else { kept += 1; console.log('  already ' + what); }
};

async function main() {
  await mongoose.connect(URI);

  const by = {};
  for (const u of await User.find({ email: /@(pestivid\.sim|example\.com)$/ }, 'name email role')) {
    by[u.email] = u;
  }
  const farmer = by['demo.farmer@pestivid.sim'];
  const buyer = by['demo.buyer@pestivid.sim'];
  const investor = by['demo.investor@pestivid.sim'];
  const alice = by['alice@example.com'];
  const bob = by['bob@example.com'];
  const charlie = by['charlie@example.com'];
  if (!farmer || !buyer || !investor || !alice || !bob || !charlie) {
    throw new Error('the demo accounts are not in this database -- is the dev server running?');
  }

  // ---- MARKET LOTS ---------------------------------------------------------
  // Every lot points at a video that is really in the database, so the evidence
  // link on the market row opens the same clip the farmer filmed. A lot with an
  // invented CID would be a row that proves nothing, on the one screen whose
  // whole job is proof.
  console.log('market lots');
  const vids = await Video.find({}, 'crop location purpose cid storageType videoFileHash uploader').lean();
  const video = (crop, location) => vids.find((v) => v.crop === crop && v.location === location);

  const lots = [
    { crop: 'Lettuce', location: 'Sunny Acres', who: farmer, min: 26000, max: 34000, ago: 0.3 },
    { crop: 'Bell Peppers', location: 'Field 3', who: alice, min: 74000, max: 88000, ago: 0.9 },
    { crop: 'Green Beans', location: 'Canal plot', who: farmer, min: 31000, max: 39000, ago: 1.4 },
    { crop: 'Wheat', location: 'North Fields', who: farmer, min: 96000, max: 118000, ago: 2.1 },
    { crop: 'Corn', location: 'Northwest Plot', who: alice, min: 58000, max: 71000, ago: 3.2 },
    { crop: 'Strawberries', location: 'Greenhouse Hydroponics', who: farmer, min: 82000, max: 97000, ago: 4.5 },
  ];

  for (const lot of lots) {
    const v = video(lot.crop, lot.location);
    if (!v) { console.log('  skipped ' + lot.crop + ' -- no video in this database'); continue; }
    const txHash = 'sim_extra_' + lot.crop.toLowerCase().replace(/[^a-z]/g, '') + '_' + Math.abs(lot.min);
    const have = await Listing.findOne({ txHash });
    if (have) { say(lot.crop + ' at ' + lot.location, false); continue; }
    await Listing.create({
      farmerWallet: lot.who._id,
      crop: v.crop, location: v.location,
      pesticide: v.pesticide, pesticideCompany: v.pesticideCompany,
      cid: v.cid, storageType: v.storageType, videoFileHash: v.videoFileHash,
      minPrice: lot.min, maxPrice: lot.max,
      status: 'active', createdAt: days(lot.ago),
      txHash, notificationSent: true,
    });
    say(lot.crop + ' at ' + lot.location + ', ' + lot.min + '-' + lot.max, true);
  }

  // ---- CONVERSATIONS -------------------------------------------------------
  // Written as real runs, not one line each: two or three messages in a row
  // from the same person is what a chat actually looks like, and it is the only
  // way to see whether the grouping -- one clock per run, the tail on the last
  // bubble -- comes out right. The last inbound message of each thread is left
  // unread, so the rail's unread mark has something to mark.
  console.log('conversations');
  const threads = [
    {
      pair: [farmer, charlie],
      lines: [
        [charlie, 'Namaste. I saw the wheat season on the funding page.', 2.4, true],
        [charlie, 'How many acres is it, and when do you sow?', 2.4, true],
        [farmer, 'Four acres. Sowing starts next week once the field dries.', 2.3, true],
        [farmer, 'I will film the field the same morning so you can see it.', 2.3, true],
        [charlie, 'That works for me. I will put in fifty thousand.', 0.4, false],
        [charlie, 'Send me the clip when it is up.', 0.35, false],
      ],
    },
    {
      pair: [buyer, alice],
      lines: [
        [buyer, 'Hello Alice. Asking about the bell peppers at Field 3.', 1.8, true],
        [alice, 'Yes, they are ready. Picked two days ago.', 1.7, true],
        [buyer, 'What was sprayed on them?', 1.6, true],
        [alice, 'Neem oil only, nothing else. It is on the video.', 1.55, true],
        [buyer, 'Good. I will take the whole lot at your asking price.', 0.2, false],
      ],
    },
    {
      pair: [investor, alice],
      lines: [
        [investor, 'I am looking at the corn pilot. Is the land your own?', 3.1, true],
        [alice, 'Own land, seven acres, papers with the panchayat.', 3.0, true],
        [investor, 'And the return you wrote -- is that after the mandi cut?', 2.9, true],
        [alice, 'After. What you see is what reaches you.', 2.85, true],
        [investor, 'Then I am in for a quarter of it.', 0.6, false],
      ],
    },
    {
      pair: [farmer, bob],
      lines: [
        [bob, 'The lettuce lot -- can you hold it two days?', 1.1, true],
        [farmer, 'I can hold it till Thursday morning.', 1.05, true],
        [farmer, 'After that it has to move, it will not keep.', 1.05, true],
        [bob, 'Understood. Thursday then.', 0.9, true],
        [bob, 'One more thing, do you deliver to the mandi or do I collect?', 0.1, false],
      ],
    },
  ];

  for (const t of threads) {
    const ids = t.pair.map((u) => u._id);
    const have = await Conversation.findOne({ participants: { $all: ids, $size: 2 } });
    const names = t.pair.map((u) => u.name.split(' ')[0]).join(' and ');
    if (have) { say('a thread between ' + names, false); continue; }
    const last = t.lines[t.lines.length - 1];
    const conv = await Conversation.create({
      participants: ids,
      lastMessageSnippet: last[1],
      lastMessageTimestamp: days(last[2]),
      createdAt: days(t.lines[0][2]),
    });
    await Message.insertMany(t.lines.map(([who, text, ago, read]) => ({
      conversationId: conv._id,
      sender: who._id,
      receiver: t.pair.find((u) => !u._id.equals(who._id))._id,
      text, timestamp: days(ago), read,
    })));
    say('a thread between ' + names + ', ' + t.lines.length + ' messages', true);
  }

  // ---- UNREAD NOTICES ------------------------------------------------------
  // The bell counts unread notices and removes itself when there are none, so
  // with everything read the badge could never be looked at at all.
  //
  // Each one carries the thing it is ABOUT. The notices page draws a chevron
  // only where a row leads somewhere, so a notice with no itemId is a row that
  // looks dead next to its neighbours -- and it would be, rightly, because
  // nothing was attached to it. These point at real rows.
  console.log('notices');
  const lastMessageTo = async (a, b) => {
    const conv = await Conversation.findOne({ participants: { $all: [a._id, b._id], $size: 2 } });
    if (!conv) return null;
    return Message.findOne({ conversationId: conv._id, receiver: a._id }).sort({ timestamp: -1 });
  };
  const lotOf = (crop, who) => Listing.findOne({ crop, farmerWallet: who._id, status: 'active' });
  const anyFunding = (title) => FundingRequest.findOne({ title: new RegExp(title, 'i') });

  const notices = [
    { to: farmer, type: 'purchase', text: 'Your Lettuce lot was bought by Bob.', ago: 0.08,
      on: async () => lotOf('Lettuce', farmer), kind: 'Listing' },
    { to: farmer, type: 'message', text: 'Bob sent you a message about the lettuce lot.', ago: 0.1,
      on: async () => lastMessageTo(farmer, bob), kind: 'Message' },
    { to: farmer, type: 'investment', text: 'Charlie put Rs 50,000 into your wheat season.', ago: 0.4,
      on: async () => anyFunding('wheat'), kind: 'FundingRequest' },
    { to: buyer, type: 'listing', text: 'A new Green Beans lot is on the market.', ago: 0.15,
      on: async () => lotOf('Green Beans', farmer), kind: 'Listing' },
    { to: buyer, type: 'message', text: 'Alice replied about the bell peppers.', ago: 0.2,
      on: async () => lastMessageTo(buyer, alice), kind: 'Message' },
    { to: investor, type: 'update', text: 'The wheat season posted an update: sowing starts next week.', ago: 0.3,
      on: async () => anyFunding('wheat'), kind: 'FundingRequest' },
    { to: investor, type: 'message', text: 'Alice answered your question about the corn pilot.', ago: 0.6,
      on: async () => lastMessageTo(investor, alice), kind: 'Message' },
  ];

  for (const n of notices) {
    const target = await n.on();
    const fields = {
      recipient: n.to._id, type: n.type, message: n.text,
      timestamp: days(n.ago), read: false,
      itemId: target ? target._id : undefined,
      itemType: target ? n.kind : undefined,
    };
    const have = await Notification.findOne({ recipient: n.to._id, message: n.text });
    if (have) {
      // A run before this one wrote it without anything attached. Give it the
      // row it is about rather than leaving a notice that leads nowhere.
      // These seven sentences belong to this script, and their whole job is to
      // be UNREAD -- the bell removes itself at zero, so a demo where they have
      // all been clicked is a demo where the badge cannot be looked at. Running
      // this again puts them back.
      const fix = (!have.itemId && target) || have.read;
      if (fix) {
        if (target) { have.itemId = target._id; have.itemType = n.kind; }
        have.read = false;
        await have.save();
        say('put back as unread: "' + n.text.slice(0, 34) + '"', true);
      } else say('a notice for ' + n.to.name.split(' ')[1], false);
      continue;
    }
    await Notification.create(fields);
    say('unread ' + (target ? '' : 'dead-end ') + 'notice for '
        + n.to.name.split(' ')[1] + ': ' + n.text, true);
  }

  console.log('');
  console.log('  ' + added + ' added, ' + kept + ' already there. Nothing was removed.');
  await mongoose.disconnect();
}

main().catch((e) => { console.error(e.message); process.exit(1); });
