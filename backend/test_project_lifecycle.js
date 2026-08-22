/**
 * Project-lifecycle and notification tests.
 *
 * Every case here is a bug that was live, and each one costs a real person
 * something:
 *
 *   1. A farmer could PUT {status:'cancelled'} on a FUNDED project, stranding the
 *      investors' principal with no refund path, then re-raise on the same video.
 *      The DELETE route had always refused this; PUT never got the guard.
 *   2. The remaining-goal ceiling compared raw doubles against a goal it
 *      advertised via toFixed(2), so 3333.33 + 6666.67 = 10000.000000000002 was
 *      refused -- the exact figure the API told the investor to send. Projects
 *      could not be funded to completion.
 *   3. Marking a GLOBAL notification read set a field shared by every user, so
 *      one person clearing a broadcast cleared it platform-wide.
 *   4. Dismissing a global added the user to dismissedBy, but the list query
 *      never consulted it, so the item came straight back.
 *   5. Registration accepted memberSince from the request body, letting a new
 *      account claim years of tenure -- exactly the signal investors judge on.
 *
 * The real routers are mounted with a stubbed auth middleware, so these exercise
 * the shipped handlers rather than a reimplementation of them.
 *
 *     node test_project_lifecycle.js
 */

const assert = require('assert');
const express = require('express');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

let pass = 0;
let fail = 0;

async function t(name, fn) {
    try {
        await fn();
        console.log(`  PASS  ${name}`);
        pass++;
    } catch (e) {
        console.log(`  FAIL  ${name}\n          ${e.message}`);
        fail++;
    }
}

(async () => {
    const mem = await MongoMemoryServer.create();
    await mongoose.connect(mem.getUri(), { dbName: 'lifecycle_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });

    const User = mongoose.model('User');
    const FundingRequest = mongoose.model('FundingRequest');
    const Notification = mongoose.model('Notification');
    const Video = mongoose.model('Video');

    const farmer = await User.create({
        name: 'Farmer One', email: 'fm@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const investor = await User.create({
        name: 'Investor One', email: 'iv@t.local', password: 'x'.repeat(20), role: 'investor',
    });
    const other = await User.create({
        name: 'Other Investor', email: 'ov@t.local', password: 'x'.repeat(20), role: 'investor',
    });

    let cidN = 0;
    const mkProject = (extra = {}) => FundingRequest.create({
        title: 'Potato season', farmerWallet: farmer._id, crop: 'Potato',
        acres: 2, amount: 1000, method: 'organic', cid: `bafy-life-${cidN++}`,
        description: 'd', timeline: 4, roi: 20, investorShare: 60,
        status: 'pending', fundedAmount: 0, ...extra,
    });

    // Mount the real routers behind a swappable identity.
    let actingAs = farmer;
    const authModule = require('./routes/auth');
    authModule.authenticateToken = (req, res, next) => { req.user = actingAs; next(); };
    for (const r of ['./routes/fundingRequests', './routes/investments', './routes/notifications']) {
        delete require.cache[require.resolve(r)];
    }
    const app = express();
    app.use(express.json());
    app.use('/api/funding-requests', require('./routes/fundingRequests'));
    app.use('/api/investments', require('./routes/investments'));
    app.use('/api/notifications', require('./routes/notifications'));
    const srv = app.listen(4877);

    const call = async (method, path, body) => {
        const r = await fetch(`http://127.0.0.1:4877${path}`, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        return { status: r.status, body: await r.json().catch(() => null) };
    };

    // ── 1. cancelling a funded project ──────────────────────────────────────
    await t('THE THEFT: farmer cannot cancel a project that holds investor money', async () => {
        const p = await mkProject({
            status: 'funded',
            fundedAmount: 1000,
            investors: [{ investorId: investor._id, amount: 1000, txHash: 'sim_x', investmentDate: new Date() }],
        });
        actingAs = farmer;
        const { status, body } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'cancelled' });
        assert.strictEqual(status, 409, `expected 409, got ${status}`);
        assert.strictEqual(body.code, 'cancel_after_funding');
        const after = await FundingRequest.findById(p._id).lean();
        assert.strictEqual(after.status, 'funded', 'project was cancelled anyway');
    });

    await t('a partially funded project cannot be cancelled either', async () => {
        const p = await mkProject({
            status: 'partially_funded',
            fundedAmount: 250,
            investors: [{ investorId: investor._id, amount: 250, txHash: 'sim_y', investmentDate: new Date() }],
        });
        actingAs = farmer;
        const { status, body } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'cancelled' });
        assert.strictEqual(status, 409, `expected 409, got ${status}`);
        assert.strictEqual(body.code, 'cancel_after_funding');
    });

    await t('a pending project nobody funded CAN still be cancelled', async () => {
        const p = await mkProject({ status: 'pending', fundedAmount: 0 });
        actingAs = farmer;
        const { status } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'cancelled' });
        assert.strictEqual(status, 200, `a legitimate cancel was blocked (${status})`);
        const after = await FundingRequest.findById(p._id).lean();
        assert.strictEqual(after.status, 'cancelled');
    });

    await t('farmer cannot declare a project funded by hand', async () => {
        const p = await mkProject({ status: 'pending', fundedAmount: 0 });
        actingAs = farmer;
        const { status, body } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'funded' });
        assert.strictEqual(status, 409, `expected 409, got ${status}`);
        assert.strictEqual(body.code, 'status_not_settable');
    });

    await t('farmer cannot skip harvest reporting by declaring completed', async () => {
        const p = await mkProject({ status: 'funded', fundedAmount: 1000 });
        actingAs = farmer;
        const { status, body } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'completed' });
        assert.strictEqual(status, 409, `expected 409, got ${status}`);
        assert.strictEqual(body.code, 'complete_via_settlement');
    });

    await t('a stranger still cannot touch someone else\'s project', async () => {
        const p = await mkProject({ status: 'pending' });
        actingAs = investor;
        const { status } = await call('PUT', `/api/funding-requests/${p._id}`, { status: 'cancelled' });
        assert.ok(status === 403 || status === 404, `expected 403/404, got ${status}`);
    });

    // ── 2. the float deadlock ───────────────────────────────────────────────
    //
    // The drift only appears once fundedAmount has been ACCUMULATED, because
    // each $inc adds a double. 215.58 + 9.99 + 142.24 lands on
    // 367.81000000000006, the API then advertises (1000 - that).toFixed(2) =
    // "632.19 still available", and 367.81000000000006 + 632.19 evaluates to
    // 1000.0000000000001 -- over the goal, so the exact quoted figure was
    // refused and the last rupee could never be raised.
    //
    // The fix has two halves and both are tested: $round stops the drift being
    // stored in the first place, and the 0.005 tolerance rescues records that
    // already hold a drifted value (every project funded before this fix).
    const DRIFTED = 367.81000000000006;      // 215.58 + 9.99 + 142.24
    await t('THE DEADLOCK: the exact advertised remainder is accepted', async () => {
        assert.ok(DRIFTED + 632.19 > 1000,
            'this JS runtime does not reproduce the float error; test is moot');
        const p = await mkProject({ status: 'partially_funded', fundedAmount: DRIFTED });
        actingAs = investor;
        const { status, body } = await call('POST', '/api/investments', {
            projectId: String(p._id), amount: 632.19,
        });
        assert.strictEqual(status, 201,
            `exact remainder refused: ${status} ${JSON.stringify(body)}`);
        const after = await FundingRequest.findById(p._id).lean();
        assert.strictEqual(after.status, 'funded', `status stuck at ${after.status}`);
        assert.strictEqual(after.fundedAmount, 1000,
            `fundedAmount stored as ${after.fundedAmount}, not rounded to the goal`);
    });

    await t('drift never accumulates in the first place', async () => {
        const p = await mkProject({ status: 'pending', fundedAmount: 0 });
        actingAs = investor;
        for (const amt of [215.58, 9.99, 142.24]) {
            const r = await call('POST', '/api/investments', {
                projectId: String(p._id), amount: amt,
            });
            assert.strictEqual(r.status, 201, `investment of ${amt} failed (${r.status})`);
        }
        const after = await FundingRequest.findById(p._id).lean();
        assert.strictEqual(after.fundedAmount, 367.81,
            `stored ${after.fundedAmount} instead of 367.81 -- drift is accumulating`);
    });

    await t('over-funding by one paisa is still refused', async () => {
        const p = await mkProject({ status: 'partially_funded', fundedAmount: DRIFTED });
        actingAs = investor;
        const { status, body } = await call('POST', '/api/investments', {
            projectId: String(p._id), amount: 632.20,
        });
        assert.strictEqual(status, 409, `over-funding was allowed (${status})`);
        assert.strictEqual(body.code, 'exceeds_remaining');
    });

    await t('the quoted remaining figure is one that would be accepted', async () => {
        const p = await mkProject({ status: 'partially_funded', fundedAmount: DRIFTED });
        actingAs = investor;
        const rej = await call('POST', '/api/investments', {
            projectId: String(p._id), amount: 999,
        });
        assert.strictEqual(rej.status, 409);
        const quoted = rej.body.remaining;
        assert.strictEqual(quoted, 632.19, `quoted ${quoted}`);
        const ok = await call('POST', '/api/investments', {
            projectId: String(p._id), amount: quoted,
        });
        assert.strictEqual(ok.status, 201,
            `the API quoted ${quoted} as available then refused it (${ok.status})`);
    });

    // ── 3. the farmer really is notified ────────────────────────────────────
    await t('investing notifies the farmer', async () => {
        const p = await mkProject({ status: 'pending', fundedAmount: 0 });
        await Notification.deleteMany({});
        actingAs = investor;
        const { status } = await call('POST', '/api/investments', {
            projectId: String(p._id), amount: 100,
        });
        assert.strictEqual(status, 201);
        const notes = await Notification.find({ recipient: farmer._id }).lean();
        assert.strictEqual(notes.length, 1,
            `farmer got ${notes.length} notifications, expected 1`);
        assert.strictEqual(notes[0].type, 'investment');
        assert.ok(/100/.test(notes[0].message), `message lacks the amount: ${notes[0].message}`);
    });

    // ── 4. per-user state on a shared global notification ───────────────────
    await t('THE BROADCAST BUG: marking a global read affects only that user', async () => {
        await Notification.deleteMany({});
        const g = await Notification.create({
            global: true, type: 'info', message: 'Platform announcement',
        });

        actingAs = investor;
        const r = await call('PUT', `/api/notifications/${g._id}/read`);
        assert.strictEqual(r.status, 200, `mark-read failed: ${r.status}`);

        // the reader sees it read
        const mine = await call('GET', `/api/notifications/user/${investor._id}`);
        const forMe = mine.body.find((n) => n._id === String(g._id));
        assert.ok(forMe, 'the notification vanished for the reader');
        assert.strictEqual(forMe.read, true, 'not marked read for the reader');

        // everyone else still sees it unread
        actingAs = other;
        const theirs = await call('GET', `/api/notifications/user/${other._id}`);
        const forThem = theirs.body.find((n) => n._id === String(g._id));
        assert.ok(forThem, 'the broadcast disappeared for another user');
        assert.strictEqual(forThem.read, false,
            'one user reading a broadcast marked it read for everyone');

        // and the shared scalar was never touched
        const raw = await Notification.findById(g._id).lean();
        assert.strictEqual(raw.read, false, 'the shared read flag was written');
    });

    await t('?read=false excludes a global this user has read', async () => {
        await Notification.deleteMany({});
        const g = await Notification.create({
            global: true, type: 'info', message: 'Another announcement',
        });
        actingAs = investor;
        await call('PUT', `/api/notifications/${g._id}/read`);
        const unread = await call('GET', `/api/notifications/user/${investor._id}?read=false`);
        assert.ok(!unread.body.some((n) => n._id === String(g._id)),
            'a global this user read still counted as unread');

        actingAs = other;
        const theirUnread = await call('GET', `/api/notifications/user/${other._id}?read=false`);
        assert.ok(theirUnread.body.some((n) => n._id === String(g._id)),
            'another user lost the unread broadcast');
    });

    await t('dismissing a global does not bring it back on the next poll', async () => {
        await Notification.deleteMany({});
        const g = await Notification.create({
            global: true, type: 'info', message: 'Dismissable announcement',
        });
        actingAs = investor;
        const del = await call('DELETE', `/api/notifications/${g._id}`);
        assert.strictEqual(del.status, 200, `dismiss failed: ${del.status}`);

        const mine = await call('GET', `/api/notifications/user/${investor._id}`);
        assert.ok(!mine.body.some((n) => n._id === String(g._id)),
            'the dismissed broadcast reappeared');

        actingAs = other;
        const theirs = await call('GET', `/api/notifications/user/${other._id}`);
        assert.ok(theirs.body.some((n) => n._id === String(g._id)),
            'one user dismissing a broadcast removed it for everyone');
    });

    await t('a direct notification is still marked read on the document', async () => {
        await Notification.deleteMany({});
        const d = await Notification.create({
            recipient: investor._id, type: 'investment', message: 'Direct note',
        });
        actingAs = investor;
        const r = await call('PUT', `/api/notifications/${d._id}/read`);
        assert.strictEqual(r.status, 200);
        const raw = await Notification.findById(d._id).lean();
        assert.strictEqual(raw.read, true, 'direct notification was not marked read');
    });

    // ── 5. forged tenure ────────────────────────────────────────────────────
    await t('THE FORGERY: registration ignores a client-supplied memberSince', async () => {
        const src = require('fs').readFileSync(require.resolve('./routes/auth'), 'utf8');
        // Strip comments first: the explanation of this bug names the field, and a
        // naive grep would match the very comment describing the fix.
        const code = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
        assert.ok(!/memberSince\s*:\s*memberSince/.test(code),
            'memberSince is still assigned from the request body');
        assert.ok(!/\bmemberSince\b[^\n]*req\.body/.test(code),
            'memberSince is still destructured from req.body');

        // and prove it end to end through the model
        const u = await User.create({
            name: 'Late Joiner', email: 'late@t.local', password: 'x'.repeat(20),
            role: 'farmer',
        });
        assert.ok(u.memberSince, 'memberSince was never set');
        assert.ok(Math.abs(u.memberSince - u.createdAt) < 5000,
            'memberSince does not track createdAt');
    });

    srv.close();
    await mongoose.disconnect();
    await mem.stop();

    console.log(`\n  ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})();
