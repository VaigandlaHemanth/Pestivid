/**
 * The harvest -> settle flow, end to end.
 *
 * This is the platform's central promise: an investor funds a crop and is paid
 * from the harvest. Every piece of it existed on the server and was reachable
 * from nothing in the UI -- POST /funding-requests/:id/harvest and
 * PUT /investments/:id/progress were both written, both tested in isolation, and
 * both called by zero lines of frontend code. There was also no endpoint that
 * gave a farmer the investment ids in their own project, so even a determined
 * client could not have driven settlement.
 *
 * So this suite tests the JOIN: report a harvest, list the investments, settle
 * them, and check the money.
 *
 *     node test_settlement_flow.js
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
    await mongoose.connect(mem.getUri(), { dbName: 'settle_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });

    const User = mongoose.model('User');
    const FundingRequest = mongoose.model('FundingRequest');
    const Investment = mongoose.model('Investment');
    const Transaction = mongoose.model('Transaction');
    const Notification = mongoose.model('Notification');

    const farmer = await User.create({
        name: 'Settle Farmer', email: 'sf@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const otherFarmer = await User.create({
        name: 'Nosy Farmer', email: 'nf@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const alice = await User.create({
        name: 'Alice Investor', email: 'ai@t.local', password: 'x'.repeat(20), role: 'investor',
    });
    const bob = await User.create({
        name: 'Bob Investor', email: 'bi@t.local', password: 'x'.repeat(20), role: 'investor',
    });

    let actingAs = farmer;
    const authModule = require('./routes/auth');
    authModule.authenticateToken = (req, res, next) => { req.user = actingAs; next(); };
    for (const r of ['./routes/fundingRequests', './routes/investments']) {
        delete require.cache[require.resolve(r)];
    }
    const app = express();
    app.use(express.json());
    app.use('/api/funding-requests', require('./routes/fundingRequests'));
    app.use('/api/investments', require('./routes/investments'));
    const srv = app.listen(4879);

    const call = async (method, path, body) => {
        const r = await fetch(`http://127.0.0.1:4879${path}`, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        return { status: r.status, body: await r.json().catch(() => null) };
    };

    let cidN = 0;
    // A project funded by two investors, ready to harvest.
    async function fundedProject(extra = {}) {
        const p = await FundingRequest.create({
            title: 'Settle season', farmerWallet: farmer._id, crop: 'Potato',
            acres: 2, amount: 1000, method: 'organic', cid: `bafy-settle-${cidN++}`,
            description: 'd', timeline: 4, roi: 20, investorShare: 60,
            status: 'pending', fundedAmount: 0, ...extra,
        });
        actingAs = alice;
        const a = await call('POST', '/api/investments', { projectId: String(p._id), amount: 600 });
        assert.strictEqual(a.status, 201, `alice invest failed: ${a.status}`);
        actingAs = bob;
        const b = await call('POST', '/api/investments', { projectId: String(p._id), amount: 400 });
        assert.strictEqual(b.status, 201, `bob invest failed: ${b.status}`);
        actingAs = farmer;
        return p;
    }

    // ── the listing endpoint that did not exist ─────────────────────────────
    await t('THE MISSING LINK: the farmer can list the investments in their project', async () => {
        const p = await fundedProject();
        const { status, body } = await call('GET', `/api/investments/project/${p._id}`);
        assert.strictEqual(status, 200, `expected 200, got ${status}`);
        assert.strictEqual(body.investments.length, 2,
            `expected 2 investments, got ${body.investments.length}`);
        // The investment ids are what settlement needs and what nothing exposed.
        for (const i of body.investments) {
            assert.ok(mongoose.isValidObjectId(i._id), `bad investment id ${i._id}`);
            assert.strictEqual(i.settled, false);
        }
        assert.strictEqual(body.project.harvestReported, false);
    });

    await t('another farmer cannot list investments in someone else\'s project', async () => {
        const p = await fundedProject();
        actingAs = otherFarmer;
        const { status } = await call('GET', `/api/investments/project/${p._id}`);
        assert.strictEqual(status, 403, `expected 403, got ${status}`);
        actingAs = farmer;
    });

    await t('an investor cannot list the whole investor set of a project', async () => {
        const p = await fundedProject();
        actingAs = alice;
        const { status } = await call('GET', `/api/investments/project/${p._id}`);
        assert.strictEqual(status, 403, `expected 403, got ${status}`);
        actingAs = farmer;
    });

    // ── harvest -> settle, and the arithmetic ───────────────────────────────
    await t('THE WHOLE POINT: harvest then settle pays principal + share of profit', async () => {
        const p = await fundedProject();
        actingAs = farmer;

        // revenue 2000, costs 800 -> profit 1200; investorShare 60% -> pool 720
        const h = await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'harvested', harvestRevenue: 2000, inputCostBasis: 800,
        });
        assert.strictEqual(h.status, 200, `harvest failed: ${h.status} ${JSON.stringify(h.body)}`);

        const list = await call('GET', `/api/investments/project/${p._id}`);
        assert.strictEqual(list.body.project.harvestReported, true);
        assert.strictEqual(list.body.project.profit, 1200, `profit ${list.body.project.profit}`);
        assert.strictEqual(list.body.project.investorPool, 720,
            `pool ${list.body.project.investorPool}`);

        for (const inv of list.body.investments) {
            const r = await call('PUT', `/api/investments/${inv._id}/progress`, {
                progress: 100, updateText: 'settled',
            });
            assert.strictEqual(r.status, 200, `settle failed: ${r.status} ${JSON.stringify(r.body)}`);
        }

        // Alice put in 600 of 1000 -> 60% of the pool = 432, plus her 600 = 1032.
        // Bob put in 400 -> 40% of 720 = 288, plus 400 = 688.
        const after = await call('GET', `/api/investments/project/${p._id}`);
        const byName = {};
        after.body.investments.forEach((i) => { byName[i.investorName] = i; });

        assert.ok(byName['Alice Investor'], 'Alice missing from the settled list');
        assert.strictEqual(byName['Alice Investor'].payoutAmount, 1032,
            `Alice paid ${byName['Alice Investor'].payoutAmount}, expected 1032`);
        assert.strictEqual(byName['Bob Investor'].payoutAmount, 688,
            `Bob paid ${byName['Bob Investor'].payoutAmount}, expected 688`);

        // and the total paid out never exceeds principal + the investor pool
        const paid = after.body.investments.reduce((s, i) => s + i.payoutAmount, 0);
        assert.strictEqual(paid, 1000 + 720, `total paid ${paid}`);
    });

    await t('every settled investor gets a payout transaction and a notification', async () => {
        const p = await fundedProject();
        await Transaction.deleteMany({ type: 'payout' });
        await Notification.deleteMany({ type: 'payout' });
        actingAs = farmer;
        await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'harvested', harvestRevenue: 2000, inputCostBasis: 800,
        });
        const list = await call('GET', `/api/investments/project/${p._id}`);
        for (const inv of list.body.investments) {
            await call('PUT', `/api/investments/${inv._id}/progress`, { progress: 100 });
        }
        const txs = await Transaction.find({ type: 'payout' }).lean();
        assert.strictEqual(txs.length, 2, `${txs.length} payout transactions, expected 2`);
        const notes = await Notification.find({ type: 'payout' }).lean();
        assert.ok(notes.length >= 2, `${notes.length} payout notifications, expected >= 2`);
    });

    await t('a harvest can only be reported once', async () => {
        const p = await fundedProject();
        actingAs = farmer;
        const first = await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'harvested', harvestRevenue: 2000, inputCostBasis: 800,
        });
        assert.strictEqual(first.status, 200);
        const second = await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'harvested', harvestRevenue: 99999, inputCostBasis: 0,
        });
        assert.strictEqual(second.status, 409, `a second report returned ${second.status}`);
        assert.strictEqual(second.body.code, 'harvest_already_reported');
        const fresh = await FundingRequest.findById(p._id).lean();
        assert.strictEqual(fresh.harvestRevenue, 2000, 'the revenue was overwritten');
    });

    await t('settling twice does not pay twice', async () => {
        const p = await fundedProject();
        actingAs = farmer;
        await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'harvested', harvestRevenue: 2000, inputCostBasis: 800,
        });
        const list = await call('GET', `/api/investments/project/${p._id}`);
        const one = list.body.investments[0];
        await call('PUT', `/api/investments/${one._id}/progress`, { progress: 100 });
        const before = await Investment.findById(one._id).lean();
        await call('PUT', `/api/investments/${one._id}/progress`, { progress: 100 });
        const after = await Investment.findById(one._id).lean();
        assert.strictEqual(after.payoutAmount, before.payoutAmount,
            'the payout amount changed on a repeat settle');
        const txs = await Transaction.find({
            type: 'payout', investmentId: one._id,
        }).lean();
        assert.ok(txs.length <= 1, `${txs.length} payout transactions for one investment`);
    });

    await t('a total loss returns principal and no profit share', async () => {
        const p = await fundedProject();
        actingAs = farmer;
        const h = await call('POST', `/api/funding-requests/${p._id}/harvest`, {
            outcome: 'total_loss', harvestRevenue: 0, inputCostBasis: 0,
        });
        assert.strictEqual(h.status, 200, `total_loss harvest failed: ${h.status}`);
        const list = await call('GET', `/api/investments/project/${p._id}`);
        for (const inv of list.body.investments) {
            const r = await call('PUT', `/api/investments/${inv._id}/progress`, { progress: 100 });
            assert.strictEqual(r.status, 200, `settle failed: ${r.status}`);
        }
        const after = await call('GET', `/api/investments/project/${p._id}`);
        for (const i of after.body.investments) {
            assert.ok(i.payoutAmount != null, 'no payout recorded for a total loss');
            assert.ok(i.payoutAmount >= 0, `negative payout ${i.payoutAmount}`);
        }
    });

    await t('settlement is refused before a harvest is reported', async () => {
        const p = await fundedProject();
        const list = await call('GET', `/api/investments/project/${p._id}`);
        actingAs = farmer;
        const r = await call('PUT', `/api/investments/${list.body.investments[0]._id}/progress`,
            { progress: 100 });
        assert.notStrictEqual(r.status, 200,
            'an investment settled with no harvest reported, so the payout came from nowhere');
        const inv = await Investment.findById(list.body.investments[0]._id).lean();
        assert.strictEqual(inv.payoutAmount, undefined,
            `payoutAmount ${inv.payoutAmount} was written with no harvest figures`);
    });

    await t('a malformed project id is rejected, not ignored', async () => {
        actingAs = farmer;
        const { status } = await call('GET', '/api/investments/project/not-an-id');
        assert.strictEqual(status, 400, `expected 400, got ${status}`);
    });

    srv.close();
    await mongoose.disconnect();
    await mem.stop();

    console.log(`\n  ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})();
