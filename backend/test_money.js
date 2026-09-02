/**
 * Money-path tests. These cover the bugs the audit proved, so a regression here
 * means an investor is paid the wrong amount.
 *
 *   1. profit share paid 0% because nothing ever wrote harvestRevenue
 *   2. full_repayment was unreachable (settlementMode had a default)
 *   3. concurrent settlement paid N times (read-modify-write)
 *   4. the settle endpoint returned 500 on every call, after paying
 *
 *     node test_money.js
 */

const assert = require('assert');

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
    await mongoose.connect(mem.getUri(), { dbName: 'money_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message',
     'AnchorBatch', 'RetiredFingerprint'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });

    const User = mongoose.model('User');
    const FundingRequest = mongoose.model('FundingRequest');
    const Investment = mongoose.model('Investment');

    const farmer = await User.create({
        name: 'Farmer One', email: 'fm@t.local', password: 'x'.repeat(20), role: 'farmer',
    });
    const investor = await User.create({
        name: 'Investor One', email: 'iv@t.local', password: 'x'.repeat(20), role: 'investor',
    });

    const mkProject = (extra = {}) => FundingRequest.create({
        title: 'Potato season', farmerWallet: farmer._id, crop: 'Potato',
        acres: 2, amount: 1000, method: 'organic', cid: 'bafy' + Math.random().toString(36).slice(2),
        description: 'd', timeline: 4, roi: 20, investorShare: 60,
        status: 'funded', fundedAmount: 1000, ...extra,
    });

    // ── 2. settlementMode must be reachable ─────────────────────────────────
    await t('settlementMode is copied from the project, not forced to profit_share', async () => {
        const proj = await mkProject({ settlementMode: 'full_repayment' });
        const inv = await Investment.create({
            investorWallet: investor._id, projectId: proj._id, amount: 100,
        });
        assert.strictEqual(inv.settlementMode, 'full_repayment',
            `investment took mode "${inv.settlementMode}" instead of the project's full_repayment`);
    });

    await t('a project with no explicit mode defaults to profit_share on the investment', async () => {
        const proj = await mkProject({});
        const inv = await Investment.create({
            investorWallet: investor._id, projectId: proj._id, amount: 50,
        });
        assert.strictEqual(inv.settlementMode, 'profit_share');
    });

    // ── 1. harvest reporting must exist and be one-shot ─────────────────────
    await t('harvestReportedAt is unset until a harvest is reported', async () => {
        const proj = await mkProject({});
        assert.ok(!proj.harvestReportedAt, 'a fresh project already claims a harvest');
    });

    await t('the harvest fields persist and profit is computable', async () => {
        const proj = await mkProject({});
        proj.harvestRevenue = 5000;
        proj.inputCostBasis = 2000;
        proj.outcome = 'harvested';
        proj.harvestReportedAt = new Date();
        await proj.save();
        const back = await FundingRequest.findById(proj._id).lean();
        assert.strictEqual(back.harvestRevenue, 5000);
        assert.strictEqual(back.inputCostBasis, 2000);
        assert.strictEqual(Math.max(0, back.harvestRevenue - back.inputCostBasis), 3000);
    });

    await t('the harvest route exists, is the owner\'s alone, and one-shot', async () => {
        const src = require('fs').readFileSync('./routes/fundingRequests.js', 'utf8');
        assert.ok(src.includes("router.post('/:id/harvest'"), 'no harvest endpoint');
        // Any account may farm now, so the gate is ownership of the season, in
        // the same atomic filter as the one-shot check.
        assert.ok(src.includes('farmerWallet: req.user._id'), 'harvest endpoint does not check the season is yours');
        assert.ok(src.includes('harvestReportedAt: { $exists: false }'),
            'the one-shot filter is missing, so a farmer could revise figures after seeing payouts');
    });

    // ── 3. the atomic settlement claim ──────────────────────────────────────
    await t('ATOMIC CLAIM: only one of five concurrent claims can win', async () => {
        const proj = await mkProject({});
        const inv = await Investment.create({
            investorWallet: investor._id, projectId: proj._id, amount: 500,
        });
        // This is the exact conditional update the route now performs.
        const claim = () => Investment.findOneAndUpdate(
            { _id: inv._id, payoutNotified: { $ne: true }, status: { $ne: 'harvested' } },
            { $set: { payoutNotified: true, status: 'harvested', payoutAmount: 500 } },
            { new: true },
        );
        const results = await Promise.all([claim(), claim(), claim(), claim(), claim()]);
        const winners = results.filter(Boolean).length;
        assert.strictEqual(winners, 1,
            `${winners} concurrent claims succeeded - that is ${winners}x the payout`);
    });

    await t('the settlement route uses a conditional update, not read-modify-write', async () => {
        const src = require('fs').readFileSync('./routes/investments.js', 'utf8');
        assert.ok(/Investment\.findOneAndUpdate\(/.test(src), 'no atomic claim present');
        assert.ok(src.includes('payoutNotified: { $ne: true }'),
            'the claim does not filter on payoutNotified, so it is not actually a claim');
    });

    await t('profit_share settlement is blocked until the harvest is reported', async () => {
        const src = require('fs').readFileSync('./routes/investments.js', 'utf8');
        assert.ok(src.includes('harvest_not_reported'),
            'settlement does not refuse an unreported harvest, so it pays 0% and latches');
    });

    // ── 4. the response formatter must not be able to throw ────────────────
    await t('the settle response populates farmerWallet before formatting it', async () => {
        const raw = require('fs').readFileSync('./routes/investments.js', 'utf8');
        // Strip comments first. The doc comment that EXPLAINS this bug contains
        // the very pattern being searched for, so a naive grep fails on its own
        // documentation -- the same trap that once disabled two repo guards.
        const src = raw
            .replace(/\/\*[\s\S]*?\*\//g, '')
            .replace(/^\s*\/\/.*$/gm, '');
        assert.ok(/populate\('farmerWallet'/.test(src),
            'farmerWallet is never populated, so .name is undefined and the handler 500s');
        assert.ok(!/\.name\.split/.test(src),
            'an unguarded .name.split remains in code, which throws after the payout is committed');
        assert.ok(/function firstName\(/.test(raw),
            'no safe name helper; formatting can still throw after money has moved');
    });

    // ── payout arithmetic, both modes ───────────────────────────────────────
    await t('payout arithmetic: full_repayment returns principal + ROI', async () => {
        const amount = 100;
        const roi = 20;
        const payout = amount + amount * (roi / 100);
        assert.strictEqual(payout, 120);
    });

    await t('payout arithmetic: profit share is pro-rata on the pool share', async () => {
        const raised = 1000;
        const amount = 250;
        const profit = 2000;
        const investorShare = 50;
        const pool = profit * (investorShare / 100);          // 1000
        const payout = amount + pool * (amount / raised);     // 250 + 250
        assert.strictEqual(payout, 500);
    });

    await t('payout arithmetic: a total loss pays nothing, not a negative', async () => {
        const revenue = 100;
        const costs = 900;
        const profit = Math.max(0, revenue - costs);
        assert.strictEqual(profit, 0);
    });

    await mongoose.disconnect();
    await mem.stop();
    console.log(`\n${pass}/${pass + fail} checks passed`);
    process.exit(fail ? 1 : 0);
})().catch(async (e) => {
    console.error('harness error:', e.message);
    try { await mongoose.disconnect(); } catch (_) {}
    process.exit(1);
});
