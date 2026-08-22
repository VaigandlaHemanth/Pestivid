/**
 * Token revocation, password change, and the authoritative role.
 *
 * A stateless JWT cannot answer "is this session still valid?", and nothing here
 * used to try. The consequences, all live:
 *
 *   - A leaked token was good for its full 24 hours with no way to kill it.
 *   - There was no password-change route at all, and the seeded demo credentials
 *     are published in seed.js.
 *   - `role` was baked into the token at login, so a demoted user kept the old
 *     role until expiry.
 *
 * These tests exercise the REAL authenticateToken against real tokens, because
 * every other suite stubs it out -- which means nothing was covering the one
 * piece of code every authenticated request runs through.
 *
 *     node test_auth_revocation.js
 */

const assert = require('assert');
const express = require('express');
const jwt = require('jsonwebtoken');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

process.env.JWT_SECRET = process.env.JWT_SECRET || 'test-secret-for-revocation-suite';

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
    await mongoose.connect(mem.getUri(), { dbName: 'auth_test' });
    ['User', 'Video', 'Listing', 'FundingRequest', 'Investment', 'Purchase',
     'Transaction', 'Notification', 'Conversation', 'Message'].forEach((m) => {
        try { require(`./models/${m}`); } catch (_) { /* optional */ }
    });
    const User = mongoose.model('User');

    // The REAL middleware this time -- no stub.
    delete require.cache[require.resolve('./routes/auth')];
    const authModule = require('./routes/auth');

    const app = express();
    app.use(express.json());
    app.use('/api/auth', authModule.router);
    // A trivial protected route, so the middleware is what is under test rather
    // than any particular business handler.
    app.get('/api/whoami', authModule.authenticateToken, (req, res) => {
        res.json({ _id: String(req.user._id), role: req.user.role, name: req.user.name });
    });
    const srv = app.listen(4881);

    const call = async (method, path, body, token) => {
        const r = await fetch(`http://127.0.0.1:4881${path}`, {
            method,
            headers: {
                'Content-Type': 'application/json',
                ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: body === undefined ? undefined : JSON.stringify(body),
        });
        let parsed = null;
        try { parsed = await r.json(); } catch (_) { /* 401s send no body */ }
        return { status: r.status, body: parsed };
    };

    const register = async (email, password, role = 'farmer') => {
        const r = await call('POST', '/api/auth/register',
            { name: 'Test Person', email, password, role });
        assert.strictEqual(r.status, 201, `register failed: ${r.status} ${JSON.stringify(r.body)}`);
        return r.body.token;
    };

    // ── the token works at all ──────────────────────────────────────────────
    await t('a freshly issued token is accepted', async () => {
        const token = await register('a1@t.local', 'password123');
        const r = await call('GET', '/api/whoami', undefined, token);
        assert.strictEqual(r.status, 200, `expected 200, got ${r.status}`);
        assert.strictEqual(r.body.role, 'farmer');
    });

    await t('no token is 401, a forged token is 403', async () => {
        const none = await call('GET', '/api/whoami');
        assert.strictEqual(none.status, 401, `no token gave ${none.status}`);
        const forged = jwt.sign({ _id: new mongoose.Types.ObjectId().toString(), role: 'farmer', tv: 0 },
            'the-wrong-secret');
        const bad = await call('GET', '/api/whoami', undefined, forged);
        assert.strictEqual(bad.status, 403, `forged token gave ${bad.status}`);
    });

    // ── revocation ──────────────────────────────────────────────────────────
    await t('THE LEAK: sign-out-everywhere kills an already-issued token', async () => {
        const token = await register('a2@t.local', 'password123');
        assert.strictEqual((await call('GET', '/api/whoami', undefined, token)).status, 200);

        const out = await call('POST', '/api/auth/sign-out-everywhere', {}, token);
        assert.strictEqual(out.status, 200, `sign-out failed: ${out.status}`);

        const after = await call('GET', '/api/whoami', undefined, token);
        assert.strictEqual(after.status, 401,
            `a revoked token still works (${after.status}) -- there is no revocation`);
        assert.strictEqual(after.body && after.body.code, 'token_revoked');
    });

    await t('a token minted with a stale version is rejected', async () => {
        const email = 'a3@t.local';
        await register(email, 'password123');
        const user = await User.findOne({ email }).lean();
        // Hand-mint a token one version behind: exactly what an attacker replaying
        // a captured token has after the account owner signs out everywhere.
        const stale = jwt.sign(
            { _id: String(user._id), role: user.role, tv: (user.tokenVersion || 0) - 1 },
            process.env.JWT_SECRET);
        const r = await call('GET', '/api/whoami', undefined, stale);
        assert.strictEqual(r.status, 401, `stale version accepted (${r.status})`);
    });

    await t('a token with NO version field is rejected, not grandfathered', async () => {
        const email = 'a4@t.local';
        await register(email, 'password123');
        const user = await User.findOne({ email }).lean();
        // The old token shape. Accepting it would leave the exact hole this closes.
        const legacy = jwt.sign({ _id: String(user._id), role: user.role },
            process.env.JWT_SECRET);
        const r = await call('GET', '/api/whoami', undefined, legacy);
        assert.strictEqual(r.status, 401, `a version-less token was accepted (${r.status})`);
    });

    await t('a token for a deleted account is rejected', async () => {
        const email = 'a5@t.local';
        const token = await register(email, 'password123');
        await User.deleteOne({ email });
        const r = await call('GET', '/api/whoami', undefined, token);
        assert.strictEqual(r.status, 401, `deleted account still authenticates (${r.status})`);
    });

    // ── role comes from the database, not the token ──────────────────────────
    await t('THE STALE ROLE: a role change takes effect immediately', async () => {
        const email = 'a6@t.local';
        const token = await register(email, 'password123', 'farmer');
        assert.strictEqual((await call('GET', '/api/whoami', undefined, token)).body.role, 'farmer');

        // Demote without touching tokenVersion: the token stays valid, but the
        // role it carries must no longer be believed.
        await User.updateOne({ email }, { $set: { role: 'buyer' } });

        const r = await call('GET', '/api/whoami', undefined, token);
        assert.strictEqual(r.status, 200, 'a valid token was rejected after a role change');
        assert.strictEqual(r.body.role, 'buyer',
            'the token\'s baked-in role was trusted over the database');
    });

    // ── password change ─────────────────────────────────────────────────────
    await t('THE MISSING ROUTE: a password can be changed', async () => {
        const email = 'a7@t.local';
        const token = await register(email, 'password123');
        const r = await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'a-longer-secret' }, token);
        assert.strictEqual(r.status, 200, `change failed: ${r.status} ${JSON.stringify(r.body)}`);
        assert.ok(r.body.token, 'no replacement token was issued');

        // the new password works
        const login = await call('POST', '/api/auth/login',
            { email, password: 'a-longer-secret' });
        assert.strictEqual(login.status, 200, `new password does not work (${login.status})`);

        // the old one does not
        const old = await call('POST', '/api/auth/login', { email, password: 'password123' });
        assert.notStrictEqual(old.status, 200, 'the old password still works');
    });

    await t('changing a password ends every other session', async () => {
        const email = 'a8@t.local';
        const stolen = await register(email, 'password123');
        // A second session for the same account -- stand-in for the attacker's.
        const mine = (await call('POST', '/api/auth/login',
            { email, password: 'password123' })).body.token;

        const r = await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'a-longer-secret' }, mine);
        assert.strictEqual(r.status, 200);

        const attacker = await call('GET', '/api/whoami', undefined, stolen);
        assert.strictEqual(attacker.status, 401,
            'the other session survived a password change, so changing it fixed nothing');

        // and the replacement token handed back does work
        const fresh = await call('GET', '/api/whoami', undefined, r.body.token);
        assert.strictEqual(fresh.status, 200,
            `the replacement token does not work (${fresh.status})`);
    });

    await t('the wrong current password is refused', async () => {
        const email = 'a9@t.local';
        const token = await register(email, 'password123');
        const r = await call('POST', '/api/auth/change-password',
            { currentPassword: 'not-it', newPassword: 'a-longer-secret' }, token);
        assert.strictEqual(r.status, 403, `expected 403, got ${r.status}`);
        assert.strictEqual(r.body.code, 'wrong_password');
        // and the password really did not change
        const login = await call('POST', '/api/auth/login', { email, password: 'password123' });
        assert.strictEqual(login.status, 200, 'the original password stopped working');
    });

    await t('a short or unchanged new password is refused', async () => {
        const token = await register('a10@t.local', 'password123');
        const short = await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'abc' }, token);
        assert.strictEqual(short.status, 400);
        assert.strictEqual(short.body.code, 'password_too_short');

        const same = await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'password123' }, token);
        assert.strictEqual(same.status, 400);
        assert.strictEqual(same.body.code, 'password_unchanged');
    });

    await t('password change requires authentication', async () => {
        const r = await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'a-longer-secret' });
        assert.strictEqual(r.status, 401, `unauthenticated change gave ${r.status}`);
    });

    await t('one user cannot change another user\'s password', async () => {
        await register('victim@t.local', 'password123');
        const attackerToken = await register('attacker@t.local', 'password123');
        // There is no user id in the request at all -- the identity comes from the
        // token -- so the only thing the attacker can change is their own. Prove
        // the victim is untouched.
        await call('POST', '/api/auth/change-password',
            { currentPassword: 'password123', newPassword: 'a-longer-secret' }, attackerToken);
        const victim = await call('POST', '/api/auth/login',
            { email: 'victim@t.local', password: 'password123' });
        assert.strictEqual(victim.status, 200, 'the victim\'s password was affected');
    });

    // ── registration still cannot forge tenure ──────────────────────────────
    await t('registration ignores a client-supplied tokenVersion', async () => {
        const r = await call('POST', '/api/auth/register', {
            name: 'Sneaky', email: 'sneaky@t.local', password: 'password123',
            role: 'farmer', tokenVersion: 9999,
        });
        assert.strictEqual(r.status, 201);
        const u = await User.findOne({ email: 'sneaky@t.local' }).lean();
        assert.strictEqual(u.tokenVersion, 0,
            `tokenVersion was mass-assigned to ${u.tokenVersion}`);
    });

    srv.close();
    await mongoose.disconnect();
    await mem.stop();

    console.log(`\n  ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})();
