// --- Local Dev Launcher: dev-server.js ---
//
// server.js needs a MongoDB instance and a handful of env vars, none of which
// ship with the repo (.env is gitignored, and MongoDB is not installed here).
//
// This launcher fills both gaps:
//   1. Boots a real mongod via mongodb-memory-server, backed by a folder on
//      disk so data survives restarts.
//   2. Sets MONGODB_URI / JWT_SECRET / PORT before requiring server.js.
//      dotenv does not overwrite variables that are already set, so a
//      hand-written backend/.env still wins for anything it defines.
//   3. Seeds demo data the first time the database comes up empty.
//
// Run with:  node dev-server.js

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

// Pick up backend/.env first if the user created one (for GROQ_API_KEY etc).
require('dotenv').config({ path: path.join(__dirname, '.env') });

const { MongoMemoryServer } = require('mongodb-memory-server');
const mongoose = require('mongoose');

const DB_PATH = path.join(__dirname, '.local-mongo-data');

// The memory server persists to DB_PATH, and seedIfEmpty() skips when any user
// exists -- so anything a test writes lives in the dev database for good. An
// e2e run left a farmer called "Phone Farmer" and two funding requests behind,
// and every screenshot afterwards showed "Phone told us" and "Ask Phone a
// question". `node dev-server.js --fresh` throws the directory away and reseeds.
const FRESH = process.argv.includes('--fresh');
const MONGO_PORT = 27017;

// Stable secret so tokens issued before a restart still validate afterwards.
//
// It did not do that. The line under this comment read `randomBytes(48)` with a
// trailing note saying "rotates each restart", which is the opposite of what the
// comment above it promised -- and the code won. Every restart invalidated every
// token that had been issued, so each tab open at the time 401'd on its next call
// and was bounced to the sign-in screen. Restarting the server to pick up a code
// change should not sign anybody out.
//
// The secret now lives beside the database it goes with: same directory, same
// lifetime, thrown away by the same --fresh. Still dev-only, still overridable.
const SECRET_PATH = path.join(DB_PATH, '.jwt-secret');
function devSecret() {
    if (process.env.JWT_SECRET) return process.env.JWT_SECRET;
    try {
        const kept = fs.readFileSync(SECRET_PATH, 'utf8').trim();
        if (kept) return kept;
    } catch { /* first run, or --fresh just removed it */ }
    const made = require('crypto').randomBytes(48).toString('base64');
    try {
        fs.mkdirSync(DB_PATH, { recursive: true });
        fs.writeFileSync(SECRET_PATH, made, { mode: 0o600 });
    } catch (e) {
        console.warn(`Could not keep the dev JWT secret (${e.message}); sessions will end at restart.`);
    }
    return made;
}
const DEV_JWT_SECRET = devSecret();

let mongod;

async function main() {
    if (FRESH && fs.existsSync(DB_PATH)) {
        console.log(`--fresh: removing ${DB_PATH}`);
        fs.rmSync(DB_PATH, { recursive: true, force: true });
    }
    fs.mkdirSync(DB_PATH, { recursive: true });

    console.log('--- Starting local MongoDB (mongodb-memory-server) ---');
    mongod = await MongoMemoryServer.create({
        instance: {
            port: MONGO_PORT,
            dbName: 'pestivid',
            dbPath: DB_PATH,
            storageEngine: 'wiredTiger'
        }
    });

    const uri = mongod.getUri('pestivid');
    console.log(`MongoDB listening at ${uri}`);
    console.log(`Data directory: ${DB_PATH}`);

    process.env.MONGODB_URI = uri;
    process.env.JWT_SECRET = process.env.JWT_SECRET || DEV_JWT_SECRET;
    process.env.PORT = process.env.PORT || '3001';
    process.env.NODE_ENV = process.env.NODE_ENV || 'development';

    await seedIfEmpty(uri);
    reportAiConfig();

    console.log('--- Starting PestiVid backend ---');
    require('./server.js');
}

// Runs seed.js as a child process the first time round. seed.js opens and
// closes its own connection, so it has to stay out of this process.
async function seedIfEmpty(uri) {
    await mongoose.connect(uri);
    const users = await mongoose.connection.db.collection('users').countDocuments();
    await mongoose.disconnect();

    if (users > 0) {
        console.log(`Database already has ${users} users — skipping seed.`);
        return;
    }

    console.log('Empty database — running seed.js ...');
    const result = spawnSync(process.execPath, ['seed.js'], {
        cwd: __dirname,
        stdio: 'inherit',
        env: process.env
    });
    if (result.status !== 0) {
        console.warn('seed.js exited non-zero — continuing with an empty database.');
    }
}

// AgriBot and the plant-analysis fallback both go through Groq. Say so loudly
// rather than letting the browser surface an opaque 500.
function reportAiConfig() {
    if (process.env.GROQ_API_KEY) {
        console.log('GROQ_API_KEY found — AgriBot chat is live.');
        return;
    }
    console.warn('');
    console.warn('  GROQ_API_KEY is not set. AgriBot chat will return an error.');
    console.warn('  Fix: create backend/.env containing one line —');
    console.warn('      GROQ_API_KEY=your_key_here');
    console.warn('  Free key: https://console.groq.com/keys');
    console.warn('');
}

async function shutdown() {
    console.log('\nShutting down...');
    try { await mongoose.disconnect(); } catch (_) {}
    if (mongod) await mongod.stop();
    process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

main().catch(async (err) => {
    console.error('Failed to start:', err);
    if (mongod) await mongod.stop();
    process.exit(1);
});
