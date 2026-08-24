// --- Main Backend Server File: server.js ---

// Import necessary Node.js modules and installed packages
const express = require('express');        // Web framework for building APIs
const mongoose = require('mongoose');      // Mongoose for interacting with MongoDB
const cors = require('cors');
const helmet = require('helmet');              // Middleware to enable Cross-Origin Resource Sharing
const dotenv = require('dotenv');          // To load environment variables from .env file
const path = require('path');              // Node.js built-in module for working with file paths

// Load environment variables from the .env file into process.env
// This should be done as early as possible.
dotenv.config();

// Create an instance of the Express application
const app = express();

// --- Middleware Setup ---

// CORS (Cross-Origin Resource Sharing) Middleware
// This allows your frontend running on one origin (e.g., http://localhost:8080 or a file:// URL)
// to make requests to your backend running on a different origin (e.g., http://localhost:3000).
// For development, '*' is used to allow requests from any origin.
// In production, you should replace '*' with your actual frontend domain(s) for security.
// CORS.
//
// The previous form was
//     origin: NODE_ENV === 'production' ? process.env.FRONTEND_URL : '*'
// which fails open: if FRONTEND_URL is unset in production, `origin` is
// undefined and the cors package answers with Access-Control-Allow-Origin: *.
// So the one configuration that was supposed to lock things down produced the
// most permissive result, silently. Refuse to start instead.
const IS_PROD = process.env.NODE_ENV === 'production';
const allowedOrigins = String(process.env.FRONTEND_URL || '')
    .split(',').map((o) => o.trim()).filter(Boolean);

if (IS_PROD && allowedOrigins.length === 0) {
    console.error('NODE_ENV=production but FRONTEND_URL is not set.');
    console.error('Set FRONTEND_URL to the exact origin(s) allowed to call this API,');
    console.error('comma-separated, e.g. https://pestivid.example');
    console.error('Refusing to start with an open CORS policy.');
    process.exit(1);
}

const corsOptions = {
    origin: IS_PROD
        ? (origin, cb) => {
            // No Origin header: same-origin, curl, or a mobile webview. Allow it;
            // CORS is not what protects those, the JWT is.
            if (!origin) return cb(null, true);
            if (allowedOrigins.includes(origin)) return cb(null, true);
            return cb(new Error(`Origin ${origin} is not allowed by CORS.`));
        }
        : '*',
    credentials: true,
    optionsSuccessStatus: 200, // Some legacy browsers choke on 204
};
app.use(cors(corsOptions));

// Security headers. There were none.
//
// contentSecurityPolicy is disabled here because this server also serves the
// single-file Vue frontend from ../frontend, which loads Tailwind, Vue, axios,
// marked, DOMPurify and the transformers.js runtime from CDNs and uses inline
// handlers. A CSP tight enough to be worth having would break the app; one loose
// enough to pass would be decoration. The frontend's CSP belongs on the static
// host that serves it, where the real origin list is known.
app.use(helmet({
    contentSecurityPolicy: false,
    // The AI model is fetched cross-origin from the HuggingFace CDN, and COEP
    // would block it.
    crossOriginEmbedderPolicy: false,
    // Video is served from IPFS gateways into <video> tags.
    crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

// Built-in Express middleware to parse incoming requests with JSON payloads.
// This makes the JSON data sent from your frontend available on req.body.
// The { limit: '10mb' } increases the maximum request body size, useful if sending base64 image data for AI analysis.
app.use(express.json({ limit: '10mb' }));


// Optional: Serve static files (like your index.html, CSS, JS bundles) from this backend.
// If you put your frontend index.html in a public folder at the same level as the backend folder,
// uncommenting the line below will make it accessible directly from this server (e.g., at http://localhost:3000).
// The root URL is the landing page, not the legacy single-page app that
// happens to be called index.html. This runs BEFORE express.static, which would
// otherwise answer "/" with frontend/index.html.
// The page's own links are relative to /app/, so serving the file verbatim at
// "/" 404s its script. A <base> makes one file correct at both URLs, and keeps
// "/" as the canonical address rather than redirecting to a path.
const LANDING = path.join(__dirname, '../frontend/app/landing.html');
let landingAtRoot = null;
app.get(['/', '/index.html'], (req, res, next) => {
    try {
        if (landingAtRoot === null || process.env.NODE_ENV !== 'production') {
            const html = require('fs').readFileSync(LANDING, 'utf8');
            landingAtRoot = html.replace('<head>', '<head><base href="/app/">');
        }
        res.type('html').send(landingAtRoot);
    } catch (err) { next(err); }
});

app.use(express.static(path.join(__dirname, '../frontend')));


// --- Database Connection ---

// Connect to the MongoDB database using the connection string from the .env file.
// Mongoose options like useNewUrlParser and useUnifiedTopology are recommended for connecting.
// --- Required configuration check ---
// Previously a missing MONGODB_URI produced an unhandled TypeError inside
// app.listen (the URI masking call), *after* "Server running on port 3001"
// had already been printed - which made it look like a startup success.
if (!process.env.MONGODB_URI) {
    console.error('FATAL: MONGODB_URI is not set. Copy backend/.env.example to backend/.env and fill it in.');
    process.exit(1);
}
if (!process.env.JWT_SECRET) {
    console.error('FATAL: JWT_SECRET is not set. Generate one with:');
    console.error('  node -e  console.log(require(`crypto`).randomBytes(48).toString(`base64`))');
    process.exit(1);
}

// ── MongoDB ─────────────────────────────────────────────────────────────────
// Works with both a local mongod and a MongoDB Atlas SRV string. Atlas is the
// deployment target: a hosted container cannot reach a localhost database, so
// the same URI has to work from anywhere.
if (!process.env.MONGODB_URI) {
    console.error('MONGODB_URI is not set. Copy .env.example to backend/.env and fill it in.');
    process.exit(1);
}

const IS_ATLAS = process.env.MONGODB_URI.startsWith('mongodb+srv://');

mongoose.connect(process.env.MONGODB_URI, {
    // Fail fast with a readable message rather than hanging. Atlas connections
    // that are going to fail almost always fail immediately.
    serverSelectionTimeoutMS: Number(process.env.MONGO_TIMEOUT_MS || 10000),
    // Atlas free tier (M0) caps connections; a small pool avoids exhausting it
    // when the host runs more than one instance.
    maxPoolSize: Number(process.env.MONGO_POOL || (IS_ATLAS ? 10 : 20)),
    retryWrites: true,
})
    .then(() => console.log(`MongoDB connected (${IS_ATLAS ? 'Atlas' : 'local'})`))
    .catch((err) => {
        // Atlas failures are nearly always one of three things and the driver's
        // own message does not say which. Naming them saves a long debug session.
        console.error(`MongoDB connection failed: ${err.message}`);
        const atlasHelp = [
            '',
            'Most likely one of:',
            '  1. Your IP is not allow-listed.  Atlas > Network Access > Add IP Address.',
            '     A hosted deployment usually has no fixed egress IP, so add 0.0.0.0/0',
            '     and rely on the database user password for access control.',
            '  2. Wrong database username or password in MONGODB_URI. A password',
            '     containing @ : / ? # or % must be percent-encoded.',
            '  3. Missing database name. The URI needs one before the "?" --',
            '     e.g. ...mongodb.net/pestivid_db?retryWrites=true&w=majority',
            '',
        ].join('\n');
        const localHelp = [
            '',
            'No local MongoDB is reachable. Either start mongod, or point',
            'MONGODB_URI at a MongoDB Atlas connection string (mongodb+srv://...).',
            '',
        ].join('\n');
        console.error(IS_ATLAS ? atlasHelp : localHelp);
        process.exit(1);
    });

// A dropped connection must not leave the server looking healthy.
mongoose.connection.on('disconnected', () =>
    console.warn('MongoDB disconnected; the driver will attempt to reconnect.'));
mongoose.connection.on('reconnected', () =>
    console.log('MongoDB reconnected.'));


// ── Periodic Bitcoin anchoring ──────────────────────────────────────────────
// A batch is anchored on an interval rather than per upload, because one Merkle
// root covers unlimited videos and OpenTimestamps is free but rate-limited by
// its calendars. Upgrading looks for Bitcoin confirmation of earlier batches,
// which normally arrives a few hours after stamping.
//
// Disabled by default in development so a local run does not talk to public
// calendar servers on every restart. Set ANCHOR_ENABLED=true to turn it on.
if (String(process.env.ANCHOR_ENABLED || '').toLowerCase() === 'true') {
    const anchorSvc = require('./services/anchor');
    const ANCHOR_EVERY_MS = Number(process.env.ANCHOR_INTERVAL_MS || 6 * 60 * 60 * 1000);
    const UPGRADE_EVERY_MS = Number(process.env.ANCHOR_UPGRADE_MS || 60 * 60 * 1000);

    const runAnchor = async () => {
        try {
            const b = await anchorSvc.anchorPending();
            if (b) {
                console.log(`Anchored ${b.videos.length} video record(s), root ` +
                            `${b.merkleRoot.slice(0, 16)}… status=${b.status}`);
            }
        } catch (e) {
            console.error('Anchoring run failed:', e.message);
        }
    };
    const runUpgrade = async () => {
        try {
            const r = await anchorSvc.upgradePending();
            if (r.upgraded) console.log(`${r.upgraded} batch(es) confirmed on Bitcoin.`);
        } catch (e) {
            console.error('Anchor upgrade run failed:', e.message);
        }
    };

    // Wait for the first connection before touching the database.
    mongoose.connection.once('open', () => {
        setTimeout(runAnchor, 30000);
        setInterval(runAnchor, ANCHOR_EVERY_MS);
        setInterval(runUpgrade, UPGRADE_EVERY_MS);
        console.log(`Bitcoin anchoring enabled (every ${ANCHOR_EVERY_MS / 3600000}h).`);
    });
} else {
    console.log('Bitcoin anchoring disabled (set ANCHOR_ENABLED=true to enable).');
}

// --- Mongoose Models ---
// Require all your Mongoose model files. This registers the schemas and models with Mongoose.
// It's important to require these after the Mongoose connection is initiated but before
// you define your routes that will use these models.
require('./models/User');
require('./models/Video');
require('./models/AnchorBatch');
require('./models/RetiredFingerprint');
require('./models/Listing');
require('./models/FundingRequest');
require('./models/Investment');
require('./models/Purchase');
require('./models/Transaction');
require('./models/Conversation');
require('./models/Message');
require('./models/Notification');


// --- API Routes ---
// Import your route modules and use them with app.use().
// Each app.use() mounts the specified router middleware at a specific path prefix.
// Requests starting with that path will be handled by the routes defined in that module.

// Import the authentication router and the authentication middleware from the auth route file
const { router: authRoutes, authenticateToken } = require('./routes/auth');

// Rate limits. There were none before, while /api/videos/upload accepted a
// 100 MB body and ran ffmpeg per request -- see middleware/rateLimits.js.
const limits = require('./middleware/rateLimits');

// trust proxy MUST be set before any rate limiter runs. Behind Render/Fly/nginx
// every request arrives from the proxy, so without this req.ip is the proxy's
// address and all limits collapse into one shared bucket -- one abusive client
// locks out everybody. Set it to the number of proxies in front of us; blanket
// `true` would let a client forge X-Forwarded-For and bypass limits entirely.
app.set('trust proxy', Number(process.env.TRUST_PROXY_HOPS || 0));

// A coarse IP-keyed backstop. It runs BEFORE any route's authenticateToken, so
// req.user does not exist yet and the per-user keying inside userOrIp() can
// never engage here -- this limiter is effectively per-IP and is documented as
// such rather than pretending otherwise. The limiters that genuinely need
// per-user keying (upload) are applied after authentication, at the route.
app.use('/api', limits.generalLimiter);

// POST only. app.use() matches every method, so an anonymous GET /api/auth/login
// fell through to the 404 handler -- and because the limiter counts only
// non-2xx responses, each 404 consumed budget. Ten credential-free GETs from any
// address therefore locked EVERY user out of login for 15 minutes.
app.post('/api/auth/login', limits.authLimiter);
app.post('/api/auth/register', limits.authLimiter);
// change-password verifies the CURRENT password, so an unlimited endpoint here is
// a password oracle: 403 means wrong, 200 means right. It needs the same limiter
// as login even though the caller is already authenticated.
app.post('/api/auth/change-password', limits.authLimiter);
app.post('/api/auth/sign-out-everywhere', limits.authLimiter);

app.use('/api/auth', authRoutes);                     // Mount authentication routes under /api/auth
app.use('/api/users', require('./routes/users'));     // Mount user-related routes under /api/users
app.use('/api/videos', require('./routes/videos'));   // Mount video routes under /api/videos
app.use('/api/listings', require('./routes/listings')); // Mount listing routes under /api/listings
app.use('/api/funding-requests', require('./routes/fundingRequests')); // Mount funding request routes under /api/funding-requests
app.use('/api/investments', require('./routes/investments')); // Mount investment routes under /api/investments
app.use('/api/purchases', require('./routes/purchases')); // Mount purchase routes under /api/purchases
app.use('/api/transactions', require('./routes/transactions')); // Mount transaction routes under /api/transactions
app.use('/api/messaging', require('./routes/messaging')); // Mount messaging routes under /api/messaging
app.use('/api/notifications', require('./routes/notifications')); // Mount notification routes under /api/notifications
app.use('/api/ai', require('./routes/ai'));           // Mount AI proxy routes under /api/ai


// The root path serves the landing page (mounted above, before the static
// middleware). This used to answer it with the string "PestiVid Backend API is
// running!", which never fired because express.static got there first.
app.get('/healthz', (req, res) => {
    res.type('text/plain').send('ok');
});


// Optional: Serve the frontend index.html for any route not matched by the API routes above.
// This is useful if you are serving your frontend build files from the backend server.
// Make sure this comes after all your API routes.
// app.get('*', (req, res) => {
//   res.sendFile(path.join(__dirname, '../frontend/index.html'));
// });


// --- Error Handling Middleware ---

// Catch 404 errors (requests that didn't match any route) and forward to error handler
app.use((req, res, next) => {
    const err = new Error('Not Found');
    err.status = 404;
    next(err); // Pass the error to the next middleware (which will be one of the error handlers below)
});

// General Error Handler Middleware
// This middleware catches errors passed via next(err) or thrown in synchronous code.
// It sends a JSON response with the error details.

// Development error handler (will print stacktrace for debugging)
if (app.get('env') === 'development') {
    app.use((err, req, res, next) => {
        console.error("Development Error:", err.stack); // Log the error stack in development
        res.status(err.status || 500); // Set status code from error, or default to 500
        res.json({
            message: err.message, // Send error message
            error: err // Send the error object itself (includes stack in dev)
        });
    });
}

// Production error handler (should not leak sensitive error details to the user)
app.use((err, req, res, next) => {
    console.error("Production Error:", err.message); // Log message in production
    res.status(err.status || 500); // Set status code
    res.json({
        message: err.message, // Send only the message in production
        error: {} // Send an empty error object or exclude it
    });
});


// --- Server Start ---

// Start the Express server and make it listen for incoming connections on the specified PORT.
// The PORT is read from the .env file, defaulting to 3000 if not set.
const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    // Mask a portion of the MongoDB URI in the console log for basic security
    const uri = process.env.MONGODB_URI || '';
    const at = uri.indexOf('@');
    const maskedUri = uri.substring(0, at > -1 ? at + 1 : 30);
    console.log(`Attempting to connect to MongoDB at: ${maskedUri}...`);
});

// Note: The authenticateToken middleware is defined in auth.js and exported from there.
// It is then required and used in this server.js file, and subsequently in other route files
// where it's imported from auth.js.