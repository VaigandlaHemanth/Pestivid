// --- Backend Routes: auth.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose (needed to access models)
const User = mongoose.model('User');        // Get the User Mongoose model (requires models/User.js to have been required in server.js)
const bcrypt = require('bcrypt');           // Import bcrypt for password comparison
const jwt = require('jsonwebtoken');        // Import jsonwebtoken for creating and verifying tokens
// const dotenv = require('dotenv');         // Usually configured in server.js, but can be here if running routes standalone
// const path = require('path');             // Usually configured in server.js

// dotenv.config(); // Uncomment this line if you ever need to run this route file in isolation


// --- Authentication Middleware ---
// This middleware is used to protect routes that require a user to be logged in.
const authenticateToken = async (req, res, next) => {
    // Get the authorization header from the request (usually 'Bearer TOKEN')
    const authHeader = req.headers['authorization'];
    // Extract the token from the 'Bearer ' scheme
    const token = authHeader && authHeader.split(' ')[1]; // Check if header exists and split

    // If no token is provided, return 401 Unauthorized
    if (token == null) {
        console.warn("Authentication failed: No token provided.");
        return res.sendStatus(401); // 401 Unauthorized
    }

    // Verify the signature first -- cheap, and it rejects garbage before we
    // touch the database.
    let payload;
    try {
        payload = jwt.verify(token, process.env.JWT_SECRET);
    } catch (err) {
        console.warn('Authentication failed: JWT verification failed.', err.message);
        return res.sendStatus(403);
    }

    // Then confirm the token has not been revoked, and read role from the
    // DATABASE rather than trusting the copy baked into the token.
    //
    // A stateless JWT alone cannot answer "is this session still valid?". Before
    // this check, a leaked token was good for its full 24 hours with no way to
    // kill it, a password change left every existing session alive, and a
    // demoted user kept their old role until the token expired. The cost is one
    // indexed lookup of two fields per request, which is the right trade when
    // the alternative is having no revocation at all.
    try {
        const User = mongoose.model('User');
        const user = await User.findById(payload._id)
            .select('_id role name email tokenVersion')
            .lean();

        if (!user) {
            // Deleted account, or a token minted against a different database.
            console.warn(`Authentication failed: no user for token subject ${payload._id}`);
            return res.sendStatus(401);
        }

        // A token with no `tv` predates this mechanism. Reject it rather than
        // grandfathering it in: accepting version-less tokens would leave exactly
        // the hole this closes, and the only cost is that everyone signs in again
        // once.
        const tv = typeof payload.tv === 'number' ? payload.tv : null;
        if (tv === null || tv !== (user.tokenVersion || 0)) {
            console.warn(`Authentication failed: token version ${tv} != `
                       + `${user.tokenVersion || 0} for user ${payload._id}`);
            return res.status(401).json({
                message: 'This session is no longer valid. Please sign in again.',
                code: 'token_revoked',
            });
        }

        req.user = {
            _id: user._id,
            role: user.role,          // authoritative, not the token's copy
            name: user.name,
            email: user.email,
        };
        return next();
    } catch (err) {
        console.error('Authentication failed: user lookup error:', err.message);
        return res.sendStatus(500);
    }
};


// --- Authentication Routes ---

// @route POST /api/auth/register
// @desc Register a new user
// @access Public
router.post('/register', async (req, res) => {
    // Extract user data from the request body
    // memberSince is deliberately NOT read from the body. It was, and a caller
    // could post memberSince:'2015-01-01' to appear as a five-year member on a
    // brand-new account -- tenure is exactly the kind of signal an investor uses
    // to decide whether to trust a farmer, so it must come from the server. The
    // pre-save hook in User.js sets it to createdAt.
    const { name, email, role, password, phone } = req.body;

    // 'admin' is a valid value on the User schema because provenance review needs
    // it, but it must never be obtainable by asking. Without this check, adding
    // the role to the enum would have made every reviewer privilege available to
    // anyone who could POST a registration.
    if (role === 'admin') {
        console.warn(`Registration refused: attempt to self-assign admin (${email}).`);
        return res.status(403).json({
            message: 'That role cannot be requested.',
            code: 'role_not_assignable',
        });
    }

    // Basic input validation
    if (!name || !email || !role || !password) {
        return res.status(400).json({ message: 'Please enter all required fields (name, email, role, password).' });
    }
    if (password.length < 6) { // Minimum password length validation (matches schema)
         return res.status(400).json({ message: 'Password must be at least 6 characters long.' });
    }
     // Validate role against enum
     if (!['farmer', 'buyer', 'investor'].includes(role)) {
          return res.status(400).json({ message: 'Invalid role specified. Must be farmer, buyer, or investor.' });
     }
    // Basic email format check (schema also validates, but frontend/early check is good)
    if (!/\S+@\S+\.\S+/.test(email)) {
        return res.status(400).json({ message: 'Please use a valid email address.' });
    }


    if (typeof email !== 'string' || typeof password !== 'string'
        || (name !== undefined && typeof name !== 'string')) {
        return res.status(400).json({ message: 'Name, email and password must be text.' });
    }

    try {
        // Check if a user with the given email already exists
        // String() is load-bearing. express.json() will happily parse
        // {"email":{"$ne":null}} into an OBJECT, and Mongoose passes an object
        // through as a query operator -- so an unauthenticated caller could match
        // an arbitrary user. Coercing to a string makes the value a literal.
        let user = await User.findOne({ email: String(email) });
        if (user) {
            // If email exists, return a conflict error
            return res.status(409).json({ message: 'Email already registered.' }); // 409 Conflict
        }

        // Create a new user instance using the Mongoose model
        user = new User({
            name,
            email,
            role,
            password, // The password will be hashed by the pre-save hook in the User model
             phone,       // Include optional fields
             // memberSince: set by the model from createdAt; never client-supplied.
            // createdAt will default in the schema
            // authMethod will default in the schema
        });

        // Save the new user document to the database
        const savedUser = await user.save();

        // Generate a JSON Web Token (JWT) for the newly registered user
        // The payload includes essential user info (_id and role) for subsequent authentication checks
         const token = jwt.sign(
             // `tv` is the tokenVersion this token is bound to. authenticateToken
             // rejects the token once the stored version moves past it, which is
             // what makes "sign out everywhere" possible at all.
             { _id: savedUser._id.toString(), role: savedUser.role,
               tv: savedUser.tokenVersion || 0 },
             process.env.JWT_SECRET, // The secret key for signing (from .env)
             { expiresIn: '24h' } // Token expiration time (e.g., 24 hours)
         );


        // Send a success response back to the frontend
        res.status(201).json({ // 201 Created
            message: 'User registered successfully',
            user: savedUser.getPublicProfile(), // Send back the user's public profile data
            token // Send the generated token
        });

    } catch (err) {
        console.error('Signup error:', err);
        // Handle Mongoose validation errors or other database errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message }); // Send validation error message
         }
         if (err.code === 11000) { // Duplicate key error (e.g., if email unique index fails for some reason)
              return res.status(409).json({ message: 'Email already registered.' });
         }
        res.status(500).json({ message: 'Server error during registration' }); // 500 Internal Server Error
    }
});

// @route POST /api/auth/login
// @desc Authenticate user & get token
// @access Public
router.post('/login', async (req, res) => {
    // Extract login credentials from the request body
    const { email, password } = req.body;

    // Basic input validation. The typeof checks are security-relevant, not
    // cosmetic: express.json() parses {"email":{"$gt":""}} into an object, and
    // Mongoose treats an object as a query OPERATOR -- so without this an
    // unauthenticated caller could select the first user in the collection and
    // only then have their password checked.
    if (!email || !password) {
        return res.status(400).json({ message: 'Please enter email and password.' });
    }
    if (typeof email !== 'string' || typeof password !== 'string') {
        return res.status(400).json({ message: 'Email and password must be text.' });
    }

    try {
        // Find the user by email in the database
        // See the note in register: an object here becomes a query operator.
        // {"email":{"$gt":""},"password":"anything"} would otherwise select the
        // first user in the collection and only then check the password.
        const user = await User.findOne({ email: String(email) });

        // If user is not found, return an error
        if (!user) {
            // Use a generic error message for security (don't reveal if email exists but password is wrong)
            return res.status(400).json({ message: 'Invalid credentials.' }); // 400 Bad Request
        }

        // Compare the provided password with the hashed password stored in the database
        // The comparePassword method is defined on the User schema (models/User.js)
        const isMatch = await user.comparePassword(password);

        // If passwords do not match, return an error
        if (!isMatch) {
            // Use a generic error message for security
            return res.status(400).json({ message: 'Invalid credentials.' }); // 400 Bad Request
        }

        // If email and password match, generate a JWT token for the authenticated user
         const token = jwt.sign(
             { _id: user._id.toString(), role: user.role,
               tv: user.tokenVersion || 0 },
             process.env.JWT_SECRET, // The secret key for signing (from .env)
             { expiresIn: '24h' } // Token expiration time
         );

        // Send a success response back to the frontend with user data and token
        res.json({ // Default status is 200 OK
            message: 'Logged in successfully',
            user: user.getPublicProfile(), // Send back the user's public profile data
            token // Send the generated token
        });

    } catch (err) {
        console.error('Login error:', err);
        res.status(500).json({ message: 'Server error during login' }); // 500 Internal Server Error
    }
});

// @route GET /api/auth/me
// @desc Get the currently logged-in user's profile
// @access Private (Requires a valid JWT token)
// @route POST /api/auth/change-password
// @desc  Change your own password, and end every other session.
// @access Private
//
// There was no way to change a password at all. Combined with seeded demo
// accounts whose credentials are published in seed.js, that meant anyone who
// read the repo could sign in as a demo user and the owner had no remedy short
// of editing the database.
//
// Changing a password MUST end other sessions. Otherwise the main reason people
// change one -- "somebody else has my password" -- is not addressed: the
// attacker's existing token keeps working. Bumping tokenVersion invalidates
// every issued token, including this caller's, so a fresh one is returned.
router.post('/change-password', authenticateToken, async (req, res) => {
    const { currentPassword, newPassword } = req.body || {};

    if (!currentPassword || !newPassword) {
        return res.status(400).json({
            message: 'Both currentPassword and newPassword are required.',
        });
    }
    if (typeof newPassword !== 'string' || newPassword.length < 8) {
        return res.status(400).json({
            message: 'The new password must be at least 8 characters.',
            code: 'password_too_short',
        });
    }
    if (newPassword === currentPassword) {
        return res.status(400).json({
            message: 'The new password must be different from the current one.',
            code: 'password_unchanged',
        });
    }

    try {
        const User = mongoose.model('User');
        // '+password' is belt-and-braces: the field is not select:false today,
        // but asking explicitly means this route keeps working if it becomes so.
        const user = await User.findById(req.user._id).select('+password');
        if (!user) {
            return res.status(404).json({ message: 'Account not found.' });
        }

        const ok = await bcrypt.compare(currentPassword, user.password);
        if (!ok) {
            // Deliberately not distinguished from any other failure in the
            // message, and logged, because this is the shape a password-guessing
            // attempt takes against an authenticated session.
            console.warn(`Password change rejected for ${req.user._id}: wrong current password`);
            return res.status(403).json({
                message: 'The current password is not correct.',
                code: 'wrong_password',
            });
        }

        // The pre-save hook hashes `password` when it is modified, so assign the
        // plaintext and let the model do it -- hashing here as well would
        // double-hash and lock the account out.
        user.password = newPassword;
        user.tokenVersion = (user.tokenVersion || 0) + 1;
        await user.save();

        const token = jwt.sign(
            { _id: user._id.toString(), role: user.role, tv: user.tokenVersion },
            process.env.JWT_SECRET,
            { expiresIn: '24h' },
        );

        console.log(`Password changed for ${user._id}; all other sessions ended.`);
        return res.json({
            message: 'Password changed. You have been signed out everywhere else.',
            token,
        });
    } catch (err) {
        console.error('Password change failed:', err.message);
        return res.status(500).json({ message: 'Could not change the password.' });
    }
});


// @route POST /api/auth/sign-out-everywhere
// @desc  Invalidate every token issued for this account, including this one.
// @access Private
//
// The remedy for "I think my token leaked" that did not exist. Cheap: one
// increment, and every outstanding token fails its version check on the next
// request.
router.post('/sign-out-everywhere', authenticateToken, async (req, res) => {
    try {
        const User = mongoose.model('User');
        const updated = await User.findByIdAndUpdate(
            req.user._id,
            { $inc: { tokenVersion: 1 } },
            { new: true },
        ).select('tokenVersion').lean();

        if (!updated) {
            return res.status(404).json({ message: 'Account not found.' });
        }

        console.log(`All sessions ended for ${req.user._id} `
                  + `(tokenVersion now ${updated.tokenVersion}).`);
        return res.json({
            message: 'Signed out everywhere. Every device will need to sign in again.',
        });
    } catch (err) {
        console.error('Sign-out-everywhere failed:', err.message);
        return res.status(500).json({ message: 'Could not end the sessions.' });
    }
});


router.get('/me', authenticateToken, async (req, res) => {
    // This route uses the authenticateToken middleware.
    // If the middleware successfully verified the token, req.user will contain the payload.

    try {
        // Find the user in the database using the _id from the token payload (req.user._id).
        // Select out the password field to avoid sending it.
        const user = await User.findById(req.user._id).select('-password');

        // If the user is not found (e.g., user deleted after token issued - rare), return 404.
        if (!user) {
            return res.status(404).json({ message: 'User not found.' }); // 404 Not Found
        }

        // Return the user's public profile data
        res.json(user.getPublicProfile()); // Default status is 200 OK

    } catch (err) {
        console.error('GET /api/auth/me error:', err);
         // Handle potential errors (e.g., invalid _id format if payload was somehow tampered)
         if (err.kind === 'ObjectId') {
             return res.status(400).json({ message: 'Invalid User ID format in token.' });
         }
        res.status(500).json({ message: 'Server error fetching user data.' }); // 500 Internal Server Error
    }
});


// --- Export the router and middleware ---

// Export the configured router so it can be used by server.js
// Export the authenticateToken middleware so it can be used by other route files
module.exports = { router, authenticateToken }; 
