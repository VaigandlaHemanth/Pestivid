// --- Backend Routes: users.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose (needed to access models and ObjectId validation)
const User = mongoose.model('User');        // Get the User Mongoose model
const { authenticateToken } = require('./auth'); // Import the authentication middleware


// --- Public User Routes ---

// @route GET /api/users/public
// @desc Get a list of public user profiles (IDs, names, roles, etc.)
// @access Public (Accessible to anyone, logged in or not)
router.get('/public', authenticateToken, async (req, res) => {
    try {
        // Find all users in the database
        // Select only fields that are appropriate for public viewing.
        // Exclude sensitive fields like 'password'. Consider if 'email' or 'phone' should be public.
        const limit  = Math.min(parseInt(req.query.limit, 10) || 50, 100);
        const offset = Math.max(parseInt(req.query.offset, 10) || 0, 0);
        const users = await User.find()
                               .select('_id name role memberSince createdAt')
                               .limit(limit)
                               .skip(offset); // Select safe fields, bounded

        // Map the Mongoose documents to plain objects, potentially adding derived fields
         const publicUsers = users.map(user => ({
             _id: user._id.toString(), // Convert ObjectId to string
             name: user.name,
             role: user.role,
             memberSince: user.memberSince ? user.memberSince.toISOString() : null, // ISO string date
             createdAt: user.createdAt ? user.createdAt.toISOString() : null, // ISO string date
             displayIdentifier: user.name.split(' ')[0] || user._id.toString().substring(0,6) + '...', // Simple display name logic from frontend
             // Add email, phone here if you want them public for everyone:
             // email: user.email,
             // phone: user.phone,
         }));


        // Send the list of public user profiles
        res.json(publicUsers); // Default status is 200 OK

    } catch (err) {
        console.error('GET /api/users/public error:', err);
        res.status(500).json({ message: 'Server error fetching public users.' }); // 500 Internal Server Error
    }
});

// @route GET /api/users/:id/public
// @desc Get a specific user's public profile details by ID
// @access Public (Accessible to anyone, logged in or not)
router.get('/:id/public', authenticateToken, async (req, res) => {
    const userId = req.params.id; // Get the user ID from the URL parameter

    // Validate the ID format before querying MongoDB
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    try {
        // Find the user by ID
        // Select fields appropriate for public viewing, potentially more than the list view.
        const user = await User.findById(userId)
                               .select('_id name role memberSince createdAt'); // email/phone deliberately excluded

        // If user is not found, return 404
        if (!user) {
            return res.status(404).json({ message: 'User not found.' }); // 404 Not Found
        }

        // Send the user's public profile data
        res.json({ // Create a clean object to send back
             _id: user._id.toString(),
             name: user.name,
             role: user.role,
             memberSince: user.memberSince ? user.memberSince.toISOString() : null,
             createdAt: user.createdAt ? user.createdAt.toISOString() : null,
             displayIdentifier: user.name.split(' ')[0] || user._id.toString().substring(0,6) + '...',
         }); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/users/${userId}/public error:`, err);
        res.status(500).json({ message: 'Server error fetching user profile.' }); // 500 Internal Server Error
    }
});


// --- Private User Routes (Require Authentication) ---

// @route GET /api/users/me
// @desc Get the currently logged-in user's profile (This route is already in auth.js /api/auth/me)
// @access Private
// router.get('/me', authenticateToken, async (req, res) => { ... } )
// ^ You don't need to duplicate this here if it's in auth.js

// @route PUT /api/users/:id/profile
// @desc Update the currently logged-in user's own profile
// @access Private (Requires authentication. User must own the profile.)
// @route PUT /api/users/me/notices
// @desc  Switch the two broadcasts on or off for the signed-in account.
// Body: { funding?: boolean, listing?: boolean }. Direct notices cannot be
// switched off here: they are about this person's own money and messages.
router.put('/me/notices', authenticateToken, async (req, res) => {
    const set = {};
    for (const k of ['funding', 'listing']) {
        if (req.body[k] === undefined) continue;
        if (typeof req.body[k] !== 'boolean') {
            return res.status(400).json({ message: `${k} must be true or false.` });
        }
        set[`notices.${k}`] = req.body[k];
    }
    if (!Object.keys(set).length) {
        return res.status(400).json({ message: 'Nothing to change: send funding and/or listing.' });
    }
    try {
        const user = await User.findByIdAndUpdate(req.user._id, { $set: set }, { new: true });
        if (!user) return res.status(404).json({ message: 'User not found.' });
        res.json(user.getPublicProfile().notices);
    } catch (err) {
        console.error('PUT /api/users/me/notices error:', err);
        res.status(500).json({ message: 'Server error saving the setting.' });
    }
});

router.put('/:id/profile', authenticateToken, async (req, res) => {
    const userId = req.params.id;   // Get the user ID from the URL parameter
    const updates = req.body;       // Get the update data from the request body

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is trying to update THEIR OWN profile.
    // req.user._id comes from the token payload set by authenticateToken.
    if (req.user._id.toString() !== userId.toString()) {
        console.warn(`Authorization failed: User ${req.user._id} attempted to update profile for user ${userId}`);
        return res.status(403).json({ message: "Forbidden: You can only update your own profile." }); // 403 Forbidden
    }

    // Define which fields users are allowed to update via this route.
    // Prevent users from changing sensitive fields like role, password, _id, etc.
    const allowedUpdates = ['name', 'email', 'phone']; // Add 'bio' or other profile fields if you add them to the schema

    // Create an object containing only the allowed updates from the request body.
    const filteredUpdates = {};
    Object.keys(updates).forEach(key => {
        if (allowedUpdates.includes(key)) {
            filteredUpdates[key] = updates[key];
        }
    });

    // If no valid update fields were provided, maybe return an error or just do nothing.
     if (Object.keys(filteredUpdates).length === 0) {
         return res.status(400).json({ message: 'No valid fields provided for update.' });
     }


    try {
        // Find the user by ID and update the filtered fields.
        // { new: true } returns the updated document.
        // { runValidators: true } runs Mongoose schema validators on the updated fields (e.g., email format, uniqueness).
        const updatedUser = await User.findByIdAndUpdate(
            userId,
            { $set: filteredUpdates }, // Use $set to apply partial updates
            { new: true, runValidators: true }
        ).select('-password'); // Exclude password from the result

        // If user is not found (shouldn't happen if token is valid but good check), return 404.
        if (!updatedUser) {
             // This scenario is unlikely if authenticateToken passed, but robust code includes it.
            return res.status(404).json({ message: 'User not found.' }); // 404 Not Found
        }

        // Send a success response with the updated public profile data
        res.json({
            message: 'Profile updated successfully',
            user: updatedUser.getPublicProfile() // Send back the updated public profile
        }); // Default status is 200 OK

    } catch (err) {
        console.error(`PUT /api/users/${userId}/profile error:`, err);
        // Handle potential Mongoose validation errors (e.g., duplicate email on update)
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
         if (err.code === 11000) { // Duplicate key error (e.g., if email update causes conflict)
              return res.status(409).json({ message: 'Email already in use.' });
         }
        res.status(500).json({ message: 'Server error updating profile.' }); // 500 Internal Server Error
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;