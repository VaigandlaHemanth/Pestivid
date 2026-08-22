// --- Backend Routes: transactions.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose (needed for ObjectId validation)
const Transaction = mongoose.model('Transaction'); // Get the Transaction Mongoose model
const { authenticateToken } = require('./auth'); // Import the authentication middleware


// --- Transaction Routes ---

// @route GET /api/transactions/user/:userId
// @desc Get all transaction records associated with a specific user ID
// @access Private (Requires authentication. User must typically fetch only their own transactions.)
router.get('/user/:userId', authenticateToken, async (req, res) => {
    const userId = req.params.userId; // Get the user ID from the URL parameter
    const authenticatedUserId = req.user._id; // Authenticated user's ID from the token

    // Validate the userId format from the URL parameter
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the user whose transactions are being requested.
    // Or allow an admin user to view any user's transactions.
    if (authenticatedUserId.toString() !== userId.toString()) {
         console.warn(`Authorization failed: User ${authenticatedUserId} attempted to view transactions for user ${userId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own transactions." }); // 403 Forbidden
    }

    try {
        // Find all transaction documents where the 'userId' field matches the requested userId.
        // Note that the 'userId' field in the Transaction schema references the User model.
        const transactions = await Transaction.find({ userId: userId })
                                             // Optional: Populate related documents (project, listing) if frontend needs details beyond just the ID
                                             // .populate('projectId', 'title crop')
                                             // .populate('listingId', 'crop location')
                                             .sort({ date: -1 }); // Sort by date, newest first

        // Map to format for frontend (e.g., ensure _id is string, format dates)
         const formattedTransactions = transactions.map(tx => ({
             _id: tx._id.toString(), // Convert ObjectId to string
             userId: tx.userId.toString(), // User ID string
             txHash: tx.txHash,
             type: tx.type,
             amount: tx.amount,
             date: tx.date ? tx.date.toISOString() : null, // ISO string date
             projectId: tx.projectId ? tx.projectId.toString() : null, // Project ID string (if linked)
             listingId: tx.listingId ? tx.listingId.toString() : null, // Listing ID string (if linked)

             // If you populated above, you could include details here:
             // projectTitle: tx.projectId ? tx.projectId.title : undefined,
             // listingCrop: tx.listingId ? tx.listingId.crop : undefined,
         }));


        // Send the list of transactions for this user
        res.json(formattedTransactions); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/transactions/user/${userId} error:`, err);
        res.status(500).json({ message: 'Server error fetching user transactions.' }); // 500 Internal Server Error
    }
});


// Note: Transaction creation happens within other routes where the event occurs
// (e.g., POST /api/investments creates an 'investment' transaction,
// POST /api/purchases creates 'purchase' and 'sale' transactions,
// PUT /api/investments/:id/progress might create a 'payout' transaction).
// There is typically no direct API endpoint for creating a general transaction from the frontend.


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;