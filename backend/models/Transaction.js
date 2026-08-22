// --- Mongoose Model: Transaction.js ---

const mongoose = require('mongoose'); // Import Mongoose library

// Define the schema for the Transaction model
const transactionSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key for each transaction document.

    userId: { // Reference to the User document of the participant in this transaction (e.g., the buyer, the investor, the farmer receiving payment)
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'User ID is required for transaction']
    },
    txHash: { // Simulated blockchain transaction hash
        type: String,
        required: [true, 'Transaction hash is required'], // Every transaction record needs a unique hash
        unique: true, // Ensure each transaction has a unique hash
        trim: true
    },
    type: { // Type of transaction
        type: String,
        required: [true, 'Transaction type is required'],
        enum: {
            values: ['investment', 'payout', 'purchase', 'sale'], // Possible types of transactions in the demo flow
            // 'investment': User (investor) sent SOL to a project
            // 'payout': User (investor) received SOL from a harvested project
            // 'purchase': User (buyer) spent SOL on a listing
            // 'sale': User (farmer) received SOL from a listing purchase
            message: 'Invalid transaction type.'
        }
    },
    amount: { // Amount of SOL involved in the transaction
        type: Number,
        required: [true, 'Transaction amount is required'],
        // Can be positive (for payout, sale) or negative (for investment, purchase) depending on how you record it,
        // or always positive and let the 'type' define inflow/outflow.
        // Let's assume positive amount and 'type' dictates direction for simplicity.
        min: [0, 'Transaction amount cannot be negative'] // Amount must be zero or positive
    },
    date: { // Date and time the transaction occurred (simulated)
        type: Date,
        default: Date.now // Default to the current date and time
    },
    // Optional references linking the transaction back to related documents for context:
    projectId: { // Link to the FundingRequest document if the type is 'investment' or 'payout'
        type: mongoose.Schema.Types.ObjectId,
        ref: 'FundingRequest',
        sparse: true // Allow null values, as not all transactions link to a project
    },
    listingId: { // Link to the Listing document if the type is 'purchase' or 'sale'
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Listing',
        sparse: true // Allow null values
    },

    // Add other relevant fields if needed, e.g.:
    // counterpartyUserId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', sparse: true }, // The other user involved (farmer for purchase, buyer for sale, farmer for investment, investor for payout)
    // fee: Number, // Transaction fee (simulated)
    // status: { type: String, enum: ['pending', 'confirmed', 'failed'], default: 'confirmed' } // Simulated blockchain confirmation status
    // tokenType: String, // e.g., 'SOL', 'USDC', custom token
});


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'transactions' (lowercase, plural) for this model.
const Transaction = mongoose.model('Transaction', transactionSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Transaction;