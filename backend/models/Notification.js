// --- Mongoose Model: Notification.js ---

const mongoose = require('mongoose'); // Import Mongoose library

// Define the schema for the Notification model
const notificationSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key.

    recipient: { // Reference to the User document who should receive this notification. Null if global.
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        sparse: true // Allows null values, as global notifications don't have a single recipient
    },
    global: { // Boolean flag indicating if this is a global notification for all users
        type: Boolean,
        default: false
    },
    type: { // The type of notification (e.g., 'info', 'success', 'message', 'listing', 'investment', 'update', 'purchase', 'payout')
        type: String,
        required: [true, 'Notification type is required'],
        enum: {
            values: ['info', 'success', 'warning', 'error', 'message', 'listing', 'investment', 'funding', 'update', 'purchase', 'sale', 'payout'], // Expanded types
            message: 'Invalid notification type.'
        }
    },
    message: { // The text content of the notification
        type: String,
        required: [true, 'Notification message is required'],
        trim: true,
        maxlength: [500, 'Notification message cannot exceed 500 characters'] // Add a max length
    },
    timestamp: { // Date and time the notification was created
        type: Date,
        default: Date.now // Default to the current date and time
    },
    // Per-user state for `global` notifications.
    //
    // A global notification is ONE document shared by every user, so any field
    // meant to describe "this user's relationship to it" has to be a set of user
    // ids, not a scalar. Two bugs came from getting that wrong: deleting a global
    // removed it for everyone (fixed with dismissedBy), and marking one read set
    // the shared `read` flag, so one user clearing a broadcast cleared it off
    // every other user's list platform-wide.
    dismissedBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    readBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],

    // Only meaningful for a DIRECT notification (recipient set, global false).
    // For a global one, read state lives in readBy and this stays false.
    read: {
        type: Boolean,
        default: false
    },

    // Optional fields to link the notification to a specific item (e.g., a listing, project, message)
    itemId: { // The ID of the related item
        type: mongoose.Schema.Types.ObjectId,
        sparse: true // Allow null values
    },
    itemType: { // The type of the related item (e.g., 'Listing', 'FundingRequest', 'Message')
        type: String,
        sparse: true // Allow null values
        // Consider using enum for stricter validation if needed
    },

    // Add other fields as needed, e.g.:
    // actionLink: String, // A URL path within the app the notification could link to
});

// Add an index on recipient and timestamp for efficient querying for a user's notifications
notificationSchema.index({ recipient: 1, timestamp: -1 });
// Add an index for global notifications
notificationSchema.index({ global: 1, timestamp: -1 });


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'notifications' (lowercase, plural) for this model.
const Notification = mongoose.model('Notification', notificationSchema);

// Export the model so it can be used in other files (like route handlers)
module.exports = Notification;