// --- Mongoose Model: Conversation.js ---

const mongoose = require('mongoose'); // Import Mongoose library

// Define the schema for the Conversation model
const conversationSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key for each conversation document.

    participants: { // An array of User IDs who are part of this conversation
        type: [{
            type: mongoose.Schema.Types.ObjectId, // Each element is a reference to a User document
            ref: 'User', // Links to the 'User' model
            required: true // Each participant ID in the array must be present
        }],
        required: [true, 'Participants are required'],
        // Optional: You could add validation to ensure there are at least 2 participants
        // Optional: For 1-on-1 chats, you could add a unique index on a sorted array of participants
        // Example unique index for 1-on-1 chats (requires sorting participant IDs before saving):
        // index({ participants: 1 }, { unique: true, partialFilterExpression: { participants: { $size: 2 } } });
        // The route handler's logic for finding/creating conversations handles this check explicitly for demo.
    },
    lastMessageSnippet: { // A short snippet of the text of the most recent message
        type: String,
        trim: true
        // Not required, as a conversation might have no messages yet
    },
    lastMessageTimestamp: { // The timestamp of the most recent message
        type: Date,
        default: Date.now // Default to the creation time, updated when a new message is sent
        // Not strictly required, but useful for sorting conversations by recency
    },
    createdAt: { // Timestamp when the conversation document was created
        type: Date,
        default: Date.now
    },

    // Add other fields as needed, e.g.:
    // isGroupChat: { type: Boolean, default: false }, // To distinguish 1-on-1 from group chats
    // name: String, // Name for group chats
    // createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }, // User who created a group chat
    // lastReadBy: { type: Map, of: Date }, // To track last read timestamp per participant
});

// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'conversations' (lowercase, plural) for this model.
const Conversation = mongoose.model('Conversation', conversationSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Conversation;