// --- Mongoose Model: Message.js ---

const mongoose = require('mongoose'); // Import Mongoose library

// Define the schema for the Message model
const messageSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key for each message document.

    conversationId: { // Reference to the Conversation document this message belongs to
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Conversation', // Links to the 'Conversation' model
        required: [true, 'Conversation ID is required']
    },
    sender: { // Reference to the User document of the sender
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Sender ID is required']
    },
    receiver: { // Reference to the User document of the direct receiver (useful for 1-on-1 chats and read status tracking)
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Receiver ID is required'] // Useful even in group chats for private messages or read receipts
    },
    text: { // The content of the message
        type: String,
        required: [true, 'Message text is required'],
        trim: true,
        maxlength: [5000, 'Message text cannot exceed 5000 characters'] // Add a max length
    },
    timestamp: { // Date and time the message was sent
        type: Date,
        default: Date.now // Default to the current date and time
    },
    read: { // Status indicating if the receiver has read the message
        type: Boolean,
        default: false // Default to unread
    },

    // Add other fields as needed, e.g.:
    // status: { type: String, enum: ['sent', 'delivered', 'read'], default: 'sent' }, // Message delivery status
    // attachments: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Attachment' }], // For file attachments
    // deletedBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }], // To track who "deleted" the message (soft delete)
    // type: { type: String, enum: ['text', 'image', 'system'], default: 'text' }, // Message type
});

// Optional: Add an index on conversationId and timestamp for efficient querying and sorting
messageSchema.index({ conversationId: 1, timestamp: 1 });


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'messages' (lowercase, plural) for this model.
const Message = mongoose.model('Message', messageSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Message;