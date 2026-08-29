// --- Backend Routes: messaging.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose
const Conversation = mongoose.model('Conversation'); // Get Conversation model
const Message = mongoose.model('Message');      // Get Message model
const User = mongoose.model('User');            // Get User model (to validate participants)
const { authenticateToken } = require('./auth'); // Import the authentication middleware
const Notification = mongoose.model('Notification'); // Get Notification model for new message alerts


// --- Test Route ---
// @route GET /api/messaging/test
// @desc Test messaging API connectivity
// @access Public
router.get('/test', (req, res) => {
    console.log('Messaging API test endpoint hit');
    res.json({
        message: 'Messaging API is working!',
        timestamp: new Date().toISOString(),
        routes: [
            'GET /api/messaging/test',
            'GET /api/messaging/conversations/:userId',
            'POST /api/messaging/conversations',
            'GET /api/messaging/conversations/:conversationId/messages',
            'POST /api/messaging/conversations/:conversationId/messages',
            'GET /api/messaging/messages/:userId'
        ]
    });
});

// --- Conversation and Message Routes ---

// @route GET /api/messaging/conversations/:userId
// @desc Get all conversations that a specific user is a participant in.
// @access Private (Requires authentication. User must typically fetch only their own conversations.)
router.get('/conversations/:userId', authenticateToken, async (req, res) => {
    const userId = req.params.userId; // Get the user ID from the URL parameter
    const authenticatedUserId = req.user._id; // Authenticated user's ID

    // Validate the userId format
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    // Authorization: the authenticated user must BE the user whose conversations
    // are being requested. Not "or an admin", which is what this comment used to
    // say and what the code below has never done -- there is no branch for it.
    // A comment promising a back door that does not exist is the kind that gets
    // one built.
    if (authenticatedUserId.toString() !== userId.toString()) {
         console.warn(`Authorization failed: User ${authenticatedUserId} attempted to view conversations for user ${userId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own conversations." }); // 403 Forbidden
    }

    try {
        // Find all conversations where the 'participants' array contains the specified userId.
        // Sort by the last message timestamp to show the most recent conversations first.
        const conversations = await Conversation.find({ participants: userId })
                                               .populate('participants', 'name role')
                                               .sort({ lastMessageTimestamp: -1 }); // Newest first

        // Map to format for frontend (e.g., ensure _id is string)
         const formattedConversations = conversations.map(conv => {
             // The old comment here said the frontend would look the other
             // participant's name up "using the users list fetched separately".
             // There is no such route a farmer is allowed to call, so every
             // thread was headed "Them" and the messages list said "A message"
             // instead of who sent it. The name comes with the conversation.
             const other = (conv.participants || [])
                 .find(p => p && p._id && p._id.toString() !== userId.toString());
             return {
                 _id: conv._id.toString(), // Conversation ID string
                 participants: (conv.participants || []).map(p => (p && p._id ? p._id.toString() : String(p))),
                 otherName: other ? other.name : null,
                 otherRole: other ? other.role : null,
                 lastMessageSnippet: conv.lastMessageSnippet,
                 lastMessageTimestamp: conv.lastMessageTimestamp ? conv.lastMessageTimestamp.toISOString() : null, // ISO string date
                 createdAt: conv.createdAt ? conv.createdAt.toISOString() : null, // ISO string date
             };
         });


        // Send the list of conversations for this user
        res.json(formattedConversations); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/messaging/conversations/${userId} error:`, err);
        res.status(500).json({ message: 'Server error fetching conversations.' }); // 500 Internal Server Error
    }
});

// @route POST /api/messaging/conversations
// @desc Create a new 1-on-1 conversation or return an existing one between the logged-in user and a target user.
// @access Private (Requires authentication)
router.post('/conversations', authenticateToken, async (req, res) => {
    const userId = req.user._id;         // The authenticated user (initiator)
    const { targetUserId } = req.body; // The ID of the other user

    // Basic input validation
    if (!targetUserId) {
        return res.status(400).json({ message: 'Missing targetUserId in request body.' }); // 400 Bad Request
    }
     // Validate targetUserId format
     if (!mongoose.Types.ObjectId.isValid(targetUserId)) {
         return res.status(400).json({ message: 'Invalid target User ID format.' }); // 400 Bad Request
     }
    // Prevent creating a conversation with self
    if (userId.toString() === targetUserId.toString()) {
        return res.status(400).json({ message: 'Cannot create a conversation with yourself.' }); // 400 Bad Request
    }

    try {
        // Verify that the target user actually exists
        const targetUserExists = await User.exists({ _id: targetUserId });
        if (!targetUserExists) {
             console.warn(`Attempted to create conversation with non-existent user ID: ${targetUserId}`);
            return res.status(404).json({ message: 'Target user not found.' }); // 404 Not Found
        }

        // For 1-on-1 conversations, we sort the participant IDs alphabetically (as strings)
        // to ensure a unique identifier for the pair, regardless of who initiates.
        const participantsSorted = [userId.toString(), targetUserId.toString()].sort();

        // Check if a conversation already exists between these two users
        let conversation = await Conversation.findOne({
            participants: {
                $size: 2, // Ensure it's a 2-person conversation
                $all: participantsSorted // Ensure both sorted participant IDs are present
            }
        });

        // If a conversation does NOT exist, create a new one
        if (!conversation) {
             console.log(`Creating new conversation between ${userId} and ${targetUserId}`);
            conversation = new Conversation({
                participants: [userId, new mongoose.Types.ObjectId(targetUserId)], // Store as ObjectIds
                lastMessageSnippet: 'Conversation started.', // Initial snippet
                lastMessageTimestamp: new Date(),
                createdAt: new Date(),
            });
            const savedConversation = await conversation.save();
             // Return the newly created conversation
             res.status(201).json(savedConversation); // 201 Created

        } else {
             console.log(`Conversation already exists between ${userId} and ${targetUserId} (ID: ${conversation._id}). Returning existing.`);
            // If conversation already exists, return the existing one
            res.json(conversation); // Default status is 200 OK
        }

    } catch (err) {
        console.error('POST /api/messaging/conversations error:', err);
         // Handle potential Mongoose validation errors or other errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
        res.status(500).json({ message: 'Server error creating or getting conversation.' }); // 500 Internal Server Error
    }
});


// @route GET /api/messaging/conversations/:conversationId/messages
// @desc Get all messages for a specific conversation.
// @access Private (Requires authentication. User must be a participant in the conversation.)
router.get('/conversations/:conversationId/messages', authenticateToken, async (req, res) => {
    const conversationId = req.params.conversationId; // Get conversation ID from URL
    const userId = req.user._id;                     // Authenticated user's ID

    // Validate the conversationId format
    if (!mongoose.Types.ObjectId.isValid(conversationId)) {
        return res.status(400).json({ message: 'Invalid Conversation ID format.' }); // 400 Bad Request
    }

    try {
        // Find the conversation by ID
        const conversation = await Conversation.findById(conversationId);

        // If conversation not found, return 404
        if (!conversation) {
            return res.status(404).json({ message: 'Conversation not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is one of the participants in the conversation.
        // Check if userId exists in the conversation's participants array.
        if (!conversation.participants.map(p => p.toString()).includes(userId.toString())) {
             console.warn(`Authorization failed: User ${userId} attempted to access messages for conversation ${conversationId} they are not in.`);
            return res.status(403).json({ message: 'Forbidden: You are not a participant in this conversation.' }); // 403 Forbidden
        }

        // Find all messages linked to this conversation ID
        // Sort messages by timestamp in ascending order (oldest first)
        const messages = await Message.find({ conversationId: conversationId })
                                     .populate('sender', 'name role')
                                     .sort({ timestamp: 1 }); // Ascending timestamp

        // Map to format for frontend (e.g., ensure IDs are strings)
         const formattedMessages = messages.map(msg => ({
             _id: msg._id.toString(), // Message ID string
             conversationId: msg.conversationId.toString(), // Conversation ID string
             // sender stays an id string so existing comparisons keep working;
             // the name rides alongside it rather than replacing it.
             sender: (msg.sender && msg.sender._id ? msg.sender._id : msg.sender).toString(),
             senderName: msg.sender && msg.sender.name ? msg.sender.name : null,
             receiver: msg.receiver.toString(), // Receiver User ID string
             text: msg.text,
             timestamp: msg.timestamp ? msg.timestamp.toISOString() : null, // ISO string date
             read: msg.read,
         }));


        // Send the list of messages for this conversation
        res.json(formattedMessages); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/messaging/conversations/${conversationId}/messages error:`, err);
        res.status(500).json({ message: 'Server error fetching messages.' }); // 500 Internal Server Error
    }
});


// @route GET /api/messaging/messages/:userId
// @desc Get all messages for a specific user across all their conversations.
// @access Private (Requires authentication. User should typically only fetch their own messages.)
router.get('/messages/:userId', authenticateToken, async (req, res) => {
    const userId = req.params.userId;        // Get user ID from URL parameter
    const authenticatedUserId = req.user._id; // Authenticated user's ID

    // Validate the userId format
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    // Authorization: Users should typically only access their own messages
    // Allow access if the authenticated user is requesting their own messages or has admin privileges
    if (userId !== authenticatedUserId.toString()) {
        console.warn(`Authorization failed: User ${authenticatedUserId} attempted to access messages for ${userId}.`);
        return res.status(403).json({ message: 'Forbidden: You can only access your own messages.' }); // 403 Forbidden
    }

    try {
        // Find all messages where the user is either sender or receiver
        const messages = await Message.find({
            $or: [
                { sender: userId },
                { receiver: userId }
            ]
        })
        .populate('sender', 'name role displayIdentifier') // Populate sender details
        .populate('receiver', 'name role displayIdentifier') // Populate receiver details
        .sort({ timestamp: 1 }); // Sort by timestamp (oldest first)

        // Map to format for frontend
        const formattedMessages = messages.map(message => ({
            _id: message._id.toString(),
            conversationId: message.conversationId.toString(),
            sender: message.sender ? message.sender._id.toString() : null,
            receiver: message.receiver ? message.receiver._id.toString() : null,
            text: message.text,
            timestamp: message.timestamp ? message.timestamp.toISOString() : null,
            read: message.read,

            // Additional sender/receiver info for display
            senderName: message.sender ? (message.sender.name || message.sender.displayIdentifier) : 'Unknown',
            receiverName: message.receiver ? (message.receiver.name || message.receiver.displayIdentifier) : 'Unknown'
        }));

        // Send the list of messages for this user
        res.json(formattedMessages); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/messaging/messages/${userId} error:`, err);
        res.status(500).json({ message: 'Server error fetching user messages.' }); // 500 Internal Server Error
    }
});


// @route POST /api/messaging/conversations/:conversationId/messages
// @desc Send a new message in a specific conversation.
// @access Private (Requires authentication. User must be a participant in the conversation.)
router.post('/conversations/:conversationId/messages', authenticateToken, async (req, res) => {
    const conversationId = req.params.conversationId; // Get conversation ID from URL
    const userId = req.user._id;                     // Authenticated user is the sender
    const { text } = req.body;                       // Get message text from body

    // Basic input validation
    if (!text || text.trim() === '') {
        return res.status(400).json({ message: 'Message text cannot be empty.' }); // 400 Bad Request
    }
     // Validate conversationId format
     if (!mongoose.Types.ObjectId.isValid(conversationId)) {
         return res.status(400).json({ message: 'Invalid Conversation ID format.' }); // 400 Bad Request
     }


    try {
        // Find the conversation by ID
        const conversation = await Conversation.findById(conversationId);

        // If conversation not found, return 404
        if (!conversation) {
            return res.status(404).json({ message: 'Conversation not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is one of the participants.
        if (!conversation.participants.map(p => p.toString()).includes(userId.toString())) {
             console.warn(`Authorization failed: User ${userId} attempted to send message in conversation ${conversationId} they are not in.`);
            return res.status(403).json({ message: 'Forbidden: You are not a participant in this conversation.' }); // 403 Forbidden
        }

        // Determine the receiver ID. In a 1-on-1 chat, it's the other participant.
        const receiverId = conversation.participants.find(p => p.toString() !== userId.toString());
        if (!receiverId) {
            // This should ideally not happen in a valid 2-person conversation
             console.error(`Error: Could not determine receiver for conversation ${conversationId} from sender ${userId}`);
            return res.status(500).json({ message: 'Could not determine message receiver.' }); // 500 Internal Server Error
        }


        // Create a new Message document instance
        const newMessage = new Message({
            conversationId: conversation._id, // Link to the conversation document
            sender: userId,       // Link to the authenticated sender user ID
            receiver: receiverId, // Link to the determined receiver user ID
            text: text.trim(),
            timestamp: new Date(), // Set the timestamp now
            read: false, // Message is initially unread for the receiver
        });

        // Save the new message document to the database
        const savedMessage = await newMessage.save();


        // Update the parent conversation's last message snippet and timestamp
        // One character, not three: three dots is a typographic tell and the
        // messages list prints this snippet straight onto the screen.
        conversation.lastMessageSnippet = savedMessage.text.substring(0, 50) + (savedMessage.text.length > 50 ? '…' : '');
        conversation.lastMessageTimestamp = savedMessage.timestamp;
        // Save the updated conversation document
        await conversation.save();


        // Optional: Create a notification for the receiver about the new message.
         // Get sender's name for the notification message
         const senderUser = req.user; // Payload from token includes _id, role, name (if included during login)
         const senderDisplayName = (senderUser.name && senderUser.name.split(' ')[0]) ||
                                   senderUser.displayIdentifier ||
                                   senderUser._id.toString().substring(0, 6) + '…';

         try {
              const notification = new Notification({
                  recipient: receiverId, // Target the receiver user
                  type: 'message', // Custom notification type
                  // Curly quotes, a real ellipsis, and the sender's name reading as a person
                  // rather than a field. Straight quotes and three dots were both
                  // showing on screen.
                  message: `${senderDisplayName} wrote: “${savedMessage.text.slice(0, 60).trim()}${savedMessage.text.length > 60 ? '…' : ''}”`,
                   itemId: savedMessage._id, // Link to the message document
                   itemType: 'Message',
                   read: false, // Initially unread for the receiver
              });
              await notification.save();
              console.log(`SIMULATING: Created notification for user ${receiverId} about new message ${savedMessage._id}.`);
         } catch (notificationError) {
             console.error('Error creating notification for new message:', notificationError);
             // Do not block the response if notifications fail
         }


        // Send a success response with the created message document
        res.status(201).json(savedMessage); // 201 Created

    } catch (err) {
        console.error('POST /api/messaging/conversations/:conversationId/messages error:', err);
         // Handle Mongoose validation errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
        res.status(500).json({ message: 'Server error sending message.' }); // 500 Internal Server Error
    }
});


// @route PUT /api/messaging/conversations/:conversationId/messages/read
// @desc Mark messages in a conversation as read for the logged-in user (who is the receiver).
// @access Private (Requires authentication. User must be a participant and the receiver of the messages.)
router.put('/conversations/:conversationId/messages/read', authenticateToken, async (req, res) => {
    const conversationId = req.params.conversationId; // Get conversation ID from URL
    const userId = req.user._id;                     // Authenticated user (who is marking as read)

    // Validate conversationId format
    if (!mongoose.Types.ObjectId.isValid(conversationId)) {
        return res.status(400).json({ message: 'Invalid Conversation ID format.' }); // 400 Bad Request
    }

    try {
        // Optional: Find the conversation by ID and validate user participation
        // This step is good practice but the updateMany query below also implicitly checks participation
        const conversation = await Conversation.findById(conversationId);
        if (!conversation) {
            return res.status(404).json({ message: 'Conversation not found.' }); // 404 Not Found
        }
        // Authorization: Ensure the authenticated user is a participant
        if (!conversation.participants.map(p => p.toString()).includes(userId.toString())) {
             console.warn(`Authorization failed: User ${userId} attempted to mark messages read in conversation ${conversationId} they are not in.`);
            return res.status(403).json({ message: 'Forbidden: You are not a participant in this conversation.' }); // 403 Forbidden
        }


        // Find messages in this conversation where:
        // 1. The conversationId matches the requested conversation.
        // 2. The message was sent TO the authenticated user (userId is the receiver).
        // 3. The message is currently marked as unread (read: false).
        // Update all matching messages to set 'read' to true.
        const result = await Message.updateMany(
            {
                conversationId: conversationId,
                receiver: userId,
                read: false
            },
            { $set: { read: true } } // Update the 'read' field to true
        );

        // 'result' object contains information about the update operation,
        // including 'nModified' (number of documents modified).
        console.log(`Marked ${result.nModified} messages in conversation ${conversationId} as read for user ${userId}.`);

        // Optional: Mark related 'message' notifications for this user/conversation as read?
        // This depends on your notification logic granularity.
        // You could find and update Notification documents where recipient is userId,
        // type is 'message', and itemId is one of the message IDs modified above.
        // Or simpler: just update notifications where recipient is userId and itemType is 'Message'.
         // await Notification.updateMany(
         //     { recipient: userId, itemType: 'Message', read: false, itemId: { $in: modifiedMessageIds } }, // Need to get modified message IDs first
         //     { $set: { read: true } }
         // );
         // Or simplest for demo: just mark notifications related to this user AND message type as read.
         await Notification.updateMany(
             { recipient: userId, type: 'message', read: false },
             { $set: { read: true } }
         );
         console.log(`SIMULATING: Marked related 'message' notifications for user ${userId} as read.`);


        // Send a success response, indicating how many messages were updated
        res.json({ message: 'Messages marked as read successfully.', modifiedCount: result.nModified }); // Default status is 200 OK

    } catch (err) {
        console.error(`PUT /api/messaging/conversations/${conversationId}/messages/read error:`, err);
        res.status(500).json({ message: 'Server error marking messages as read.' }); // 500 Internal Server Error
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;