// --- Backend Routes: notifications.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose
const Notification = mongoose.model('Notification'); // Get Notification model
const { authenticateToken } = require('./auth'); // Import the authentication middleware


// --- Notification Routes ---

// @route GET /api/notifications/user/:userId
// @desc Get all notifications for a specific user (including global notifications).
// @access Private (Requires authentication. User must typically fetch only their own notifications.)
// Query Params: ?read=<true|false> (Optional: Filter by read status)
router.get('/user/:userId', authenticateToken, async (req, res) => {
    const userId = req.params.userId; // Get the user ID from the URL parameter
    const authenticatedUserId = req.user._id; // Authenticated user's ID

    // Validate the userId format
    if (!mongoose.Types.ObjectId.isValid(userId)) {
        return res.status(400).json({ message: 'Invalid User ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the user whose notifications are being requested.
    // Or allow an admin user to view any user's notifications.
    if (authenticatedUserId.toString() !== userId.toString()) {
         console.warn(`Authorization failed: User ${authenticatedUserId} attempted to view notifications for user ${userId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own notifications." }); // 403 Forbidden
    }

    try {
        // Build the filter object.
        // Notifications are either for the specific user OR are global.
        // A global notification the user has dismissed must not come back. The
        // dismissedBy set was added when DELETE stopped removing globals for
        // everyone, but this query never consulted it -- so "delete" appeared to
        // do nothing and the item reappeared on the next poll.
        const filter = {
            dismissedBy: { $ne: userId },
            $or: [
                { recipient: userId },  // addressed to this user
                { global: true },       // platform-wide broadcast
            ],
        };
        // ?after=<ISO date>: only what arrived since then. A page that polls
        // for arrivals has no use for the whole history it already holds. A
        // date that does not parse is ignored rather than guessed at.
        if (req.query.after !== undefined) {
            const after = new Date(String(req.query.after));
            if (!Number.isNaN(after.getTime())) filter.timestamp = { $gt: after };
        }

        const notifications = await Notification.find(filter)
                                               .sort({ timestamp: -1 })
                                               .lean();

        // Per-user read state has to be resolved BEFORE any read filter is
        // applied, because for a global it lives in readBy rather than in `read`.
        // Filtering in the query on `read` would have used the shared flag.
        const isReadForUser = (n) => (n.global
            ? (n.readBy || []).some((u) => String(u) === String(userId))
            : Boolean(n.read));

        let visible = notifications;
        if (req.query.read !== undefined) {
            const want = String(req.query.read).toLowerCase();
            if (want === 'true') visible = visible.filter(isReadForUser);
            else if (want === 'false') visible = visible.filter((n) => !isReadForUser(n));
            // anything else: ignore the parameter rather than guessing
        }

        // Map to format for frontend (e.g., ensure _id is string, format dates)
         const formattedNotifications = visible.map(notification => ({
             _id: notification._id.toString(), // Notification ID string
             recipient: notification.recipient ? notification.recipient.toString() : null, // Recipient User ID string (null if global)
             global: notification.global,
             type: notification.type,
             message: notification.message,
             timestamp: notification.timestamp ? notification.timestamp.toISOString() : null, // ISO string date
             read: isReadForUser(notification),
             itemId: notification.itemId ? notification.itemId.toString() : null, // Related item ID string (if linked)
             itemType: notification.itemType, // Related item type (if linked)
         }));


        // Send the list of notifications for this user
        res.json(formattedNotifications); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/notifications/user/${userId} error:`, err);
        res.status(500).json({ message: 'Server error fetching user notifications.' }); // 500 Internal Server Error
    }
});


// @route PUT /api/notifications/:id/read
// @desc Mark a specific notification as read.
// @access Private (Requires authentication. User must be the recipient OR it's a global notification OR user is admin.)
router.put('/:id/read', authenticateToken, async (req, res) => {
    const notificationId = req.params.id; // Get the notification ID from the URL parameter
    const userId = req.user._id;         // Authenticated user's ID

    // Validate the notificationId format
    if (!mongoose.Types.ObjectId.isValid(notificationId)) {
        return res.status(400).json({ message: 'Invalid Notification ID format.' }); // 400 Bad Request
    }

    try {
        // Find the notification by ID
        const notification = await Notification.findById(notificationId);

        // If notification is not found, return 404
        if (!notification) {
            return res.status(404).json({ message: 'Notification not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is authorized to mark this notification as read.
        // They must be the recipient, OR it must be a global notification, OR they must be an admin.
        const isRecipient = notification.recipient && notification.recipient.toString() === userId.toString();
        const isGlobal = notification.global === true;

        if (!isRecipient && !isGlobal) {
             console.warn(`Authorization failed: User ${userId} attempted to mark notification ${notificationId} as read (Recipient: ${notification.recipient}, Global: ${notification.global}).`);
            return res.status(403).json({ message: "Forbidden: You do not have permission to mark this notification as read." }); // 403 Forbidden
        }

        // Check if the notification is already read (optional, maybe return 304 Not Modified)
        // if (notification.read) {
        //     return res.status(304).send(); // 304 Not Modified
        // }

        // Mark as read -- per user for a global, on the document for a direct one.
        //
        // Setting notification.read on a global marked the platform-wide broadcast
        // read for EVERY user, because they all share this single document. Any
        // one user dismissing an announcement made it vanish for all of them.
        if (isGlobal && !isRecipient) {
            await Notification.updateOne(
                { _id: notificationId },
                { $addToSet: { readBy: userId } },
            );
        } else {
            notification.read = true;
            await notification.save();
        }

        const fresh = await Notification.findById(notificationId).lean();
        res.json({
            message: 'Notification marked as read successfully.',
            notification: {
                ...fresh,
                // Report the state for THIS user, not the raw shared field.
                read: fresh.global
                    ? (fresh.readBy || []).some((u) => String(u) === String(userId))
                    : fresh.read,
                readBy: undefined,
                dismissedBy: undefined,
            },
        });

    } catch (err) {
        console.error(`PUT /api/notifications/${notificationId}/read error:`, err);
        // Handle potential Mongoose validation errors or other errors
        res.status(500).json({ message: 'Server error marking notification as read.' }); // 500 Internal Server Error
    }
});


// Note: Notification creation happens within other routes where events occur
// (e.g., POST /api/messaging/conversations/:conversationId/messages creates a 'message' notification,
// POST /api/listings creates a 'listing' notification,
// POST /api/funding-requests creates a 'funding' notification,
// PUT /api/investments/:id/progress might create 'payout' or 'update' notifications).
// There is typically no direct API endpoint for creating a general notification from the frontend,
// as notifications should be server-generated based on specific events.


// @route DELETE /api/notifications/:id
// @desc Delete a specific notification (e.g., dismiss permanently)
// @access Private (Requires authentication. User must be the recipient OR it's a global notification OR user is admin.)
// NOTE: This route is optional. You might just mark as read instead of deleting.
router.delete('/:id', authenticateToken, async (req, res) => {
    const notificationId = req.params.id;
    const userId = req.user._id;

     // Validate the notificationId format
     if (!mongoose.Types.ObjectId.isValid(notificationId)) {
         return res.status(400).json({ message: 'Invalid Notification ID format.' }); // 400 Bad Request
     }

     try {
         // Find the notification
         const notification = await Notification.findById(notificationId);

         if (!notification) {
             return res.status(404).json({ message: 'Notification not found.' });
         }

         // Authorization: Ensure user is authorized to delete (same logic as PUT /read)
         const isRecipient = notification.recipient && notification.recipient.toString() === userId.toString();
         const isGlobal = notification.global === true; // Allow anyone to dismiss global notifications? Or only admin? Let's allow anyone for demo.

         if (!isRecipient && !isGlobal) {
              console.warn(`Authorization failed: User ${userId} attempted to delete notification ${notificationId} they are not authorized for.`);
             return res.status(403).json({ message: "Forbidden: You do not have permission to delete this notification." });
         }

         if (isGlobal && !isRecipient) {
             // A global notification belongs to everyone, so one user must not be
             // able to delete it for the whole platform. Record a per-user
             // dismissal instead and leave the document intact.
             await Notification.updateOne(
                 { _id: notificationId },
                 { $addToSet: { dismissedBy: userId } }
             );
             return res.json({ message: 'Notification dismissed.' });
         }

         // Only the actual recipient can delete their own notification.
         await notification.deleteOne();

         res.json({ message: 'Notification deleted successfully.' }); // Default status is 200 OK

     } catch (err) {
         console.error(`DELETE /api/notifications/${notificationId} error:`, err);
         res.status(500).json({ message: 'Server error deleting notification.' });
     }
});



// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;