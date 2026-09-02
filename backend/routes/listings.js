// --- Backend Routes: listings.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose (needed for ObjectId validation)
const Listing = mongoose.model('Listing');  // Get the Listing Mongoose model
const Video = mongoose.model('Video');      // Get Video model to link to
const ipfs = require('../services/ipfsUpload');   // for the public gateway address
const { authenticateToken } = require('./auth');
const { requireFundableVideo } = require('../services/videoEligibility'); // Import the authentication middleware
const Purchase = mongoose.model('Purchase'); // To verify a caller actually bought this listing
const Notification = mongoose.model('Notification'); // Get Notification model for buyer notifications


/* A LOT'S VIDEO FRAME, AND WHERE TO PLAY IT.
 *
 * A listing stores a cid and nothing else about the file, so the market page had
 * a cid it could not show and no address it could play. Its own board says so:
 * the 92x68 row thumbnail and the 176px panel behind the selected lot were "a
 * flat #37322d fill, permanently" -- a play triangle drawn over nothing, on the
 * one screen where a buyer is deciding whether to trust a stranger's crop.
 *
 * One query for the whole page rather than one per lot. `poster` is select:false
 * on the model so it never rides along by accident; it has to be asked for.
 */
const frames = async (listings) => {
    const cids = [...new Set(listings.map(l => l.cid).filter(Boolean))];
    if (!cids.length) return {};
    const vids = await Video.find({ cid: { $in: cids } }).select('+poster cid').lean();
    return Object.fromEntries(vids.map(v => [v.cid, v.poster || null]));
};

// Helper function to generate a simulated blockchain transaction hash
const generateSimulatedTxHash = (prefix = 'sim_tx') => {
    // Generates a unique-enough string for demo purposes based on timestamp and random number
    return `${prefix}_${Date.now().toString(16)}${Math.random().toString(16).substring(2, 12)}`;
};


// --- Marketplace (Public) & Farmer's Listings Routes ---

// @route GET /api/listings
// @desc Get all active marketplace listings (can be filtered by query parameters)
// @access Public (Accessible to anyone, logged in or not)
// Query Params (matching frontend filters): ?crop=<cropName>, ?location=<text>, ?pesticideCompany=<text>
router.get('/', async (req, res) => {   // public marketplace browse -- intentionally unauthenticated

    try {
        // Start with filter for active listings
        const filter = { status: 'active' };

        // Add filters based on query parameters from the frontend (buyerFilters)
        if (req.query.crop) {
            // Case-insensitive exact match for crop type
            filter.crop = req.query.crop.trim(); // Assume frontend sends exact match value from select
            // If frontend sends partial text, use regex: filter.crop = { $regex: req.query.crop.trim(), $options: 'i' };
        }
        if (req.query.location) {
             // Case-insensitive partial match for location
             filter.location = { $regex: req.query.location.trim(), $options: 'i' };
        }
        if (req.query.pesticideCompany) {
             // Case-insensitive partial match for pesticideCompany
             filter.pesticideCompany = { $regex: req.query.pesticideCompany.trim(), $options: 'i' };
        }
        // Add other filters as needed

        // Find listing documents based on the filter
        // Populate the farmerWallet field to get the farmer's name for display
        const listings = await Listing.find(filter)
                                    .populate('farmerWallet', 'name role displayIdentifier') // Populate farmer's _id, name, role
                                    .sort({ createdAt: -1 }); // Sort by newest listing first


        // Map to format for frontend (e.g., ensure _id is string, format dates)
         const poster = await frames(listings);
         const formattedListings = listings.map(listing => ({
             _id: listing._id.toString(),
             farmerWallet: listing.farmerWallet ? listing.farmerWallet._id.toString() : null,
             farmerName: listing.farmerWallet ? (listing.farmerWallet.name.split(' ')[0] || listing.farmerWallet.displayIdentifier) : 'Unknown Farmer',
             crop: listing.crop,
             location: listing.location,
             pesticide: listing.pesticide,
             pesticideCompany: listing.pesticideCompany,
             cid: listing.cid,                       // public marketing video
             poster: poster[listing.cid] || null,    // one frame, for the thumbnail
             gateway: listing.cid ? ipfs.gatewayUrl(listing.cid) : null,
             storageType: listing.storageType,
             videoFileHash: listing.videoFileHash,   // shown in the purchase modal
             minPrice: listing.minPrice,
             maxPrice: listing.maxPrice,
             status: listing.status,
             createdAt: listing.createdAt ? listing.createdAt.toISOString() : null, // ISO string date
             notificationSent: listing.notificationSent,
             txHash: listing.txHash,
             // isNew: client-side state, not stored in DB
         }));


        // Send the list of active listings
        res.json(formattedListings); // Default status is 200 OK

    } catch (err) {
        console.error('GET /api/listings error:', err);
        res.status(500).json({ message: 'Server error fetching listings.' }); // 500 Internal Server Error
    }
});

// @route GET /api/listings/farmer/:farmerId
// @desc Get listings created by a specific farmer (all statuses)
// @access Private (Requires authentication. User should typically only fetch their own listings)
router.get('/farmer/:farmerId', authenticateToken, async (req, res) => {
    const farmerId = req.params.farmerId; // Get the farmer ID from the URL parameter
    const userId = req.user._id;         // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(farmerId)) {
        return res.status(400).json({ message: 'Invalid Farmer ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the farmer whose listings are being requested.
    // Or allow an admin user to view any farmer's listings.
    if (userId.toString() !== farmerId.toString()) {
         console.warn(`Authorization failed: User ${userId} attempted to view listings for user ${farmerId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own listings." }); // 403 Forbidden
    }

    try {
        // Find listing documents for the specified farmer (all statuses)
        // Populate farmerWallet field if needed (though we just verified the ID matches req.user._id)
        const listings = await Listing.find({ farmerWallet: farmerId })
                                     .populate('farmerWallet', 'name role displayIdentifier')
                                     .sort({ createdAt: -1 }); // Sort by newest first

        // Map to format for frontend
        const poster = await frames(listings);
        const formattedListings = listings.map(listing => ({
            _id: listing._id.toString(),
            farmerWallet: listing.farmerWallet ? listing.farmerWallet._id.toString() : null,
            farmerName: listing.farmerWallet ? (listing.farmerWallet.name.split(' ')[0] || listing.farmerWallet.displayIdentifier) : 'Unknown Farmer',
            crop: listing.crop,
            location: listing.location,
            pesticide: listing.pesticide,
            pesticideCompany: listing.pesticideCompany,
            cid: listing.cid,
            poster: poster[listing.cid] || null,
            gateway: listing.cid ? ipfs.gatewayUrl(listing.cid) : null,
            storageType: listing.storageType,
            videoFileHash: listing.videoFileHash,
            minPrice: listing.minPrice,
            maxPrice: listing.maxPrice,
            status: listing.status,
            createdAt: listing.createdAt ? listing.createdAt.toISOString() : null,
            notificationSent: listing.notificationSent,
            txHash: listing.txHash,
        }));


        // Send the list of listings for this farmer
        res.json(formattedListings); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/listings/farmer/${farmerId} error:`, err);
        res.status(500).json({ message: 'Server error fetching farmer listings.' }); // 500 Internal Server Error
    }
});


// @route POST /api/listings
// @desc Create a new marketplace listing
// @access Private (Requires authentication. Must be a farmer.)
router.post('/', authenticateToken, async (req, res) => {
    // Authorization: Ensure the authenticated user has the 'farmer' role.
    if (req.user.role !== 'farmer') {
         console.warn(`Authorization failed: User ${req.user._id} with role ${req.user.role} attempted to create a listing.`);
        return res.status(403).json({ message: "Forbidden: Only users with the 'farmer' role can create listings." }); // 403 Forbidden
    }

    // Extract listing details from the request body
    const { cid, minPrice, maxPrice, notifyBuyers } = req.body; // Get video CID and price range

    // Basic input validation
    if (!cid || minPrice === undefined || maxPrice === undefined) {
         return res.status(400).json({ message: "Missing required listing fields (cid, minPrice, maxPrice)." }); // 400 Bad Request
    }

    const parsedMinPrice = parseFloat(minPrice);
    const parsedMaxPrice = parseFloat(maxPrice);

     if (isNaN(parsedMinPrice) || isNaN(parsedMaxPrice) || parsedMinPrice <= 0 || parsedMaxPrice <= 0 || parsedMinPrice > parsedMaxPrice) {
         return res.status(400).json({ message: "Invalid price range provided. minPrice and maxPrice must be positive numbers, and maxPrice >= minPrice." }); // 400 Bad Request
     }
      if (parsedMinPrice > parsedMaxPrice) {
          // Double check max >= min although the previous check might catch it
          return res.status(400).json({ message: "Maximum price cannot be less than minimum price." }); // 400 Bad Request
      }


    try {
        // Debug: Log the received data
        console.log('Received listing data:', {
            cid, minPrice, maxPrice, notifyBuyers
        });
        console.log('User ID:', req.user._id);

        // Find the associated video metadata by CID and ensure it belongs to the authenticated farmer.
        // Same gate as funding requests: a listing is a money claim too, and a
        // buyer sees the same integrity language. See services/videoEligibility.js.
        const eligible = await requireFundableVideo(cid, req.user._id, { purpose: 'sell' });
        if (!eligible.ok) {
            return res.status(eligible.status).json({ message: eligible.message, code: eligible.code });
        }
        const video = eligible.video;
        console.log('Found video:', video);

        // If video is not found or doesn't belong to the farmer, return 404 or 403.
        if (!video) {
             console.warn(`Validation failed: Video with CID ${cid} not found for farmer ${req.user._id} during listing creation.`);
            return res.status(404).json({ message: "Associated video not found or does not belong to you." }); // 404 Not Found
        }

         // Check if this video is already used in an active listing.
         // This prevents a farmer from listing the same batch (represented by the same video) multiple times simultaneously.
         const existingActiveListing = await Listing.findOne({ cid: cid, status: 'active' });
         if (existingActiveListing) {
             console.warn(`Validation failed: Video with CID ${cid} is already used in active listing ${existingActiveListing._id}.`);
             return res.status(409).json({ message: "This video is already being used in an active marketplace listing." }); // 409 Conflict
         }


        // Create a new Listing document instance with video data populated manually
        const newListing = new Listing({
            farmerWallet: req.user._id, // Link to the authenticated farmer's user ID
            cid: video.cid, // Link the video CID
            minPrice: parsedMinPrice,
            maxPrice: parsedMaxPrice,
            // Manually populate fields from the video instead of relying on pre-save hook
            crop: video.crop,
            location: video.location,
            pesticide: video.pesticide || '',
            pesticideCompany: video.pesticideCompany || '',
            storageType: video.storageType,
            videoFileHash: video.videoFileHash,
            status: 'active', // Default status
            createdAt: new Date(), // Set creation date
            txHash: generateSimulatedTxHash('sim_list'), // Simulate a blockchain transaction hash for listing creation
            notificationSent: false, // Will be true if notifyBuyers is true and notification succeeds
        });

        console.log('Creating listing with populated data:', {
            crop: video.crop,
            location: video.location,
            pesticide: video.pesticide,
            pesticideCompany: video.pesticideCompany
        });

        // Save the new listing document to the database
        const savedListing = await newListing.save();

        // Optional: Send notification to potential buyers if requested.
        // This should ideally be a background task or message queue, not blocking the API response.
        // For this demo, we'll create a global notification document.
        if (notifyBuyers) {
             try {
                  const notification = new Notification({
                      global: true, // This is a global notification for the marketplace
                      type: 'listing', // Custom notification type
                      message: `“${savedListing.crop}” is now listed by a farmer.`,
                      itemId: savedListing._id, // Link to the new listing document
                      itemType: 'Listing',
                  });
                  await notification.save();
                 console.log(`SIMULATING: Created global notification for new listing ${savedListing._id}.`);

                 // Mark the listing as notified after successful notification creation
                  savedListing.notificationSent = true;
                  await savedListing.save(); // Save again to update the notificationSent flag

             } catch (notificationError) {
                 console.error('Error creating notification for new listing:', notificationError);
                 // Do not block listing creation if notification fails, but log it.
             }
        }

        // Populate the farmerWallet field on the saved document before sending it back,
        // so the frontend immediately has the farmer's name associated.
        await savedListing.populate('farmerWallet', 'name role displayIdentifier');


        // Send a success response with the created listing document
        res.status(201).json(savedListing); // 201 Created

    } catch (err) {
        console.error('POST /api/listings error:', err);
         // Handle Mongoose validation errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
         // Handle duplicate key error (less likely now with specific checks, but keep)
          if (err.code === 11000) {
              return res.status(409).json({ message: `A listing with this video CID already exists.` });
         }
        res.status(500).json({ message: 'Server error creating listing.' }); // 500 Internal Server Error
    }
});

// @route DELETE /api/listings/:id
// @desc Cancel a marketplace listing
// @access Private (Requires authentication. Must be a farmer and own the listing. Cannot be sold.)
router.delete('/:id', authenticateToken, async (req, res) => {
    const listingId = req.params.id; // Get the listing ID from the URL parameter
    const userId = req.user._id;     // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(listingId)) {
        return res.status(400).json({ message: 'Invalid Listing ID format.' }); // 400 Bad Request
    }

    try {
        // Find the listing by ID
        const listing = await Listing.findById(listingId);

        // If listing is not found, return 404
        if (!listing) {
            return res.status(404).json({ message: 'Listing not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is the farmer who owns the listing.
        if (listing.farmerWallet.toString() !== userId.toString()) {
             console.warn(`Authorization failed: User ${userId} attempted to cancel listing ${listingId} owned by ${listing.farmerWallet}`);
            return res.status(403).json({ message: "Forbidden: You can only cancel your own listings." }); // 403 Forbidden
        }

        // Prevent cancellation if the listing has already been sold.
        if (listing.status === 'sold') {
             console.warn(`Attempted to cancel sold listing: ${listingId}`);
            return res.status(400).json({ message: "Cannot cancel listing: It has already been sold." }); // 400 Bad Request
        }

        // Update the listing status to 'cancelled'.
        // We use findByIdAndUpdate for atomicity if multiple updates were possible,
        // but directly modifying and saving is fine here too.
        const updatedListing = await Listing.findByIdAndUpdate(
            listingId,
            { $set: { status: 'cancelled' } },
            { new: true } // Return the updated document
        );

        // Check again if the update was successful and returned a document
        if (!updatedListing) {
             console.error(`Error during update: Listing ${listingId} not found after initial find.`);
             return res.status(404).json({ message: 'Listing not found after update attempt.' });
        }

        // Optional: Notify interested buyers or remove related notifications? (More complex logic)

        // Send a success response with the updated listing document
         res.json({ message: 'Listing cancelled successfully.', listing: updatedListing }); // Default status is 200 OK

    } catch (err) {
        console.error(`DELETE/PUT /api/listings/${listingId} cancel error:`, err);
        res.status(500).json({ message: 'Server error cancelling listing.' }); // 500 Internal Server Error
    }
});


// @route PUT /api/listings/:id/notify
// @desc Send a 'new listing' notification to buyers for a specific listing.
// @access Private (Requires authentication. Must be a farmer and own the listing.)
router.put('/:id/notify', authenticateToken, async (req, res) => {
    const listingId = req.params.id; // Get the listing ID from the URL parameter
    const userId = req.user._id;     // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(listingId)) {
        return res.status(400).json({ message: 'Invalid Listing ID format.' }); // 400 Bad Request
    }

    try {
        // Find the listing by ID
        const listing = await Listing.findById(listingId);

        // If listing is not found, return 404
        if (!listing) {
            return res.status(404).json({ message: 'Listing not found.' }); // 404 Not Found
        }

        // Authorization: Ensure the authenticated user is the farmer who owns the listing.
        if (listing.farmerWallet.toString() !== userId.toString()) {
             console.warn(`Authorization failed: User ${userId} attempted to notify for listing ${listingId} owned by ${listing.farmerWallet}`);
            return res.status(403).json({ message: "Forbidden: You can only send notifications for your own listings." }); // 403 Forbidden
        }

        // Optional: Check if notification was already sent recently to avoid spamming
        // if (listing.notificationSent && (Date.now() - listing.updatedAt) < someCooldownPeriod) {
        //     return res.status(400).json({ message: "Notification for this listing was sent recently." });
        // }
        // For this demo, we'll allow re-sending unless the notificationSent flag is true (though the flag is basic)

        // Create a global notification for buyers
         try {
              const notification = new Notification({
                  global: true, // This notification is for the marketplace (buyers)
                  type: 'listing', // Custom notification type
                  message: `“${listing.crop}” is now listed by a farmer.`,
                  itemId: listing._id, // Link to the listing document
                  itemType: 'Listing',
              });
              await notification.save();
             console.log(`SIMULATING: Created global notification for listing ${listing._id}.`);

              // Update the listing's notificationSent flag
              listing.notificationSent = true;
              // Use findByIdAndUpdate for a simpler update if only changing the flag
              const updatedListing = await Listing.findByIdAndUpdate(
                   listingId,
                   { $set: { notificationSent: true } },
                   { new: true } // Return the updated document
               );


              res.json({ message: 'Buyers notified successfully.', listing: updatedListing }); // Default status is 200 OK

         } catch (notificationError) {
             console.error('Error creating notification:', notificationError);
              res.status(500).json({ message: 'Server error sending notification.' }); // 500 Internal Server Error
         }


    } catch (err) {
        console.error(`PUT /api/listings/${listingId}/notify error:`, err);
        // Handle potential errors during lookup
        res.status(500).json({ message: 'Server error notifying buyers.' }); // 500 Internal Server Error
    }
});


// @route  GET /api/listings/:id/media
// @desc   Authoritative media reference for a listing, released only to people
//         entitled to it: the farmer who owns it, or a buyer who purchased it.
// @access Private (owner or purchaser)
//
// Why this exists separately from the browse response:
// the browse route publishes `cid` because the listing video is public marketing
// for a physical crop sale. If any media is ever made genuinely paid, it belongs
// here and NOT in the browse payload -- publishing a CID is publishing the file,
// since anyone can fetch it straight from an IPFS gateway. This endpoint is the
// gated path, and it also gives a buyer a verifiable post-purchase record.
router.get('/:id/media', authenticateToken, async (req, res) => {
    const listingId = req.params.id;

    if (!mongoose.Types.ObjectId.isValid(listingId)) {
        return res.status(400).json({ message: 'Invalid Listing ID format.' });
    }

    try {
        const listing = await Listing.findById(listingId)
                                     .select('_id farmerWallet cid storageType videoFileHash crop');
        if (!listing) {
            return res.status(404).json({ message: 'Listing not found.' });
        }

        const isOwner = listing.farmerWallet
            && listing.farmerWallet.toString() === req.user._id.toString();

        // A purchase row is the entitlement record.
        const purchase = isOwner ? null : await Purchase.findOne({
            listingId: listing._id,
            buyerWallet: req.user._id
        }).select('_id purchaseDate txHash');

        if (!isOwner && !purchase) {
            console.warn(`Media denied: user ${req.user._id} has no entitlement to listing ${listingId}`);
            return res.status(403).json({
                message: 'Forbidden: you must purchase this listing to access its media.'
            });
        }

        return res.json({
            listingId: listing._id.toString(),
            crop: listing.crop,
            cid: listing.cid,
            storageType: listing.storageType || 'ipfs',
            videoFileHash: listing.videoFileHash,
            entitlement: isOwner ? 'owner' : 'purchaser',
            purchaseDate: purchase && purchase.purchaseDate ? purchase.purchaseDate.toISOString() : null,
            txHash: purchase ? purchase.txHash : null
        });

    } catch (err) {
        console.error('GET /api/listings/:id/media error:', err);
        return res.status(500).json({ message: 'Server error fetching listing media.' });
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;