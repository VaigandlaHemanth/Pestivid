// --- Backend Routes: purchases.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose
const Purchase = mongoose.model('Purchase');  // Get the Purchase Mongoose model
const Listing = mongoose.model('Listing');  // Get Listing model to update its status
const Transaction = mongoose.model('Transaction'); // Get Transaction model to record purchase/sale
const { authenticateToken } = require('./auth'); // Import the authentication middleware
const Notification = mongoose.model('Notification'); // Get Notification model for farmer notifications


// Helper function to generate a simulated blockchain transaction hash
const generateSimulatedTxHash = (prefix = 'sim_tx') => {
    // Generates a unique-enough string for demo purposes based on timestamp and random number
    return `${prefix}_${Date.now().toString(16)}${Math.random().toString(16).substring(2, 12)}`;
};


// --- Purchase Routes ---

// @route GET /api/purchases/buyer/:buyerId
// @desc Get all purchases made by a specific buyer (for their history)
// @access Private (Requires authentication. User should typically only fetch their own purchases)
router.get('/buyer/:buyerId', authenticateToken, async (req, res) => {
    const buyerId = req.params.buyerId; // Get the buyer ID from the URL parameter
    const userId = req.user._id;        // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(buyerId)) {
        return res.status(400).json({ message: 'Invalid Buyer ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the buyer whose history is being requested.
    // Or allow an admin user to view any buyer's history.
    if (userId.toString() !== buyerId.toString()) {
         console.warn(`Authorization failed: User ${userId} attempted to view purchases for user ${buyerId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own purchases." }); // 403 Forbidden
    }

    try {
        // Find purchase documents for the specified buyer
        // Populate related farmer details for display in the history
        const purchases = await Purchase.find({ buyerWallet: buyerId })
                                       .populate('farmerWallet', 'name role displayIdentifier') // Populate farmer details
                                        // Note: Redundant fields like crop, location etc. are stored directly in Purchase
                                        // document, so we don't need to populate the listing here if that's sufficient.
                                       .sort({ purchaseDate: -1 }); // Sort by newest purchase first

        // Map to format for frontend (e.g., ensure _id is string, format dates)
         const formattedPurchases = purchases.map(purchase => ({
             _id: purchase._id.toString(),
             buyerWallet: purchase.buyerWallet.toString(), // Buyer user ID string

             // Redundant fields copied from listing
             farmerWallet: purchase.farmerWallet ? purchase.farmerWallet._id.toString() : null, // Farmer user ID string
             farmerName: purchase.farmerWallet ? (purchase.farmerWallet.name.split(' ')[0] || purchase.farmerWallet.displayIdentifier) : 'Unknown Farmer',
             crop: purchase.crop,
             location: purchase.location,
             pesticide: purchase.pesticide,
             pesticideCompany: purchase.pesticideCompany,
             cid: purchase.cid, // Video CID
             videoStorageType: purchase.videoStorageType,
             videoFileHash: purchase.videoFileHash,

             // Purchase-specific details
             listingId: purchase.listingId ? purchase.listingId.toString() : null, // Original listing ID string
             price: purchase.price,
             purchaseDate: purchase.purchaseDate ? purchase.purchaseDate.toISOString() : null, // ISO string date
             txHash: purchase.txHash, // Purchase transaction hash
         }));


        // Send the list of purchases for this buyer
        res.json(formattedPurchases); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/purchases/buyer/${buyerId} error:`, err);
        res.status(500).json({ message: 'Server error fetching purchases.' }); // 500 Internal Server Error
    }
});


// @route GET /api/purchases/farmer/:farmerId
// @desc Get all purchases of listings belonging to a specific farmer (their sales)
// @access Private (Requires authentication. User should typically only fetch their own sales)
router.get('/farmer/:farmerId', authenticateToken, async (req, res) => {
    const farmerId = req.params.farmerId; // Get the farmer ID from the URL parameter
    const userId = req.user._id;          // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(farmerId)) {
        return res.status(400).json({ message: 'Invalid Farmer ID format.' }); // 400 Bad Request
    }

    // Authorization: Users should typically only access their own sales data
    // Allow access if the authenticated user is the farmer or has admin privileges
    if (farmerId !== userId.toString()) {
        console.warn(`Authorization failed: User ${userId} attempted to access farmer sales for ${farmerId}.`);
        return res.status(403).json({ message: 'Forbidden: You can only access your own sales data.' }); // 403 Forbidden
    }

    try {
        // Find purchase documents where the farmer is the seller
        // Populate buyer details for display in the farmer's sales history
        const sales = await Purchase.find({ farmerWallet: farmerId })
                                   .populate('buyerWallet', 'name role displayIdentifier') // Populate buyer details
                                   .sort({ purchaseDate: -1 }); // Sort by newest sale first

        // Map to format for frontend
        const formattedSales = sales.map(sale => ({
            _id: sale._id.toString(),
            farmerWallet: sale.farmerWallet.toString(), // Farmer user ID string

            // Buyer information
            buyerWallet: sale.buyerWallet ? sale.buyerWallet._id.toString() : null, // Buyer user ID string
            buyerName: sale.buyerWallet ? (sale.buyerWallet.name.split(' ')[0] || sale.buyerWallet.displayIdentifier) : 'Unknown Buyer',

            // Product details (copied from listing)
            crop: sale.crop,
            location: sale.location,
            pesticide: sale.pesticide,
            pesticideCompany: sale.pesticideCompany,
            cid: sale.cid, // Video CID
            videoStorageType: sale.videoStorageType,
            videoFileHash: sale.videoFileHash,

            // Sale-specific details
            listingId: sale.listingId ? sale.listingId.toString() : null, // Original listing ID string
            price: sale.price,
            purchaseDate: sale.purchaseDate ? sale.purchaseDate.toISOString() : null, // ISO string date
            txHash: sale.txHash, // Purchase transaction hash
        }));

        // Send the list of sales for this farmer
        res.json(formattedSales); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/purchases/farmer/${farmerId} error:`, err);
        res.status(500).json({ message: 'Server error fetching farmer sales.' }); // 500 Internal Server Error
    }
});


// @route POST /api/purchases
// @desc Record a new purchase transaction and update the corresponding listing status
// @access Private (Requires authentication. Must be a buyer.)
router.post('/', authenticateToken, async (req, res) => {
    // Authorization: Ensure the authenticated user has the 'buyer' role.
    if (req.user.role !== 'buyer') {
         console.warn(`Authorization failed: User ${req.user._id} with role ${req.user.role} attempted to record a purchase.`);
        return res.status(403).json({ message: "Forbidden: Only users with the 'buyer' role can record purchases." }); // 403 Forbidden
    }

    // Extract purchase details from the request body
    const { listingId, offerPrice } = req.body; // Get the listing ID and the accepted offer price

    // Basic input validation
    if (!listingId || offerPrice === undefined) {
         return res.status(400).json({ message: "Missing required purchase fields (listingId, offerPrice)." }); // 400 Bad Request
    }

    const parsedPrice = parseFloat(offerPrice);
    if (isNaN(parsedPrice) || parsedPrice <= 0) {
         return res.status(400).json({ message: "Invalid offer price. Price must be a positive number." }); // 400 Bad Request
    }
     // Validate listingId format
     if (!mongoose.Types.ObjectId.isValid(listingId)) {
         return res.status(400).json({ message: 'Invalid Listing ID format.' }); // 400 Bad Request
     }


    // --- Transaction and Update Logic (Wrap in a transaction if using replica set) ---
    // In production, for operations that update multiple documents (like this, updating Listing and creating Purchase/Transaction),
    // you should use MongoDB transactions if your deployment is a replica set. This ensures
    // either all operations succeed or all fail, maintaining data consistency.
    // For a standalone MongoDB server or simple demo, we'll just perform operations sequentially.

    try {
        // --- 1. Find and Update the Listing Status ---
        // Find the listing by ID and ensure it's still 'active'
        const listing = await Listing.findById(listingId);

        // If listing is not found, return 404
        if (!listing) {
            return res.status(404).json({ message: 'Listing not found.' }); // 404 Not Found
        }

        // Prevent buying if the listing is not 'active' (e.g., already sold or cancelled by someone else)
        if (listing.status !== 'active') {
             console.warn(`Attempted to purchase listing ${listingId} with status "${listing.status}"`);
            return res.status(400).json({ message: `Cannot purchase listing: It is currently "${listing.status}".` }); // 400 Bad Request
        }

        // Prevent buying your own listing (should ideally be blocked on frontend too)
         if (listing.farmerWallet.toString() === req.user._id.toString()) {
              return res.status(400).json({ message: "Cannot purchase your own listing." });
         }

        // --- Server-side price validation ---
        // The price band was previously enforced ONLY in the browser
        // (public/index.html), so a crafted request could buy any listing for
        // any amount above zero. The seller's own listing is the source of truth.
        if (parsedPrice < listing.minPrice || parsedPrice > listing.maxPrice) {
            console.warn(`Rejected out-of-band offer ${parsedPrice} on listing ${listingId} (band ${listing.minPrice}-${listing.maxPrice})`);
            return res.status(400).json({
                message: `Offer must be between ${listing.minPrice} and ${listing.maxPrice}.`
            });
        }


        // Claim the listing atomically.
        // The `status: 'active'` guard above is a separate round-trip, so on its
        // own it does not prevent two concurrent buyers both passing it and both
        // creating a Purchase. Putting the condition INSIDE the filter makes
        // MongoDB itself the arbiter: exactly one caller can match and flip it.
        const updatedListing = await Listing.findOneAndUpdate(
             { _id: listingId, status: 'active' },
             { $set: { status: 'sold' } },
             { new: true } // Return the updated document
         );

        // Lost the race: someone else claimed this listing first.
        if (!updatedListing) {
             return res.status(409).json({ message: 'Listing is no longer available.' });
        }


        // --- 2. Create the Purchase Document ---
        // Simulate transaction hash for the purchase
        const purchaseTxHash = generateSimulatedTxHash('sim_purchase');

        const newPurchase = new Purchase({
            buyerWallet: req.user._id, // Link to the authenticated buyer
            listingId: updatedListing._id, // Link to the sold listing's ID
            price: parsedPrice, // Record the final agreed price
            txHash: purchaseTxHash, // Use the generated transaction hash
            purchaseDate: new Date(), // Set purchase date
            // Redundant fields (farmerWallet, crop, etc.) will be populated by the pre-save hook on Purchase.js
        });

        const savedPurchase = await newPurchase.save();


        // --- 3. Create Transaction Records ---
         // Transaction for the buyer (outflow - purchase)
         const buyerTransaction = new Transaction({
             userId: req.user._id, // The buyer's ID
             txHash: savedPurchase.txHash, // Use the same hash as the purchase record
             type: 'purchase',
             amount: savedPurchase.price, // Amount spent
             listingId: savedPurchase.listingId, // Link to the listing
             date: savedPurchase.purchaseDate, // Date of the transaction
         });
         await buyerTransaction.save();
         console.log(`SIMULATING: Created transaction ${buyerTransaction._id} for buyer ${req.user._id}`);

         // Transaction for the farmer (inflow - sale)
         const farmerTransaction = new Transaction({
              userId: updatedListing.farmerWallet, // The farmer's ID
              txHash: generateSimulatedTxHash('sim_sale'), // Generate a separate hash for the sale transaction record
              type: 'sale',
              amount: savedPurchase.price, // Amount received (same as purchase price)
              listingId: updatedListing._id, // Link to the listing
              date: savedPurchase.purchaseDate, // Date of the transaction (same as purchase date)
         });
         await farmerTransaction.save();
         console.log(`SIMULATING: Created transaction ${farmerTransaction._id} for farmer ${updatedListing.farmerWallet}`);


        // --- 4. Send Notifications ---
        // Notify the farmer that their listing was purchased
         try {
              // Need the buyer's name for the notification message
              // We can get it from req.user since the buyer is the authenticated user
             const buyerUser = req.user; // The payload from the token includes _id, role, and potentially name
             const buyerDisplayName = (buyerUser.name && buyerUser.name.split(' ')[0]) || buyerUser._id.toString().substring(0, 6) + '...';

              const notification = new Notification({
                  recipient: updatedListing.farmerWallet, // Farmer's ID
                  type: 'purchase', // Custom notification type
                  message: `“${savedPurchase.crop || 'your lot'}” was bought by ${buyerDisplayName}.`,
                  itemId: savedPurchase.listingId, // Link to the listing document
                  itemType: 'Listing',
                  read: false, // Initially unread for the farmer
              });
              await notification.save();
              console.log(`SIMULATING: Notified farmer ${updatedListing.farmerWallet} about purchase of listing ${updatedListing._id}.`);

             // The buyer's own receipt notice. It was a comment, so the person
             // who paid got nothing while the person who was paid got a bell.
             // type purchase + Listing sends a buyer's row to their orders page.
             const paid = '₹' + Math.round(Number(savedPurchase.price) || 0).toLocaleString('en-IN');
             const buyerNotification = new Notification({
                 recipient: req.user._id,
                 type: 'purchase',
                 message: `You bought “${savedPurchase.crop || 'a lot'}” for ${paid}. The receipt is on your orders page.`,
                 itemId: savedPurchase.listingId,
                 itemType: 'Listing',
                 read: false,
             });
             await buyerNotification.save();

         } catch (notificationError) {
             console.error('Error creating notification after purchase:', notificationError);
             // Do not block the response if notifications fail
         }


        // --- 5. Prepare Response ---
        // Populate related documents on the saved purchase document before sending it back
        await savedPurchase.populate('farmerWallet', 'name role displayIdentifier'); // Populate farmer details
        await savedPurchase.populate('listingId'); // Optionally populate the listing if frontend needs it (redundant fields already copied)

         // Map to format for frontend (similar structure to the GET response)
         const formattedPurchase = {
             _id: savedPurchase._id.toString(),
             buyerWallet: savedPurchase.buyerWallet.toString(),
             listingId: savedPurchase.listingId ? savedPurchase.listingId._id.toString() : null,

             farmerWallet: savedPurchase.farmerWallet ? savedPurchase.farmerWallet._id.toString() : null,
             farmerName: savedPurchase.farmerWallet ? (savedPurchase.farmerWallet.name.split(' ')[0] || savedPurchase.farmerWallet.displayIdentifier) : 'Unknown Farmer',
             crop: savedPurchase.crop,
             location: savedPurchase.location,
             pesticide: savedPurchase.pesticide,
             pesticideCompany: savedPurchase.pesticideCompany,
             cid: savedPurchase.cid,
             videoStorageType: savedPurchase.videoStorageType,
             videoFileHash: savedPurchase.videoFileHash,

             price: savedPurchase.price,
             purchaseDate: savedPurchase.purchaseDate ? savedPurchase.purchaseDate.toISOString() : null,
             txHash: savedPurchase.txHash,
         };


        // Send a success response with the created purchase document
        res.status(201).json(formattedPurchase); // 201 Created

    } catch (err) {
        console.error('POST /api/purchases error:', err);
        // Handle potential Mongoose validation errors or other errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
        res.status(500).json({ message: 'Server error recording purchase.' }); // 500 Internal Server Error
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;