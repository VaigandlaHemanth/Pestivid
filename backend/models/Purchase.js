// --- Mongoose Model: Purchase.js ---

const mongoose = require('mongoose'); // Import Mongoose library
// No need to explicitly require Listing here, as we access it via mongoose.model() in the pre-save hook


// Define the schema for the Purchase model
const purchaseSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key for each purchase document.

    buyerWallet: { // Reference to the User document of the buyer
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Buyer ID is required']
    },
    listingId: { // Reference to the Listing document that was purchased
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Listing', // Links to the 'Listing' model
        required: [true, 'Listing ID is required']
    },

    // Fields populated from the associated Listing for denormalization
    // This makes querying a buyer's purchase history faster without always populating the full listing
    farmerWallet: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }, // Farmer ID (copied from listing)
    crop: { type: String }, // Copied from listing
    location: { type: String }, // Copied from listing
    pesticide: { type: String }, // Copied from listing
    pesticideCompany: { type: String }, // Copied from listing
    cid: { type: String }, // Video CID from the listing
    videoStorageType: { type: String }, // Video storage type from the listing
    videoFileHash: { type: String }, // Video file hash from the listing


    // Purchase-specific details:
    price: { // The final price paid for this purchase (the offer price accepted)
        type: Number,
        required: [true, 'Price is required'],
        min: [0, 'Price cannot be negative']
    },
    purchaseDate: {
        type: Date,
        default: Date.now // Timestamp when the purchase was recorded
    },
    txHash: { // Simulated blockchain transaction hash for this purchase transaction
        type: String,
        required: [true, 'Transaction hash is required'], // Make txHash required for a purchase record
        unique: true, // Each purchase transaction should have a unique hash
        trim: true
    },

    // Add other fields relevant to a purchase if needed, e.g.:
    // quantity: Number, // Quantity purchased (if listings support quantity)
    // status: { type: String, enum: ['pending', 'shipped', 'delivered'], default: 'delivered' } // Status after purchase
});

// --- Mongoose Middleware (Hooks) ---

// Pre-save hook: This runs before saving a document.
// Used here to automatically populate redundant fields from the linked Listing document
// when a new purchase is created, based on the provided listingId.
purchaseSchema.pre('save', async function(next) {
    // 'this' refers to the document being saved

    // Only run this logic if it's a new document AND a listingId is present
    // Also, ensure the listing-related fields aren't already populated
    if (this.isNew && this.listingId && (!this.farmerWallet || !this.crop || !this.location || !this.cid)) {
        try {
            // Find the corresponding Listing document by listingId
            // Use .select() to get only necessary fields (matching the redundant fields in the schema)
            // Use .lean() to get a plain JavaScript object, faster for reading
            const Listing = mongoose.model('Listing'); // Get the model within the hook
            const listing = await Listing.findById(this.listingId).select('farmerWallet crop location pesticide pesticideCompany cid storageType videoFileHash').lean();

            if (listing) {
                // Populate the purchase fields from the listing document
                this.set({
                    farmerWallet: listing.farmerWallet, // Populate the farmer's ID
                    crop: listing.crop,
                    location: listing.location,
                    pesticide: listing.pesticide,
                    pesticideCompany: listing.pesticideCompany,
                    cid: listing.cid,
                    videoStorageType: listing.storageType,
                    videoFileHash: listing.videoFileHash,
                });
                 console.log(`Populated purchase fields from listing ${this.listingId}`);
            } else {
                // If the listing is not found, this purchase might be invalid.
                // Log a warning or prevent saving. A purchase should always link to a valid listing.
                console.error(`Validation Error: Listing with ID ${this.listingId} not found during purchase save.`);
                const error = new Error('Associated listing not found.');
                error.name = 'ValidationError'; // Mimic Mongoose validation error
                return next(error); // Prevent saving if listing is invalid
            }
        } catch (err) {
            // If there's a database error during the listing lookup
            console.error('Error in purchase pre-save hook:', err);
            return next(err); // Pass the error to Mongoose
        }
    }

    // Proceed with the save operation
    next();
});


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'purchases' (lowercase, plural) for this model.
const Purchase = mongoose.model('Purchase', purchaseSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Purchase;
