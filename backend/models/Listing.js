// --- Mongoose Model: Listing.js ---

const mongoose = require('mongoose'); // Import Mongoose library
const Video = mongoose.model('Video'); // Import the Video model to use in pre-save hook


// Define the schema for the Listing model
const listingSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key.

    farmerWallet: { // Reference to the User document of the farmer listing the crop
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Farmer ID is required']
    },

    // Fields copied/linked from the associated video for easier querying and display:
    crop: {
        type: String,
        required: [true, 'Crop type is required'],
        trim: true
    },
    location: {
        type: String,
        required: [true, 'Location is required'],
        trim: true
    },
    pesticide: { // Pesticide used (or 'None', 'Organic', etc.) - Optional
        type: String,
        trim: true
    },
    pesticideCompany: { // Company name for the pesticide used - Optional
        type: String,
        trim: true
    },
    cid: { // The CID/Key of the associated video evidence
        type: String,
        required: [true, 'Video CID/Key is required'],
        trim: true
        // Note: While unique video CIDs exist, a CID can be used in multiple listings
        // if the listing represents a batch from the same video evidence.
        // However, in this demo, we link one video to one active listing.
        // A compound unique index on { cid, status } where status is 'active' could enforce this.
        // For simplicity, we'll handle the 'already listed' check in the route handler.
    },
    storageType: { // Storage type of the associated video (e.g., 'storj', 'ipfs')
        type: String,
        enum: ['storj', 'ipfs'],
        default: 'ipfs' // Default value, should ideally match the video document
    },
    videoFileHash: { // SHA256 hash of the associated video file
        type: String,
        trim: true
    },

    // Listing-specific fields:
    minPrice: {
        type: Number,
        required: [true, 'Minimum price is required'],
        min: [0, 'Minimum price cannot be negative']
    },
    maxPrice: {
        type: Number,
        required: [true, 'Maximum price is required'],
        min: [0, 'Maximum price cannot be negative']
        // Custom validation to ensure maxPrice >= minPrice can be added if not handled in route
    },
    status: {
        type: String,
        default: 'active', // Initial status when created
        enum: {
            values: ['active', 'sold', 'cancelled'], // Possible states for a listing
            message: 'Invalid listing status.'
        }
    },
    createdAt: {
        type: Date,
        default: Date.now // Timestamp when the listing was created
    },
    notificationSent: { // Flag to track if the 'new listing' notification was sent to buyers
        type: Boolean,
        default: false
    },
    txHash: { // Simulated blockchain transaction hash for the listing creation event
        type: String,
        trim: true
        // Not strictly required by DB schema, but useful for tracking
    },

    // Add other fields as needed for your marketplace, e.g.:
    // quantity: Number, // Quantity of crop being sold
    // unit: String, // Unit of measurement (e.g., 'kg', 'bushels', 'crates')
    // expiryDate: Date, // Date the listing expires
    // photos: [String], // Array of image URLs
    // description: String, // A specific description for the listing
});

// --- Mongoose Middleware (Hooks) ---

// Pre-save hook: This function runs before a Listing document is saved.
// Used here to automatically populate some fields from the linked Video document
// when a new listing is created, based on the provided cid.
listingSchema.pre('save', async function(next) {
    // 'this' refers to the document being saved
    console.log('Pre-save hook triggered for listing:', {
        isNew: this.isNew,
        cid: this.cid,
        farmerWallet: this.farmerWallet,
        currentCrop: this.crop,
        currentLocation: this.location
    });

    // Only run this logic if it's a new document AND a cid and farmerWallet are present
    if (this.isNew && this.cid && this.farmerWallet) {
        try {
            console.log('Looking for video with CID:', this.cid, 'and farmerWallet:', this.farmerWallet);

            // Find the corresponding Video document by CID and farmerWallet
            // Use .lean() to get a plain JavaScript object, faster for reading
            const video = await mongoose.model('Video').findOne({
                cid: this.cid,
                farmerWallet: this.farmerWallet
            }).select('crop location pesticide pesticideCompany storageType videoFileHash').lean();

            console.log('Found video in pre-save hook:', video);

            if (video) {
                // Populate the listing fields from the video document
                this.set({
                    crop: video.crop,
                    location: video.location,
                    pesticide: video.pesticide,
                    pesticideCompany: video.pesticideCompany,
                    storageType: video.storageType, // Use video's storage type
                    videoFileHash: video.videoFileHash, // Use video's hash
                });
                console.log(`Populated listing fields from video ${this.cid}:`, {
                    crop: video.crop,
                    location: video.location,
                    pesticide: video.pesticide,
                    pesticideCompany: video.pesticideCompany
                });
            } else {
                // If the video is not found or doesn't belong to the farmer,
                // this indicates an issue. You could prevent saving by throwing an error.
                console.error(`Validation Warning: Video with CID ${this.cid} not found for farmer ${this.farmerWallet} during listing save.`);
                // Depending on strictness, you might want to throw an error here:
                // const error = new Error('Associated video not found or does not belong to the farmer.');
                // error.name = 'ValidationError'; // Mimic Mongoose validation error
                // return next(error);
                // For this demo, we'll just log and save with potentially missing video data.
            }
        } catch (err) {
            // If there's a database error during the video lookup
            console.error('Error in listing pre-save hook:', err);
            return next(err); // Pass the error to Mongoose
        }
    } else {
        console.log('Pre-save hook skipped - conditions not met');
    }

    // Proceed with the save operation
    next();
});


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'listings' (lowercase, plural) for this model.
const Listing = mongoose.model('Listing', listingSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Listing;
