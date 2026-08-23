// --- Mongoose Model: FundingRequest.js ---

const mongoose = require('mongoose'); // Import Mongoose library
const Video = mongoose.model('Video'); // Import the Video model to use in pre-save hook


// Define the schema for the FundingRequest model
const fundingRequestSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key.

    farmerWallet: { // Reference to the User document of the farmer requesting funding
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Farmer ID is required']
    },

    // Project Details:
    title: {
        type: String,
        required: [true, 'Project title is required'],
        trim: true,
        maxlength: [100, 'Project title cannot exceed 100 characters']
    },
    crop: {
        type: String,
        required: [true, 'Crop type is required'],
        trim: true
    },
    acres: { // Land size for the project
        type: Number,
        required: [true, 'Land size (acres) is required'],
        min: [0.1, 'Land size must be at least 0.1 acres']
    },
    amount: { // Total funding amount required, in rupees
        type: Number,
        required: [true, 'Funding amount is required'],
        min: [1, 'Funding amount must be at least 1 rupee']
    },
    method: { // Growing method
        type: String,
        required: [true, 'Growing method is required'],
        enum: {
            values: ['organic', 'conventional', 'hydroponic', 'aquaponic', 'regenerative'],
            message: 'Invalid growing method.'
        }
    },
    description: {
        type: String,
        required: [true, 'Project description is required'],
        trim: true,
        maxlength: [1000, 'Description cannot exceed 1000 characters'] // Add a max length
    },
    timeline: { // Expected duration of the project (in months)
        type: Number,
        required: [true, 'Timeline is required'],
        min: [1, 'Timeline must be at least 1 month'],
        max: [36, 'Timeline cannot exceed 36 months'] // Example max
    },
    roi: { // Expected Return on Investment for investors (as a percentage)
        type: Number,
        required: [true, 'Expected ROI is required'],
        min: [0, 'ROI cannot be negative'],
        max: [100, 'ROI cannot exceed 100%']
    },
    // --- Settlement terms -------------------------------------------------
    // The stated goal offers the farmer two ways to settle: share the profit, or
    // repay the whole fund. Only one formula existed before -- amount * roi/100 --
    // which is neither: it pays the yield and never returns the principal.
    settlementMode: {
        type: String,
        default: 'profit_share',
        enum: {
            values: ['profit_share', 'full_repayment'],
            message: 'settlementMode must be profit_share or full_repayment.'
        }
    },
    // Reported by the farmer at harvest. A profit share cannot be computed
    // without it, which is why investorShare was previously unusable.
    harvestRevenue: { type: Number, min: [0, 'Revenue cannot be negative'] },
    inputCostBasis: { type: Number, min: [0, 'Costs cannot be negative'] },
    harvestReportedAt: { type: Date },
    // Crops fail. Without this the model implicitly promises a return, which is
    // both dishonest and the exact thing that sank the real crowdfunding platforms.
    outcome: {
        type: String,
        default: 'pending',
        enum: {
            values: ['pending', 'harvested', 'partial_loss', 'total_loss'],
            message: 'Invalid harvest outcome.'
        }
    },

    investorShare: { // Percentage of profit shared with investors
        type: Number,
        required: [true, 'Investor share is required'],
        min: [0, 'Investor share cannot be negative'],
        max: [100, 'Investor share cannot exceed 100%']
    },

    // Video Evidence:
    cid: { // The CID/Key of the associated video evidence for this project
        type: String,
        required: [true, 'Video CID/Key is required'],
        trim: true
        // Similar to Listing, could add unique index if needed
    },
    videoStorageType: { // Storage type of the associated video
        type: String,
        enum: ['storj', 'ipfs'],
        default: 'ipfs' // Default value, should ideally match the video document
    },
    videoFileHash: { // SHA256 hash of the associated video file
        type: String,
        trim: true
    },

    // Funding Progress:
    fundedAmount: { // Total funding received so far, in rupees
        type: Number,
        default: 0,
        min: [0, 'Funded amount cannot be negative']
    },
    status: { // Current status of the funding request
        type: String,
        default: 'pending', // Initial status
        enum: {
            values: ['pending', 'partially_funded', 'funded', 'completed', 'cancelled'], // Possible states
            message: 'Invalid funding request status.'
        }
    },
    createdAt: {
        type: Date,
        default: Date.now // Timestamp when the request was created
    },

    // Embedded Investor Details:
    // Array of investors who have contributed to this project.
    // This is an embedded array, meaning investor data is stored directly within the FundingRequest document.
    investors: [{
        // Note: Mongoose automatically adds a default _id to embedded documents, but we don't typically need to reference it.
        investorId: { // Reference to the User document of the investor
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User', // Links to the 'User' model
            required: true
        },
        amount: { // Amount invested by this specific investor in this project
            type: Number,
            required: true,
            min: 0
        },
        txHash: { // Simulated transaction hash for this specific investment event
            type: String,
            trim: true
        },
        investmentDate: { // Date of this specific investment
            type: Date,
            default: Date.now
        },
        // Add other investment-specific details if needed
    }],

    // Embedded Farmer Updates:
    // Array of updates posted by the farmer regarding the project progress.
    updates: [{
        // No separate `id` field: Mongoose auto-generates `_id` for embedded
        // documents, and that is what the frontend keys off. A previously
        // required String `id` was never populated by either push site, so
        // every project update failed validation.
        date: { type: String, required: true }, // Keeping as String date for demo consistency
        text: { type: String, required: true, trim: true },
        // Add timestamp: Date if you switch to Date objects for dates
    }],

    // Add other fields as needed, e.g.:
    // photos: [String], // Array of additional photos/videos related to the project
    // requiredItems: [String], // List of things funding will pay for
    // budgetBreakdown: Object, // Detailed budget breakdown
    // expectedHarvestDate: Date,
});

// --- Mongoose Middleware (Hooks) ---

// Pre-save hook: This runs before saving a document.
// Used here to populate video-related fields from the linked Video document
// when a new request is created, based on the provided cid and farmerWallet.
fundingRequestSchema.pre('save', async function(next) {
    // 'this' refers to the document being saved

    // Only run this logic if it's a new document AND a cid and farmerWallet are present
    // Also, ensure the video-related fields aren't already populated
    if (this.isNew && this.cid && this.farmerWallet && (!this.videoStorageType || !this.videoFileHash)) {
        try {
            // Find the corresponding Video document by CID and farmerWallet
            // Use .select() to get only necessary fields
            // Use .lean() to get a plain JavaScript object, faster for reading
            const video = await mongoose.model('Video').findOne({
                cid: this.cid,
                farmerWallet: this.farmerWallet
            }).select('storageType videoFileHash').lean();

            if (video) {
                // Populate the funding request fields from the video document
                this.set({
                    videoStorageType: video.storageType, // Use video's storage type
                    videoFileHash: video.videoFileHash, // Use video's hash
                });
                console.log(`Populated funding request video fields from video ${this.cid}`);
            } else {
                // If the video is not found or doesn't belong to the farmer,
                // this indicates an issue. Log a warning or prevent saving.
                console.warn(`Validation Warning: Video with CID ${this.cid} not found for farmer ${this.farmerWallet} during funding request save.`);
                // Depending on strictness, you might want to throw an error here:
                // const error = new Error('Associated video not found or does not belong to the farmer.');
                // error.name = 'ValidationError'; // Mimic Mongoose validation error
                // return next(error);
                // For this demo, we'll just log and save with potentially missing video data.
            }
        } catch (err) {
            // If there's a database error during the video lookup
            console.error('Error in funding request pre-save hook:', err);
            return next(err); // Pass the error to Mongoose
        }
    }

    // Ensure fundedAmount does not exceed total amount
     if (this.fundedAmount > this.amount) {
         this.fundedAmount = this.amount;
     }

     // Derive status from fundedAmount -- but ONLY for the funding-progress
     // states. This block previously ran on every save() and unconditionally
     // overwrote `status`, which made 'completed' and 'cancelled' permanently
     // unreachable: a route would set one, call save(), and have it silently
     // rewritten back to 'funded'/'partially_funded'/'pending'.
     // Terminal states are owned by the routes and must not be derived.
     const TERMINAL = ['completed', 'cancelled'];
     if (!TERMINAL.includes(this.status)) {
         if (this.fundedAmount >= this.amount) {
             this.status = 'funded';
         } else if (this.fundedAmount > 0) {
             this.status = 'partially_funded';
         } else {
             this.status = 'pending';
         }
     }


    // Proceed with the save operation
    next();
});


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'fundingrequests' (lowercase, plural) for this model.
const FundingRequest = mongoose.model('FundingRequest', fundingRequestSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = FundingRequest;
