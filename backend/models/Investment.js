// --- Mongoose Model: Investment.js ---

const mongoose = require('mongoose'); // Import Mongoose library
// No need to explicitly require FundingRequest here, as we access it via mongoose.model() in the hook

// Define the schema for the Investment model
const investmentSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) as the primary key for each investment document.

    investorWallet: { // Reference to the User document of the investor who made this investment
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User', // Links to the 'User' model
        required: [true, 'Investor ID is required']
    },
    projectId: { // Reference to the FundingRequest document that was invested in
        type: mongoose.Schema.Types.ObjectId,
        ref: 'FundingRequest', // Links to the 'FundingRequest' model
        required: [true, 'Project ID is required']
    },

    // Fields populated from the associated FundingRequest for denormalization
    // This makes querying an investor's portfolio faster without always populating the full project
    projectTitle: { type: String }, // Copied from project
    farmerWallet: { type: mongoose.Schema.Types.ObjectId, ref: 'User' }, // Farmer ID (copied from project)
    crop: { type: String }, // Copied from project
    method: { type: String }, // Copied from project
    description: { type: String }, // Copied from project
    acres: { type: Number }, // Copied from project
    timeline: { type: Number }, // Copied from project (months)
    roi: { type: Number }, // Expected ROI (%) (copied from project)
    investorShare: { type: Number }, // Investor share (%) for this project (copied from project)
    // NO DEFAULT, on purpose. With default:'profit_share' this field was always
    // truthy, so the settlement code's
    //     investment.settlementMode || project.settlementMode || 'profit_share'
    // could never reach the project -- a project agreed as full_repayment settled
    // as profit share, and the investor lost the whole agreed return. The mode is
    // copied from the project below; undefined here means "ask the project".
    settlementMode: { type: String, enum: ['profit_share', 'full_repayment'] },
    // How payoutAmount was arrived at, so an investor can audit the number
    // instead of being handed a bare figure.
    payoutBasis: { type: String },
    cid: { type: String }, // Video CID from the project
    videoStorageType: { type: String }, // Video storage type from the project
    videoFileHash: { type: String }, // Video file hash from the project


    // Investment-specific details:
    amount: { // The specific amount of SOL invested by this investor in this project
        type: Number,
        required: [true, 'Investment amount is required'],
        min: [0, 'Investment amount cannot be negative']
    },
    status: { // Status of this specific investment (might follow project status, or track payout)
        type: String,
        default: 'active', // Initial status
        enum: {
            values: ['active', 'growing', 'harvested', 'cancelled'], // Possible states for an individual investment
            // 'active': Investment made, project is pending/partially funded
            // 'growing': Project is partially funded/funded and in progress
            // 'harvested': Project is completed, payout simulated/processed
            // 'cancelled': Project was cancelled before completion
            message: 'Invalid investment status.'
        }
    },
    progress: { // Simulated progress for this investment (mirrors project progress for demo)
        type: Number,
        default: 0,
        min: [0, 'Progress cannot be negative'],
        max: [100, 'Progress cannot exceed 100']
    },
    investmentDate: {
        type: Date,
        default: Date.now // Timestamp when this investment was made
    },
    txHash: { // Simulated blockchain transaction hash for this investment transaction
        type: String,
        trim: true
    },

    // Payout details (populated when project status becomes 'harvested' or 'completed')
    payoutAmount: { // Actual calculated payout amount (simulated)
        type: Number
        // Can be calculated when status changes to 'harvested'
    },
    payoutDate: { // Date the payout was simulated
        type: Date
    },
    payoutTxHash: { // Simulated transaction hash for the payout transaction
        type: String,
        trim: true
    },
    payoutNotified: { // Flag to track if the payout notification was sent
        type: Boolean,
        default: false
    },


    // Add other fields relevant to an individual investment if needed, e.g.:
    // shareOfProject: Number, // Calculated percentage share based on investment amount / total funded amount
});

// --- Mongoose Middleware (Hooks) ---

// Pre-save hook: This runs before saving a document.
// Used here to automatically populate redundant fields from the linked FundingRequest document
// when a new investment is created, based on the provided projectId.
investmentSchema.pre('save', async function(next) {
    // 'this' refers to the document being saved

    // Only run this logic if it's a new document AND a projectId is present
    if (this.isNew && this.projectId) {
        try {
            // Find the corresponding FundingRequest document by projectId
            // Use .select() to get only necessary fields (matching the redundant fields in the schema)
            // Use .lean() to get a plain JavaScript object, faster for reading
            const FundingRequest = mongoose.model('FundingRequest'); // Get the model within the hook
            const project = await FundingRequest.findById(this.projectId).select('title farmerWallet crop method description acres timeline roi investorShare settlementMode cid videoStorageType videoFileHash').lean();

            if (project) {
                // Populate the investment fields from the project document
                this.set({
                    projectTitle: project.title,
                    farmerWallet: project.farmerWallet, // Populate the farmer's ID
                    crop: project.crop,
                    method: project.method,
                    description: project.description,
                    acres: project.acres,
                    timeline: project.timeline,
                    roi: project.roi,
                    investorShare: project.investorShare,
                    // The agreed settlement terms, frozen at investment time so a
                    // later edit to the project cannot change what was promised.
                    settlementMode: project.settlementMode || 'profit_share',
                    cid: project.cid,
                    videoStorageType: project.videoStorageType,
                    videoFileHash: project.videoFileHash,
                });
                console.log(`Populated investment fields from project ${this.projectId}`);
            } else {
                // If the project is not found, this investment might be orphaned or invalid.
                // Log a warning or prevent saving depending on strictness.
                console.warn(`Validation Warning: FundingRequest with ID ${this.projectId} not found during investment save.`);
                // Depending on strictness, you might want to throw an error here:
                // const error = new Error('Associated project not found.');
                // error.name = 'ValidationError'; // Mimic Mongoose validation error
                // return next(error);
                // For this demo, we'll just log and save with potentially missing project data.
            }
        } catch (err) {
            // If there's a database error during the project lookup
            console.error('Error in investment pre-save hook:', err);
            return next(err); // Pass the error to Mongoose
        }
    }

    // Proceed with the save operation
    next();
});


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'investments' (lowercase, plural) for this model.
const Investment = mongoose.model('Investment', investmentSchema);

// Export the model so it can be used in other files (like route handlers or other models)
module.exports = Investment;
