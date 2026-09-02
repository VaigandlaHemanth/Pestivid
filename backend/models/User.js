// --- Mongoose Model: User.js ---

const mongoose = require('mongoose'); // Import Mongoose library
const bcrypt = require('bcrypt');     // Import bcrypt for password hashing

// Define the schema for the User model
const userSchema = new mongoose.Schema({
    // MongoDB automatically adds an _id field (ObjectId) which serves as the primary key.
    // We don't need to define a separate 'id' field unless we have a specific reason
    // to use a custom ID format instead of MongoDB's default ObjectId.

    name: {
        type: String,
        required: [true, 'Name is required'], // Name is required, with a custom error message
        trim: true // Remove leading/trailing whitespace
    },
    email: {
        type: String,
        required: [true, 'Email is required'],
        unique: true, // Ensure email addresses are unique across all users
        trim: true,
        lowercase: true, // Store email in lowercase for case-insensitive uniqueness
        // Basic email format validation (more robust validation can be added)
        match: [/\S+@\S+\.\S+/, 'Please use a valid email address']
    },
    role: {
        type: String,
        required: [true, 'Role is required'],
        enum: {
            // NOTE: there is deliberately no 'admin' role. Several routes used to
            // read `req.user.role !== 'admin'` as an escape hatch; because 'admin'
            // was never a valid value those branches were unreachable dead code and
            // have been removed. If an admin role is ever added, it must come with a
            // real grant path -- never via self-service registration or profile edit.
            // 'admin' reviews provenance flags. It is NOT self-assignable:
            // routes/auth.js refuses it at registration, so the only way to get
            // it is a direct database change by whoever runs the platform. Adding
            // a privileged value to a public enum without that guard would be
            // handing it out.
            // 'member' is the one kind of account there is now: it films, funds
            // and buys. farmer, buyer and investor are the words accounts made
            // before the merge carry; they mean member.
            values: ['member', 'farmer', 'buyer', 'investor', 'admin'],
            message: 'Invalid role. Must be farmer, buyer, or investor.'
        }
    },
    password: {
        type: String,
        required: [true, 'Password is required'],
        // Eight, matching the create-account screen and the auth routes. Six was
        // the farmer's six-digit code, and that page no longer exists.
        minlength: [8, 'Password must be at least 8 characters long']
        // NOTE: In a real production app, password policies should be much stronger.
    },
    createdAt: {
        type: Date,
        default: Date.now // Default to the current date and time when a document is created
    },
    phone: { // Optional phone number field
        type: String,
        trim: true
    },
    // Bumped to invalidate every token already issued for this user.
    //
    // The JWT is stateless, so before this there was NO way to revoke one: a
    // leaked token stayed valid for its full 24 hours, changing a password left
    // every existing session alive, and a role baked into a token at login kept
    // working for 24h after the role changed. The seeded demo credentials are
    // public, which makes all three worse.
    //
    // Every token carries the tokenVersion it was signed with, and
    // authenticateToken rejects any token whose version is behind. Bumping this
    // is therefore "sign out everywhere", and it is cheap.
    tokenVersion: {
        type: Number,
        default: 0,
        required: true,
    },

    memberSince: { // Date the user joined PestiVid
        type: Date,
        default: Date.now // Default to creation date, can be overridden
    },
    authMethod: { // How the user was authenticated/created (for demo purposes)
        type: String,
        default: 'email_pass_demo' // e.g., 'email_pass', 'wallet_connect', 'demo_seeded'
    },
    // walletAddress: { // Field to store the user's blockchain wallet address (e.g., Solana address)
    //     type: String,
    //     unique: true, // Wallet address should be unique
    //     sparse: true // Allows multiple documents to have null/undefined walletAddress
    //     // Required: true // Make required if wallet connection is mandatory for signup
    // },
});

// --- Mongoose Middleware (Hooks) ---

// Pre-save hook: This function runs before a User document is saved to the database.
// We use it to hash the password if it's new or has been modified.
userSchema.pre('save', async function(next) {
    // 'this' refers to the document being saved

    // Only hash the password if it has been modified (or is new)
    if (this.isModified('password')) {
        try {
            // Generate a salt (random value used in hashing)
            const salt = await bcrypt.genSalt(10); // 10 is a common cost factor
            // Hash the plain-text password using the salt
            this.password = await bcrypt.hash(this.password, salt);
        } catch (err) {
            // If hashing fails, pass the error to the next middleware or save operation
            return next(err);
        }
    }

     // Ensure memberSince is set if it's a new document and wasn't provided
     if (this.isNew && !this.memberSince) {
         this.memberSince = this.createdAt; // Default to creation date
     }

     // Optional: Simulate phone number and memberSince for seeded/demo users if not provided
     // This logic might be better in the seeding script, but can be here for robustness
      if (this.isNew) {
          // Simulate phone if not provided
           if (!this.phone) {
              let hash = 0;
              for (let i = 0; i < (this._id ? this._id.toString() : '').length; i++) {
                  hash = (hash << 5) - hash + (this._id.toString()).charCodeAt(i);
                  hash |= 0;
              }
              const num = Math.abs(hash % 9000) + 1000;
              const area = Math.abs(hash % 800) + 100;
              const prefix = Math.abs(hash % 800) + 100;
              this.phone = `(${area}) ${prefix}-${num}`;
           }
           // Simulate memberSince if not provided (already handled above, but double check if needed)
           // if (!this.memberSince) { ... }
      }


    // Proceed with the save operation
    next();
});


// --- User Schema Methods ---

// Method to compare an entered password with the hashed password stored in the document.
// This will be used during login.
userSchema.methods.comparePassword = async function(candidatePassword) {
    try {
        // Use bcrypt.compare to safely compare the plain candidatePassword with the hashed password (this.password)
        const isMatch = await bcrypt.compare(candidatePassword, this.password);
        return isMatch; // Returns true if they match, false otherwise
    } catch (err) {
        // If comparison fails (e.g., due to hashing error), throw the error
        throw new Error('Error comparing passwords'); // Or pass error to callback/promise rejection
    }
};

// Method to return a "public" version of the user document, excluding sensitive fields.
// This is used when sending user data to the frontend (e.g., after login, in profiles).
userSchema.methods.getPublicProfile = function() {
    // 'this' refers to the user document instance
    return {
        _id: this._id.toString(), // Convert ObjectId to string for frontend consistency
        name: this.name,
        email: this.email, // Include email in public profile for this demo
        role: this.role,
        phone: this.phone, // Include phone in public profile for this demo
        memberSince: this.memberSince ? this.memberSince.toISOString() : null, // Send date as ISO string
        createdAt: this.createdAt ? this.createdAt.toISOString() : null, // Send date as ISO string
        // walletAddress: this.walletAddress, // Include wallet address if it exists
        displayIdentifier: this.name.split(' ')[0] || this._id.toString().substring(0,6) + '...', // Simple display name
    };
};


// --- Create and Export the Model ---

// Create the Mongoose model from the schema.
// Mongoose automatically creates a collection named 'users' (lowercase, plural) for this model.
const User = mongoose.model('User', userSchema);

// Export the model so it can be used in other files (like route handlers)
module.exports = User;
