// --- Database Seeding Script: seed.js ---

// Import necessary Node.js modules and installed packages
const mongoose = require('mongoose');      // Mongoose for interacting with MongoDB
const dotenv = require('dotenv');          // To load environment variables from .env file
const bcrypt = require('bcrypt');          // Needed to hash passwords for demo users

// Load environment variables from the .env file
// This should be done as early as possible in a script that uses environment variables.
dotenv.config();

// Require all your Mongoose model files. This is crucial so Mongoose
// registers the schemas and models before we try to use them in the script.
const User = require('./models/User');
const Video = require('./models/Video');
const Listing = require('./models/Listing');
const FundingRequest = require('./models/FundingRequest');
const Investment = require('./models/Investment');
const Purchase = require('./models/Purchase');
const Transaction = require('./models/Transaction');
const Conversation = require('./models/Conversation');
const Message = require('./models/Message');
const Notification = require('./models/Notification');


// Get the MongoDB connection string from environment variables
const MONGODB_URI = process.env.MONGODB_URI;

// Helper function to generate a simulated blockchain transaction hash
const generateSimulatedTxHash = (prefix = 'sim_tx') => {
    // Generates a unique-enough string for demo purposes based on timestamp and random number
    return `${prefix}_${Date.now().toString(16)}${Math.random().toString(16).substring(2, 12)}`;
};


// Main asynchronous function to perform the database seeding
const seedDB = async () => {
    try {
        // --- Connect to MongoDB ---
        await mongoose.connect(MONGODB_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            // useCreateIndex: true,    // Deprecated in Mongoose 6.x
            // useFindAndModify: false, // Deprecated in Mongoose 6.x
            serverSelectionTimeoutMS: 5000, // Timeout after 5s if server not found
            connectTimeoutMS: 10000 // Give up initial connection after 10s
        });
        console.log('MongoDB connected successfully for seeding.');

        // --- Clear existing data from all relevant collections ---
        console.log('Clearing existing data...');
        assertSafeToSeed();

        assertSafeToSeed();


        await User.deleteMany({});
        await Video.deleteMany({});
        await Listing.deleteMany({});
        await FundingRequest.deleteMany({});
        await Investment.deleteMany({});
        await Purchase.deleteMany({});
        await Transaction.deleteMany({});
        await Conversation.deleteMany({});
        await Message.deleteMany({});
        await Notification.deleteMany({});
        console.log('Existing data cleared from all collections.');

        // --- Create Demo Users ---
        console.log('Creating demo users...');
        // Using a static password for all demo users, hashed securely
        const demoPasswordPlain = 'password123';
        const demoPasswordHash = await bcrypt.hash(demoPasswordPlain, 10);

        const users = [
            { name: 'Demo Farmer', email: 'demo.farmer@pestivid.sim', role: 'farmer', password: demoPasswordHash, memberSince: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }, // ~30 days ago
            { name: 'Demo Buyer', email: 'demo.buyer@pestivid.sim', role: 'buyer', password: demoPasswordHash, memberSince: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000) }, // ~15 days ago
            { name: 'Demo Investor', email: 'demo.investor@pestivid.sim', role: 'investor', password: demoPasswordHash, memberSince: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000) }, // ~45 days ago
            { name: 'Demo Admin', email: 'demo.admin@pestivid.sim', role: 'admin', password: demoPasswordHash, memberSince: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000) }, // registration refuses role 'admin', so this seed is the only way one exists
            { name: 'Alice Farmer', email: 'alice@example.com', role: 'farmer', password: await bcrypt.hash('alice123', 10), memberSince: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000) }, // ~60 days ago
            { name: 'Bob Buyer', email: 'bob@example.com', role: 'buyer', password: await bcrypt.hash('bob123', 10), memberSince: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000) }, // ~20 days ago
            { name: 'Charlie Investor', email: 'charlie@example.com', role: 'investor', password: await bcrypt.hash('charlie123', 10), memberSince: new Date(Date.now() - 50 * 24 * 60 * 60 * 1000) }, // ~50 days ago
            // Add more demo users here if you want more variety
        ];
        // Insert many users into the database and get the created documents
        const createdUsers = await User.insertMany(users);

        // Find specific users by email or role to use their _id for linking
        const demoFarmerUser = createdUsers.find(u => u.email === 'demo.farmer@pestivid.sim');
        const demoBuyerUser = createdUsers.find(u => u.email === 'demo.buyer@pestivid.sim');
        const demoInvestorUser = createdUsers.find(u => u.email === 'demo.investor@pestivid.sim');
        const aliceUser = createdUsers.find(u => u.email === 'alice@example.com');
        const bobUser = createdUsers.find(u => u.email === 'bob@example.com');
        const charlieUser = createdUsers.find(u => u.email === 'charlie@example.com');

        console.log(`${createdUsers.length} demo users created.`);

        // --- Create Demo Videos (linked to farmers) ---
        console.log('Creating demo videos...');
        const videos = [
            // Videos by Demo Farmer
            { cid: 'bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi', storageType: 'ipfs', videoFileHash: 'sim_hash_tomato_video_1', farmerWallet: demoFarmerUser._id, crop: 'Tomatoes', pesticide: 'Neem Oil', location: 'Green Valley Farm', pesticideCompany: 'Organic Solutions', purpose: 'sell', uploadTimestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000) }, // 2 days ago
            { cid: 'bafybeihwev36kewudjgwrgghmnan2dwf7yas43gq2yck6jcpucsrkkhzwa', storageType: 'ipfs', videoFileHash: 'sim_hash_wheat_video_1', farmerWallet: demoFarmerUser._id, crop: 'Wheat', pesticide: 'Crop Rotation', location: 'North Fields', pesticideCompany: 'Self-Managed', purpose: 'funding', uploadTimestamp: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000) }, // 4 days ago
             { cid: 'bafybeicgmdpvw4duutrmdxl4a2dhlfldsoqodc2llgmljt4g24k5sx4tgy', storageType: 'ipfs', videoFileHash: 'sim_hash_lettuce_video_1', farmerWallet: demoFarmerUser._id, crop: 'Lettuce', pesticide: 'Spinosad', location: 'Sunny Acres', pesticideCompany: 'SafeGrow Ltd.', purpose: 'agristream', uploadTimestamp: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000) }, // 1 day ago
            { cid: 'bafybeif4tcbxn7hizx7xqn36yqurgkznnhz4f25ihf2a75qjpcflm6zape', storageType: 'ipfs', videoFileHash: 'sim_hash_strawberry_video_1', farmerWallet: demoFarmerUser._id, crop: 'Strawberries', pesticide: 'None (Hydro)', location: 'Greenhouse Hydroponics', pesticideCompany: 'HydroFarm', purpose: 'funding', uploadTimestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }, // 7 days ago
            // Videos by Alice Farmer
             { cid: 'bafybeifq72n3z7y3q9f3u8b3x2e2d2c2b2a292827262524232221202f2e2d', storageType: 'ipfs', videoFileHash: 'sim_hash_peppers_video_1', farmerWallet: aliceUser._id, crop: 'Bell Peppers', pesticide: 'Organic', location: 'Field 3', pesticideCompany: 'NatureCare', purpose: 'sell', uploadTimestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000) }, // 3 days ago
             { cid: 'bafybeidr4pzkr5kkh5664gxfb4o73j322v4r5v6w7x8y9z0a1b1c1d1e1f1g', storageType: 'ipfs', videoFileHash: 'sim_hash_corn_video_1', farmerWallet: aliceUser._id, crop: 'Corn', pesticide: 'Integrated Pest Mgmt', location: 'Northwest Plot', pesticideCompany: 'FarmGuard', purpose: 'funding', uploadTimestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000) }, // 10 days ago
        ];
        // Make the seeded videos look like real uploads.
        //
        // services/videoEligibility.js refuses to let a video back a funding
        // request or listing unless it has a server-computed hash AND a
        // fingerprint -- that gate is what stops the whole provenance pipeline
        // being bypassed via the legacy metadata route. Seed data written before
        // that gate existed carries placeholder hashes like
        // 'sim_hash_tomato_video_1' and no fingerprint, so every demo funding flow
        // would be blocked with 'video_not_verified'.
        //
        // These values are SIMULATED, deterministically derived from the CID so
        // reseeding is stable. They stand in for a real upload; they are not
        // claims about real files.
        const crypto = require('crypto');

/**
 * Refuse to wipe a database that is not obviously local or test.
 *
 * This script opens with User.deleteMany({}) and nine more, against whatever
 * MONGODB_URI happens to be set to. backend/.env now points at a live Atlas
 * cluster, so running `node seed.js` in the wrong terminal destroys real farmer
 * records, funding requests and investments -- and a free M0 tier has no backup.
 *
 * Set SEED_CONFIRM=yes to reset a remote database deliberately.
 */
function assertSafeToSeed() {
    const uri = process.env.MONGODB_URI || '';
    const isLocal = /(^|\/\/)(127\.0\.0\.1|localhost)/.test(uri);
    const looksLikeTest = /(test|dev|local|staging)/i.test(uri);
    const confirmed = String(process.env.SEED_CONFIRM || '').toLowerCase() === 'yes';

    if (isLocal || looksLikeTest || confirmed) return;

    const redacted = uri.replace(/\/\/[^@]*@/, '//<credentials>@');
    console.error([
        '',
        'REFUSING TO SEED.',
        '',
        '  MONGODB_URI does not look local or test-like:',
        '    ' + redacted,
        '',
        '  This script deletes every user, video, listing, funding request,',
        '  investment, purchase, transaction, conversation, message and',
        '  notification before inserting demo data.',
        '',
        '  To reset that database on purpose:',
        '    SEED_CONFIRM=yes node seed.js',
        '',
    ].join('\n'));
    process.exit(1);
}





        for (const v of videos) {
            v.videoFileHash = crypto.createHash('sha256')
                .update(`seed:${v.cid}`).digest('hex');
            v.hashComputedBy = 'server';
            v.fingerprint = {
                // 12 distinct 64-bit dHashes, derived from the CID so no two
                // seeded videos collide and trip the duplicate detector.
                frameHashes: Array.from({ length: 12 }, (_, i) => crypto
                    .createHash('sha256').update(`seed:${v.cid}:${i}`)
                    .digest('hex').slice(0, 16)),
                nFrames: 12,
                durationSeconds: 12,
                algorithm: 'dhash64/9x8',
            };
            v.provenance = { flags: [], reviewState: 'none' };
        }

        const createdVideos = await Video.insertMany(videos);

        // Find specific videos to link them to listings/funding requests
        const demoTomatoVideo = createdVideos.find(v => v.crop === 'Tomatoes' && v.farmerWallet.equals(demoFarmerUser._id));
        const demoWheatVideo = createdVideos.find(v => v.crop === 'Wheat' && v.farmerWallet.equals(demoFarmerUser._id));
        const demoStrawberryVideo = createdVideos.find(v => v.crop === 'Strawberries' && v.farmerWallet.equals(demoFarmerUser._id));
        const demoLettuceVideo = createdVideos.find(v => v.crop === 'Lettuce' && v.farmerWallet.equals(demoFarmerUser._id));
        const alicePeppersVideo = createdVideos.find(v => v.crop === 'Bell Peppers' && v.farmerWallet.equals(aliceUser._id));
        const aliceCornVideo = createdVideos.find(v => v.crop === 'Corn' && v.farmerWallet.equals(aliceUser._id));


        console.log(`${createdVideos.length} demo videos created.`);


        // --- Create Demo Listings (linked to farmers and videos) ---
        console.log('Creating demo listings...');
        const listings = [
            // Listing by Demo Farmer
            {
                farmerWallet: demoFarmerUser._id,
                crop: demoTomatoVideo.crop, location: demoTomatoVideo.location, pesticide: demoTomatoVideo.pesticide, pesticideCompany: demoTomatoVideo.pesticideCompany,
                cid: demoTomatoVideo.cid, storageType: demoTomatoVideo.storageType, videoFileHash: demoTomatoVideo.videoFileHash,
                minPrice: 44000, maxPrice: 52000, status: 'active', createdAt: new Date(Date.now() - 1.5 * 24 * 60 * 60 * 1000), // 1.5 days ago
                txHash: generateSimulatedTxHash('sim_list_tomato_demo'), notificationSent: true // Already notified buyers
            },
             {
                 farmerWallet: demoFarmerUser._id,
                 crop: demoLettuceVideo.crop, location: demoLettuceVideo.location, pesticide: demoLettuceVideo.pesticide, pesticideCompany: demoLettuceVideo.pesticideCompany,
                 cid: demoLettuceVideo.cid, storageType: demoLettuceVideo.storageType, videoFileHash: demoLettuceVideo.videoFileHash,
                 minPrice: 28000, maxPrice: 36000, status: 'active', createdAt: new Date(Date.now() - 0.5 * 24 * 60 * 60 * 1000), // 0.5 days ago
                 txHash: generateSimulatedTxHash('sim_list_lettuce_demo'), notificationSent: true // Already notified buyers
             },
             // Listing by Alice Farmer
             {
                farmerWallet: aliceUser._id,
                crop: alicePeppersVideo.crop, location: alicePeppersVideo.location, pesticide: alicePeppersVideo.pesticide, pesticideCompany: alicePeppersVideo.pesticideCompany,
                cid: alicePeppersVideo.cid, storageType: alicePeppersVideo.storageType, videoFileHash: alicePeppersVideo.videoFileHash,
                minPrice: 78000, maxPrice: 92000, status: 'active', createdAt: new Date(Date.now() - 2.5 * 24 * 60 * 60 * 1000), // 2.5 days ago
                txHash: generateSimulatedTxHash('sim_list_peppers_alice'), notificationSent: true // Already notified buyers
            },
            // Add a listing that will be marked as 'sold' later
             {
                farmerWallet: demoFarmerUser._id,
                crop: 'Cucumbers', location: 'Field 2', pesticide: 'Organic Spray', pesticideCompany: 'BioGro',
                cid: 'bafybeidr4pzkr5kkh5664gxfb4o73j322v4r5v6w7x8y9z0a1b1c1d1e1f1h', storageType: 'ipfs', videoFileHash: 'sim_hash_cucumbers_video_1',
                minPrice: 38000, maxPrice: 48000, status: 'active', createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
                txHash: generateSimulatedTxHash('sim_list_cukes_demo'), notificationSent: true
            },
        ];
        const createdListings = await Listing.insertMany(listings);
        const demoTomatoListing = createdListings.find(l => l.crop === 'Tomatoes' && l.farmerWallet.equals(demoFarmerUser._id));
        const demoCucumbersListing = createdListings.find(l => l.crop === 'Cucumbers' && l.farmerWallet.equals(demoFarmerUser._id)); // Find the one to be sold
        const alicePepperListing = createdListings.find(l => l.crop === 'Bell Peppers' && l.farmerWallet.equals(aliceUser._id));


        console.log(`${createdListings.length} demo listings created.`);

        // --- Create Demo Funding Requests (linked to farmers and videos) ---
        console.log('Creating demo funding requests...');
        const fundingRequests = [
            // Request by Demo Farmer
            {
                farmerWallet: demoFarmerUser._id, title: 'Organic Wheat Expansion', crop: demoWheatVideo.crop, acres: 10, amount: 500000, method: 'organic',
                cid: demoWheatVideo.cid, videoStorageType: demoWheatVideo.storageType, videoFileHash: demoWheatVideo.videoFileHash,
                description: 'Expand organic wheat production to meet local demand. Funds for new harvester.', timeline: 6, roi: 15, investorShare: 20,
                fundedAmount: 155000, status: 'partially_funded', createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // 3 days ago
                 investors: [], // Investors will be added below
                 updates: [{id:'upd_wheat_1', date: new Date(Date.now() - 2.8*24*60*60*1000).toLocaleDateString(), text:'Initial planting complete, crops look healthy.'}] // Embedded update
            },
             {
                 farmerWallet: demoFarmerUser._id, title: 'Hydroponic Strawberry Setup', crop: demoStrawberryVideo.crop, acres: 1, amount: 1200000, method: 'hydroponic',
                 cid: demoStrawberryVideo.cid, videoStorageType: demoStrawberryVideo.storageType, videoFileHash: demoStrawberryVideo.videoFileHash,
                 description: 'Setting up a new hydroponic system for year-round strawberry production.', timeline: 4, roi: 22, investorShare: 25,
                 fundedAmount: 0, status: 'pending', createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
                 investors: [], updates: []
             },
             // Request by Alice Farmer
             {
                farmerWallet: aliceUser._id, title: 'Regenerative Corn Pilot', crop: aliceCornVideo.crop, acres: 5, amount: 800000, method: 'regenerative',
                cid: aliceCornVideo.cid,
                videoStorageType: aliceCornVideo.storageType, videoFileHash: aliceCornVideo.videoFileHash,
                description: 'Pilot project for regenerative corn farming on 5 acres. Focus on soil health and biodiversity.', timeline: 9, roi: 18, investorShare: 22,
                fundedAmount: 400000, status: 'partially_funded', createdAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000), // 10 days ago
                investors: [], updates: []
            },
            // Add a request that is already fully funded
             {
                farmerWallet: demoFarmerUser._id, title: 'Green Bean Expansion', crop: 'Green Beans', acres: 3, amount: 300000, method: 'organic',
                cid: demoLettuceVideo.cid, // Using lettuce video as placeholder
                videoStorageType: demoLettuceVideo.storageType, videoFileHash: demoLettuceVideo.videoFileHash,
                description: 'Expanding organic green bean patch. Need funds for trellising materials.', timeline: 3, roi: 12, investorShare: 15,
                fundedAmount: 300000, status: 'funded', createdAt: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000), // 15 days ago
                investors: [], updates: [] // Investors added below
            },
        ];
        const createdFundingRequests = await FundingRequest.insertMany(fundingRequests);

        // Find specific requests to link investments to them
        const demoWheatRequest = createdFundingRequests.find(r => r.crop === 'Wheat' && r.farmerWallet.equals(demoFarmerUser._id));
        const demoStrawberryRequest = createdFundingRequests.find(r => r.crop === 'Strawberries' && r.farmerWallet.equals(demoFarmerUser._id));
        const aliceCornRequest = createdFundingRequests.find(r => r.crop === 'Corn' && r.farmerWallet.equals(aliceUser._id));
        const demoGreenBeansRequest = createdFundingRequests.find(r => r.crop === 'Green Beans' && r.farmerWallet.equals(demoFarmerUser._id));


        console.log(`${createdFundingRequests.length} demo funding requests created.`);

        // --- Create Demo Investments (linked to investor users and requests) ---
        console.log('Creating demo investments...');
        const investments = [
            // Investments by Demo Investor in Demo Farmer's Wheat project
            {
                investorWallet: demoInvestorUser._id,
                projectId: demoWheatRequest._id,
                amount: 100000,
                investmentDate: new Date(Date.now() - 2.5 * 24 * 60 * 60 * 1000), // 2.5 days ago
                txHash: generateSimulatedTxHash('sim_invest_wheat_demo_1'),
                 status: 'growing', progress: 50, // Simulate some progress
            },
             {
                 investorWallet: demoInvestorUser._id,
                 projectId: demoWheatRequest._id,
                 amount: 55000, // Total funded = 100000 + 55000 = 155000, matches wheatRequest.fundedAmount
                 investmentDate: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
                 txHash: generateSimulatedTxHash('sim_invest_wheat_demo_2'),
                  status: 'growing', progress: 50, // Simulate some progress
             },
             // Investment by Charlie Investor in Alice Farmer's Corn project
             {
                 investorWallet: charlieUser._id,
                 projectId: aliceCornRequest._id,
                 amount: 400000, // Matches fundedAmount on request
                 investmentDate: new Date(Date.now() - 9 * 24 * 60 * 60 * 1000), // 9 days ago
                 txHash: generateSimulatedTxHash('sim_invest_corn_charlie_1'),
                  status: 'growing', progress: 30, // Simulate some progress
             },
            // Investment by Demo Investor in Demo Farmer's Green Beans (fully funded)
            {
                investorWallet: demoInvestorUser._id,
                projectId: demoGreenBeansRequest._id,
                amount: 300000, // Matches total amount
                investmentDate: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000), // 14 days ago
                txHash: generateSimulatedTxHash('sim_invest_greenbeans_demo'),
                 status: 'harvested', progress: 100, // Mark as harvested
                 payoutAmount: 300000 * (demoGreenBeansRequest.roi / 100), // Calculate expected payout
                 payoutDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // Payout happened 5 days ago
                 payoutTxHash: generateSimulatedTxHash('sim_payout_greenbeans_demo'),
                 payoutNotified: true,
            },
        ];

         // Update the investors array in the funding requests with the investment details
         // This matches the embedded array structure in the FundingRequest model
         for (const inv of investments) {
              const req = createdFundingRequests.find(r => r._id.equals(inv.projectId));
              if (req) {
                  req.investors.push({
                      investorId: inv.investorWallet,
                      amount: inv.amount,
                      txHash: inv.txHash,
                      investmentDate: inv.investmentDate,
                  });
                  // Don't save here yet, batch saves later if needed, or rely on pre-save hook (though pre-save for embedded is tricky)
                  // Let's just rely on the embedded array being populated correctly in the FundingRequest object itself
              }
         }

         // Save the updated Funding Requests that now have investors listed in their embedded array
         // This is important because the investors array is part of the FundingRequest document
         for (const req of createdFundingRequests) {
             // We only updated the 'investors' array in the objects in our 'createdFundingRequests' array
             // We need to save these changes back to the database.
             await req.save();
         }


        const createdInvestments = await Investment.insertMany(investments);
        // Note: The pre-save hook on Investment will populate the redundant fields like projectTitle, farmerWallet etc.


        console.log(`${createdInvestments.length} demo investments created.`);

        // --- Create Demo Purchases ---
        console.log('Creating demo purchases...');
        // Let's have Bob Buyer purchase the demo farmer's Cucumber listing
        if (demoCucumbersListing && bobUser && demoFarmerUser) {
            const purchaseTxHash = generateSimulatedTxHash('sim_purchase_cukes_bob');
            const purchaseDate = new Date(Date.now() - 0.1 * 24 * 60 * 60 * 1000); // Very recent

            const purchase = new Purchase({
                 buyerWallet: bobUser._id,
                 listingId: demoCucumbersListing._id,
                 price: 49500, // an offer inside the range the farmer listed
                 purchaseDate: purchaseDate,
                 txHash: purchaseTxHash,
                 // farmerWallet, crop, etc. will be populated by pre-save hook
            });
            const createdPurchase = await purchase.save();
            console.log(`Demo purchase created: ${createdPurchase._id}`);

             // Manually update the listing status to 'sold'
             demoCucumbersListing.status = 'sold';
             await demoCucumbersListing.save();
             console.log(`Demo Cucumber listing ${demoCucumbersListing._id} marked as sold.`);

             // Create associated transactions for this purchase
              // Transaction for the buyer (purchase)
              const buyerTransaction = new Transaction({
                  userId: bobUser._id,
                  txHash: purchaseTxHash,
                  type: 'purchase',
                  amount: purchase.price,
                  listingId: demoCucumbersListing._id,
                  date: purchase.purchaseDate,
              });
               await buyerTransaction.save();

              // Transaction for the farmer (sale)
              const farmerTransaction = new Transaction({
                   userId: demoFarmerUser._id, // The farmer who sold
                   txHash: generateSimulatedTxHash('sim_sale_cukes_demo'), // Separate Tx hash for the farmer's sale record
                   type: 'sale',
                   amount: purchase.price,
                   listingId: demoCucumbersListing._id,
                   date: purchase.purchaseDate,
              });
              await farmerTransaction.save();
              console.log('Demo purchase/sale transactions created.');
        } else {
            console.log('Could not create demo purchase: Cucumber listing, Bob buyer, or Demo Farmer user not found.');
        }

        console.log('Demo purchases created.');

        // --- Create Demo Conversations and Messages ---
         console.log('Creating demo conversations...');
         // Participants sorted for consistent finding
         const demoConvParticipants1 = [demoFarmerUser._id, demoBuyerUser._id].sort();
         const demoConvParticipants2 = [demoFarmerUser._id, demoInvestorUser._id].sort();
         const demoConvParticipants3 = [aliceUser._id, bobUser._id].sort(); // Alice and Bob conversation

         const conversations = [
             {
                 participants: demoConvParticipants1, // Demo Farmer <-> Demo Buyer
                 lastMessageSnippet: 'Yes, about the tomatoes...',
                 lastMessageTimestamp: new Date(Date.now() - 0.02 * 24 * 60 * 60 * 1000), // Very recent
             },
              {
                  participants: demoConvParticipants2, // Demo Farmer <-> Demo Investor
                  lastMessageSnippet: 'Thanks for the update!',
                  lastMessageTimestamp: new Date(Date.now() - 0.5 * 24 * 60 * 60 * 1000), // More recent
              },
              {
                  participants: demoConvParticipants3, // Alice Farmer <-> Bob Buyer
                  lastMessageSnippet: 'Are the peppers organic?',
                  lastMessageTimestamp: new Date(Date.now() - 1.2 * 24 * 60 * 60 * 1000),
              }
         ];
         const createdConvs = await Conversation.insertMany(conversations);
         const convFarmerBuyer = createdConvs.find(c => c.participants.map(p=>p.toString()).join(',') === demoConvParticipants1.map(p=>p.toString()).join(','));
         const convFarmerInvestor = createdConvs.find(c => c.participants.map(p=>p.toString()).join(',') === demoConvParticipants2.map(p=>p.toString()).join(','));
         const convAliceBob = createdConvs.find(c => c.participants.map(p=>p.toString()).join(',') === demoConvParticipants3.map(p=>p.toString()).join(','));

         console.log(`${createdConvs.length} demo conversations created.`);

         console.log('Creating demo messages...');
         const messages = [];
          if (convFarmerBuyer) {
             messages.push({ conversationId: convFarmerBuyer._id, sender: demoBuyerUser._id, receiver: demoFarmerUser._id, text: 'Hi farmer!', timestamp: new Date(Date.now() - 0.04 * 24 * 60 * 60 * 1000), read: true }); // Assume farmer read older messages
             messages.push({ conversationId: convFarmerBuyer._id, sender: demoFarmerUser._id, receiver: demoBuyerUser._id, text: 'Hello! Interested?', timestamp: new Date(Date.now() - 0.03 * 24 * 60 * 60 * 1000), read: true }); // Assume buyer read older messages
             messages.push({ conversationId: convFarmerBuyer._id, sender: demoBuyerUser._id, receiver: demoFarmerUser._id, text: 'Yes, about the tomatoes. Are they organic?', timestamp: new Date(Date.now() - 0.02 * 24 * 60 * 60 * 1000), read: false }); // Farmer hasn't read yet
          }
          if (convFarmerInvestor) {
              messages.push({ conversationId: convFarmerInvestor._id, sender: demoInvestorUser._id, receiver: demoFarmerUser._id, text: 'How is the wheat project progressing?', timestamp: new Date(Date.now() - 0.6 * 24 * 60 * 60 * 1000), read: true });
              messages.push({ conversationId: convFarmerInvestor._id, sender: demoFarmerUser._id, receiver: demoInvestorUser._id, text: 'Growth looks good, soil tests positive.', timestamp: new Date(Date.now() - 0.55 * 24 * 60 * 60 * 1000), read: true });
              messages.push({ conversationId: convFarmerInvestor._id, sender: demoInvestorUser._id, receiver: demoFarmerUser._id, text: 'Thanks for the update!', timestamp: new Date(Date.now() - 0.5 * 24 * 60 * 60 * 1000), read: false }); // Farmer hasn't read yet
          }
          if (convAliceBob) {
              messages.push({ conversationId: convAliceBob._id, sender: bobUser._id, receiver: aliceUser._id, text: 'Hi Alice, interested in your peppers.', timestamp: new Date(Date.now() - 1.5 * 24 * 60 * 60 * 1000), read: true });
               messages.push({ conversationId: convAliceBob._id, sender: aliceUser._id, receiver: bobUser._id, text: 'Great! What info do you need?', timestamp: new Date(Date.now() - 1.4 * 24 * 60 * 60 * 1000), read: true });
               messages.push({ conversationId: convAliceBob._id, sender: bobUser._id, receiver: aliceUser._id, text: 'Are they organic?', timestamp: new Date(Date.now() - 1.2 * 24 * 60 * 60 * 1000), read: false }); // Alice hasn't read yet
          }

         const createdMessages = await Message.insertMany(messages);
         console.log(`${createdMessages.length} demo messages created.`);


        // --- Create Demo Notifications ---
         console.log('Creating demo notifications...');
         const notifications = [
              // Notification for Demo Farmer about purchase
             { recipient: demoFarmerUser._id, type: 'purchase', message: `Your listing "${demoCucumbersListing?.crop || 'N/A'}" was purchased by Bob Buyer!`, timestamp: new Date(Date.now() - 0.1 * 24 * 60 * 60 * 1000), read: false, itemId: demoCucumbersListing?._id, itemType: 'Listing' },
              // Notification for Demo Investor about investment
             { recipient: demoInvestorUser._id, type: 'investment', message: 'Your investment of ₹1,00,000 in the wheat season successful!', timestamp: new Date(Date.now() - 2.5 * 24 * 60 * 60 * 1000), read: false, itemId: investments.find(i => i.projectId.equals(demoWheatRequest._id) && i.amount === 10.0)?._id, itemType: 'Investment' },
             { recipient: demoInvestorUser._id, type: 'investment', message: 'Your investment of ₹55,000 in the wheat season successful!', timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), read: false, itemId: investments.find(i => i.projectId.equals(demoWheatRequest._id) && i.amount === 5.5)?._id, itemType: 'Investment' },
              // Notification for Demo Investor about update (assuming farmer added an update)
             { recipient: demoInvestorUser._id, type: 'update', message: `Update posted for project "${demoWheatRequest?.title || 'N/A'}": Initial planting...`, timestamp: new Date(Date.now() - 2.8 * 24 * 60 * 60 * 1000), read: true, itemId: demoWheatRequest?._id, itemType: 'FundingRequest' }, // Mark read
             // Notification for Demo Investor about payout (Green Beans)
             { recipient: demoInvestorUser._id, type: 'payout', message: `Payout received for "Green Bean Expansion": ${investments.find(i => i.projectId.equals(demoGreenBeansRequest._id))?.payoutAmount?.toLocaleString('en-IN') || 'N/A'}!`, timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), read: false, itemId: investments.find(i => i.projectId.equals(demoGreenBeansRequest._id))?._id, itemType: 'Investment' },


             // Global notifications
             { global: true, type: 'info', message: 'Welcome to PestiVid Demo Platform! Explore features.', timestamp: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000), read: false }, // Older global notification
              { global: true, type: 'listing', message: 'New listing: "Lettuce" from Demo Farmer is available!', timestamp: new Date(Date.now() - 0.5 * 24 * 60 * 60 * 1000), read: false, itemId: demoLettuceVideo?._id, itemType: 'Listing' }, // New global listing notification
               { global: true, type: 'funding', message: 'New opportunity: "Regenerative Corn Pilot" by Alice Farmer!', timestamp: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000), read: false, itemId: aliceCornRequest?._id, itemType: 'FundingRequest' }, // New global funding notification
         ];
         const createdNotifications = await Notification.insertMany(notifications);
          console.log(`${createdNotifications.length} demo notifications created.`);

        // --- Create Demo Avatar Messages (Optional) ---
        console.log('Creating demo avatar messages...');


        console.log('\nDatabase seeding complete!');

    } catch (error) {
        console.error('Database seeding failed:', error);
         // Log specific Mongoose validation errors if any
         if (error.errors) {
             console.error('Mongoose Validation Errors:', error.errors);
         }
         if (error.code === 11000) {
             console.error('MongoDB Duplicate Key Error:', error.message);
         }

    } finally {
         // Close the MongoDB connection regardless of success or failure
        if (mongoose.connection.readyState === 1) {
             await mongoose.connection.close();
             console.log('MongoDB connection closed.');
        }
        // You might not want to exit the process immediately if running as a module
        // For a standalone seed script, process.exit() is appropriate.
        // process.exit(error ? 1 : 0); // Exit with 1 on error, 0 otherwise
    }
};

// Execute the seeding function
seedDB();
