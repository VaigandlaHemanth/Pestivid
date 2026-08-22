// --- Backend Routes: investments.js ---

const express = require('express');         // Import Express
const router = express.Router();            // Create a new router instance
const mongoose = require('mongoose');       // Import Mongoose
const Investment = mongoose.model('Investment'); // Get the Investment Mongoose model
const FundingRequest = mongoose.model('FundingRequest'); // Get FundingRequest model to update it
const Transaction = mongoose.model('Transaction'); // Get Transaction model to record investments/payouts
const { authenticateToken } = require('./auth'); // Import the authentication middleware

/**
 * First name of a populated user ref, or a safe fallback.
 *
 * Formatting a response must never be able to throw. The settlement handler used
 * `investment.farmerWallet.name.split(' ')[0]` on a ref that was not populated,
 * so it returned HTTP 500 on every call -- after the payout Transaction had
 * already been written, leaving the caller unable to tell whether the money had
 * moved.
 */
function firstName(userRef, fallback) {
    if (!userRef || typeof userRef !== 'object') return fallback || 'Unknown';
    const name = userRef.name;
    if (typeof name === 'string' && name.trim()) return name.trim().split(' ')[0];
    return userRef.displayIdentifier || fallback || 'Unknown';
}

const Notification = mongoose.model('Notification'); // Get Notification model for notifications


// Helper function to generate a simulated blockchain transaction hash
const generateSimulatedTxHash = (prefix = 'sim_tx') => {
    // Generates a unique-enough string for demo purposes based on timestamp and random number
    return `${prefix}_${Date.now().toString(16)}${Math.random().toString(16).substring(2, 12)}`;
};


// --- Investment Routes ---

// @route GET /api/investments/project/:projectId
// @desc  The investments in one project, for the farmer who owns it.
// @access Private (farmer, owner of the project only)
//
// Why this exists: settlement is driven per investment through
// PUT /api/investments/:id/progress, but nothing gave the farmer the list of
// investment ids in their own project. The FundingRequest holds an embedded
// `investors` array with investor ids and amounts, but not the Investment
// document ids, so the UI had no way to reach the settlement path at all. The
// entire payout pipeline was complete on the server and unreachable from the app,
// which meant no investor could ever actually be paid.
//
// This deliberately does NOT settle anything -- it only lists. Settlement stays
// in the one audited, tested endpoint rather than being duplicated here.
//
// Scope: the investor's identity is included because the farmer already sees it
// in the project's embedded investors array, but nothing else about the investor
// is exposed.
router.get('/project/:projectId', authenticateToken, async (req, res) => {
    const { projectId } = req.params;

    if (!mongoose.Types.ObjectId.isValid(projectId)) {
        return res.status(400).json({ message: 'Invalid project ID format.' });
    }

    try {
        const project = await FundingRequest.findById(projectId)
            .select('farmerWallet title status harvestReportedAt outcome '
                  + 'harvestRevenue inputCostBasis investorShare settlementMode')
            .lean();

        if (!project) {
            return res.status(404).json({ message: 'Project not found.' });
        }
        if (String(project.farmerWallet) !== String(req.user._id)) {
            console.warn(`Forbidden: user ${req.user._id} asked for investments in `
                       + `project ${projectId} owned by ${project.farmerWallet}`);
            return res.status(403).json({
                message: 'Forbidden: you can only see investments in your own projects.',
            });
        }

        const investments = await Investment.find({ projectId })
            .select('_id amount status progress payoutAmount payoutDate '
                  + 'settlementMode investorWallet investmentDate')
            .populate('investorWallet', 'name displayIdentifier')
            .sort({ investmentDate: 1 })
            .lean();

        const profit = Math.max(0,
            (project.harvestRevenue || 0) - (project.inputCostBasis || 0));

        return res.json({
            project: {
                _id: String(project._id),
                title: project.title,
                status: project.status,
                harvestReported: Boolean(project.harvestReportedAt),
                harvestReportedAt: project.harvestReportedAt || null,
                outcome: project.outcome || null,
                harvestRevenue: project.harvestRevenue != null ? project.harvestRevenue : null,
                inputCostBasis: project.inputCostBasis != null ? project.inputCostBasis : null,
                investorShare: project.investorShare,
                settlementMode: project.settlementMode || null,
                profit,
                investorPool: Math.round(profit * ((project.investorShare || 0) / 100) * 100) / 100,
            },
            investments: investments.map((i) => ({
                _id: String(i._id),
                amount: i.amount,
                status: i.status,
                progress: i.progress,
                settled: i.payoutAmount != null,
                payoutAmount: i.payoutAmount != null ? i.payoutAmount : null,
                payoutDate: i.payoutDate || null,
                investorName: i.investorWallet ? i.investorWallet.name : 'Investor',
                investorIdentifier: i.investorWallet
                    ? i.investorWallet.displayIdentifier : null,
                investmentDate: i.investmentDate || null,
            })),
        });
    } catch (err) {
        console.error(`GET /api/investments/project/${projectId} error:`, err.message);
        return res.status(500).json({ message: 'Server error loading project investments.' });
    }
});


// @route GET /api/investments/investor/:investorId
// @desc Get all investments made by a specific investor (for their portfolio)
// @access Private (Requires authentication. User should typically only fetch their own investments)
router.get('/investor/:investorId', authenticateToken, async (req, res) => {
    const investorId = req.params.investorId; // Get the investor ID from the URL parameter
    const userId = req.user._id;           // Authenticated user's ID

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(investorId)) {
        return res.status(400).json({ message: 'Invalid Investor ID format.' }); // 400 Bad Request
    }

    // Authorization: Ensure the authenticated user is the investor whose portfolio is being requested.
    // Or allow an admin user to view any investor's portfolio.
    if (userId.toString() !== investorId.toString()) {
         console.warn(`Authorization failed: User ${userId} attempted to view investments for user ${investorId}`);
        return res.status(403).json({ message: "Forbidden: You can only view your own investments." }); // 403 Forbidden
    }

    try {
        // Find investment documents for the specified investor
        // Populate related project and farmer details for display in the portfolio
        const investments = await Investment.find({ investorWallet: investorId })
                                           .populate('projectId', 'title status fundedAmount amount updates') // Populate related project details
                                           .populate('farmerWallet', 'name role displayIdentifier') // Populate farmer details
                                            // Note: Redundant fields like projectTitle, crop, roi etc. are stored directly in Investment
                                            // document, so we don't need to populate them here if that's sufficient.
                                           .sort({ investmentDate: -1 }); // Sort by newest investment first

        // Map to format for frontend (e.g., ensure _id is string, format dates)
         const formattedInvestments = investments.map(investment => {
             // Determine project status based on the populated project if available, otherwise use investment status
             const projectStatus = investment.projectId ? investment.projectId.status : 'unknown';
             const projectUpdates = investment.projectId ? (investment.projectId.updates || []) : [];

             return {
                 _id: investment._id.toString(),
                 investorWallet: investment.investorWallet.toString(), // Investor user ID string
                 projectId: investment.projectId ? investment.projectId._id.toString() : null, // Project ID string
                 projectTitle: investment.projectTitle, // Redundant field
                 farmerWallet: investment.farmerWallet ? investment.farmerWallet._id.toString() : null, // Farmer user ID string
                 farmerName: firstName(investment.farmerWallet, 'Unknown Farmer'),
                 crop: investment.crop, // Redundant field
                 method: investment.method, // Redundant field
                 description: investment.description, // Redundant field
                 acres: investment.acres, // Redundant field
                 timeline: investment.timeline, // Redundant field
                 roi: investment.roi, // Redundant field
                 investorShare: investment.investorShare, // Redundant field
                 cid: investment.cid, // Redundant video CID
                 videoStorageType: investment.videoStorageType, // Redundant
                 videoFileHash: investment.videoFileHash, // Redundant

                 amount: investment.amount, // Amount invested by this investor

                 // Use investment status for portfolio display
                 status: investment.status, // Status of this specific investment
                 progress: investment.progress, // Simulated progress of this specific investment

                 investmentDate: investment.investmentDate ? investment.investmentDate.toISOString() : null, // ISO string date
                 txHash: investment.txHash, // Investment transaction hash

                 // How this investment settles. The UI needs it to describe the
                 // terms truthfully: 'full_repayment' returns principal only,
                 // 'profit_share' returns principal PLUS investorShare% of the
                 // realised profit. Without it the frontend was left guessing,
                 // and it guessed wrong.
                 settlementMode: investment.settlementMode || null,

                 // Payout details
                 payoutAmount: investment.payoutAmount,
                 payoutDate: investment.payoutDate ? investment.payoutDate.toISOString() : null,
                 payoutTxHash: investment.payoutTxHash,
                 payoutNotified: investment.payoutNotified,

                 // Include project updates if needed for the investment details modal
                 updates: projectUpdates.map(update => ({
                     _id: update._id ? update._id.toString() : null,
                     date: update.date, // Keeping string format
                     text: update.text,
                 })),
             };
         });


        // Send the list of investments for this investor
        res.json(formattedInvestments); // Default status is 200 OK

    } catch (err) {
        console.error(`GET /api/investments/investor/${investorId} error:`, err);
        res.status(500).json({ message: 'Server error fetching investments.' }); // 500 Internal Server Error
    }
});


// @route POST /api/investments
// @desc Create a new investment record and update the corresponding funding request
// @access Private (Requires authentication. Must be an investor.)
router.post('/', authenticateToken, async (req, res) => {
    // Authorization: Ensure the authenticated user has the 'investor' role.
    if (req.user.role !== 'investor') {
         console.warn(`Authorization failed: User ${req.user._id} with role ${req.user.role} attempted to create an investment.`);
        return res.status(403).json({ message: "Forbidden: Only users with the 'investor' role can make investments." }); // 403 Forbidden
    }

    // Extract investment details from the request body
    const { projectId, amount } = req.body; // Get the project ID and investment amount

    // Basic input validation
    if (!projectId || amount === undefined) {
         return res.status(400).json({ message: "Missing required investment fields (projectId, amount)." }); // 400 Bad Request
    }

    const parsedAmount = parseFloat(amount);
    if (isNaN(parsedAmount) || parsedAmount <= 0) {
         return res.status(400).json({ message: "Invalid investment amount. Amount must be a positive number." }); // 400 Bad Request
    }
     // Validate projectId format
     if (!mongoose.Types.ObjectId.isValid(projectId)) {
         return res.status(400).json({ message: 'Invalid Project ID format.' }); // 400 Bad Request
     }


    // --- Transaction and Update Logic (Wrap in a transaction if using replica set) ---
    // In production, for operations that update multiple documents (like this, updating FundingRequest and creating Investment/Transaction),
    // you should use MongoDB transactions if your deployment is a replica set. This ensures
    // either all operations succeed or all fail, maintaining data consistency.
    // For a standalone MongoDB server or simple demo, we'll just perform operations sequentially.

    try {
        // --- 1. Find and Update the Funding Request ---
        // Find the funding request by ID
        // Use findByIdAndUpdate with $inc for atomic increment and $push for adding investor to embedded array
        // Identifiers are generated BEFORE the write so they can be used as the
        // join key, instead of searching the embedded array afterwards by
        // (investor, exact float amount, 5-second window) -- which returned the
        // wrong entry for two same-amount investments inside that window.
        const txHash = generateSimulatedTxHash('sim_invest');
        const investmentDate = new Date();

        // Atomic, guarded increment. `findByIdAndUpdate` with $inc alone was
        // atomic but UNCONDITIONAL, so nothing stopped funding past the goal --
        // and findOneAndUpdate does not run the pre('save') clamp in
        // FundingRequest.js, so this filter is the only real ceiling.
        // Atomic, guarded increment expressed as an aggregation-pipeline update.
        // Two reasons for the pipeline form rather than plain $inc/$push:
        //   1. The filter must reject anything that would push past the goal --
        //      plain $inc is atomic but unconditional, so nothing stopped
        //      over-funding.
        //   2. findOneAndUpdate does NOT run the pre('save') hook in
        //      FundingRequest.js, so the status transition has to happen here.
        //      Pipeline stages run in order, so stage 2 sees stage 1's new
        //      fundedAmount. Terminal states are never overwritten.
        const updatedFundingRequest = await FundingRequest.findOneAndUpdate(
            {
                _id: projectId,
                status: { $in: ['pending', 'partially_funded'] },
                // Half-a-paisa tolerance, and it is not sloppiness -- it is the
                // fix for a deadlock. fundedAmount accumulates as a double, so
                // after a few investments it holds a value like 3333.33 whose
                // binary representation is slightly off. The remaining goal is
                // ADVERTISED as (amount - fundedAmount).toFixed(2), e.g.
                // "6666.67 still available", but 3333.33 + 6666.67 evaluates to
                // 10000.000000000002 -- greater than the 10000 goal -- so the
                // exact figure the API just told the investor to send was
                // refused, every time, leaving the project permanently
                // unfundable at the last rupee.
                //
                // 0.005 is smaller than the smallest representable over-fund
                // (one paisa, 0.01), so this admits the exact remainder and
                // still rejects real over-funding. The proper fix is integer
                // paise throughout, which is a schema-wide change; this is the
                // correct behaviour in the meantime.
                $expr: {
                    $lte: [
                        { $add: ['$fundedAmount', parsedAmount] },
                        { $add: ['$amount', 0.005] },
                    ],
                }
            },
            [
                {
                    $set: {
                        // $round keeps the stored total from drifting to
                        // 10000.000000000002 and being rendered that way.
                        fundedAmount: { $round: [{ $add: ['$fundedAmount', parsedAmount] }, 2] },
                        investors: {
                            $concatArrays: ['$investors', [{
                                investorId: new mongoose.Types.ObjectId(req.user._id),
                                amount: parsedAmount,
                                txHash: txHash,
                                investmentDate: investmentDate
                            }]]
                        }
                    }
                },
                {
                    $set: {
                        status: {
                            $cond: [
                                // Same tolerance, mirrored: a project funded to
                                // within half a paisa of its goal is funded, not
                                // "partially funded" forever.
                                { $gte: ['$fundedAmount', { $subtract: ['$amount', 0.005] }] },
                                'funded',
                                'partially_funded'
                            ]
                        }
                    }
                }
            ],
            { new: true } // Return the updated document
        );

        // The filter can miss for three different reasons -- tell them apart so
        // the investor gets an accurate message rather than a bare 404.
        if (!updatedFundingRequest) {
            const existing = await FundingRequest.findById(projectId).select('amount fundedAmount status');
            if (!existing) {
                return res.status(404).json({ message: 'Funding request not found.' });
            }
            if (!['pending', 'partially_funded'].includes(existing.status)) {
                return res.status(409).json({ message: `This project is not accepting investment (status: ${existing.status}).` });
            }
            // Round to paise the same way the ceiling now compares, so the figure
            // quoted here is a figure that will actually be accepted.
            const remaining = Math.max(0, Math.round((existing.amount - existing.fundedAmount) * 100) / 100);
            return res.status(409).json({
                message: `Amount exceeds the remaining goal. ${remaining.toFixed(2)} still available.`,
                code: 'exceeds_remaining',
                remaining,
            });
        }

        // Check if the investment amount exceeded the remaining needed amount BEFORE the update
        // The update already happened, so check based on the updated fundedAmount.
        // This check is better done before the update to provide a precise error message.
        // A better flow: Find -> Validate amount -> If valid, perform $inc and $push -> Check status transitions.
        // Let's re-fetch the request after the update to get the correct total funded amount status.
        // Or, rely on the pre-save hook in FundingRequest.js to update status based on fundedAmount.
        // Assuming the pre-save hook handles funded/partially_funded status transition.

        // --- 2. Create the Investment Document (Main Record) ---
        // No lookup needed: txHash and investmentDate were generated above and
        // written into the embedded entry, so they are already the join key.

        const newInvestment = new Investment({
            investorWallet: req.user._id, // Link to the authenticated investor
            projectId: updatedFundingRequest._id, // Link to the funding request
            amount: parsedAmount,
            txHash: txHash,
            investmentDate: investmentDate,
            // Redundant fields (projectTitle, farmerWallet, crop, etc.) will be populated by the pre-save hook on Investment.js
            status: 'active', // Initial status for an individual investment record
            progress: 0, // Initial progress
        });

        const savedInvestment = await newInvestment.save();


        // --- 3. Create the Transaction Record ---
         const transaction = new Transaction({
             userId: req.user._id, // The investor's ID
             txHash: savedInvestment.txHash, // Use the same transaction hash as the investment record
             type: 'investment', // Transaction type
             amount: savedInvestment.amount, // Amount invested
             projectId: savedInvestment.projectId, // Link to the project
             date: savedInvestment.investmentDate, // Date of the investment transaction
         });
         await transaction.save();
         console.log(`SIMULATING: Created transaction ${transaction._id} for investor ${req.user._id} for investment ${savedInvestment._id}`);


        // --- 4. Send Notifications ---
        // Notify the farmer about the new investment
         try {
              const farmerNotification = new Notification({
                  recipient: updatedFundingRequest.farmerWallet, // Farmer's ID
                  type: 'investment', // Custom notification type
                  message: `${firstName(req.user, 'An investor')} invested ${parsedAmount.toFixed(2)} SOL in your project "${updatedFundingRequest.title}"!`,
                  itemId: updatedFundingRequest._id, // Link to the funding request document
                  itemType: 'FundingRequest',
                  read: false,
              });
              await farmerNotification.save();
              console.log(`SIMULATING: Notified farmer ${updatedFundingRequest.farmerWallet} about new investment in project ${updatedFundingRequest._id}.`);

             // Optional: Notify the investor about their successful investment (less critical, frontend might handle this)
             // const investorNotification = new Notification({ ... }); await investorNotification.save();

         } catch (notificationError) {
             console.error('Error creating notification after investment:', notificationError);
             // Do not block the response if notifications fail
         }


        // --- 5. Prepare Response ---
        // Populate fields on the saved investment document before sending it back
        await savedInvestment.populate('projectId', 'title status fundedAmount amount updates');
        await savedInvestment.populate('farmerWallet', 'name role displayIdentifier'); // Populate related documents

        // Map to format for frontend (similar to GET /investor/:id)
         const formattedInvestment = {
             _id: savedInvestment._id.toString(),
             investorWallet: savedInvestment.investorWallet.toString(),
             projectId: savedInvestment.projectId ? savedInvestment.projectId._id.toString() : null,
             projectTitle: savedInvestment.projectTitle,
             farmerWallet: savedInvestment.farmerWallet ? savedInvestment.farmerWallet._id.toString() : null,
             farmerName: firstName(savedInvestment.farmerWallet, 'Unknown Farmer'),
             crop: savedInvestment.crop,
             method: savedInvestment.method,
             description: savedInvestment.description,
             acres: savedInvestment.acres,
             timeline: savedInvestment.timeline,
             roi: savedInvestment.roi,
             investorShare: savedInvestment.investorShare,
             cid: savedInvestment.cid,
             videoStorageType: savedInvestment.videoStorageType,
             videoFileHash: savedInvestment.videoFileHash,

             amount: savedInvestment.amount,

             status: savedInvestment.status,
             progress: savedInvestment.progress,

             investmentDate: savedInvestment.investmentDate ? savedInvestment.investmentDate.toISOString() : null,
             txHash: savedInvestment.txHash,

             payoutAmount: savedInvestment.payoutAmount,
             payoutDate: savedInvestment.payoutDate ? savedInvestment.payoutDate.toISOString() : null,
             payoutTxHash: savedInvestment.payoutTxHash,
             payoutNotified: savedInvestment.payoutNotified,

             // Include project updates (from the updatedFundingRequest or re-fetch if needed)
             // For simplicity, we can omit updates in the immediate response or fetch them separately on the frontend
             // updates: updatedFundingRequest.updates.map(...) // Need to fetch updated request again or structure the findByIdAndUpdate response differently
         };


        // Send a success response with the created investment document
        res.status(201).json(formattedInvestment); // 201 Created

    } catch (err) {
        console.error('POST /api/investments error:', err);
         // Handle Mongoose validation errors or other errors
         if (err.name === 'ValidationError') {
              // This might catch errors from the FundingRequest update or Investment creation
              return res.status(400).json({ message: err.message });
         }
        res.status(500).json({ message: 'Server error creating investment.' }); // 500 Internal Server Error
    }
});


// @route PUT /api/investments/:id/progress
// @desc Simulate updating investment progress and triggering payout
// @access Private (Requires authentication. Ideally admin or internal process.)
// Request Body: { progress: number, updateText: string (optional) }
// NOTE: This route is primarily for simulating progress/payout in a demo.
// Actual project progress tracking and payout would likely be more complex
// and potentially managed by backend Cron jobs or external triggers.
router.put('/:id/progress', authenticateToken, async (req, res) => {
     // Authorization is enforced below, once the investment's parent project is loaded:
     // only the farmer who owns the project may report progress on it.

    const investmentId = req.params.id; // Get the investment ID from the URL parameter
    const userId = req.user._id;         // Authenticated user making the request
    const { progress, updateText } = req.body; // Expected new progress value (0-100) and optional update text

    // Validate the ID format
    if (!mongoose.Types.ObjectId.isValid(investmentId)) {
        return res.status(400).json({ message: 'Invalid Investment ID format.' }); // 400 Bad Request
    }

    // Validate progress value
    const parsedProgress = parseInt(progress, 10);
    if (isNaN(parsedProgress) || parsedProgress < 0 || parsedProgress > 100) {
         return res.status(400).json({ message: 'Invalid progress value provided. Must be a number between 0 and 100.' }); // 400 Bad Request
    }

    try {
        // Find the investment document
        const investment = await Investment.findById(investmentId)
                                           // harvestReportedAt is LOAD-BEARING: the guard below refuses to
                                           // settle a profit_share investment until the farmer has reported the
                                           // harvest, and it tests this exact field. It was missing from this
                                           // projection, so the guard read undefined on every request and blocked
                                           // settlement permanently -- even for a project whose harvest had been
                                           // reported correctly. The whole payout path was unreachable, and the
                                           // error message blamed the farmer for not doing what they had done.
                                           .populate('projectId', 'title status fundedAmount amount '
                                               + 'updates farmerWallet settlementMode harvestRevenue '
                                               + 'inputCostBasis investorShare outcome harvestReportedAt')
                                           .populate('investorWallet', 'name role displayIdentifier'); // Populate investor details


        // If investment is not found, return 404
        if (!investment) {
            return res.status(404).json({ message: 'Investment not found.' }); // 404 Not Found
        }

        // --- Authorization: only the farmer who owns the parent project ---
        // This endpoint moves an investment to 'harvested' and writes a payout
        // Transaction, so it must not be reachable by an arbitrary token.
        const ownerId = investment.projectId && investment.projectId.farmerWallet;
        if (!ownerId || ownerId.toString() !== userId.toString()) {
            console.warn(`Forbidden: user ${userId} tried to update progress on investment ${investmentId}`);
            return res.status(403).json({ message: 'Forbidden: only the project owner can update progress.' });
        }

        // --- Apply Progress and Status Updates ---

        // Check if the investment is already harvested/cancelled (terminal states)
         if (investment.status === 'harvested' || investment.status === 'cancelled') {
             // Prevent updating progress if in a terminal state, unless progress is already 100
              if (parsedProgress < 100 || investment.status === 'cancelled') {
                 return res.status(400).json({ message: `Cannot update progress for investment in terminal state "${investment.status}".` });
              }
         }


        // Update the progress value
        investment.progress = parsedProgress;

        let payoutTriggered = false; // Flag to track if payout logic ran

        // Check status transitions based on progress
        if (investment.progress >= 100 && investment.status !== 'harvested') {
            // Investment is now completed/harvested
            investment.status = 'harvested';

            // --- Trigger Payout Logic (Simulated) ---
            // This payout logic runs when an individual investment reaches 100% progress.
            // The project status on FundingRequest is updated when all investments are harvested (handled below).
             if (!investment.payoutNotified) {
                 console.log(`SIMULATING: Triggering payout for investment ${investment._id} (Project: ${investment.projectId})`);

                 // --- Settlement -------------------------------------------
                 // The old line was `investment.amount * (investment.roi / 100)`,
                 // which returns ONLY the yield and never the principal: a 100 at
                 // 15% ROI paid out 15, not 115. It also ignored investorShare
                 // entirely, so the "profit share" the product advertised was
                 // never computed. Both settlement modes are now real, and a
                 // failed harvest is representable instead of implicitly promising
                 // a return.
                 const project = investment.projectId;
                 const mode = investment.settlementMode
                     || (project && project.settlementMode) || 'profit_share';
                 const outcome = (project && project.outcome) || 'harvested';

                 let payout = 0;
                 let basis = '';

                 if (outcome === 'total_loss') {
                     payout = 0;
                     basis = 'Total crop loss reported by the farmer. No payout.';
                 } else if (mode === 'full_repayment') {
                     // Principal back, plus the agreed return on it.
                     const interest = investment.amount * ((investment.roi || 0) / 100);
                     payout = investment.amount + interest;
                     basis = `full_repayment: principal ${investment.amount.toFixed(2)}`
                           + ` + ${investment.roi || 0}% return ${interest.toFixed(2)}`;
                 } else {
                     // Profit share: the investor's pro-rata slice of the pool's
                     // agreed share of ACTUAL realised profit. Needs real numbers
                     // from the farmer, which is why harvestRevenue exists now.
                     const revenue = (project && project.harvestRevenue) || 0;
                     const costs = (project && project.inputCostBasis) || 0;
                     const profit = Math.max(0, revenue - costs);
                     const poolShare = profit * (((project && project.investorShare) || 0) / 100);
                     const raised = (project && project.fundedAmount) || 0;
                     const proRata = raised > 0 ? investment.amount / raised : 0;
                     payout = investment.amount + (poolShare * proRata);
                     basis = `profit_share: principal ${investment.amount.toFixed(2)}`
                           + ` + ${proRata.toFixed(4)} of ${((project && project.investorShare) || 0)}%`
                           + ` of profit ${profit.toFixed(2)} (revenue ${revenue} - costs ${costs})`;
                     if (revenue === 0) {
                         basis += ' -- NOTE: no harvest revenue reported yet, so only principal is returned.';
                     }
                 }

                 // A profit-share settlement with no reported harvest pays out
                 // exactly the principal, silently, and then LATCHES: payoutNotified
                 // and status are set, so the investor can never be paid the
                 // difference once real numbers arrive. Investors are recruited on
                 // "receive N% of the profits", so paying 0% and closing the record
                 // is the worst possible failure. Refuse instead.
                 if (mode === 'profit_share' && outcome !== 'total_loss'
                     && !(project && project.harvestReportedAt)) {
                     return res.status(409).json({
                         message: 'The farmer has not reported the harvest for this '
                                + 'project yet, so the profit share cannot be '
                                + 'calculated. Settlement is blocked until then.',
                         code: 'harvest_not_reported',
                     });
                 }

                 const payoutAmount = Math.round(payout * 100) / 100;
                 const payoutDate = new Date();
                 // NOT a blockchain transaction: an internal reference only. See
                 // generateSimulatedTxHash above -- there is no chain integration.
                 const payoutTxHash = generateSimulatedTxHash('ref_payout');

                 // CLAIM THE SETTLEMENT ATOMICALLY, before any money is written.
                 //
                 // This used to be read-modify-write: the `if (!investment.payoutNotified)`
                 // guard above was evaluated against a document read earlier, so two
                 // concurrent requests both passed it and both wrote a payout
                 // Transaction. Five concurrent calls produced five payouts for one
                 // investment. The conditional update makes exactly one caller win;
                 // everyone else gets null and stops.
                 const claimed = await Investment.findOneAndUpdate(
                     { _id: investment._id, payoutNotified: { $ne: true },
                       status: { $ne: 'harvested' } },
                     { $set: {
                         payoutNotified: true,
                         status: 'harvested',
                         payoutAmount,
                         payoutBasis: basis,
                         settlementMode: mode,
                         payoutDate,
                         payoutTxHash,
                     } },
                     { new: true },
                 );

                 if (!claimed) {
                     // Another request settled it while we were working. Not an
                     // error: report the settled state rather than paying twice.
                     const already = await Investment.findById(investment._id).lean();
                     return res.status(200).json({
                         message: 'This investment was already settled.',
                         alreadySettled: true,
                         payoutAmount: already ? already.payoutAmount : null,
                     });
                 }

                 investment.payoutAmount = payoutAmount;
                 investment.payoutBasis = basis;
                 investment.settlementMode = mode;
                 investment.payoutDate = payoutDate;
                 investment.payoutTxHash = payoutTxHash;
                 investment.payoutNotified = true;
                 investment.status = 'harvested';

                 // Create a payout transaction record for the investor
                  const payoutTransaction = new Transaction({
                      userId: investment.investorWallet, // The investor who receives payout
                      txHash: investment.payoutTxHash,
                      type: 'payout',
                      amount: investment.payoutAmount,
                      projectId: investment.projectId,
                      date: investment.payoutDate,
                  });
                  await payoutTransaction.save();
                  console.log(`SIMULATING: Created payout transaction ${payoutTransaction._id} for investor ${investment.investorWallet}`);

                 // Notify the investor about the payout
                  try {
                       const payoutNotification = new Notification({
                           recipient: investment.investorWallet,
                           type: 'payout', // Custom type
                           message: `Payout received for "${investment.projectTitle}": ${investment.payoutAmount.toFixed(2)} SOL!`,
                           itemId: investment._id, // Link to the investment document
                           itemType: 'Investment',
                           read: false,
                       });
                       await payoutNotification.save();
                       console.log(`SIMULATING: Notified investor ${investment.investorWallet} about payout for ${investment._id}.`);
                  } catch (notifError) {
                       console.error('Error notifying investor about payout:', notifError);
                  }

                 // Notify the farmer that payout was processed for this investment
                  try {
                       const farmerNotification = new Notification({
                           recipient: investment.farmerWallet, // The farmer's ID
                           type: 'payout', // Re-using 'payout' type, or could create 'investment_harvested'
                           message: `Payout processed for an investment in "${investment.projectTitle}". Investor received ${investment.payoutAmount.toFixed(2)} SOL.`,
                           itemId: investment._id,
                           itemType: 'Investment',
                           read: false,
                       });
                       await farmerNotification.save();
                       console.log(`SIMULATING: Notified farmer ${investment.farmerWallet} about payout processed for ${investment._id}.`);
                  } catch (notifError) {
                       console.error('Error notifying farmer about payout:', notifError);
                  }

                 investment.payoutNotified = true; // Mark as notified to prevent duplicate payouts
                 payoutTriggered = true; // Set flag
             }

            // Check if all investments in the project are now harvested
            // This logic should ideally run after the investment document is saved with the new status
             // Let's do this check after saving the investment document below.


        } else if (investment.progress > 0 && investment.status === 'active') {
            // If progress is greater than 0 but not yet 100, and status is still 'active', change to 'growing'.
            investment.status = 'growing';
        } else if (investment.progress === 0 && investment.status !== 'active' && investment.status !== 'harvested' && investment.status !== 'cancelled') {
             // If progress goes back to 0 (unlikely scenario), maybe revert status?
             // Or just keep it as 'growing' if it was already.
             // Let's ensure it's at least 'active' if progress is 0.
            if (investment.status !== 'growing') { // Avoid changing from 'growing' if progress just went to 0
                 investment.status = 'active';
            }
        }


        // --- Add Optional Update Text (if provided in body) ---
         if (updateText && updateText.trim() !== '' && investment.projectId) {
             // We need to add this update to the FundingRequest document, not the Investment document.
             // Find the related FundingRequest document.
             const fundingRequestToUpdate = await FundingRequest.findById(investment.projectId);

             if (fundingRequestToUpdate) {
                 const newUpdate = {
                     _id: new mongoose.Types.ObjectId(), // Use Mongoose ObjectId for embedded document ID
                     date: new Date().toLocaleDateString(), // Keep string date format for demo consistency
                     text: updateText.trim(),
                 };
                  if (!fundingRequestToUpdate.updates) {
                      fundingRequestToUpdate.updates = [];
                  }
                 fundingRequestToUpdate.updates.push(newUpdate);

                 // Save the updated FundingRequest document
                 await fundingRequestToUpdate.save();
                  console.log(`Added new update to Funding Request ${fundingRequestToUpdate._id}`);

                 // Notify ALL investors in this project about the new update
                  const investorsInProject = await Investment.find({ projectId: investment.projectId }).distinct('investorWallet');
                   if (investorsInProject && investorsInProject.length > 0) {
                       const notifications = investorsInProject.map(investorId => ({
                           recipient: investorId,
                           type: 'update', // Custom type
                           message: `Update posted for project "${fundingRequestToUpdate.title}" (${newUpdate.date}): ${newUpdate.text.substring(0, 50)}...`,
                           itemId: fundingRequestToUpdate._id,
                           itemType: 'FundingRequest',
                           read: false,
                       }));
                       await Notification.insertMany(notifications);
                       console.log(`SIMULATING: Notified ${investorsInProject.length} investors about update for request ${fundingRequestToUpdate._id}`);
                   }

             } else {
                 console.warn(`Update text provided, but associated Funding Request ${investment.projectId} not found.`);
                 // Decide how to handle - maybe create a standalone update log?
             }
         }


        // --- Save the Updated Investment Document ---
        const updatedInvestment = await investment.save();

        // --- Post-Save Check for Project Status ---
         // After saving the individual investment, check if THIS project is now fully harvested.
         if (payoutTriggered && updatedInvestment.status === 'harvested' && updatedInvestment.projectId) {
             const totalInvestmentsInProject = await Investment.countDocuments({ projectId: updatedInvestment.projectId });
             const harvestedInvestmentsInProject = await Investment.countDocuments({ projectId: updatedInvestment.projectId, status: 'harvested' });

             // If all investments are harvested, update the project status to 'completed'
             if (totalInvestmentsInProject > 0 && totalInvestmentsInProject === harvestedInvestmentsInProject) {
                 const fundingRequest = await FundingRequest.findById(updatedInvestment.projectId);
                 if (fundingRequest && fundingRequest.status !== 'completed') {
                      fundingRequest.status = 'completed';
                      await fundingRequest.save();
                      console.log(`SIMULATING: Project ${fundingRequest._id} marked as completed because all investments are harvested.`);

                      // Optional: Notify farmer/investors that the entire project is completed
                       try {
                           const completionNotification = new Notification({
                               recipient: fundingRequest.farmerWallet, // Notify the farmer
                               type: 'update', // Or a 'completion' type
                               message: `Your project "${fundingRequest.title}" is now completed!`,
                               itemId: fundingRequest._id,
                               itemType: 'FundingRequest',
                               read: false,
                           });
                            // Also notify all investors that the project is completed (they already got payout notifs)
                            // const investorsInProject = await Investment.find({ projectId: fundingRequest._id }).distinct('investorWallet');
                            // ... create notifications for investors ...

                           await completionNotification.save();
                           console.log(`SIMULATING: Notified farmer ${fundingRequest.farmerWallet} that project ${fundingRequest._id} is completed.`);
                       } catch (notifError) {
                            console.error('Error notifying farmer about project completion:', notifError);
                       }
                 }
             }
         }


        // Populate fields for the response
        // Mongoose 8: Document.populate() returns a Promise, so these cannot be chained.
        await updatedInvestment.populate('projectId', 'title status fundedAmount amount updates farmerWallet');
        await updatedInvestment.populate('investorWallet', 'name role displayIdentifier');
        // farmerWallet MUST be populated here. Without it the formatter below
        // reads .name off a bare ObjectId, .split throws, and this endpoint
        // returned HTTP 500 on every single call -- AFTER the payout Transaction
        // was already written. The caller could not tell whether the money had
        // moved, which is exactly what drove operators to retry and trigger the
        // duplicate-payout race.
        await updatedInvestment.populate('farmerWallet', 'name role displayIdentifier');

         // Map to format for frontend (similar structure to the GET response)
         const formattedUpdatedInvestment = {
             _id: updatedInvestment._id.toString(),
             investorWallet: updatedInvestment.investorWallet
                 ? (updatedInvestment.investorWallet._id || updatedInvestment.investorWallet).toString()
                 : null,
             projectId: updatedInvestment.projectId ? updatedInvestment.projectId._id.toString() : null,
             projectTitle: updatedInvestment.projectTitle,
             farmerWallet: updatedInvestment.farmerWallet ? updatedInvestment.farmerWallet._id.toString() : null,
             // Defensive: a farmer record could be missing a name. Formatting a
             // response must never be able to fail after money has moved.
             farmerName: firstName(updatedInvestment.farmerWallet, 'Unknown Farmer'),
             crop: updatedInvestment.crop,
             method: updatedInvestment.method,
             description: updatedInvestment.description,
             acres: updatedInvestment.acres,
             timeline: updatedInvestment.timeline,
             roi: updatedInvestment.roi,
             investorShare: updatedInvestment.investorShare,
             cid: updatedInvestment.cid,
             videoStorageType: updatedInvestment.videoStorageType,
             videoFileHash: updatedInvestment.videoFileHash,

             amount: updatedInvestment.amount,

             status: updatedInvestment.status,
             progress: updatedInvestment.progress,

             investmentDate: updatedInvestment.investmentDate ? updatedInvestment.investmentDate.toISOString() : null,
             txHash: updatedInvestment.txHash,

             payoutAmount: updatedInvestment.payoutAmount,
             payoutDate: updatedInvestment.payoutDate ? updatedInvestment.payoutDate.toISOString() : null,
             payoutTxHash: updatedInvestment.payoutTxHash,
             payoutNotified: updatedInvestment.payoutNotified,

             // Include project updates if they were added/updated (fetch from the updated FundingRequest document if necessary)
             // Since we populated projectId with updates, we can access them here
             updates: updatedInvestment.projectId && updatedInvestment.projectId.updates
                 ? updatedInvestment.projectId.updates.map(update => ({
                     _id: update._id ? update._id.toString() : null, date: update.date, text: update.text,
                 }))
                 : [], // Empty array if no project or no updates

         };


        // Send a success response with the updated investment document
        res.json({ message: 'Investment progress updated successfully.', investment: formattedUpdatedInvestment }); // Default status is 200 OK

    } catch (err) {
        console.error(`PUT /api/investments/${investmentId}/progress error:`, err);
        // Handle potential Mongoose validation errors
         if (err.name === 'ValidationError') {
              return res.status(400).json({ message: err.message });
         }
        res.status(500).json({ message: 'Server error updating investment progress.' }); // 500 Internal Server Error
    }
});


// --- Export the router ---

// Export the configured router so it can be used by server.js
module.exports = router;