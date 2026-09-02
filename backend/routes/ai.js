// --- Backend Routes: ai.js ---

const express = require('express');
const router = express.Router();
const dotenv = require('dotenv');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { authenticateToken } = require('./auth');
const multer = require('multer');
const FormData = require('form-data');

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GROQ_API_KEY   = process.env.GROQ_API_KEY;

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent';
const GROQ_API_URL   = 'https://api.groq.com/openai/v1/chat/completions';
// Groq retires model ids roughly yearly: llama3-70b-8192 was decommissioned
// 2025-08-30 and llama-3.3-70b-versatile deprecated 2026-06-17. Never inline one.
const GROQ_MODEL       = process.env.GROQ_MODEL || 'openai/gpt-oss-120b';
const GROQ_MODEL_SMALL = process.env.GROQ_MODEL_SMALL || 'openai/gpt-oss-20b';
const FLASK_URL      = process.env.FLASK_URL || 'http://127.0.0.1:5000';

// Server-owned persona. Never accept this from the client.
const GROQ_TIMEOUT_MS = Number(process.env.GROQ_TIMEOUT_MS || 30000);
// The RAG probe must fail FAST. It used GROQ_TIMEOUT_MS (30s), so pointing
// FLASK_URL at a sleeping free-tier host made every chat message hang ~33s
// before the Groq fallback answered -- indistinguishable from the app being
// broken. A local Flask replies in well under 3s; anything slower is not there.
const RAG_PROBE_TIMEOUT_MS = Number(process.env.RAG_PROBE_TIMEOUT_MS || 3000);
const MAX_HISTORY_TURNS = Number(process.env.MAX_HISTORY_TURNS || 8);
const MAX_QUESTION_CHARS = Number(process.env.MAX_QUESTION_CHARS || 4000);

// Kept in sync with GROQ_UNGROUNDED_PROMPT in flask_server.py. A live test
// exposed why that matters: without the dose rule below, this endpoint happily
// returned a table of "typical field rates" for late-blight fungicides, while
// the Flask path correctly refused the same question. This is the endpoint the
// UI actually calls, so it needs the stricter prompt, not the looser one.
const AGRIBOT_SYSTEM_PROMPT = `You are AgriBot, an agricultural assistant for potato farmers on the PestiVid platform.

You have NO document retrieval for this answer, so you are working from general
knowledge. Behave accordingly:

- NEVER state a pesticide dose, concentration, application rate, spray interval or
  pre-harvest interval. Not even a typical or approximate one, and not in a table.
  Say instead that the rate must be read off the product label and confirmed with a
  licensed agronomist or the local agricultural extension officer.
- Do not name a product as registered or approved for a crop. Registration is
  jurisdiction-specific and you cannot verify it.
- Prefer cultural and preventive measures (rotation, drainage, spacing, certified
  seed, sanitation, scouting). Those are safe to recommend in detail.
- You may name active ingredients as options, but always without a rate.
- Be explicit when you are uncertain.
- If a question is not about agriculture, say so and do not answer it.
- Advise contacting an extension officer for anything urgent or field-wide.

Provide practical, concise advice on crop diseases and pest management, planting and
harvesting schedules, soil health, irrigation, and sustainable practices. Always
prioritise farmer safety and environmental sustainability.

Write PLAIN TEXT. No Markdown of any kind: no asterisks for emphasis, no hyphens or
digits starting a list, no headings, no tables, no code fences. The answer is shown
in a chat bubble that renders text exactly as you write it, so a farmer reading it
on a phone sees every asterisk you type. Short sentences and short paragraphs, one
idea each.`;

// Simple file upload handling using multer
const UPLOAD_DIR = path.join(__dirname, '..', 'temp_uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

// Configure multer for file uploads
const upload = multer({
    dest: UPLOAD_DIR,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});


// ─────────────────────────────────────────────────────────────────────────────
// @route  POST /api/ai/agribot
// @desc   AgriBot chat — calls Groq directly (no Flask needed).
//         Works for any authenticated farmer.
// @access Private / farmer
// ─────────────────────────────────────────────────────────────────────────────
router.post('/agribot', authenticateToken, async (req, res) => {
    // Anyone signed in may ask. A buyer checking a lot's leaf and an investor
    // asking about blight have the same question a farmer has.
    // Validate the request BEFORE checking server configuration. With the
    // config check first, an oversized or malformed question reported a 500
    // "not configured", which blames the operator for a client error and hides
    // the real problem from the caller.
    const { question } = req.body;
    if (!question || typeof question !== 'string' || !question.trim()) {
        return res.status(400).json({ message: 'question is required.' });
    }
    if (question.length > MAX_QUESTION_CHARS) {
        return res.status(413).json({
            message: `Question too long (max ${MAX_QUESTION_CHARS} characters).` });
    }

    if (!GROQ_API_KEY) {
        return res.status(500).json({ message: 'AgriBot service is not configured (missing GROQ_API_KEY).' });
    }

    // Conversation memory. Every message used to be stateless: the frontend kept
    // history in localStorage and never sent it, so "my potatoes have late
    // blight" followed by "what dose?" produced an answer with no knowledge of
    // the first message. Bounded so a caller cannot inflate the context window.
    const history = Array.isArray(req.body.history)
        ? req.body.history
            .filter(m => m && ['user', 'assistant'].includes(m.role)
                          && typeof m.content === 'string' && m.content.trim())
            .slice(-MAX_HISTORY_TURNS)
            .map(m => ({ role: m.role, content: m.content.slice(0, MAX_QUESTION_CHARS) }))
        : [];

    // --- Prefer the retrieval pipeline when the Flask AI server is up. ---
    // The README advertised "a localized Pinecone RAG knowledge base", but this
    // endpoint only ever called Groq directly -- the RAG existed in the
    // notebooks and flask_server.py and reached no user. Try it first, and fall
    // back to a direct Groq call rather than failing.
    try {
        const ragRes = await axios.post(`${FLASK_URL}/chat`,
            { question, history },
            { timeout: RAG_PROBE_TIMEOUT_MS });
        const answer = ragRes.data && (ragRes.data.answer || ragRes.data.text);
        if (answer) {
            return res.json({
                question,
                answer,            // same field name as the Groq path below
                source: 'rag',
                retrieved: ragRes.data.retrieved || null,
                grounded: Boolean(ragRes.data.retrieved && ragRes.data.retrieved.length)
            });
        }
    } catch (ragErr) {
        console.log('agribot: RAG unavailable (%s), using direct Groq',
                    ragErr.code || ragErr.message);
    }

    try {
        const groqRes = await axios.post(
            GROQ_API_URL,
            {
                model: GROQ_MODEL,
                messages: [
                    {
                        role: 'system',
                        content: AGRIBOT_SYSTEM_PROMPT
                    },
                    ...history,
                    { role: 'user', content: question }
                ],
                max_tokens: 1200,   // headroom: reasoning models spend part of this on reasoning
                temperature: 0.7
            },
            {
                headers: {
                    Authorization: `Bearer ${GROQ_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: GROQ_TIMEOUT_MS
            }
        );

        // gpt-oss and other reasoning models spend part of the completion budget
        // on internal reasoning, so `content` can come back EMPTY while the call
        // itself succeeds (verified live: 53 of 83 completion tokens were
        // reasoning tokens). Returning that empty string would show the farmer a
        // blank reply and look like a silent failure.
        const msg = groqRes.data?.choices?.[0]?.message || {};
        const answer = (msg.content || '').trim();
        if (!answer) {
            console.warn('Groq returned empty content (reasoning_tokens=%s). Raise max_tokens.',
                groqRes.data?.usage?.completion_tokens_details?.reasoning_tokens);
            return res.status(502).json({
                message: 'The assistant returned an empty answer. Please try again.',
                reason: 'empty_completion'
            });
        }
        return res.json({ question, answer });

    } catch (error) {
        console.error('AgriBot /agribot Groq error:', error.response?.data || error.message);
        if (error.response) {
            return res.status(error.response.status).json({
                message: error.response.data?.error?.message || 'Error from Groq.',
            });
        }
        return res.status(500).json({ message: 'Error communicating with AgriBot service.' });
    }
});


// ─────────────────────────────────────────────────────────────────────────────
// @route  POST /api/ai/predict-proxy
// ─────────────────────────────────────────────────────────────────────────────
// @route  POST /api/ai/predict-proxy
// @desc   Plant disease analysis — tries Flask ML server first, falls back to Groq LLM.
// @access Private / farmer
// ─────────────────────────────────────────────────────────────────────────────
// Reject non-farmers BEFORE multer writes the body to disk. Previously the
// 10MB upload was persisted and then abandoned on the 403 path, so any
// authenticated non-farmer could fill the disk with orphaned temp files.
const farmersOnly = (req, res, next) => {
    // Anyone signed in may ask. A buyer checking a lot's leaf and an investor
    // asking about blight have the same question a farmer has.
    next();
};

router.post('/predict-proxy', authenticateToken, farmersOnly, upload.single('file'), async (req, res) => {

    // Check if file was uploaded
    if (!req.file) {
        return res.status(400).json({ message: 'No image file uploaded.' });
    }

    // --- Strategy 1: Forward to Flask ML server (real model inference) ---
    try {
        console.log('predict-proxy: Trying Flask server at', FLASK_URL + '/predict');
        console.log('predict-proxy: File uploaded:', req.file.originalname, 'Size:', req.file.size);

        // Create form data to send to Flask
        const formData = new FormData();
        formData.append('file', fs.createReadStream(req.file.path), {
            filename: req.file.originalname || 'image.jpg',
            contentType: req.file.mimetype || 'image/jpeg'
        });

        // Forward to Flask server
        const flaskRes = await axios.post(`${FLASK_URL}/predict`, formData, {
            headers: {
                ...formData.getHeaders(),
            },
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            timeout: 60000, // 60s timeout for model inference
        });

        console.log('predict-proxy: Flask responded successfully');
        const result = flaskRes.data;
        result._source = 'flask-ml';
        result._note   = 'Classification from the Flask model server. Treatment text may come from a curated table rather than a model.';
        
        // Clean up uploaded file
        fs.unlinkSync(req.file.path);
        
        return res.json(result);

    } catch (flaskError) {
        console.log('predict-proxy: Flask unavailable or error:', flaskError.message);
        console.error('Flask error details:', flaskError.response?.data || flaskError.message);
        
        // Clean up uploaded file
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        // Fall through to Groq fallback
    }

    // --- No fabrication fallback. If the model is unavailable, say so. ---
    // Previously this asked an LLM to "make a realistic, randomised disease
    // prediction" with no image attached, and returned it to the farmer as a
    // real diagnosis with a fabricated confidence and probability vector.
    return res.status(503).json({
        error: 'model_unavailable',
        message: 'Disease detection is offline. No diagnosis was produced.'
    });
});


// ─────────────────────────────────────────────────────────────────────────────
// @route  POST /api/ai/analyze-plant
// @desc   Gemini Vision proxy (original — kept for backward compat).
// @access Private / farmer
// ─────────────────────────────────────────────────────────────────────────────
router.post('/analyze-plant', authenticateToken, async (req, res) => {
    // Anyone signed in may ask. A buyer checking a lot's leaf and an investor
    // asking about blight have the same question a farmer has.
    if (!GEMINI_API_KEY || GEMINI_API_KEY.startsWith('YOUR_GEMINI')) {
        return res.status(500).json({ message: 'Plant analysis service is not configured on the server.' });
    }

    const { mimeType, base64Data } = req.body;
    if (!mimeType || !base64Data) {
        return res.status(400).json({ message: 'Missing image data (mimeType or base64Data).' });
    }

    try {
        const prompt = 'Analyze this plant image. Identify: PLANT: [Name]; DISEASE: [Name/Healthy/Not Apparent/Unknown]; TREATMENT: [Suggestions]. Respond strictly in this format.';
        const requestBody = {
            contents: [{ parts: [{ text: prompt }, { inline_data: { mime_type: mimeType, data: base64Data } }] }],
            generationConfig: { temperature: 0.4, topK: 32, topP: 1, maxOutputTokens: 1024 }
        };
        const geminiResponse = await axios.post(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, requestBody, {
            headers: { 'Content-Type': 'application/json' }
        });
        const text = geminiResponse.data?.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) return res.json({ rawText: text });
        return res.status(500).json({ message: 'AI service returned unexpected format.' });
    } catch (error) {
        console.error('Gemini API error:', error.response?.data || error.message);
        if (error.response) {
            return res.status(error.response.status).json({ message: error.response.data?.error?.message || 'Error from AI service.' });
        }
        return res.status(500).json({ message: 'Error communicating with plant analysis service.' });
    }
});


// ─────────────────────────────────────────────────────────────────────────────
// @route  POST /api/ai/chatbot
// @desc   Groq chatbot proxy (original — kept for backward compat).
// @access Private / farmer
// ─────────────────────────────────────────────────────────────────────────────
router.post('/chatbot', authenticateToken, async (req, res) => {
    // Anyone signed in may ask. A buyer checking a lot's leaf and an investor
    // asking about blight have the same question a farmer has.
    // The caller used to supply `systemPrompt` AND the whole message array,
    // which made this an open LLM proxy billed to the operator's Groq key: any
    // registered user could set an arbitrary persona and use it for anything.
    // The system prompt is now server-owned and the history is bounded.
    const { messages } = req.body;
    if (!Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ message: 'Invalid chat data format.' });
    }
    const ALLOWED_ROLES = ['user', 'assistant'];
    const history = messages
        .filter(m => m && ALLOWED_ROLES.includes(m.role) && typeof m.content === 'string')
        .slice(-8)
        .map(m => ({ role: m.role, content: m.content.slice(0, 4000) }));
    if (history.length === 0) {
        return res.status(400).json({ message: 'No usable messages supplied.' });
    }
    const systemPrompt = AGRIBOT_SYSTEM_PROMPT;

    if (!GROQ_API_KEY) {
        return res.status(500).json({ message: 'AgriBot service is not configured on the server.' });
    }
    try {
        const groqResponse = await axios.post(
            GROQ_API_URL,
            { messages: [{ role: 'system', content: systemPrompt }, ...history], model: GROQ_MODEL_SMALL, temperature: 0.7, max_tokens: 1024, top_p: 1, stream: false },
            { headers: { Authorization: `Bearer ${GROQ_API_KEY}`, 'Content-Type': 'application/json' }, timeout: GROQ_TIMEOUT_MS }
        );
        const content = groqResponse.data?.choices?.[0]?.message?.content;
        if (content) return res.json({ text: content });
        return res.status(500).json({ message: 'AgriBot returned unexpected format.' });
    } catch (error) {
        console.error('Groq API error:', error.response?.data || error.message);
        if (error.response) return res.status(error.response.status).json({ message: error.response.data?.error?.message || 'Error from AI service.' });
        return res.status(500).json({ message: 'Error communicating with AgriBot service.' });
    }
});


module.exports = router;