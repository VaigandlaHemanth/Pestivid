# Fix Log

Remediation of the findings from the 2026-08-21 static audit.
One entry per task. `Phase` numbers refer to the repair plan.

---

## Phase 0 — Containment (complete, local only)

### 0.3 Werkzeug debugger disabled — was remote code execution
All four Flask servers ran `app.run(host='0.0.0.0', debug=True)`. Werkzeug's
`/console` is a Python REPL running as the server user, and the debugger PIN
plus the developer's LAN IP were published in `RUNNING_STATUS.md`.

- `flask_server.py:529`, `flask_server_simple.py:192`,
  `flask_server_with_models.py:338`, `flask_server_with_frontend.py:200`
- Now: `host` / `port` / `debug` all read from env, defaulting to
  `127.0.0.1:5000` with debug **off**.
- `import os` added to `flask_server_with_models.py` (it had none).
- PIN and LAN IP redacted from `RUNNING_STATUS.md`.

Verify: `curl http://127.0.0.1:5000/console` -> 404.

### 0.4 Three fabricated prediction paths removed
None of these ever opened the uploaded image.

| File | Was | Now |
|---|---|---|
| `flask_server_simple.py` `/predict` | hardcoded `'Potential Fungal or Pest Issue'` / `'75%'` | `503 model_unavailable` |
| `flask_server_with_frontend.py` `/api/predict` | same hardcoded pair | `503 model_unavailable` |
| `backend/routes/ai.js` `/predict-proxy` fallback | asked an LLM to *"make a realistic, randomised disease prediction"* with no image attached, returning a fabricated confidence and probability vector | `503 model_unavailable` |

Also `ai.js:139` no longer claims canned output came from `"CLIP + T5 ML models"`.

Verify: upload any image with Flask down -> `503`, no disease name, no percentage.

### 0.5 Unauthenticated PII harvest closed
`GET /api/users/public` (every user id, unpaginated) chained with
`GET /api/users/:id/public` (that user's email + phone) dumped the whole user
base with no token. Both routes were `@access Public`.

- `routes/users.js:15,53` — `authenticateToken` added to both.
- List route now bounded: `?limit` capped at 100, `?offset` supported.
- `email` and `phone` removed from both the `.select()` and the response body.

Verify: `curl /api/users/public` with no token -> 401; with a token -> no
`email` or `phone` in any row.

### 0.6 Authorization restored on the payout endpoint
`PUT /api/investments/:id/progress` moves an investment to `harvested` and
writes a payout `Transaction`. Its only guard was a `console.warn` with the
`403` **commented out**, and the `role !== 'admin'` test could never be true
because `admin` is not in the `User` role enum.

- `routes/investments.js` — dead admin block removed; replaced with a real
  check that `investment.projectId.farmerWallet` equals `req.user._id`.

Verify: as an unrelated user, `PUT` someone else's investment -> 403.

### 0.7 Server-side price validation
`grep -c 'minPrice\|maxPrice' routes/purchases.js` returned **0** — the price
band existed only in the browser, so a crafted request bought any listing for
any amount above zero.

- `routes/purchases.js` — offer now validated against the listing's own
  `minPrice`/`maxPrice` after the self-purchase check.

Verify: `curl -X POST /api/purchases -d '{"offerPrice":0.01,...}'` -> 400.

### 0.8 .gitignore gaps closed
- Added: `backend/.local-mongo-data/` (a live WiredTiger database),
  `backend/temp_uploads/`, `temp_image.jpg`, `*.bak`, `backend/db/*.json`.
- Untracked (files kept on disk): `backend/db/{users,listings,videos}.json`
  (real bcrypt hashes), `public/index_new.html`,
  `public/index_original.html.bak` (both held live Pinata JWTs).
- `Pestivid/.gitignore` — removed the bare `*.json`, which would have silently
  ignored `package.json` and any config file added later.

Nothing depends on the JSON store: no module imports `simple-db`, and
`simple-db.js` recreates the directory and files on demand.

### 0.2 (local half) Hardcoded secrets removed from source
Rotation itself needs dashboard access and is **still outstanding** — see below.

| Location | Was | Now |
|---|---|---|
| `flask_server.py:149` | `os.getenv(...) or "pcsk_..."` | raises `RuntimeError` if unset — missing config must not fall back to a shared credential |
| `backend/dev-server.js:30` | committed JWT secret literal | `process.env.JWT_SECRET` or a random 48-byte value per restart |
| `backend/test-supabase.js:16-17` | real project ref + publishable key | `<your-project-ref>` / `<your-publishable-key>` |
| `public/index.html:1401` (+ the two duplicate copies) | live Pinata JWT with `scopedKeySecret` | `YOUR_PINATA_JWT_MUST_BE_REPLACED_HERE`, which triggers the config-warning banner the page already had at line 411 |
| 20+ docs and `.bat` files | Supabase key, project ref, Werkzeug PIN, LAN IP | redacted |

Added `.env.example` and `backend/.env.example` covering every value the code
now requires, including `GROQ_MODEL` (kept in config because Groq retires model
ids about yearly).

Verified: a repo-wide sweep for `pcsk_`, `sb_publishable_`, `gsk_`, the Pinata
JWT prefix, the debugger PIN and the LAN IP returns **nothing**.

---

## Still outstanding — needs your account access

1. **Rotate the exposed credentials.** They are in git history on both repos, so
   editing the files does not retire them. Pinecone (account-wide: can list,
   read and delete every index), the three Pinata scoped keys, the Supabase
   publishable key, and Groq/Cohere as a precaution.
2. **Repo visibility.** Check whether the repository should be public at all
   while the exposed credentials are still live.
3. **Force-reset** the accounts whose bcrypt hashes were committed in
   `backend/db/users.json`.
4. **Optional hardening:** purge the secrets from git history with
   `git-filter-repo` or BFG. This rewrites history and needs a force-push, so it
   is deliberately not done here.

---

## Phase 1 — Make it run (complete)

All changes verified against a running server on in-memory MongoDB
(`npm run dev:mem`). Live results are quoted per task.

### 1.1 Groq model id moved to configuration
`llama3-70b-8192` was decommissioned 2025-08-30, so every AI call had been
returning `400 model_decommissioned` for ~12 months. Its documented replacement
`llama-3.3-70b-versatile` was itself deprecated 2026-06-17 — so the fix is not a
new literal, it is *removing the literal*.

- 5 code sites now read config: `ai.js` (`GROQ_MODEL`, `GROQ_MODEL_SMALL`),
  `flask_server.py` x2, `simple_ai_agent.py` (new `self.groq_model`).
- 3 notebook sites (`nowwor.ipynb`, `tes.ipynb`) now use
  `os.getenv("GROQ_MODEL", ...)`.
- Default: `openai/gpt-oss-120b` (131,072-token context, up from 8,192).

Verified: `grep -rn "llama3-70b-8192" --include=*.js --include=*.py` -> only a
comment. `POST /api/ai/agribot` now returns
`"AgriBot service is not configured (missing GROQ_API_KEY)"` — a config error,
not a dead-model error.

### 1.2 Silent fallback made observable
`get_crop_advice` swallowed every failure, served a canned keyword dictionary,
and still reported `status: "success"`. That is why the outage went unnoticed.

- The fallback now returns `status: "degraded"` with a `reason`, and logs the
  Groq HTTP status and response body (Groq puts a machine-readable
  `error.code` such as `model_decommissioned` there).
- `_get_local_advice(query, reason=...)` — all three call sites pass a reason.
- `analyze_crop_image_description` and `get_seasonal_advice` never call an LLM
  at all; both now declare `source: "Rule-based ... (no LLM)"` so a caller
  cannot mistake them for model output.

### 1.3 Dependency floors corrected (latent, not yet firing)
Worth being precise: the audit described what a **fresh** `pip install` resolves
to today, not this machine. Locally it is Python 3.11.7 / transformers 4.52.4,
where `T5TokenizerFast` still exists. The breakage is latent, and the pins stop
it landing later.

- `transformers>=4.38` -> `>=4.40,<5` (5.x removed `T5TokenizerFast`).
- `langchain>=0.2` -> `>=0.3,<0.4` (`>=0.2` spanned two majors of API drift).
- `pinecone-client` **removed** — renamed to `pinecone` at v5.1.0; the old name
  is now a tombstone that raises a bare `Exception`, which bypasses the
  `ImportError` guard in `flask_server.py` and kills the process at startup.
  `langchain-pinecone` already depends on `pinecone`.
- `langchain-pinecone` annotated: requires python <3.14.
- `T5TokenizerFast` -> `AutoTokenizer` in 5 files (works on 4.x and 5.x).
- Notebooks: `langchain.text_splitter` -> `langchain_text_splitters`,
  `langchain.schema` -> `langchain_core.documents`.

### 1.4 One backend entry point
- `server.js` now fails fast with a readable message when `MONGODB_URI` or
  `JWT_SECRET` is missing. Previously a missing URI threw an unhandled
  `TypeError` inside `app.listen` — *after* printing "Server running on port
  3001", so it looked like a successful start. The URI-masking call is also
  defensive now.
- Deleted `server-simple.js` and `server-supabase.js`. Both `require()`
  directories (`routes-simple/`, `routes-supabase/`) that
  `git log --all` confirms **never existed in any commit**.
- `dev-server.js` is now tracked, `mongodb-memory-server@^11.2.0` declared in
  `devDependencies`, and `npm run dev:mem` added.

Verified: with no env, `node server.js` prints
`FATAL: MONGODB_URI is not set...` and exits 1.

### 1.5 `mounted()` promoted to a real lifecycle hook
It was defined at 4-space indent *inside* `methods: {`, i.e. a method named
"mounted" that Vue 2 never calls. Session restore, localStorage hydration and
the `beforeunload` persistence handler were all dead code.

- `methods` is now closed after the last real method; `mounted` sits at column 0
  alongside `el`, `data`, `computed`, `watch`, `methods`.

Verified: the `<script>` block was extracted and passed `node --check`;
`grep -nE "^(data|computed|watch|methods):|^async mounted"` shows `mounted` at
top level.

### 1.6 Chained `Document.populate()` — 3 sites, not 1
In Mongoose 8 `Document.populate()` returns a **Promise**, so `.populate()`
chained onto it is `undefined` and throws `TypeError`. Confirmed by execution:
`typeof doc.populate('x').populate === 'undefined'`.

It threw *after* the investment was saved and the funding total incremented, so
the money committed and the caller got a 500 — then retried and double-invested.

- Split into sequential awaits at `investments.js:250`, `investments.js:538`,
  `fundingRequests.js:507`.

Verified: `POST /api/investments` -> **201** with a fully populated body
(was 500).

### 1.7 Dependencies upgraded — 15 vulnerabilities to 0
| Package | Was | Now |
|---|---|---|
| multer | 1.4.5-lts.2 | **2.2.0** |
| axios | 1.9.0 | **1.19.0** |
| mongoose | 8.15.1 | **8.24.3** |
| form-data | 4.0.3 | **4.0.6** |
| bcrypt | 5.1.1 | **6.0.0** |
| express | 4.21.2 | **4.22.2** |
| jws (transitive) | <3.2.3 | **3.2.3** |

`npm audit`: 2 critical + 10 high + 3 moderate -> **0**.

Note `jws <3.2.3` is *Improperly Verifies HMAC Signature* — a JWT signature
bypass reached transitively through `jsonwebtoken`, and invisible in the
top-level dependency list.

Both major bumps were compatibility-tested rather than assumed:
- bcrypt 6: hash/compare round-trip works, output is still 60-char `$2b$`, and
  it still verifies a hash generated by 5.x.
- multer 2: the existing `{ dest, limits }` config constructs fine and yields a
  `DiskStorage`.

### Live end-to-end verification

| Check | Result |
|---|---|
| `GET /api/users/public` no token | **401** |
| `GET /api/users/:id/public` no token | **401** |
| fields returned with a token | `_id,name,role,memberSince,createdAt,displayIdentifier` — no email, no phone |
| `?limit=2` | 2 rows |
| purchase at 0.01 on a 0.5–0.9 listing | **400** `Offer must be between 0.5 and 0.9.` |
| buyer PUTs an investor's investment progress | **403** `only the project owner can update progress` |
| `POST /api/investments` | **201**, populated |
| `POST /api/ai/predict-proxy` with Flask down | **503** `model_unavailable` — no fabricated diagnosis |
| backend boot | connects, seeds, serves on 3001 |

---

## Phase 2 — Close the remaining holes (complete)

### 2.1 Twelve unreachable admin branches removed

Every one read `X && req.user.role !== 'admin'`. Because `admin` is not in the
`User` role enum that clause is always true, so the branches were dead code that
*looked* like authorization. Removing the clause is behaviourally a no-op and
makes each check say what it actually does.

Sites: fundingRequests 1, investments 1, listings 1, messaging 2,
notifications 3, purchases 2, transactions 1, videos 1.

`models/User.js` now documents why there is no admin role, so the pattern does
not come back.

### 2.2 Money paths made atomic

**purchases.js** — the `status !== 'active'` guard was a separate round-trip from
the mutation, so two concurrent buyers could both pass it. `findByIdAndUpdate`
became `findOneAndUpdate({ _id, status: 'active' }, ...)`, which makes MongoDB
the arbiter, plus a 409 for the loser. An unreachable duplicate
`if (!updatedListing)` block that I introduced while editing was removed.

**investments.js** — rewritten as an aggregation-pipeline update:

- filter: status in {pending, partially_funded} **and**
  `$expr: fundedAmount + amount <= amount`. Plain `$inc` was atomic but
  *unconditional*, so nothing prevented over-funding.
- stage 1 increments `fundedAmount` and appends the investor entry.
- stage 2 derives `status` from the new total. This is required because
  `findOneAndUpdate` does **not** run the `pre('save')` hook, so without it a
  fully funded project stayed `partially_funded`. I hit exactly this during
  testing and fixed it.
- 409 responses now distinguish not-found / not-accepting / exceeds-remaining.
- `txHash` and `investmentDate` are generated **before** the write and used as
  the join key. The old code searched the embedded array afterwards by
  (investor, exact float amount, 5-second window) and returned the wrong entry
  for two same-amount investments inside that window.

### 2.3 Two model hooks that made features impossible

- `FundingRequest` derived `status` on **every** save, outside the
  `if (this.isNew)` guard, so `completed` and `cancelled` were silently rewritten
  back. Terminal states are now excluded from derivation.
- The `updates[]` subdocument required a String `id` that **neither** push site
  ever set, so every project update threw
  `ValidationError: updates.0.id: Path id is required`. Both sites set `_id`, and
  the frontend keys off `update._id` (index.html:692, 1140), so the phantom field
  was removed and all four serialisers now emit `_id`.

### 2.4 Stopped reporting failure as success

- `sendMessage` fell back to `sendMessageLocal()` and told the user
  *Message sent (stored locally)*. Nothing was sent and the recipient never
  received it. It now restores the typed text and reports the failure.
- `startOrGoToConversation` created an `isLocal: true` conversation on error and
  announced *ready (local mode)* — a chat that could never deliver anything.
  Removed.
- `sendMessageLocal` had zero callers afterwards; the 35-line function was
  deleted.
- `showFarmerProfile` called `GET /api/users/:id`, **which does not exist** (the
  real route is `/users/:id/public`), so it *always* fell through to fabricating
  a profile including a fake email `<id>@pestivid.demo`. Endpoint corrected; the
  fabrication replaced with a visible error.

Checked and found already correct: `createListing` and `sendChatMessage` both
surface real errors. My initial grep heuristic flagged them as false positives.

### 2.5 Remaining access-control leaks

- **The paywall bypass.** `GET /api/listings` was unauthenticated and returned
  `cid` — the IPFS content id of the very video buyers pay for. Anyone could
  fetch it for free. `cid`, `storageType` and `videoFileHash` are now stripped
  from the browse response, and the route requires a token. Same treatment for
  `GET /api/videos`. Buyers still receive `cid` on their `Purchase` record.
- **The `?farmerId=` bypass was real.** `GET /api/videos/farmer/:farmerId`
  correctly returns 403 for other users, but the unauthenticated
  `GET /api/videos` accepted `?farmerId=` and served the same data, making that
  403 meaningless.
- **ReDoS.** `filter.crop = { $regex: req.query.crop }` took raw user input.
  Replaced with `safeCrop()`, which caps length at 60 and drops everything that
  is not a word character, space or hyphen — a whitelist rather than an escape,
  so there is no metacharacter left to get wrong.
- **Global notification deletion.** Any user could delete a `global`
  notification for *every* user. Global notifications now record a per-user
  dismissal (`dismissedBy` on the model); only the actual recipient can delete.
- **Open LLM proxy.** `POST /api/ai/chatbot` let the caller supply the entire
  system prompt and message array, billed to the operator's Groq key. The prompt
  is now a server-side constant (`AGRIBOT_SYSTEM_PROMPT`, shared with
  `/agribot`), roles are filtered to user/assistant, history capped at 8 turns
  and 4000 chars per message.
- **Upload before authorization.** multer wrote the 10MB body to disk *before*
  the role check, orphaning files on the 403 path. A `farmersOnly` middleware now
  runs first.
- **No request timeout** on any Groq call, so a hung upstream locked the chat
  input forever. All calls now use `GROQ_TIMEOUT_MS` (default 30s).

### 2.6 Shared temp file — cross-farmer diagnosis

Every `/predict` request wrote to the same hardcoded `temp_image.jpg`, and the
Flask dev server is threaded, so two concurrent uploads overwrote each other:
one farmer could be diagnosed from another farmer's photograph, or have their
file unlinked mid-read. Now `tempfile.mkstemp()` per request.

Also `/predict` required **both** CLIP and T2T to be loaded, so a missing
recommender disabled disease detection entirely. Classification now needs only
CLIP, and the recommendation falls back to the curated table.

### 2.7 XSS sink closed and supply chain pinned

`renderMarkdown` fed `marked.parse()` straight into `v-html` — and returned the
**raw string** when marked was unavailable, i.e. it failed *open* into the sink.
It now sanitises with DOMPurify and fails *closed* to escaped text.

`axios` and `marked` were loaded **unpinned**, so whatever the CDN published next
would execute in every visitor's browser. All four scripts are now
version-pinned with real SHA-384 SRI hashes, computed from the actual downloaded
files, plus `crossorigin` and `referrerpolicy`.

### 2.8 Runtime configuration

`http://localhost:3001` and `:5000` were baked into `index.html` and `js/app.js`,
so the app could not be deployed and every request would be blocked as mixed
content over HTTPS. Replaced with a `window.__PESTVID` block whose defaults are
port-aware, because the app is run three different ways:

| How it is served | Resolves to |
|---|---|
| reverse proxy on 80/443, or backend serving `public/` on 3001 | `/api`, `/ml` (same-origin) |
| `public/server.js` on port 3000 (**does not proxy `/api`**) | `http://host:3001/api`, `http://host:5000` |
| opened as `file://` | `http://127.0.0.1:3001/api` |

Both branches were checked by evaluating the config block directly under Node
with a stubbed `window.location`.

### Verification

| Check | Result |
|---|---|
| 10 concurrent purchases of one listing | **1x 201, 9x rejected** — exactly one Purchase |
| invest over the remaining goal | **409** Amount exceeds the remaining goal |
| invest exactly the remainder | **201** |
| invest 1 more after full | **409** |
| final DB state | `50/50 funded`, `40/80 partially_funded`, `0/120 pending` — no over-funding, correct transitions |
| `GET /videos` and `/listings` with no token | **401** both |
| `GET /videos?farmerId=...` with no token | **401** (bypass closed) |
| `cid` / `videoFileHash` in browse response | **absent** |
| ReDoS payload as `?crop=` | **200**, no hang |
| `/ai/chatbot` with attacker `systemPrompt` | prompt ignored, server-owned |
| all JS and Python | parses / compiles |
| `npm audit` | **found 0** |

### Regression sweep — all zero

hardcoded model ids, `debug=True`, `host='0.0.0.0'`, hardcoded localhost,
`role !== 'admin'`, chained `.populate()`, unpinned CDN scripts.

### Process note

Three of my own edits broke a file and were caught by the syntax check before
moving on: a mangled regex character class in `videos.js`, a stray `\x01`
control character from a bad regex backreference in `Notification.js`, and an
apostrophe inside a single-quoted JS string in `index.html`. Nested
shell/Python/JS quoting is the recurring cause; every edited file is now
`node --check`ed or `py_compile`d immediately after writing.

One earlier verification run was invalid and was redone: `pkill` does not work
on Windows, so a stale server kept serving old code and two Phase 2 results were
measured against pre-fix behaviour. Killed with `taskkill //F` and re-tested from
a fresh seed. Phase 0 and Phase 1 results were unaffected, since that server was
started after those edits.

---

## Follow-up to 2.5 — the cid decision, reconsidered

I flagged at the end of Phase 2 that stripping `cid` from the listings browse
response might break video previews. On investigation **my Phase 2 change was
wrong on both counts**, so it has been reverted:

- `loadPublicPlatformData()` fetches `GET /api/listings` **anonymously** and its
  own comment says *"public endpoint, no auth required"*. Requiring a token
  broke marketplace browsing for logged-out visitors.
- The browse card plays the listing video directly
  (`<video :src="getVideoUrl(l.cid, ...)">`, index.html ~642) and the purchase
  modal reads `selectedListing.videoFileHash` (~1162). Both fields are needed.

Re-reading the product: buyers purchase a physical **crop batch**; the video is
public marketing that evidences pesticide provenance. So the video was never the
paid artefact and there was no paywall to bypass. The original audit finding was
about *videos* as paid content and I applied it to *listings* by analogy — that
was my error, not the audit's.

Verified separately that `GET /listings/farmer/:farmerId` never lost `cid`, so
the farmer's own dashboard (`l.cid.substring(0, 8)`, which would have thrown)
was unaffected.

`GET /api/videos` **keeps** its authentication: nothing fetches it anonymously.

### New: `GET /api/listings/:id/media` (purchase-gated)

Added the gated path anyway, because it is the correct home for any media that
does become genuinely paid — publishing a CID *is* publishing the file, since
anyone can fetch it from an IPFS gateway.

Entitlement is the farmer who owns the listing, or a buyer with a matching
`Purchase` row. Returns `cid`, `storageType`, `videoFileHash`, plus the
`entitlement` kind and the purchase date / txHash as a verifiable receipt.

| Caller | Result |
|---|---|
| no token | **401** |
| investor who never purchased | **403** |
| farmer who owns the listing | **200** `entitlement: owner` |
| buyer **before** purchasing | **403** |
| buyer **after** purchasing | **200** `entitlement: purchaser` |
| malformed id | **400** |

---

## Phase 3 — Delete the dead weight (complete)

Removed **89 files**. Every deletion is a `git rm`, so all of it is recoverable
from history.

### 3.1 The abandoned Supabase branch — 21 files
`backend/models-supabase/` (3), `config/supabase.js`, `package-supabase.json`,
`setup-supabase.js`, `test-supabase.js`, `test-supabase-connection.js`, and 14
Supabase / cloud-migration documents.

The migration never happened: `server.js` still uses Mongoose, the route modules
it needed were never committed in any revision, and `@supabase/supabase-js` was
in no manifest. Two launcher scripts started MongoDB while telling the user it
was running on Supabase. `backend/config/` is now empty and gone.

`CREATE_TABLES.sql` went with it. Worth recording why, in case it is ever
revived: it created all 11 tables with **no Row Level Security**, and the only
RLS remediation anywhere in the repo used `USING (true) / WITH CHECK (true)` on
5 of 11 tables — which is not row-level security, it is the absence of it with
extra steps.

### 3.2 One Flask server, one launcher — 14 files
Deleted `flask_server_simple.py`, `flask_server_with_models.py`,
`flask_server_with_frontend.py`, `serve_frontend.py`, `test.py`, and **all nine**
`.bat` launchers.

Those nine started five different combinations across ports
3000/3001/3002/5000/8080, and `start_all_servers.bat` opened the frontend over
`file://`, where every fetch fails CORS. `flask_server.py` is now the only AI
server and `start.bat` the only launcher, with one documented port map:

| Port | Serves |
|---|---|
| 3001 | Node API **and** the static frontend |
| 5000 | Flask AI server (optional) |

The new `start.bat` refuses to run without `backend/.env`, and points at
`npm run dev:mem` when MongoDB is not installed.

### 3.3 Forty documents to three
Kept `README.md`, `FIXLOG.md`, `requirements.txt`. Deleted 37, including nine
overlapping "status" files (`FINAL_STATUS`, `PROJECT_RUNNING`,
`PROJECT_RUNNING_NOW`, `RUNNING_STATUS`, `SETUP_COMPLETE`, `PROJECT_STATUS`, …),
six "fix summary" files, and four mutually contradictory getting-started guides
(`START_HERE` vs `START_HERE_NOW` vs `OPEN_ME` vs `README_FIRST`).

The README was rewritten so every claim is checkable, and each one was then
checked against the code. Removed as false or unsupported:

| Old claim | Reality |
|---|---|
| "Production Ready" badge | no tests, no CI, no Dockerfile |
| "Real-time messaging" | no WebSocket, no SSE, not even polling on the server |
| "blockchain" / web3 | the only reference is a commented-out `walletAddress` field |
| "75%+ accuracy" | that string was the hardcoded value in the fake endpoint |
| InstructBLIP component | never successfully loaded, let alone trained |
| Port 3002 | the server listens on 3001 |

It now states plainly that disease detection needs weights that are not in the
repo, that the 84.10% figure is an upper bound and should not be quoted, that
the treatment recommender is effectively a lookup table with no dose or
jurisdiction data, and that the RAG pipeline is not wired into the UI.

### 3.4 Orphaned frontend and dead API surface — 10 files
Verified unreferenced before deleting, rather than assuming:

| File | Evidence |
|---|---|
| `public/css/landing.css` | 0 references in any HTML |
| `public/css/styles.css` | 0 references; a dead fork of the inline `<style>` |
| `public/js/components/ChatInterface.vue` | 0 references; a `.vue` file cannot load without a build step |
| `public/fresh.html`, `test-analysis.html`, `clear-cache.html`, `test_live.html` | scratch pages |
| `public/index_new.html`, `index_original.html.bak` | duplicate copies that held live Pinata JWTs |

**Kept** `chat-interface.css`, `js/app.js`, `js/components/ChatInterface.js` and
`js/pages/MessagingPage.js` — all four *are* referenced by `index.html`, and all
four return 200 from the running server.

Dead routes removed and unmounted from `server.js`:

- `routes/avatarmessages.js` + `models/AvatarMessage.js` — the only writer of
  `AvatarMessage` was the route itself, and the frontend never called it: a
  self-contained dead loop.
- `routes/conversations.js` — 0 frontend calls, duplicated `messaging.js`, and
  its `POST` was a guaranteed 500 (`ObjectId()` without `new` on Mongoose 8).

**Deliberately kept `routes/notifications.js`.** The frontend never calls it, so
it looks dead — but `new Notification(...)` appears in **six** route files, so
the data is actively produced server-side. Deleting the read path would orphan
six writers. The real gap is missing frontend wiring, not dead backend code.

### A regression I caused and fixed

Deleting `models/AvatarMessage.js` broke `seed.js`, which required it at line 24.
The seed threw, the database stayed empty, and **login started failing** — every
authenticated route returned 403. Caught by the post-deletion smoke test.
Removed the require, the `deleteMany`, and the 18-line avatar-message seed block.

This is the argument for smoke-testing after deletions specifically: `node
--check` passes on a file whose `require()` target no longer exists, because the
failure is at runtime, not parse time.

### Verification

Fresh in-memory MongoDB, full reseed, every surviving router exercised:

```
seed: 6 users · 6 videos · 4 listings · 4 funding requests
      4 investments · 3 conversations · 9 messages · 8 notifications

login                                     OK
GET /listings                             200   (public)
GET /funding-requests                     200   (public)
GET /videos                               200
GET /users/public                         200
GET /auth/me                              200
GET /notifications/user/:id               200
GET /messaging/conversations/:id          200
GET /purchases/buyer/:id                  200
GET /transactions/user/:id                200
GET /investments/investor/:id             200

GET /api/avatarmessages                   404   (deleted)
GET /api/conversations                    404   (deleted)

GET /            (frontend)               200   292,432 bytes
  chat-interface.css 200 · app.js 200 · ChatInterface.js 200 · MessagingPage.js 200
```

Every README claim re-checked against code: 11 routers mounted, 10 models
present, `npm run dev:mem` exists, both `.env.example` files present,
`/listings/:id/media` present, `/listings` and `/funding-requests` unauthenticated,
no web3/socket.io dependency, `GROQ_MODEL` config-driven in all four call sites,
DOMPurify wired with 4 SRI hashes.

### File count

| | Before | After |
|---|---|---|
| Root `.md` / `.txt` | 40 | 3 |
| `.bat` launchers | 9 | 1 |
| Flask servers | 4 | 1 |
| Node entry points | 4 | 2 (`server.js`, `dev-server.js`) |
| Backend routers | 13 | 11 |
| Mongoose models | 11 | 10 |

---

## Phase 4 — Make the model honest (code complete, training not run)

No ML code was executed: this machine has no GPU and running it was explicitly
out of scope. Everything below is **prepared and statically verified**, ready to
run on Kaggle or Colab. The pure-logic functions were unit-tested against
synthetic data, which found one real bug (see below).

### New: `Pestivid/train_potato.py`

A single leak-free pipeline replacing the notebook's two competing ones. The
notebook had 111 cells with a duplicate pipeline starting around cell 45, so
patching it in place would have left two versions of every decision.

| # | Change | Detail |
|---|---|---|
| 4.1 | **Recover the missing 39% of the dataset** | `collect_images()` is recursive, case-insensitive and multi-extension, and asserts against the 3,076 figure from the dataset paper with a per-class breakdown. The old `glob("*/*.jpg")` went one level deep, matched lowercase only, and found 1,885 with no check. |
| 4.2 | **No text branch at all** | `Head(features) -> logits`. Nothing but pixels reaches the model, so the leak is structurally impossible rather than merely absent. |
| 4.3 | **DINOv2 backbone, frozen** | `facebook/dinov2-large`, features extracted once. `--backbone clip` reproduces the old backbone for comparison. Frozen means a linear probe trains in seconds. |
| 4.4 | **Augmentation that runs** | `RandomResizedCrop` + `TrivialAugmentWide` + `RandomErasing`, train split only. Augmentation must happen *before* a frozen backbone, so it cannot be applied to cached features — the script extracts `--tta N` augmented copies instead (default 4). |
| 4.5 | **Class-balanced loss** | Cui et al. 2019 effective-number weighting, and **macro-F1 is the selection and headline metric**, not accuracy. |
| 4.6 | **Grouped k-fold + calibration** | `StratifiedGroupKFold`, with a dHash + union-find near-duplicate grouper so multiple frames of one plant cannot straddle a split. Temperature scaling fitted on an inner split; ECE reported; per-fold mean ± std. |
| 4.7 | **Abstention + explainability** | Mahalanobis OOD gate on L2-normalised features, a confidence floor, and an attention-rollout overlay. |

Also emits `model_card.md` with the honest numbers, the per-class table, the
abstention thresholds, and an explicit "supersedes" note.

### New: `pestvid_complete_project/potato_infer.py`

Inference that matches training — which the old code did not:

| | Old | New |
|---|---|---|
| Forward passes per image | **7** (one per candidate prompt) | 1 |
| What is read | the 7×7 diagonal | the logits |
| Confidence | softmax over 7 scalars from 7 *separate* passes | temperature-scaled probability, fold-ensembled |
| Reject option | none | OOD gate + confidence floor |
| `torch.load` | no `weights_only` | `weights_only=True` |

`get_classifier()` returns `None` when artifacts are absent, so the caller must
answer 503 rather than guess. Verified: it does.

### `flask_server.py` rewired

- The `CLIPFineTuner` class, `get_clip_disease_prediction`, `get_t2t_recommendation`
  and the 7 `text_prompts` are **deleted** — 113 lines. The file went 530 → 453.
- `/predict` calls the new classifier, returns `not_a_leaf` / `uncertain`
  verdicts **with no disease name and no treatment text**, and supports
  `?heatmap=1` for the attention overlay.
- The Flan-T5 recommender is deliberately not loaded. Its documented curve ends
  at loss 3.2368 after 100 optimizer steps on 7 examples, so it never learned
  them; the quality heuristics always fired and the curated dict supplied the
  text anyway. Now the dict is the only source, which is at least reviewable.
- Treatment text ships with an explicit `recommendation_source` stating it has
  no dose, no pre-harvest interval and no jurisdiction check.

Two startup messages were corrected because they were actively misleading:
`"OPTIONAL RAG DEPENDENCIES MISSING (Python 3.14 compat)"` misdiagnosed a hard
`python < 3.14` floor in `langchain-pinecone` as a transient gap, and
`"Core disease prediction (CLIP/T5) still works fine!"` was false.

### Documentation corrected

- `potatoleaf-vlm-fc93c1.ipynb` — a **SUPERSEDED** banner as cell 0, with the
  leak shown in code and a nine-row comparison table. Kept for provenance.
- `Pestivid/README.md` — the Results section now marks 84.10% **WITHDRAWN**,
  explains the mechanism, and gives the real published baselines for this
  dataset (EfficientNetV2B3 73.63, MobileNetV3-L 72.03, ResNet50 68.17, best
  87.82) with the guidance to treat anything above ~88% as leakage.

### A bug found by unit-testing

`fit_temperature` returned **0.0019** on a perfectly separable validation split.
The optimum genuinely is `T -> 0` when the model is never wrong, but `logits / T`
then overflows at inference. Added a clamp to `[0.05, 10]` plus a warning, since
a degenerate fit also signals a validation split that is too small or too easy
to calibrate on.

### Verification (no GPU, no models loaded)

| Check | Result |
|---|---|
| `train_potato.py`, `potato_infer.py`, `flask_server.py` | compile |
| `train_potato.py --help` | CLI resolves |
| dHash: identical vs random image | 0 bits vs 32 bits apart |
| Class-balanced weights on the real 11:1 imbalance | Nematode 3.234 vs Fungi 0.404 (8.0×), sums to n_classes |
| ECE: uniform vs confidently-wrong | 0.006 vs 0.880 |
| Mahalanobis: in-dist vs synthetic OOD | median 30.4 vs 464.3; **100%** of OOD flagged at the p99 threshold |
| Temperature on inflated / deflated logits | ×3 → T=3.333, ×0.3 → T=0.333 (inverts the scale exactly) |
| Degenerate temperature | clamped to 0.05 with a warning |
| `get_classifier()` with no artifacts | returns `None` → caller answers 503 |
| `import flask_server` with no artifacts | imports cleanly, `CLIP_LOADED=False`, 7 routes registered |
| Notebook after edit | valid JSON, 112 cells |

### What still needs a GPU

```bash
python Pestivid/train_potato.py \
    --data-root "<kaggle>/Potato Leaf Disease Dataset in Uncontrolled Environment" \
    --backbone dinov2 --folds 5 --tta 4 --out artifacts
```

Then copy `artifacts/` next to `flask_server.py` (or set `PESTIVID_ARTIFACTS`).

Two things to expect, both intended:

1. **The honest macro-F1 will be lower than 84.10%.** That is the point of the
   phase. A defensible number with error bars and a calibration curve is worth
   more than an unreproducible one.
2. **The image count must come out at 3,076.** If `collect_images()` warns, stop
   and fix the path before training — do not train on a truncated dataset.

Not done, and out of scope for this phase: replacing the curated treatment table
with a CIBRC-backed jurisdiction-aware dataset (Phase 5), and wiring the
`not_a_leaf` / `uncertain` verdicts into the frontend UI, which currently expects
a disease name in every response.

---

## Phase 5 — Rebuild the recommender (complete)

### 5.1 The T2T model is gone
Every reference removed: `T2T_BASE_MODEL_NAME`, `T2T_LOADED`, `t2t_tokenizer`,
the loader, the generation helper, the brittle quality heuristics, and the
`/health` field. `reload_and_verify_models.py`, `test_model_loading.py` and
`test_t2t_model.py` were deleted — they existed only to verify the two `.pth`
files. `test_setup.py` now checks for `artifacts/` and `treatments.json`.

It was never worth reviving. The documented loss curve ends at **3.2368** after
100 optimizer steps on 7 examples — perplexity ~25 for a model whose only job was
verbatim recall, so it never learned them.

### 5.2 `treatments.json` + `treatments.py`

7 conditions, 47 cultural practices, 11 chemical entries, 5 cited sources
(CIBRC major-uses, CIBRC registered products, CIBRC banned/restricted,
ICAR-CPRI, CABI Plantwise).

**The safety invariant.** A dose, pre-harvest interval or application rate is
emitted only for an entry whose `status == "reviewed"`. Enforced in one place
(`_sanitise_chemical`) rather than trusted to callers.

Every chemical entry is currently `needs_verification`, and that is deliberate:
the CIBRC crop-wise tables are PDFs I could not machine-read, and **inventing a
dose for an agrochemical is exactly the harm this table exists to prevent**.
Populating them is data entry against the cited PDFs, not a code change. The
loader logs a warning while any remain unverified.

Tested adversarially: a dose typed into the JSON *without* setting
`status: reviewed` is withheld, and a warning names the dropped fields. A
properly reviewed entry does serve its dose. So the failure mode is safe by
construction, not by discipline.

Domain detail worth keeping: the Phytophthora entry carries an explicit note
that **metalaxyl resistance in *P. infestans* is established in India**, so it
must not be used as a solo curative. A static string could never express that;
a versioned table with resistance notes can.

### 5.3 Healthy is a first-class case
`is_disease: false`, zero chemical options, six monitoring practices, and an
explicit `do_not`: *"Do not spray because a scan came back healthy."* The
deployed prompt previously asked the model for treatment of
`"potato healthy disease"`.

---

## Phase 6 — Rebuild the chatbot (complete)

### 6.1 RAG is now in the user's path
`/api/ai/agribot` tries `POST {FLASK_URL}/chat` first and falls back to direct
Groq. Responses carry `source: 'rag' | 'llm'` and `grounded: bool`, and the UI
records both — an ungrounded answer deserves less trust and now says so.

Caught while wiring this: my first version returned `{text}` while the existing
Groq path returned `{answer}`, and the frontend reads `.answer`. Unified.

### 6.2 Conversation memory
Last 8 turns, roles filtered to user/assistant, 4000 chars each, on all three
paths (`/agribot`, Flask `/chat`, `ask_via_groq`). The frontend now sends the
history it was already keeping in `localStorage` and never transmitting.

### 6.3 Multilingual embeddings
`COHERE_EMBED_MODEL` defaults to `embed-multilingual-v3.0` — same vendor, same
API, same 1024 dimensions, so the index geometry still fits. `PINECONE_INDEX` is
configurable too, because **switching the model requires a re-embed**: vectors
from the English and multilingual models are not comparable. Documented in
`.env.example` rather than left as a trap.

### 6.4 Retrieval rebuilt
- **Score floor** (`RAG_SCORE_FLOOR`, default 0.30). Without it, an out-of-corpus
  question retrieved the three least-bad chunks and the model answered
  confidently from them.
- **Abstention** when nothing clears the floor, instead of answering from nothing.
- **Citations**: page numbers now travel as metadata. The ingestion notebook was
  injecting `--- Page N ---` inline into the text, which embedded as semantic
  noise *and* was unrecoverable at answer time. Now it chunks per page and
  carries `page` in metadata.
- **Injection defence**: retrieved text is wrapped in `<<<SOURCE [n]>>>` markers
  and the prompt states that content inside them is reference material, not
  instructions.
- **Dose refusal in the prompt**: never state a dose or PHI unless a source gives
  it explicitly.
- **Ungrounded path hardened**: temperature 0.7 → 0.2, and a separate persona
  that refuses to state doses at all, because it has no retrieval to anchor it.

**Three audit claims I have to correct here.** I reported them and they were
wrong:

| Claim | Reality |
|---|---|
| Cohere `input_type` asymmetry is wrong | The notebook uses `embed_documents` for indexing and `embed_query` for queries. The LangChain wrapper handles `input_type` correctly. |
| `similarity_search` drops chunks lacking `text` | Ingestion sets `"text": chunk` in metadata, with a comment saying why. Not applicable to this corpus. |
| Re-running ingestion duplicates the corpus | Ids are `leaf_train_chunk_{i}` — deterministic, so upsert overwrites. Already idempotent. |

Only the page-marker problem was real. I have kept the defensive `text`-key
fallback and the `page` metadata anyway, since both are cheap and the fallback
protects against a differently-built index.

### 6.6 Evaluation that can detect hallucination
`eval_rag.py` measures hit rate, citation rate, abstention rate, and
**dose leaks** — an answer stating a dose that no retrieved source supplied.
`--ci` exits non-zero on regression.

This replaces `tes.ipynb`'s scorer, which rated a fully hallucinated answer
**5.0/10** with most points for word count and term frequency. A metric that
cannot detect hallucination launders it.

`--make-golden` scaffolds a golden set including prompt-injection probes, with a
note to build the real one from Kisan Call Centre transcripts — real farmer
questions with the advisor's actual answer — rather than invented ones.

Targets, from Farmer.Chat (arXiv:2409.08916): context precision 71%,
faithfulness high on ~80%, answer rate ~75%. Their finding that **66% of
unanswered queries were content gaps** is the important one: one PDF will not
reach those numbers, and that is a corpus problem, not a prompt problem.

---

## Phase 7 — Product (partial: 7.1, 7.4a, 7.5 done)

### 7.1 Late-blight forecasting — `blight_risk.py`
Implements the **Smith Period** (Smith 1956): two consecutive days with min temp
≥ 10 °C and ≥ 11 hours at RH ≥ 90%. Chosen over a bespoke rule because it is
published and independently verifiable.

Risk ladder: `low → moderate → high → very_high`, each with a specific action.
Plus a **spray-window** check — rainfastness, wind band, and Delta T via a
Stull wet-bulb approximation — because telling a farmer *which* fungicide is half
the advice and *when* decides whether it works or washes off.

Exposed as `POST /blight-risk`.

**INDO-BLIGHTCAST is deliberately not encoded.** It is the India-specific
calibration from ICAR-CPRI Shimla, validated separately across the
Indo-Gangetic plains, plateau and hills — and I could not verify its
coefficients against the primary source. Inventing thresholds for a spray
decision is the same class of error as inventing a dose. The interface is shaped
so it can be added as a second scorer with `calibrated_for="IN"`, and every
response carries a caveat naming the gap.

### 7.4a Abstention verdicts rendered in the UI
Closes the gap I flagged at the end of Phase 4: the backend returned
`not_a_leaf` / `uncertain` verdicts the frontend could not display. Now:

- `not_a_leaf` → "Not a potato leaf", no disease name, no treatment
- `uncertain` → "Uncertain — no diagnosis given"
- `model_unavailable` → a clear offline message, `parsedAnalysis` cleared
- a successful prediction surfaces cultural practices, chemical options, `do_not`,
  escalation advice, and an explicit warning when doses are withheld
- confidence is labelled "uncalibrated" when the model says so

Full confidence-gated *escalation into the marketplace* (7.4b) is not built —
that is a product flow, not a patch.

### 7.5 CI — `.github/workflows/ci.yml` + `tools/guards.py`
Three jobs: backend (parse → **boot** → smoke → `npm audit`), python (compile,
guards, safety invariants), ml-leak-gate.

The backend job **boots the server**, not just parses it, because
`node --check` passes on a file whose `require()` target is missing — that is
exactly how deleting `models/AvatarMessage.js` broke `seed.js` in Phase 3.

**`tools/guards.py` — and why it is a script, not greps.** My first version was
inline `grep -v` lines. Two things went wrong, both instructive:

1. **Four of six guards failed on their own documentation.** The comments
   explaining each bug contained the forbidden pattern. A guard that cannot
   coexist with its own explanation is a guard someone deletes.
2. Fixing that by stripping strings *and* comments made **two guards silently
   never fire** — because a fabrication prompt, a hardcoded model id and a bind
   address *are* string literals.

So each guard now declares whether it needs strings kept. Verified both
directions: all 6 pass on the clean tree, and all 6 fail against a probe file
containing one deliberate regression each. A guard that has never been seen to
fail is not a guard.

Guards: no-fabricated-diagnosis, no-hardcoded-model-id, no-debug-server,
no-credential-literal, no-label-leak, no-chained-document-populate.

### Not built, and why
| Item | Reason |
|---|---|
| 7.2 on-device inference + PWA | Needs a trained model first (Phase 4 has not been run), then distillation and quantisation. Weeks, and it is the right *next* project. |
| 7.3 Voice-first Indic pipeline | Needs Bhashini/ASR credentials and a language-priority decision that is yours, not mine. |
| 7.4b Escalation into the marketplace | A product flow spanning pricing and expert onboarding. |
| CIBRC dose population | Data entry against PDFs. The schema and the safety gate are ready. |

---

## Final test pass

### Static
| Check | Result |
|---|---|
| Python files compile | 10 + `train_potato.py` |
| JS files parse | 1,334 |
| `index.html` Vue block | parses |
| `treatments.json`, `golden.json`, `ci.yml` | valid |
| 3 notebooks | valid JSON (14 / 18 / 112 cells) |
| Regression guards | **6/6 pass** |
| Guards catch regressions | **6/6 fail on probes** |

### Logic — 29/29
treatments (5): invariant holds, injected dose withheld, reviewed dose served,
Healthy not a disease, unknown class fails safe.
blight (10): Smith boundaries at 9.9/10.0 °C and 10/11 h, risk ladder, empty
input, spray blockers and allowances, advisory serialises.
eval (7): g/L, kg/ha and PHI detected; dose-free text ignored; abstention and
citation detectors both directions.
infer (1): no artifacts → `None`, so the caller 503s.
train (6): rarest class gets the largest weight, weights sum to n_classes,
ECE ≈ 0 / > 0.8, degenerate temperature clamped, dataset size asserted.

### Live backend — 26/26
auth 3 · public-vs-protected 5 · deleted surface 2 · money 3 · entitlement 4 ·
ai 4 · frontend 1 · input-validation ordering 5.

### Live Flask — 14/14
health 2 · predict refuses with no diagnosis in the payload 2 ·
blight-risk 5 (including 400/413 malformed) · chat 4 · advice 2.

### One real bug found by this pass
`/agribot` and `/chatbot` checked `GROQ_API_KEY` **before** validating input, so
an oversized question returned `500 not configured` — blaming the operator for a
client error and hiding the real cause. Reordered so validation runs first;
oversized now returns 413 and malformed returns 400 on both routes.

---

## Round 2 — honesty fixes, settlement model, and the first live LLM test

### The blockchain claims

I previously removed "blockchain" from the README as a false claim. That was
wrong: it is an unbuilt *requirement*, not an overclaim. Corrected — the UI now
says "planned, not yet implemented" rather than either claiming it works or
denying it was ever intended.

What was genuinely indefensible, and is now removed:

- **`solanaExplorerUrl()`** built `https://explorer.solana.com/tx/<hash>?cluster=devnet`
  for hashes produced by `generateSimulatedTxHash()` — values that have never
  existed on any chain. It was rendered as a clickable link in the transaction
  table, the purchase modal and the investment modal, on screens that ask
  someone for money. A nearby "(Demo Link)" label does not fix that. Removed at
  all three sites; the identifier is still shown, labelled "Platform reference
  (not a blockchain transaction)".
- Eight landing-page and modal claims corrected: "video-verified", "complete
  confidence and traceability", "Solana blockchain-simulated transactions",
  "All investments and harvests recorded on immutable ledger", "Platform
  Verification (Simulated)". Replaced with what is actually true — IPFS gives a
  content address that changes if the file changes, blockchain anchoring is
  planned, and the funding flow is a demonstration with no real money.

### Client-side payout fabrication removed

`updateInvestmentProgress()` fabricated the entire investment lifecycle in the
investor's own browser: it derived `progress` from elapsed wall-clock time since
`investmentDate`, added a pseudo-random increment seeded from a hash of the
investment `_id`, and on reaching 100 set `status='harvested'`, computed
`payoutAmount`, minted a `payoutTxHash` from `Math.random()`, pushed a
`status:'confirmed'` payout Transaction, and wrote it all to `localStorage`.

None of it reached the server. Two browsers showed different harvest states for
the same investment, and the payee was inventing their own payout.

Meanwhile `PUT /api/investments/:id/progress` already existed and already
authorised correctly — only the farmer who owns the parent project — and **no UI
ever called it**. Replaced the function with `refreshInvestments()`, which
fetches server state. Same class of defect as the fabricated diagnoses removed
in Phase 0.

### G8 settlement: both modes now exist

The goal offers the farmer two ways to settle — share the profit, or repay the
whole fund. Neither was implemented. The only formula was
`payoutAmount = amount * (roi / 100)`, which returns the yield and never the
principal: 100 at 15% ROI paid out **15, not 115**. `investorShare` was
required, validated, copied to the Investment, displayed as "Your Share %" — and
used in **zero calculations**, because there was no revenue figure to take a
share of.

Added to `FundingRequest`: `settlementMode` enum `['profit_share','full_repayment']`,
`harvestRevenue`, `inputCostBasis`, `harvestReportedAt`, and an `outcome` enum
`['pending','harvested','partial_loss','total_loss']` — crops fail, and without
that field the model implicitly promised a return, which is exactly what sank
the real crowdfunding platforms.

`routes/investments.js` now computes:

| Mode | Payout |
|---|---|
| `full_repayment` | principal + (principal x roi%) |
| `profit_share` | principal + pro-rata share of (`investorShare`% of realised profit) |
| `outcome: total_loss` | 0 |

Every payout records a human-readable `payoutBasis` string so an investor can
audit the arithmetic instead of being handed a bare number. `payoutTxHash` is
now prefixed `ref_payout` rather than `sim_payout`, and commented as an internal
reference.

### First live LLM test — and what it caught

The user supplied a Groq key, so the chatbot was exercised end to end for the
first time. Keys were written only to `.env` and `backend/.env`, both confirmed
gitignored before writing, and the `no-credential-literal` guard (which matches
`gsk_`) still passes.

Confirmed working:

- `openai/gpt-oss-120b` is live and correct — "late blight" -> *Phytophthora
  infestans*; "brown concentric rings" -> Early Blight / *Alternaria solani*.
  The Phase 1 default was the right choice.
- Conversation memory works: a bare "What dose?" was answered in the context of
  the previous late-blight turn.
- The off-topic guard works: a poem request plus "ignore your instructions" was
  declined.

**`GET /openai/v1/models` returned 13 models and NOT ONE llama chat model** —
only `openai/gpt-oss-*`, `qwen/qwen3.6-27b`, whisper and prompt-guard. That is
direct confirmation that `llama3-70b-8192` is gone, not an inference from
documentation. Also noted for later: `whisper-large-v3` (the ASR for a voice
feature) and `meta-llama/llama-prompt-guard-2-*` (a prompt-injection classifier)
are available on the same key at no extra cost.

Two real bugs the live test found, both invisible to static analysis:

1. **Empty completions.** `gpt-oss` is a reasoning model — 53 of 83 completion
   tokens went to reasoning on a short answer. With a low `max_tokens` the call
   returns HTTP 200 with **empty `content`**, which would have shown the farmer a
   blank reply. All three call sites now detect this, log the reasoning-token
   count, and return an explicit error instead of an empty string. `max_tokens`
   raised 500/600 -> 1200.

2. **A safety gap between the two chat paths.** Asked for a mancozeb dose, the
   Flask path refused correctly but `/api/ai/agribot` returned a table of
   "typical field rates" — because the dose-refusal rule existed only in
   `GROQ_UNGROUNDED_PROMPT` (Flask) and not in `AGRIBOT_SYSTEM_PROMPT` (Node).
   Node is the path the UI actually calls, so the weaker prompt was the live one.
   `AGRIBOT_SYSTEM_PROMPT` now carries the full rule set and is documented as
   needing to stay in sync. Retested with a deliberately insistent prompt
   ("give me the exact rate per litre") and it refuses, pointing at the product
   label and a licensed agronomist.

That second one is the argument for live testing: no amount of reading would
have surfaced a divergence between two prompts that are individually reasonable.

### Test results

| Suite | Result |
|---|---|
| Python compile (9 files + train_potato.py) | pass |
| Node parse (all routes, models, entry points, frontend JS) | pass |
| `index.html` Vue block | parses |
| treatments.json / golden.json / ci.yml / 3 notebooks | valid |
| Regression guards | **6/6 pass** |
| Logic tests | **29/29 pass** |
| Live backend (incl. real LLM calls) | **29/29 pass** |
| Fake-chain sweep | 0 explorer links, 0 payout minting, 0 "immutable ledger" |

---

## Phase 4 closure — the pipeline now provably runs

**Training itself is still yours to run.** No GPU here, the Kaggle dataset is not
on disk, and running ML code was explicitly out of scope. What was closeable was
the verification gap: `train_potato.py` had only ever been statically checked and
unit-tested per function. The orchestration had never executed once.

### New: `Pestivid/test_train_pipeline.py`

Runs the entire pipeline on CPU in seconds, with no GPU, no dataset and no model
download:

- synthetic class-correlated images, with a deliberate near-duplicate planted in
  every class so the grouping code has real work to do
- the frozen backbone replaced by a stub returning deterministic, separable
  pseudo-features, so no weights are downloaded and no real inference runs

It asserts 35 things: collection (including uppercase `.JPEG`), duplicate
grouping, class-balanced weighting, transform shapes, augmentation actually being
stochastic, one head written per fold, every artifact present, macro-F1 in range
and above chance, temperature fitted and inside the clamp, and — the important
part — that the checkpoint carries every key `potato_infer.py` reads, that the
head reloads, and that the OOD scorer separates far points from centroids.

**35/35 pass.**

### Two real bugs it found, both invisible to static analysis

**1. `int32` vs `int64` — would have crashed every Windows training run.**
`y_all = np.array([...])` inherits numpy's platform default, which is **int32 on
Windows**. `F.cross_entropy` requires Long, so temperature fitting died with
`RuntimeError: expected scalar type Long but found Int`. On Linux, where numpy
defaults to int64, it would have passed — a platform-dependent failure that only
appears when you run it. Fixed at six sites, plus a defensive `targets.long()`
inside `fit_temperature` so it is safe however it is called.

**2. The trainer and the inference loader could never talk to each other.**
`train_potato.Head` is an `nn.Module` holding `self.net`, so its state_dict keys
are `net.1.weight`. `potato_infer._build_head` returned a bare `nn.Sequential`,
expecting `1.weight`. So **`potato_infer.py` could not load any checkpoint
`train_potato.py` produced** — and both files compiled, and both passed their own
tests. This is the one that would have hurt most: train for an hour on Colab,
then discover the artifact is unloadable. `potato_infer` now defines `_Head`
mirroring the trainer exactly, with a comment saying the two must change
together. Verified: state_dict keys match for both the linear-probe and
hidden-layer shapes.

### Train-to-infer handoff verified

With both backbones stubbed, `train_potato.main()` writes artifacts and
`PotatoClassifier` loads them: 3 folds ensembled, classes recovered, OOD params
loaded, `predict()` returns a valid verdict with probabilities summing to 1, and
the attention heatmap renders. **Handoff verified.**

### CI additions

Two new steps in the `ml-leak-gate` job so neither bug can return silently:

- the full pipeline integration test (CPU torch, seconds)
- an explicit head-parity assertion comparing `train_potato.Head` and
  `potato_infer._build_head` state_dict keys at both shapes

### Phase 4 status

| Item | State |
|---|---|
| Leak-free pipeline written | done |
| Runs end-to-end | **verified, 35/35** |
| Artifacts load into inference | **verified** |
| Guarded in CI | **done** |
| Trained on the real dataset | **yours** — one Colab/Kaggle session |

The remaining step is `python Pestivid/train_potato.py --data-root <dataset>
--backbone dinov2 --folds 5 --tta 4 --out artifacts`, then copy `artifacts/`
next to `flask_server.py`. Expect the honest macro-F1 to be lower than 84.10%,
and expect `collect_images()` to report 3,076 images — if it warns, fix the path
before training.
