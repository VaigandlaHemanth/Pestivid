# PestVid

An agricultural platform combining a crop marketplace, project funding, farmer
messaging, and AI-assisted plant disease diagnosis.

> **Status: work in progress, not production ready.** A full static audit in
> August 2026 found the AI features non-functional and several security holes.
> Remediation phases 0–7 are complete, and the classifier is **trained** on all
> 3,076 images with no label leak: macro-F1 0.7180 ± 0.0116, accuracy
> 0.7585 ± 0.0102, ECE 0.0401 on 5-fold StratifiedGroupKFold.
> [FIXLOG.md](FIXLOG.md) records exactly what was changed and verified.
> Read [Known limitations](#known-limitations) before demoing this.

---

## What actually works

| Feature | State |
|---|---|
| Register / login (JWT, bcrypt) | Working |
| Crop listing marketplace + purchase | Working, server-validated prices |
| Project funding + investment | Working, atomic and goal-bounded |
| Transaction history | Working |
| Farmer ↔ buyer messaging | Working (polling, not real-time) |
| AgriBot chat | Working **only** with a `GROQ_API_KEY` |
| Disease detection from a photo | **Working** — DINOv2-base + linear probe, macro-F1 **0.718 ± 0.012** (5-fold grouped CV) |
| Late-blight risk forecasting | Working — `POST /blight-risk` |
| Treatment guidance | Working — curated table; doses withheld until CIBRC-verified |
| RAG expert chat | Working when Flask is up and the index is populated; falls back to direct Groq otherwise |

Anything not in this table does not exist. In particular there is no blockchain,
no on-chain settlement, and no real payment processing — `txHash` values are
simulated strings, and prices are denominated in a fictional unit.

---

## Requirements

- **Node.js** 18+
- **MongoDB** running locally, **or** use the in-memory mode below
- **Python 3.13** — only for the optional AI server
  (`langchain-pinecone` requires `python < 3.14`, so 3.14 will not install)

---

## Setup

```bash
git clone <this repo>
cd pestvid_complete_project

# 1. Backend dependencies
cd backend && npm install && cd ..

# 2. Configuration — both files are required, neither has a default
cp .env.example .env                  # AI server keys
cp backend/.env.example backend/.env  # database + JWT + Groq

# 3. Generate a JWT secret and put it in backend/.env
node -e "console.log(require('crypto').randomBytes(48).toString('base64'))"
```

The server refuses to start without `MONGODB_URI` and `JWT_SECRET`. That is
deliberate — it used to print `Server running on port 3001` and *then* crash on
an unhandled `TypeError`, which looked like a successful boot.

### Optional: the Python AI server

```bash
python -m venv .venv313 --python=3.13    # or however you pin 3.13
pip install -r requirements.txt
```

---

## Running

**Windows:** double-click `start.bat` — the only launcher.

**No MongoDB installed?** This starts an in-memory MongoDB, seeds it, and runs
the API:

```bash
cd backend && npm run dev:mem
```

**Manual:**

```bash
cd backend && npm start          # API + frontend, port 3001
python flask_server.py           # AI server, port 5000 (optional)
```

Then open **http://127.0.0.1:3001**.

Do not open `public/index.html` from disk — `file://` breaks CORS on every
request. The Node server serves the frontend itself.

### Port map

| Port | Serves |
|---|---|
| 3001 | Node/Express API **and** the static frontend |
| 5000 | Flask AI server (optional) |

That is the whole map. The frontend resolves its own API base at runtime via
`window.__PESTVID`, so it also works behind a reverse proxy on 80/443 with no
rebuild.

### Seeded demo accounts

| Role | Email | Password |
|---|---|---|
| Farmer | `demo.farmer@pestivid.sim` | `password123` |
| Buyer | `demo.buyer@pestivid.sim` | `password123` |
| Investor | `demo.investor@pestivid.sim` | `password123` |

---

## Architecture

```
public/index.html          Vue 2 (CDN, no build step) — the entire frontend
  └── window.__PESTVID     runtime API config, port-aware

backend/                   Node 18 + Express + Mongoose
  ├── server.js            the only entry point
  ├── dev-server.js        same app on an in-memory MongoDB (npm run dev:mem)
  ├── routes/              11 routers, all JWT-authenticated except where noted
  ├── models/              10 Mongoose schemas
  └── seed.js              demo data

flask_server.py            Optional Python AI server
simple_ai_agent.py         Groq client + a rule-based fallback
```

### API

Everything is under `/api`. All routes require
`Authorization: Bearer <token>` except the four marked public.

| Method | Route | Notes |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | **public** |
| GET | `/auth/me` | |
| GET | `/users/public`, `/users/:id/public` | paginated; no email or phone |
| PUT | `/users/:id/profile` | own profile only; role is immutable |
| GET | `/listings` | **public** marketplace browse |
| GET | `/listings/:id/media` | owner or purchaser only |
| POST/DELETE | `/listings`, `/listings/:id` | farmer only |
| GET | `/videos`, `/videos/farmer/:id` | own videos only |
| GET | `/funding-requests` | **public** |
| POST/PUT/DELETE | `/funding-requests` | farmer only |
| POST | `/investments` | investor only; atomic, goal-bounded |
| PUT | `/investments/:id/progress` | project owner only |
| POST | `/purchases` | buyer only; price validated server-side |
| GET | `/purchases/buyer/:id`, `/transactions/user/:id` | own records only |
| GET/POST | `/messaging/*` | participants only |
| GET/PUT/DELETE | `/notifications/*` | recipient only |
| POST | `/ai/agribot`, `/ai/chatbot` | farmer only; server-owned prompt |
| POST | `/ai/predict-proxy` | farmer only; `503` when the model is absent |

Flask AI server on 5000: `POST /predict`, `POST /chat`,
`POST /simple-ai-advice`.

---

## Known limitations

Being explicit, because the previous documentation was not.

**Disease detection does not work out of the box.** It needs
`best_vlm_model.pth` (~688 MB) and `best_t2t_recommendation_model.pth`
(~293 MB), which exceed GitHub's file limit and are gitignored. Without them
`/predict` returns `503 model_unavailable`. It will not guess, and it will not
invent a diagnosis — three code paths that previously fabricated one (including
one that asked an LLM to *"make a realistic, randomised disease prediction"*
with no image attached) have been removed.

**The old 84.10% is withdrawn; the real number is lower.** That figure came from
a pipeline that fed the ground-truth label into the model through a frozen text
branch. The honest measurement, from `train_potato.py` over all 3,076 images with
5-fold StratifiedGroupKFold (near-duplicate photographs grouped so they cannot
straddle a split):

| Metric | Value |
|---|---|
| **macro-F1** | **0.7180 ± 0.0116** |
| accuracy | 0.7585 ± 0.0102 |
| ECE (after temperature scaling) | 0.0401 |

For context, the dataset paper's own baselines: EfficientNetV2B3 73.63,
MobileNetV3-Large 72.03, ResNet50 68.17 (accuracy). This is a frozen backbone
with a linear probe and no augmentation, on a *harder* grouped split than theirs.

Per-class F1 is uneven and this README will not hide it: Bacteria 0.96 and Virus
0.82 are strong; Nematode is 0.45, because it has only 68 images in the entire
dataset. Below the calibrated confidence floor the model abstains instead of
guessing, and a Mahalanobis gate rejects images that are not potato leaves.

**The treatment recommender is effectively a lookup table.** The fine-tuned
Flan-T5 checkpoint was trained on 7 examples for 100 optimizer steps and ended
at loss 3.24, so it never learned them. In practice a curated 7-entry dictionary
supplies the text. Treatment advice names real agrochemicals but carries **no
dose, no pre-harvest interval, no PPE guidance and no jurisdiction check** —
treat it as illustrative, never as agronomic instruction.

**The RAG chat is not wired into the UI.** `/api/ai/agribot` calls Groq directly
with no retrieval. The Pinecone/Cohere pipeline exists only in the notebooks and
in `flask_server.py`.

**Messaging is not real-time.** There is no WebSocket and no SSE.

**No automated tests, no CI, no Dockerfile.** The `test_*.py` files are
print-based scripts, not a test suite.

**Groq model ids expire.** `GROQ_MODEL` is configuration for a reason:
`llama3-70b-8192` was decommissioned 2025-08-30 and its own replacement
`llama-3.3-70b-versatile` was deprecated 2026-06-17. If chat starts returning
400s, update `GROQ_MODEL` in `.env` — not the source.

---

## Security notes

- Rotate the credentials in `.env` if this repo was ever public. Earlier commits
  contained a live Pinecone key, three Pinata JWTs and a Supabase key. Removing
  them from the working tree does not remove them from git history.
- Never set `FLASK_DEBUG=1` on a shared or public host. It enables the Werkzeug
  debugger, which is a remote code execution surface.
- The frontend stores its JWT in `localStorage`, which is readable by any script
  on the page. Markdown rendered into the DOM is sanitised with DOMPurify and all
  CDN scripts are version-pinned with SRI hashes, but an httpOnly cookie would
  be stronger.

---

---

## Testing

```bash
python tools/guards.py                 # 6 regression guards for the audited defect classes
cd backend && npm run dev:mem          # boot on in-memory MongoDB, auto-seeded
python eval_rag.py --make-golden       # scaffold the RAG golden set
python eval_rag.py --golden golden.json --ci --max-dose-leaks 0
```

CI runs three jobs on every push: backend (parse → **boot** → smoke → `npm audit`),
python (compile → guards → safety invariants), and an ML label-leak gate.

The backend job boots the server rather than only parsing it, because
`node --check` passes on a file whose `require()` target is missing — which is
exactly how a model deletion once broke `seed.js` and silently disabled login.

## Training the classifier

```bash
python ml-training/train_potato.py     --data-root "<dataset>/Potato Leaf Disease Dataset in Uncontrolled Environment"     --backbone dinov2 --folds 5 --tta 4 --out artifacts
cp -r artifacts 
```

Two things to expect, both intended:

1. **The honest macro-F1 will be lower than the old 84.10%.** That figure was
   measured with the ground-truth label inside the model input. Published results
   on this dataset for reference: EfficientNetV2B3 73.63%, MobileNetV3-L 72.03%,
   best 87.82%. Treat anything above ~88% as leakage until proven otherwise.
2. **The image count must come out at 3,076.** If `collect_images()` warns, fix
   the path before training — do not train on a truncated dataset.

## Companion repo

The models and notebooks live separately in **Pestivid** — CLIP fine-tuning, the
Flan-T5 recommender, and the RAG pipeline. Train the weights there, then place
the two `.pth` files in this project root.

## License

Proprietary. All rights reserved.
