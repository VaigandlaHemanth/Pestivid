# Pestivid — Potato Leaf Disease Detection & Pesticide Recommendation

An end-to-end deep learning system that **identifies potato leaf diseases from images** and **recommends targeted pesticide treatments** using Vision-Language Models (VLMs) and a Retrieval-Augmented Generation (RAG) chatbot.

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Running the Notebooks](#running-the-notebooks)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Pestivid** addresses a critical agricultural challenge: rapid, accurate identification of potato leaf diseases and actionable treatment advice. The project consists of two tightly integrated components:

1. **Vision-Language Model (VLM) Pipeline** — Fine-tunes OpenAI's CLIP (`clip-vit-base-patch32`) and Salesforce's InstructBLIP (`instructblip-flan-t5-xl`) on 1,885 potato leaf images across 7 disease categories. A companion Text-to-Text (T2T) model (`google/flan-t5-small`) is trained to generate detailed pesticide recommendations from the predicted disease class.

2. **RAG Chatbot** — Ingests a plant disease research PDF, embeds it with Cohere `embed-english-v3.0` (1024-dimensional), stores vectors in Pinecone, and answers natural language queries using Groq-hosted `llama3-70b-8192` orchestrated by LangGraph.

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Input                           │
│          (Leaf Image  or  Text Question)                │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             ▼                          ▼
┌────────────────────────┐   ┌────────────────────────────┐
│  VLM Classification    │   │   RAG Chatbot Pipeline     │
│  (potatoleaf-vlm)      │   │   (nowwor.ipynb)           │
│                        │   │                            │
│  CLIP ViT-B/32         │   │  PDF → PyMuPDF → Chunks   │
│      ↓                 │   │      ↓                    │
│  Disease Prediction    │   │  Cohere embed-english-v3.0 │
│      ↓                 │   │      ↓                    │
│  Flan-T5-Small (T2T)   │   │  Pinecone index "hi"      │
│      ↓                 │   │      ↓                    │
│  Pesticide             │   │  LangGraph (Retrieve →    │
│  Recommendation        │   │    Generate)              │
│                        │   │      ↓                    │
│                        │   │  Groq llama3-70b-8192     │
└────────────────────────┘   └────────────────────────────┘
```

---

## Notebooks

| Notebook | Purpose | Platform |
|---|---|---|
| `potatoleaf-vlm-fc93c1.ipynb` | CLIP fine-tuning, InstructBLIP LoRA fine-tuning, Flan-T5 pesticide recommendation training, combined inference | Kaggle (GPU P100) |
| `nowwor.ipynb` | Full RAG chatbot — PDF ingestion, embedding, Pinecone upsert, LangGraph workflow, Q&A | Local / Any Python env |
| `tes.ipynb` | Lightweight chatbot test harness — connects to existing Pinecone index and runs queries without re-ingesting the PDF | Local / Any Python env |

---

## Dataset

**"Potato Leaf Disease Dataset in Uncontrolled Environment"** from Kaggle.

| Class | Images |
|---|---|
| Bacteria | 342 |
| Fungi | 452 |
| Healthy | 175 |
| Nematode | 47 |
| Pest | 415 |
| Phytophthora | 151 |
| Virus | 303 |
| **Total** | **1,885** |

Images are RGB photographs of potato leaves captured in real-world (uncontrolled) field environments.

> **Note:** The dataset is not included in this repository due to its size. Download it from [Kaggle](https://www.kaggle.com/datasets) and place it under the path expected by the VLM notebook.

---

## Technology Stack

| Component | Technology |
|---|---|
| Image Classification | CLIP `clip-vit-base-patch32` (fine-tuned), InstructBLIP `instructblip-flan-t5-xl` (LoRA) |
| Pesticide Recommendation | Flan-T5-Small (fine-tuned T2T) |
| Embedding | Cohere `embed-english-v3.0` (1024-D) |
| Vector Store | Pinecone (index `"hi"`, 1024 dimensions) |
| LLM (Chatbot) | Groq `llama3-70b-8192` (temperature 0.1) |
| RAG Orchestration | LangGraph `StateGraph` |
| PDF Parsing | PyMuPDF (`fitz`) |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Deep Learning Framework | PyTorch, HuggingFace Transformers |
| Quantization | BitsAndBytes 4-bit NF4 |
| Parameter-Efficient Fine-Tuning | PEFT / LoRA |

---

## Getting Started

### Prerequisites

```
Python >= 3.9
PyTorch >= 2.0 (with CUDA for VLM training)
```

Install dependencies:

```bash
# For the VLM notebook (run on Kaggle or a GPU machine)
pip install torch torchvision torchinfo transformers peft bitsandbytes accelerate \
    scikit-learn matplotlib seaborn opencv-python pillow nltk tqdm

# For the RAG chatbot notebooks
pip install langchain langchain-cohere langchain-pinecone langchain-groq \
    langgraph pinecone-client python-dotenv PyMuPDF cohere
```

### Environment Variables

Create a `.env` file (used by the chatbot notebooks):

```env
COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### Running the Notebooks

1. **VLM Training** — Open `potatoleaf-vlm-fc93c1.ipynb` on Kaggle with GPU acceleration enabled. Attach the Potato Leaf Disease dataset and run all cells.

2. **RAG Chatbot (Full Pipeline)** — Open `nowwor.ipynb`, set your API keys in `.env`, place `leaf_train.pdf` at the expected path, and run all cells. This will ingest the PDF, embed chunks, upsert to Pinecone, and start the Q&A system.

3. **RAG Chatbot (Query Only)** — Open `tes.ipynb` to query the already-populated Pinecone index without re-ingesting the PDF.

---

## Results

> **The figures below are not valid and are kept only for provenance.**
> See [Why the old numbers are withdrawn](#why-the-old-numbers-are-withdrawn).

### CLIP Fine-Tuning (25 epochs) — WITHDRAWN

| Metric | Reported | Status |
|---|---|---|
| Test Accuracy | 84.10% | **invalid** — measured with the ground-truth label present in the model input |
| Best Validation Loss | 0.7833 (epoch 25) | still improving at the final epoch, i.e. undertrained |
| Best Validation Accuracy | 81.69% | same leak; also used for checkpoint selection |
| Trainable Parameters | 14,441,991 | accurate |

Per-class figures are withdrawn for the same reason. Note also that the Nematode
row rested on **7** test images, where a single error moves recall ~14 points.

### Why the old numbers are withdrawn

`CLIPFineTuner.forward()` computed:

```python
combined = image_features * 0.7 + text_features * 0.3
```

and the dataset selected the text with `text = self.text_prompts[label]` — the
ground-truth label. Because `unfreeze_layers=2` unfroze only the last two
**vision** layers, the text tower stayed frozen, and there were exactly 7 prompt
strings. So `text_features` was a fixed lookup table of 7 constant vectors: a
class-indexed additive constant with zero within-class variance. Formally
`H(y | combined) = 0` — the input contained the answer.

The same leak was present in the validation and test sets, so checkpoint
selection ran through it too. **84.10% is an upper bound of unknown tightness on
image-only accuracy, and nothing in this repo bounds it from below.**

The coefficient is a red herring: the first `nn.Linear` can rescale that
direction freely, so even `text_features * 0.001` would leak completely.

Inference made it worse. `get_clip_disease_prediction` looped over all 7
candidate prompts and read the 7×7 diagonal, then softmaxed 7 scalars gathered
from 7 **separate** forward passes — a computation the model was never trained to
produce, and a "confidence" that is not a probability of anything.

### Current pipeline

Use **`train_potato.py`**, which has no text branch. For reference, published
results on this exact 7-class dataset:

| Model | Accuracy |
|---|---|
| EfficientNet-LITE + Kernel-Ensemble SVM (Front. Plant Sci. 2025) | 87.82% |
| EfficientNetV2B3 (dataset paper baseline) | 73.63% |
| MobileNetV3-Large | 72.03% |
| ResNet50 | 68.17% |
| VGG-16 | 59.81% |

Treat anything above ~88% on this dataset as leakage until proven otherwise.

```bash
python train_potato.py --data-root <dataset> --backbone dinov2 --folds 5
```

It reports **macro-F1 with 5-fold StratifiedGroupKFold** (near-duplicate
photographs of the same plant grouped so they cannot straddle a split), plus a
calibration curve and an out-of-distribution gate, and writes a model card.
Expect the honest number to be **lower** than 84.10% — that is the point.

### RAG Chatbot

The chatbot successfully answers domain-specific questions about potato plant diseases with grounded, citation-backed responses from the ingested PDF research document.

---

## Repository Structure

```
├── potatoleaf-vlm-fc93c1.ipynb   # VLM training & pesticide recommendation
├── nowwor.ipynb                  # Full RAG chatbot pipeline
├── tes.ipynb                     # RAG chatbot query-only test harness
├── README.md                     # This file
└── TECHNICAL_DOCUMENTATION.md    # Exhaustive technical documentation
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is for educational and research purposes.
