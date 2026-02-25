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

### CLIP Fine-Tuning (25 Epochs)

| Metric | Value |
|---|---|
| Test Accuracy | **84.10%** |
| Best Validation Loss | 0.7833 (epoch 25) |
| Best Validation Accuracy | 81.69% |
| Trainable Parameters | 14,441,991 (9.53% of 151M total) |

**Per-Class Performance (Test Set):**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bacteria | 1.00 | 0.98 | 0.99 | 51 |
| Fungi | 0.78 | 0.82 | 0.80 | 68 |
| Healthy | 0.82 | 0.88 | 0.85 | 26 |
| Nematode | 1.00 | 0.86 | 0.92 | 7 |
| Pest | 0.71 | 0.81 | 0.76 | 63 |
| Phytophthora | 1.00 | 0.64 | 0.78 | 22 |
| Virus | 0.93 | 0.83 | 0.87 | 46 |

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
