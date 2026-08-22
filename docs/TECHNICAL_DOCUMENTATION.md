# Pestivid — Technical Documentation

> Exhaustive engineering reference covering every component, algorithm, hyperparameter, and design decision in the Pestivid project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [End-to-End Project Flow](#2-end-to-end-project-flow)
3. [Component 1 — Vision-Language Model Pipeline](#3-component-1--vision-language-model-pipeline)
   - [3.1 Dataset](#31-dataset)
   - [3.2 Image Preprocessing](#32-image-preprocessing)
   - [3.3 Text Preprocessing](#33-text-preprocessing)
   - [3.4 Custom Dataset & DataLoader](#34-custom-dataset--dataloader)
   - [3.5 CLIP Fine-Tuning Architecture](#35-clip-fine-tuning-architecture)
   - [3.6 Training Configuration](#36-training-configuration)
   - [3.7 Training Results](#37-training-results)
   - [3.8 Evaluation Metrics](#38-evaluation-metrics)
   - [3.9 Pesticide Recommendation (T2T Model)](#39-pesticide-recommendation-t2t-model)
   - [3.10 Combined Inference Pipeline](#310-combined-inference-pipeline)
   - [3.11 InstructBLIP with LoRA (Experimental)](#311-instructblip-with-lora-experimental)
4. [Component 2 — RAG Chatbot](#4-component-2--rag-chatbot)
   - [4.1 PDF Ingestion Pipeline](#41-pdf-ingestion-pipeline)
   - [4.2 Text Chunking Algorithm](#42-text-chunking-algorithm)
   - [4.3 Embedding Model](#43-embedding-model)
   - [4.4 Vector Store (Pinecone)](#44-vector-store-pinecone)
   - [4.5 LangGraph RAG Workflow](#45-langgraph-rag-workflow)
   - [4.6 LLM Configuration](#46-llm-configuration)
   - [4.7 Prompt Engineering](#47-prompt-engineering)
   - [4.8 Query Functions](#48-query-functions)
5. [Component 3 — Chatbot Test Harness](#5-component-3--chatbot-test-harness)
6. [Design Decisions & Rationale](#6-design-decisions--rationale)
7. [Known Limitations & Future Work](#7-known-limitations--future-work)
8. [File Inventory](#8-file-inventory)

---

## 1. Project Overview

Pestivid is an agricultural AI system with two complementary capabilities:

| Capability | Input | Output |
|---|---|---|
| **Disease Classification + Pesticide Recommendation** | Potato leaf image | Disease class (1 of 7) + detailed treatment plan |
| **Domain Expert Chatbot** | Natural language question | Research-grounded answer about plant diseases |

The project is implemented across three Jupyter notebooks, all designed to run on Kaggle (VLM) or any Python environment (chatbot).

---

## 2. End-to-End Project Flow

```
                              ┌──────────────────────┐
                              │   Potato Leaf Image   │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │     CLIP ViT-B/32     │
                              │  (Fine-Tuned on 1885  │
                              │   images, 7 classes)  │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  CLIPFineTuner Model  │
                              │  image_features*0.7 + │
                              │  text_features*0.3    │
                              │       ↓               │
                              │  Linear(512,512)→ReLU │
                              │  →Dropout(0.1)        │
                              │  →Linear(512,7)       │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Predicted Disease    │
                              │  e.g. "Fungi"         │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Flan-T5-Small (T2T)  │
                              │  Input: "Recommend    │
                              │  treatment for potato │
                              │  disease: Fungi"      │
                              │       ↓               │
                              │  Generated treatment  │
                              │  recommendation       │
                              └──────────────────────┘


    ┌───────────────────────────────────────────────────────┐
    │              RAG Chatbot (Parallel Path)               │
    │                                                       │
    │  User Question                                        │
    │       ↓                                               │
    │  Cohere embed-english-v3.0 → query vector (1024-D)    │
    │       ↓                                               │
    │  Pinecone similarity_search(k=3)                      │
    │       ↓                                               │
    │  Top-3 document chunks                                │
    │       ↓                                               │
    │  LangGraph: Retrieve → Generate                       │
    │       ↓                                               │
    │  Groq llama3-70b-8192 (temperature=0.1)               │
    │       ↓                                               │
    │  Grounded Expert Answer                               │
    └───────────────────────────────────────────────────────┘
```

---

## 3. Component 1 — Vision-Language Model Pipeline

**Notebook:** `potatoleaf-vlm-fc93c1.ipynb`  
**Runtime:** Kaggle with GPU (NVIDIA Tesla P100, 16 GB VRAM)

### 3.1 Dataset

| Property | Value |
|---|---|
| Name | Potato Leaf Disease Dataset in Uncontrolled Environment |
| Source | Kaggle |
| Total Images | 1,885 |
| Image Format | JPEG, RGB |
| Classes | 7 |
| Class Distribution | Bacteria: 342, Fungi: 452, Healthy: 175, Nematode: 47, Pest: 415, Phytophthora: 151, Virus: 303 |
| Split Ratio | 70% train / 15% validation / 15% test |
| Split Method | Stratified random split (`sklearn.model_selection.train_test_split`) |
| Random Seed | 123 |

**Class imbalance note:** The dataset is moderately imbalanced. Nematode (47) has ~10× fewer samples than Fungi (452). No oversampling or class-weighted loss was applied — this is a known limitation.

### 3.2 Image Preprocessing

**Training transforms** (with data augmentation):

```python
transforms.Compose([
    transforms.Resize((224, 224)),        # Resize to CLIP's expected input
    transforms.RandomHorizontalFlip(),    # 50% probability horizontal flip
    transforms.RandomRotation(10),        # ±10° random rotation
    transforms.ToTensor(),                # [0, 255] → [0.0, 1.0] float tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet channel means
        std=[0.229, 0.224, 0.225]         # ImageNet channel stds
    ),
])
```

**Validation/Test transforms** (no augmentation):

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Why these transforms:**
- **224×224:** CLIP ViT-B/32 expects 224×224 input. The vision transformer divides the image into 7×7 patches of 32×32 pixels each.
- **ImageNet normalization:** CLIP was pretrained on ImageNet-scale data with these statistics. Matching them ensures the pretrained feature extractors receive inputs in their expected distribution.
- **Horizontal flip + rotation:** Simple geometric augmentations that create viewpoint invariance without distorting disease symptoms. More aggressive augmentations (e.g., color jitter) were avoided because color is a key diagnostic signal for plant diseases.

### 3.3 Text Preprocessing

A general text preprocessing pipeline is defined (though primarily used for auxiliary analysis, not for CLIP's own tokenizer):

```python
def preprocess_text(text):
    text = text.lower()                                    # Lowercase
    text = re.sub(r'[^a-z0-9\s]', '', text)               # Remove punctuation
    tokens = word_tokenize(text)                           # NLTK word tokenizer
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_tokens)
```

CLIP's own text processing uses `CLIPProcessor` (a CLIP-specific BPE tokenizer with max sequence length of 77 tokens).

### 3.4 Custom Dataset & DataLoader

**`CLIPPotatoDataset`** — Returns a dictionary per sample:

| Key | Shape | Description |
|---|---|---|
| `pixel_values` | `[3, 224, 224]` | Preprocessed image tensor |
| `input_ids` | `[77]` | CLIP-tokenized text prompt (padded to max length) |
| `attention_mask` | `[77]` | Attention mask for text tokens |
| `labels` | scalar | Integer class index (0–6) |

**Text prompts per class:**

| Class | Prompt |
|---|---|
| Bacteria | "a potato leaf infected with bacterial disease" |
| Fungi | "a potato leaf infected with fungal disease" |
| Healthy | "a healthy potato leaf with no disease" |
| Nematode | "a potato leaf infected with nematode disease" |
| Pest | "a potato leaf damaged by pests" |
| Phytophthora | "a potato leaf infected with phytopthora disease" |
| Virus | "a potato leaf infected with viral disease" |

**Why natural language prompts:** CLIP was trained using contrastive learning between images and free-form text. Providing descriptive disease text lets the model leverage its pretrained multimodal alignment — the text branch encodes semantic disease information that guides the image branch.

**DataLoader configuration:**

| Parameter | Value |
|---|---|
| `batch_size` | 32 |
| `shuffle` | True (train), False (val/test) |
| `num_workers` | `os.cpu_count()` |
| `collate_fn` | Custom function stacking `pixel_values`, `input_ids`, `attention_mask`, `labels` |

### 3.5 CLIP Fine-Tuning Architecture

**Base model:** `openai/clip-vit-base-patch32`

| Property | Value |
|---|---|
| Vision encoder | ViT-B/32 (12 transformer layers, patch size 32×32, hidden dim 768) |
| Text encoder | 12-layer transformer, max seq length 77 |
| Projection dimension | 512 |
| Total parameters | ~151M |

**`CLIPFineTuner` architecture:**

```
CLIP ViT-B/32 (frozen except last 2 vision transformer layers)
    ↓
vision_model → pooled output → visual_projection → image_features (512-D)
text_model   → pooled output → text_projection   → text_features  (512-D)
    ↓
L2-normalize both feature vectors
    ↓
combined = image_features * 0.7 + text_features * 0.3    ← weighted fusion
    ↓
Linear(512, 512) → ReLU → Dropout(0.1) → Linear(512, 7)  ← classification head
    ↓
logits (7-D)
```

**Selective unfreezing strategy:**

- All CLIP parameters are frozen by default
- The last `unfreeze_layers=2` vision transformer layers are unfrozen
- The classification head is always trainable
- **Trainable parameters:** 14,441,991 (9.53% of total)

**Why this architecture:**
- **Frozen base:** Preserves CLIP's rich pretrained representations. With only 1,885 images, full fine-tuning would risk catastrophic forgetting.
- **Unfreeze last 2 layers:** Allows the model to adapt its high-level visual features to the specific domain (diseased potato leaves) while keeping lower-level features (edges, textures) intact.
- **70/30 image-text fusion:** Images are the primary diagnostic signal; text provides supplementary semantic context. The 0.7/0.3 weighting reflects this priority.
- **Dropout(0.1):** Mild regularization given the small dataset to prevent overfitting of the classification head.

### 3.6 Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Weight decay variant of Adam; prevents large weight accumulation in fine-tuned layers. Standard for transformer fine-tuning. |
| Learning Rate | 2e-5 | Conservative LR for fine-tuning pretrained transformers. Larger LRs (e.g., 1e-3) would destroy pretrained weights. |
| Weight Decay | AdamW default (0.01) | Implicit L2 regularization through AdamW's decoupled weight decay. |
| Loss Function | `nn.CrossEntropyLoss()` | Standard multi-class classification loss. Combines LogSoftmax + NLLLoss. Suitable for mutually exclusive classes. |
| Epochs | 25 | Empirically chosen; validation loss continues decreasing through epoch 25. |
| Batch Size | 32 | Fits in P100 16 GB VRAM with CLIP-B/32 + gradients. |
| Random Seed | 123 | Applied to `torch.manual_seed`, `random.seed`, `np.random.seed`, and `torch.cuda.manual_seed`. |
| Gradient Clipping | None | Not applied; training was stable without it. |
| Learning Rate Scheduler | None | Constant LR throughout training. |

**Checkpoint strategy:** Best model saved whenever validation loss improves. Checkpoint includes: model state dict, optimizer state dict, loss, epoch, and validation accuracy.

### 3.7 Training Results

**Epoch-by-epoch progression (selected):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 1.9092 | 28.89% | 1.8767 | 55.11% |
| 5 | 1.7038 | 81.98% | 1.6982 | 78.91% |
| 10 | 1.3580 | 98.81% | 1.4219 | 81.82% |
| 15 | 0.9962 | 99.63% | 1.1374 | 81.91% |
| 20 | 0.7009 | 99.70% | 0.9182 | 82.03% |
| 25 | 0.4883 | 99.70% | 0.7833 | 81.69% |

**Observations:**
- Training accuracy converges rapidly (~99% by epoch 8), indicating the model memorizes the training set.
- Validation accuracy plateaus around 80–83%, suggesting some overfitting but still strong generalization.
- Validation loss continues to decrease monotonically, suggesting the model's confidence is still improving even as accuracy plateaus.

### 3.8 Evaluation Metrics

**Test set accuracy: 84.10%** (283 test images)

**Full classification report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bacteria | 1.00 | 0.98 | 0.99 | 51 |
| Fungi | 0.78 | 0.82 | 0.80 | 68 |
| Healthy | 0.82 | 0.88 | 0.85 | 26 |
| Nematode | 1.00 | 0.86 | 0.92 | 7 |
| Pest | 0.71 | 0.81 | 0.76 | 63 |
| Phytophthora | 1.00 | 0.64 | 0.78 | 22 |
| Virus | 0.93 | 0.83 | 0.87 | 46 |
| **Weighted Avg** | **0.85** | **0.84** | **0.84** | **283** |
| **Macro Avg** | **0.89** | **0.83** | **0.85** | **283** |

**Analysis:**
- **Bacteria** and **Nematode** achieve perfect precision — the model never makes a false positive for these classes.
- **Phytophthora** has the lowest recall (0.64) — the model misses 36% of Phytophthora samples, likely confused with Fungi (both are fungal-like/oomycete diseases with visually similar lesion patterns).
- **Pest** has the lowest precision (0.71) — some non-pest images are misclassified as pest damage.
- The high macro-average precision (0.89) indicates the model is generally confident and correct when it predicts a class.

### 3.9 Pesticide Recommendation (T2T Model)

A **Text-to-Text model** is fine-tuned to generate structured treatment recommendations.

**Model:** `google/flan-t5-small` (~80M parameters, instruction-tuned)

**Training data:** 7 hand-crafted input-output pairs, one per disease class:

| Input | Output |
|---|---|
| `"Recommend treatment for potato disease: Bacteria"` | Detailed paragraph covering disease info, pesticides, application instructions, and cultural practices |
| `"Recommend treatment for potato disease: Fungi"` | ... |
| ... (one per class) | ... |

**T2T Training Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for HuggingFace T5 fine-tuning |
| Learning Rate | 5e-5 | Slightly higher than CLIP LR because T5 is fully fine-tuned on a memorization task |
| Epochs | 50 | High epoch count intentional — with only 7 samples, the model needs many passes to memorize the outputs |
| Batch Size | 4 | Small batch because dataset is tiny (7 samples, 2 batches/epoch) |
| LR Scheduler | Linear decay with 0 warmup steps | Gradual LR reduction over the 50 epochs |
| Max Input Length | 128 tokens | Sufficient for the short input prompts |
| Max Target Length | 512 tokens | Accommodates the full treatment recommendation text |
| Label Padding | pad_token_id replaced with -100 | Standard HuggingFace practice: -100 indices are ignored by CrossEntropyLoss |

**T2T Training Results:**

| Epoch | Average Loss |
|---|---|
| 10 | 4.0325 |
| 20 | 3.6229 |
| 30 | 3.4167 |
| 40 | 3.3092 |
| 50 | 3.2368 |

**Pesticide Recommendation Database (per class):**

Each class has a structured entry containing:
- **Disease Information** — Pathogen identification and key symptoms
- **Recommended Pesticides** — 3-5 specific products (e.g., "Copper Hydroxide", "Mancozeb-based fungicides")
- **Application Instructions** — Timing, frequency, coverage
- **Cultural Practices** — 4-5 preventive measures (crop rotation, resistant varieties, sanitation)

Example for **Bacteria:**
- Pesticides: Copper-based bactericides, Streptomycin sulfate, Kasugamycin
- Application: Foliar spray every 7-10 days during disease pressure
- Cultural: Certified seed potatoes, 3-4 year crop rotation, improved drainage

**T2T Inference:** Uses beam search (`num_beams=4`, `early_stopping=True`) at inference time for higher-quality generation.

### 3.10 Combined Inference Pipeline

The end-to-end inference combines both models:

1. **Input:** Path to a potato leaf image
2. **CLIP prediction:** `get_clip_disease_prediction()` runs the image through all 7 class text prompts, collects logits per class, applies softmax to get probabilities, and selects the argmax class
3. **T2T generation:** `get_t2t_recommendation()` takes the predicted disease name, constructs the input prompt, and generates the treatment recommendation
4. **Visualization:** `display_combined_prediction_and_recommendations()` creates a matplotlib figure showing the image, class probability bar chart, and treatment text

### 3.11 InstructBLIP with LoRA (Experimental)

The notebook also contains an **experimental** second approach using Salesforce's InstructBLIP-Flan-T5-XL, a much larger Vision-Language Model (~4B parameters).

**Model:** `Salesforce/instructblip-flan-t5-xl`

**LoRA configuration:**

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | "none" |
| Target modules | `["q", "k"]` (query and key projections in attention layers) |
| Trainable parameters | 9,437,184 (0.23% of 4B total) |

**Quantization:** BitsAndBytes 4-bit NF4 quantization with `bfloat16` compute dtype to fit in GPU memory.

**Prompt for InstructBLIP:**
```
Examine the image of a potato leaf closely.
Based on visual evidence, what is the primary disease or condition affecting it?
Choose your answer from the following list only: Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus.
Respond with only the single, most likely class name.
```

**Status:** This approach encountered **CUDA Out-of-Memory errors** on the Kaggle P100 (16 GB VRAM). The InstructBLIP model at 4-bit quantization still exceeds the available VRAM after CLIP training fills the GPU. The LoRA adapters were also not saved to disk before the OOM occurred, preventing inference. This remains an incomplete experiment.

---

## 4. Component 2 — RAG Chatbot

**Notebook:** `nowwor.ipynb`  
**Runtime:** Any Python environment with API key access (no GPU required)

### 4.1 PDF Ingestion Pipeline

**Source document:** `leaf_train.pdf` (a plant disease research document)

**Extraction library:** PyMuPDF (`fitz`)

```python
doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc[page_num]
    page_text = page.get_text()
    full_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
doc.close()
```

**Why PyMuPDF:**
- Fast C-based PDF parsing (10-100× faster than pure Python alternatives)
- Handles complex PDF layouts, multi-column text, and embedded fonts
- Returns clean Unicode text without needing OCR for text-based PDFs

### 4.2 Text Chunking Algorithm

**Algorithm:** `RecursiveCharacterTextSplitter` from LangChain

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1000 characters | Balances information density per chunk with embedding model context window. Too small → fragments sentences and loses context. Too large → dilutes relevant information with noise, reduces retrieval precision. 1000 chars ≈ 150-200 words ≈ a full paragraph, which typically captures one complete idea. |
| `chunk_overlap` | 200 characters | 20% overlap ensures no information is lost at chunk boundaries. If a key sentence spans two chunks, the overlap guarantees it appears fully in at least one chunk. |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Hierarchical splitting: prefers paragraph breaks (`\n\n`), then line breaks (`\n`), then sentence boundaries (`. `), then word boundaries (` `), and finally character-level splitting as a last resort. This preserves semantic coherence — the splitter always tries the most natural break point first. |
| `length_function` | `len` (character count) | Simple character counting for chunk measurement. Token-based counting would be more precise for the embedding model but adds complexity. |

**Why `RecursiveCharacterTextSplitter`:**
- It recursively tries each separator in order, producing the most semantically coherent chunks possible
- Unlike `CharacterTextSplitter` (which uses only one separator), it adapts to document structure
- Unlike fixed-window splitting, it respects natural text boundaries

### 4.3 Embedding Model

**Model:** Cohere `embed-english-v3.0`

| Property | Value |
|---|---|
| Dimensions | 1024 |
| Max input tokens | 512 |
| Training approach | Contrastive learning on large-scale web data |
| Language | English |
| Compression | Not applied (full 1024-D vectors stored) |

**Why Cohere `embed-english-v3.0`:**
- State-of-the-art retrieval performance on MTEB benchmarks
- Optimized for search/retrieval use cases (asymmetric semantic similarity)
- 1024 dimensions provides rich semantic representation with acceptable storage cost
- Cloud API — no local GPU required for embedding

**Embedding workflow:**
1. Each text chunk is embedded individually via `embeddings.embed_documents([chunk])[0]`
2. A 0.2-second sleep between API calls prevents rate limiting
3. Each embedding is paired with rich metadata (document name, chunk_id, total_chunks, full text, content_type, source)

### 4.4 Vector Store (Pinecone)

| Property | Value |
|---|---|
| Index name | `"hi"` |
| Dimensions | 1024 |
| Total vectors (after ingestion) | 2,682 |
| Metric | Cosine similarity (default) |
| Cloud | Pinecone serverless |

**Vector metadata schema:**

```json
{
  "document": "leaf_train.pdf",
  "file_path": "potato_leaf_disease - Copy/train/leaf_train.pdf",
  "file_type": "pdf",
  "chunk_id": 0,
  "total_chunks": N,
  "text": "full chunk text here",
  "content_type": "plant_disease_research",
  "source": "leaf_train_pdf"
}
```

The `"text"` key in metadata serves a dual purpose:
1. Returned with search results so the LLM can read the chunk content
2. Prevents LangChain's "no text key in metadata" warnings

**Why Pinecone:**
- Managed vector database — zero operational overhead
- Sub-100ms query latency at scale
- Supports metadata filtering (useful for multi-document scenarios)
- Free tier sufficient for project scale

### 4.5 LangGraph RAG Workflow

The RAG system is implemented as a **LangGraph `StateGraph`** with two nodes:

```
Entry Point → [Retrieve] → [Generate] → END
```

**State definition:**

```python
class RAGState(TypedDict):
    question: str           # User's input question
    documents: List[Document]  # Retrieved document chunks
    answer: str             # Generated answer
```

**Node 1 — `retrieve_documents`:**
- Input: `RAGState` with `question` populated
- Action: `vector_store.similarity_search(question, k=3)`
- Output: Populates `documents` with top-3 most similar chunks
- **k=3 rationale:** Provides sufficient context without overwhelming the LLM's context window. With 1000-char chunks, 3 chunks ≈ 3000 characters ≈ 750 tokens of context.

**Node 2 — `generate_answer`:**
- Input: `RAGState` with `question` and `documents` populated
- Action: Constructs a prompt from context + question, invokes Groq LLM
- Output: Populates `answer` with the LLM's response

**Why LangGraph:**
- Explicit state management — each node transforms a well-defined state object
- Easily extensible — additional nodes (e.g., re-ranking, fact-checking) can be inserted into the graph
- Visual debugging — the graph structure can be rendered for inspection
- Production-ready — integrates with LangSmith for tracing and monitoring

### 4.6 LLM Configuration

**Model:** Groq-hosted `llama3-70b-8192`

| Property | Value |
|---|---|
| Provider | Groq (cloud inference) |
| Model | Meta LLaMA 3 70B |
| Context window | 8,192 tokens |
| Temperature | 0.1 |
| Interface | `langchain_groq.ChatGroq` |

**Why this model:**
- **LLaMA 3 70B:** Best open-source model at the time of development. Strong reasoning and factual accuracy.
- **Groq hosting:** Extremely fast inference (~500 tokens/second via Groq's LPU hardware), free tier API access.
- **Temperature 0.1:** Near-deterministic output. For scientific/agricultural advice, factual accuracy is paramount — creative/diverse outputs are undesirable.
- **8,192 token context:** Sufficient for the system prompt (~100 tokens) + 3 retrieved chunks (~750 tokens) + question (~30 tokens) + generated answer (~500 tokens).

### 4.7 Prompt Engineering

The system prompt positions the LLM as a domain expert:

```
You are an expert plant pathologist with deep knowledge of plant diseases,
their symptoms, causes, and management strategies.

Based on the following research context about plant diseases, provide a
comprehensive and accurate answer to the question.

Research Context:
{context}

Question: {question}

Instructions:
- Provide detailed, scientific information
- Include specific symptoms, pathogens, and management strategies when relevant
- If the information is not in the context, clearly state that
- Use proper scientific terminology

Answer:
```

**Design decisions:**
- **Role assignment** ("expert plant pathologist") activates the LLM's domain knowledge and primes appropriate vocabulary
- **Grounding instruction** ("based on the following research context") prevents hallucination by anchoring answers to retrieved documents
- **Explicit out-of-scope handling** ("if the information is not in the context, clearly state that") prevents the model from fabricating answers when the knowledge base lacks relevant information
- **Scientific register** ("proper scientific terminology") ensures outputs match the domain's expected communication style

### 4.8 Query Functions

Three query interfaces are provided:

| Function | Description | Shows Sources |
|---|---|---|
| `ask_plant_expert(question)` | Standard Q&A with clean output | No |
| `query_plant_diseases(question)` | Alias for `ask_plant_expert` | No |
| `detailed_plant_query(question, show_sources=True)` | Advanced query with source document preview | Yes (first 100 chars of each source) |

All functions invoke the same LangGraph workflow and return the answer string.

---

## 5. Component 3 — Chatbot Test Harness

**Notebook:** `tes.ipynb`

A lightweight notebook that **skips PDF ingestion** and connects directly to the existing Pinecone index `"hi"`. It reuses the same:
- Cohere `embed-english-v3.0` embeddings
- Pinecone connection
- Groq `llama3-70b-8192` LLM
- LangGraph `StateGraph` with Retrieve → Generate nodes

**Differences from `nowwor.ipynb`:**
- No PDF processing or vector upload code
- State class renamed to `PlantRAGState` (functionally identical)
- Prompt slightly modified: "You are a world-class plant pathologist with expertise in disease diagnosis and management"
- Includes an index stats check on startup to verify the knowledge base is populated
- Additional `detailed_query()` function shows sources by default

**Purpose:** Rapid iteration and testing of the Q&A system without the time/cost of re-ingesting the PDF and re-embedding all chunks.

---

## 6. Design Decisions & Rationale

| Decision | Alternatives Considered | Why This Choice |
|---|---|---|
| CLIP for classification | ResNet-50, EfficientNet, ViT standalone | CLIP's multimodal pretraining provides zero-shot transfer ability and text-guided classification. Disease class prompts give the model semantic context beyond pixel patterns. With only 1,885 images, leveraging pretrained multimodal representations outperforms training a vision-only model from scratch. |
| 70/30 image-text fusion | Equal weighting, learned attention, concatenation | The image is the primary evidence for disease; text supplements it. A fixed 70/30 split avoids the overhead of learning fusion weights on a small dataset. |
| AdamW optimizer | SGD with momentum, Adam, LAMB | AdamW is the standard choice for transformer fine-tuning. Its decoupled weight decay provides regularization without interfering with the adaptive learning rate. |
| Learning rate 2e-5 | 1e-4, 1e-3, 5e-5 | Standard fine-tuning LR for CLIP/ViT models. Higher LRs destabilize pretrained weights; lower LRs converge too slowly. |
| Cohere embeddings | OpenAI `ada-002`, Sentence-BERT, Vertex AI | Cohere `embed-english-v3.0` is specifically optimized for retrieval tasks (vs. OpenAI's general-purpose embeddings). 1024 dimensions are a good balance between representation richness and storage cost. |
| `chunk_size=1000` | 500, 2000, 4000 | 1000 characters captures approximately one full paragraph — enough context for coherent retrieval without diluting signal. |
| `chunk_overlap=200` | 0, 100, 500 | 20% overlap prevents information loss at boundaries without excessive duplication (which would inflate the vector index size). |
| Groq LLaMA 3 70B | GPT-4, Claude, LLaMA 3 8B | Best performance at zero cost. Groq's free tier provides fast inference. 70B parameters generate more accurate and detailed responses than 8B. GPT-4/Claude would add API costs. |
| LangGraph | LangChain LCEL, raw Python | LangGraph provides explicit state management and node-based composition, making the RAG workflow debuggable and extensible. |
| Pinecone | Chroma, Weaviate, FAISS | Managed service with zero ops. Free tier is sufficient. Metadata filtering support future multi-document scenarios. |
| Flan-T5-Small for recommendations | Rule-based lookup, GPT-4 API | A fine-tuned T2T model generates more natural text than templates while being fully offline (no API dependency). Flan-T5-Small is small enough to train in minutes. |

---

## 7. Known Limitations & Future Work

| Limitation | Impact | Potential Fix |
|---|---|---|
| Class imbalance (Nematode: 47 vs Fungi: 452) | Lower recall for minority classes | Weighted CrossEntropyLoss, SMOTE oversampling, or collecting more minority-class images |
| No learning rate scheduler | May miss optimal convergence | Add cosine annealing or linear warmup+decay |
| T2T model memorizes 7 samples | Cannot generalize to unseen diseases | Expand training data with more diseases and variations |
| InstructBLIP OOM on P100 | Larger VLM approach incomplete | Use A100 or multi-GPU setup, or switch to InstructBLIP-Flan-T5-Base |
| Single PDF knowledge base | Chatbot knowledge is narrow | Ingest multiple research papers, textbooks, and agricultural guidelines |
| No re-ranking for retrieval | Top-k results may not be optimal | Add a cross-encoder re-ranking step after initial retrieval |
| API key exposure in notebooks | Security concern | Move all keys to environment variables or a secrets manager |
| No CI/CD or automated testing | Manual verification only | Add pytest tests for data pipeline, model inference, and RAG accuracy |
| Dataset images not in repo | Users must download separately | Provide a download script or use DVC for data versioning |

---

## 8. File Inventory

| File | Size | Description | Status |
|---|---|---|---|
| `potatoleaf-vlm-fc93c1.ipynb` | ~2 MB | CLIP fine-tuning, T2T training, InstructBLIP experiment | CLIP: fully trained & evaluated. T2T: trained. InstructBLIP: OOM. |
| `nowwor.ipynb` | ~200 KB | Full RAG chatbot — PDF ingestion through Q&A | Fully working with saved outputs |
| `tes.ipynb` | ~100 KB | RAG chatbot test harness (query-only) | Fully working with saved outputs |
| `README.md` | ~6 KB | Project overview and getting started guide | Complete |
| `TECHNICAL_DOCUMENTATION.md` | This file | Exhaustive technical reference | Complete |
