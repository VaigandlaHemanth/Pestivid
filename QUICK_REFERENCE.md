# Technical Reference Guide## System OverviewPestivid consists of two integrated subsystems:1. **Classification Model**: EfficientNetB0-based image classifier for plant disease detection2. **RAG System**: Semantic search engine for disease information retrieval---## Classification Model Specifications### Architecture- **Base Model**: EfficientNetB0 (pre-trained ImageNet)- **Input**: 256×256×3 RGB image- **Output**: 3-class softmax probabilities- **Frozen Layers**: All backbone- **Trainable Layers**: Dropout + Dense(3)- **Total Parameters**: 4.2M### Training Configuration```Loss Function: Sparse Categorical CrossentropyOptimizer: Adam (LR: 1e-4)Epochs: 20Batch Size: 32Validation Split: 80/20Augmentation: Flip, Rotation, Zoom, Contrast```### Performance| Metric | Value ||--------|-------|| Training Accuracy | 95%+ || Validation Accuracy | 85-92% || Test Accuracy | 80-90% || Inference Latency (GPU) | 50-100ms || Model Size | 50-100MB |### Loss Function**Sparse Categorical Crossentropy**:- Formula: `-log(p_true_class)`- Use Case: Integer labels (0, 1, 2)- Advantage: No one-hot encoding required### Optimizer**Adam**:- Learning Rate: 1e-4- Beta1: 0.9 (momentum)- Beta2: 0.999 (RMSprop)- Epsilon: 1e-7---## Data Pipeline### Dataset**Source**: PlantVillage**Classes** (3):- Early blight- Late blight- Healthy**Specifications**:- Format: JPG/PNG- Size: 256×256 (after resizing)- Color Space: RGB- Normalization: [0, 1] range### Preprocessing Steps1. **Load**: Read from disk2. **Resize**: To 256×256 pixels3. **Convert**: BGR → RGB4. **Normalize**: Divide by 255.0### Augmentation StrategyApplied per batch during training:- **Flip**: Horizontal + Vertical- **Rotation**: ±20%- **Zoom**: ±20%- **Contrast**: ±20%### Data Splitting- **Training**: 80% - Weight updates- **Validation**: 20% - Hyperparameter tuning---## RAG System Specifications### Pipeline Flow```PDF Documents    ↓Text Extraction (pdfplumber)    ↓Text Chunking (500 chars, 50 overlap)    ↓Embedding Generation (Vertex AI)    ↓Vector Upload (Pinecone)    ↓Semantic Search Enabled```### Text Chunking- **Chunk Size**: 500 characters- **Overlap**: 50 characters- **Rationale**: Context preservation, token optimization### Embedding Model- **Provider**: Google Vertex AI- **Model**: text-embedding-005- **Dimensions**: 768-D- **Latency**: 100-200ms per chunk- **Cost**: $0.025 per 1M tokens### Vector Database- **Platform**: Pinecone- **Index**: nowchat- **Dimensions**: 768- **Metric**: Cosine Similarity- **Batch Size**: 5 documents- **Rate Limit**: 10 seconds between batches### Query Process```User Question    ↓Generate Embedding (768-D)    ↓Cosine Similarity Search    ↓Retrieve Top-5 Results    ↓Return Relevant Chunks```---## Inference Specifications### Image Classification**Input**: Single plant leaf image**Process**:1. Load and resize to 256×2562. Normalize (divide by 255)3. Add batch dimension: (1, 256, 256, 3)4. Forward pass through model5. Softmax probabilities: [P0, P1, P2]6. Argmax to get class index7. Return class name + confidence**Output**: ```{    "class": "Late blight",    "confidence": 0.90,    "probabilities": {        "Early blight": 0.05,        "Late blight": 0.90,        "Healthy": 0.05    }}```### Latency- GPU: 50-100ms- CPU: 500-800ms---## RAG Query**Input**: Text question**Process**:1. Generate embedding from question2. Compute cosine similarity with stored vectors3. Retrieve top-K similar chunks4. Return with source metadata**Latency**:- Embedding: 150-200ms- Search: 50-100ms- Total: 200-300ms---## Error Handling### PDF Extraction- **Retries**: 3 attempts- **Timeout**: Per-file timeout- **Fallback**: Skip corrupted files- **Logging**: Record failed documents### Embedding Generation- **Retries**: 3 attempts with 5-second delay- **Rate Limiting**: Respect API limits- **Fallback**: Skip on critical errors- **Logging**: Track failures### Vector Upload- **Batch Processing**: 5 vectors per request- **Error Recovery**: Pause on failure, resume with start_batch- **Logging**: Upload progress tracking---## Configuration Parameters### Training```pythonIMAGE_SIZE = 256BATCH_SIZE = 32EPOCHS = 20NUM_CLASSES = 3LEARNING_RATE = 1e-4DROPOUT_RATE = 0.5VALIDATION_SPLIT = 0.2```### Data Augmentation```pythonFLIP_MODE = "horizontal_and_vertical"ROTATION_FACTOR = 0.2ZOOM_HEIGHT_FACTOR = 0.2ZOOM_WIDTH_FACTOR = 0.2CONTRAST_FACTOR = 0.2```### RAG Pipeline```pythonCHUNK_SIZE = 500CHUNK_OVERLAP = 50EMBEDDING_DIMENSION = 768EMBEDDING_MODEL = "text-embedding-005"BATCH_UPLOAD_SIZE = 5RATE_LIMIT_SLEEP = 10  # secondsEMBEDDING_RETRIES = 3EMBEDDING_RETRY_DELAY = 5  # seconds```---## Files and Locations### Model Checkpoint- **Primary**: `models/plant_disease_model_EfficientNetB0.h5`- **Format**: HDF5 (Keras)- **Size**: ~50-100MB- **Alternatives**:   - `models/best_model.h5`  - `models/1.keras` (native Keras format)### Notebooks- **Training**: `EfficientNet.ipynb`- **Evaluation**: `training.ipynb`- **Preprocessing**: `Pestivid.ipynb`### Scripts- **RAG Pipeline**: `upload_to_pinecone.py`---## Dependencies### Core- TensorFlow >= 2.10- NumPy >= 1.21- OpenCV >= 4.5- LangChain >= 0.0.200- pdfplumber >= 0.7- Pinecone >= 2.2.5- google-cloud-vertexai >= 1.0### Optional- Matplotlib (visualization)- scikit-learn (metrics)- tqdm (progress bars)- tiktoken (token counting)
---

## Key Metrics to Monitor

### Model Performance

- **Accuracy**: % correct predictions
- **Loss**: Cross-entropy value
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (P × R) / (P + R)

### System Performance

- **Inference Latency**: Model prediction time
- **Embedding Latency**: Vector generation time
- **Query Latency**: Search response time
- **Throughput**: Requests per second
- **Error Rate**: Failed requests percentage

---

## API Implementation Example

### Classification Endpoint

```python
@app.post("/predict")
def predict(image: UploadFile) -> dict:
    img = load_and_preprocess(image)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return {
        "class": CLASS_NAMES[class_idx],
        "confidence": float(confidence),
        "probabilities": predictions[0].tolist()
    }
```

### RAG Query Endpoint

```python
@app.post("/query")
def query_disease(question: str) -> dict:
    question_embedding = embeddings.embed_query(question)
    results = index.query(question_embedding, top_k=5)
    return {
        "question": question,
        "results": results["matches"],
        "count": len(results["matches"])
    }
```

---

## Deployment Requirements

### Hardware (GPU Recommended)

- **GPU**: NVIDIA (CUDA 11.8+)
- **VRAM**: 2GB+ for model + batch
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 200GB (model + data)

### Software

- Python 3.8+
- CUDA Toolkit (for TensorFlow GPU)
- cuDNN 8.1+

### Credentials

- **GCP Service Account**: For Vertex AI
- **Pinecone API Key**: For vector database
- **OpenAI API Key** (optional): For LLM responses

---

## Monitoring Checklist

- [ ] Model prediction accuracy per class
- [ ] Inference latency percentiles (p50, p95, p99)
- [ ] Embedding generation latency
- [ ] Vector search latency
- [ ] Error rates and error types
- [ ] API request throughput
- [ ] GPU/CPU utilization
- [ ] Memory usage trends
- [ ] Database query performance
- [ ] Cache hit rates

---

**Repository**: https://github.com/VaigandlaHemanth/Pestivid  
**Status**: Production-Ready  
**Last Updated**: January 29, 2026
