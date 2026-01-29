# Pestivid - Complete Technical Documentation

**Date**: January 29, 2026  
**Project**: Plant Disease Detection + RAG Chatbot  
**Repository**: https://github.com/VaigandlaHemanth/Pestivid

---

## 🎯 PROJECT OVERVIEW

Pestivid is a **dual-component AI system** for plant disease detection and interactive querying:

1. **Deep Learning Model** - Multi-class plant disease classification
2. **RAG Chatbot** - Question-answering system using vector database

---

## 📊 SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                     PESTIVID SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Plant Images    │         │  Plant Disease   │         │
│  │  (PlantVillage)  │         │  Documents (PDF) │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                           │                    │
│           ▼                           ▼                    │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ EfficientNet.ipynb          │upload_to_pinecone.py│        │
│  │ (Model Training) │         │ (PDF → Embeddings)│         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                           │                    │
│           ▼                           ▼                    │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Trained Model    │         │  Pinecone Vector │         │
│  │ (Disease Classes)│         │  Database (RAG)  │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                           │                    │
│           └──────────────┬────────────┘                    │
│                          ▼                                 │
│                   ┌─────────────┐                          │
│                   │  Inference  │                          │
│                   │  & Results  │                          │
│                   └─────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 COMPONENT 1: DEEP LEARNING MODEL

### **File: EfficientNet.ipynb**

#### **Purpose**
Train a multi-class plant disease classifier using transfer learning with EfficientNetB0.

#### **Dataset**
- **Source**: PlantVillage dataset
- **Classes**: 3 classes (e.g., Early blight, Late blight, Healthy)
- **Image Size**: 256×256 pixels
- **Batch Size**: 32
- **Train/Validation Split**: 80/20

#### **Model Architecture**

```
Input: (256, 256, 3)
       ↓
   EfficientNetB0 (pre-trained on ImageNet)
   ├─ Frozen base layers (transfer learning)
   ├─ Global Average Pooling
   ├─ Dropout(0.5)
       ↓
   Dense Layer: 3 nodes
   ├─ Activation: Softmax
       ↓
   Output: (3,) - class probabilities
```

#### **Key Configuration**
```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 20
MODEL_NAME = "EfficientNetB0"
```

#### **Preprocessing Pipeline**

1. **Normalization**
   - Resizing to 256×256
   - Rescaling: pixel values / 255.0 (normalize to [0,1])

2. **Data Augmentation**
   - RandomFlip: horizontal and vertical
   - RandomRotation: ±20%
   - RandomZoom: ±20%
   - RandomContrast: ±20%

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(factor=0.2),
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    tf.keras.layers.RandomContrast(factor=0.2),
])
```

#### **Loss Function & Optimizer**

| Component | Value |
|-----------|-------|
| **Loss Function** | Sparse Categorical Crossentropy |
| **Optimizer** | Adam (learning rate: 1e-4) |
| **Metrics** | Accuracy |

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

**Why Sparse Categorical Crossentropy?**
- Labels are integers (0, 1, 2)
- No need for one-hot encoding
- Computationally efficient

#### **Training Process**

```python
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)
```

**Training Details**:
- Epochs: 20
- Validation performed after each epoch
- Early stopping can be added via callbacks

#### **Model Output**

- **Saved as**: `models/plant_disease_model_EfficientNetB0.h5`
- **Format**: HDF5 (Keras model format)
- **Size**: ~50-100 MB (depending on ImageNet weights)

#### **Inference**

```python
# Load model
model = load_model("models/plant_disease_model_EfficientNetB0.h5")

# Preprocess image
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, 0)

# Predict
predictions = model.predict(img_array)  # Shape: (1, 3)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
```

---

## 📁 COMPONENT 2: TRAINING & EVALUATION

### **File: training.ipynb**

#### **Purpose**
Train a CNN from scratch (or use transfer learning) for plant disease classification with detailed evaluation.

#### **Model Architecture (CNN-based)**

```
Input: (256, 256, 3)
       ↓
Conv2D(32, 3×3, ReLU) + MaxPool(2×2)
Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
Conv2D(64, 3×3, ReLU) + MaxPool(2×2)
       ↓
Flatten()
       ↓
Dense(64, ReLU)
       ↓
Dense(3, Softmax)  ← Output (3 classes)
```

#### **Data Splitting**

```python
train_ds, val_ds, test_ds = get_dataset_partitions_tf(
    dataset,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    shuffle=True
)
```

| Set | Ratio | Purpose |
|-----|-------|---------|
| **Training** | 80% | Model learning |
| **Validation** | 10% | Hyperparameter tuning |
| **Test** | 10% | Final evaluation |

#### **Loss Function & Optimizer**

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

#### **Training Configuration**

```python
EPOCHS = 50
BATCH_SIZE = 32
IMAGE_SIZE = 256
```

#### **Performance Metrics**

```python
# Training loop
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50
)

# Evaluation on test set
score = model.evaluate(test_ds, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])
```

**Metrics Tracked**:
- **Accuracy**: Percentage of correct predictions
- **Loss**: Sparse Categorical Crossentropy value
- **Validation Accuracy**: Accuracy on validation set

#### **Visualization**

```python
# Extract metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training vs validation
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
```

#### **Batch Prediction**

```python
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
```

#### **Model Checkpointing**

```python
save_path = os.path.join(model_dir, f"{next_version}.keras")
model.save(save_path)
```

---

## 📁 COMPONENT 3: DATA PREPROCESSING

### **File: Pestivid.ipynb**

#### **Purpose**
Preprocess plant disease images, generate importance maps, and create training datasets.

#### **Key Functions**

##### **1. Image Normalization**
```python
def normalize_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 512×512
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
    return img.astype(np.float32) / 255.0        # Normalize to [0,1]
```

**Process**:
- Read image in BGR (OpenCV default)
- Resize to 512×512 pixels
- Convert to RGB color space
- Normalize pixel values to [0, 1]

##### **2. Label Normalization**
```python
def normalize_label(label):
    label = cv2.resize(label, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    label = label.astype(np.float32)
    if label.ndim == 3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label /= label.max() if label.max() != 0 else 1.0
    return label
```

**Process**:
- Resize label mask to 512×512
- Convert to grayscale if needed
- Normalize to [0, 1]

##### **3. Importance Map Computation**
```python
def compute_importance_map(label):
    weights = label / (np.sum(label) + 1e-6)
    return weights
```

**Purpose**: Assigns higher weights to pixels with higher disease presence.

##### **4. Similarity Map Computation**
```python
def compute_similarity_map(img):
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    similarity_map = np.exp(-edge_strength / (np.max(edge_strength) + 1e-6))
    return similarity_map
```

**Purpose**: 
- Identifies edge regions using Sobel operators
- High similarity in smooth regions, low at edges
- Used for spatial smoothness in weakly-supervised learning

#### **Dataset Output Structure**

```
preprocessed_data/
├── images/              # Normalized RGB images (512×512)
├── pseudo_labels/       # Disease labels/masks
├── importance_maps/     # Weighted importance masks
└── similarity_maps/     # Spatial similarity maps
```

---

## 📁 COMPONENT 4: RAG CHATBOT

### **File: upload_to_pinecone.py**

#### **Purpose**
Build a Retrieval-Augmented Generation (RAG) system that converts PDF documents into vector embeddings and uploads them to Pinecone for semantic search and question-answering.

#### **System Flow**

```
┌──────────────┐
│ PDF Files    │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Extract Text         │
│ (pdfplumber)         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Split into Chunks    │
│ (RecursiveCharTextSpl)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Generate Embeddings  │
│ (Vertex AI / Gemini) │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Upload to Pinecone   │
│ (Vector Database)    │
└──────────────────────┘
```

#### **1. PDF Text Extraction**

```python
def extract_text(file_path: str) -> str | None:
    try:
        merged_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    merged_text += txt + "\n"
        return merged_text
    except Exception as e:
        print(f"⚠️ Failed to process {file_path}: {e}")
        return None
```

**Library**: `pdfplumber` - Extracts text from PDFs with layout preservation

#### **2. Text Chunking Strategy**

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 500 characters per chunk
    chunk_overlap=50     # 50 character overlap between chunks
)

splits = splitter.split_text(text)
```

**Configuration**:
- **Chunk Size**: 500 characters
  - Balances context window and granularity
  - ~80-100 tokens per chunk
- **Overlap**: 50 characters
  - Preserves context at chunk boundaries
  - Improves retrieval quality

#### **3. Embedding Generation (Google Vertex AI)**

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"gcp-service-account.json"

from langchain_google_vertexai import VertexAIEmbeddings

vertex_embeddings = VertexAIEmbeddings(
    project="rising-abacus-461617-d2",
    location="us-central1",
    model_name="text-embedding-005"  # Google's latest embedding model
)

def get_embedding(text: str, retries: int = 3, delay: int = 5) -> list[float] | None:
    for attempt in range(retries):
        try:
            return vertex_embeddings.embed_query(text)
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
```

**Embedding Model**: 
- **Name**: `text-embedding-005` (Google Gemini)
- **Vector Dimension**: 768-D (default)
- **Cost**: $0.025 per 1M tokens

#### **4. Pinecone Vector Database Upload**

```python
from pinecone import Pinecone

pinecone_api_key = "pcsk_xxxxxxxxxxxx"
index_name = "nowchat"

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Batch upload
for batch_num in range(total_batches):
    vectors: list[dict] = []
    for item in batch:
        embedding = get_embedding(item["text"])
        if embedding is None:
            continue
        vectors.append({
            "id": item["id"],
            "values": embedding,
            "metadata": item["metadata"]
        })
    
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} vectors.")
        except Exception as e:
            print(f"❌ Failed: {e}")
            break
    
    time.sleep(10)  # Rate limiting
```

**Configuration**:
- **Batch Size**: 5 documents per batch
- **Sleep Time**: 10 seconds between batches (rate limiting)
- **Resume Support**: Set `start_batch` to resume from a specific batch

#### **5. Metadata Structure**

```python
all_chunks.append({
    "id": f"{filename}-{chunk_index}",
    "text": chunk_content,
    "metadata": {"source": filename}
})
```

**Stored in Pinecone**:
- **ID**: Unique identifier for retrieval
- **Vector**: 768-D embedding
- **Metadata**: Source document filename

#### **Key Features**

| Feature | Implementation |
|---------|-----------------|
| **Error Handling** | 3 retries with exponential backoff |
| **Batch Processing** | Process in batches of 5 |
| **Resume Support** | Update `start_batch` to resume |
| **Rate Limiting** | 10-second sleep between batches |
| **Logging** | Progress indicators for each batch |

---

## 🔄 DATA FLOW SUMMARY

### **Training Pipeline**
1. Load PlantVillage dataset (256×256 images, 3 classes)
2. Apply data augmentation (rotation, zoom, flip, contrast)
3. Train EfficientNetB0 with sparse categorical crossentropy loss
4. Validate on 20% held-out data
5. Save model to `models/plant_disease_model_EfficientNetB0.h5`

### **Inference Pipeline**
1. Load trained model
2. Preprocess input image (resize, normalize)
3. Generate predictions (3 classes, softmax probabilities)
4. Return predicted class and confidence score

### **RAG Chatbot Pipeline**
1. Extract text from PDF documents
2. Split text into 500-character chunks (with 50-char overlap)
3. Generate 768-D embeddings using Vertex AI
4. Upload embeddings to Pinecone
5. Enable semantic search and Q&A

---

## 🛠️ DEPENDENCIES & VERSIONS

### **Deep Learning**
- TensorFlow 2.x
- Keras (included with TensorFlow)
- tf-addons (advanced augmentations)

### **Data Processing**
- NumPy
- OpenCV (cv2)
- Pandas
- scikit-learn (metrics)

### **RAG Chatbot**
- LangChain
- pdfplumber
- Pinecone
- Google Cloud Vertex AI
- tiktoken (token counting)

### **Optional**
- PyTorch + segmentation-models-pytorch (for segmentation)
- Matplotlib (visualization)
- tqdm (progress bars)

---

## 📈 PERFORMANCE METRICS

### **Model Evaluation Metrics**
- **Accuracy**: Percentage of correct classifications
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Class-wise classification breakdown

### **RAG System Metrics**
- **Embedding Quality**: Vector similarity / cosine distance
- **Retrieval Precision**: Relevance of top-K results
- **Response Latency**: Time to generate embeddings and retrieve results

---

## 🔐 CREDENTIALS & CONFIGURATION

### **Required Files**
1. **GCP Service Account JSON** (for Vertex AI)
   - Path: `rising-abacus-461617-d2-49c712714ba6.json`
   - Permissions: Vertex AI Embedding User

2. **Pinecone API Key**
   - Format: `pcsk_xxxxxxxxxxxx`
   - Purpose: Vector database access

### **Configuration Variables**
```python
# EfficientNet Training
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# RAG Pipeline
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-005"
BATCH_SIZE_UPLOAD = 5

# Pinecone
INDEX_NAME = "nowchat"
VECTOR_DIMENSION = 768
```

---

## 💾 MODEL ARTIFACTS

### **Trained Models**
- `models/plant_disease_model_EfficientNetB0.h5` - Main model
- `models/1.keras` - Alternative format
- `models/best_model.h5` - Checkpoint

### **Dataset**
- `PlantVillage/` - Training images (3 classes)
  - Early blight
  - Late blight
  - Healthy

### **Preprocessed Data**
- `preprocessed_data/images/` - Normalized images
- `preprocessed_data/pseudo_labels/` - Label masks
- `preprocessed_data/importance_maps/` - Weighted masks
- `preprocessed_data/similarity_maps/` - Spatial similarity

---

## 🚀 DEPLOYMENT CHECKLIST

- [ ] Verify model accuracy on test set
- [ ] Load model and test inference
- [ ] Prepare PDF documents for RAG
- [ ] Test embedding generation with sample texts
- [ ] Verify Pinecone connection and upload
- [ ] Test end-to-end inference pipeline
- [ ] Document class names and confidence thresholds
- [ ] Set up monitoring for model predictions
- [ ] Configure backup and versioning strategy

---

## 📝 NOTES & RECOMMENDATIONS

### **Model Improvements**
- Fine-tune EfficientNetB0 by unfreezing top layers
- Implement MixUp/CutMix for advanced augmentation
- Use ensemble methods (multiple models)
- Add focal loss for class imbalance

### **RAG System Enhancements**
- Implement multi-level chunking (paragraph/sentence level)
- Add metadata filtering before retrieval
- Use hybrid search (keyword + semantic)
- Implement re-ranking of results
- Add LLM-based response generation

### **Production Considerations**
- Containerize with Docker
- Set up API endpoints (FastAPI/Flask)
- Implement caching for common queries
- Monitor embedding latency
- Set up alerts for model drift

---

**Last Updated**: January 29, 2026  
**Status**: Production-Ready  
**Maintainer**: VaigandlaHemanth
