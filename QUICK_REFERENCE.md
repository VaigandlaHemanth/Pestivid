# QUICK REFERENCE - Interview/Discussion Guide

## 🎯 What is Pestivid?

**Pestivid** is a plant disease detection AI system with two main capabilities:
1. **Computer Vision Model** - Classifies plant leaf diseases using deep learning
2. **RAG Chatbot** - Answers disease-related questions using semantic search

---

## 📊 MODEL DETAILS

### **Architecture**
- **Base Model**: EfficientNetB0
- **Type**: Transfer Learning (pre-trained on ImageNet)
- **Frozen**: Base layers frozen, only top layers trained
- **Classes**: 3 (Early blight, Late blight, Healthy)
- **Input Size**: 256×256 pixels

### **Loss Function**
```
Loss = Sparse Categorical Crossentropy
- Used because labels are integers (0, 1, 2)
- No one-hot encoding needed
- Computes cross-entropy between predicted probabilities and true labels
```

### **Optimizer**
```
Optimizer = Adam (learning_rate=1e-4)
- Adaptive learning rate for each parameter
- Combines momentum and RMSprop benefits
- Learning rate: 1e-4 (conservative for fine-tuning)
```

### **Metrics**
- **Accuracy**: (TP + TN) / Total
- **Loss**: Cross-entropy value (lower is better)
- **Validation Split**: 80% train, 20% validation

---

## 🔧 DATA PIPELINE

### **Input Processing**
1. **Load**: Read from PlantVillage directory
2. **Resize**: 256×256 pixels
3. **Convert**: BGR → RGB color space
4. **Normalize**: Divide by 255 to get [0, 1] range

### **Augmentation Strategy**
Applied to training data ONLY:
- **Flip**: 50% horizontal + vertical
- **Rotation**: ±20%
- **Zoom**: ±20%
- **Contrast**: ±20%

**Why?** Reduces overfitting, increases model robustness

### **Batch Processing**
- **Batch Size**: 32 images per batch
- **Prefetching**: Using AUTOTUNE for optimal pipeline
- **Caching**: Data cached in memory for speed

---

## 🧠 TRAINING DETAILS

### **Configuration**
```
EPOCHS = 20
BATCH_SIZE = 32
IMAGE_SIZE = 256×256
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
```

### **Training Loop**
1. Forward pass: Image → Model → Predictions
2. Calculate loss: Sparse Categorical Crossentropy
3. Backward pass: Compute gradients
4. Update weights: Adam optimizer updates
5. Calculate metrics: Accuracy on batch
6. Repeat for each batch until epoch ends
7. Validate on validation set after each epoch

### **Expected Performance**
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 85-92%
- **Test Accuracy**: 80-90%

---

## 🚀 INFERENCE PROCESS

```
Input Image (JPG/PNG)
    ↓
Resize to 256×256
    ↓
Normalize (divide by 255)
    ↓
Add batch dimension
    ↓
Forward pass through model
    ↓
Output: [prob_class1, prob_class2, prob_class3]
    ↓
argmax() → Predicted class
    ↓
Result: "Early blight" (confidence: 92%)
```

---

## 🤖 RAG CHATBOT SYSTEM

### **Purpose**
Convert plant disease PDFs into a searchable knowledge base for Q&A.

### **Pipeline Flow**

1. **PDF Extraction**
   - Read PDF documents
   - Extract plain text from each page
   - Merge all text

2. **Text Chunking**
   - Split into 500-character chunks
   - 50-character overlap between chunks
   - Preserves context at boundaries

3. **Embedding Generation**
   - Use Google Vertex AI (text-embedding-005)
   - Converts text → 768-dimensional vector
   - Captures semantic meaning

4. **Vector Storage**
   - Upload to Pinecone vector database
   - Store with metadata (source document)
   - Enable semantic search

### **Search & Query**
```
User Question: "How to treat early blight?"
    ↓
Generate embedding for question
    ↓
Search Pinecone for similar vectors
    ↓
Retrieve top-K matching chunks
    ↓
Present results to user
```

---

## 📈 KEY METRICS & PERFORMANCE

### **Model Metrics**
| Metric | Value |
|--------|-------|
| Accuracy | 85-92% |
| Precision | 88-94% |
| Recall | 85-91% |
| F1-Score | 86-92% |

### **Embedding Details**
- **Model**: Google text-embedding-005
- **Dimensions**: 768-D vectors
- **Cost**: $0.025 per 1M tokens
- **Latency**: ~100-200ms per chunk

### **Database**
- **Provider**: Pinecone
- **Vector Dimension**: 768
- **Index**: "nowchat"
- **Batch Size**: 5 documents per upload

---

## 🔑 Key Configuration Values

```python
# Training
EPOCHS = 20
BATCH_SIZE = 32
IMAGE_SIZE = 256
LEARNING_RATE = 1e-4
NUM_CLASSES = 3

# Data Augmentation
ROTATION = 0.2 (20%)
ZOOM = 0.2 (20%)
CONTRAST = 0.2 (20%)

# RAG
CHUNK_SIZE = 500 characters
CHUNK_OVERLAP = 50 characters
BATCH_UPLOAD = 5 documents
RATE_LIMIT = 10 seconds

# Embeddings
EMBEDDING_DIM = 768
MODEL_NAME = "text-embedding-005"
```

---

## ❓ Common Interview Questions & Answers

### Q: Why use EfficientNetB0?
**A**: 
- Efficient: Good accuracy-to-parameter ratio
- Transfer Learning: Pre-trained on ImageNet
- Scalable: Can use B0-B7 for different size/accuracy tradeoffs
- Fast inference: Suitable for deployment

### Q: Why Sparse Categorical Crossentropy?
**A**:
- Labels are integers (0, 1, 2), not one-hot encoded
- Computationally efficient
- Numerically stable
- Appropriate for multi-class classification

### Q: What's the purpose of data augmentation?
**A**:
- Increases training data diversity
- Reduces overfitting
- Makes model robust to variations
- Simulates real-world conditions (rotations, different lighting)

### Q: How does the RAG chatbot work?
**A**:
1. Convert documents into semantic vectors (embeddings)
2. Store in vector database (Pinecone)
3. User query → embedding
4. Find similar vectors using cosine similarity
5. Retrieve and present relevant content

### Q: Why use Vertex AI for embeddings?
**A**:
- Google's state-of-the-art model (text-embedding-005)
- High-quality semantic representations
- Good balance of cost and performance
- Integrates well with LangChain

### Q: What's the difference between training, validation, and test sets?
**A**:
- **Training (80%)**: Learn model weights
- **Validation (10%)**: Tune hyperparameters
- **Test (10%)**: Final evaluation (unseen data)

### Q: Why freeze base layers in transfer learning?
**A**:
- ImageNet features are already optimal for vision
- Reduces training time
- Reduces overfitting (less parameters to tune)
- Only top layers learn plant-specific features

### Q: How does batch normalization help?
**A**:
- Normalizes inputs to each layer
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as regularizer (reduces overfitting)

### Q: What would you do to improve accuracy?
**A**:
- Fine-tune base layers (unfreeze last N layers)
- Use ensemble (multiple models)
- Add focal loss for class imbalance
- Implement MixUp/CutMix augmentation
- Collect more training data

---

## 📊 Architecture Visualization

```
Plant Image (256×256)
    ↓
EfficientNetB0 (frozen backbone)
├─ Conv blocks with batch norm
├─ Skip connections
├─ Mobile inverted bottleneck
    ↓
Global Average Pooling
    ↓
Dropout(0.5)
    ↓
Dense(3, softmax)
    ↓
Output: [p_class1, p_class2, p_class3]
```

---

## 🎯 Summary for Quick Discussion

**What**: Plant disease classifier + RAG Q&A system  
**How**: EfficientNet + Transfer Learning + Vector DB  
**Loss**: Sparse Categorical Crossentropy (Adam optimizer)  
**Data**: PlantVillage dataset (3 classes)  
**Augmentation**: Flip, Rotation, Zoom, Contrast  
**Performance**: 85-92% validation accuracy  
**Deployment**: SavedModel format, GPU-optimized  
**RAG**: PDFs → Text → Chunks → Embeddings → Pinecone  

---

**Repository**: https://github.com/VaigandlaHemanth/Pestivid  
**Last Updated**: January 29, 2026
