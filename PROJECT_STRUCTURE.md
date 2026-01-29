# Pestivid - Plant Disease Detection & RAG Chatbot

A comprehensive solution for plant disease detection using deep learning (EfficientNet) and a Retrieval-Augmented Generation (RAG) chatbot powered by Pinecone vector database.

## Project Structure

```
Pestivid/
├── EfficientNet.ipynb           # Main model training with EfficientNet
├── Pestivid.ipynb               # Core preprocessing and data pipeline
├── training.ipynb               # Training metrics and model evaluation
├── upload_to_pinecone.py        # RAG chatbot - PDF to vector DB pipeline
├── models/                      # Trained model files
│   ├── best_model.h5
│   ├── 1.keras
│   └── plant_disease_model_EfficientNetB0.h5
├── PlantVillage/                # Training dataset
├── chatbot/                      # Chatbot implementation
│   └── Upload-to-Pinecone-Python-Script/  # Vector DB integration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Working Files

### Notebooks
- **EfficientNet.ipynb**: Deep learning model for plant disease classification
  - Loads data from PlantVillage dataset
  - Implements EfficientNetB0 architecture
  - Includes data augmentation and transfer learning

- **Pestivid.ipynb**: Main data preprocessing pipeline
  - Image normalization and preprocessing
  - Label encoding and data validation
  - Pseudo-label generation

- **training.ipynb**: Training, evaluation, and metrics
  - Model training loop
  - Performance metrics calculation
  - Visualization and analysis

### Python Scripts
- **upload_to_pinecone.py**: RAG Pipeline
  - Extracts text from PDF documents
  - Generates embeddings using Vertex AI (Google Cloud)
  - Uploads to Pinecone vector database
  - Supports batch processing and resume capability

## Setup & Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies
- TensorFlow 2.x
- Keras
- OpenAI / Google Vertex AI
- Pinecone
- LangChain
- pdfplumber

## Usage

### 1. Train the Model
Run `EfficientNet.ipynb` to train the plant disease detection model.

### 2. Upload Knowledge Base to Pinecone
```bash
python upload_to_pinecone.py
```
Configure before running:
- Set `GOOGLE_APPLICATION_CREDENTIALS` path
- Update `pdf_folder` path to your PDF documents
- Set Pinecone API key and index name

## Features

✅ **Plant Disease Detection**: Multi-class classification using EfficientNet  
✅ **RAG Chatbot**: Question-answering based on knowledge base  
✅ **Vector Database Integration**: Pinecone for semantic search  
✅ **Batch Processing**: Resume support for large datasets  
✅ **Cloud Integration**: Google Vertex AI embeddings  

## Notes

- Model files are stored in `models/` directory
- Large datasets (PlantVillage) should be downloaded separately
- Credentials and sensitive data are in `.gitignore`
- See `upload_to_pinecone.py` for detailed configuration options

## Removed Files

The following were removed as they were test/experimental files:
- Untitled.ipynb, Untitled1.ipynb (test notebooks)
- test*.ipynb files (incomplete tests)
- Various experimental scripts (new.py, gcptry.py, etc.)

---
**Status**: Production-ready core components  
**Last Updated**: January 2025
