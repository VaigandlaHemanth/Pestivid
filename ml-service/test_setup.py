"""Quick setup test for PestVid"""
print("Testing PestVid Setup...")
print("=" * 50)

# Test imports
try:
    import flask
    print("✅ Flask installed")
except:
    print("❌ Flask not installed - run: pip install flask")

try:
    import flask_cors
    print("✅ Flask-CORS installed")
except:
    print("❌ Flask-CORS not installed - run: pip install flask-cors")

try:
    import requests
    print("✅ Requests installed")
except:
    print("❌ Requests not installed - run: pip install requests")

try:
    from dotenv import load_dotenv
    print("✅ Python-dotenv installed")
except:
    print("❌ Python-dotenv not installed - run: pip install python-dotenv")

try:
    from simple_ai_agent import pestivid_agent
    print("✅ AI Agent loaded")
    result = pestivid_agent.get_quick_tips()
    print(f"✅ AI Agent working - {len(result)} tips available")
except Exception as e:
    print(f"❌ AI Agent error: {e}")

# Test optional AI packages
print("\n" + "=" * 50)
print("Optional AI Packages (for advanced features):")
print("=" * 50)

try:
    import torch
    print(f"✅ PyTorch installed - version {torch.__version__}")
except:
    print("⚠️  PyTorch not installed (needed for disease prediction)")

try:
    import transformers
    print("✅ Transformers installed")
except:
    print("⚠️  Transformers not installed (needed for disease prediction)")

try:
    from PIL import Image
    print("✅ Pillow installed")
except:
    print("⚠️  Pillow not installed (needed for image processing)")

try:
    from langchain_cohere import CohereEmbeddings
    print("✅ LangChain Cohere installed")
except:
    print("⚠️  LangChain packages not installed (needed for RAG chat)")

try:
    from pinecone import Pinecone
    print("✅ Pinecone installed")
except:
    print("⚠️  Pinecone not installed (needed for RAG chat)")

# Check for model files
print("\n" + "=" * 50)
print("Model Files:")
print("=" * 50)

import os
import pathlib
# The .pth checkpoints are gone. Disease detection now loads artifacts/ produced
# by Pestivid/train_potato.py, and treatment text comes from treatments.json.
if os.path.isdir("artifacts") and list(pathlib.Path("artifacts").glob("fold_*_head.pt")):
    print("OK: trained classifier artifacts found")
else:
    print("INFO: no artifacts/ -- /predict will return 503 until you train")
    print("      python Pestivid/train_potato.py --data-root <dataset>")
if os.path.exists("treatments.json"):
    print("OK: treatments.json present")
else:
    print("MISSING: treatments.json")

# Check environment variables
print("\n" + "=" * 50)
print("Environment Variables:")
print("=" * 50)

from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    print(f"✅ GROQ_API_KEY set ({groq_key[:10]}...)")
else:
    print("⚠️  GROQ_API_KEY not set (will use local knowledge base)")

cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    print(f"✅ COHERE_API_KEY set")
else:
    print("⚠️  COHERE_API_KEY not set (RAG chat will not work)")

pinecone_key = os.getenv("PINECONE_API_KEY")
if pinecone_key:
    print(f"✅ PINECONE_API_KEY set")
else:
    print("⚠️  PINECONE_API_KEY not set (RAG chat will not work)")

print("\n" + "=" * 50)
print("Summary:")
print("=" * 50)
print("✅ = Working")
print("⚠️  = Optional (some features won't work)")
print("❌ = Required (must install)")
print("\nYou can start Flask server with: python flask_server.py")
print("Basic AI features will work even without optional packages!")
