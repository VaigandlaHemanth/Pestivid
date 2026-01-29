import os
import time
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# ── STEP 1: Set Application Default Credentials (ADC) ─────────────────────────────
# Point this to your Vertex AI service‐account JSON (must have "Vertex AI Embedding User" role).
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  r"rising-abacus-461617-d2-49c712714ba6.json"

# ── Vertex AI (Gemini) Embeddings ───────────────────────────────────────────────
from langchain_google_vertexai import VertexAIEmbeddings

# ── Pinecone (v2.2.5) ─────────────────────────────────────────────────────────────
import pinecone


# ── Configuration ───────────────────────────────────────────────────────────────

# 1) Vertex AI (Gemini) Embeddings via ADC
PROJECT = "rising-abacus-461617-d2"    # ← your GCP Project ID
REGION  = "us-central1"                # ← region where Vertex AI is enabled
MODEL   = "text-embedding-005"    # ← correct Gemini embedding model name

vertex_embeddings = VertexAIEmbeddings(
    project=PROJECT,
    location=REGION,
    model_name=MODEL,
)

# 2) Pinecone configuration (v2.2.5)
pinecone_api_key = "pcsk_2zLsPR_TW281dRvebjuvjaL6MbQLawuMjQyiYWj6wog7FSddx6otQaFj4ESRenCCnqYnmh"
index_name = "nowchat"  # Use the existing 768-dim index

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# 3) PDF‐folder path (adjust if needed)
pdf_folder = r"potato_leaf_disease - Copy/train"


# ── Helper: Extract text from a PDF safely ────────────────────────────────────
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


# ── Helper: Split text into chunks ―――――――――――――――――――――――――――――――――――――――
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ── Helper: Generate an embedding via Vertex AI (Gemini) ―――――――――――――――――――
def get_embedding(text: str, retries: int = 3, delay: int = 5) -> list[float] | None:
    """
    Use VertexAIEmbeddings.embed_query(...) to get a vector for `text`.
    Retries on transient errors up to `retries` times.
    """
    for attempt in range(retries):
        try:
            return vertex_embeddings.embed_query(text)
        except Exception as e:
            print(f"⚠️ Embedding failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds…")
                time.sleep(delay)
            else:
                print("❌ Skipping this chunk after multiple failed attempts.")
                return None


# ── Build list of all chunks from all PDFs ――――――――――――――――――――――――――――――――――――
all_chunks: list[dict] = []

for filename in os.listdir(pdf_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    file_path = os.path.join(pdf_folder, filename)
    print(f"Processing {filename}…")

    text = extract_text(file_path)
    if not text:
        continue

    splits = splitter.split_text(text)
    for i, chunk in enumerate(splits):
        all_chunks.append({
            "id":       f"{filename}-{i}",
            "text":     chunk,
            "metadata": {"source": filename}
        })


# ── Upload to Pinecone in Batches (with resume support) ―――――――――――――――――――――
start_batch = 0    # Change if resuming from a specific batch
batch_size  = 5    # smaller batch size

total_batches = ((len(all_chunks) - 1) // batch_size) + 1

for batch_num in range(start_batch, total_batches):
    start_idx = batch_num * batch_size
    batch     = all_chunks[start_idx : start_idx + batch_size]
    print(f"Uploading batch {batch_num + 1} of {total_batches}")

    vectors: list[dict] = []
    for item in batch:
        embedding = get_embedding(item["text"])
        if embedding is None:
            continue
        vectors.append({
            "id":       item["id"],
            "values":   embedding,
            "metadata": item["metadata"]
        })

    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"✅ Uploaded {len(vectors)} vectors.")
        except Exception as e:
            print(f"❌ Failed to upload batch {batch_num + 1}: {e}")
            print("❗ Stopping script to prevent data loss. Update `start_batch` to resume later.")
            break
    else:
        print("⚠️ No vectors in this batch to upload (all embeddings failed).")

    time.sleep(10)  # longer pause between batches

print("✅ Script execution completed.")
