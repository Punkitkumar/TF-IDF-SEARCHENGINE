import os
import re
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

qData_folder = "Leetcode-Scraping/qData"
if not os.path.exists(qData_folder):
    qData_folder = "Leetcode scraping/qData"

target_str = "Example 1:"

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    return [term.lower() for term in text.strip().split()]

documents = []
qlinks = []

# Load Qlinks
with open(os.path.join(qData_folder, "Qlink.txt"), "r", encoding="utf-8", errors="ignore") as f:
    qlinks = [line.strip() for line in f.readlines()]

print("Loading raw documents...")
for i in range(1, 2040):
    file_path = os.path.join(qData_folder, f"{i}/{i}.txt")
    if not os.path.exists(file_path):
        documents.append("")
        continue
        
    doc = ""
    with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
        for line in f:
            if target_str in line:
                break
            doc += line
    documents.append(doc)

# 1. BM25 Indexing
print("Tokenizing documents for BM25...")
tokenized_corpus = [preprocess(doc) for doc in documents]

print("Initializing BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

# 2. Semantic Search Indexing
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding documents into semantic vectors (this may take a minute)...")
# normalize_embeddings=True makes inner product act as cosine similarity
embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)

print("Building FAISS index...")
dimension = embeddings.shape[1]
# IndexFlatIP does Inner Product (which is Cosine Similarity if vectors are normalized)
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

print("Saving BM25 index, FAISS index, qlinks, and raw documents to data/ directory...")
with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "wb") as f:
    pickle.dump(bm25, f)
    
faiss.write_index(faiss_index, os.path.join(DATA_DIR, "faiss_index.bin"))

with open(os.path.join(DATA_DIR, "qlinks.pkl"), "wb") as f:
    pickle.dump(qlinks, f)

with open(os.path.join(DATA_DIR, "documents.pkl"), "wb") as f:
    pickle.dump(documents, f)

print(f"Processed {len(documents)} documents successfully! Indexes are ready.")