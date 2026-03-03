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

cfData_folder = "cfData"

target_str = "Example 1:"

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    return [term.lower() for term in text.strip().split()]

documents = []
qlinks = []

print("Loading raw LeetCode documents...")
# Load Leetcode Qlinks
if os.path.exists(os.path.join(qData_folder, "Qlink.txt")):
    with open(os.path.join(qData_folder, "Qlink.txt"), "r", encoding="utf-8", errors="ignore") as f:
        qlinks.extend([line.strip() for line in f.readlines()])

    for i in range(1, 2040):
        file_path = os.path.join(qData_folder, f"{i}/{i}.txt")
        if not os.path.exists(file_path):
            if len(documents) < len(qlinks): # keep arrays aligned 
                documents.append("")
            continue
            
        doc = ""
        with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
            for line in f:
                if target_str in line:
                    break
                doc += line
        documents.append(doc)

print("Loading raw Codeforces documents...")
# Load Codeforces Qlinks
if os.path.exists(os.path.join(cfData_folder, "Qlink.txt")):
    cf_links = []
    with open(os.path.join(cfData_folder, "Qlink.txt"), "r", encoding="utf-8", errors="ignore") as f:
        cf_links = [line.strip() for line in f.readlines()]
    
    qlinks.extend(cf_links)

    for i in range(1, len(cf_links) + 1):
        file_path = os.path.join(cfData_folder, f"{i}/{i}.txt")
        if not os.path.exists(file_path):
            documents.append("")
            continue
            
        doc = ""
        with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
            # Codeforces formatting logic (it already has names embedded)
            doc = f.read()
        documents.append(doc)

print(f"Total Combined Documents: {len(documents)}")

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