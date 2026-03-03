from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="LeetCode Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

DATA_DIR = "data"

# Global states
bm25 = None
faiss_index = None
qlinks = []
embedder = None

print("Loading application state...")
try:
    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
        bm25 = pickle.load(f)

    with open(os.path.join(DATA_DIR, "qlinks.pkl"), "rb") as f:
        qlinks = pickle.load(f)
        
    faiss_index = faiss.read_index(os.path.join(DATA_DIR, "faiss_index.bin"))
    
    # Load the very small embedding model into memory
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Application fully loaded.")
except Exception as e:
    print(f"Warning: Data files or model not found. Error: {e}")
    print("Make sure to run prepare.py first.")
    
class SearchQuery(BaseModel):
    query: str

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    return [term.lower() for term in text.strip().split()]

def min_max_normalize(scores):
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search(query_obj: SearchQuery):
    if not bm25 or not faiss_index or not embedder:
        return {"results": [], "error": "Search indices not fully loaded"}
        
    query_text = query_obj.query.strip()
    if not query_text:
         return {"results": []}

    # 1. BM25 Scores (Keyword Matching)
    tokenized_query = preprocess(query_text)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 2. FAISS Scores (Semantic Matching)
    # Get embedding for query 
    query_embedding = embedder.encode([query_text], normalize_embeddings=True)
    
    # Search the entire index to get semantic similarities for all documents
    # (In massive production we'd limit k, but for 2000 docs doing a full scan and blend is extremely fast)
    faiss_scores = np.zeros(len(bm25_scores))
    top_k = len(bm25_scores) # scan everything to blend
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # distances are cosine similarities because we used IndexFlatIP and normalized embeddings
    for i, idx in enumerate(indices[0]):
        if idx != -1:
             faiss_scores[idx] = distances[0][i]

    # Combine Scores using min-max scaling to give equal weight to BM25 and Semantic Search
    norm_bm25 = min_max_normalize(bm25_scores)
    norm_faiss = min_max_normalize(faiss_scores)
    
    # Hybrid Score (e.g., 40% keyword, 60% semantic)
    hybrid_scores = 0.4 * norm_bm25 + 0.6 * norm_faiss

    # Sort and get top K results
    top_n = 20
    top_n_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    
    results = []
    for idx in top_n_indices:
        score = hybrid_scores[idx]
        if score > 0.05: # Only return highly relevant results
            link = qlinks[idx] if idx < len(qlinks) else ""
            if "leetcode.com" in link:
                try: 
                    title_slug = link.rstrip("/").split("/")[-1]
                    title_parts = title_slug.replace("-", " ")
                    title = title_parts.title()
                except:
                    title = "LeetCode Problem"
            else:
                title = "Question Source"
                
            results.append({
                "title": title,
                "url": link,
                "score": round(float(score), 3)
            })
            
    return {"results": results}
