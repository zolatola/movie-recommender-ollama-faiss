"""
recommender.py
--------------------------
Handles FAISS search and Ollama embedding generation.
"""

import numpy as np
import faiss
import pickle
import requests

index = faiss.read_index("models/movie_index.faiss")
with open("models/movie_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def get_embedding(text):
    """Generate embedding for input text using Ollama."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": "nomic-embed-text", "input": text}
    response = requests.post(url, json=payload)
    return np.array(response.json()["embedding"], dtype="float32").reshape(1, -1)

def get_recommendations(query, k=5):
    """Return top K similar movies to query."""
    qv = get_embedding(query)
    dist, idx = index.search(qv, k)
    results = []
    for i, d in zip(idx[0], dist[0]):
        title = metadata.iloc[i]["title"]
        sim = 1 / (1 + d)
        results.append((title, sim))
    return results
