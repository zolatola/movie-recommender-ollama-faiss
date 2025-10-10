"""
prepare_data.py
--------------------------
Prepares TMDB movie data for recommendation.
Generates Ollama embeddings and stores FAISS index + metadata.
"""

import os
import json
import faiss
import numpy as np
import pandas as pd
import requests
import pickle

# Create directories
os.makedirs("models", exist_ok=True)

# Load TMDB datasets
movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
credits = pd.read_csv("data/credits.csv")

# Merge on 'id'
credits['id'] = credits['id'].astype(str)
movies['id'] = movies['id'].astype(str)
df = movies.merge(credits, on='id')

# Combine text fields
def combine_text(row):
    return f"{row['title']} {row['overview']} {row.get('genres', '')} {row.get('cast', '')}"

df['combined'] = df.apply(combine_text, axis=1)
df = df[['id', 'title', 'combined']].dropna().head(1000)

def get_embedding(text):
    """Uses Ollama embedding model to generate a vector."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": "nomic-embed-text", "input": text}
    response = requests.post(url, json=payload)
    return np.array(response.json()["embedding"], dtype="float32")

print("🔹 Generating embeddings...")
embeddings = [get_embedding(t) for t in df['combined']]
embeddings = np.vstack(embeddings)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save
faiss.write_index(index, "models/movie_index.faiss")
with open("models/movie_metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("FAISS index and metadata saved.")
