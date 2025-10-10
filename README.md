# Movie Recommendation System (Content-Based · FAISS · Ollama · Flask)

A modern **content-based movie recommender system** built with **Python**, **Ollama LLM embeddings**, **FAISS vector search**, and a clean **Flask web interface**.

This project demonstrates how to combine **Large Language Models (LLMs)** with **vector databases** to build smart, semantic movie recommendations — similar to Netflix or IMDb’s “More Like This” feature.

---

## Features

- Uses **content-based filtering** (recommends movies similar in plot, cast, and genre)  
- Runs fully **offline** using local embeddings via **Ollama**  
- Uses **FAISS** for high-speed semantic similarity search  
- Interactive **Flask web app** for easy use  
- Built with **TMDB Movie Metadata** from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

---

## How It Works

1. **Data Preparation**
   - Loads and cleans movie data from TMDB’s `movies_metadata.csv` and `credits.csv`.

2. **Embedding Generation**
   - Each movie’s plot and cast are converted into vector embeddings using **Ollama**’s `nomic-embed-text` model.

3. **Vector Indexing**
   - Embeddings are stored in a **FAISS** vector index for fast similarity search.

4. **Recommendation**
   - When a user searches for a movie or description, the system embeds that input and retrieves the closest matches.

5. **Frontend**
   - Flask renders the results on a clean, responsive web interface.

---

## Tech Stack

 Backend: Python, Flask 
 LLM Embeddings: Ollama (`nomic-embed-text`) 
 Vector Search: FAISS 
 Data: TMDB Movie Metadata (Kaggle) 
 Frontend: HTML, CSS, Jinja Templates 

---

##  Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender-ollama-faiss.git
cd movie-recommender-ollama-faiss
