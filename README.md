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

### 1. Clone the Repository

`git clone https://github.com/zolatola/movie-recommender-ollama-faiss.git
cd movie-recommender-ollama-faiss`

### 2. Install Dependencies

Make sure Python 3.9+ is installed, then run:

`pip install -r requirements.txt`

### 3. Download the TMDB Dataset

From Kaggle:
`TMDB Movie Metadata Dataset`

Place these files inside the data/ folder:

`data/movies_metadata.csv`
`data/credits.csv`

### 4. Install and Run Ollama

Download Ollama from `https://ollama.com/download`

Start the Ollama server from the terminal:

`ollama serve`


Pull the embedding model:

`ollama pull nomic-embed-text`


Verify it’s working:

`curl http://localhost:11434/api/tags`

### 5. Prepare Embeddings and FAISS Index
`python prepare_data.py`

### 6. Run the Flask Web App
`python app.py`


Open your browser and visit:
`http://127.0.0.1:5000`

Type a movie title (e.g. Inception) or a description (“space exploration and survival”) and hit Find Recommendations.
