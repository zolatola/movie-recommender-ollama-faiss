# Local Movie recommendation system using LLMs and FAISS
Content-based movie recommendation system that runs entirely on the local machine.
Uses [Ollama](https://ollama.com) to generate semantic embeddings of each
movie's genres + overview, ranks the catalog by cosine similarity, and can
ask a local LLM to explain *why* a given movie was recommended.

## How it works

- **Data**: the bundled `top10K-TMDB-movies.csv` (10,000 popular movies with
  title, genre, overview, rating, etc.). Other movie CSV files are optional.
- **Content-based filtering**: each movie's title + genres + overview is
  turned into a vector using an Ollama embedding model
  (`nomic-embed-text` by default). Recommendations are the nearest
  neighbors by cosine similarity — no user history or ratings-matrix needed.
- **Two ways to search**:
  1. *Similar to a movie* - pick a movie you like, get similar ones.
  2. *Describe what you want* - type a free-text plot description,
     it's embedded the same way and matched against the catalog.
- **LLM explanations**: click "Why this recommendation?" on any result and
  a local chat model (`llama3.2` by default) writes a short, specific
  reason it fits.
- Embeddings are cached to disk (`cache/`) keyed by dataset + model, so the
  slow part (embedding thousands of movies) only happens once.

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install it (macOS,
Windows, or Linux). Then pull the two models this app uses:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Make sure the server is running (it usually auto-starts; otherwise run
`ollama serve` in a terminal).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

(Python 3.10+ recommended.)

### 3. (Optional) Pre-build embeddings from the command line

The app can build embeddings for you on first run inside the UI, but for
the full 10K-movie catalog you may prefer to kick this off ahead of time:

```bash
python build_embeddings.py --max-movies 3000
```

Drop `--max-movies` to embed the whole catalog (10,000 movies - can take a
while depending on the hardware; each is a small API call to your local
Ollama server). Progress checkpointed every 50 movies, so it's safe to
interrupt and resume.

### 4. Run the app

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. In the sidebar you can:

- Point at a different Ollama host/port
- Change the embedding/chat model names
- Change how many of the top-popularity movies to load (smaller = faster
  first build)
- Upload own CSV (needs `title`, `genre`, `overview` columns at minimum)

## Project layout

```
app.py                 Streamlit UI
data.py                CSV loading/cleaning
ollama_client.py        Thin HTTP client for Ollama's /api/embeddings and /api/chat
embeddings_store.py     Disk-cached, resumable embedding computation
recommender.py          Cosine-similarity recommendation engine + LLM explanations
build_embeddings.py     CLI to pre-build the embedding cache
requirements.txt
top10K-TMDB-movies.csv  Bundled dataset
```

## Troubleshooting

- **"Ollama not reachable"** — make sure `ollama serve` is running and the
  host in the sidebar matches (default `http://localhost:11434`).
- **Embedding build is slow** — lower "Catalog size" in the sidebar, or use
  a smaller/faster embedding model.
- **Model not found errors** — double check you ran `ollama pull <model>`
  for whatever name you typed in the sidebar; it must match exactly.
