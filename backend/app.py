"""
app.py
--------------------------
Flask backend for the Movie Recommender.
Integrates FAISS + Ollama for similarity and TMDB API for visuals.
"""

import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv
from recommender import get_recommendations

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

def fetch_movie_details(title):
    """Fetch poster, overview, and IMDb link from TMDB."""
    base = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    res = requests.get(base, params=params).json()
    if not res.get("results"):
        return None
    movie = res["results"][0]
    movie_id = movie["id"]

    details = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}",
        params={"api_key": TMDB_API_KEY},
    ).json()

    imdb_id = details.get("imdb_id")
    return {
        "title": movie["title"],
        "overview": movie.get("overview", "No overview available."),
        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
        "tmdb_url": f"https://www.themoviedb.org/movie/{movie_id}",
        "imdb_url": f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    query, results = "", []
    if request.method == "POST":
        query = request.form["query"]
        recs = get_recommendations(query)
        for title, score in recs:
            d = fetch_movie_details(title)
            if d:
                d["similarity"] = round(score, 2)
                results.append(d)
    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
