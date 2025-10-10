"""
app.py
-----------------------------------
Flask web interface for the Movie Recommendation System.
Displays recommendations with TMDB posters, overviews, and IMDb links.
"""

import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv
from recommender import get_recommendations

# Load .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

def fetch_movie_details(movie_title):
    """
    Fetch poster, overview, TMDB link, and IMDb link using TMDB API.
    """
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    response = requests.get(base_url, params=params).json()

    if not response.get("results"):
        return None

    movie = response["results"][0]
    movie_id = movie["id"]

    # Get IMDb ID
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    details = requests.get(details_url, params={"api_key": TMDB_API_KEY}).json()
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
    query = ""
    results = []

    if request.method == "POST":
        query = request.form["query"]
        recs = get_recommendations(query)
        for title, score in recs:
            details = fetch_movie_details(title)
            if details:
                details["similarity"] = round(score, 2)
                results.append(details)

    return render_template("index.html", results=results, query=query)

if __name__ == "__main__":
    app.run(debug=True)
