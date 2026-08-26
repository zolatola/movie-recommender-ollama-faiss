"""
Data loading + cleaning for the movie recommender.
Works with the TMDB top-10K CSV export (id,title,genre,original_language,
overview,popularity,release_date,vote_average,vote_count).
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "id", "title", "genre", "original_language",
    "overview", "popularity", "release_date", "vote_average", "vote_count",
]


def load_and_clean(csv_path: str, max_movies: int | None = None) -> pd.DataFrame:
    """Load the CSV, drop unusable rows, and return a clean, popularity-sorted DataFrame."""
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # Drop rows with no overview or no genre -- there's nothing to build content
    # features from for those, so they can't be recommended meaningfully.
    df = df.dropna(subset=["overview", "genre", "title"]).copy()
    df["overview"] = df["overview"].astype(str).str.strip()
    df = df[df["overview"].str.len() > 10]

    df["genre"] = df["genre"].astype(str)
    df["genre_list"] = df["genre"].apply(
        lambda g: [x.strip() for x in g.split(",") if x.strip()]
    )

    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)

    df = df.drop_duplicates(subset=["title", "release_year"])
    df = df.sort_values("popularity", ascending=False).reset_index(drop=True)

    if max_movies is not None:
        df = df.head(max_movies).reset_index(drop=True)

    # Stable row id used to index into the embeddings matrix (independent of TMDB id).
    df["row_id"] = np.arange(len(df))

    return df


def embedding_text(row) -> str:
    """Build the text blob that gets embedded for a movie.

    Genres are repeated to give them real weight against the (usually much
    longer) overview text -- otherwise a 400-word overview would swamp a
    two-word genre tag in the embedding.
    """
    genres = " ".join(row["genre_list"] * 3)
    return f"Title: {row['title']}. Genres: {genres}. Overview: {row['overview']}"
