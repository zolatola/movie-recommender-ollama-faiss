"""
Fetches poster images and YouTube trailer links from the TMDB API.

The bundled dataset's `id` column is the movie's TMDB id, so no title
matching/searching is needed -- one request per movie
(GET /movie/{id}?append_to_response=videos) gets both the poster path and
the trailer list in a single call.

Requires a free TMDB API key: https://www.themoviedb.org/settings/api
Entirely optional -- if no key is configured, the app just shows a
placeholder instead of a poster and skips the trailer link.
"""

from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

TMDB_API_BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w342"
YOUTUBE_WATCH = "https://www.youtube.com/watch?v="

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "tmdb_media.json")


class TMDBError(Exception):
    pass


def _load_disk_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_disk_cache(cache: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        pass  # caching is a nice-to-have, never fatal


def _pick_trailer_key(videos: list[dict]) -> Optional[str]:
    """Prefer an official YouTube trailer; fall back to any YouTube trailer,
    then any YouTube teaser."""
    def score(v):
        return (
            v.get("site") == "YouTube",
            v.get("type") == "Trailer",
            v.get("official", False),
        )
    candidates = [v for v in videos if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser")]
    if not candidates:
        return None
    best = max(candidates, key=score)
    return best.get("key")


def fetch_movie_media(tmdb_id: int, api_key: str, timeout: float = 8.0) -> dict:
    """Single-movie fetch: {"poster_url": str|None, "trailer_url": str|None}."""
    try:
        r = requests.get(
            f"{TMDB_API_BASE}/movie/{tmdb_id}",
            params={"api_key": api_key, "append_to_response": "videos"},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise TMDBError(str(e))

    poster_path = data.get("poster_path")
    poster_url = f"{IMG_BASE}{poster_path}" if poster_path else None

    videos = data.get("videos", {}).get("results", [])
    trailer_key = _pick_trailer_key(videos)
    trailer_url = f"{YOUTUBE_WATCH}{trailer_key}" if trailer_key else None

    return {"poster_url": poster_url, "trailer_url": trailer_url}


def fetch_media_batch(tmdb_ids: list[int], api_key: str, cache: dict) -> dict:
    """Fetch poster/trailer for a list of TMDB ids, using and updating `cache`
    (mutated in place, and persisted to disk). Missing ids are fetched
    concurrently since these are small, independent HTTP calls.

    Returns {tmdb_id: {"poster_url": ..., "trailer_url": ...}} for every id
    in tmdb_ids (never raises -- failed lookups just come back empty).
    """
    if not api_key:
        return {tid: {"poster_url": None, "trailer_url": None} for tid in tmdb_ids}

    result = {}
    to_fetch = []
    for tid in tmdb_ids:
        key = str(tid)
        if key in cache:
            result[tid] = cache[key]
        else:
            to_fetch.append(tid)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_id = {
                pool.submit(fetch_movie_media, tid, api_key): tid for tid in to_fetch
            }
            for future in as_completed(future_to_id):
                tid = future_to_id[future]
                try:
                    media = future.result()
                except TMDBError:
                    media = {"poster_url": None, "trailer_url": None}
                result[tid] = media
                if media["poster_url"] or media["trailer_url"]:
                    cache[str(tid)] = media  # only cache real hits; retry failures later
        _save_disk_cache(cache)

    return result
