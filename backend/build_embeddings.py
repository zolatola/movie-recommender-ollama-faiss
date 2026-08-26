"""
Optional CLI to pre-build the embedding cache before launching the app --
handy if you want the full 10K catalog ready without waiting inside the UI.

Usage:
    python build_embeddings.py
    python build_embeddings.py --csv my_movies.csv --max-movies 5000 --model nomic-embed-text
"""

import argparse
import os
import sys

from data import load_and_clean, embedding_text
from embeddings_store import get_or_build_embeddings
from ollama_client import is_running, OllamaError

APP_DIR = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(APP_DIR, "top10K-TMDB-movies.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--max-movies", type=int, default=None, help="Limit to top N by popularity")
    p.add_argument("--model", default="nomic-embed-text")
    p.add_argument("--host", default="http://localhost:11434")
    args = p.parse_args()

    if not is_running(args.host):
        print(f"Ollama isn't reachable at {args.host}. Start it with `ollama serve` "
              f"and make sure `ollama pull {args.model}` has been run.")
        sys.exit(1)

    print(f"Loading {args.csv} ...")
    df = load_and_clean(args.csv, max_movies=args.max_movies)
    print(f"{len(df)} movies loaded. Building embeddings with '{args.model}' "
          f"(this is cached to disk and only needs to run once per catalog size/model)...")

    def cb(done, total):
        pct = 100 * done / total
        bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
        print(f"\r[{bar}] {done}/{total} ({pct:.1f}%)", end="", flush=True)

    try:
        get_or_build_embeddings(df, embedding_text, args.model, args.host, args.csv, progress_cb=cb)
    except OllamaError as e:
        print(f"\nFailed: {e}")
        sys.exit(1)

    print("\nDone. Cached embeddings are in ./cache — the Streamlit app will pick them up automatically.")


if __name__ == "__main__":
    main()
