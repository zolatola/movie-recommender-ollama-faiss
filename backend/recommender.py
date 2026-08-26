"""
Content-based recommendation engine.

Primary mode: cosine similarity over Ollama embeddings (semantic --
understands that "heist thriller" and "con artists pull off a robbery"
are related even without shared vocabulary).

Fallback mode: TF-IDF cosine similarity over the same text. Used
automatically when Ollama is unreachable, so the app still works; the UI
always shows which mode is active.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import embedding_text
from ollama_client import get_embedding, chat, OllamaError


class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.embeddings: np.ndarray | None = None  # Ollama semantic vectors
        self._tfidf_matrix = None
        self._tfidf = None
        self.mode = "none"  # "ollama" | "tfidf"

    # ---------- setup ----------

    def set_ollama_embeddings(self, embeddings: np.ndarray):
        assert embeddings.shape[0] == len(self.df), "embedding/row count mismatch"
        # L2-normalize once so cosine similarity is just a dot product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = embeddings / norms
        self.mode = "ollama"

    def build_tfidf_fallback(self):
        texts = self.df.apply(embedding_text, axis=1)
        self._tfidf = TfidfVectorizer(max_features=25000, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        self.mode = "tfidf"

    # ---------- lookups ----------

    def find_titles(self, query: str, limit: int = 15) -> pd.DataFrame:
        q = query.strip().lower()
        if not q:
            return self.df.head(limit)
        mask = self.df["title"].str.lower().str.contains(q, na=False)
        return self.df[mask].sort_values("popularity", ascending=False).head(limit)

    def row_by_id(self, row_id: int) -> pd.Series:
        return self.df.loc[self.df["row_id"] == row_id].iloc[0]

    # ---------- recommendation ----------

    def recommend_similar_to(
        self, row_id: int, top_n: int = 10, genre_filter: list[str] | None = None
    ) -> pd.DataFrame:
        idx = self.df.index[self.df["row_id"] == row_id][0]
        sims = self._sims_from_vector(idx)
        return self._rank(sims, exclude_idx={idx}, top_n=top_n, genre_filter=genre_filter)

    def recommend_for_query(
        self,
        query_text: str,
        top_n: int = 10,
        genre_filter: list[str] | None = None,
        embed_model: str = "",
        host: str = "",
    ) -> pd.DataFrame:
        if self.mode == "ollama":
            vec = np.array(get_embedding(query_text, model=embed_model, host=host), dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            sims = self.embeddings @ vec
        else:
            q_vec = self._tfidf.transform([query_text])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        return self._rank(sims, exclude_idx=set(), top_n=top_n, genre_filter=genre_filter)

    def _sims_from_vector(self, idx: int) -> np.ndarray:
        if self.mode == "ollama":
            return self.embeddings @ self.embeddings[idx]
        return cosine_similarity(self._tfidf_matrix[idx], self._tfidf_matrix).ravel()

    def _rank(self, sims: np.ndarray, exclude_idx: set, top_n: int, genre_filter):
        order = np.argsort(-sims)
        results = []
        for idx in order:
            if idx in exclude_idx:
                continue
            row = self.df.iloc[idx]
            if genre_filter:
                if not set(row["genre_list"]) & set(genre_filter):
                    continue
            results.append((idx, sims[idx]))
            if len(results) >= top_n:
                break
        out = self.df.iloc[[i for i, _ in results]].copy()
        out["similarity"] = [s for _, s in results]
        return out

    # ---------- LLM explanation ----------

    def explain(
        self,
        anchor_description: str,
        candidate_row: pd.Series,
        chat_model: str,
        host: str,
    ) -> str:
        """Ask the local LLM for a one/two-sentence reason this rec fits."""
        prompt = (
            "You are a sharp, friendly movie recommender. In 1-2 short sentences, "
            "explain why the CANDIDATE movie is a good recommendation given the "
            "USER CONTEXT. Be specific (mention tone, themes, or plot similarities). "
            "Do not repeat the plot summary verbatim, do not use bullet points.\n\n"
            f"USER CONTEXT: {anchor_description}\n\n"
            f"CANDIDATE: {candidate_row['title']} ({candidate_row.get('release_year', '')}) "
            f"- Genres: {', '.join(candidate_row['genre_list'])}\n"
            f"Overview: {candidate_row['overview']}\n\n"
            "Answer:"
        )
        try:
            return chat(
                messages=[{"role": "user", "content": prompt}],
                model=chat_model,
                host=host,
                timeout=45,
            )
        except OllamaError as e:
            return f"(couldn't reach local LLM for an explanation: {e})"
