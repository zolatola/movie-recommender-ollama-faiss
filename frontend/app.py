import os
import streamlit as st
import pandas as pd

from data import load_and_clean, embedding_text
from recommender import ContentRecommender
from ollama_client import is_running, list_models, OllamaError
from embeddings_store import get_or_build_embeddings, load_cached

APP_DIR = os.path.dirname(__file__)
DEFAULT_CSV = os.path.join(APP_DIR, "top10K-TMDB-movies.csv")

st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
:root {
    --card-bg: #1a1d29;
    --accent: #e8a13c;
}
.stApp { background: linear-gradient(180deg, #12131a 0%, #191b25 100%); }
h1, h2, h3 { font-family: 'Georgia', serif; }
.hero-title { font-size: 2.6rem; font-weight: 800; margin-bottom: 0; }
.hero-sub { color: #9aa0b4; font-size: 1.05rem; margin-top: 0.2rem; }
.mode-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em;
}
.mode-ollama { background: #1f3d2c; color: #6fe098; border: 1px solid #2e6b45; }
.mode-tfidf { background: #3d3320; color: #f0c975; border: 1px solid #6b5a2e; }
.movie-card {
    background: var(--card-bg); border-radius: 14px; padding: 18px 20px;
    margin-bottom: 14px; border: 1px solid #2a2d3d;
}
.movie-title { font-size: 1.15rem; font-weight: 700; color: #f2f2f5; }
.movie-meta { color: #9aa0b4; font-size: 0.85rem; margin-bottom: 8px; }
.genre-tag {
    display: inline-block; background: #262a3d; color: #b9c0d4;
    padding: 2px 10px; border-radius: 999px; font-size: 0.72rem;
    margin-right: 5px; margin-bottom: 5px;
}
.sim-bar-bg { background: #262a3d; border-radius: 6px; height: 7px; width: 100%; margin-top: 6px;}
.sim-bar-fg { background: var(--accent); border-radius: 6px; height: 7px; }
.rating-badge {
    background: #262a3d; color: var(--accent); font-weight: 700;
    border-radius: 8px; padding: 3px 9px; font-size: 0.85rem; float: right;
}
.explain-box {
    background: #20243a; border-left: 3px solid var(--accent);
    padding: 8px 12px; border-radius: 6px; margin-top: 8px;
    font-size: 0.88rem; color: #d8dcec; font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- state ----
defaults = {
    "host": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    "chat_model": "llama3.2",
    "catalog_size": 3000,
    "recommender": None,
    "csv_path": DEFAULT_CSV,
    "explanations": {},
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    uploaded = st.file_uploader("Movie CSV (optional — uses bundled TMDB set if empty)", type=["csv"])
    if uploaded is not None:
        tmp_path = os.path.join(APP_DIR, "cache", "_uploaded.csv")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.csv_path = tmp_path

    st.session_state.host = st.text_input("Ollama host", st.session_state.host)

    ollama_up = is_running(st.session_state.host)
    if ollama_up:
        st.success("Ollama is running")
        try:
            models = list_models(st.session_state.host)
        except OllamaError:
            models = []
    else:
        st.error("Ollama not reachable — will fall back to TF-IDF")
        models = []

    st.session_state.embed_model = st.text_input(
        "Embedding model", st.session_state.embed_model,
        help="e.g. nomic-embed-text, mxbai-embed-large. Run `ollama pull nomic-embed-text`.",
    )
    st.session_state.chat_model = st.text_input(
        "Chat model (for 'why this?' explanations)", st.session_state.chat_model,
        help="e.g. llama3.2, mistral, qwen2.5. Run `ollama pull llama3.2`.",
    )
    if models:
        with st.expander("Models detected on your Ollama server"):
            for m in models:
                st.write(f"• {m}")

    st.session_state.catalog_size = st.slider(
        "Catalog size (top N most popular movies)",
        min_value=500, max_value=10000, value=st.session_state.catalog_size, step=500,
        help="Smaller = faster first-time embedding build. You can re-run with a bigger catalog anytime.",
    )

    st.markdown("---")
    build_clicked = st.button("🔨 Build / load embeddings", use_container_width=True, type="primary")
    st.caption("Embeddings are cached to disk — this is only slow the first time for a given catalog size + model.")

# ---------------------------------------------------------------- header ---
st.markdown('<div class="hero-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Content-based movie recommendations, powered by local Ollama embeddings & LLMs — nothing leaves your machine.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------- build pipeline ---
def load_recommender():
    csv_path = st.session_state.csv_path
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}. Upload one in the sidebar.")
        return None

    df = load_and_clean(csv_path, max_movies=st.session_state.catalog_size)
    rec = ContentRecommender(df)

    if is_running(st.session_state.host):
        cached = load_cached(csv_path, len(df), st.session_state.embed_model)
        if cached is not None:
            rec.set_ollama_embeddings(cached)
            return rec

        progress = st.progress(0.0, text="Embedding movies with Ollama…")
        status = st.empty()

        def cb(done, total):
            progress.progress(done / total, text=f"Embedding movies with Ollama… {done}/{total}")

        try:
            emb = get_or_build_embeddings(
                df, embedding_text, st.session_state.embed_model,
                st.session_state.host, csv_path, progress_cb=cb,
            )
            rec.set_ollama_embeddings(emb)
            progress.empty()
            status.empty()
            return rec
        except OllamaError as e:
            progress.empty()
            st.warning(f"Ollama embedding failed ({e}). Falling back to TF-IDF.")
            rec.build_tfidf_fallback()
            return rec
    else:
        rec.build_tfidf_fallback()
        return rec


if build_clicked or st.session_state.recommender is None:
    with st.spinner("Loading catalog…"):
        st.session_state.recommender = load_recommender()

rec: ContentRecommender = st.session_state.recommender

if rec is None:
    st.stop()

mode_html = (
    '<span class="mode-badge mode-ollama">● Semantic mode — Ollama embeddings</span>'
    if rec.mode == "ollama"
    else '<span class="mode-badge mode-tfidf">● Fallback mode — TF-IDF (start Ollama for better results)</span>'
)
st.markdown(mode_html, unsafe_allow_html=True)
st.caption(f"Catalog loaded: {len(rec.df):,} movies")
st.write("")

# --------------------------------------------------------------- helpers --
ALL_GENRES = sorted({g for gl in rec.df["genre_list"] for g in gl})


def render_results(results: pd.DataFrame, anchor_description: str, key_prefix: str):
    if results.empty:
        st.info("No matches found — try loosening the genre filter.")
        return

    cols = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 2]:
            genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in row["genre_list"])
            sim_pct = max(0.0, min(1.0, float(row["similarity"]))) * 100
            year = row["release_year"] if pd.notna(row["release_year"]) else "—"

            st.markdown(f"""
            <div class="movie-card">
                <span class="rating-badge">★ {row['vote_average']:.1f}</span>
                <div class="movie-title">{row['title']}</div>
                <div class="movie-meta">{year} · {row['vote_count']:,} votes · popularity {row['popularity']:.0f}</div>
                <div>{genres_html}</div>
                <div class="sim-bar-bg"><div class="sim-bar-fg" style="width:{sim_pct:.0f}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Overview"):
                st.write(row["overview"])

            explain_key = f"{key_prefix}_{row['row_id']}"
            if st.button("✨ Why this recommendation?", key=f"btn_{explain_key}"):
                with st.spinner("Asking the local LLM…"):
                    st.session_state.explanations[explain_key] = rec.explain(
                        anchor_description, row,
                        chat_model=st.session_state.chat_model,
                        host=st.session_state.host,
                    )
            if explain_key in st.session_state.explanations:
                st.markdown(
                    f'<div class="explain-box">{st.session_state.explanations[explain_key]}</div>',
                    unsafe_allow_html=True,
                )


# ------------------------------------------------------------------ tabs --
tab1, tab2 = st.tabs(["🎯 Similar to a movie", "💬 Describe what you want"])

with tab1:
    st.write("Pick a movie you like — CineMatch finds others with a similar feel, themes and tone.")
    search = st.text_input("Search for a movie", placeholder="e.g. Inception, Parasite, Spirited Away…")
    matches = rec.find_titles(search, limit=20)

    if matches.empty:
        st.warning("No movies match that search.")
    else:
        options = {
            f"{r.title} ({r.release_year if pd.notna(r.release_year) else '—'})": r.row_id
            for r in matches.itertuples()
        }
        choice = st.selectbox("Matches", list(options.keys()))
        genre_filter1 = st.multiselect("Filter recommendations by genre (optional)", ALL_GENRES, key="gf1")
        top_n1 = st.slider("Number of recommendations", 4, 20, 8, key="n1")

        if st.button("🎬 Recommend", type="primary", key="rec_btn_1"):
            chosen_id = options[choice]
            chosen_row = rec.row_by_id(chosen_id)
            results = rec.recommend_similar_to(chosen_id, top_n=top_n1, genre_filter=genre_filter1 or None)
            st.subheader(f"Because you liked *{chosen_row['title']}*")
            render_results(
                results,
                anchor_description=f"The user liked the movie '{chosen_row['title']}' "
                                    f"({', '.join(chosen_row['genre_list'])}): {chosen_row['overview']}",
                key_prefix="movie",
            )

with tab2:
    st.write("Describe a mood, plot, or vibe in your own words — the local LLM's embedding finds movies that match.")
    query = st.text_area(
        "What are you in the mood for?",
        placeholder="e.g. a slow-burn psychological thriller with an unreliable narrator, "
                    "or a lighthearted animated adventure for family movie night",
        height=90,
    )
    genre_filter2 = st.multiselect("Filter by genre (optional)", ALL_GENRES, key="gf2")
    top_n2 = st.slider("Number of recommendations", 4, 20, 8, key="n2")

    if st.button("🔍 Find movies", type="primary", key="rec_btn_2"):
        if not query.strip():
            st.warning("Type a description first.")
        else:
            results = rec.recommend_for_query(
                query, top_n=top_n2, genre_filter=genre_filter2 or None,
                embed_model=st.session_state.embed_model, host=st.session_state.host,
            )
            st.subheader("Top matches for you")
            render_results(results, anchor_description=f"The user wants: {query}", key_prefix="query")

st.markdown("---")
st.caption(
    "Runs 100% locally: movie data stays on your machine, embeddings and explanations are generated by your local Ollama server."
)
