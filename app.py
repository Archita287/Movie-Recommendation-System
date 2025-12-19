import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# --------------------------------------------------
# Load & REDUCE data aggressively
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sampled_movies_ratings.csv")
    df = df.dropna()

    # 🔴 HARD LIMIT FOR STREAMLIT CLOUD
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=132629)

    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    return df

df = load_data()

st.caption(f"Dataset size used: {df.shape[0]} rows")

# --------------------------------------------------
# TF-IDF similarity (very light)
# --------------------------------------------------
@st.cache_data
def compute_similarity(genres):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=1500
    )
    tfidf_matrix = tfidf.fit_transform(genres)
    return cosine_similarity(tfidf_matrix)

# --------------------------------------------------
# SVD (minimum viable)
# --------------------------------------------------
@st.cache_data
def compute_svd(df):
    user_movie = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    svd = TruncatedSVD(
        n_components=5,   # 🔴 reduced
        random_state=42
    )

    latent = svd.fit_transform(user_movie)
    reconstructed = np.dot(latent, svd.components_)

    scaler = MinMaxScaler((0.5, 5))
    reconstructed = scaler.fit_transform(reconstructed)

    return pd.DataFrame(
        reconstructed,
        index=user_movie.index,
        columns=user_movie.columns
    )

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
method = st.sidebar.selectbox(
    "Recommendation Method",
    ["Content-Based (Genres)", "Collaborative Filtering (SVD)"]
)

# --------------------------------------------------
# Content-Based Recommendation
# --------------------------------------------------
if method == "Content-Based (Genres)":
    movie_name = st.selectbox("Select a movie", df["title"].unique())

    if st.button("Recommend Movies"):
        with st.spinner("Computing similarity..."):
            cosine_sim = compute_similarity(df["genres"])
            indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
            idx = indices[movie_name]

            scores = list(enumerate(cosine_sim[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

            movie_idxs = [i[0] for i in scores]
            result = df.iloc[movie_idxs][["title", "genres"]]

        st.success("Top 10 similar movies")
        st.dataframe(result)

# --------------------------------------------------
# Collaborative Filtering
# --------------------------------------------------
else:
    user_id = st.selectbox("Select User ID", df["userId"].unique())

    if st.button("Recommend Movies"):
        with st.spinner("Running SVD model..."):
            preds = compute_svd(df)
            top_movies = preds.loc[user_id].sort_values(ascending=False).head(10)

            result = (
                df[df["movieId"].isin(top_movies.index)]
                [["title"]]
                .drop_duplicates()
            )

        st.success("Top 10 recommended movies")
        st.dataframe(result)
