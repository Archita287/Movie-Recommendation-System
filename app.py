import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# --------------------------------------------------
# Load Data (FULL 10K RECORDS)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sampled_movies_ratings.csv")
    df = df.dropna()
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    return df

df = load_data()

st.caption(f"Dataset loaded: {df.shape[0]} records")

# --------------------------------------------------
# TF-IDF + Cosine Similarity (Content-Based)
# --------------------------------------------------
@st.cache_data
def build_tfidf_similarity(genres):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(genres)
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# --------------------------------------------------
# SVD Model (Collaborative Filtering)
# --------------------------------------------------
@st.cache_data
def build_svd_predictions(df):
    user_movie_matrix = df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    svd = TruncatedSVD(
        n_components=20,
        random_state=42
    )

    latent_matrix = svd.fit_transform(user_movie_matrix)
    reconstructed_matrix = np.dot(latent_matrix, svd.components_)

    scaler = MinMaxScaler(feature_range=(0.5, 5))
    reconstructed_matrix = scaler.fit_transform(reconstructed_matrix)

    predictions = pd.DataFrame(
        reconstructed_matrix,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.columns
    )

    return predictions

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
method = st.sidebar.selectbox(
    "Choose Recommendation Method",
    ["Content-Based (Genres)", "Collaborative Filtering (SVD)"]
)

# --------------------------------------------------
# Content-Based Recommendation
# --------------------------------------------------
if method == "Content-Based (Genres)":
    st.subheader("🎥 Content-Based Recommendation")

    movie_name = st.selectbox(
        "Select a Movie",
        sorted(df["title"].unique())
    )

    if st.button("Recommend Movies"):
        with st.spinner("Computing similarity using TF-IDF..."):
            cosine_sim = build_tfidf_similarity(df["genres"])
            indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
            idx = indices[movie_name]

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

            movie_indices = [i[0] for i in sim_scores]
            recommendations = df.iloc[movie_indices][["title", "genres"]]

        st.success("Top 10 similar movies")
        st.dataframe(recommendations)

# --------------------------------------------------
# Collaborative Filtering Recommendation
# --------------------------------------------------
else:
    st.subheader("👥 Collaborative Filtering (SVD)")

    user_id = st.selectbox(
        "Select User ID",
        sorted(df["userId"].unique())
    )

    if st.button("Recommend Movies"):
        with st.spinner("Predicting ratings using SVD..."):
            predictions = build_svd_predictions(df)
            top_movies = predictions.loc[user_id].sort_values(ascending=False).head(10)

            recommended_movies = (
                df[df["movieId"].isin(top_movies.index)]
                [["movieId", "title"]]
                .drop_duplicates()
            )

        st.success("Top 10 recommended movies")
        st.dataframe(recommended_movies)
