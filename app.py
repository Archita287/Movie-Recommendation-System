import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("sampled_movies_ratings.csv")
    df = df.dropna()
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    return df

df = load_data()

# ---------------- TF-IDF SIMILARITY ----------------
@st.cache_data
def build_tfidf_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_sim = build_tfidf_similarity(df)

movie_indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

def recommend_content(movie_title, n=5):
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_idxs = [i[0] for i in sim_scores]
    return df.iloc[movie_idxs][["title", "genres"]]

# ---------------- SVD COLLABORATIVE FILTERING ----------------
@st.cache_data
def build_svd(data):
    user_movie = data.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd.fit_transform(user_movie)

    reconstructed = np.dot(matrix_reduced, svd.components_)
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_movie.index,
        columns=user_movie.columns
    )

    scaler = MinMaxScaler()
    reconstructed_df[:] = scaler.fit_transform(reconstructed_df)

    return reconstructed_df

svd_predictions = build_svd(df)

def recommend_user(user_id, n=5):
    if user_id not in svd_predictions.index:
        return None

    user_ratings = svd_predictions.loc[user_id].sort_values(ascending=False)
    top_movies = user_ratings.head(n).index

    return df[df["movieId"].isin(top_movies)][["title", "genres"]].drop_duplicates()

# ---------------- STREAMLIT UI ----------------
st.sidebar.header("🔍 Recommendation Type")
option = st.sidebar.radio(
    "Choose Recommendation Method",
    ("Content-Based (Movie Similarity)", "Collaborative (User-Based)")
)

# -------- CONTENT BASED UI --------
if option == "Content-Based (Movie Similarity)":
    st.subheader("🎥 Content-Based Recommendation")

    movie_name = st.selectbox(
        "Select a movie",
        sorted(df["title"].unique())
    )

    if st.button("Recommend Similar Movies"):
        recommendations = recommend_content(movie_name)
        st.success("Recommended Movies:")
        st.dataframe(recommendations)

# -------- USER BASED UI --------
else:
    st.subheader("👤 User-Based Recommendation (SVD)")

    user_id = st.selectbox(
        "Select User ID",
        sorted(df["userId"].unique())
    )

    if st.button("Recommend Movies"):
        recs = recommend_user(user_id)

        if recs is None:
            st.error("User not found!")
        else:
            st.success("Recommended Movies:")
            st.dataframe(recs)
