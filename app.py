import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sampled_movies_ratings.csv")
    df = df.dropna()
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    return df

df = load_data()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("🔍 Recommendation Method")

option = st.sidebar.selectbox(
    "Choose Method",
    ["Content-Based (Genres)", "Collaborative Filtering (SVD)"]
)

# ----------------------------------------------------
# CONTENT-BASED RECOMMENDATION
# ----------------------------------------------------
if option == "Content-Based (Genres)":
    st.subheader("🎥 Content-Based Movie Recommendation")

    movie_name = st.selectbox(
        "Select a Movie",
        sorted(df["title"].unique())
    )

    if st.button("Recommend Movies"):
        with st.spinner("Creating TF-IDF vectors and similarity matrix..."):
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(df["genres"])

            cosine_sim = cosine_similarity(tfidf_matrix)

            movie_indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
            idx = movie_indices[movie_name]

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

            movie_indices_list = [i[0] for i in sim_scores]
            recommendations = df.iloc[movie_indices_list][["title", "genres"]]

        st.success("Top 10 similar movies")
        st.dataframe(recommendations)

# ----------------------------------------------------
# COLLABORATIVE FILTERING (SVD)
# ----------------------------------------------------
else:
    st.subheader("👥 Collaborative Filtering using SVD")

    user_id = st.selectbox(
        "Select User ID",
        sorted(df["userId"].unique())
    )

    if st.button("Recommend Movies"):
        with st.spinner("Training SVD model and predicting ratings..."):
            user_movie_matrix = df.pivot_table(
                index="userId",
                columns="movieId",
                values="rating"
            ).fillna(0)

            svd = TruncatedSVD(n_components=15, random_state=42)
            latent_matrix = svd.fit_transform(user_movie_matrix)

            reconstructed_matrix = np.dot(latent_matrix, svd.components_)

            scaler = MinMaxScaler(feature_range=(0.5, 5))
            reconstructed_matrix = scaler.fit_transform(reconstructed_matrix)

            predictions = pd.DataFrame(
                reconstructed_matrix,
                index=user_movie_matrix.index,
                columns=user_movie_matrix.columns
            )

            top_movies = predictions.loc[user_id].sort_values(ascending=False).head(10)

            movie_titles = df[df["movieId"].isin(top_movies.index)][
                ["movieId", "title"]
            ].drop_duplicates()

        st.success("Top 10 recommended movies")
        st.dataframe(movie_titles)
