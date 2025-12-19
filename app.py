import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Movie Recommendation System", layout="centered")
st.title("🎬 Movie Recommendation System")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Archita287/Movie-Recommendation-System/main/sampled_movies_ratings.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    df["genres"] = df["genres"].str.replace("|", " ", regex=False)
    return df

df = load_data()

# --------------------------------------------------
# CONTENT-BASED FILTERING (TF-IDF)
# --------------------------------------------------
@st.cache_data
def build_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["genres"])
    return cosine_similarity(tfidf_matrix)

cosine_sim = build_similarity(df)

def content_recommend(movie, n=10):
    matches = df[df["title"] == movie]
    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    indices = [i[0] for i in scores]
    return df.iloc[indices]["title"].unique()

# --------------------------------------------------
# COLLABORATIVE FILTERING (SAFE SVD)
# --------------------------------------------------
@st.cache_resource
def build_svd(data):
    matrix = data.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # 🔑 CRITICAL FIX
    max_components = min(matrix.shape[0], matrix.shape[1]) - 1
    n_components = min(20, max_components)

    svd = TruncatedSVD(n_components=n_components, random_state=132629)
    latent = svd.fit_transform(matrix)
    reconstructed = np.dot(latent, svd.components_)

    scaler = MinMaxScaler(feature_range=(0.5, 5))
    reconstructed = scaler.fit_transform(reconstructed)

    return pd.DataFrame(
        reconstructed,
        index=matrix.index,
        columns=matrix.columns
    )

predicted = build_svd(df)

def svd_recommend(user_id, n=10):
    if user_id not in predicted.index:
        return None

    top_ids = (
        predicted.loc[user_id]
        .sort_values(ascending=False)
        .head(n)
        .index
    )
    return df[df["movieId"].isin(top_ids)]["title"].unique()

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
choice = st.radio(
    "Select Recommendation Type",
    ["Content-Based (Genres)", "Collaborative Filtering (SVD)"]
)

if choice == "Content-Based (Genres)":
    movie = st.selectbox("Select Movie", sorted(df["title"].unique()))
    if st.button("Recommend"):
        recs = content_recommend(movie)
        if len(recs) == 0:
            st.warning("No recommendations found.")
        else:
            st.success("Recommended Movies:")
            for i, m in enumerate(recs, 1):
                st.write(f"{i}. {m}")

else:
    user = st.number_input(
        "Enter User ID",
        min_value=int(df["userId"].min()),
        max_value=int(df["userId"].max()),
        step=1
    )

    if st.button("Recommend"):
        recs = svd_recommend(user)
        if recs is None:
            st.warning("User ID not found in training data.")
        else:
            st.success("Recommended Movies:")
            for i, m in enumerate(recs, 1):
                st.write(f"{i}. {m}")

st.markdown("---")
st.caption("TF-IDF Content Filtering + Robust SVD Collaborative Filtering")
