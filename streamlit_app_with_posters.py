
import streamlit as st
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# TMDb API Key (securely embedded)
TMDB_API_KEY = "688e7cab1012d2f55568d2e9ec227b0d"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                          sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movieId', 'title'])
    return ratings, movies

def fetch_movie_poster(movie_title):
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_title
        }
        response = requests.get(TMDB_SEARCH_URL, params=params)
        data = response.json()
        if data["results"]:
            poster_path = data["results"][0].get("poster_path", None)
            if poster_path:
                return TMDB_IMG_BASE + poster_path
        return None
    except:
        return None

def collaborative_recommendations(user_id, ratings, movies, top_n=5):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_movie_matrix.index:
        return ["User not found in dataset."], []

    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    similarity = cosine_similarity(user_vector, user_movie_matrix)[0]
    sim_df = pd.DataFrame({'userId': user_movie_matrix.index, 'similarity': similarity})
    sim_df = sim_df[sim_df['userId'] != user_id].sort_values(by='similarity', ascending=False)

    top_users = sim_df.head(5)['userId'].tolist()
    top_movies = ratings[ratings['userId'].isin(top_users)].groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n)
    recommended_movies = movies[movies['movieId'].isin(top_movies.index)]
    return recommended_movies['title'].tolist(), recommended_movies['title'].tolist()

# Streamlit UI
st.title("🎬 Collaborative Movie Recommender with Posters")
ratings, movies = load_data()

user_ids = ratings['userId'].unique()
user_id = st.selectbox("Select your User ID:", sorted(user_ids))

if st.button("Get Recommendations"):
    recommendations, titles = collaborative_recommendations(user_id, ratings, movies)
    st.subheader("Recommended for You:")
    for title in titles:
        st.markdown(f"**{title}**")
        poster_url = fetch_movie_poster(title)
        if poster_url:
            st.image(poster_url, width=120)
        else:
            st.write("_Poster not available_")
