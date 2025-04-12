
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                          sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movieId', 'title'])
    return ratings, movies

def collaborative_recommendations(user_id, ratings, movies, top_n=5):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_movie_matrix.index:
        return ["User not found in dataset."]
    
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    similarity = cosine_similarity(user_vector, user_movie_matrix)[0]
    sim_df = pd.DataFrame({'userId': user_movie_matrix.index, 'similarity': similarity})
    sim_df = sim_df[sim_df['userId'] != user_id].sort_values(by='similarity', ascending=False)
    
    top_users = sim_df.head(5)['userId'].tolist()
    top_movies = ratings[ratings['userId'].isin(top_users)].groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n)
    recommendations = movies[movies['movieId'].isin(top_movies.index)]['title'].tolist()
    return recommendations

# Main app
st.title("🎬 Collaborative Movie Recommendation (Cloud Compatible)")
ratings, movies = load_data()

user_ids = ratings['userId'].unique()
user_id = st.selectbox("Select your User ID:", sorted(user_ids))

if st.button("Get Recommendations"):
    recommendations = collaborative_recommendations(user_id, ratings, movies)
    st.subheader("Top Recommendations for You:")
    for movie in recommendations:
        st.write(f"- {movie}")
