
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
                         sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movieId', 'title'])
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                          sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    return movies, ratings

def collaborative_recommendations(user_id, ratings_df, movies_df, top_n=5):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    movie_ids = movies_df['movieId'].unique()
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated_movies = [mid for mid in movie_ids if mid not in rated_movies]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in unrated_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = [movies_df[movies_df['movieId'] == mid]['title'].values[0] for mid, _ in predictions[:top_n]]
    return top_movies

# Content-Based filtering
def recommend_movies(selected_title, movies_df, similarity_matrix):
    idx = movies_df[movies_df['title'] == selected_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_titles = [movies_df.iloc[i[0]]['title'] for i in sim_scores]
    return recommended_titles

# Main app
st.title("🎬 Hybrid Movie Recommendation System")
st.write("Choose content-based or collaborative filtering to get recommendations.")

movies_df, ratings_df = load_data()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

mode = st.radio("Choose Recommendation Mode:", ["Content-Based", "Collaborative Filtering"])

if mode == "Content-Based":
    movie_list = movies_df['title'].tolist()
    selected_movie = st.selectbox("Select a movie you like:", movie_list)
    if st.button("Get Content-Based Recommendations"):
        recommendations = recommend_movies(selected_movie, movies_df, cosine_sim)
        st.subheader("You might also like:")
        for title in recommendations:
            st.write(f"- {title}")

if mode == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID (1-943):", min_value=1, max_value=943, value=1)
    if st.button("Get Collaborative Recommendations"):
        recommendations = collaborative_recommendations(int(user_id), ratings_df, movies_df)
        st.subheader("Recommended for You:")
        for title in recommendations:
            st.write(f"- {title}")
