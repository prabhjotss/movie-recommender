import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data and pre-trained model
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('🎬 Movie Recommendation System')

# Movie selection box
selected_movie = st.selectbox(
    'Select a movie you like:',
    movies['title'].values
)

def recommend(movie):
    # Find the index of the selected movie
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    # Get top 5 similar movies (excluding itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

if st.button('Show Recommendations'):
    recommendations = recommend(selected_movie)
    st.subheader('Top 5 Recommended Movies:')
    for m in recommendations:
        st.write(f'🎥 {m}')
