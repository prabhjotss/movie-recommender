import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Sample dataset (you can replace this with a larger one, e.g., TMDB or IMDb dataset)
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Tenet', 'Memento'],
    'overview': [
        'A thief who steals corporate secrets through dream-sharing technology.',
        'A team of explorers travel through a wormhole in space to ensure humanity’s survival.',
        'When the menace known as the Joker wreaks havoc, Batman must accept one of the greatest tests of his ability.',
        'Protagonist manipulates the flow of time to prevent World War III.',
        'A man suffering from short-term memory loss uses notes and tattoos to hunt for his wife’s murderer.'
    ]
}

movies = pd.DataFrame(data)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Save data and model
pickle.dump(movies.to_dict(), open('movies_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
