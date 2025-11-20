from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# ---------------- Load movie data ----------------
movies_data = pd.read_csv("movie.csv")
movies_data.fillna('', inplace=True)

# Columns in your dataset:
# movie_id, movie_name, year, genre, overview, director, cast

selected_features = ['genre', 'overview', 'director', 'cast']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = (
    movies_data['genre'] + ' ' +
    movies_data['overview'] + ' ' +
    movies_data['director'] + ' ' +
    movies_data['cast']
)

# Vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# ---------------- Recommendation function ----------------
def recommend_movies(prompt, top_n=10):
    prompt_vector = vectorizer.transform([prompt])
    similarity_scores = cosine_similarity(prompt_vector, feature_vectors)
    sorted_indices = np.argsort(similarity_scores[0])[::-1][:top_n]
    recommended = [movies_data.iloc[i]['movie_name'] for i in sorted_indices]
    return recommended


# ---------------- HTML Template ----------------
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Movie Recommender</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: #f2f5f9;
            color: #333;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 30px 15px;
        }

        .container {
            width: 100%;
            max-width: 720px;
            background: #ffffff;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.08);
        }

        h2 {
            text-align: center;
            margin-bottom: 20px;
            font-weight: 700;
            font-size: 28px;
            color: #2a4d8f;
        }

        form {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
        }

        input[type=text] {
            flex: 1;
            padding: 14px;
            border-radius: 12px;
            border: 2px solid #cfd8e3;
            background: #f9fbfd;
            color: #333;
            font-size: 16px;
            outline: none;
            transition: 0.3s;
        }

        input[type=text]:focus {
            border-color: #2a4d8f;
            background: #ffffff;
        }

        input[type=submit] {
            padding: 14px 22px;
            background: #2a4d8f;
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            transition: 0.3s;
        }

        input[type=submit]:hover {
            background: #19386b;
        }

        .movie-card {
            background: #f5f7fa;
            padding: 15px;
            border-radius: 12px;
            margin: 8px 0;
            font-size: 16px;
            border: 1px solid #e2e8f0;
            transition: 0.3s;
        }

        .movie-card:hover {
            background: #e8eef7;
            border-color: #c3d4ec;
            transform: translateY(-3px);
        }

        h3 {
            margin-top: 25px;
            margin-bottom: 10px;
            font-size: 22px;
            font-weight: 600;
            color: #2a4d8f;
        }

        footer {
            text-align: center;
            margin-top: 25px;
            font-size: 14px;
            color: #666;
        }

        footer span {
            color: #2a4d8f;
            font-weight: 600;
        }

        @media (max-width: 600px) {
            form {
                flex-direction: column;
            }
            input[type=submit] {
                width: 100%;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <h2>🎬 Movie Recommendation System</h2>

        <form method="post">
            <input type="text" name="prompt" placeholder="Enter movie name or description" required>
            <input type="submit" value="Recommend">
        </form>

        {% if recommendations %}
            <h3>Recommended Movies</h3>
            <ul style="list-style:none; padding:0;">
                {% for movie in recommendations %}
                    <li class="movie-card">{{ movie }}</li>
                {% endfor %}
            </ul>
        {% endif %}

        <footer>Made by <span>Prabhjot Singh</span> ⭐</footer>
    </div>
</body>
</html>
"""

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        prompt = request.form['prompt']
        recommendations = recommend_movies(prompt, top_n=5)
    return render_template_string(html_template, recommendations=recommendations)


# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
