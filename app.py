from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# ---------------- Load movie data ----------------
movies_data = pd.read_csv(r'E:\Btechthesis\movies.csv')
movies_data.fillna('', inplace=True)

selected_features = ['genres','keywords','tagline','cast','director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# ---------------- Recommendation function ----------------
def recommend_movies(prompt, top_n=10):
    prompt_vector = vectorizer.transform([prompt])
    similarity_scores = cosine_similarity(prompt_vector, feature_vectors)
    sorted_indices = np.argsort(similarity_scores[0])[::-1][:top_n]
    recommended = [movies_data.iloc[i]['title'] for i in sorted_indices]
    return recommended

# ---------------- HTML Template with Modern CSS ----------------
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Movie Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            padding-top:30px;
            margin: 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: #fff;
            padding: 40px;
            border-radius: 15px;
            max-width: 700px;
            width: 90%;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
            text-align: center;
        }
        h2 {
            margin-bottom: 30px;
            color: #333;
        }
        form input[type=text] {
            width: 70%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #764ba2;
            border-radius: 30px;
            outline: none;
            transition: 0.3s;
        }
        form input[type=text]:focus {
            border-color: #667eea;
        }
        form input[type=submit] {
            padding: 15px 30px;
            font-size: 16px;
            margin-left: 10px;
            border: none;
            border-radius: 30px;
            background: #764ba2;
            color: #fff;
            cursor: pointer;
            transition: 0.3s;
        }
        form input[type=submit]:hover {
            background: #667eea;
        }
        .movie-card {
            background: #f4f4f4;
            border-radius: 10px;
            padding: 15px 20px;
            margin: 10px 0;
            text-align: left;
            font-weight: 500;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: 0.3s;
        }
        .movie-card:hover {
            background: #e1d4f7;
            transform: translateY(-3px);
        }
        ul {
            list-style: none;
            padding: 0;
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
            <h3>Recommended Movies:</h3>
            <ul>
                {% for movie in recommendations %}
                    <li class="movie-card">{{ movie }}</li>
                {% endfor %}
            </ul>
        {% endif %}
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
    app.run(debug=True)
